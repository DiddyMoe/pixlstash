import re
import sys

from fastapi import (
    Depends,
    HTTPException,
    Query,
    Request,
)
from pydantic import ConfigDict
from sqlmodel import select
from typing import Annotated, Optional

from pixlstash.database import OCR_TEXT_MATCH_WEIGHT
from pixlstash.db_models import (
    Face,
    Picture,
    SortMechanism,
)
from pixlstash.inference.cpu_query_encoders import CpuQueryEncodersNotReadyError
from pixlstash.pixl_logging import get_logger
from pixlstash.utils.service.filter_helpers import (
    collect_set_filter_ids,
    fetch_scope_allowed_picture_ids,
    fetch_set_candidate_ids,
    is_scrapheap_view_id,
    is_truthy_flag,
    narrow_picture_project_ids,
    normalize_set_mode,
    project_membership_exists_clause,
    project_unassigned_clause,
    request_targets_scrapheap,
)
from pixlstash.utils.query.predicate_filter import PredicateFilter
from pixlstash.utils.sqlite_ints import SqliteInt, parse_sqlite_int
from pixlstash.utils.service.scope_table import scope_id_subquery
from pixlstash.utils.sql_chunking import chunked

from ._helpers import (
    _face_cluster_ids,
    _fetch_hidden_picture_ids,
)
from ._listing import (
    PictureListFilters,
    StoredPictureRow,
    _storable_picture_ids,
    enforce_list_id_cap,
    parse_picture_filter,
    shared_picture_ids,
)


logger = get_logger(__name__)


class PictureMetadataResponse(StoredPictureRow):
    """Serialized picture metadata as returned by search/CRUD endpoints.

    The full picture model is large and dynamic; common fields are enumerated
    here for documentation and ``extra="allow"`` preserves every other field
    (including the search-only ``likeness_score`` overlay)."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    format: Optional[str] = None
    score: Optional[int] = None
    likeness_score: Optional[float] = None
    text_match: bool = False


SEARCH_DEFAULT_THRESHOLD = 0.5
# The result pill's "In text" switch: only the pictures whose own text matched
# the query. A view param, not an export option, so an export of a search view
# carries it with the rest of the view's query and packages what the grid shows.
TEXT_MATCH_ONLY_PARAM = "text_match_only"


def search_results(
    server,
    request,
    query: str,
    *,
    offset: int = 0,
    limit: int = sys.maxsize,
    threshold: float = SEARCH_DEFAULT_THRESHOLD,
) -> list[tuple[Picture, float]]:
    """The pictures a text search shows, in the order it shows them.

    Everything that decides which pictures a search answers with, and their
    order, is here: the listing's filters (one parse with the listing), sets,
    people, projects, face clusters, the Scrapheap flag, shared-only, the
    token's scope, the ranking, the hidden tags and the "In text" switch
    (``text_match_only``). ``GET /pictures/search`` serializes what this
    returns, and an export of a search view packages it, so the two cannot
    disagree.

    Args:
        server: The running server.
        request: The request whose query string carries the view's filters
            and whose state carries the caller's token. A caller that is not
            the search route passes a request whose query string holds the
            view's params only: every key in it is read as the search reads
            it.
        query: The search text.
        offset: Results to skip.
        limit: Most results to return.
        threshold: Lowest relevance a result may have.

    Returns:
        ``(picture, relevance)`` pairs, best first unless the request sorts.

    Raises:
        HTTPException: 400 for an empty query, a bad sort or an id SQLite
            cannot store; 422 for a filter the listing refuses; 503 when the
            query encoders failed to load.
    """
    enforce_list_id_cap(request)
    unassigned_by = request.query_params.get("unassigned_by")
    # Every narrowing filter the listing honours, parsed where it parses
    # them, so a search narrows as the grid it searches.
    narrowing = parse_picture_filter(server, request)
    query_params = {}
    format = None
    character_id = None
    set_id = None
    set_ids = None
    set_mode = "union"
    project_id = None
    sort = None
    descending = True
    unscored = False
    favourite_only = False
    comfyui_models = []
    comfyui_loras = []
    tags_filter = []
    tags_rejected_filter = []
    base_set_id_raw = None
    stack_state = None
    imported_since = None
    if request.query_params:
        query_params = dict(request.query_params)
        query_params.pop("query", None)
        query_params.pop("offset", None)
        query_params.pop("limit", None)
        character_id = query_params.pop("character_id", None)
        set_id = query_params.pop("set_id", None)
        set_ids = request.query_params.getlist("set_ids")
        set_mode = query_params.pop("set_mode", "union")
        base_set_id_raw = query_params.pop("base_set_id", None)
        project_id = query_params.pop("project_id", None)
        sort = query_params.pop("sort", None)
        desc_val = query_params.pop("descending", descending)
        descending = (
            desc_val.lower() == "true" if isinstance(desc_val, str) else bool(desc_val)
        )
        # The membership and pagination params keep their bespoke
        # ``query_params`` pop handling.
        _predicate_filter = narrowing
        format = _predicate_filter.format or []
        unscored = _predicate_filter.unscored
        favourite_only = _predicate_filter.favourite_only
        comfyui_models = _predicate_filter.comfyui_models_filter or []
        comfyui_loras = _predicate_filter.comfyui_loras_filter or []
        tags_filter = _predicate_filter.tags_filter or []
        tags_rejected_filter = _predicate_filter.tags_rejected_filter or []
        stack_state = _predicate_filter.stack_state
        # A search in the Inbox searches the Inbox: the unassigned view
        # (both halves) added within its period, as the listing reads it.
        if request.query_params.get("inbox") is not None:
            character_id = "UNASSIGNED"
            unassigned_by = None
            imported_since = _predicate_filter.imported_since
    min_score = narrowing.min_score
    max_score = narrowing.max_score
    smart_score_bucket = narrowing.smart_score_bucket
    resolution_bucket = narrowing.resolution_bucket
    if not query:
        raise HTTPException(
            status_code=400, detail="Query parameter is required for search"
        )

    only_deleted = request_targets_scrapheap(request.query_params)
    if is_scrapheap_view_id(character_id):
        character_id = None
    # The Scrapheap wins over the unassigned view and the Inbox, as in
    # ``select_pictures_for_listing``: those read live pictures only, so the
    # search would otherwise find nothing in the heap.
    if only_deleted and character_id == "UNASSIGNED":
        character_id = None
    candidate_ids = None
    sort_mech = None
    normalized_set_mode = normalize_set_mode(set_mode)
    set_filter_ids = collect_set_filter_ids(
        set_id_value=set_id,
        set_ids_values=set_ids,
    )
    if normalized_set_mode == "difference" and base_set_id_raw is not None:
        try:
            _base_id = int(base_set_id_raw)
            if _base_id in set_filter_ids:
                set_filter_ids = [_base_id] + [
                    s for s in set_filter_ids if s != _base_id
                ]
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid base_set_id value %r in /pictures/search request",
                base_set_id_raw,
            )

    if sort:
        try:
            sort_mech = SortMechanism.from_string(sort, descending=descending)
        except ValueError as ve:
            logger.error("Invalid sort mechanism for search: %s", ve)
            raise HTTPException(status_code=400, detail="Invalid sort mechanism")

    if set_filter_ids:
        candidate_ids = server.vault.db.run_immediate_read_task(
            fetch_set_candidate_ids,
            set_ids=set_filter_ids,
            set_mode=normalized_set_mode,
            deleted_only=only_deleted,
        )
    elif character_id is not None:
        if character_id == "UNASSIGNED":

            def fetch_unassigned_ids(session):
                assignment_project_id = None
                assignment_unassigned_project = False
                if project_id == "UNASSIGNED":
                    assignment_unassigned_project = True
                elif project_id is not None:
                    try:
                        assignment_project_id = parse_sqlite_int(project_id)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid project_id %r for UNASSIGNED search assignment scope; treating as global scope",
                            project_id,
                        )
                unassigned_conditions = Picture.build_unassigned_conditions(
                    enforce_stack_assignment=True,
                    assignment_project_id=assignment_project_id,
                    assignment_unassigned_project=assignment_unassigned_project,
                    unassigned_by=unassigned_by,
                )
                query_stmt = select(Picture.id).where(
                    *unassigned_conditions,
                    Picture.deleted.is_(False),
                )
                if imported_since is not None:
                    query_stmt = query_stmt.where(Picture.imported_at >= imported_since)
                return list(session.exec(query_stmt).all())

            candidate_ids = set(
                server.vault.db.run_immediate_read_task(fetch_unassigned_ids)
            )
        elif character_id in ("ALL", ""):
            candidate_ids = None
        elif str(character_id).isdigit():
            try:
                character_id_value = parse_sqlite_int(character_id)
            except ValueError:
                logger.warning(
                    "Refusing a character_id past SQLite's range in /pictures/search"
                )
                raise HTTPException(status_code=400, detail="Invalid character_id")

            def fetch_character_ids(session, character_id_value):
                # A subquery, not a fetched id list: a person can be in
                # more pictures than SQLite binds in one statement.
                character_picture_ids = select(Face.picture_id).where(
                    Face.character_id == character_id_value
                )
                rows = session.exec(
                    select(Picture.id).where(
                        Picture.id.in_(character_picture_ids),
                        Picture.deleted.is_(False),
                    )
                ).all()
                return list(rows)

            candidate_ids = set(
                server.vault.db.run_immediate_read_task(
                    fetch_character_ids, character_id_value
                )
            )
    else:
        search_character_ids_raw = request.query_params.getlist("character_ids")
        search_character_id_list = []
        for _v in search_character_ids_raw:
            try:
                _cid = parse_sqlite_int(_v)
                if _cid > 0:
                    search_character_id_list.append(_cid)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid character_ids value %r in /pictures/search request",
                    _v,
                )
        # Once each, in the caller's order: ``difference`` keeps the first
        # person as its base, and a repeat of that person later in the list
        # would subtract their own pictures from it.
        search_character_id_list = list(dict.fromkeys(search_character_id_list))
        if search_character_id_list:
            search_character_mode_raw = query_params.get("character_mode", "union")
            search_character_mode = (
                (search_character_mode_raw or "union").strip().lower()
            )
            if search_character_mode not in {
                "union",
                "intersection",
                "difference",
                "xor",
            }:
                search_character_mode = "union"

            def fetch_character_ids_by_mode(session, ids, mode):
                # The caller's list has no cap: read in chunks under SQLite's
                # bound-parameter limit. Each row names its own person; the
                # modes below combine the merged rows in Python.
                rows = [
                    row
                    for id_chunk in chunked(ids)
                    for row in session.exec(
                        select(Face.character_id, Face.picture_id).where(
                            Face.character_id.in_(id_chunk)
                        )
                    ).all()
                ]
                members_by_char: dict[int, set[int]] = {cid: set() for cid in ids}
                for cid, pid in rows:
                    members_by_char.setdefault(int(cid), set()).add(int(pid))
                if mode == "intersection":
                    result: set[int] | None = None
                    for cid in ids:
                        current = members_by_char.get(cid, set())
                        result = set(current) if result is None else result & current
                    candidate_pids = list(result or set())
                elif mode == "difference":
                    first = members_by_char.get(ids[0], set())
                    rest_set: set[int] = set()
                    for cid in ids[1:]:
                        rest_set |= members_by_char.get(cid, set())
                    candidate_pids = list(first - rest_set)
                elif mode == "xor":
                    xor_u: set[int] = set()
                    for cid in ids:
                        xor_u |= members_by_char.get(cid, set())
                    xor_i: set[int] | None = None
                    for cid in ids:
                        cur = members_by_char.get(cid, set())
                        xor_i = set(cur) if xor_i is None else xor_i & cur
                    candidate_pids = list(xor_u - (xor_i or set()))
                else:
                    union_pids: set[int] = set()
                    for cid in ids:
                        union_pids |= members_by_char.get(cid, set())
                    candidate_pids = list(union_pids)
                if not candidate_pids:
                    return []
                candidate_scope = scope_id_subquery(
                    session, candidate_pids, name="_pixlstash_search_people_ids"
                )
                rows2 = session.exec(
                    select(Picture.id).where(
                        Picture.id.in_(candidate_scope),
                        Picture.deleted.is_(False),
                    )
                ).all()
                return list(rows2)

            candidate_ids = set(
                server.vault.db.run_immediate_read_task(
                    fetch_character_ids_by_mode,
                    search_character_id_list,
                    search_character_mode,
                )
            )

    # One of the person's face clusters (``face_cluster``), as the listing
    # takes it.
    face_cluster_ids = _face_cluster_ids(
        server,
        request,
        character_id,
        request.query_params.get("face_cluster"),
        request.query_params.get("face_cluster_threshold"),
    )
    if face_cluster_ids is not None:
        candidate_ids = (
            face_cluster_ids
            if candidate_ids is None
            else set(candidate_ids) & face_cluster_ids
        )

    if project_id is not None:

        def fetch_project_ids(
            session,
            project_id_value: str,
            deleted_only: bool,
        ):
            query_stmt = select(Picture.id)
            if project_id_value == "UNASSIGNED":
                query_stmt = query_stmt.where(project_unassigned_clause(Picture))
            else:
                try:
                    parsed_project_id = parse_sqlite_int(project_id_value)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid project_id",
                    )
                query_stmt = query_stmt.where(
                    project_membership_exists_clause(parsed_project_id, Picture)
                )
            if deleted_only:
                query_stmt = query_stmt.where(Picture.deleted.is_(True))
            else:
                query_stmt = query_stmt.where(Picture.deleted.is_(False))
            return list(session.exec(query_stmt).all())

        project_candidate_ids = set(
            server.vault.db.run_immediate_read_task(
                fetch_project_ids,
                project_id,
                only_deleted,
            )
        )
        candidate_ids = (
            project_candidate_ids
            if candidate_ids is None
            else candidate_ids & project_candidate_ids
        )

    # The listing's filters the semantic search takes no argument for
    # narrow its candidates instead, compiled by the same
    # ``PredicateFilter`` the listing's ``Picture.find`` compiles them with.
    filter_clauses = PredicateFilter(
        tags_confidence_above_filter=narrowing.tags_confidence_above_filter,
        tags_confidence_below_filter=narrowing.tags_confidence_below_filter,
        face_filter=narrowing.face_filter,
        impossible_sources=narrowing.impossible_sources,
        file_path_prefix=narrowing.file_path_prefix,
        import_source_folder=narrowing.import_source_folder,
        workflow_structural_hashes=narrowing.workflow_structural_hashes,
        tag_state=narrowing.tag_state,
        unprocessable_ids=narrowing.unprocessable_ids,
        # The search applies the deleted scope itself.
        apply_deleted_filter=False,
    ).predicates()
    reference_folder_id = request.query_params.get("reference_folder_id")
    if reference_folder_id is not None:
        # An integer SQLite stores: ``PictureListFilters`` refused any other.
        filter_clauses.append(
            Picture.reference_folder_id == parse_sqlite_int(reference_folder_id)
        )
    id_filter = None
    if "id" in request.query_params:
        id_filter = _storable_picture_ids(request.query_params.getlist("id"))
    if filter_clauses or id_filter is not None:

        def fetch_filtered_ids(session):
            clauses = list(filter_clauses)
            if id_filter is not None:
                # Up to ``LIST_MAX_IDS`` (1000, ``enforce_list_id_cap``) ids,
                # past SQLite's historical 999-parameter limit, and the other
                # clauses bind their own values: read from a temp table.
                clauses.append(
                    Picture.id.in_(
                        scope_id_subquery(
                            session, id_filter, name="_pixlstash_search_id_filter"
                        )
                    )
                )
            return set(session.exec(select(Picture.id).where(*clauses)).all())

        filtered_ids = server.vault.db.run_immediate_read_task(fetch_filtered_ids)
        candidate_ids = (
            filtered_ids if candidate_ids is None else set(candidate_ids) & filtered_ids
        )

    shared_ids = shared_picture_ids(server, request)
    if shared_ids is not None:
        candidate_ids = (
            shared_ids if candidate_ids is None else set(candidate_ids) & shared_ids
        )

    # Token scope enforcement: restrict candidate_ids to the pictures
    # allowed by the token's resource scope.  This prevents a scoped
    # token (e.g. picture_set, project, character) from seeing pictures
    # outside its authorised scope by passing arbitrary filter params.
    scope_allowed = fetch_scope_allowed_picture_ids(server, request)
    if scope_allowed is not None:
        candidate_ids = (
            scope_allowed if candidate_ids is None else candidate_ids & scope_allowed
        )

    if candidate_ids is not None and not candidate_ids:
        return []

    # Encoded here, never inside the database task: the DB writer is one
    # thread, and an encode there holds every write for a model call.
    try:
        query_embedding = server.vault.generate_text_embedding(query)
        clip_query_embedding = server.vault.generate_clip_text_embedding(query)
    except CpuQueryEncodersNotReadyError as exc:
        # 503 rather than a 500: nothing is wrong with the request; the
        # CPU query encoders failed to load at start-up.
        logger.warning("Text search cannot encode its query: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    def find_by_text(session, query, offset, limit):
        words = re.findall(r"\b\w+\b", query.lower())
        semantic_offset = 0 if sort_mech else offset
        semantic_limit = sys.maxsize if sort_mech else limit
        candidate_size = len(candidate_ids) if candidate_ids else None

        def log_semantic_results(label: str, rows):
            if rows is None:
                return
            if not rows:
                logger.debug(
                    "Semantic search %s: no results (query=%r, words=%s, threshold=%s, format=%s, only_deleted=%s, candidate_ids=%s)",
                    label,
                    query,
                    words,
                    threshold,
                    format,
                    only_deleted,
                    candidate_size,
                )
                return
            preview = [
                {
                    "id": getattr(pic, "id", None),
                    "score": round(float(score), 4),
                }
                for pic, score in rows[:10]
            ]
            logger.debug(
                "Semantic search %s: results=%d (query=%r, words=%s, threshold=%s, format=%s, only_deleted=%s, candidate_ids=%s) top=%s",
                label,
                len(rows),
                query,
                words,
                threshold,
                format,
                only_deleted,
                candidate_size,
                preview,
            )

        results = Picture.semantic_search(
            session,
            query,
            words,
            query_embedding=query_embedding,
            clip_query_embedding=clip_query_embedding,
            text_match_weight=OCR_TEXT_MATCH_WEIGHT,
            offset=semantic_offset,
            limit=semantic_limit,
            threshold=threshold,
            format=format,
            select_fields=Picture.metadata_fields(),
            only_deleted=only_deleted,
            candidate_ids=list(candidate_ids) if candidate_ids else None,
            min_score=min_score,
            max_score=max_score,
            smart_score_bucket=smart_score_bucket,
            resolution_bucket=resolution_bucket,
            unscored=unscored,
            favourite_only=favourite_only,
            comfyui_models_filter=comfyui_models or None,
            comfyui_loras_filter=comfyui_loras or None,
            tags_filter=tags_filter or None,
            tags_rejected_filter=tags_rejected_filter or None,
            stack_state=stack_state,
        )

        log_semantic_results("base", results)

        if not sort_mech:
            return results

        if not results:
            return []

        score_map = {
            pic.id: score
            for pic, score in results
            if pic is not None and getattr(pic, "id", None) is not None
        }
        if not score_map:
            return []

        # Every picture the search matched, so possibly more ids than
        # SQLite binds in one statement: passed as a temp-table scope.
        result_scope = scope_id_subquery(
            session, score_map.keys(), name="_pixlstash_search_result_ids"
        )
        sorted_pics = Picture.find(
            session,
            sort_mech=sort_mech,
            offset=offset,
            limit=limit,
            select_fields=Picture.metadata_fields(),
            format=format,
            only_deleted=only_deleted,
            id=result_scope,
            min_score=min_score,
            max_score=max_score,
            smart_score_bucket=smart_score_bucket,
            resolution_bucket=resolution_bucket,
            unscored=unscored,
            favourite_only=favourite_only,
            comfyui_models_filter=comfyui_models or None,
            comfyui_loras_filter=comfyui_loras or None,
            tags_filter=tags_filter or None,
            tags_rejected_filter=tags_rejected_filter or None,
            stack_state=stack_state,
        )
        sorted_results = [(pic, score_map.get(pic.id, 0.0)) for pic in sorted_pics]
        log_semantic_results(f"sorted_{sort_mech.key.name}", sorted_results)
        return sorted_results

    # "In text" narrows before the page is cut, so a page holds the matches
    # that belong on it rather than the text matches of an unnarrowed page.
    text_match_only = is_truthy_flag(request.query_params.get(TEXT_MATCH_ONLY_PARAM))
    page = slice(offset, offset + limit)
    if text_match_only:
        offset, limit = 0, sys.maxsize
    results = server.vault.db.run_immediate_read_task(
        find_by_text, query, offset, limit
    )
    if results:
        hidden_ids = _fetch_hidden_picture_ids(
            server,
            request,
            [
                getattr(pic, "id", None)
                for pic, _score in results
                if pic is not None and getattr(pic, "id", None) is not None
            ],
        )
        if hidden_ids:
            results = [
                result
                for result in results
                if result[0] is not None
                and getattr(result[0], "id", None) not in hidden_ids
            ]
    if text_match_only and results:
        text_matched_ids = server.vault.db.run_immediate_read_task(
            Picture.ids_matching_text,
            query,
            [pic.id for pic, _score in results if pic is not None],
        )
        results = [
            result
            for result in results
            if result[0] is not None and result[0].id in text_matched_ids
        ][page]
    return results


def register_routes(router, server):
    @router.get(
        "/pictures/search",
        summary="Search pictures by text",
        description="Performs semantic text search across pictures with optional sort, filtering, and candidate scoping.",
        response_model=list[PictureMetadataResponse],
    )
    def search_pictures(
        request: Request,
        query: str,
        offset: Annotated[SqliteInt, Query()] = 0,
        limit: Annotated[SqliteInt, Query()] = sys.maxsize,
        threshold: float = Query(SEARCH_DEFAULT_THRESHOLD),
        # Read by search_results from the query string, as an export of the
        # view reads it; declared here for the schema.
        text_match_only: bool = Query(
            False,
            description=(
                "Only the pictures whose own text matched the query (the "
                "result pill's In text switch), narrowed before offset and "
                "limit."
            ),
        ),
        # The listing's filters, validated as the listing validates them.
        filters: PictureListFilters = Depends(),
    ):
        results = search_results(
            server,
            request,
            query,
            offset=offset,
            limit=limit,
            threshold=threshold,
        )
        rows = [Picture.serialize_with_likeness(r) for r in results]
        # Per row, whether the picture's own text matched: the result pill counts
        # and narrows on this without a second search (#1197).
        text_matched_ids = (
            server.vault.db.run_immediate_read_task(
                Picture.ids_matching_text, query, [row.get("id") for row in rows]
            )
            if rows
            else set()
        )
        for row in rows:
            row["text_match"] = row.get("id") in text_matched_ids
        # Rows come from `metadata_fields()`, which carries the raw
        # `Picture.project_id`, and `PictureMetadataResponse` sets
        # `extra="allow"` so the response model filters nothing (issue #719,
        # §16.6).
        narrow_picture_project_ids(server, request, rows)
        return rows
