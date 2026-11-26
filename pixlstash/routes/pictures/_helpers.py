import concurrent.futures
import os
import re
import uuid

from fastapi import (
    HTTPException,
    Request,
)
from sqlalchemy import (
    func,
)
from sqlmodel import Session, select

from pixlstash.db_models import (
    Picture,
    Tag,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.services import face_clusters, import_dedup_service, scrapheap_service
from pixlstash.services.layout_move_service import resolve_placement
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.service.caption_utils import (
    normalize_hidden_tags,
)
from pixlstash.utils.service.scope_table import scope_id_subquery
from pixlstash.utils.sql_chunking import chunked, in_json
from pixlstash.utils.sqlite_ints import is_sqlite_int, parse_sqlite_int


logger = get_logger(__name__)

_ONE_PERSON_DETAIL = "face_cluster needs one person's character_id"


_SIDECAR_TAG_PATTERN = re.compile(
    r"^[a-z0-9]+(?:[ _-][a-z0-9]+){0,2}$",
    re.IGNORECASE,
)

MEDIA_TYPE_BY_FORMAT = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "avif": "image/avif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "m4v": "video/mp4",
}


def _face_cluster_ids(
    server, request: Request, character_id, cluster_raw, threshold_raw
):
    """The picture ids of the person's "Cluster N" (``face_cluster``), or
    ``None`` without one. A cluster past the last is an empty set.

    Owner only, as ``GET /face-clusters`` is: the grouping spans the person's
    whole library, so a scoped token could read it through the pictures it
    can see. *character_id* is an int for one person; anything else is a 422.
    """
    if cluster_raw is None:
        return None
    server.auth.require_unscoped_owner(request)
    if isinstance(character_id, str) and character_id.isdigit():
        try:
            character_id = parse_sqlite_int(character_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_ONE_PERSON_DETAIL) from exc
    if not is_sqlite_int(character_id):
        raise HTTPException(status_code=422, detail=_ONE_PERSON_DETAIL)
    try:
        number = int(cluster_raw)
        threshold = (
            face_clusters.SAME_PERSON if threshold_raw is None else float(threshold_raw)
        )
    except (TypeError, ValueError) as exc:
        # The raw values are not echoed: neither parsed.
        raise HTTPException(
            status_code=422,
            detail="face_cluster must be a whole number and "
            "face_cluster_threshold a number",
        ) from exc
    if number < 1 or not 0.05 <= threshold <= 0.95:
        raise HTTPException(
            status_code=422,
            detail="face_cluster counts from 1; face_cluster_threshold is 0.05-0.95",
        )
    clusters = face_clusters.person_cluster_ids(server.vault, character_id, threshold)
    if clusters is None or number > len(clusters):
        return set()
    return set(clusters[number - 1])


def _get_hidden_tags_from_request(server, request: Request) -> list[str]:
    if request.query_params.get("apply_tag_filter", "").lower() != "true":
        return []
    try:
        user = server.auth.get_user_for_request(request)
    except HTTPException:
        user = server.auth.get_user()
    if not user:
        return []
    normalized = normalize_hidden_tags(getattr(user, "hidden_tags", None))
    return normalized or []


def _fetch_hidden_picture_ids(server, request: Request, picture_ids: list[int]):
    hidden_tags = _get_hidden_tags_from_request(server, request)
    if not hidden_tags or not picture_ids:
        return set()
    hidden_tag_set = {str(tag).strip().lower() for tag in hidden_tags if tag}

    def fetch_hidden(session: Session, ids: list[int], tags: set[str]):
        # Every row of a search or listing, so possibly more ids than SQLite
        # binds in one statement: read from a temp table instead. The owner's
        # hidden tags have no cap either, so they bind as one JSON array.
        id_scope = scope_id_subquery(session, ids, name="_pixlstash_hidden_check_ids")
        rows = session.exec(
            select(Tag.picture_id).where(
                Tag.picture_id.in_(id_scope),
                Tag.tag.is_not(None),
                in_json(func.lower(Tag.tag), tags),
            )
        ).all()
        return {row for row in rows if row is not None}

    return server.vault.db.run_immediate_read_task(
        fetch_hidden, list(picture_ids), hidden_tag_set
    )


def _create_picture_imports(
    server,
    uploaded_files,
    dest_folder,
    progress_callback=None,
    on_file_failed=None,
    compared: set[int] | None = None,
):
    """
    Given a list of (content, ext, name), create Picture objects for new images,
    skipping content already in the vault (live OR scrapheaped) by pixel_sha.
    Returns (fingerprints, existing_map, scrapheaped_map, new_picture_map)

    **A scrapheaped match is skipped too, and returned separately** for the
    caller to restore or report. This call
    site used to ask ``Picture.find(..., pixel_shas=shas)``, whose
    ``include_deleted`` defaults to False, so a soft-deleted picture was
    invisible here and its file imported again as a brand-new second row. The
    lookup now goes through :mod:`pixlstash.services.import_dedup_service`,
    which sees soft-deleted rows *for this query only*, ``Picture.find``'s
    default is unchanged, so no listing, search, count or dedup query gains
    deleted rows. See that module for the full rationale and for why a
    permanently purged file is correctly NOT a match.

    Args:
        server: The server instance.
        uploaded_files: List of ``(content, ext, name)`` tuples. *content*
            is the file's bytes, or the path of a file holding them, which is
            hashed and copied a chunk at a time and never read whole (the
            one-shot upload route spools its uploads to disk). A path is
            copied, not moved: the caller removes it.
        dest_folder: Destination folder for images.
        progress_callback: Optional callable invoked after each image is written
            to disk. Receives no arguments. Used for incremental progress tracking.
        on_file_failed: Optional callable ``(index, exc)`` for a file that
            could not be published (undecodable, unreadable, no thumbnail).
            With it, the failure is handed over and the batch goes on: one bad
            file does not cost the batch its other files. Without it, the
            files this call already published are removed and the error is
            raised, so a caller that stops does not leave files behind with
            no row, which the next root scan would index as new pictures.
            Either way a failed file has no entry in ``new_picture_map``.
        compared: Optional set the ids of the candidate pictures this lookup
            compared the files with are added to. Hand it to
            :func:`import_publish_service.withdraw_imported_meanwhile`, so
            the look again under the import lock reads no file for a
            candidate already compared here.

    Returns:
        ``(fingerprints, existing_map, scrapheaped_map, new_picture_map)``.
        Maps use ``(sampled_sha, size_bytes, full_sha)`` keys, so sampled-hash
        collisions remain distinct throughout a batch. ``new_picture_map`` has
        one Picture per unique new file; repeated identical uploads share it.
    """

    def create_fingerprint(content):
        if isinstance(content, str):
            return (
                ImageUtils.calculate_hash_from_file_path(content),
                os.path.getsize(content),
                ImageUtils.calculate_full_hash_from_file_path(content),
            )
        return (
            ImageUtils.calculate_hash_from_bytes(content),
            len(content),
            ImageUtils.calculate_full_hash_from_bytes(content),
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fingerprints = list(
            executor.map(
                create_fingerprint,
                (content for content, *_ in uploaded_files),
            )
        )

    candidates = server.vault.db.run_immediate_read_task(
        import_dedup_service.load_match_candidates_in_session,
        [(sampled, size) for sampled, size, _full in fingerprints],
    )
    existing_map, scrapheaped_map = import_dedup_service.partition_confirmed_matches(
        candidates, fingerprints, server.vault.image_root
    )
    if compared is not None:
        compared.update(row.id for rows in candidates.values() for row in rows)

    # Placement on write (v1.11 Phase 4b). Resolved once for the batch: every
    # file in it goes to the same folder, and `None` means this library has no
    # layout - or the caller is writing somewhere that is not its root.
    subfolder = resolve_placement(server.vault.db, dest_folder)

    new_picture_map = {}
    for index, (file_entry, fingerprint) in enumerate(
        zip(uploaded_files, fingerprints)
    ):
        if (
            fingerprint in existing_map
            or fingerprint in scrapheaped_map
            or fingerprint in new_picture_map
        ):
            continue
        sampled_sha, size_bytes, _full_sha = fingerprint
        content, ext, original_name = file_entry
        pic_uuid = str(uuid.uuid4()) + ext
        logger.debug(f"Importing picture from uploaded content as id={pic_uuid}")
        try:
            if isinstance(content, str):
                picture = ImageUtils.create_picture_from_file(
                    image_root_path=dest_folder,
                    source_file_path=content,
                    picture_uuid=pic_uuid,
                    pixel_sha=sampled_sha,
                    subfolder=subfolder,
                    record_source_name=False,
                )
                # The spooled file's name is the route's; the upload's is
                # the picture's.
                picture.original_file_name = original_name
            else:
                picture = ImageUtils.create_picture_from_bytes(
                    image_root_path=dest_folder,
                    image_bytes=content,
                    picture_uuid=pic_uuid,
                    pixel_sha=sampled_sha,
                    original_file_name=original_name,
                    subfolder=subfolder,
                )
            new_picture_map[fingerprint] = picture
        except Exception as exc:
            if on_file_failed is None:
                logger.error(
                    "Import of file %d of %d (%s) failed; removing the %d file(s) "
                    "this batch already published: %s",
                    index + 1,
                    len(uploaded_files),
                    original_name,
                    len(new_picture_map),
                    exc,
                )
                for published in new_picture_map.values():
                    ImageUtils.unpublish_picture(dest_folder, published)
                raise
            logger.warning(
                "Import of file %d of %d (%s, %d bytes) failed; importing the rest: %s",
                index + 1,
                len(uploaded_files),
                original_name,
                size_bytes,
                exc,
                exc_info=True,
            )
            on_file_failed(index, exc)
            continue
        if progress_callback is not None:
            progress_callback()

    return fingerprints, existing_map, scrapheaped_map, new_picture_map


def _normalise_sidecar_stem(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename or ""))[0].strip().lower()


def _normalise_vocab_token(value: str) -> str:
    if not value:
        return ""
    return " ".join(str(value).replace("_", " ").strip().lower().split())


def _parse_sidecar_tags(raw_text: str) -> list[str]:
    text = (raw_text or "").strip()
    if not text or "," not in text:
        return []

    tags_raw = [part.strip() for part in text.replace("\n", ",").split(",")]
    tags_raw = [tag for tag in tags_raw if tag]
    if len(tags_raw) < 2:
        return []

    seen = set()
    parsed = []
    for raw_tag in tags_raw:
        # Lenient sanity check: 1-3 words per tag using space/dash/underscore.
        compact_raw = " ".join(raw_tag.strip().split())
        if not _SIDECAR_TAG_PATTERN.fullmatch(compact_raw):
            continue
        # Preserve sidecar tag semantics (e.g. "1girl") while still
        # normalising separators/spacing for storage.
        candidate = _normalise_vocab_token(compact_raw)
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        parsed.append(candidate)
    return parsed


def _enrich_scrapheap_retention(server, pics: list[dict]) -> list[dict]:
    """Add ``purge_at``, ``auto_purge_exempt`` and ``auto_purge_exempt_reason``.

    All three are computed server-side so the countdown, the badge, and the
    retention grace maths have exactly one implementation. The values are read
    fresh from the DB (``deleted_at``, ``reference_folder_id``, locked-set
    membership) rather than from the request's field projection, so
    ``fields=grid`` gets them too.

    **The listing must agree with the sweep.** Both exemptions the sweep honours
    are applied here, through the same helpers the finder uses
    (``fetch_no_delete_folder_ids`` and ``locked_scrapheap_picture_ids``, the
    latter covering the live-stack-sibling freeze), so the two can never
    disagree. Omitting the lock check made the grid render a permanent, urgent
    "purges today" countdown on a locked picture the sweep would never touch.

    Args:
        server: The application server with vault access.
        pics: Scrapheap picture dicts to enrich (mutated in place).

    Returns:
        The same list, with all three keys set on every dict.
    """
    if not pics:
        return pics
    picture_ids = [
        int(p.get("id"))
        for p in pics
        if isinstance(p, dict) and p.get("id") is not None
    ]
    if not picture_ids:
        return pics

    rows = scrapheap_service.fetch_scrapheap_rows(server.vault, picture_ids)
    no_delete_folder_ids = scrapheap_service.fetch_no_delete_folder_ids(server.vault)
    locked_ids = scrapheap_service.locked_scrapheap_picture_ids(
        server.vault, picture_ids
    )
    retention_days = server.vault.scrapheap_retention_days
    reduced_at = server.vault.scrapheap_retention_reduced_at

    reason_by_id: dict[int, str | None] = {}
    purge_at_by_id: dict[int, str | None] = {}
    for row in rows:
        if row.id is None:
            continue
        row_id = int(row.id)
        is_protected = row.is_protected(no_delete_folder_ids)
        is_locked = row_id in locked_ids
        purge_at = scrapheap_service.compute_purge_at(
            row.deleted_at, retention_days, reduced_at, is_protected, is_locked
        )
        reason_by_id[row_id] = scrapheap_service.auto_purge_exemption(
            is_protected, is_locked
        )
        purge_at_by_id[row_id] = purge_at.isoformat() if purge_at else None

    for pic in pics:
        if not isinstance(pic, dict):
            continue
        pic_id = pic.get("id")
        pic_id = int(pic_id) if pic_id is not None else None
        if pic_id not in reason_by_id:
            # Vanished between the listing query and this enrichment: report it
            # as exempt with no deadline rather than advertising a countdown we
            # did not actually compute. The reason is null because we no longer
            # know which exemption would have applied.
            pic["purge_at"] = None
            pic["auto_purge_exempt"] = True
            pic["auto_purge_exempt_reason"] = None
            continue
        reason = reason_by_id[pic_id]
        pic["auto_purge_exempt"] = reason is not None
        pic["auto_purge_exempt_reason"] = reason
        pic["purge_at"] = purge_at_by_id.get(pic_id)
    return pics


def _enrich_stack_counts(server, pics: list[dict]) -> list[dict]:
    """Add stack_count field to each dict in pics by querying the DB.

    Args:
        server: The application server with vault.db access.
        pics: List of picture dicts to enrich.

    Returns:
        New list with a stack_count key added to every dict.
    """
    if not pics:
        return pics
    picture_ids = [
        int(p.get("id"))
        for p in pics
        if isinstance(p, dict) and p.get("id") is not None
    ]
    if not picture_ids:
        return pics

    def fetch_stack_info(session: Session, ids: list[int]):
        # A listing page has no fixed cap (``limit`` defaults to
        # ``sys.maxsize``), so both reads run in chunks under SQLite's
        # bound-parameter limit. The ids and stack ids are unique, and the
        # count groups by the chunked column, so no row or count is split.
        id_stack_rows = [
            row
            for id_chunk in chunked(sorted(set(ids)))
            for row in session.exec(
                select(Picture.id, Picture.stack_id).where(
                    Picture.id.in_(id_chunk),
                    Picture.deleted.is_(False),
                )
            ).all()
        ]
        stack_ids = sorted(
            {
                int(stack_id)
                for _pic_id, stack_id in id_stack_rows
                if stack_id is not None
            }
        )
        if not stack_ids:
            return id_stack_rows, []
        stack_count_rows = [
            row
            for stack_chunk in chunked(stack_ids)
            for row in session.exec(
                select(Picture.stack_id, func.count(Picture.id))
                .where(
                    Picture.stack_id.in_(stack_chunk),
                    # Not ``IS 0``, which beats the stack index from one id ("Probe cost").
                    Picture.deleted.is_not(True),
                )
                .group_by(Picture.stack_id)
            ).all()
        ]
        return id_stack_rows, stack_count_rows

    id_stack_rows, stack_count_rows = server.vault.db.run_immediate_read_task(
        fetch_stack_info, picture_ids
    )
    stack_id_by_picture_id = {
        int(pic_id): stack_id for pic_id, stack_id in id_stack_rows
    }
    stack_count_by_stack_id = {
        int(stack_id): int(count)
        for stack_id, count in stack_count_rows
        if stack_id is not None
    }
    enriched: list[dict] = []
    for pic in pics:
        if not isinstance(pic, dict):
            enriched.append(pic)
            continue
        picture_id = pic.get("id")
        if picture_id is None:
            enriched.append(pic)
            continue
        numeric_id = int(picture_id)
        stack_id = pic.get("stack_id")
        if stack_id is None:
            stack_id = stack_id_by_picture_id.get(numeric_id)
        stack_count = 0
        if stack_id is not None:
            stack_count = stack_count_by_stack_id.get(int(stack_id), 1)
        enriched.append(
            {
                **pic,
                "stack_id": stack_id,
                "stack_count": stack_count,
            }
        )
    return enriched


# enforce_picture_scope and its private _picture_id_in_scoped_* helpers live in
# pixlstash/authz/membership.py - the single home for object-membership checks.
# The centralised authz gate calls them; handlers no longer do (Step 5).
