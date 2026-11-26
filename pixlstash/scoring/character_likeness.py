"""Character-likeness scoring: face↔reference similarity via InsightFace embeddings.

Split out of the former ``pixlstash.picture_scoring`` module (Backend Refactor
Phase 2 §4.6). Smart-score computation lives in the sibling
:mod:`pixlstash.scoring.smart_score`. Import the public names from
:mod:`pixlstash.scoring`.
"""

import struct
import time

import numpy as np
from sqlalchemy import and_, asc, case, desc, exists, func, literal, or_, text
from sqlalchemy.orm import aliased
from sqlalchemy.sql import Select
from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models import (
    DEFAULT_SMART_SCORE_PENALIZED_TAGS,
    DEFAULT_SMART_SCORE_PENALIZED_TAG_WEIGHT,
    Face,
    Picture,
    PictureSetMember,
    Tag,
)
from pixlstash.services.face_clusters import reference_cluster_faces
from pixlstash.utils.quality.smart_score_utils import smart_score_penalised_tags
from pixlstash.utils.service.filter_helpers import combine_likeness_scores
from pixlstash.utils.service.scope_table import scope_id_subquery
from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)


def select_reference_faces_for_character(
    session: Session,
    character_id: int,
    max_refs: int = 10,
) -> list[Face]:
    """Select reference faces for a character using simple, deterministic rules.

    With a reference cluster (``services/face_clusters.py``) the references
    are the person's faces on it, most central first. Without one: 5-star
    pictures, then 4-star, then the rest ranked by how many penalised tags
    their picture carries, each stage oldest picture first, then picture id,
    then face id.

    Every stage is one ``LIMIT``ed statement. SQLite (no ``ANALYZE``
    statistics here) drives the two rated stages from the score range of
    ``ix_picture_grid_score`` and the rest from the person's faces; measured
    on a 60k-face person and on a library of 60k 5-star pictures, that beats
    forcing the face-driven plan for every stage (docs/backend_architecture.md
    §25.30). The rest is ranked in SQL with a per-picture penalised-tag
    weight, so only the faces picked are loaded, never the person's whole
    face list.
    No stage loads ``Picture`` rows, whose embeddings are not deferred.

    Args:
        session: Database session to query faces and pictures.
        character_id: Character id to select reference faces for.
        max_refs: Maximum number of reference faces to return.

    Returns:
        A list of Face objects to use as reference faces.
    """

    chosen = reference_cluster_faces(session, character_id, max_refs)
    if chosen:
        return chosen

    min_refs = min(5, max_refs)

    eligible = (
        Face.character_id == character_id,
        Face.features.is_not(None),
        Picture.deleted.is_(False),
    )
    base_query = (
        select(Face).join(Picture, Face.picture_id == Picture.id).where(*eligible)
    )
    oldest_first = (
        Picture.created_at.asc(),
        Picture.id.asc(),
        Face.id.asc(),
    )

    representatives = list(
        session.exec(
            base_query.where(Picture.score >= 5).order_by(*oldest_first).limit(max_refs)
        ).all()
    )

    logger.debug(
        "[reference_faces] character_id=%s target_count=%s five_star_rows=%s",
        character_id,
        max_refs,
        len(representatives),
    )

    if len(representatives) >= max_refs:
        return representatives

    selected_face_ids = {face.id for face in representatives}
    selected_picture_ids = {face.picture_id for face in representatives}

    remaining_faces = session.exec(
        base_query.where(Picture.score >= 4)
        .where(~Picture.id.in_(selected_picture_ids))
        .order_by(*oldest_first)
        .limit(max_refs - len(representatives))
    ).all()
    logger.debug(
        "[reference_faces] character_id=%s four_five_rows=%s selected_pictures=%s",
        character_id,
        len(remaining_faces),
        len(selected_picture_ids),
    )
    for face in remaining_faces:
        if face.id in selected_face_ids:
            continue
        selected_face_ids.add(face.id)
        representatives.append(face)

    if len(representatives) >= min_refs:
        return representatives

    # The rest: least penalised picture first, then oldest; a picture with no
    # date goes last. Stage 1's pictures are excluded, stage 2's faces too.
    rest_faces = session.exec(
        base_query.where(~Picture.id.in_(selected_picture_ids))
        .where(~Face.id.in_(selected_face_ids))
        .order_by(
            _penalised_tag_weight(),
            Picture.created_at.is_(None),
            *oldest_first,
        )
        .limit(min_refs - len(representatives))
    ).all()
    logger.debug(
        "[reference_faces] character_id=%s rest_rows=%s selected_pictures=%s",
        character_id,
        len(rest_faces),
        len(selected_picture_ids),
    )
    for face in rest_faces:
        selected_face_ids.add(face.id)
        representatives.append(face)

    if len(representatives) >= min_refs:
        return representatives

    fallback_face = session.exec(
        base_query.order_by(desc(Picture.score), *oldest_first).limit(1)
    ).first()
    if fallback_face and fallback_face.id not in selected_face_ids:
        representatives.append(fallback_face)

    logger.debug(
        "[reference_faces] character_id=%s final_faces=%s",
        character_id,
        len(representatives),
    )

    return representatives


def _penalised_tag_weight():
    """Return the per-picture sum of penalised-tag weights, as a SQL expression.

    Correlated on ``Picture.id``; a picture with no penalised tag weighs 0.
    Tags match case-insensitively on the stored text, as the smart score's
    penalty does. The names and weights are the built-in defaults, rendered
    into the SQL rather than bound, so the statement binds no parameter per
    tag under SQLite's per-statement limit
    (https://docs.sqlalchemy.org/en/20/core/sqlelement.html#sqlalchemy.sql.expression.literal).
    """
    penalised_tags = smart_score_penalised_tags(
        None,
        DEFAULT_SMART_SCORE_PENALIZED_TAGS,
        default_weight=DEFAULT_SMART_SCORE_PENALIZED_TAG_WEIGHT,
    )
    weights = {
        name: float(penalised_tags.get(name, 0.0))
        for name in {str(tag).strip().lower() for tag in penalised_tags if tag}
    }
    zero = literal(0.0, literal_execute=True)
    if not weights:
        return zero
    tag_name = func.lower(Tag.tag)
    weight = case(
        *(
            (
                tag_name == literal(name, literal_execute=True),
                literal(value, literal_execute=True),
            )
            for name, value in sorted(weights.items())
        ),
        else_=zero,
    )
    return (
        select(func.coalesce(func.sum(weight), zero))
        .where(Tag.picture_id == Picture.id)
        .where(tag_name.in_([literal(name, literal_execute=True) for name in weights]))
        .correlate(Picture)
        .scalar_subquery()
    )


def compute_character_likeness_for_faces(
    reference_faces: list[Face],
    candidate_faces: list[Face],
    combine: str = "softmax",
) -> dict[int, float]:
    """Compute likeness scores for candidate faces against reference faces.

    Args:
        reference_faces: Reference faces to compare against.
        candidate_faces: Candidate faces to score.
        combine: How to aggregate each candidate face's cosine similarity
            across the character's multiple reference faces.

            - ``"softmax"`` (default): the legacy softmax-weighted average
              (alpha=5), which leans toward the best-matching reference faces.
              This is the behaviour every existing caller relied on, so it stays
              the default and their scores are unchanged.
            - ``"mean"`` / ``"max"`` / ``"min"`` / ``"harmonic_mean"`` /
              ``"geometric_mean"``: reduce across reference faces via
              :func:`combine_likeness_scores`. ``"max"`` scores a face on its
              single best-matching reference (lenient), ``"min"`` requires
              matching every reference (strict), ``"mean"`` is the plain average.

    Returns:
        A mapping of face_id to likeness score.
    """

    if not reference_faces or not candidate_faces:
        return {}

    ref_arrs = []
    for ref_face in reference_faces:
        if ref_face.features is None:
            continue
        ref_arr = np.frombuffer(ref_face.features, dtype=np.float32)
        if ref_arr.size == 0:
            continue
        ref_arrs.append(ref_arr)

    if not ref_arrs:
        return {}

    face_vectors = []
    face_ids = []
    for face in candidate_faces:
        if face.features is None:
            continue
        arr_face = np.frombuffer(face.features, dtype=np.float32)
        if arr_face.size == 0:
            continue
        face_vectors.append(arr_face)
        face_ids.append(face.id)

    if not face_vectors:
        return {}

    cand = np.stack(face_vectors)
    ref = np.stack(ref_arrs)
    cand_norm = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-8)
    ref_norm = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-8)
    # (N_cand, N_ref) per-candidate, per-reference cosine similarity.
    sims = np.clip(cand_norm @ ref_norm.T, -1.0, 1.0)

    if combine == "softmax":
        alpha = 5.0
        weights = np.exp(alpha * sims)
        denom = np.sum(weights, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        per_candidate = np.sum(weights * sims, axis=1) / denom.squeeze(1)
    else:
        # combine_likeness_scores reduces across axis 0; we want to reduce across
        # reference faces (axis 1), so transpose to (N_ref, N_cand) and it
        # returns one score per candidate face.
        per_candidate = combine_likeness_scores(sims.T, combine)

    return {
        face_id: float(likeness)
        for face_id, likeness in zip(face_ids, per_candidate, strict=False)
    }


def match_faces_to_characters(
    session: Session,
    faces: list[Face],
    character_ids: list[int],
) -> dict[int, int]:
    """Say which of one picture's *faces* is which of *character_ids*.

    Liberal on purpose: the run loaded these people, so they are in it. Each
    person is scored against every face that names nobody yet, by likeness to
    their reference faces, and the strongest pair is taken first, then the
    strongest of what is left, and so on: the obvious face settles its person
    and the less obvious ones take what remains. A person gets at most one
    face, a face at most one person, and a person some face already names is
    left out.

    Whoever is still without a face (no reference faces to score, or no face
    with features left) takes the largest face nobody took, in turn, until
    the people or the faces run out.

    Args:
        session: Database session to read reference faces with.
        faces: The picture's real faces (no ``face_index == -1`` sentinel).
        character_ids: The people the picture is of.

    Returns:
        face id -> character id, for the faces to assign.
    """
    named = {face.character_id for face in faces if face.character_id is not None}
    people = [cid for cid in dict.fromkeys(character_ids) if cid not in named]
    free = [face for face in faces if face.character_id is None]
    candidates = [face for face in free if face.features]
    scored = []
    for character_id in people:
        likeness = compute_character_likeness_for_faces(
            select_reference_faces_for_character(session, character_id),
            candidates,
        )
        scored.extend(
            (score, character_id, face_id) for face_id, score in likeness.items()
        )

    pairs: dict[int, int] = {}
    for _score, character_id, face_id in sorted(scored, reverse=True):
        if face_id not in pairs and character_id not in pairs.values():
            pairs[face_id] = character_id

    left_people = [cid for cid in people if cid not in pairs.values()]
    left_faces = sorted(
        (face for face in free if face.id not in pairs),
        key=lambda face: (face.width or 0) * (face.height or 0),
        reverse=True,
    )
    for character_id, face in zip(left_people, left_faces):
        pairs[face.id] = character_id
    return pairs


def find_pictures_by_character_likeness(
    server,
    character_id,
    reference_character_id,
    offset,
    limit,
    descending,
    candidate_ids=None,
):
    """List pictures by likeness to a character.

    Args:
        server: The server object.
        character_id: Character id to filter pictures by (or "ALL" or "UNASSIGNED").
        reference_character_id: Character id to use as reference for likeness scoring.
        offset: The number of items to skip before starting to collect the result set.
        limit: The maximum number of items to return.
        descending: Whether to sort in descending order.
        candidate_ids: Optional list of candidate picture ids to filter by.
    """
    reference_character_id = int(reference_character_id)

    timing_start = time.perf_counter()

    reference_faces = server.vault.db.run_task(
        select_reference_faces_for_character,
        reference_character_id,
        10,
        priority=DBPriority.IMMEDIATE,
    )
    timing_after_refs = time.perf_counter()

    if not reference_faces:
        logger.warning("No reference faces found for character id=%s", character_id)
        return []

    def get_all_faces(session, character_id, candidate_ids=None):
        query = select(Face).join(Picture, Face.picture_id == Picture.id)
        if character_id == "ALL" or character_id is None:
            pass
        elif character_id == "UNASSIGNED":
            query = query.where(Face.character_id.is_(None))
        else:
            query = query.where(Face.character_id == int(character_id))
        if candidate_ids is not None:
            if not candidate_ids:
                return []
            # A candidate list (a set's members, a face cluster) has no size
            # cap, so it is read from a temp table rather than bound per id.
            query = query.where(
                Face.picture_id.in_(
                    scope_id_subquery(
                        session, candidate_ids, name="_pixlstash_likeness_candidates"
                    )
                )
            )
        return session.exec(query).all()

    candidate_faces = server.vault.db.run_task(
        get_all_faces, character_id, candidate_ids
    )
    timing_after_candidates = time.perf_counter()
    if not candidate_faces:
        logger.warning("No unassigned faces found")
        return []

    character_likeness_map = compute_character_likeness_for_faces(
        reference_faces,
        candidate_faces,
    )
    if not character_likeness_map:
        logger.warning(
            "No reference face features found for character id=%s", character_id
        )
        return []
    timing_after_likeness = time.perf_counter()

    picture_likeness_map = {}
    for face in candidate_faces:
        pic_id = face.picture_id
        likeness = character_likeness_map.get(face.id, 0.0)
        if pic_id not in picture_likeness_map:
            picture_likeness_map[pic_id] = likeness
        else:
            picture_likeness_map[pic_id] = max(picture_likeness_map[pic_id], likeness)

    sorted_ids = sorted(
        picture_likeness_map.items(),
        key=lambda item: item[1],
        reverse=descending,
    )
    sorted_ids = [pid for pid, _ in sorted_ids]

    if character_id == "UNASSIGNED" and sorted_ids:

        def filter_unassigned_ids(session: Session, picture_ids: list[int]):
            if not picture_ids:
                return []
            assigned_faces = exists(
                select(Face.id).where(
                    Face.picture_id == Picture.id,
                    Face.character_id.is_not(None),
                )
            )
            in_set = exists(
                select(PictureSetMember.picture_id).where(
                    PictureSetMember.picture_id == Picture.id
                )
            )
            rows = session.exec(
                select(Picture.id)
                .where(
                    Picture.id.in_(
                        scope_id_subquery(
                            session, picture_ids, name="_pixlstash_likeness_unassigned"
                        )
                    )
                )
                .where(~assigned_faces)
                .where(~in_set)
                .where(Picture.deleted.is_(False))
            ).all()
            return [row for row in rows]

        eligible_ids = set(server.vault.db.run_task(filter_unassigned_ids, sorted_ids))
        sorted_ids = [pid for pid in sorted_ids if pid in eligible_ids]

    selected_ids = sorted_ids[offset : offset + limit]
    if not selected_ids:
        return []

    candidate_pics = server.vault.db.run_task(
        Picture.find,
        id=selected_ids,
        select_fields=Picture.metadata_fields(),
    )
    timing_after_fetch = time.perf_counter()

    logger.debug(
        "[LIKELINESS TIMING] refs=%.3fms candidates=%.3fms likeness=%.3fms fetch=%.3fms total=%.3fms",
        (timing_after_refs - timing_start) * 1000.0,
        (timing_after_candidates - timing_after_refs) * 1000.0,
        (timing_after_likeness - timing_after_candidates) * 1000.0,
        (timing_after_fetch - timing_after_likeness) * 1000.0,
        (timing_after_fetch - timing_start) * 1000.0,
    )

    # Imported here, not at the top: the listing module imports
    # ``pixlstash.scoring``, so a module-level import is circular.
    from pixlstash.routes.pictures._listing import stored_picture_dict

    pic_map = {pic.id: pic for pic in candidate_pics}
    results = []
    for pic_id in selected_ids:
        pic = pic_map.get(pic_id)
        if not pic:
            continue
        # Read through the column types: a stored value its column does not
        # declare is null in its row, not a 500 for the page.
        pic_dict = stored_picture_dict(pic)
        pic_dict["character_likeness"] = max(0.0, picture_likeness_map.get(pic_id, 0.0))
        results.append(pic_dict)

    return results


def pack_reference_blobs(reference_faces: list) -> bytes | None:
    """Pack reference face feature vectors into a binary blob for the character_face_likeness SQL function.

    Args:
        reference_faces: List of Face objects whose features will be packed.

    Returns:
        Bytes object with header (n_refs, vec_size as little-endian int32) followed by
        pre-normalised float32 vectors concatenated, or None if no valid features found.
    """
    vecs = []
    for face in reference_faces:
        if face.features is None:
            continue
        arr = np.frombuffer(face.features, dtype=np.float32)
        if arr.size == 0:
            continue
        vecs.append(arr)
    if not vecs:
        return None
    vec_size = vecs[0].size
    vecs = [v for v in vecs if v.size == vec_size]
    if not vecs:
        return None
    matrix = np.stack(vecs).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    matrix_norm = (matrix / norms).astype(np.float32)
    header = struct.pack("<ii", len(vecs), vec_size)
    return header + matrix_norm.tobytes()


def _scoped_stack_leader_clause(character_id, candidate_scope: Select | None):
    """Build the scoped-leader stack-collapse clause for CHARACTER_LIKENESS queries.

    Mirrors the ``id_scope`` branch of ``Picture.find`` (picture.py): a stack is
    represented by its lowest-positioned live member *inside the filtered scope*,
    instead of being dropped whenever its global position-0 leader is out of
    scope (e.g. the leader has no face assigned to the viewed character while a
    child does). Shared by the list and count queries so the two can never
    drift.

    The sibling scope is: not deleted, has a Face row satisfying the same
    character filter the outer query applies, and (when ``candidate_ids`` is
    given) is itself in the candidate set - the candidate scope carries
    token-scope narrowing, so omitting it would let a share-scoped stack be
    represented by an out-of-scope sibling. Tie-break is identical to
    ``Picture.find``: ``coalesce(stack_position, 999999)`` ascending, then
    lower id wins.

    Args:
        character_id: Character filter (int, None/""/"ALL", or "UNASSIGNED").
        candidate_scope: Optional picture-id restriction (token scope / filters),
            as a SELECT of the ids (``scope_id_subquery``) so its length binds
            no parameters.

    Returns:
        A SQLAlchemy boolean clause to apply with ``.where()``.
    """
    sibling = aliased(Picture)
    sib_face = aliased(Face)
    if character_id == "UNASSIGNED":
        other_sib_face = aliased(Face)
        sibling_in_scope = and_(
            exists(
                select(sib_face.id).where(
                    sib_face.picture_id == sibling.id,
                    sib_face.character_id.is_(None),
                )
            ),
            ~exists(
                select(other_sib_face.id).where(
                    other_sib_face.picture_id == sibling.id,
                    other_sib_face.character_id.is_not(None),
                )
            ),
        )
    elif character_id is not None and character_id != "" and character_id != "ALL":
        sibling_in_scope = exists(
            select(sib_face.id).where(
                sib_face.picture_id == sibling.id,
                sib_face.character_id == int(character_id),
            )
        )
    else:
        sibling_in_scope = exists(
            select(sib_face.id).where(sib_face.picture_id == sibling.id)
        )

    # ``deleted IS NOT 1`` rather than ``deleted IS 0`` on purpose. As an
    # equality term ``deleted`` lets the planner serve the sibling lookup from
    # ``ix_picture_deleted``, which ties on cost with ``ix_picture_stack_id``
    # (no ``sqlite_stat1``, see ``Picture.__table_args__``) and wins or loses
    # on index-creation order. When it won, every face row walked every live
    # picture: 6.5 s for one page on a 12k-picture library, 0.13 s the other
    # way. ``IS NOT`` is not an indexable term, so only the stack_id indexes
    # can answer the correlated lookup, whichever order the indexes were built
    # in. (``~sibling.deleted`` does not do this: SQLAlchemy compiles it to
    # ``deleted = 0`` for SQLite, which is indexable again.)
    conditions = [
        sibling.stack_id == Picture.stack_id,
        sibling.deleted.is_not(True),
        sibling_in_scope,
    ]
    if candidate_scope is not None:
        conditions.append(sibling.id.in_(candidate_scope))
    cur_pos = func.coalesce(Picture.stack_position, 999999)
    sib_pos = func.coalesce(sibling.stack_position, 999999)
    conditions.append(
        or_(
            sib_pos < cur_pos,
            and_(sib_pos == cur_pos, sibling.id < Picture.id),
        )
    )
    has_higher_ranked_sibling = exists(select(sibling.id).where(*conditions))
    return or_(Picture.stack_id.is_(None), ~has_higher_ranked_sibling)


def find_pictures_by_character_likeness_sql(
    server,
    character_id,
    reference_character_id,
    offset: int,
    limit: int,
    descending: bool,
    candidate_ids: list[int] | None = None,
    deleted_only: bool = False,
    stack_leaders_only: bool = False,
) -> list[dict]:
    """List pictures by character likeness using SQL ORDER BY with LIMIT/OFFSET.

    Uses the character_face_likeness SQLite scalar function so sorting and pagination
    happen entirely at the SQL layer, enabling the fast grid streaming path.

    Args:
        server: The server object.
        character_id: Character id to filter candidate faces by (int, None/"ALL", or "UNASSIGNED").
        reference_character_id: Character whose reference faces define the likeness target.
        offset: Number of rows to skip.
        limit: Maximum number of rows to return.
        descending: If True, highest-likeness pictures come first.
        candidate_ids: Optional list of picture ids to restrict the search to.
        deleted_only: If True, restrict to deleted (scrapheap) pictures only.
        stack_leaders_only: If True, collapse each stack to one representative
            using scoped-leader semantics (see _scoped_stack_leader_clause).

    Returns:
        List of picture metadata dicts with a "character_likeness" field added.
    """
    reference_character_id = int(reference_character_id)
    timing_start = time.perf_counter()

    reference_faces = server.vault.db.run_task(
        select_reference_faces_for_character,
        reference_character_id,
        10,
        priority=DBPriority.IMMEDIATE,
    )
    if not reference_faces:
        logger.warning(
            "No reference faces found for character id=%s", reference_character_id
        )
        return []

    refs_blob = pack_reference_blobs(reference_faces)
    if refs_blob is None:
        logger.warning(
            "No valid reference face features for character id=%s",
            reference_character_id,
        )
        return []

    timing_after_refs = time.perf_counter()

    def run_query(session: Session):
        # The candidate list has no size cap, so it is read from a temp table:
        # the GROUP BY, ORDER BY and LIMIT/OFFSET below then hold over the
        # whole list, which chunking could not keep.
        candidate_scope = (
            scope_id_subquery(
                session, candidate_ids, name="_pixlstash_likeness_candidates"
            )
            if candidate_ids is not None
            else None
        )
        max_likeness = func.max(
            func.character_face_likeness(Face.features, refs_blob)
        ).label("likeness_score")
        order_expr = (
            desc(text("likeness_score")) if descending else asc(text("likeness_score"))
        )
        deleted_filter = (
            Picture.deleted.is_(True) if deleted_only else Picture.deleted.is_(False)
        )
        query = (
            select(Face.picture_id, max_likeness)
            .join(Picture, Face.picture_id == Picture.id)
            .where(deleted_filter)
            .group_by(Face.picture_id)
            .order_by(order_expr)
            .limit(limit)
            .offset(offset)
        )
        if stack_leaders_only:
            query = query.where(
                _scoped_stack_leader_clause(character_id, candidate_scope)
            )
        if character_id == "UNASSIGNED":
            other_face = aliased(Face)
            query = query.where(Face.character_id.is_(None))
            query = query.where(
                ~exists(
                    select(other_face.id)
                    .where(
                        other_face.picture_id == Face.picture_id,
                        other_face.character_id.is_not(None),
                    )
                    .correlate(Face)
                )
            )
        elif character_id is not None and character_id != "" and character_id != "ALL":
            query = query.where(Face.character_id == int(character_id))
        if candidate_scope is not None:
            query = query.where(Face.picture_id.in_(candidate_scope))
        return session.exec(query).all()

    rows = server.vault.db.run_task(run_query, priority=DBPriority.IMMEDIATE)
    timing_after_query = time.perf_counter()

    if not rows:
        return []

    selected_ids = [row[0] for row in rows]
    likeness_by_pic = {row[0]: float(row[1]) for row in rows}

    candidate_pics = server.vault.db.run_task(
        Picture.find,
        id=selected_ids,
        select_fields=Picture.metadata_fields(),
        only_deleted=deleted_only,
    )
    timing_after_fetch = time.perf_counter()

    logger.debug(
        "[LIKENESS SQL TIMING] refs=%.3fms query=%.3fms fetch=%.3fms total=%.3fms",
        (timing_after_refs - timing_start) * 1000.0,
        (timing_after_query - timing_after_refs) * 1000.0,
        (timing_after_fetch - timing_after_query) * 1000.0,
        (timing_after_fetch - timing_start) * 1000.0,
    )

    # Imported here, not at the top: the listing module imports
    # ``pixlstash.scoring``, so a module-level import is circular.
    from pixlstash.routes.pictures._listing import stored_picture_dict

    pic_map = {pic.id: pic for pic in candidate_pics}
    results = []
    for pic_id in selected_ids:
        pic = pic_map.get(pic_id)
        if not pic:
            continue
        # Read through the column types: a stored value its column does not
        # declare is null in its row, not a 500 for the page.
        pic_dict = stored_picture_dict(pic)
        pic_dict["character_likeness"] = max(0.0, likeness_by_pic.get(pic_id, 0.0))
        results.append(pic_dict)

    return results


def count_pictures_by_character_likeness(
    server,
    character_id,
    candidate_ids: list[int] | None = None,
    deleted_only: bool = False,
    stack_leaders_only: bool = False,
) -> int:
    """Count pictures that would be returned by a CHARACTER_LIKENESS sort query.

    Does not require the reference character or likeness scoring - it simply counts
    distinct picture_ids matching the character and candidate filters.

    Args:
        server: The server object.
        character_id: Character id filter (int, None/"ALL", or "UNASSIGNED").
        candidate_ids: Optional list of picture ids to restrict the count to.
        deleted_only: If True, restrict to deleted (scrapheap) pictures only.
        stack_leaders_only: If True, collapse each stack to one representative
            using scoped-leader semantics (see _scoped_stack_leader_clause), so
            the count matches what find_pictures_by_character_likeness_sql yields.

    Returns:
        Total number of distinct matching pictures.
    """

    def run_count(session: Session) -> int:
        candidate_scope = (
            scope_id_subquery(
                session, candidate_ids, name="_pixlstash_likeness_candidates"
            )
            if candidate_ids is not None
            else None
        )
        deleted_filter = (
            Picture.deleted.is_(True) if deleted_only else Picture.deleted.is_(False)
        )
        query = (
            select(func.count(func.distinct(Face.picture_id)))
            .join(Picture, Face.picture_id == Picture.id)
            .where(deleted_filter)
        )
        if stack_leaders_only:
            query = query.where(
                _scoped_stack_leader_clause(character_id, candidate_scope)
            )
        if character_id == "UNASSIGNED":
            inner_face = aliased(Face)
            query = query.where(Face.character_id.is_(None))
            query = query.where(
                ~exists(
                    select(inner_face.id)
                    .where(
                        inner_face.picture_id == Face.picture_id,
                        inner_face.character_id.is_not(None),
                    )
                    .correlate(Face)
                )
            )
        elif character_id is not None and character_id != "" and character_id != "ALL":
            query = query.where(Face.character_id == int(character_id))
        if candidate_scope is not None:
            query = query.where(Face.picture_id.in_(candidate_scope))
        result = session.exec(query).first()
        return int(result) if result is not None else 0

    return server.vault.db.run_task(run_count, priority=DBPriority.IMMEDIATE)
