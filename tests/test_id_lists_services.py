"""Service, task and scoring reads take a picture id list of any length.

A select-all, the Scrapheap, a set, a person or a project hands these paths
one id per picture, and an owner's library runs to hundreds of thousands. An
``IN`` list binds one parameter per id, and SQLite refuses a statement past
``SQLITE_LIMIT_VARIABLE_NUMBER``: 999 on builds before 3.32, 32,766 on the
python.org Windows and macOS builds. Each path here reads the list from a temp
table (``scope_id_subquery``) or, where the statement neither aggregates nor
orders nor limits across the list, binds it ``SQLITE_ID_CHUNK`` ids at a time.

Every connection of the module's vault is pinned to the historical 999, and
``test_the_ceiling_is_pinned`` proves the pin took, so a path that went back to
a plain ``IN`` fails with "too many SQL variables" on any build. Each list is
1,100 ids plus pictures that must be left out (binned, out of scope or
absent), and the tests compare the ids that come back, not how many.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import delete, event, func, insert
from sqlalchemy.exc import OperationalError
from sqlmodel import select

from pixlstash.database import (
    VaultDatabase,
    _after_flush_layout_marker,
    _after_flush_smart_score_reset,
)
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureLikeness,
    PictureLikenessQueue,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Quality,
    Tag,
    TagPrediction,
)
from pixlstash.db_models.character_reference import CharacterReferenceMember
from pixlstash.db_models.dedup import DedupVerdict
from pixlstash.db_models.review import Review
from pixlstash.db_models.tag_suggestion import TagSuggestion
from pixlstash.maintenance import MaintenanceMixin
from pixlstash.scoring.smart_score import (
    fetch_anomaly_confidences,
    fetch_applied_anomaly_tags,
    fetch_main_face_mpx,
    fetch_smart_score_data,
    find_pictures_by_smart_score,
)
from pixlstash.services import (
    comfyui_service,
    dedup_tier_service,
    dedup_verdict_service,
    face_clusters,
    mixed_stack_service,
    operation_log_service,
    review_service,
    scrapheap_service,
    search_query_service,
    tag_suggestion_service,
    workflow_library_service,
)
from pixlstash.services.operation_log_service import (
    FACET_CHARACTERS,
    FACET_SETS,
    FACET_STACK,
    FACET_TAG_PREDICTIONS,
    FACET_TAGS,
)
from pixlstash.tasks.likeness_task import LikenessTask
from pixlstash.utils.likeness.likeness_parameter_utils import LikenessParameterUtils
from pixlstash.utils.quality.anomaly_penalty import ANOMALY_PENALTY_TAGS
from pixlstash.utils.service.picture_stats import (
    PictureStatsParams,
    _build_filtered_picture_subquery,
)
from pixlstash.utils.sql_chunking import in_json

VARIABLE_CEILING = 999
GROUP = 1_100
SMALL = 50
# Ids no picture has: they pad a list without matching anything.
ABSENT_IDS = list(range(10**9, 10**9 + 50))
ANOMALY_TAG = sorted(ANOMALY_PENALTY_TAGS)[0]
EMBEDDING = np.ones(8, dtype=np.float32).tobytes()
LIKENESS_BLOB = np.full(4, 0.25, dtype=np.float32).tobytes()
PHASH = "0f" * 8
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
RUN_FILED_WORKFLOW = "manual:example-workflow"


def _pin_variable_limit(vault_db: VaultDatabase) -> None:
    """Lower every new connection of *vault_db* to the historical 999."""

    def set_limit(dbapi_conn, _record):
        dbapi_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, VARIABLE_CEILING)

    event.listen(vault_db._engine, "connect", set_limit)
    vault_db._engine.dispose()


def _face_features(index: int) -> bytes:
    """A distinct, non-zero ArcFace-sized embedding per face."""
    rng = np.random.default_rng(index)
    return (rng.standard_normal(16).astype(np.float32) + 1.0).tobytes()


def _insert_pictures(session, prefix: str, count: int, **columns) -> list[int]:
    """Bulk-insert *count* pictures named ``prefix_N`` and return their ids."""
    rows = []
    for index in range(count):
        row = {
            "file_path": f"{prefix}_{index}.png",
            "format": "png",
            "deleted": False,
            "width": 64,
            "height": 48,
            "pixel_sha": f"{prefix}-{index}",
            "created_at": BASE_TIME + timedelta(minutes=index),
        }
        row.update(columns)
        rows.append(row)
    session.execute(insert(Picture.__table__), rows)
    session.flush()
    return list(
        session.exec(
            select(Picture.id)
            .where(Picture.file_path.like(f"{prefix}\\_%", escape="\\"))
            .order_by(Picture.id)
        ).all()
    )


def _seed(session) -> SimpleNamespace:
    ids = SimpleNamespace()
    ready = {
        "image_embedding": EMBEDDING,
        "likeness_parameters": LIKENESS_BLOB,
        "perceptual_hash": PHASH,
    }
    # The scope: live, embedded, tagged, faced. ``out`` is the same kind of
    # picture outside every list; ``binned`` is in every list but scrapheaped.
    ids.scope = _insert_pictures(session, "scope", GROUP, **ready)
    ids.out = _insert_pictures(session, "out", SMALL, **ready)
    ids.binned = _insert_pictures(session, "binned", SMALL, deleted=True, **ready)
    # No embedding: the smart-score sort's unscored tail.
    ids.unscored = _insert_pictures(session, "unscored", GROUP)
    # Written to by the tests below, one group each.
    ids.restore = _insert_pictures(session, "restore", GROUP, deleted=True)
    ids.meta = _insert_pictures(
        session, "meta", GROUP, likeness_parameters=None, perceptual_hash=None
    )
    ids.layout = _insert_pictures(session, "layout", GROUP)
    ids.dedup = _insert_pictures(session, "dedup", 2 * GROUP)

    stacks = [{"name": name} for name in ("unstack", "split")] + [
        {"name": f"dedup-{index}"} for index in range(GROUP)
    ]
    session.execute(insert(PictureStack.__table__), stacks)
    session.flush()
    stack_rows = session.exec(select(PictureStack.id, PictureStack.name)).all()
    by_name = {name: int(stack_id) for stack_id, name in stack_rows}
    ids.unstack_stack = by_name["unstack"]
    ids.split_stack = by_name["split"]
    ids.unstack = _insert_pictures(
        session, "unstack", GROUP, stack_id=ids.unstack_stack
    )
    ids.split_live = _insert_pictures(session, "splitlive", 3, stack_id=ids.split_stack)
    ids.split_binned = _insert_pictures(
        session, "splitbinned", GROUP, stack_id=ids.split_stack, deleted=True
    )
    # Two pictures per stack, so a group naming one of each folds GROUP stacks.
    for index, picture_id in enumerate(ids.dedup):
        session.execute(
            Picture.__table__.update()
            .where(Picture.__table__.c.id == picture_id)
            .values(stack_id=by_name[f"dedup-{index // 2}"], stack_position=index % 2)
        )

    # One workflow card of GROUP variants, a picture each, newer by index.
    # Index 0 is rated best and index 1 next, and the last is rated as high
    # as index 0 for a read that drops the card. The newest and best rated
    # picture of variant VARIABLE_CEILING sits in the Scrapheap, and the
    # newest kept picture is filed under a run-filed workflow while its
    # variant is the last.
    ids.variants = _insert_pictures(session, "variant", GROUP)
    for index, picture_id in enumerate(ids.variants):
        session.execute(
            Picture.__table__.update()
            .where(Picture.__table__.c.id == picture_id)
            .values(
                workflow_structural_hash=f"variant-{index}",
                workflow_instance_hash=f"instance-{index}",
                score={0: 5, 1: 4, GROUP - 1: 5}.get(index),
            )
        )
    late = BASE_TIME + timedelta(days=30)
    (ids.variant_binned,) = _insert_pictures(
        session,
        "variantbinned",
        1,
        deleted=True,
        workflow_structural_hash=f"variant-{VARIABLE_CEILING}",
        workflow_instance_hash="instance-binned",
        score=5,
        created_at=late,
    )
    (ids.variant_run,) = _insert_pictures(
        session,
        "variantrun",
        1,
        run_workflow_id=RUN_FILED_WORKFLOW,
        workflow_structural_hash=f"variant-{GROUP - 1}",
        created_at=late,
    )

    session.execute(
        insert(Quality.__table__),
        [
            {"picture_id": pid, "sharpness": 0.2, "contrast": 0.1, "brightness": 0.5}
            for pid in ids.scope
        ],
    )
    tag_rows = [{"picture_id": pid, "tag": "signature"} for pid in ids.scope]
    tag_rows += [{"picture_id": pid, "tag": ANOMALY_TAG} for pid in ids.scope]
    tag_rows += [{"picture_id": pid, "tag": "rare"} for pid in ids.scope[:10]]
    session.execute(insert(Tag.__table__), tag_rows)

    prediction = {
        "model_version": "v43",
        "status": "PENDING",
        "label_state": "UNKNOWN",
        "predicted_at": BASE_TIME,
    }
    prediction_rows = [
        {"picture_id": pid, "tag": ANOMALY_TAG, "confidence": 0.9, **prediction}
        for pid in ids.scope + ids.out
    ]
    prediction_rows += [
        {"picture_id": pid, "tag": "dog", "confidence": 0.7, **prediction}
        for pid in ids.scope + ids.out
    ]
    # Bulk accept resolves a pair only when both sides are confident: the five
    # chosen suspects, the shared twin, and one suspect outside the scope.
    ids.twin = ids.out[0]
    ids.chosen = ids.scope[100:105]
    ids.out_suspect = ids.out[1]
    prediction_rows += [
        {"picture_id": pid, "tag": "cat", "confidence": 0.95, **prediction}
        for pid in [*ids.chosen, ids.twin, ids.out_suspect]
    ]
    session.execute(insert(TagPrediction.__table__), prediction_rows)

    session.execute(insert(Review.__table__), [{"tag": "cat", "status": "OPEN"}])
    session.flush()
    ids.review = int(session.exec(select(Review.id)).one())
    suggestion = {
        "tag": "cat",
        "direction": "add",
        "source": "near_neighbor",
        "status": "PENDING",
        "twin_picture_id": ids.twin,
        "twin_sim": 0.9,
        "review_id": ids.review,
    }
    suggestion_rows = [
        {"picture_id": pid, "score": 0.5 + index * 1e-4, **suggestion}
        for index, pid in enumerate(ids.scope)
    ]
    # Outscores everything in scope, so it leads any list that ignores scope.
    suggestion_rows.append({"picture_id": ids.out_suspect, "score": 0.99, **suggestion})
    session.execute(insert(TagSuggestion.__table__), suggestion_rows)

    session.execute(insert(Character.__table__), [{"name": "person"}])
    session.execute(insert(PictureSet.__table__), [{"name": "set"}])
    session.flush()
    ids.character = int(session.exec(select(Character.id)).one())
    ids.set = int(session.exec(select(PictureSet.id)).one())
    session.execute(
        insert(PictureSetMember.__table__),
        [{"set_id": ids.set, "picture_id": pid} for pid in ids.scope + ids.binned],
    )
    faced = ids.scope + ids.binned
    session.execute(
        insert(Face.__table__),
        [
            {
                "picture_id": pid,
                "frame_index": 0,
                "face_index": 0,
                "character_id": ids.character,
                "features": _face_features(index),
                "model_pack": "buffalo_l",
            }
            for index, pid in enumerate(faced)
        ],
    )
    session.execute(
        insert(CharacterReferenceMember.__table__),
        [{"character_id": ids.character, "picture_id": pid} for pid in faced],
    )
    session.commit()
    return ids


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("id_lists_services")
    vault_db = VaultDatabase(str(root / "vault.db"))
    try:
        ids = vault_db.run_task(_seed)
        _pin_variable_limit(vault_db)
        notified: list = []
        vault = SimpleNamespace(
            db=vault_db,
            image_root=str(root),
            notify=lambda *args, **kwargs: notified.append(args),
            get_pixlstash_tagger_meta_path=lambda: None,
            get_pixlstash_tagger_threshold_offset=lambda: 0.0,
            get_pixlstash_acceptance_threshold=lambda: 0.5,
        )
        yield SimpleNamespace(
            ids=ids,
            vault=vault,
            server=SimpleNamespace(vault=vault, auth=None),
            notified=notified,
        )
    finally:
        vault_db.close()


def _read(env, fn, *args):
    return env.vault.db.run_immediate_read_task(fn, *args)


def _write(env, fn, *args):
    return env.vault.db.run_task(fn, *args)


def test_the_ceiling_is_pinned(env):
    """A plain IN over more ids than the pin allows fails, so every test below
    exercises the long-list path rather than passing on a roomy build."""
    ids = env.ids.scope
    assert len(ids) > VARIABLE_CEILING
    with pytest.raises(OperationalError, match="too many SQL variables"):
        _read(
            env,
            lambda s: s.exec(select(Picture.id).where(Picture.id.in_(ids))).all(),
        )


# --- Picture stats (request-driven: /pictures/stats) -------------------------


def _stats_params(**overrides) -> PictureStatsParams:
    values = dict(
        only_deleted=False,
        set_filter_ids=[],
        set_mode="union",
        character_id_list=[],
        character_mode="union",
        character_id_raw=None,
        project_id_raw=None,
        format_filter=[],
        min_score=None,
        max_score=None,
        unscored=False,
        smart_score_bucket=None,
        resolution_bucket=None,
        file_path_prefix=None,
        import_source_folder=None,
        tags_filter=[],
        rejected_tags=[],
        face_filter=None,
        stack_state=None,
        confidence_tag=None,
        confidence_above=[],
        confidence_below=[],
        include=set(),
        penalised_tag_set=None,
        penalised_cooc_both=False,
    )
    values.update(overrides)
    return PictureStatsParams(**values)


def _stats_ids(env, params) -> set[int]:
    def run(session):
        subquery = _build_filtered_picture_subquery(session, params)
        return set(session.exec(select(subquery.c.id)).all())

    return _read(env, run)


def test_stats_scope_reads_a_token_scope_past_the_ceiling(env):
    ids = env.ids
    params = _stats_params(scoped_picture_ids=ids.scope + ids.binned + ABSENT_IDS)
    assert _stats_ids(env, params) == set(ids.scope)


def test_stats_set_filter_reads_a_large_set(env):
    ids = env.ids
    params = _stats_params(set_filter_ids=[ids.set])
    assert _stats_ids(env, params) == set(ids.scope)


def test_stats_character_filter_reads_a_large_person(env):
    ids = env.ids
    params = _stats_params(character_id_list=[ids.character])
    assert _stats_ids(env, params) == set(ids.scope)


# --- Search queries (request-driven: likeness search, set signature) ---------


def test_signature_tags_aggregate_over_a_large_set(env):
    """A tag on exactly half the set is a signature tag only when every
    member's tags were counted: a read that stopped short leaves it below the
    share the whole set's size sets."""
    ids = env.ids
    half = ids.scope[::2]
    assert len(half) * 2 == len(ids.scope)

    def tag_half(session):
        session.execute(
            insert(Tag.__table__), [{"picture_id": pid, "tag": "half"} for pid in half]
        )
        session.commit()

    def untag_half(session):
        session.execute(delete(Tag.__table__).where(Tag.__table__.c.tag == "half"))
        session.commit()

    _write(env, tag_half)
    try:
        tags = search_query_service.fetch_signature_tags(env.vault.db, set(ids.scope))
    finally:
        _write(env, untag_half)
    assert tags == sorted({"signature", ANOMALY_TAG, "half"})


def test_face_candidates_read_a_large_candidate_set(env):
    ids = env.ids
    rows = search_query_service.fetch_face_candidates(
        env.vault.db, set(ids.scope + ids.binned + ABSENT_IDS)
    )
    assert {pid for pid, _faces in rows} == set(ids.scope)


def test_candidate_clip_embeddings_read_a_large_candidate_set(env):
    ids = env.ids
    rows = search_query_service.fetch_candidate_clip_embeddings(
        env.vault.db, ids.scope + ids.binned + ABSENT_IDS
    )
    assert {pid for pid, _emb in rows} == set(ids.scope)


def test_source_clip_embeddings_read_a_large_source_set(env):
    ids = env.ids
    rows = search_query_service.fetch_source_clip_embeddings(
        env.vault.db, ids.scope + ids.binned + ABSENT_IDS
    )
    assert {pid for pid, _blob in rows} == set(ids.scope)


# --- Tag suggestions and reviews (request-driven: a scoped token) ------------


def _top_in_scope(env, count: int) -> list[int]:
    """The in-scope suspects by score, highest first (scores rise with id)."""
    return list(reversed(env.ids.scope))[:count]


def test_list_suggestions_ranks_over_the_whole_scope(env):
    ids = env.ids
    rows = tag_suggestion_service.list_suggestions(
        env.vault,
        tag="cat",
        limit=20,
        picture_ids=set(ids.scope + ids.binned + ABSENT_IDS),
    )
    assert [row.picture_id for row in rows] == _top_in_scope(env, 20)


def test_confidence_map_reads_a_large_id_list(env):
    ids = env.ids
    confidences = _read(
        env,
        tag_suggestion_service._confidence_map,
        ids.scope + ids.binned + ABSENT_IDS,
        "dog",
    )
    assert set(confidences) == set(ids.scope)


def test_bulk_accept_dry_run_respects_a_large_scope(env):
    ids = env.ids
    result = tag_suggestion_service.bulk_accept(
        env.vault,
        "cat",
        0.5,
        dry_run=True,
        picture_ids=set(ids.scope + ABSENT_IDS),
    )
    assert {item["picture_id"] for item in result["sample"]} == set(ids.chosen)


def test_review_queue_ranks_over_the_whole_scope(env):
    ids = env.ids
    items = review_service.list_review_suggestions(
        env.vault,
        ids.review,
        limit=20,
        picture_ids=set(ids.scope + ids.binned + ABSENT_IDS),
    )
    assert [item["picture_id"] for item in items] == _top_in_scope(env, 20)


# --- Person reference clusters (request-driven: up to 5,000 references) -----


def test_reference_cluster_reads_a_large_cluster(env):
    ids = env.ids
    assert _read(env, face_clusters.reference_cluster_ids, ids.character) == sorted(
        ids.scope
    )


def test_reference_cluster_faces_read_a_large_cluster(env):
    ids = env.ids
    faces = _read(env, face_clusters.reference_cluster_faces, ids.character, 5000)
    assert {face.picture_id for face in faces} == set(ids.scope)


# --- Workflow pictures (request-driven: a run's or plan's pictures) ---------


def test_kept_picture_files_reads_a_large_list(env):
    ids = env.ids
    kept = _read(
        env,
        workflow_library_service.kept_picture_files,
        ids.scope + ids.binned + ABSENT_IDS,
    )
    assert set(kept) == set(ids.scope)


def test_picture_sizes_reads_a_large_list(env):
    ids = env.ids
    sizes = _read(
        env,
        workflow_library_service.picture_sizes,
        ids.scope + ids.binned + ABSENT_IDS,
    )
    assert set(sizes) == set(ids.scope + ids.binned)


def _card(count: int) -> list[str]:
    """The first *count* of the seeded card's variants."""
    return [f"variant-{index}" for index in range(count)]


def test_a_cards_newest_pictures_read_a_card_past_the_ceiling(env, key_list_plan):
    """One workflow's variants bind as one list, however many inputs it ran.

    The card stops short of the last variants, so their newer pictures must
    not answer, nor the card's newest, which is in the Scrapheap.
    """
    ids = env.ids
    card = _card(VARIABLE_CEILING + 1)
    with key_list_plan(
        env.vault.db._engine,
        "workflow_structural_hash IN",
        "ix_picture_workflow_structural_hash",
    ):
        newest = _read(env, workflow_library_service.variant_picture_ids, card, 3)
    assert newest == [ids.variants[VARIABLE_CEILING - n] for n in range(3)]


def test_a_cards_best_pictures_read_a_card_past_the_ceiling(env, key_list_plan):
    ids = env.ids
    card = _card(VARIABLE_CEILING + 1)
    with key_list_plan(
        env.vault.db._engine,
        "workflow_structural_hash IN",
        "ix_picture_workflow_structural_hash",
    ):
        best = _read(env, workflow_library_service.best_picture_ids, card, 2)
    assert best == [ids.variants[0], ids.variants[1]]


def test_a_cards_run_instances_read_a_card_past_the_ceiling(env, key_list_plan):
    card = _card(VARIABLE_CEILING + 1)
    with key_list_plan(
        env.vault.db._engine,
        "workflow_structural_hash IN",
        "ix_picture_workflow_structural_hash",
    ):
        newest = _read(
            env, workflow_library_service.instance_hashes_for_variants, card, None, 2
        )
        rated = _read(
            env, workflow_library_service.instance_hashes_for_variants, card, 4, 5
        )
    assert newest == [f"instance-{VARIABLE_CEILING - n}" for n in range(2)]
    assert rated == ["instance-1", "instance-0"]


def test_a_run_filed_cards_pictures_read_a_card_past_the_ceiling(env):
    """With a run-filed workflow live its id is one more key, beside the whole
    card, and the run's picture is filed under it rather than its variant."""
    ids = env.ids
    keys = [*_card(GROUP), RUN_FILED_WORKFLOW]
    newest = _read(
        env,
        workflow_library_service.variant_picture_ids,
        keys,
        3,
        [RUN_FILED_WORKFLOW],
    )
    assert newest == [ids.variant_run, ids.variants[-1], ids.variants[-2]]

    without_run = _read(
        env,
        workflow_library_service.variant_picture_ids,
        _card(GROUP),
        1,
        [RUN_FILED_WORKFLOW],
    )
    assert without_run == [ids.variants[-1]]


# --- Scrapheap restore (request-driven: restore all) -------------------------


def test_restore_brings_back_a_large_selection(env):
    ids = env.ids
    restored = scrapheap_service.restore_pictures(
        env.vault, ids.restore + [ids.out[2]] + ABSENT_IDS, batch_id="restore-batch"
    )
    assert set(restored) == set(ids.restore)
    still_binned = _read(
        env,
        lambda s: set(
            s.exec(
                select(Picture.id).where(
                    Picture.deleted.is_(True),
                    Picture.file_path.like("restore\\_%", escape="\\"),
                )
            ).all()
        ),
    )
    assert still_binned == set()


# --- Mixed stacks (request-driven: a large stack) ---------------------------


def test_unstack_dissolves_a_large_stack(env):
    ids = env.ids
    _write(
        env,
        lambda s: mixed_stack_service.unstack_in_session(s, ids.unstack_stack),
    )
    stacked = _read(
        env,
        lambda s: set(
            s.exec(
                select(Picture.id).where(Picture.stack_id == ids.unstack_stack)
            ).all()
        ),
    )
    assert stacked == set()


def test_split_names_every_scrapheaped_member_of_a_large_stack(env):
    ids = env.ids
    with pytest.raises(mixed_stack_service.MixedStackError) as raised:
        _write(
            env,
            lambda s: mixed_stack_service.split_stranded_in_session(
                s, ids.split_stack, picture_ids=ids.split_binned
            ),
        )
    assert str(sorted(ids.split_binned)) in str(raised.value)


# --- Dedup verdicts (request-driven: a large duplicate group) ---------------


def test_stack_members_folds_a_large_group(env):
    ids = env.ids
    # One picture of each two-picture stack: the group names GROUP pictures
    # and folds GROUP stacks, so both reads are past the ceiling.
    named = ids.dedup[::2]
    cover = named[0]

    def run(session):
        stack_id = dedup_verdict_service._stack_members(session, named, cover)
        session.commit()
        return stack_id

    stack_id = _write(env, run)
    members = _read(
        env,
        lambda s: dict(
            s.exec(
                select(Picture.id, Picture.stack_position).where(
                    Picture.stack_id == stack_id
                )
            ).all()
        ),
    )
    assert set(members) == set(ids.dedup)
    assert members[cover] == 0


def test_verdict_stack_standing_reads_a_large_verdict(env):
    ids = env.ids

    def standing(member_ids):
        row = DedupVerdict(
            signature="sig",
            verdict="stack",
            picture_ids=json.dumps(member_ids),
            excluded_picture_ids="[]",
            stack_id=ids.split_stack,
            decided_at=BASE_TIME,
        )
        return _read(env, dedup_verdict_service._verdict_stack_still_standing, row)

    together = ids.split_live + ids.split_binned
    assert standing(together) is True
    # A live member outside the stack means the group is visible to the queue.
    assert standing(together + [ids.out[3]]) is False


# --- Smart score (request-driven: the sort over a set or the Scrapheap) -----


def test_applied_anomaly_tags_read_a_large_list(env):
    ids = env.ids
    applied = _read(env, fetch_applied_anomaly_tags, ids.scope + ids.out)
    assert set(applied) == set(ids.scope)


def test_anomaly_confidences_read_a_large_list(env):
    ids = env.ids
    probs, _human = _read(env, fetch_anomaly_confidences, ids.scope + ids.out)
    # ``out`` carries the prediction but not the tag, so it is not charged.
    assert set(probs) == set(ids.scope)


def test_smart_score_data_reads_a_large_candidate_set(env):
    ids = env.ids
    _good, _bad, candidates, _config = fetch_smart_score_data(
        env.server,
        None,
        candidate_ids=ids.scope + ids.binned + ABSENT_IDS,
        penalised_tags={},
    )
    assert {candidate["id"] for candidate in candidates} == set(ids.scope)


def test_smart_score_sort_pages_a_large_unscored_set(env):
    ids = env.ids
    results = find_pictures_by_smart_score(
        env.server,
        None,
        0,
        sys.maxsize,
        True,
        candidate_ids=ids.unscored + ids.binned + ABSENT_IDS,
        penalised_tags={},
    )
    # Unscored pictures sort newest first; created_at rises with id.
    assert [row["id"] for row in results] == sorted(ids.unscored, reverse=True)


def test_main_face_size_reads_a_large_list(env):
    def seed(session):
        faced = _insert_pictures(session, "facempx", GROUP, width=1000, height=800)
        faceless = _insert_pictures(session, "facempxnone", SMALL)
        session.execute(
            insert(Face.__table__),
            [
                {
                    "picture_id": pid,
                    "frame_index": 0,
                    "face_index": 0,
                    "bbox": "[100, 100, 300, 300]",
                }
                for pid in faced
            ],
        )
        session.commit()
        return faced, faceless

    faced, faceless = _write(env, seed)

    sizes = _read(env, fetch_main_face_mpx, faced + faceless + ABSENT_IDS)

    assert set(sizes) == set(faced)
    # One 200 x 200 stored box each, shrunk back by the stored-box scale.
    (size,) = set(sizes.values())
    assert 0 < size <= 0.04


# --- Likeness (background: up to 2,048 pictures a cycle) --------------------


def test_likeness_blobs_and_quality_read_a_large_list(env):
    ids = env.ids
    blobs = _read(
        env, LikenessParameterUtils.fetch_blobs_for_ids, ids.scope + ABSENT_IDS
    )
    assert {pid for pid, blob in blobs.items() if blob == LIKENESS_BLOB} == set(
        ids.scope
    )
    quality = _read(
        env, LikenessParameterUtils.fetch_quality_for_ids, ids.scope + ABSENT_IDS
    )
    assert set(quality) == set(ids.scope)


def test_likeness_picture_params_read_a_large_list(env):
    ids = env.ids
    params, updates = _read(
        env,
        LikenessParameterUtils.fetch_picture_params_for_ids,
        ids.scope + ABSENT_IDS,
        env.vault.image_root,
    )
    assert set(params) == set(ids.scope)
    assert updates == {}


def test_likeness_writes_reach_a_large_batch(env):
    ids = env.ids
    updates = [
        {"id": pid, "likeness_parameters": LIKENESS_BLOB}
        for pid in ids.meta + ABSENT_IDS
    ]
    _write(env, LikenessParameterUtils.write_blob_updates, updates)
    _write(
        env,
        LikenessParameterUtils.update_picture_metadata,
        {pid: {"perceptual_hash": PHASH} for pid in ids.meta + ABSENT_IDS},
    )
    written = _read(
        env,
        lambda s: set(
            s.exec(
                select(Picture.id).where(
                    Picture.likeness_parameters == LIKENESS_BLOB,
                    Picture.perceptual_hash == PHASH,
                    Picture.file_path.like("meta\\_%", escape="\\"),
                )
            ).all()
        ),
    )
    assert written == set(ids.meta)


def _queued(env) -> set[int]:
    return _read(
        env, lambda s: set(s.exec(select(PictureLikenessQueue.picture_id)).all())
    )


def _clear_likeness(session):
    session.exec(delete(PictureLikenessQueue))
    session.exec(delete(PictureLikeness))
    session.commit()


def test_likeness_reset_reaches_a_large_list(env):
    ids = env.ids
    _write(env, _clear_likeness)
    pairs = [
        (ids.scope[index], ids.scope[index + 1]) for index in range(0, GROUP - 1, 2)
    ]
    kept = (ids.out[4], ids.out[5])

    def seed(session):
        session.execute(
            insert(PictureLikeness.__table__),
            [
                {"picture_id_a": a, "picture_id_b": b, "likeness": 0.9, "metric": "m"}
                for a, b in pairs + [kept]
            ],
        )
        session.commit()

    _write(env, seed)
    _write(env, LikenessParameterUtils.reset_likeness_for_pictures, ids.scope)
    remaining = _read(
        env,
        lambda s: set(
            s.exec(
                select(PictureLikeness.picture_id_a, PictureLikeness.picture_id_b)
            ).all()
        ),
    )
    assert remaining == {kept}
    assert _queued(env) == set(ids.scope)


def test_likeness_prune_reaches_a_large_batch(env):
    ids = env.ids
    _write(env, _clear_likeness)
    # Two partners per picture_id_a, so a top-1 prune drops exactly the weaker.
    strong = [(pid, ids.out[6], 0.9) for pid in ids.scope]
    weak = [(pid, ids.out[7], 0.1) for pid in ids.scope]

    def seed(session):
        session.execute(
            insert(PictureLikeness.__table__),
            [
                {"picture_id_a": a, "picture_id_b": b, "likeness": v, "metric": "m"}
                for a, b, v in strong + weak
            ],
        )
        session.commit()

    def prune(session):
        PictureLikeness.bulk_prune_below_top_k(session, ids.scope, 1)
        session.commit()

    _write(env, seed)
    _write(env, prune)
    remaining = _read(
        env,
        lambda s: set(
            s.exec(
                select(PictureLikeness.picture_id_a, PictureLikeness.picture_id_b)
            ).all()
        ),
    )
    assert remaining == {(a, b) for a, b, _v in strong}


def test_likeness_work_batch_takes_a_large_queue(env):
    ids = env.ids
    _write(env, _clear_likeness)
    _write(
        env,
        lambda s: (
            PictureLikenessQueue.enqueue(s, ids.scope + ids.binned),
            s.commit(),
        ),
    )
    taken = _write(env, LikenessTask.fetch_work_batch, 2048)
    assert {item[0] for item in taken} == set(ids.scope)
    # A binned picture is not a candidate, so it keeps its queue row.
    assert _queued(env) == set(ids.binned)


# --- Layout tracker (background: a bulk move dirties every picture) ---------


def test_layout_marker_stamps_a_large_flush(env):
    ids = env.ids

    def run(session):
        session.info["_layout_dirty_pids"] = set(ids.layout)
        session.info["_library_has_layout"] = True
        _after_flush_layout_marker(session, None)
        session.commit()

    _write(env, run)
    stamped = _read(
        env,
        lambda s: set(
            s.exec(
                select(Picture.id).where(Picture.layout_check_due_at.is_not(None))
            ).all()
        ),
    )
    assert stamped == set(ids.layout)


# --- Operation log and stack facts (an undoable gesture over a large set) ---


def _insert_stacked(session, prefix: str, count: int) -> dict[int, tuple[int, str]]:
    """Give *count* new pictures a one-member stack each: ``{stack: (pic, name)}``."""
    session.execute(
        insert(PictureStack.__table__),
        [{"name": f"{prefix}-{index}"} for index in range(count)],
    )
    session.flush()
    stacks = session.exec(
        select(PictureStack.id, PictureStack.name)
        .where(PictureStack.name.like(f"{prefix}-%"))
        .order_by(PictureStack.id)
    ).all()
    session.execute(
        insert(Picture.__table__),
        [
            {
                "file_path": f"{name}.png",
                "format": "png",
                "deleted": False,
                "pixel_sha": name,
                "created_at": BASE_TIME,
                "stack_id": int(stack_id),
                "stack_position": 0,
            }
            for stack_id, name in stacks
        ],
    )
    session.flush()
    pictures = dict(
        session.exec(
            select(Picture.stack_id, Picture.id).where(
                Picture.file_path.like(f"{prefix}-%")
            )
        ).all()
    )
    session.commit()
    return {int(stack_id): (int(pictures[stack_id]), name) for stack_id, name in stacks}


def test_operation_capture_reads_a_large_selection(env):
    ids = env.ids
    stacked = _write(env, lambda s: _insert_stacked(s, "capture", GROUP))
    wanted = ids.scope + [picture_id for picture_id, _name in stacked.values()]

    state = _read(
        env, operation_log_service.capture_state_in_session, wanted + ABSENT_IDS
    )

    assert set(state) == {str(picture_id) for picture_id in wanted}
    for picture_id in ids.scope:
        entry = state[str(picture_id)]
        assert "signature" in entry[FACET_TAGS]
        assert ANOMALY_TAG in entry[FACET_TAG_PREDICTIONS]
        assert entry[FACET_SETS] == [ids.set]
        assert list(entry[FACET_CHARACTERS].values()) == [ids.character]
    # GROUP stacks, so the stack-name read is past the ceiling too.
    assert {
        stack_id: state[str(picture_id)][FACET_STACK]
        for stack_id, (picture_id, _name) in stacked.items()
    } == {
        stack_id: {"id": stack_id, "name": name, "position": 0}
        for stack_id, (_picture_id, name) in stacked.items()
    }


def test_stack_facts_read_a_large_page_of_stacks(env):
    stacked = _write(env, lambda s: _insert_stacked(s, "facts", GROUP))

    facts = _read(env, dedup_tier_service.load_stack_facts, [*stacked, *ABSENT_IDS])

    assert {stack_id: fact.member_ids for stack_id, fact in facts.items()} == {
        stack_id: (picture_id,) for stack_id, (picture_id, _name) in stacked.items()
    }


# --- Face-box tracker (background: a face pass or restore moves many boxes) --


def test_smart_score_reset_clears_a_large_flush(env):
    def seed(session):
        moved = _insert_pictures(session, "facebox", GROUP, smart_score=0.5)
        kept = _insert_pictures(session, "faceboxkept", SMALL, smart_score=0.5)
        session.commit()
        return moved, kept

    moved, kept = _write(env, seed)

    def run(session):
        session.info["_face_box_dirty_pids"] = set(moved) | set(ABSENT_IDS)
        _after_flush_smart_score_reset(session, None)
        session.commit()

    _write(env, run)
    scores = _read(
        env,
        lambda s: dict(
            s.exec(
                select(Picture.id, Picture.smart_score).where(
                    Picture.file_path.like("facebox%")
                )
            ).all()
        ),
    )
    assert {pid for pid, score in scores.items() if score is None} == set(moved)
    assert {scores[pid] for pid in kept} == {0.5}


# --- ComfyUI run outputs (a run's output count is the workflow's) -----------


def test_holding_a_run_bins_a_large_output_set(env):
    """A held run soft-deletes every output, and its lock check sees every id:
    the one output a locked set froze is the last, past the first chunk."""

    def seed(session):
        outputs = _insert_pictures(session, "held", GROUP)
        others = _insert_pictures(session, "notheld", SMALL)
        session.execute(
            insert(PictureSet.__table__), [{"name": "held-locked", "locked": True}]
        )
        session.flush()
        locked_set = int(
            session.exec(
                select(PictureSet.id).where(PictureSet.name == "held-locked")
            ).one()
        )
        session.execute(
            insert(PictureSetMember.__table__),
            [{"set_id": locked_set, "picture_id": outputs[-1]}],
        )
        session.commit()
        return outputs, others, locked_set

    def unlock(session, set_id):
        session.execute(
            PictureSet.__table__.update()
            .where(PictureSet.__table__.c.id == set_id)
            .values(locked=False)
        )
        session.commit()

    outputs, others, locked_set = _write(env, seed)
    try:
        held = _write(env, comfyui_service._hold_in_session, outputs + ABSENT_IDS)
        binned = _read(
            env,
            lambda s: set(
                s.exec(
                    select(Picture.id).where(
                        Picture.deleted.is_(True),
                        Picture.file_path.like("%held\\_%", escape="\\"),
                    )
                ).all()
            ),
        )
    finally:
        # Every other test here reads no lock; leave none behind.
        _write(env, unlock, locked_set)
    assert held == set(outputs[:-1])
    assert binned == set(outputs[:-1])
    assert binned.isdisjoint(others)


def test_naming_a_runs_person_reaches_a_large_output_set(env):
    """One person named for a run is deferred onto every output (none has
    faces yet), past the first chunk, and onto nothing else."""

    def seed(session):
        outputs = _insert_pictures(session, "lorarun", GROUP)
        others = _insert_pictures(session, "loraother", SMALL)
        session.commit()
        return outputs, others

    outputs, others = _write(env, seed)
    comfyui_service._assign_outputs_to_lora_people(
        env.server, [env.ids.character], outputs + ABSENT_IDS
    )
    pending = _read(
        env,
        lambda s: dict(
            s.exec(
                select(Picture.id, Picture.pending_character_id).where(
                    Picture.file_path.like("lora%")
                )
            ).all()
        ),
    )
    named = {pid for pid, person in pending.items() if person == env.ids.character}
    assert named == set(outputs)
    assert {pending[pid] for pid in others} == {None}


# --- Startup missing-file cleanup (a library whose files moved) -------------


def test_missing_file_cleanup_removes_a_large_set(tmp_path):
    vault_db = VaultDatabase(str(tmp_path / "vault.db"))
    try:

        def seed(session):
            missing = _insert_pictures(session, "gone", GROUP)
            present = _insert_pictures(session, "here", 3)
            session.commit()
            return missing, present

        _missing, present = vault_db.run_task(seed)
        for index in range(3):
            (tmp_path / f"here_{index}.png").write_bytes(b"png")
        _pin_variable_limit(vault_db)

        cleaner = MaintenanceMixin()
        cleaner.vault = SimpleNamespace(db=vault_db, image_root=str(tmp_path))
        cleaner._cleanup_missing_pictures()

        left = vault_db.run_immediate_read_task(
            lambda s: set(s.exec(select(Picture.id)).all())
        )
        assert left == set(present)
    finally:
        vault_db.close()


# --- in_json: a membership bound as one JSON array --------------------------


def _ids_where(env, predicate) -> list[int]:
    return _read(env, lambda s: list(s.exec(select(Picture.id).where(predicate))))


def test_in_json_reads_a_list_past_the_ceiling(env):
    """Every value binds as one parameter, so a list past the pinned ceiling
    reads whole; repeated and absent values change nothing."""
    ids = env.ids
    values = ids.scope + ids.scope[:10] + ABSENT_IDS
    assert len(values) > VARIABLE_CEILING
    assert sorted(_ids_where(env, in_json(Picture.id, values))) == ids.scope


def test_in_json_of_nothing_matches_nothing(env):
    """An empty list is a valid clause that matches no row, not a missing
    filter that matches every row."""
    assert _ids_where(env, in_json(Picture.id, [])) == []


def test_in_json_compares_text_case_sensitively(env):
    """The match is SQLite's ``=``: a name in other capitals matches only once
    both sides are lowered."""
    ids = env.ids
    rare = set(ids.scope[:10])

    def carriers(predicate) -> set[int]:
        return set(
            _read(
                env,
                lambda s: s.exec(
                    select(Tag.picture_id).where(
                        Tag.picture_id.in_(ids.scope[:20]), predicate
                    )
                ).all(),
            )
        )

    assert carriers(in_json(Tag.tag, ["rare", "rare"])) == rare
    assert carriers(in_json(Tag.tag, ["RARE"])) == set()
    assert carriers(in_json(func.lower(Tag.tag), ["rare"])) == rare
