"""Removing a penalised tag removes it from tagging.

The owner's X on a penalised tag stores the name as excluded (hub, every
library), and every vault session carries that set. These tests pin the three
halves of the contract: the score no longer counts the name, no producer writes
it again, and the cleanup removes what is already there.

One logged-in server is shared by the module (see ``tests/test_operation_log.py``
for the shape and why), with the tagging sweeps that own ``Tag`` and
``TagPrediction`` taken out of the planner so the rows a test seeds are the rows
it asserts on.
"""

import gc
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

import pixlstash.database as database_module
from pixlstash.auth import COMFYUI_SCOPE
from pixlstash.db_models import (
    TAG_PENDING_SENTINEL,
    TARGET_TAG_EXCLUSION,
    Operation,
    Picture,
    PictureSet,
    PictureSetMember,
    Review,
    Tag,
    TagPrediction,
    User,
    is_tag_sentinel,
)
from pixlstash.db_models.library_settings import LibrarySettings
from pixlstash.db_models.tag_suggestion import TagSuggestion
from pixlstash.event_types import EventType
from pixlstash.scoring.smart_score import attach_anomaly_inputs
from pixlstash.server import Server
from pixlstash.services import (
    tag_exclusion_service,
    tag_scan_service,
    tag_suggestion_service,
)
from pixlstash.tasks import TaskType, tag_exclusion_cleanup_task
from pixlstash.tasks.missing_tag_prediction_finder import MissingTagPredictionFinder
from pixlstash.tasks.tag_exclusion_cleanup_task import TagExclusionCleanupTask
from pixlstash.tasks.watch_folder_import_task import WatchFolderImportTask
from pixlstash.utils.caption_file_utils import DEFAULT_TAGS_SUFFIX
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.quality.tag_exclusion import (
    EXCLUDED_TAGS_GENERATION_KEY,
    EXCLUDED_TAGS_KEY,
    SESSION_INFO_KEY,
    expand_aliases,
    filter_tag_names,
    session_tag_exclusions,
)
from pixlstash.utils.service.label_ledger import HUMAN, NEG, POS, record_human_label
from pixlstash.utils.service.tag_prediction_utils import (
    recompute_anomaly_tag_uncertainty,
)
from tests.utils import bearer_client, mint_read_token, upload_pictures_and_wait

API = "/api/v1"

# The sweeps whose tasks rewrite the rows these tests seed by hand.
_CONFLICTING_FINDERS = (
    TaskType.TAGGER,
    TaskType.TAG_PREDICTION_BACKFILL,
    TaskType.SMART_SCORE,
    TaskType.TEXT_EMBEDDING,
)


@pytest.fixture(scope="module")
def _env():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
        config_path = os.path.join(temp_dir.name, "server-config.json")
        with open(config_path, "w") as fh:
            fh.write(json.dumps({"port": 8000}))
        server = Server(config_path)
        for task_type in _CONFLICTING_FINDERS:
            server.vault._planner_work_finders.pop(task_type)
        server.vault._work_planner.detach_finders(_CONFLICTING_FINDERS)
        try:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200, resp.text
            yield client, server
        finally:
            server.close()
    finally:
        temp_dir.cleanup()
        gc.collect()


@pytest.fixture
def client(_env):
    return _env[0]


@pytest.fixture
def server(_env):
    return _env[1]


@pytest.fixture(autouse=True)
def _no_exclusions(_env):
    """Every test starts and ends with nothing excluded."""
    _client, server = _env
    server.vault.db.set_tag_exclusions(
        frozenset(), generation=server.vault.db.tag_exclusion_generation + 1
    )
    yield
    server.vault.db.set_tag_exclusions(
        frozenset(), generation=server.vault.db.tag_exclusion_generation + 1
    )


_counter = [0]


def _upload(client):
    """Upload a fresh, content-distinct in-memory PNG and return its id."""
    _counter[0] += 1
    n = _counter[0]
    img = Image.new("RGB", (20 + n, 20 + n), color=(n * 11 % 256, n * 5 % 256, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return upload_pictures_and_wait(
        client, [("file", (f"excl{n}.png", buf.getvalue(), "image/png"))]
    )["results"][0]["picture_id"]


def _exclude(server, *names):
    server.vault.db.set_tag_exclusions(
        expand_aliases(names),
        generation=server.vault.db.tag_exclusion_generation + 1,
    )


def _seed(server, picture_id, tags, predictions=(), human_pos=()):
    """Write Tag rows and built-in tagger predictions for *picture_id*."""

    def _write(session):
        for tag in tags:
            session.add(Tag(tag=tag, picture_id=picture_id))
        for tag, confidence in predictions:
            session.add(
                TagPrediction(
                    picture_id=picture_id,
                    tag=tag,
                    confidence=confidence,
                    model_version="test-v1",
                    predicted_at=datetime.now(timezone.utc),
                )
            )
        for tag in human_pos:
            session.add(
                TagPrediction(
                    picture_id=picture_id,
                    tag=tag,
                    confidence=1.0,
                    model_version="test-v1",
                    label_state=POS,
                    label_source=HUMAN,
                    predicted_at=datetime.now(timezone.utc),
                )
            )
        session.commit()

    server.vault.db.run_task(_write)


def _tags(server, picture_id):
    return sorted(
        server.vault.db.run_task(
            lambda session: [
                row.tag
                for row in session.exec(
                    select(Tag).where(Tag.picture_id == picture_id)
                ).all()
                if not is_tag_sentinel(row.tag)
            ]
        )
    )


def _prediction_tags(server, picture_id):
    return sorted(
        server.vault.db.run_task(
            lambda session: [
                row.tag
                for row in session.exec(
                    select(TagPrediction).where(TagPrediction.picture_id == picture_id)
                ).all()
            ]
        )
    )


# ---------------------------------------------------------------------------
# What one name covers
# ---------------------------------------------------------------------------


def test_a_parent_covers_its_merge_children_and_nothing_else():
    assert expand_aliases(["Malformed Hand "]) == frozenset(
        {"malformed hand", "extra digit", "missing digit"}
    )
    # A child covers only itself; a family sibling is never pulled in.
    assert expand_aliases(["extra digit"]) == frozenset({"extra digit"})
    assert expand_aliases(["flux chin"]) == frozenset({"flux chin"})


# ---------------------------------------------------------------------------
# The vault snapshot
# ---------------------------------------------------------------------------


def test_writer_and_read_sessions_carry_the_snapshot(server):
    _exclude(server, "flux chin")

    writer = server.vault.db.run_task(lambda session: session_tag_exclusions(session))
    reader = server.vault.db.run_immediate_read_task(
        lambda session: session_tag_exclusions(session)
    )

    assert writer == frozenset({"flux chin"})
    assert reader == frozenset({"flux chin"})


def test_tagger_settings_set_the_snapshot_and_keep_other_keys(server):
    previous = dict(server.vault._tagger_settings)
    try:
        settings = dict(previous)
        settings[EXCLUDED_TAGS_KEY] = ["flux chin", "waxy skin"]
        settings[EXCLUDED_TAGS_GENERATION_KEY] = (
            server.vault.db.tag_exclusion_generation + 1
        )
        server.vault.set_tagger_settings(settings)

        assert server.vault.db.tag_exclusions == frozenset({"flux chin", "waxy skin"})
        # Filling in plugin defaults keeps the new keys and the plugin config.
        assert server.vault._tagger_settings[EXCLUDED_TAGS_KEY] == [
            "flux chin",
            "waxy skin",
        ]
        assert server.vault._tagger_settings["plugins"] == previous["plugins"]
        assert server.vault.db.run_task(
            lambda session: session.info[SESSION_INFO_KEY]
        ) == frozenset({"flux chin", "waxy skin"})
    finally:
        restored = dict(previous)
        restored[EXCLUDED_TAGS_GENERATION_KEY] = (
            server.vault.db.tag_exclusion_generation + 1
        )
        server.vault.set_tagger_settings(restored)
    assert server.vault.db.tag_exclusions == frozenset()


def test_an_older_decision_never_replaces_a_newer_one(server):
    base = server.vault.db.tag_exclusion_generation
    server.vault.db.set_tag_exclusions({"test-a", "test-b"}, generation=base + 2)
    server.vault.db.set_tag_exclusions({"test-a"}, generation=base + 1)

    assert server.vault.db.tag_exclusion_generation == base + 2
    assert server.vault.db.tag_exclusions == frozenset({"test-a", "test-b"})
    # The same generation again is the same decision: accepted.
    server.vault.db.set_tag_exclusions({"test-b"}, generation=base + 2)
    assert server.vault.db.tag_exclusions == frozenset({"test-b"})


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _anomaly_inputs(server, picture_id, *, read_session):
    candidates = [{"id": picture_id}]

    def _attach(session):
        attach_anomaly_inputs(session, candidates, apply_thresholds=None)
        return candidates[0]["anomaly_probs"], set(candidates[0]["anomaly_human"])

    if read_session:
        return server.vault.db.run_immediate_read_task(_attach)
    return server.vault.db.run_task(_attach)


@pytest.mark.parametrize("read_session", [True, False], ids=["request", "background"])
def test_an_excluded_tag_is_no_scoring_input(client, server, read_session):
    pid = _upload(client)
    _seed(
        server,
        pid,
        ["flux chin", "waxy skin", "extra digit"],
        predictions=[("flux chin", 0.97), ("waxy skin", 0.9), ("extra digit", 0.95)],
        human_pos=["malformed teeth"],
    )

    probs, human = _anomaly_inputs(server, pid, read_session=read_session)
    assert set(probs) == {"flux chin", "waxy skin", "extra digit", "malformed teeth"}

    # The family sibling stays; a child goes with its excluded parent; a human
    # verdict on an excluded name is not an input either.
    _exclude(server, "flux chin", "malformed hand", "malformed teeth")
    probs, human = _anomaly_inputs(server, pid, read_session=read_session)

    assert set(probs) == {"waxy skin"}
    assert probs["waxy skin"] == pytest.approx(0.9)
    assert human == set()


def test_uncertainty_ignores_an_excluded_tag(client, server):
    pid = _upload(client)
    # Not applied, confidently predicted: the review queue's strongest signal.
    _seed(server, pid, [], predictions=[("flux chin", 0.95), ("waxy skin", 0.2)])

    def _uncertainty(session):
        recompute_anomaly_tag_uncertainty(session, pid)
        session.commit()
        return session.get(Picture, pid).anomaly_tag_uncertainty

    assert server.vault.db.run_task(_uncertainty) == pytest.approx(0.95)
    _exclude(server, "flux chin")
    assert server.vault.db.run_task(_uncertainty) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Writing tags by hand, and the tag list
# ---------------------------------------------------------------------------


def test_adding_a_removed_tag_by_hand_is_refused(client, server):
    pid = _upload(client)
    _exclude(server, "flux chin")

    r = client.post(f"{API}/pictures/{pid}/tags", json={"tag": " Flux Chin"})
    assert r.status_code == 409, r.text
    assert "'flux chin' was removed from tagging" in r.json()["detail"]
    assert _tags(server, pid) == []
    assert _prediction_tags(server, pid) == []

    r = client.post(f"{API}/pictures/{pid}/tags", json={"tag": "portrait"})
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["portrait"]


def _prediction_statuses(server, picture_id):
    return server.vault.db.run_task(
        lambda session: {
            row.tag: (row.status, row.label_state)
            for row in session.exec(
                select(TagPrediction).where(TagPrediction.picture_id == picture_id)
            ).all()
        }
    )


def _confirm_operations(server, picture_id):
    return server.vault.db.run_task(
        lambda session: [
            op.id
            for op in session.exec(
                select(Operation).where(Operation.op_type == "pictures.tags.confirm")
            ).all()
            if picture_id in json.loads(op.target_ids or "[]")
        ]
    )


def test_confirming_a_prediction_for_a_removed_tag_is_refused(client, server):
    pid = _upload(client)
    _seed(server, pid, [], predictions=[("flux chin", 0.8), ("waxy skin", 0.9)])
    before = _prediction_statuses(server, pid)
    _exclude(server, "flux chin")

    r = client.post(f"{API}/pictures/{pid}/tag_predictions/flux%20chin/confirm")
    assert r.status_code == 409, r.text
    assert "'flux chin' was removed from tagging" in r.json()["detail"]
    assert _tags(server, pid) == []
    assert _prediction_statuses(server, pid) == before
    assert _confirm_operations(server, pid) == []

    r = client.post(f"{API}/pictures/{pid}/tag_predictions/waxy%20skin/confirm")
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["waxy skin"]
    assert len(_confirm_operations(server, pid)) == 1


def _remove_tags(server, picture_id):
    def _delete(session):
        for row in session.exec(select(Tag).where(Tag.picture_id == picture_id)).all():
            session.delete(row)
        session.commit()

    server.vault.db.run_task(_delete)


def test_restoring_a_picture_leaves_out_a_removed_tag_and_says_so(client, server):
    pid = _upload(client)
    _seed(server, pid, ["portrait", "Flux Chin"])
    snapshot_id = server.vault.snapshot_service.create_snapshot("MANUAL").id
    url = f"{API}/snapshots/{snapshot_id}/restore"

    # With nothing removed from tagging the restore brings both back.
    _remove_tags(server, pid)
    control = client.post(f"{url}/picture/{pid}", json={})
    assert control.status_code == 200, control.text
    assert _tags(server, pid) == ["Flux Chin", "portrait"]

    _exclude(server, "flux chin")
    _remove_tags(server, pid)
    r = client.post(f"{url}/picture/{pid}", json={})
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["portrait"]
    assert r.json()["skipped_excluded"] == 1

    _remove_tags(server, pid)
    r = client.post(
        f"{url}/batch", json={"resources": [{"type": "picture", "id": pid}]}
    )
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["portrait"]
    assert r.json()["skipped_excluded"] == 1
    assert control.json()["skipped_excluded"] == 0


def _cleared(server, picture_id, tags):
    """Leave *picture_id* as an impossible-tags clear does: no Tag, a human NEG."""

    def _write(session):
        for tag in tags:
            record_human_label(session, picture_id, tag, NEG)
        session.commit()

    server.vault.db.run_task(_write)


def test_undoing_an_impossible_tags_clear_leaves_out_a_removed_tag(client, server):
    url = f"{API}/pictures/impossible-tags/restore"
    control_pid, pid = _upload(client), _upload(client)
    _cleared(server, control_pid, ["portrait", "Flux Chin"])
    _cleared(server, pid, ["portrait", "Flux Chin"])

    pairs = [{"picture_id": control_pid, "tag": t} for t in ("portrait", "Flux Chin")]
    control = client.post(url, json={"pairs": pairs})
    assert control.status_code == 200, control.text
    assert _tags(server, control_pid) == ["Flux Chin", "portrait"]
    assert control.json()["restored"] == 2
    assert control.json()["skipped_excluded"] == 0

    _exclude(server, "flux chin")
    pairs = [{"picture_id": pid, "tag": t} for t in ("portrait", "Flux Chin")]
    r = client.post(url, json={"pairs": pairs})
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["portrait"]
    assert r.json()["restored"] == 1
    assert r.json()["skipped_excluded"] == 1
    # The removed tag keeps the negative the clear recorded; the other is reset.
    statuses = _prediction_statuses(server, pid)
    assert statuses["Flux Chin"][1] == NEG
    assert statuses["portrait"][1] != NEG


def test_replacing_tags_drops_a_removed_one_and_says_so(client, server):
    pid = _upload(client)
    _seed(server, pid, ["test-old tag"])
    _exclude(server, "flux chin")

    r = client.patch(f"{API}/pictures/{pid}", json={"tags": ["Flux Chin", "portrait"]})
    assert r.status_code == 200, r.text
    assert r.json()["skipped_excluded"] == ["flux chin"]
    assert _tags(server, pid) == ["portrait"]

    # Nothing permitted is left: like an empty list, the tags stay as they are.
    r = client.patch(f"{API}/pictures/{pid}", json={"tags": ["flux chin"]})
    assert r.status_code == 200, r.text
    assert r.json()["skipped_excluded"] == ["flux chin"]
    assert _tags(server, pid) == ["portrait"]

    # A replace that names nothing removed reports nothing skipped.
    r = client.patch(f"{API}/pictures/{pid}", json={"tags": ["portrait", "smile"]})
    assert r.status_code == 200, r.text
    assert r.json()["skipped_excluded"] == []
    assert _tags(server, pid) == ["portrait", "smile"]


def test_the_tag_list_leaves_out_a_removed_tag(client, server):
    pid = _upload(client)
    _seed(server, pid, ["flux chin", "test-kept tag"])

    def _listed():
        r = client.get(f"{API}/tags")
        assert r.status_code == 200, r.text
        return {row["tag"] for row in r.json()}

    assert {"flux chin", "test-kept tag"} <= _listed()
    _exclude(server, "flux chin")
    listed = _listed()
    assert "test-kept tag" in listed
    assert "flux chin" not in listed


# ---------------------------------------------------------------------------
# Importing with a tags sidecar
# ---------------------------------------------------------------------------


def _png_bytes():
    _counter[0] += 1
    n = _counter[0]
    img = Image.new("RGB", (20 + n, 20 + n), color=(n * 11 % 256, n * 5 % 256, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"side{n}", buf.getvalue()


def _picture_id_by_name(server, file_name):
    return server.vault.db.run_task(
        lambda session: session.exec(
            select(Picture.id).where(Picture.original_file_name == file_name)
        ).one()
    )


def _tag_rows(server, picture_id):
    """Every Tag row of *picture_id*, the pending sentinel included."""
    return sorted(
        server.vault.db.run_task(
            lambda session: [
                row.tag
                for row in session.exec(
                    select(Tag).where(Tag.picture_id == picture_id)
                ).all()
            ]
        )
    )


@pytest.fixture
def hashed_tags(monkeypatch):
    """Record, per metadata hash, the tag rows it was computed over.

    Background sweeps also write hashed picture columns, so comparing the
    stored hash with a fresh one races them; the tag rows behind the stored
    hash are what a tag write has to keep current.
    """
    tags_by_hash = {}
    compute = database_module._compute_picture_metadata_hashes

    def _recording(session, picture_ids):
        hashes = compute(session, picture_ids)
        for pid, digest in hashes.items():
            tags_by_hash[digest] = sorted(
                session.execute(select(Tag.tag).where(Tag.picture_id == pid))
                .scalars()
                .all()
            )
        return hashes

    monkeypatch.setattr(database_module, "_compute_picture_metadata_hashes", _recording)
    return tags_by_hash


def _tags_behind_the_hash(server, picture_id, hashed_tags):
    stored = server.vault.db.run_task(
        lambda session: session.get(Picture, picture_id).metadata_hash
    )
    return hashed_tags.get(stored)


def _import_with_sidecar(client, server, via, caption, *, png=None, stem=None):
    """Import one picture with a ``.txt`` tags sidecar and return its id."""
    new_stem, new_png = _png_bytes()
    png = png or new_png
    stem = stem or new_stem
    files = [
        ("file", (f"{stem}.png", png, "image/png")),
        ("file", (f"{stem}.txt", caption, "text/plain")),
    ]
    if via == "upload":
        upload_pictures_and_wait(client, files)
    elif via == "staging":
        r = client.post(f"{API}/pictures/import/staging", json={})
        assert r.status_code == 200, r.text
        staging_id = r.json()["staging_id"]
        r = client.post(
            f"{API}/pictures/import/staging/{staging_id}/files", files=files
        )
        assert r.status_code == 200, r.text
        r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
        assert r.status_code == 200, r.text
        deadline = time.time() + 30
        while True:
            r = client.get(f"{API}/pictures/import/staging/{staging_id}/status")
            assert r.status_code == 200, r.text
            if r.json()["stage"] in ("completed", "failed"):
                break
            assert time.time() < deadline, r.json()
            time.sleep(0.1)
        assert r.json()["stage"] == "completed", r.json()
    else:
        with tempfile.TemporaryDirectory() as folder:
            image_path = os.path.join(folder, f"{stem}.png")
            with open(image_path, "wb") as fh:
                fh.write(png)
            with open(os.path.join(folder, f"{stem}.txt"), "wb") as fh:
                fh.write(caption)
            candidate = {"file_path": image_path, "import_source_folder": folder}
            WatchFolderImportTask(server.vault.db, [candidate], 1)._run_task()
    return _picture_id_by_name(server, f"{stem}.png")


_IMPORT_PATHS = ("upload", "staging", "watch folder")


@pytest.mark.parametrize("via", _IMPORT_PATHS)
def test_an_imported_sidecar_never_brings_a_removed_tag_back(
    client, server, hashed_tags, via
):
    _exclude(server, "flux chin")

    pid = _import_with_sidecar(client, server, via, b"flux chin, portrait")

    assert _tag_rows(server, pid) == ["portrait"]
    assert _tags_behind_the_hash(server, pid, hashed_tags) == ["portrait"]


@pytest.mark.parametrize("via", _IMPORT_PATHS)
def test_a_sidecar_of_only_removed_tags_still_counts_as_tagged(
    client, server, hashed_tags, via
):
    """The owner tagged it by hand, so the tagger is not asked to tag it again."""
    _exclude(server, "flux chin", "waxy skin")

    pid = _import_with_sidecar(client, server, via, b"Flux Chin, waxy_skin")

    assert _tag_rows(server, pid) == []
    assert _tags_behind_the_hash(server, pid, hashed_tags) == []


def test_a_duplicate_upload_replaces_tags_with_the_permitted_ones_only(
    client, server, hashed_tags
):
    stem, png = _png_bytes()
    pid = _import_with_sidecar(
        client, server, "upload", b"portrait, smile", png=png, stem=stem
    )
    assert _tag_rows(server, pid) == ["portrait", "smile"]
    _exclude(server, "flux chin", "waxy skin")

    again = _import_with_sidecar(
        client, server, "upload", b"flux chin, beach", png=png, stem=stem
    )
    assert again == pid
    assert _tag_rows(server, pid) == ["beach"]
    assert _tags_behind_the_hash(server, pid, hashed_tags) == ["beach"]

    # Nothing permitted is left: like an empty sidecar, the tags stay.
    _import_with_sidecar(
        client, server, "upload", b"flux chin, waxy skin", png=png, stem=stem
    )
    assert _tag_rows(server, pid) == ["beach"]
    assert _tags_behind_the_hash(server, pid, hashed_tags) == ["beach"]


# ---------------------------------------------------------------------------
# Reviews and tag suggestions
# ---------------------------------------------------------------------------


def _suggest(server, picture_id, tag, direction, *, status="PENDING", twin_id=None):
    """Write one near-neighbour suggestion row and return its id."""

    def _write(session):
        row = TagSuggestion(
            picture_id=picture_id,
            tag=tag,
            direction=direction,
            source=tag_scan_service.SOURCE,
            score=0.95,
            twin_picture_id=twin_id,
            twin_sim=0.99 if twin_id is not None else None,
            status=status,
        )
        session.add(row)
        session.commit()
        return row.id

    return server.vault.db.run_task(_write)


def _suggestion_status(server, suggestion_id):
    return server.vault.db.run_task(
        lambda session: session.get(TagSuggestion, suggestion_id).status
    )


def _suggestion_tags(server, picture_id):
    return sorted(
        server.vault.db.run_task(
            lambda session: [
                row.tag
                for row in session.exec(
                    select(TagSuggestion).where(TagSuggestion.picture_id == picture_id)
                ).all()
            ]
        )
    )


def test_a_scan_or_review_for_a_removed_tag_is_refused(client, server):
    pid = _upload(client)
    r = client.post(
        f"{API}/tag_suggestions/scan", json={"tag": "portrait", "project": None}
    )
    assert r.status_code == 200, r.text
    _exclude(server, "flux chin")

    r = client.post(
        f"{API}/tag_suggestions/scan", json={"tag": " Flux Chin", "project": None}
    )
    assert r.status_code == 409, r.text
    assert "'flux chin' was removed from tagging" in r.json()["detail"]

    r = client.post(f"{API}/reviews", json={"tag": "Flux Chin", "picture_ids": [pid]})
    assert r.status_code == 409, r.text
    assert (
        server.vault.db.run_task(
            lambda session: session.exec(
                select(Review.id).where(Review.tag == "Flux Chin")
            ).all()
        )
        == []
    )
    assert _suggestion_tags(server, pid) == []

    r = client.post(
        f"{API}/reviews", json={"tag": "test-reviewed tag", "picture_ids": [pid]}
    )
    assert r.status_code == 200, r.text
    assert _suggestion_tags(server, pid) == ["test-reviewed tag"]


def test_refreshing_a_review_for_a_removed_tag_is_refused(client, server):
    reviews = {}
    for tag, source in (
        ("test-refreshed tag", tag_scan_service.SOURCE),
        ("test-boundary tag", tag_scan_service.MODEL_SOURCE),
    ):
        r = client.post(f"{API}/reviews", json={"tag": tag, "source": source})
        assert r.status_code == 200, r.text
        reviews[tag] = r.json()["id"]
        r = client.post(f"{API}/reviews/{reviews[tag]}/refresh")
        assert r.status_code == 200, (tag, r.text)
    _exclude(server, *reviews)

    for tag, review_id in reviews.items():
        r = client.post(f"{API}/reviews/{review_id}/refresh")
        assert r.status_code == 409, (tag, r.text)
        assert f"'{tag}' was removed from tagging" in r.json()["detail"]


def test_a_scan_finishing_after_the_removal_writes_no_suspects(client, server):
    pid = _upload(client)
    suspect = {
        "picture_id": pid,
        "direction": "add",
        "score": 0.9,
        "reason": "test-reason",
        "twin_picture_id": None,
        "twin_sim": None,
        "neighbors": [],
    }
    _exclude(server, "flux chin")

    def _write(tag):
        return server.vault.db.run_task(
            tag_scan_service._write_suspects,
            tag,
            tag_scan_service.SOURCE,
            [suspect],
            None,
            False,
        )

    assert _write("Flux Chin") == {"new": 0, "prev_reviewed": 0}
    assert _write("test-scanned tag") == {"new": 1, "prev_reviewed": 0}
    assert _suggestion_tags(server, pid) == ["test-scanned tag"]


def test_acting_on_a_suggestion_for_a_removed_tag_is_refused(client, server):
    pid = _upload(client)
    other = _upload(client)
    twin = _upload(client)
    _seed(server, other, ["flux chin"])
    kept = _suggest(server, pid, "test-accepted tag", "add")
    add = _suggest(server, pid, "flux chin", "add")
    undo = _suggest(server, pid, "Flux Chin", "remove", status="ACCEPTED")
    to_twin = _suggest(server, other, "flux chin", "remove", twin_id=twin)
    _exclude(server, "flux chin")

    r = client.post(f"{API}/tag_suggestions/{kept}/accept")
    assert r.status_code == 200, r.text
    assert _tags(server, pid) == ["test-accepted tag"]

    for path in (f"{add}/accept", f"{undo}/reopen", f"{to_twin}/fix-twin"):
        r = client.post(f"{API}/tag_suggestions/{path}")
        assert r.status_code == 409, (path, r.text)
        assert "'flux chin' was removed from tagging" in r.json()["detail"]
    # Swap untags the suspect first and then tags the twin: the refusal of the
    # second half must leave the first undone too.
    r = client.post(f"{API}/tag_suggestions/{to_twin}/swap")
    assert r.status_code == 409, r.text

    assert _tags(server, pid) == ["test-accepted tag"]
    assert _tags(server, other) == ["flux chin"]
    assert _tags(server, twin) == []
    assert [_suggestion_status(server, s) for s in (add, undo, to_twin)] == [
        "PENDING",
        "ACCEPTED",
        "PENDING",
    ]


def test_bulk_review_actions_leave_a_removed_tag_alone_and_say_so(client, server):
    suspect = _upload(client)
    twin = _upload(client)
    undone = _upload(client)
    _seed(server, twin, ["flux chin"], predictions=[("flux chin", 0.97)])
    _seed(server, suspect, [], predictions=[("flux chin", 0.96)])
    _suggest(server, suspect, "flux chin", "add", twin_id=twin)
    decided = _suggest(server, undone, "flux chin", "remove", status="ACCEPTED")

    def _bulk_accept(**kw):
        return tag_suggestion_service.bulk_accept(
            server.vault, "flux chin", 0.9, picture_ids={suspect}, **kw
        )

    assert _bulk_accept(dry_run=True)["count"] == 1
    _exclude(server, "flux chin")

    dry = _bulk_accept(dry_run=True)
    assert (dry["count"], dry["skipped_excluded"]) == (0, [suspect])
    done = _bulk_accept()
    assert (done["accepted_ids"], done["skipped_excluded"]) == ([], [suspect])
    assert _tags(server, suspect) == []

    r = client.post(f"{API}/tag_suggestions/bulk-reopen", json={"ids": [decided]})
    assert r.status_code == 200, r.text
    assert (r.json()["count"], r.json()["skipped_excluded"]) == (0, [undone])
    assert _suggestion_status(server, decided) == "ACCEPTED"
    assert _tags(server, undone) == []


def test_the_suggestion_queue_leaves_out_a_removed_tag(client, server):
    pid = _upload(client)
    removed = _suggest(server, pid, "Flux Chin", "add")
    kept = _suggest(server, pid, "test-queued tag", "add")

    def _listed():
        r = client.get(f"{API}/tag_suggestions", params={"limit": 500})
        assert r.status_code == 200, r.text
        return {row["id"] for row in r.json()}

    assert {removed, kept} <= _listed()
    _exclude(server, "flux chin")
    listed = _listed()
    assert kept in listed
    assert removed not in listed


# ---------------------------------------------------------------------------
# Removing a penalised tag with X
# ---------------------------------------------------------------------------

PENALISED = f"{API}/users/me/penalised-tags"
CLEANUP = f"{PENALISED}/cleanup"


@pytest.fixture
def owner_settings(server):
    """Put the owner's weights and removed tags back as they were.

    The stored generation starts level with the one in force, which the
    module's other tests move without writing the owner's settings; an owner
    write below it would be an older decision and change nothing.
    """
    user_id = server.auth.user.id

    def _read(session):
        user = session.get(User, user_id)
        return user.tagger_settings, user.smart_score_penalised_tags

    tagger_settings, weights = server.hub_engine.run_immediate_read_task(_read)

    def _store(generation):
        settings = json.loads(tagger_settings or "{}")
        settings[EXCLUDED_TAGS_GENERATION_KEY] = generation

        def _write(session):
            user = session.get(User, user_id)
            user.tagger_settings = json.dumps(settings)
            user.smart_score_penalised_tags = weights
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

        user = server.hub_engine.run_task(_write)
        server.auth.user = user
        server._user = user
        return settings

    _store(server.vault.db.tag_exclusion_generation)
    yield
    _wait_cleanup(server)
    server.vault.set_tagger_settings(
        _store(server.vault.db.tag_exclusion_generation + 1)
    )


@pytest.fixture(scope="module")
def other_library(_env, tmp_path_factory):
    """A second library that is not the active one."""
    folder = tmp_path_factory.mktemp("test-other-library")
    return _env[1].library_registry.create(str(folder), "Other")


def _cleanup_status(client):
    r = client.get(CLEANUP)
    assert r.status_code == 200, r.text
    return r.json()


def _wait_cleanup(server, timeout=60.0):
    """Wait until no cleanup job of this vault is running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = tag_exclusion_service.status(server.vault)
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError(f"cleanup still running: {status}")


def _job(status, job_id):
    return next(job for job in status["jobs"] if job["id"] == job_id)


def _lock(server, picture_id):
    def _write(session):
        locked = PictureSet(name=f"test-locked {picture_id}", locked=True)
        session.add(locked)
        session.flush()
        session.add(PictureSetMember(set_id=locked.id, picture_id=picture_id))
        session.commit()

    server.vault.db.run_task(_write)


def _set_picture(server, picture_id, **values):
    def _write(session):
        pic = session.get(Picture, picture_id)
        for key, value in values.items():
            setattr(pic, key, value)
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_write)


def _picture(server, picture_id, *fields):
    return server.vault.db.run_task(
        lambda session: tuple(
            getattr(session.get(Picture, picture_id), f) for f in fields
        )
    )


def _negative_labels(server, picture_id):
    return server.vault.db.run_task(
        lambda session: sorted(
            row.tag
            for row in session.exec(
                select(TagPrediction).where(
                    TagPrediction.picture_id == picture_id,
                    TagPrediction.label_state == NEG,
                )
            ).all()
        )
    )


def _cleanup_receipts(server, job_id):
    def _read(session):
        rows = session.exec(
            select(Operation).where(Operation.target_type == TARGET_TAG_EXCLUSION)
        ).all()
        receipts = [(row.undoable, json.loads(row.after_state)) for row in rows]
        return [
            (undoable, after) for undoable, after in receipts if after["job"] == job_id
        ]

    return server.vault.db.run_task(_read)


def test_removing_a_penalised_tag_cleans_this_library_only(
    client, server, owner_settings, other_library, hashed_tags, monkeypatch
):
    # One row per batch, so the locked picture (seeded first, lowest row ids)
    # proves the lock filter runs before the batch limit, not after it.
    monkeypatch.setattr(tag_exclusion_cleanup_task, "BATCH_SIZE", 1)
    rebuilds = []
    monkeypatch.setattr(
        tag_exclusion_cleanup_task.tag_health_service,
        "start_rebuild",
        lambda vault: rebuilds.append(vault),
    )
    r = client.put(f"{PENALISED}/test-blob", json={"weight": 3})
    assert r.status_code == 200, r.text
    assert r.json()["smart_score_penalised_tags"]["test-blob"] == 3

    locked = _upload(client)
    _seed(server, locked, ["test-blob"], predictions=[("test-blob", 0.9)])
    _lock(server, locked)
    plain = _upload(client)
    _seed(
        server, plain, ["test-blob", "test-sibling"], predictions=[("Test-Blob", 0.8)]
    )
    _suggest(server, plain, "test-blob", "add")
    _set_picture(server, plain, smart_score=4.2, text_embedding=b"test-embedding")
    scrapped = _upload(client)
    _seed(server, scrapped, ["Test-Blob"], human_pos=["test-blob"])
    # A decided suggestion is the review history, not a pending tag.
    decided = _suggest(server, scrapped, "test-blob", "remove", status="DISMISSED")
    _set_picture(server, scrapped, deleted=True)
    with server.library_switch.open_other_library(other_library.uuid) as (_lib, vault):

        def _elsewhere(session):
            pic = Picture(file_path="test-other.png")
            session.add(pic)
            session.flush()
            session.add(Tag(tag="test-blob", picture_id=pic.id))
            session.commit()
            return pic.id

        elsewhere = vault.db.run_task(_elsewhere)

    r = client.delete(f"{PENALISED}/test-blob")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "test-blob" not in body["smart_score_penalised_tags"]
    assert "test-blob" in body["excluded"]
    job_id = body["cleanup"]["id"]
    assert body["cleanup"]["names"] == ["test-blob"]
    _wait_cleanup(server)

    status = _cleanup_status(client)
    job = _job(status, job_id)
    assert job["state"] == "completed"
    assert (job["pictures"], job["scrapheap"], job["locked"]) == (2, 1, 1)
    assert job["removed"] == {"tags": 2, "predictions": 2, "suggestions": 1}
    assert (status["remaining"], status["locked"], status["state"]) == (0, 1, "idle")
    assert "test-blob" in status["excluded"]

    assert _tags(server, plain) == ["test-sibling"]
    assert _prediction_tags(server, plain) == []
    assert _suggestion_tags(server, plain) == []
    assert _picture(server, plain, "smart_score", "text_embedding") == (
        None,
        b"test-embedding",
    )
    # The tag rows the stored hash was computed over are the ones left (the
    # tagger is held back here, so the upload's pending marker is still one).
    assert _tags_behind_the_hash(server, plain, hashed_tags) == [
        TAG_PENDING_SENTINEL,
        "test-sibling",
    ]
    assert _tags(server, scrapped) == []
    assert _prediction_tags(server, scrapped) == []
    assert _suggestion_tags(server, scrapped) == ["test-blob"]
    assert _suggestion_status(server, decided) == "DISMISSED"
    # Removing is not a verdict: no negative label is left behind.
    assert _negative_labels(server, plain) == []
    assert _negative_labels(server, scrapped) == []
    # A locked set freezes its pictures, this tag included.
    assert _tags(server, locked) == ["test-blob"]
    assert _prediction_tags(server, locked) == ["test-blob"]

    receipts = _cleanup_receipts(server, job_id)
    assert len(receipts) == 1
    undoable, after = receipts[0]
    assert undoable is False
    assert after["counts"]["pictures"] == 2
    assert rebuilds == [server.vault]

    with server.library_switch.open_other_library(other_library.uuid) as (_lib, vault):
        there = vault.db.run_immediate_read_task(
            lambda session: [
                row.tag
                for row in session.exec(
                    select(Tag).where(Tag.picture_id == elsewhere)
                ).all()
            ]
        )
    assert there == ["test-blob"]


def test_a_cleanup_stops_when_the_tag_is_added_back(client, server):
    pid = _upload(client)
    _seed(server, pid, ["test-blob", "test-kept"])
    _set_picture(server, pid, smart_score=4.2, text_embedding=b"test-embedding")
    base = server.vault.db.tag_exclusion_generation
    server.vault.db.set_tag_exclusions(frozenset({"test-blob"}), generation=base + 1)
    names = frozenset({"test-blob"})

    # Added back and removed again since the job was planned: same names, a
    # newer generation. The job must not run on the old decision.
    stale = TagExclusionCleanupTask(server.vault, names, generation=base)
    assert stale._run_task()["state"] == "superseded"
    assert _tags(server, pid) == ["test-blob", "test-kept"]
    # Not removed at all any more.
    server.vault.db.set_tag_exclusions(frozenset(), generation=base + 2)
    gone = TagExclusionCleanupTask(server.vault, names, generation=base + 2)
    assert gone._run_task()["state"] == "superseded"
    assert _tags(server, pid) == ["test-blob", "test-kept"]
    assert _picture(server, pid, "smart_score") == (4.2,)

    server.vault.db.set_tag_exclusions(frozenset({"test-blob"}), generation=base + 3)
    current = TagExclusionCleanupTask(server.vault, names, generation=base + 3)
    assert current._run_task()["state"] == "completed"
    assert _tags(server, pid) == ["test-kept"]
    # The score was computed over the old tags; the text embedding is kept. Run
    # directly, no settings write invalidated the score first.
    assert _picture(server, pid, "smart_score", "text_embedding") == (
        None,
        b"test-embedding",
    )


def test_every_stored_spelling_of_a_removed_tag_is_found(client, server):
    # Case outside ASCII and whitespace other than spaces: the database's own
    # lower() and trim() do not fold either.
    spellings = ["ÄRT", "\tflux chin\t"]
    only = _upload(client)
    _seed(server, only, spellings)
    kept = _upload(client)
    _seed(server, kept, [*spellings, "test-kept"])
    _exclude(server, "ärt", "flux chin")
    names = frozenset({"ärt", "flux chin"})

    # A picture whose only real tags are removed ones has nothing to back-fill.
    candidates = server.vault.db.run_task(
        lambda session: [
            picture.id
            for picture in MissingTagPredictionFinder._fetch_missing_predictions(
                session, 1_000_000
            )
        ]
    )
    assert kept in candidates
    assert only not in candidates

    # The count of that work agrees: taking the picture out changes nothing,
    # while taking out one that has a real tag left does.
    def _count():
        return server.vault.db.run_task(server.vault._count_missing_tag_predictions)

    counted = _count()
    _set_picture(server, only, deleted=True)
    assert _count() == counted
    _set_picture(server, only, deleted=False)
    _set_picture(server, kept, deleted=True)
    assert _count() == counted - 1
    _set_picture(server, kept, deleted=False)

    task = TagExclusionCleanupTask(
        server.vault, names, generation=server.vault.db.tag_exclusion_generation
    )
    assert task._run_task()["state"] == "completed"
    assert _tags(server, only) == []
    assert _tags(server, kept) == ["test-kept"]
    assert tag_exclusion_service.status(server.vault)["remaining"] == 0


def test_a_cleanup_that_never_ran_reads_as_interrupted(
    client, server, owner_settings, monkeypatch
):
    pid = _upload(client)
    _seed(server, pid, ["test-blob"])
    assert client.put(f"{PENALISED}/test-blob", json={"weight": 2}).status_code == 200
    with monkeypatch.context() as patched:
        patched.setattr(server.vault, "submit_task", lambda task: None)
        r = client.delete(f"{PENALISED}/test-blob")
    assert r.status_code == 200, r.text

    status = _cleanup_status(client)
    assert (status["state"], status["remaining"]) == ("interrupted", 1)
    assert _tags(server, pid) == ["test-blob"]

    # Retry.
    r = client.post(CLEANUP, json={})
    assert r.status_code == 200, r.text
    assert r.json()["cleanup"]["names"] == sorted(status["excluded"])
    _wait_cleanup(server)
    status = _cleanup_status(client)
    assert (status["state"], status["remaining"]) == ("idle", 0)
    assert _tags(server, pid) == []

    # Only a removed tag can be cleaned up.
    r = client.post(CLEANUP, json={"names": ["test-not removed"]})
    assert r.status_code == 422, r.text


def test_a_removal_while_a_cleanup_waits_takes_its_names_over(
    client, server, owner_settings, monkeypatch
):
    pid = _upload(client)
    _seed(server, pid, ["test-blob", "test-smudge", "test-kept"])
    for name in ("test-blob", "test-smudge"):
        assert client.put(f"{PENALISED}/{name}", json={"weight": 2}).status_code == 200
    held = []
    with monkeypatch.context() as patched:
        # The runner takes the jobs but has not started them yet.
        patched.setattr(
            server.vault, "submit_task", lambda task: held.append(task) or "test-held"
        )
        first = client.delete(f"{PENALISED}/test-blob")
        assert first.status_code == 200, first.text
        r = client.post(CLEANUP, json={})
        assert r.status_code == 409, r.text
        second = client.delete(f"{PENALISED}/test-smudge")
        assert second.status_code == 200, second.text

    assert first.json()["cleanup"]["names"] == ["test-blob"]
    assert second.json()["cleanup"]["names"] == ["test-blob", "test-smudge"]
    # The first job was planned on an older decision and stops; the second
    # one does the work of both.
    assert held[0]._run_task()["state"] == "superseded"
    assert held[1]._run_task()["state"] == "completed"
    assert _tags(server, pid) == ["test-kept"]


def test_adding_one_tag_back_keeps_removing_the_others(
    client, server, owner_settings, monkeypatch
):
    pid = _upload(client)
    _seed(server, pid, ["test-blob", "test-smudge", "test-kept"])
    for name in ("test-blob", "test-smudge"):
        assert client.put(f"{PENALISED}/{name}", json={"weight": 2}).status_code == 200
    held = []
    with monkeypatch.context() as patched:
        patched.setattr(
            server.vault, "submit_task", lambda task: held.append(task) or "test-held"
        )
        assert client.delete(f"{PENALISED}/test-blob").status_code == 200
        assert client.delete(f"{PENALISED}/test-smudge").status_code == 200
        back = client.put(f"{PENALISED}/test-smudge", json={"weight": 2})
        assert back.status_code == 200, back.text
        # Adding a tag that was never removed starts nothing.
        plain = client.put(f"{PENALISED}/test-kept", json={"weight": 1})
        assert plain.status_code == 200, plain.text

    # Adding test-smudge back moves the generation, so the job carrying both
    # names stops; a new one finishes test-blob alone.
    assert back.json()["cleanup"]["names"] == ["test-blob"]
    assert plain.json()["cleanup"] is None
    assert [task.params["names"] for task in held] == [
        ["test-blob"],
        ["test-blob", "test-smudge"],
        ["test-blob"],
    ]
    assert [task._run_task()["state"] for task in held] == [
        "superseded",
        "superseded",
        "completed",
    ]
    assert _tags(server, pid) == ["test-kept", "test-smudge"]


def _sync_tags(server, on):
    """Turn the library's tag sidecar write-back on or off."""

    def _write(session):
        row = session.exec(select(LibrarySettings)).first()
        row.sync_tags = on
        session.add(row)
        session.commit()

    server.vault.db.run_task(_write)


def _tags_sidecar(server, picture_id):
    """The path of the picture's tag sidecar under the default suffix."""
    (path,) = _picture(server, picture_id, "file_path")
    image_path = ImageUtils.resolve_picture_path(server.vault.image_root, path)
    return os.path.splitext(image_path)[0] + DEFAULT_TAGS_SUFFIX


def _read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_a_tag_sidecar_that_cannot_be_written_keeps_its_tags_until_a_retry(
    client, server, owner_settings, monkeypatch
):
    failing = _upload(client)
    _seed(server, failing, ["test-blob", "test-fail kept"])
    written = _upload(client)
    _seed(server, written, ["test-blob", "test-kept"])
    # A sidecar edited by hand after it was read in: the folder scan owns it.
    edited = _upload(client)
    _seed(server, edited, ["test-blob", "test-edited"])
    edited_file = _tags_sidecar(server, edited)
    with open(edited_file, "w", encoding="utf-8") as fh:
        fh.write("test-blob, test-edited, test-by hand")
    read_in_at = os.path.getmtime(edited_file) - 60
    _set_picture(server, edited, tags_file=edited_file, tags_file_mtime=read_in_at)
    real_write = tag_exclusion_cleanup_task.write_sidecar
    in_writer = []

    def _write_sidecar(path, content):
        in_writer.append(threading.current_thread() is server.vault.db._task_worker)
        if "test-fail kept" in content:
            return None
        return real_write(path, content)

    monkeypatch.setattr(tag_exclusion_cleanup_task, "write_sidecar", _write_sidecar)
    _sync_tags(server, True)
    try:
        assert (
            client.put(f"{PENALISED}/test-blob", json={"weight": 2}).status_code == 200
        )
        r = client.delete(f"{PENALISED}/test-blob")
        assert r.status_code == 200, r.text
        _wait_cleanup(server)

        status = _cleanup_status(client)
        job = _job(status, r.json()["cleanup"]["id"])
        assert (job["state"], job["sidecar_failed"]) == ("failed", 1)
        assert (status["state"], status["remaining"]) == ("interrupted", 1)
        # The picture whose sidecar was not written keeps its tags, so its
        # rows and its file still say the same.
        assert _tags(server, failing) == ["test-blob", "test-fail kept"]
        assert _picture(server, failing, "tags_file") == (None,)
        assert _tags(server, written) == ["test-kept"]
        (tags_file,) = _picture(server, written, "tags_file")
        assert _read_text(tags_file) == "test-kept"
        # Written in the batch's own writer turn: nothing else records a
        # sidecar for the picture in between.
        assert in_writer == [True, True]
        assert _tags(server, edited) == ["test-edited"]
        assert _read_text(edited_file) == "test-blob, test-edited, test-by hand"
        assert _picture(server, edited, "tags_file_mtime") == (read_in_at,)

        # Retry, with the file writable again.
        monkeypatch.setattr(tag_exclusion_cleanup_task, "write_sidecar", real_write)
        r = client.post(CLEANUP, json={})
        assert r.status_code == 200, r.text
        _wait_cleanup(server)
    finally:
        _sync_tags(server, False)

    status = _cleanup_status(client)
    assert _job(status, r.json()["cleanup"]["id"])["state"] == "completed"
    assert (status["state"], status["remaining"]) == ("idle", 0)
    assert _tags(server, failing) == ["test-fail kept"]
    (tags_file,) = _picture(server, failing, "tags_file")
    assert _read_text(tags_file) == "test-fail kept"


def test_a_cancelled_cleanup_leaves_each_picture_agreeing_with_its_sidecar(
    client, server, monkeypatch
):
    # One row per batch, so the job is cancelled with one picture done.
    monkeypatch.setattr(tag_exclusion_cleanup_task, "BATCH_SIZE", 1)
    pictures = [_upload(client) for _ in range(3)]
    sidecars = {}
    for pid in pictures:
        _seed(server, pid, ["test-cut", f"test-kept {pid}"])
        sidecars[pid] = _tags_sidecar(server, pid)
        with open(sidecars[pid], "w", encoding="utf-8") as fh:
            fh.write(f"test-cut, test-kept {pid}")
        _set_picture(
            server,
            pid,
            tags_file=sidecars[pid],
            tags_file_mtime=os.path.getmtime(sidecars[pid]),
        )
    _exclude(server, "test-cut")
    task = TagExclusionCleanupTask(
        server.vault, {"test-cut"}, generation=server.vault.db.tag_exclusion_generation
    )
    real_notify = server.vault.notify
    caller = threading.current_thread()

    def _notify(event_type, data=None):
        # The first batch is in: cancel the job.
        if (
            event_type == EventType.CHANGED_TAGS
            and threading.current_thread() is caller
        ):
            task.on_cancel()
        return real_notify(event_type, data)

    monkeypatch.setattr(server.vault, "notify", _notify)
    _sync_tags(server, True)
    try:
        assert task._run_task()["state"] == "cancelled"
    finally:
        _sync_tags(server, False)

    first, *rest = pictures
    assert _tags(server, first) == [f"test-kept {first}"]
    assert _read_text(sidecars[first]) == f"test-kept {first}"
    for pid in rest:
        assert _tags(server, pid) == ["test-cut", f"test-kept {pid}"]
        assert _read_text(sidecars[pid]) == f"test-cut, test-kept {pid}"
    carrying = server.vault.db.run_immediate_read_task(
        tag_exclusion_cleanup_task.pictures_carrying, {"test-cut"}
    )
    assert carrying == set(rest)
    assert tag_exclusion_service.status(server.vault)["state"] == "interrupted"


def test_adding_a_removed_tag_back_lets_the_tagger_use_it_again(
    client, server, owner_settings
):
    pid = _upload(client)
    _seed(server, pid, ["malformed hand", "extra digit", "test-kept"])
    before = server.vault.db.tag_exclusion_generation

    r = client.delete(f"{PENALISED}/Malformed%20Hand")
    assert r.status_code == 200, r.text
    assert {"malformed hand", "extra digit", "missing digit"} <= set(
        r.json()["excluded"]
    )
    _wait_cleanup(server)
    assert _tags(server, pid) == ["test-kept"]
    removed = server.vault.db.tag_exclusion_generation
    assert removed > before

    r = client.put(f"{PENALISED}/malformed%20hand", json={"weight": 2})
    assert r.status_code == 200, r.text
    assert r.json()["smart_score_penalised_tags"]["malformed hand"] == 2
    assert not {"malformed hand", "extra digit", "missing digit"} & set(
        r.json()["excluded"]
    )
    assert not {"malformed hand", "extra digit"} & server.vault.db.tag_exclusions
    assert server.vault.db.tag_exclusion_generation > removed
    # Tags removed earlier are not put back.
    assert _tags(server, pid) == ["test-kept"]


def test_two_removals_leave_the_newer_decision_in_force_whatever_the_order(
    client, server, owner_settings, monkeypatch
):
    for name in ("test-a", "test-b"):
        assert client.put(f"{PENALISED}/{name}", json={"weight": 3}).status_code == 200
    user_id = server.auth.user.id
    paused, release = threading.Event(), threading.Event()
    real_wake = server.vault.wake

    def _wake():
        # The first removal has written the hub and stops before it publishes.
        if threading.current_thread().name == "test-request-a":
            paused.set()
            assert release.wait(10), "test release timed out"
        real_wake()

    monkeypatch.setattr(server.vault, "wake", _wake)
    monkeypatch.setattr(server.vault, "submit_task", lambda task: None)
    errors = []

    def _remove_a():
        try:
            tag_exclusion_service.remove_penalised_tag(server, user_id, "test-a")
        except Exception as exc:
            errors.append(repr(exc))

    first = threading.Thread(target=_remove_a, name="test-request-a")
    first.start()
    try:
        assert paused.wait(10), "the first removal never reached its publish"
        tag_exclusion_service.remove_penalised_tag(server, user_id, "test-b")
    finally:
        release.set()
        first.join(10)
    assert not first.is_alive()
    assert errors == []

    stored = server.hub_engine.run_immediate_read_task(
        lambda session: json.loads(session.get(User, user_id).tagger_settings)
    )
    assert set(stored[EXCLUDED_TAGS_KEY]) == {"test-a", "test-b"}
    # The older decision published last and changed nothing.
    assert (
        server.vault.db.tag_exclusion_generation == stored[EXCLUDED_TAGS_GENERATION_KEY]
    )
    assert server.vault.db.tag_exclusions == frozenset({"test-a", "test-b"})
    permitted = server.vault.db.run_task(
        lambda session: filter_tag_names(
            ["test-a", "test-b"], session_tag_exclusions(session)
        )
    )
    assert permitted == []


def test_a_penalised_tag_name_is_checked(client, server, owner_settings):
    for name, expected in (
        ("%20%20", 422),
        ("x" * 129, 422),
        ("__tag", 422),
        ("test-never penalised", 404),
    ):
        r = client.delete(f"{PENALISED}/{name}")
        assert r.status_code == expected, (name, r.text)
    for weight in (0, 6):
        r = client.put(f"{PENALISED}/test-blob", json={"weight": weight})
        assert r.status_code == 422, (weight, r.text)
    r = client.put(f"{PENALISED}/{'x' * 129}", json={"weight": 2})
    assert r.status_code == 422, r.text

    r = client.put(f"{PENALISED}/{'y' * 128}", json={"weight": 2})
    assert r.status_code == 200, r.text
    r = client.put(f"{PENALISED}/%20Test-Blob%20", json={"weight": 2})
    assert r.status_code == 200, r.text
    assert r.json()["smart_score_penalised_tags"]["test-blob"] == 2
    r = client.delete(f"{PENALISED}/%20TEST-blob%20")
    assert r.status_code == 200, r.text
    assert "test-blob" in r.json()["excluded"]
    assert "test-blob" in server.vault.db.tag_exclusions


def test_a_tag_with_a_slash_can_be_penalised_and_removed(
    client, server, owner_settings
):
    r = client.put(f"{PENALISED}/test%2Fslash", json={"weight": 2})
    assert r.status_code == 200, r.text
    assert r.json()["smart_score_penalised_tags"]["test/slash"] == 2
    r = client.delete(f"{PENALISED}/test%2Fslash")
    assert r.status_code == 200, r.text
    assert "test/slash" in r.json()["excluded"]
    assert "test/slash" in server.vault.db.tag_exclusions


def test_only_the_owner_removes_a_penalised_tag(client, server, owner_settings):
    pid = _upload(client)
    comfyui_key = client.post(
        f"{API}/users/me/token",
        json={"description": "test-comfyui", "scope": COMFYUI_SCOPE},
    )
    assert comfyui_key.status_code == 200, comfyui_key.text
    # Each credential with a route it may read, to show it works.
    readers = {
        "read": (
            bearer_client(server, mint_read_token(client, "test-read")),
            f"{API}/users/me/penalised-tags",
        ),
        "scoped read": (
            bearer_client(
                server,
                mint_read_token(
                    client, "test-scoped read", resource_type="picture", resource_id=pid
                ),
            ),
            f"{API}/users/me/penalised-tags",
        ),
        "comfyui": (
            bearer_client(server, comfyui_key.json()["token"]),
            f"{API}/pictures",
        ),
    }
    calls = (
        ("get", CLEANUP, None),
        ("post", CLEANUP, {}),
        ("put", f"{PENALISED}/test-blob", {"weight": 2}),
        ("delete", f"{PENALISED}/flux%20chin", None),
    )
    # Positive control: the owner reaches every route.
    assert client.get(CLEANUP).status_code == 200
    assert client.put(f"{PENALISED}/test-blob", json={"weight": 2}).status_code == 200
    for label, (reader, readable) in readers.items():
        assert reader.get(readable).status_code == 200, label
        for method, path, payload in calls:
            kwargs = {} if payload is None else {"json": payload}
            r = getattr(reader, method)(path, **kwargs)
            assert r.status_code == 403, (label, method, path, r.text)
    assert "flux chin" not in server.vault.db.tag_exclusions
