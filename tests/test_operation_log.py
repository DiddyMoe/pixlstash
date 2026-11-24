"""Tests for the append-only operation log and metadata undo/redo (DAM 1.2).

Covers the three things the design rests on:

1. **Recording** - a metadata mutation appends exactly one operation carrying the
   changed facets, the batch id, and the WS-envelope provenance; a no-op
   mutation appends nothing.
2. **Undo / redo** - undo restores the recorded ``before`` state, redo restores
   ``after``, a bulk action is ONE undoable unit via its batch id, and recording
   a new operation invalidates the redo stack.
3. **The invariants** - the log is append-only (undo mutates only the lifecycle
   markers), the service never reads the origin contextvar, and a locked picture
   set is not walked around by undo.
4. **The scrapheap lifecycle** - a move to the Scrapheap and a restore out of it
   are recorded symmetrically and are reversible in both directions, a bulk move
   is one batch and one Undo, a **permanent** delete is recorded nowhere, and
   undoing a move whose picture has since been purged is refused outright
   (410) rather than half-applied.
5. **The tag-review decisions** (§21.2) - confirming and rejecting a tag
   prediction are recorded, and their undo reverses the *whole* decision: the
   Tag row, the prediction status AND the human-label ledger, with the derived
   ``anomaly_tag_uncertainty`` recomputed and the cached smart score dropped.
   A half-undo - the tag back but the rejection still on file - is the failure
   these tests exist to prevent.
6. **Read cost** - the history lookups build only the row they act on, and the
   read-only listing and search paths never queue on the single writer or
   issue a write. A thumbnail batch stays on the writer queue: it reads a
   picture's geometry and its face boxes from one committed state.
7. **The event loop** - the edits, uploads and token checks that wait on a
   database do so on a worker thread, so a request parked behind the writer or
   a held engine never stops the server answering the next one.
"""

import asyncio
import contextlib
import gc
import io
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event
from sqlalchemy import inspect as sa_inspect
from sqlmodel import delete, select, update

from pixlstash.auth import TOKEN_BIND_PENDING
from pixlstash.database import DBPriority
from pixlstash.db_models import (
    DEFAULT_SMART_SCORE_PENALIZED_TAGS,
    Character,
    Face,
    Operation,
    Picture,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Tag,
    TagPrediction,
    User,
    UserToken,
    is_tag_sentinel,
)
from pixlstash.routes.pictures import _misc
from pixlstash.server import Server
from pixlstash.services import operation_log_service, tag_prediction_service
from pixlstash.services.stack_membership import expand_picture_ids_to_stacks
from pixlstash.tasks import TaskType
from pixlstash.tasks.tag_task import TagTask
from tests.utils import bearer_client, mint_read_token, upload_pictures_and_wait

API = "/api/v1"


# Backfill finders whose task owns a row or column this module hand-writes.
# They are taken out of the planner for the lifetime of the module server (see
# ``_disable_conflicting_backfill``).
_CONFLICTING_FINDERS = (
    # TagTask clears the pending sentinel and adds its own labels beside any
    # seeded Tag; it also rewrites tag_prediction and anomaly_tag_uncertainty and
    # invalidates smart_score. Every `_tags()` assertion here reads that table.
    TaskType.TAGGER,
    # TagPredictionBackfillTask rewrites tag_prediction, which the review
    # decision tests seed by hand and then assert is absent again after an undo.
    TaskType.TAG_PREDICTION_BACKFILL,
    # SmartScoreTask fills any NULL smart_score. Dropping the cached score is
    # part of what an undo must do, so a sweep that refills it turns the
    # `score is None` assertion into a coin flip. It is only reachable at all
    # once the tagger stops being in flight, which detaching TAGGER causes:
    # detach_finders() marks a removed finder exhausted so its dependents are
    # not blocked for ever.
    TaskType.SMART_SCORE,
)


def _disable_conflicting_backfill(server):
    """Take the backfill finders that fight this module's fixtures out of the planner.

    These tests write ``Tag`` and ``TagPrediction`` rows straight into the vault
    and then assert on the exact tag list and the exact prediction/ledger state
    that an undo leaves behind. The tagging sweep owns those same rows: it
    adds real labels beside the seeded tag, and recomputes the two
    derived score columns beside them.

    The per-test servers this module used to build hid the race. A vault that
    had just come up had nothing to backfill and no model loaded, so the sweep
    was still in a long backoff when the test ended. The shared server is warm
    and its work pool only grows (``reset_operation_log`` never wipes
    ``picture``), so the sweep lands *inside* a test instead: a run of this file
    submits several ``TagTask`` batches, and a batch that lands between a seed
    and its assertion fails it. Two or three failures per run, never the same
    ones.

    Only these finders go, and the planner keeps running. Detaching the whole
    set the way ``tests/test_picture_mutation_scope.py`` does is not an option
    here: that module imports its library once up front, whereas nearly every
    test in this one uploads, and the import endpoint refuses a picture outright
    with ``Cannot import: no TaskType.FACE_EXTRACTION finder is registered with
    the work planner.`` Adding ``FACE_EXTRACTION`` to the tuple above was tried
    and turned 51 of these 63 tests red.

    Returns the names of the finders it removed, so ``reset_operation_log`` can
    re-check before every test that they are still gone.
    """
    for task_type in _CONFLICTING_FINDERS:
        server.vault._planner_work_finders.pop(task_type)
    # detach_finders() edits the planner's finder structures under its own lock,
    # so this is safe against the loop thread that is running right now.
    return server.vault._work_planner.detach_finders(_CONFLICTING_FINDERS)


@pytest.fixture(scope="module")
def _env():
    """One logged-in Server + TestClient shared by every test in this module.

    Building a Server (migrations, vault start-up, route registration) and
    tearing it down again costs well over a second, and minting the login
    credential costs another half-second in deliberately slow password
    hashing. Both used to be paid ~50 times over. They are paid once here
    instead, and per-test isolation comes from ``reset_operation_log`` below.
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
        server_config_path = os.path.join(temp_dir.name, "server-config.json")
        with open(server_config_path, "w") as fh:
            fh.write(json.dumps({"port": 8000}))
        server = Server(server_config_path)
        disabled_finders = _disable_conflicting_backfill(server)
        try:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200
            yield client, server, disabled_finders
        finally:
            # The detachment does not need undoing: it edits this server's own
            # planner, and closing the server destroys it.
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
def reset_operation_log(_env):
    """Start every test from an empty operation log and an empty Scrapheap.

    This is an audit-log suite: assertions read the newest entry, count the
    entries of a type, or require that nothing at all was recorded. Sharing one
    Server would otherwise let an earlier test's entries be read as this
    test's: an assertion that passes for the wrong reason, which is worse than
    one that fails. Truncating ``operation`` (nothing references it by fk,
    and only request-driven code writes it) puts each test back on the clean
    log its assertions assume, including the derived undo/redo state, which is
    read straight off these rows.

    The Scrapheap is the other thing a test can observe globally:
    ``POST /pictures/scrapheap/restore`` with no body restores *every*
    soft-deleted picture, so pictures a previous test left scrapheaped would
    inflate its ``restored_count``. Clearing the flag is the whole reset: the
    endpoint's own query is ``Picture.deleted is True`` and nothing else.

    The third check is that the finders ``_disable_conflicting_backfill``
    removed are still gone, so a later test cannot silently run with the tagging
    sweep rewriting the rows it seeds.

    Note the reset deliberately does not wipe ``picture``: the stale-claim /
    id-reuse hazard that forces other shared-server modules to stop the
    schedulers before truncating is not reachable from here.

    All three integrity checks live here rather than in a "runs last" canary
    test on purpose: the CI gate shards tests individually, so a canary only
    guards its own shard while an autouse fixture runs ahead of every test in
    every shard. They assert the log and the Scrapheap *are* empty rather than
    comparing counts, because a leaked row is exactly what corrupts a count.
    """
    _client, server, disabled_finders = _env

    def _reset(session):
        session.exec(delete(Operation))
        session.exec(
            update(Picture)
            .where(Picture.deleted.is_(True))
            .values(deleted=False, deleted_at=None)
        )
        session.commit()

    server.vault.db.run_task(_reset)

    running = server.vault._work_planner.registered_finder_names()
    assert running.isdisjoint(disabled_finders), (
        "a backfill finder that rewrites this module's Tag / TagPrediction / "
        f"smart_score fixtures is running again: {sorted(running & disabled_finders)}"
    )
    assert _operations(server) == [], (
        "the operation log must be empty at the start of every test; the "
        "truncation above is what makes this module's shared Server safe"
    )
    assert _scrapheaped(server) == [], (
        "the Scrapheap must be empty at the start of every test, or a "
        "whole-Scrapheap restore picks up another test's pictures"
    )
    yield


_counter = [0]


def _upload(client):
    """Upload a fresh, content-distinct in-memory PNG and return its id."""
    _counter[0] += 1
    n = _counter[0]
    img = Image.new("RGB", (16 + n, 16 + n), color=(n * 7 % 256, n * 13 % 256, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return upload_pictures_and_wait(
        client, [("file", (f"op{n}.png", buf.getvalue(), "image/png"))]
    )["results"][0]["picture_id"]


def _tags(server, picture_id):
    """The picture's user-visible tags.

    The pending-retag sentinel (``__tag``) is machine bookkeeping written by the
    importer, not a user tag; it is filtered out here so the assertions read as
    the user experiences them. It IS part of the recorded before/after state -
    see the round-trip assertion in the recording test.
    """
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


def _operations(server, **filters):
    return operation_log_service.list_operations(server.vault, limit=100, **filters)


def _scrapheaped(server):
    """The ids currently sitting in the Scrapheap, newest id last."""
    return server.vault.db.run_task(
        lambda session: sorted(
            int(pid)
            for pid in session.exec(
                select(Picture.id).where(Picture.deleted.is_(True))
            ).all()
        )
    )


def _lifecycle(server, picture_id):
    """``(deleted, deleted_at)`` straight off the row, or ``None`` if purged."""

    def _read(session):
        picture = session.get(Picture, picture_id)
        if picture is None:
            return None
        return (bool(picture.deleted), picture.deleted_at)

    return server.vault.db.run_task(_read)


def _stack_state(server, stack_id, picture_ids):
    """``(row_exists, {picture_id: stack_id})`` for the orphan-row assertions."""

    def _read(session):
        row = session.get(PictureStack, stack_id)
        pointers = {
            int(pic.id): pic.stack_id
            for pic in session.exec(
                select(Picture).where(Picture.id.in_(picture_ids))
            ).all()
        }
        return (row is not None, pointers)

    return server.vault.db.run_task(_read)


def _visible(client, picture_id):
    """Whether the picture shows up in the ordinary (non-scrapheap) listing."""
    resp = client.get(f"{API}/pictures", params={"limit": 500})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    pictures = payload["pictures"] if isinstance(payload, dict) else payload
    return picture_id in {int(pic["id"]) for pic in pictures}


def _seed_prediction(
    server, picture_id, tag, confidence=0.99, status="PENDING", model_version="test-v1"
):
    """Write the tagger prediction row a review decision adjudicates."""

    def _insert(session):
        session.add(
            TagPrediction(
                picture_id=picture_id,
                tag=tag,
                confidence=confidence,
                model_version=model_version,
                status=status,
                predicted_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

    server.vault.db.run_task(_insert)


def _prediction(server, picture_id, tag):
    """The prediction row + its human-label ledger, or ``None`` if there is none.

    This is what the tagger and the training exporter read: ``label_source`` is
    what makes a POS/NEG real supervision, so an undo that leaves it set has not
    undone the decision no matter what the tag list says.
    """

    def _read(session):
        row = session.exec(
            select(TagPrediction).where(
                TagPrediction.picture_id == picture_id,
                TagPrediction.tag == tag,
            )
        ).first()
        if row is None:
            return None
        return {
            "model_version": row.model_version,
            "confidence": row.confidence,
            "status": row.status,
            "label_state": row.label_state,
            "label_source": row.label_source,
            "label_model_version": row.label_model_version,
            "label_confidence": row.label_confidence,
        }

    return server.vault.db.run_task(_read)


def _picture_scores(server, picture_id):
    """``(anomaly_tag_uncertainty, smart_score)`` - the two derived values."""

    def _read(session):
        picture = session.get(Picture, picture_id)
        return (picture.anomaly_tag_uncertainty, picture.smart_score)

    return server.vault.db.run_task(_read)


def _set_smart_score(server, picture_id, value):
    def _write(session):
        picture = session.get(Picture, picture_id)
        picture.smart_score = value
        session.add(picture)
        session.commit()

    server.vault.db.run_task(_write)


def _lock_picture(server, picture_id, name="frozen"):
    def _lock(session):
        picture_set = PictureSet(name=name, locked=True)
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()

    server.vault.db.run_task(_lock)


def _purge_forever(client, ids):
    """Permanently destroy scrapheap rows through the real preview->confirm flow."""
    preview = client.post(f"{API}/pictures/scrapheap/delete-preview", json={"ids": ids})
    assert preview.status_code == 200, preview.text
    token = preview.json()["confirm_token"]
    resp = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"picture_ids": ids, "include_protected": True, "confirm_token": token},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Pure unit tests - no server needed
# ---------------------------------------------------------------------------


def test_diff_states_keeps_only_changed_facets():
    before = {"1": {"tags": ["a"], "score": 3, "description": "x"}}
    after = {"1": {"tags": ["a", "b"], "score": 3, "description": "x"}}
    before_delta, after_delta = operation_log_service.diff_states(before, after)
    assert before_delta == {"1": {"tags": ["a"]}}
    assert after_delta == {"1": {"tags": ["a", "b"]}}


def test_diff_states_drops_unchanged_pictures_entirely():
    state = {"1": {"tags": ["a"]}, "2": {"tags": []}}
    before_delta, after_delta = operation_log_service.diff_states(state, state)
    assert before_delta == {}
    assert after_delta == {}


def test_request_context_reads_the_request_never_a_contextvar():
    """The §15 rule at the op-log's entry point: provenance comes off the request.

    A live ``origin_client_id_var`` must not leak into the recorded operation -
    the recorder runs on the DB worker thread where that contextvar is dead, so
    reading it anywhere downstream is the silent-misattribution bug.
    """
    from pixlstash.utils.request_origin import origin_client_id_var

    class _State:
        auth_user_id = 7
        origin_client_id = "tab-from-request"

    class _Request:
        state = _State()

    token = origin_client_id_var.set("contextvar-tab-should-be-ignored")
    try:
        context = operation_log_service.request_context(_Request())
    finally:
        origin_client_id_var.reset(token)

    assert context == {
        "actor": "7",
        "source": "ui",
        "origin_client_id": "tab-from-request",
        "batch_id": None,
    }

    # No X-Client-Id on the request -> the envelope's own defaults, still not
    # the (live) contextvar.
    class _BareState:
        pass

    class _BareRequest:
        state = _BareState()

    token = origin_client_id_var.set("contextvar-tab-should-be-ignored")
    try:
        bare = operation_log_service.request_context(_BareRequest())
    finally:
        origin_client_id_var.reset(token)
    assert bare == {
        "actor": None,
        "source": "external",
        "origin_client_id": None,
        "batch_id": None,
    }


def test_request_context_takes_the_gesture_batch_id_from_the_request():
    """One gesture, one batch: the correlation id rides the request, too.

    ``fallback_batch_id`` is what a handler that is a bulk action in its own
    right passes so it stays batched when the caller sent no header; a header
    the middleware accepted wins over it, which is what makes a compound client
    gesture a single undo unit.
    """

    class _State:
        auth_user_id = 7
        origin_client_id = "tab-1"
        operation_batch_id = "cli-gesture-1"

    class _Request:
        state = _State()

    assert operation_log_service.request_context(_Request())["batch_id"] == (
        "cli-gesture-1"
    )
    assert operation_log_service.request_context(
        _Request(), fallback_batch_id="srv-abc"
    )["batch_id"] == ("cli-gesture-1")

    class _BareState:
        pass

    class _BareRequest:
        state = _BareState()

    assert operation_log_service.request_context(_BareRequest())["batch_id"] is None
    assert (
        operation_log_service.request_context(
            _BareRequest(), fallback_batch_id="srv-abc"
        )["batch_id"]
        == "srv-abc"
    )


def test_a_client_batch_id_can_never_impersonate_a_server_minted_one():
    """The namespace guard: ``cli-`` is the client's, ``srv-`` is the server's.

    A caller-supplied correlation id is a grouping hint over its own history
    (§21.2), but it must not be able to attach its requests to a batch the
    server created - so the validator only accepts the ``cli-`` namespace and
    ``new_batch_id`` only mints ``srv-``.
    """
    from pixlstash.utils.request_origin import (
        MAX_OPERATION_BATCH_ID_LENGTH,
        sanitize_operation_batch_id,
    )

    server_minted = operation_log_service.new_batch_id()
    assert server_minted.startswith(operation_log_service.SERVER_BATCH_ID_PREFIX)
    assert sanitize_operation_batch_id(server_minted) is None

    assert sanitize_operation_batch_id("cli-abcd1234") == "cli-abcd1234"
    # Rejected, never raised and never truncated: a bad header must not 500 and
    # a crafted long value must not collide with a legitimate short one.
    for rejected in (
        None,
        "",
        "srv-deadbeef",
        "abcd1234",
        "cli-",
        "cli-a",
        "cli-has spaces",
        "cli-../../etc/passwd",
        "cli-" + "a" * MAX_OPERATION_BATCH_ID_LENGTH,
    ):
        assert sanitize_operation_batch_id(rejected) is None, rejected


def test_service_module_never_reads_the_origin_contextvar():
    """Structural guard: a future edit cannot reintroduce a contextvar read.

    The same failure shape ``test_source_origin_read_from_data_only`` pins for
    the broadcaster, pinned here for the operation log - both run off the
    request's task, so both must take origin from data passed to them.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(operation_log_service))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "origin_client_id_var" not in referenced, (
        "operation_log_service reads the origin contextvar. It runs on the DB "
        "worker thread where that contextvar is dead - origin must be passed in "
        "explicitly (docs/backend_architecture.md §15)."
    )
    imported = {
        alias_module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias_module in ([node.module] if node.module else [])
    }
    assert not any("request_origin" in module for module in imported)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_tag_add_records_one_undoable_operation_with_provenance(client, server):
    picture_id = _upload(client)
    resp = client.post(
        f"{API}/pictures/{picture_id}/tags",
        json={"tag": "sunset"},
        headers={"X-Client-Id": "tab-1"},
    )
    assert resp.status_code == 200, resp.text

    operations = _operations(server, op_type="pictures.tags.add")
    assert len(operations) == 1
    operation = operations[0]
    assert operation["target_ids"] == [picture_id]
    assert operation["target_count"] == 1
    assert operation["undoable"] is True
    assert operation["status"] == "applied"
    # WS-envelope provenance, carried from the request header.
    assert operation["origin_client_id"] == "tab-1"
    assert operation["source"] == "ui"

    # The recorded state is the RAW tag list, sentinel included: adding the
    # first real tag also consumes the importer's pending-retag sentinel, and
    # undo must put that back or the picture silently leaves the retag queue.
    detail = client.get(f"{API}/operations/{operation['id']}").json()
    assert detail["before"] == {str(picture_id): {"tags": ["__tag"]}}
    assert detail["after"] == {str(picture_id): {"tags": ["sunset"]}}


def test_no_op_mutation_records_nothing(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    before = len(_operations(server))
    # Adding the same tag again changes nothing.
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    assert len(_operations(server)) == before


@pytest.mark.parametrize("failure_stage", ["capture", "record", "serialize"])
def test_recording_failure_rolls_back_the_tag_mutation(
    monkeypatch, failure_stage, client, server
):
    """The receipt is part of the write: no receipt means no domain mutation."""
    picture_id = _upload(client)
    before_operations = len(_operations(server))

    def add_tag(session, pid):
        session.add(Tag(picture_id=pid, tag="atomic"))
        session.flush()

    if failure_stage == "capture":
        real_capture = operation_log_service.capture_state_in_session
        calls = 0

        def fail_second_capture(session, picture_ids):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("after capture failed")
            return real_capture(session, picture_ids)

        monkeypatch.setattr(
            operation_log_service,
            "capture_state_in_session",
            fail_second_capture,
        )
    elif failure_stage == "record":
        monkeypatch.setattr(
            operation_log_service,
            "record_operation_in_session",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("record failed")
            ),
        )
    else:
        monkeypatch.setattr(
            operation_log_service,
            "serialize",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("serialize failed")
            ),
        )

    with pytest.raises(RuntimeError):
        operation_log_service.run_recorded_metadata_task(
            server.vault,
            add_tag,
            picture_id,
            op_type="test.atomic-tag",
            picture_ids=[picture_id],
        )

    assert _tags(server, picture_id) == []
    assert len(_operations(server)) == before_operations


def test_recorded_callback_cannot_commit_independently(client, server):
    """Fail closed if a future callback reintroduces a premature commit."""
    picture_id = _upload(client)

    def bad_callback(session, pid):
        session.add(Tag(picture_id=pid, tag="escaped"))
        session.commit()

    with pytest.raises(RuntimeError, match="attempted to commit independently"):
        operation_log_service.run_recorded_metadata_task(
            server.vault,
            bad_callback,
            picture_id,
            op_type="test.bad-commit",
            picture_ids=[picture_id],
        )

    assert _tags(server, picture_id) == []
    assert _operations(server, op_type="test.bad-commit") == []


def test_an_empty_diff_is_recorded_only_when_targets_are_declared(client, server):
    """The empty-diff escape hatch for operations with no picture facet.

    Default behaviour is unchanged - an empty diff records nothing, so a no-op
    endpoint cannot consume a Ctrl+Z. Passing ``empty_diff_target_ids`` records
    the row anyway, with empty payloads and those targets: the path the dedup
    keep-separate verdict uses, whose whole restore is its post-restore hook.
    """
    picture_id = _upload(client)

    def record(session, target_ids):
        # Serialized in-session: the row would be detached once the DB
        # worker closes the session.
        operation = operation_log_service.record_operation_in_session(
            session,
            op_type="test.no_facet",
            before={},
            after={},
            summary="No facet changed",
            empty_diff_target_ids=target_ids,
        )
        return (
            operation_log_service.serialize(operation, include_state=True)
            if operation is not None
            else None
        )

    assert server.vault.db.run_task(record, None) is None
    operation = server.vault.db.run_task(record, [picture_id])
    assert operation is not None
    assert operation["target_ids"] == [picture_id]
    assert operation["target_count"] == 1
    assert operation["before"] == {}
    assert operation["after"] == {}
    assert operation["undoable"] is True
    assert operation["summary"] == "No facet changed"
    # Undo finds it, writes no picture facet, and still reports (and
    # announces) the declared targets - their domain state is what the
    # operation's post-restore hook changes.
    undone = client.post(f"{API}/operations/undo", json={})
    assert undone.status_code == 200, undone.text
    assert undone.json()["picture_ids"] == [picture_id]
    assert [op["op_type"] for op in undone.json()["operations"]] == ["test.no_facet"]


def test_bulk_rating_is_one_operation_over_many_targets(client, server):
    ids = [_upload(client) for _ in range(3)]
    resp = client.post(
        f"{API}/pictures/apply-scores",
        json={"scores": {str(pid): 4 for pid in ids}, "only_unscored": False},
    )
    assert resp.status_code == 200, resp.text

    operations = _operations(server, op_type="pictures.score")
    assert len(operations) == 1
    assert operations[0]["target_ids"] == sorted(ids)
    assert operations[0]["target_count"] == 3


def test_set_membership_is_recorded_and_undone(client, server):
    """The membership facet, end to end through the real endpoints."""
    picture_id = _upload(client)
    created = client.post(f"{API}/picture_sets", json={"name": "trip"})
    assert created.status_code == 200, created.text
    set_id = created.json()["picture_set"]["id"]

    resp = client.post(f"{API}/picture_sets/{set_id}/members/{picture_id}")
    assert resp.status_code == 200, resp.text

    def members(session):
        return sorted(
            int(row.picture_id)
            for row in session.exec(
                select(PictureSetMember).where(PictureSetMember.set_id == set_id)
            ).all()
        )

    assert server.vault.db.run_task(members) == [picture_id]

    operations = _operations(server, op_type="picture_sets.members.add")
    assert len(operations) == 1
    assert operations[0]["target_ids"] == [picture_id]

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert server.vault.db.run_task(members) == []
    assert client.post(f"{API}/operations/redo").status_code == 200
    assert server.vault.db.run_task(members) == [picture_id]


def test_record_failure_rolls_back_set_membership(monkeypatch, client, server):
    """Membership joins cannot commit without their operation receipt."""
    picture_id = _upload(client)
    created = client.post(f"{API}/picture_sets", json={"name": "atomic-set"})
    set_id = created.json()["picture_set"]["id"]
    monkeypatch.setattr(
        operation_log_service,
        "record_operation_in_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("membership receipt failed")
        ),
    )

    with pytest.raises(RuntimeError, match="membership receipt failed"):
        client.post(f"{API}/picture_sets/{set_id}/members/{picture_id}")

    members = server.vault.db.run_task(
        lambda session: list(
            session.exec(
                select(PictureSetMember).where(
                    PictureSetMember.set_id == set_id,
                    PictureSetMember.picture_id == picture_id,
                )
            ).all()
        )
    )
    assert members == []
    assert _operations(server, op_type="picture_sets.members.add") == []


def test_stacking_is_recorded_and_undone(client, server):
    """The stack facet: undo unstacks, redo re-stacks the same pictures."""
    ids = [_upload(client) for _ in range(2)]
    resp = client.post(f"{API}/stacks", json={"picture_ids": ids})
    assert resp.status_code == 200, resp.text

    def stack_ids(session):
        return sorted(
            (int(row.id), row.stack_id)
            for row in session.exec(select(Picture).where(Picture.id.in_(ids))).all()
        )

    stacked = server.vault.db.run_task(stack_ids)
    assert all(stack_id is not None for _pid, stack_id in stacked)

    operations = _operations(server, op_type="stacks.create")
    assert len(operations) == 1
    assert operations[0]["target_ids"] == sorted(ids)

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert all(
        stack_id is None for _pid, stack_id in server.vault.db.run_task(stack_ids)
    )

    # Redo re-points the pictures at the same stack row.
    assert client.post(f"{API}/operations/redo").status_code == 200
    assert server.vault.db.run_task(stack_ids) == stacked


def test_undo_of_stack_creation_deletes_the_emptied_stack_row(client, server):
    """Issue #643 (CSO finding C3): undoing a stack creation must not leave an
    orphaned empty PictureStack row behind - the mirror of undo-of-dissolve
    recreating the row."""
    ids = [_upload(client) for _ in range(2)]
    resp = client.post(f"{API}/stacks", json={"picture_ids": ids})
    assert resp.status_code == 200, resp.text
    stack_id = int(resp.json()["id"])

    row_exists, pointers = _stack_state(server, stack_id, ids)
    assert row_exists
    assert all(pointer == stack_id for pointer in pointers.values())

    assert client.post(f"{API}/operations/undo").status_code == 200

    row_exists, pointers = _stack_state(server, stack_id, ids)
    assert all(pointer is None for pointer in pointers.values())
    assert not row_exists, "undo left an orphaned empty PictureStack row"


def test_undo_keeps_a_stack_an_outside_picture_still_points_at(client, server):
    """A stack with members the restored operations never touched survives the
    cleanup: over-deletion would break a live pointer, which is worse than the
    row leak the cleanup fixes."""
    ids = [_upload(client) for _ in range(2)]
    resp = client.post(f"{API}/stacks", json={"picture_ids": ids})
    assert resp.status_code == 200, resp.text
    stack_id = int(resp.json()["id"])

    # A third picture joins the stack via the (unrecorded) add-members
    # endpoint, so it is outside the operation about to be undone.
    outsider = _upload(client)
    resp = client.post(
        f"{API}/stacks/{stack_id}/members", json={"picture_ids": [outsider]}
    )
    assert resp.status_code == 200, resp.text

    assert client.post(f"{API}/operations/undo").status_code == 200

    row_exists, pointers = _stack_state(server, stack_id, ids + [outsider])
    assert all(pointers[pid] is None for pid in ids)
    assert pointers[outsider] == stack_id
    assert row_exists, "cleanup deleted a stack that still had a member"


def test_undo_of_a_dissolve_recreates_the_row_and_redo_deletes_it_again(client, server):
    """The pre-existing dissolve-undo behaviour (row recreation) still holds,
    and its redo now removes the recreated row again - matching the forward
    dissolve, which deletes the row itself."""
    ids = [_upload(client) for _ in range(2)]
    resp = client.post(f"{API}/stacks", json={"picture_ids": ids})
    assert resp.status_code == 200, resp.text
    stack_id = int(resp.json()["id"])

    # Removing one member leaves <= 1 behind: the dissolve branch unstacks
    # the survivor too and deletes the stack row.
    resp = client.request(
        "DELETE",
        f"{API}/stacks/{stack_id}/members",
        json={"picture_ids": [ids[0]]},
    )
    assert resp.status_code == 200, resp.text
    row_exists, pointers = _stack_state(server, stack_id, ids)
    assert not row_exists
    assert all(pointer is None for pointer in pointers.values())

    # Undo recreates the row under its original id and restores both members.
    assert client.post(f"{API}/operations/undo").status_code == 200
    row_exists, pointers = _stack_state(server, stack_id, ids)
    assert row_exists, "undo of a dissolve no longer recreates the stack row"
    assert all(pointer == stack_id for pointer in pointers.values())

    # Redo replays the dissolve: members off, and the emptied row goes too.
    assert client.post(f"{API}/operations/redo").status_code == 200
    row_exists, pointers = _stack_state(server, stack_id, ids)
    assert all(pointer is None for pointer in pointers.values())
    assert not row_exists, "redo of a dissolve left the recreated row behind"


# ---------------------------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------------------------


def test_undo_restores_before_state_and_redo_restores_after(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    assert _tags(server, picture_id) == ["sunset"]

    state = client.get(f"{API}/operations/undo-state").json()
    assert state["can_undo"] is True
    assert state["can_redo"] is False

    undo = client.post(f"{API}/operations/undo")
    assert undo.status_code == 200, undo.text
    assert undo.json()["picture_ids"] == [picture_id]
    assert _tags(server, picture_id) == []

    state = client.get(f"{API}/operations/undo-state").json()
    assert state["can_undo"] is False
    assert state["can_redo"] is True

    redo = client.post(f"{API}/operations/redo")
    assert redo.status_code == 200, redo.text
    assert _tags(server, picture_id) == ["sunset"]


def test_redo_after_two_undos_replays_the_older_operation_first(client, server):
    """Redo is the mirror of undo: the last one undone is the first one redone.

    Undo walks history newest-first, so after two undos the most recently
    undone operation is the OLDER one. Replaying the newer one first would lay
    its after-state over a history that has not reached it yet, and the older
    redo that follows would then roll the picture back to the middle state
    while both rows read "applied".
    """
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "one"})
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "two"})
    first, second = sorted(
        (row for row in _operations(server) if row["op_type"] == "pictures.tags.add"),
        key=lambda row: row["id"],
    )[-2:]

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _tags(server, picture_id) == []

    state = client.get(f"{API}/operations/undo-state").json()
    assert state["next_redo"]["id"] == first["id"]

    redo = client.post(f"{API}/operations/redo")
    assert redo.status_code == 200, redo.text
    assert [op["id"] for op in redo.json()["operations"]] == [first["id"]]
    assert _tags(server, picture_id) == ["one"]

    state = client.get(f"{API}/operations/undo-state").json()
    assert state["next_redo"]["id"] == second["id"]
    assert client.post(f"{API}/operations/redo").status_code == 200
    assert sorted(_tags(server, picture_id)) == ["one", "two"]


def test_undo_of_bulk_rating_reverts_every_target(client, server):
    ids = [_upload(client) for _ in range(3)]
    client.post(
        f"{API}/pictures/apply-scores",
        json={"scores": {str(pid): 4 for pid in ids}, "only_unscored": False},
    )

    def scores(session):
        return sorted(
            (int(row.id), row.score)
            for row in session.exec(select(Picture).where(Picture.id.in_(ids))).all()
        )

    assert [score for _pid, score in server.vault.db.run_task(scores)] == [4, 4, 4]

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert [score for _pid, score in server.vault.db.run_task(scores)] == [
        None,
        None,
        None,
    ]


def _favourite_ids(client, ids, **params):
    """Which of *ids* the listing returns with "Favourites only" on."""
    resp = client.get(
        f"{API}/pictures", params={"limit": 500, "favourite": "1", **params}
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    pictures = payload["pictures"] if isinstance(payload, dict) else payload
    return sorted({int(pic["id"]) for pic in pictures} & set(ids))


def _favourite_count(client):
    resp = client.get(f"{API}/pictures/count", params={"favourite": "1"})
    assert resp.status_code == 200, resp.text
    return resp.json()["count"]


def test_favourite_narrows_the_listing_and_one_undo_reverts_it(client, server):
    ids = [_upload(client) for _ in range(3)]
    marked = ids[:2]
    count_before = _favourite_count(client)
    resp = client.post(
        f"{API}/pictures/favourite", json={"picture_ids": marked, "favourite": True}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated_ids"] == sorted(marked)
    assert _favourite_ids(client, ids) == sorted(marked)
    # The grid's projection carries the mark, so a tile can draw its heart.
    grid = client.get(f"{API}/pictures", params={"fields": "grid", "limit": 500})
    grid_rows = grid.json()
    grid_rows = grid_rows["pictures"] if isinstance(grid_rows, dict) else grid_rows
    hearts = {int(p["id"]): p["favourite"] for p in grid_rows if int(p["id"]) in ids}
    assert hearts == {pid: pid in marked for pid in ids}
    # The count the filter panel shows runs the same predicate.
    assert _favourite_count(client) == count_before + len(marked)
    # So do search's and the stats sidebar's parsers (`from_query_params`).
    stats = client.get(f"{API}/pictures/stats", params={"favourite": "1"})
    assert stats.status_code == 200, stats.text
    unfiltered = client.get(f"{API}/pictures/stats")
    assert stats.json() != unfiltered.json()

    # Marking again changes nothing and is not an operation of its own.
    again = client.post(
        f"{API}/pictures/favourite", json={"picture_ids": marked, "favourite": True}
    )
    assert again.json()["updated_ids"] == []
    assert len(_operations(server, op_type="pictures.favourite")) == 1

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _favourite_ids(client, ids) == []
    assert client.post(f"{API}/operations/redo").status_code == 200
    assert _favourite_ids(client, ids) == sorted(marked)

    unmark = client.post(
        f"{API}/pictures/favourite", json={"picture_ids": ids, "favourite": False}
    )
    assert unmark.json()["updated_ids"] == sorted(marked)
    assert _favourite_ids(client, ids) == []


def test_favourites_only_shows_a_stack_by_its_favourite_member(client, server):
    """The heart is often on a variant, not the cover: the stack must still show."""
    cover, variant = _upload(client), _upload(client)
    resp = client.post(f"{API}/stacks", json={"picture_ids": [cover, variant]})
    assert resp.status_code == 200, resp.text

    def positions(session):
        return {
            int(p.id): p.stack_position
            for p in session.exec(
                select(Picture).where(Picture.id.in_([cover, variant]))
            ).all()
        }

    by_position = sorted(
        server.vault.db.run_task(positions).items(), key=lambda kv: kv[1]
    )
    member = by_position[1][0]  # not the stack's cover
    client.post(
        f"{API}/pictures/favourite", json={"picture_ids": [member], "favourite": True}
    )
    shown = _favourite_ids(
        client, [cover, variant], fields="grid", stack_leaders_only="true"
    )
    assert shown == [member]


def test_favourite_refuses_a_malformed_body(client):
    picture_id = _upload(client)
    for body in (
        {"picture_ids": [picture_id], "favourite": "yes"},
        {"picture_ids": [], "favourite": True},
        {"picture_ids": ["abc"], "favourite": True},
        # int() would read each of these as picture 1.
        {"picture_ids": [True], "favourite": True},
        {"picture_ids": [1.9], "favourite": True},
        {"picture_ids": [str(picture_id)], "favourite": True},
    ):
        r = client.post(f"{API}/pictures/favourite", json=body)
        assert r.status_code == 400, (body, r.text)


def test_a_share_token_cannot_favourite_a_picture(client, server):
    """The negative direction with the owner's own write as its control."""
    picture_id = _upload(client)
    body = {"picture_ids": [picture_id], "favourite": True}
    for token in (
        mint_read_token(
            client,
            "favourite picture token",
            resource_type="picture",
            resource_id=picture_id,
        ),
        mint_read_token(client, "favourite global token"),
    ):
        share = bearer_client(server, token)
        assert share.get(f"{API}/pictures").status_code == 200
        assert share.post(f"{API}/pictures/favourite", json=body).status_code == 403
    assert _favourite_ids(client, [picture_id]) == []
    assert client.post(f"{API}/pictures/favourite", json=body).status_code == 200
    assert _favourite_ids(client, [picture_id]) == [picture_id]


def test_undo_is_last_in_first_out(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "one"})
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "two"})
    assert _tags(server, picture_id) == ["one", "two"]

    client.post(f"{API}/operations/undo")
    assert _tags(server, picture_id) == ["one"]
    client.post(f"{API}/operations/undo")
    assert _tags(server, picture_id) == []


def test_named_undo_rejects_a_stale_operation_without_writes(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "one"})
    older = _operations(server)[0]
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "two"})

    response = client.post(f"{API}/operations/{older['id']}/undo")

    assert response.status_code == 409, response.text
    assert "must be undone first" in response.json()["detail"]
    assert _tags(server, picture_id) == ["one", "two"]
    assert [op["status"] for op in _operations(server)[:2]] == [
        "applied",
        "applied",
    ]


def test_batch_undo_rejects_an_interleaved_newer_operation_without_writes(
    client, server
):
    """A latest batch id is still stale when another unit landed inside it."""
    first, second, unrelated = [_upload(client) for _ in range(3)]
    batch_id = operation_log_service.new_batch_id()

    def add_tag(session, picture_id, tag):
        session.add(Tag(picture_id=picture_id, tag=tag))
        session.flush()

    operation_log_service.run_recorded_metadata_task(
        server.vault,
        add_tag,
        first,
        "batched",
        op_type="test.batch",
        picture_ids=[first],
        batch_id=batch_id,
    )
    operation_log_service.run_recorded_metadata_task(
        server.vault,
        add_tag,
        unrelated,
        "newer",
        op_type="test.unrelated",
        picture_ids=[unrelated],
    )
    operation_log_service.run_recorded_metadata_task(
        server.vault,
        add_tag,
        second,
        "batched",
        op_type="test.batch",
        picture_ids=[second],
        batch_id=batch_id,
    )

    response = client.post(f"{API}/operations/batches/{batch_id}/undo")

    assert response.status_code == 409, response.text
    assert client.post(f"{API}/operations/undo").status_code == 409
    assert _tags(server, first) == ["batched"]
    assert _tags(server, second) == ["batched"]
    assert _tags(server, unrelated) == ["newer"]
    assert all(op["status"] == "applied" for op in _operations(server)[:3])


@pytest.mark.parametrize("address", ["operation", "batch"])
def test_concurrent_explicit_undo_has_one_winner_and_one_409(address, client, server):
    first_client = client
    second_client = TestClient(server.api)
    try:
        login = second_client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        )
        assert login.status_code == 200
        picture_id = _upload(first_client)
        batch_id = operation_log_service.new_batch_id()

        def add_tag(session, pid):
            session.add(Tag(picture_id=pid, tag="once"))
            session.flush()

        _result, operation = operation_log_service.run_recorded_metadata_task(
            server.vault,
            add_tag,
            picture_id,
            op_type="test.concurrent",
            picture_ids=[picture_id],
            batch_id=batch_id,
        )
        path = (
            f"{API}/operations/{operation['id']}/undo"
            if address == "operation"
            else f"{API}/operations/batches/{batch_id}/undo"
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(lambda c: c.post(path), [first_client, second_client])
            )

        assert sorted(response.status_code for response in responses) == [200, 409]
        assert _tags(server, picture_id) == []
        rows = _operations(server, batch_id=batch_id)
        assert [row["status"] for row in rows] == ["undone"]
    finally:
        second_client.close()


def test_recording_a_new_operation_invalidates_the_redo_stack(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "one"})
    client.post(f"{API}/operations/undo")
    assert client.get(f"{API}/operations/undo-state").json()["can_redo"] is True

    # A new change moves the history on; the undone operation can no longer
    # be replayed onto it.
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "two"})
    assert client.get(f"{API}/operations/undo-state").json()["can_redo"] is False
    assert client.post(f"{API}/operations/redo").status_code == 409

    superseded = _operations(server, status="superseded")
    assert len(superseded) == 1
    # The row survives - this is an audit log, not a stack that pops.
    assert superseded[0]["op_type"] == "pictures.tags.add"


def test_undo_with_nothing_to_undo_is_409(client, server):
    assert client.post(f"{API}/operations/undo").status_code == 409
    assert client.post(f"{API}/operations/redo").status_code == 409


@contextmanager
def _hydrated_operation_ids():
    """Collect the id of every ``Operation`` the ORM builds inside the block.

    The ``load`` event fires once per row a query materialises into a session,
    so this sees every row the SQL returned, not only the one the caller kept.
    ``reset_operation_log`` empties the table before each test, so every row
    seen here belongs to the calling test.
    """
    loaded: list[int] = []

    def _on_load(target, _context):
        loaded.append(target.id)

    event.listen(Operation, "load", _on_load)
    try:
        yield loaded
    finally:
        event.remove(Operation, "load", _on_load)


def test_history_lookups_build_only_the_operation_they_act_on(client, server):
    """Undo state, undo and redo each hydrate just the row they return.

    ``Result.first()`` adds no LIMIT and an ORM result is buffered, so an
    unlimited "newest applied" lookup builds the whole applied history to keep
    one row. Two rows on each side of the history make that visible.
    """
    picture_id = _upload(client)
    for tag in ("one", "two", "three", "four"):
        response = client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": tag})
        assert response.status_code == 200, response.text
    four, three, two, one = [op["id"] for op in _operations(server)]
    for _ in range(2):
        assert client.post(f"{API}/operations/undo").status_code == 200

    with _hydrated_operation_ids() as loaded:
        state = client.get(f"{API}/operations/undo-state").json()
    assert state["next_undo"]["id"] == two
    assert state["next_redo"]["id"] in (three, four)
    assert sorted(loaded) == sorted([two, state["next_redo"]["id"]])

    with _hydrated_operation_ids() as loaded:
        response = client.post(f"{API}/operations/undo")
    assert response.status_code == 200, response.text
    assert [op["id"] for op in response.json()["operations"]] == [two]
    assert loaded == [two]

    next_redo = client.get(f"{API}/operations/undo-state").json()["next_redo"]["id"]
    with _hydrated_operation_ids() as loaded:
        response = client.post(f"{API}/operations/redo")
    assert response.status_code == 200, response.text
    assert [op["id"] for op in response.json()["operations"]] == [next_redo]
    assert loaded == [next_redo]
    assert one not in loaded


def test_stale_batch_check_builds_only_the_newest_blocking_operation(client, server):
    """The interleaved-batch refusal names, and builds, one blocking row."""
    picture_id = _upload(client)
    batch_id = operation_log_service.new_batch_id()

    def add_tag(session, tag):
        session.add(Tag(picture_id=picture_id, tag=tag))
        session.flush()

    def record(tag, op_type, batch=None):
        _result, operation = operation_log_service.run_recorded_metadata_task(
            server.vault,
            add_tag,
            tag,
            op_type=op_type,
            picture_ids=[picture_id],
            batch_id=batch,
        )
        return operation["id"]

    oldest_member = record("batched-a", "test.batch", batch_id)
    record("unrelated-older", "test.unrelated")
    newest_blocker = record("unrelated-newer", "test.unrelated")
    newest_member = record("batched-b", "test.batch", batch_id)

    with _hydrated_operation_ids() as loaded:
        response = client.post(f"{API}/operations/batches/{batch_id}/undo")

    assert response.status_code == 409, response.text
    assert (
        f"operation {newest_blocker} must be undone first"
        in (response.json()["detail"])
    )
    assert sorted(loaded) == sorted([newest_member, oldest_member, newest_blocker])


def test_batch_undo_reverts_the_whole_bulk_action_in_one_call(client, server):
    """One bulk action = one batch id = one Undo, exactly as the sweep needs.

    Recorded through the service (the sweep's own service does not exist yet),
    then reverted through the public batch endpoint.
    """
    ids = [_upload(client) for _ in range(2)]
    batch_id = operation_log_service.new_batch_id()

    def _tag_one(session, picture_id, tag):
        session.add(Tag(picture_id=picture_id, tag=tag))
        session.flush()

    for picture_id in ids:
        operation_log_service.run_recorded_metadata_task(
            server.vault,
            _tag_one,
            picture_id,
            "swept",
            op_type="test.sweep",
            picture_ids=[picture_id],
            batch_id=batch_id,
            summary="Swept",
            actor="1",
            source="ui",
            origin_client_id="tab-sweep",
        )

    assert all(_tags(server, pid) == ["swept"] for pid in ids)
    members = _operations(server, batch_id=batch_id)
    assert len(members) == 2

    resp = client.post(f"{API}/operations/batches/{batch_id}/undo")
    assert resp.status_code == 200, resp.text
    assert sorted(resp.json()["picture_ids"]) == sorted(ids)
    assert all(_tags(server, pid) == [] for pid in ids)
    assert all(
        op["status"] == "undone" for op in _operations(server, batch_id=batch_id)
    )

    # A second call has nothing left to revert.
    assert client.post(f"{API}/operations/batches/{batch_id}/undo").status_code == 409


def test_undoing_one_member_of_a_batch_reverts_the_whole_batch(client, server):
    ids = [_upload(client) for _ in range(2)]
    batch_id = operation_log_service.new_batch_id()

    def _tag_one(session, picture_id, tag):
        session.add(Tag(picture_id=picture_id, tag=tag))
        session.flush()

    for picture_id in ids:
        operation_log_service.run_recorded_metadata_task(
            server.vault,
            _tag_one,
            picture_id,
            "swept",
            op_type="test.sweep",
            picture_ids=[picture_id],
            batch_id=batch_id,
        )

    newest = _operations(server, batch_id=batch_id)[0]
    resp = client.post(f"{API}/operations/{newest['id']}/undo")
    assert resp.status_code == 200, resp.text
    # Both members reverted - a partially-undone bulk action cannot exist.
    assert len(resp.json()["operations"]) == 2
    assert all(_tags(server, pid) == [] for pid in ids)


def test_log_is_append_only_across_undo_and_redo(client, server):
    """Undo/redo move the lifecycle marker only; no row is rewritten or removed."""
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})

    def snapshot(session):
        return [
            (
                row.id,
                row.op_type,
                row.target_ids,
                row.before_state,
                row.after_state,
                row.created_at,
            )
            for row in session.exec(select(Operation).order_by(Operation.id)).all()
        ]

    recorded = server.vault.db.run_task(snapshot)
    assert len(recorded) == 1

    client.post(f"{API}/operations/undo")
    client.post(f"{API}/operations/redo")

    assert server.vault.db.run_task(snapshot) == recorded


def test_undo_refuses_to_write_a_picture_frozen_by_a_locked_set(client, server):
    """A locked set is a hard freeze; undo must not be the way around it."""
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    _lock_picture(server, picture_id)

    resp = client.post(f"{API}/operations/undo")
    assert resp.status_code == 423, resp.text
    assert _tags(server, picture_id) == ["sunset"]
    # The operation stays applied - a refused undo must not half-commit.
    assert _operations(server)[0]["status"] == "applied"


@contextlib.contextmanager
def _removed_from_tagging(server, *names):
    """Run the block with *names* removed from tagging, as the X in Smart Score does."""
    server.vault.db.set_tag_exclusions(frozenset(names))
    try:
        yield
    finally:
        server.vault.db.set_tag_exclusions(frozenset())


def test_undo_and_redo_never_put_back_a_tag_removed_from_tagging(client, server):
    """A history step that would re-add a removed tag is refused whole.

    The log stores the exact prior state, so applying part of it would leave a
    step marked undone that was not. The refusal names the tag and leaves the
    pictures and the step's status as they were; an unrelated step still undoes.
    """
    removed = _upload(client)
    redone = _upload(client)
    unrelated = _upload(client)
    client.post(f"{API}/pictures/{removed}/tags", json={"tag": "Test Gone"})
    client.post(f"{API}/pictures/{removed}/tags/remove_all", json={"tag": "Test Gone"})
    removal = _operations(server)[0]
    client.post(f"{API}/pictures/{redone}/tags", json={"tag": "test gone"})
    assert client.post(f"{API}/operations/undo").status_code == 200
    addition = _operations(server, status="undone")[0]
    assert _tags(server, removed) == [] and _tags(server, redone) == []

    with _removed_from_tagging(server, "test gone"):
        redo = client.post(f"{API}/operations/redo")
        assert redo.status_code == 409, redo.text
        assert "test gone" in redo.json()["detail"]
        assert "removed from tagging" in redo.json()["detail"]
        assert _tags(server, redone) == []
        assert client.get(f"{API}/operations/{addition['id']}").json()["status"] == (
            "undone"
        )

        undo = client.post(f"{API}/operations/undo")
        assert undo.status_code == 409, undo.text
        assert "removed from tagging" in undo.json()["detail"]
        assert _tags(server, removed) == []
        assert client.get(f"{API}/operations/{removal['id']}").json()["status"] == (
            "applied"
        )

        client.post(f"{API}/pictures/{unrelated}/tags", json={"tag": "sunset"})
        client.post(
            f"{API}/pictures/{unrelated}/tags/remove_all", json={"tag": "sunset"}
        )
        assert client.post(f"{API}/operations/undo").status_code == 200
        assert _tags(server, unrelated) == ["sunset"]


# ---------------------------------------------------------------------------
# Scrapheap lifecycle (soft delete / restore)
# ---------------------------------------------------------------------------


def test_scrapheap_move_is_recorded_and_undo_brings_the_picture_back(client, server):
    """The core promise: a move to the Scrapheap is reversible from the log."""
    picture_id = _upload(client)
    assert _visible(client, picture_id) is True

    resp = client.delete(
        f"{API}/pictures/{picture_id}", headers={"X-Client-Id": "tab-1"}
    )
    assert resp.status_code == 200, resp.text
    deleted, deleted_at = _lifecycle(server, picture_id)
    assert deleted is True
    assert deleted_at is not None
    assert _visible(client, picture_id) is False

    operations = _operations(server, op_type="pictures.scrapheap.move")
    assert len(operations) == 1
    operation = operations[0]
    assert operation["target_ids"] == [picture_id]
    assert operation["undoable"] is True
    assert operation["summary"] == "Moved 1 picture to the Scrapheap"
    assert operation["origin_client_id"] == "tab-1"

    # The recorded facet is the soft-delete state itself, retention stamp
    # included, so undo restores the deadline rather than inventing one.
    detail = client.get(f"{API}/operations/{operation['id']}").json()
    assert detail["before"][str(picture_id)]["deleted"] == {
        "deleted": False,
        "deleted_at": None,
    }
    assert detail["after"][str(picture_id)]["deleted"]["deleted"] is True
    assert detail["after"][str(picture_id)]["deleted"]["deleted_at"] is not None

    undo = client.post(f"{API}/operations/undo")
    assert undo.status_code == 200, undo.text
    assert undo.json()["restored_picture_ids"] == [picture_id]
    assert undo.json()["scrapheaped_picture_ids"] == []
    assert _lifecycle(server, picture_id) == (False, None)
    assert _visible(client, picture_id) is True

    # Redo puts it back in the Scrapheap, with the SAME retention stamp.
    redo = client.post(f"{API}/operations/redo")
    assert redo.status_code == 200, redo.text
    assert redo.json()["scrapheaped_picture_ids"] == [picture_id]
    assert redo.json()["restored_picture_ids"] == []
    assert _lifecycle(server, picture_id) == (True, deleted_at)
    assert _visible(client, picture_id) is False


def test_scrapheap_lifecycle_announces_restored_not_added(client, server):
    """End to end: the WS envelope calls a scrapheap comeback ``restored``.

    Both the op-log undo path and the explicit restore endpoint put a card back,
    and both used to say ``added`` - which the SPA reads as "new to the vault"
    and answers with the sidebar's NEW marker on the counts that grew. That is a
    lie for a picture that has been in the library all along, so the comeback
    gets its own kind. Genuine imports keep ``added``; a re-scrapheap on redo
    keeps ``removed``.
    """
    picture_id = _upload(client)
    emitted: list[dict] = []
    real_notify = server.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and "change_kind" in data:
            emitted.append({"event": event_type, **data})
        return real_notify(event_type, data)

    server.vault.notify = _capture

    assert (
        client.delete(
            f"{API}/pictures/{picture_id}", headers={"X-Client-Id": "tab-1"}
        ).status_code
        == 200
    )
    # The move itself: the card goes away.
    assert emitted[-1]["change_kind"] == "removed"
    assert emitted[-1]["picture_ids"] == [picture_id]

    emitted.clear()
    assert client.post(f"{API}/operations/undo").status_code == 200
    undo_kinds = {
        kind["change_kind"]
        for kind in emitted
        if picture_id in (kind.get("picture_ids") or [])
    }
    assert undo_kinds == {"restored"}, emitted
    # Origin travels in ``data`` (§15) - the undo path runs on the DB worker
    # thread where the contextvar is dead.
    assert all("origin_client_id" in kind for kind in emitted)

    emitted.clear()
    assert client.post(f"{API}/operations/redo").status_code == 200
    redo_kinds = {
        kind["change_kind"]
        for kind in emitted
        if picture_id in (kind.get("picture_ids") or [])
    }
    assert redo_kinds == {"removed"}, emitted

    # …and the explicit endpoint agrees with the undo path.
    emitted.clear()
    assert (
        client.post(
            f"{API}/pictures/scrapheap/restore", json={"picture_ids": [picture_id]}
        ).status_code
        == 200
    )
    endpoint_kinds = {
        kind["change_kind"]
        for kind in emitted
        if picture_id in (kind.get("picture_ids") or [])
    }
    assert "restored" in endpoint_kinds, emitted
    assert "added" not in endpoint_kinds, emitted


def test_bulk_scrapheap_move_is_one_batch_and_one_undo(client, server):
    """Bulk = one batch id = one Undo, matching the log's grouping rule."""
    ids = [_upload(client) for _ in range(3)]
    resp = client.request("DELETE", f"{API}/pictures", json={"picture_ids": ids})
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted_count"] == 3

    operations = _operations(server, op_type="pictures.scrapheap.move")
    assert len(operations) == 1
    operation = operations[0]
    assert operation["target_ids"] == sorted(ids)
    assert operation["target_count"] == 3
    assert operation["batch_id"]
    assert operation["summary"] == "Moved 3 pictures to the Scrapheap"
    assert all(_lifecycle(server, pid)[0] is True for pid in ids)

    # The batch endpoint reverts the whole move in one call.
    undo = client.post(f"{API}/operations/batches/{operation['batch_id']}/undo")
    assert undo.status_code == 200, undo.text
    assert sorted(undo.json()["restored_picture_ids"]) == sorted(ids)
    assert all(_lifecycle(server, pid) == (False, None) for pid in ids)

    redo = client.post(f"{API}/operations/redo")
    assert redo.status_code == 200, redo.text
    assert all(_lifecycle(server, pid)[0] is True for pid in ids)


def test_restore_from_the_scrapheap_is_recorded_symmetrically(client, server):
    """Undoing a restore puts the pictures back - the history stays coherent."""
    ids = [_upload(client) for _ in range(2)]
    client.request("DELETE", f"{API}/pictures", json={"picture_ids": ids})
    stamps = {pid: _lifecycle(server, pid)[1] for pid in ids}
    assert all(stamp is not None for stamp in stamps.values())

    resp = client.post(f"{API}/pictures/scrapheap/restore")
    assert resp.status_code == 200, resp.text
    assert resp.json()["restored_count"] == 2
    assert all(_lifecycle(server, pid) == (False, None) for pid in ids)

    operations = _operations(server, op_type="pictures.scrapheap.restore")
    assert len(operations) == 1
    restore_op = operations[0]
    assert restore_op["target_ids"] == sorted(ids)
    assert restore_op["batch_id"]
    assert restore_op["summary"] == "Restored 2 pictures from the Scrapheap"

    # Undoing the restore is a re-scrapheap, stamp and all.
    undo = client.post(f"{API}/operations/undo")
    assert undo.status_code == 200, undo.text
    assert sorted(undo.json()["scrapheaped_picture_ids"]) == sorted(ids)
    for pid in ids:
        assert _lifecycle(server, pid) == (True, stamps[pid])

    # And redoing it restores them again.
    assert client.post(f"{API}/operations/redo").status_code == 200
    assert all(_lifecycle(server, pid) == (False, None) for pid in ids)


def test_restore_of_an_id_that_is_not_scrapheaped_records_nothing(client, server):
    picture_id = _upload(client)
    before = len(_operations(server))
    resp = client.post(
        f"{API}/pictures/scrapheap/restore", json={"picture_ids": [picture_id]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["restored_count"] == 0
    assert len(_operations(server)) == before


def test_undoing_a_move_whose_picture_was_purged_refuses_and_changes_nothing(
    client, server
):
    """The fail-closed edge case: a purge is permanent, so the undo is refused.

    Same contract as the locked-set guard - the WHOLE request is refused with a
    specific error, nothing is written, and the operation stays ``applied``
    rather than being marked undone over a change that did not happen.
    """
    picture_id = _upload(client)
    assert client.delete(f"{API}/pictures/{picture_id}").status_code == 200
    assert _operations(server, op_type="pictures.scrapheap.move")

    purged = _purge_forever(client, [picture_id])
    assert purged["deleted_count"] == 1
    assert _lifecycle(server, picture_id) is None

    resp = client.post(f"{API}/operations/undo")
    assert resp.status_code == 410, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "pictures_purged"
    assert detail["picture_ids"] == [picture_id]
    assert "permanently deleted" in detail["message"]

    # Refused, not half-applied.
    move = _operations(server, op_type="pictures.scrapheap.move")[0]
    assert move["status"] == "applied"
    assert move["undone_at"] is None


def test_a_partially_purged_bulk_move_refuses_the_whole_undo(client, server):
    """No silent partial success: one purged target refuses the entire batch."""
    ids = [_upload(client) for _ in range(3)]
    client.request("DELETE", f"{API}/pictures", json={"picture_ids": ids})
    _purge_forever(client, [ids[1]])
    assert _lifecycle(server, ids[1]) is None

    resp = client.post(f"{API}/operations/undo")
    assert resp.status_code == 410, resp.text
    assert resp.json()["detail"]["picture_ids"] == [ids[1]]

    # The survivors are untouched - the refusal rolled the whole thing back.
    assert _lifecycle(server, ids[0])[0] is True
    assert _lifecycle(server, ids[2])[0] is True
    assert (
        _operations(server, op_type="pictures.scrapheap.move")[0]["status"] == "applied"
    )


def test_permanent_deletes_record_no_operation(client, server):
    """Purge / Empty Scrapheap are NOT undoable and must leave no log row."""
    ids = [_upload(client) for _ in range(2)]
    client.request("DELETE", f"{API}/pictures", json={"picture_ids": ids})
    before = {op["id"] for op in _operations(server)}

    # Named selection, then "Empty Scrapheap" (no ids at all).
    _purge_forever(client, [ids[0]])
    preview = client.post(f"{API}/pictures/scrapheap/delete-preview", json=None)
    assert preview.status_code == 200, preview.text
    emptied = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={
            "include_protected": True,
            "confirm_token": preview.json()["confirm_token"],
        },
    )
    assert emptied.status_code == 200, emptied.text

    assert {op["id"] for op in _operations(server)} == before
    assert _operations(server, op_type="pictures.scrapheap.purge") == []


def test_scrapheap_move_of_a_stack_member_snapshots_the_whole_stack(client, server):
    """normalize_stack_positions renumbers siblings; undo must put them back."""
    ids = [_upload(client) for _ in range(3)]
    stacked = client.post(f"{API}/stacks", json={"picture_ids": ids})
    assert stacked.status_code == 200, stacked.text

    def positions(session):
        return {
            int(row.id): row.stack_position
            for row in session.exec(select(Picture).where(Picture.id.in_(ids))).all()
        }

    before = server.vault.db.run_task(positions)
    leader = next(pid for pid, pos in before.items() if pos == 0)

    assert client.delete(f"{API}/pictures/{leader}").status_code == 200
    after = server.vault.db.run_task(positions)
    assert after != before, "deleting the leader should promote a sibling"

    operation = _operations(server, op_type="pictures.scrapheap.move")[0]
    # Every renumbered sibling is in the snapshot, not just the deleted one.
    assert set(operation["target_ids"]) >= {
        pid for pid in ids if before[pid] != after[pid]
    }

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert server.vault.db.run_task(positions) == before
    assert _lifecycle(server, leader) == (False, None)


def test_undo_announces_the_kept_stack_siblings_by_their_stacks(
    client, server, monkeypatch, key_list_plan
):
    """Undoing a move across two stacks re-announces each stack's other kept
    members, read by stack id, and not a member still in the Scrapheap. The
    undo's lock check expands the moved pictures to their stacks the same way."""
    first = [_upload(client) for _ in range(3)]
    second = [_upload(client) for _ in range(2)]
    loose = _upload(client)
    for members in (first, second):
        stacked = client.post(f"{API}/stacks", json={"picture_ids": members})
        assert stacked.status_code == 200, stacked.text
    assert client.delete(f"{API}/pictures/{first[2]}").status_code == 200
    moved = [first[0], second[0]]
    resp = client.request("DELETE", f"{API}/pictures", json={"picture_ids": moved})
    assert resp.status_code == 200, resp.text
    emitted: list[dict] = []
    real_notify = server.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and data.get("fields") == ["stack_count"]:
            emitted.append(data)
        return real_notify(event_type, data)

    monkeypatch.setattr(server.vault, "notify", _capture)

    with key_list_plan(
        server.vault.db._engine,
        "FROM picture \nWHERE picture.stack_id IN",
        "ix_picture_stack_id",
    ):
        assert client.post(f"{API}/operations/undo").status_code == 200
    assert [data["picture_ids"] for data in emitted] == [[first[1], second[1]]]
    assert _lifecycle(server, first[2])[0] is True

    seeds = [*moved, loose]
    kept = sorted([*first[:2], *second, loose])
    with key_list_plan(
        server.vault.db._engine,
        "FROM picture \nWHERE picture.stack_id IN",
        "ix_picture_stack_id",
    ):
        assert server.vault.db.run_task(expand_picture_ids_to_stacks, seeds) == kept
    assert server.vault.db.run_task(
        expand_picture_ids_to_stacks, seeds, True
    ) == sorted([*kept, first[2]])
    assert server.vault.db.run_task(expand_picture_ids_to_stacks, [loose]) == [loose]
    assert server.vault.db.run_task(expand_picture_ids_to_stacks, []) == []


def test_scrapheap_move_of_a_locked_picture_is_refused_and_records_nothing(
    client, server
):
    picture_id = _upload(client)
    _lock_picture(server, picture_id)

    assert client.delete(f"{API}/pictures/{picture_id}").status_code == 423
    assert _lifecycle(server, picture_id) == (False, None)
    assert _operations(server, op_type="pictures.scrapheap.move") == []


def test_scrapheap_summaries_count_the_recorded_change_not_the_request():
    """A skipped (locked / already-deleted) picture must not inflate the toast."""
    move = operation_log_service.scrapheap_move_summary
    restore = operation_log_service.scrapheap_restore_summary
    facet = operation_log_service.FACET_DELETED

    after = {
        "1": {facet: {"deleted": True, "deleted_at": "2026-07-29T00:00:00"}},
        "2": {facet: {"deleted": True, "deleted_at": "2026-07-29T00:00:00"}},
        # A stack sibling that was only renumbered is not a move.
        "3": {"stack": {"id": 1, "name": None, "position": 0}},
    }
    assert move({}, after) == "Moved 2 pictures to the Scrapheap"
    assert restore({}, after) is None

    restored = {"9": {facet: {"deleted": False, "deleted_at": None}}}
    assert restore({}, restored) == "Restored 1 picture from the Scrapheap"
    assert move({}, restored) is None


# ---------------------------------------------------------------------------
# Tag-review decisions: confirm / reject (§21.2)
# ---------------------------------------------------------------------------

# An anomaly-vocabulary tag, so the decision moves the scorer's inputs too.
ANOMALY_TAG = "watermark"


def test_reject_is_recorded_and_undo_clears_the_human_negative(client, server):
    """The reported gap: a reject raised no receipt and could not be undone.

    Undo must leave the ledger as if the reject never happened - status back to
    PENDING and ``label_source`` back to null. A row still carrying a human NEG
    is one the tagger and the training exporter would go on treating as refused,
    which is the half-undo this facet exists to prevent.
    """
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)

    resp = client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/reject",
        headers={"X-Client-Id": "tab-1"},
    )
    assert resp.status_code == 200, resp.text

    operations = _operations(server, op_type="pictures.tags.reject")
    assert len(operations) == 1
    operation = operations[0]
    assert operation["target_ids"] == [picture_id]
    assert operation["undoable"] is True
    assert operation["status"] == "applied"
    assert operation["summary"] == f"Removed tag '{ANOMALY_TAG}'"
    assert operation["origin_client_id"] == "tab-1"
    assert operation["source"] == "ui"

    rejected = _prediction(server, picture_id, ANOMALY_TAG)
    assert rejected["status"] == "REJECTED"
    assert rejected["label_state"] == "NEG"
    assert rejected["label_source"] == "human"
    # The tagger version/confidence the reviewer overruled is snapshotted.
    assert rejected["label_model_version"] == "test-v1"
    assert rejected["label_confidence"] == 0.99

    # The recorded facet is the prediction/ledger state, not an inverse.
    detail = client.get(f"{API}/operations/{operation['id']}").json()
    assert set(detail["before"][str(picture_id)]) == {"tag_predictions"}
    assert (
        detail["before"][str(picture_id)]["tag_predictions"][ANOMALY_TAG][
            "label_source"
        ]
        is None
    )

    assert client.post(f"{API}/operations/undo").status_code == 200
    reopened = _prediction(server, picture_id, ANOMALY_TAG)
    assert reopened["status"] == "PENDING"
    assert reopened["label_state"] == "UNKNOWN"
    assert reopened["label_source"] is None
    assert reopened["label_model_version"] is None
    assert reopened["label_confidence"] is None
    # The tagger's own live fields are never rolled back by an undo.
    assert reopened["model_version"] == "test-v1"
    assert reopened["confidence"] == 0.99

    assert client.post(f"{API}/operations/redo").status_code == 200
    assert _prediction(server, picture_id, ANOMALY_TAG)["label_state"] == "NEG"


def test_reject_of_a_hand_added_tag_round_trips_the_synthetic_manual_row(
    client, server
):
    """A tag the tagger never predicted has its NEG parked on an invented row.

    That row is the only prediction row a user action can create, so it is also
    the only one an undo may delete - and a redo has to be able to rebuild it.
    """
    picture_id = _upload(client)
    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"}
        ).status_code
        == 200
    )
    # A content tag outside the tagger's label space: no prediction on file.
    assert _prediction(server, picture_id, "sunset") is None

    resp = client.post(f"{API}/pictures/{picture_id}/tag_predictions/sunset/reject")
    assert resp.status_code == 200, resp.text
    synthetic = _prediction(server, picture_id, "sunset")
    assert synthetic["model_version"] == "manual"
    assert synthetic["status"] == "REJECTED"
    assert synthetic["label_state"] == "NEG"

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _prediction(server, picture_id, "sunset") is None

    assert client.post(f"{API}/operations/redo").status_code == 200
    rebuilt = _prediction(server, picture_id, "sunset")
    assert rebuilt["model_version"] == "manual"
    assert rebuilt["status"] == "REJECTED"
    assert rebuilt["label_state"] == "NEG"


def test_redo_never_rebuilds_a_label_row_for_a_tag_removed_from_tagging(client, server):
    """A label on a removed tag is data about it, and is not written back either."""
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "test-gone label"})
    resp = client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/test-gone label/reject"
    )
    assert resp.status_code == 200, resp.text
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _prediction(server, picture_id, "test-gone label") is None

    with _removed_from_tagging(server, "test-gone label"):
        redo = client.post(f"{API}/operations/redo")
        assert redo.status_code == 409, redo.text
        assert "removed from tagging" in redo.json()["detail"]
        assert _prediction(server, picture_id, "test-gone label") is None
        assert _operations(server, status="undone")[0]["op_type"] == (
            "pictures.tags.reject"
        )


def test_confirm_is_recorded_and_undo_removes_the_tag_row_it_created(client, server):
    """Confirm promotes a prediction to a Tag; undo takes both back."""
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)
    assert _tags(server, picture_id) == []

    resp = client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/confirm"
    )
    assert resp.status_code == 200, resp.text
    assert _tags(server, picture_id) == [ANOMALY_TAG]

    operations = _operations(server, op_type="pictures.tags.confirm")
    assert len(operations) == 1
    assert operations[0]["summary"] == f"Confirmed tag '{ANOMALY_TAG}'"
    assert operations[0]["target_ids"] == [picture_id]
    confirmed = _prediction(server, picture_id, ANOMALY_TAG)
    assert confirmed["status"] == "CONFIRMED"
    assert confirmed["label_state"] == "POS"

    # Both facets moved, so both are recorded and both come back.
    detail = client.get(f"{API}/operations/{operations[0]['id']}").json()
    assert set(detail["before"][str(picture_id)]) == {"tags", "tag_predictions"}

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _tags(server, picture_id) == []
    reopened = _prediction(server, picture_id, ANOMALY_TAG)
    assert reopened["status"] == "PENDING"
    assert reopened["label_state"] == "UNKNOWN"
    assert reopened["label_source"] is None

    assert client.post(f"{API}/operations/redo").status_code == 200
    assert _tags(server, picture_id) == [ANOMALY_TAG]
    assert _prediction(server, picture_id, ANOMALY_TAG)["label_state"] == "POS"


def test_undo_recomputes_the_derived_uncertainty_and_drops_the_cached_score(
    client, server
):
    """Derived data is re-derived on restore, never snapshotted.

    ``anomaly_tag_uncertainty`` is a function of the labels, and ``smart_score``
    is a cache of a function of them. Restoring a snapshot of either would let
    the pair drift out of step with the rows the undo just wrote.
    """
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)

    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/confirm"
        ).status_code
        == 200
    )
    uncertainty, _score = _picture_scores(server, picture_id)
    # Tag applied: the model was 0.99 sure, so the human/model gap is 0.01.
    assert uncertainty == pytest.approx(0.01)

    # Pretend the background scorer has since filled the cache back in.
    _set_smart_score(server, picture_id, 0.5)

    assert client.post(f"{API}/operations/undo").status_code == 200
    uncertainty, score = _picture_scores(server, picture_id)
    # Tag gone again: the gap is the model's raw confidence, recomputed.
    assert uncertainty == pytest.approx(0.99)
    # And the cached score the confirm had moved is dropped for recompute.
    assert score is None


def test_removing_a_tag_chip_end_to_end_is_fully_undoable(client, server):
    """The overlay gesture the bug was reported against, start to finish.

    Removing a chip is two requests - ``tags/remove_all`` then the reject that
    makes the removal durable supervision. Each records its own operation (no
    batch id, exactly like the other single-picture tag ops), and walking both
    back leaves the tag applied with **no** rejection on file.

    This is the *unbatched* path - no ``X-Operation-Batch-Id``, as an external
    caller would issue it. The in-app gesture stamps both requests with one id
    and takes a single Ctrl+Z; see
    ``test_removing_a_tag_chip_with_a_gesture_batch_id_is_one_undo_step``.
    """
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    assert _tags(server, picture_id) == ["sunset"]

    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tags/remove_all", json={"tag": "sunset"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tag_predictions/sunset/reject"
        ).status_code
        == 200
    )
    assert _tags(server, picture_id) == []
    assert _prediction(server, picture_id, "sunset")["label_state"] == "NEG"

    recorded = _operations(server)
    assert [op["op_type"] for op in recorded] == [
        "pictures.tags.reject",
        "pictures.tags.remove_all",
        "pictures.tags.add",
    ]
    assert all(op["batch_id"] is None for op in recorded)

    # Undo is last-in-first-out: the reject, then the removal.
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _prediction(server, picture_id, "sunset") is None
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _tags(server, picture_id) == ["sunset"]
    assert _prediction(server, picture_id, "sunset") is None


def test_removing_a_tag_chip_with_a_gesture_batch_id_is_one_undo_step(client, server):
    """The fix: one user gesture, one history step, one Ctrl+Z.

    The lightbox stamps both requests of the chip delete with the same
    ``X-Operation-Batch-Id``. They still record two operations - the log stays a
    faithful record of what happened - but they share a batch, and undoing the
    newest reverts the whole batch, so a single Ctrl+Z brings the tag back AND
    clears the human NEG the reject wrote.
    """
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    assert _tags(server, picture_id) == ["sunset"]

    headers = {"X-Operation-Batch-Id": "cli-gesture-abc123"}
    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tags/remove_all",
            json={"tag": "sunset"},
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tag_predictions/sunset/reject",
            headers=headers,
        ).status_code
        == 200
    )
    assert _tags(server, picture_id) == []
    assert _prediction(server, picture_id, "sunset")["label_state"] == "NEG"

    gesture = _operations(server, batch_id="cli-gesture-abc123")
    assert [op["op_type"] for op in gesture] == [
        "pictures.tags.reject",
        "pictures.tags.remove_all",
    ]
    # The `pictures.tags.add` that seeded the tag is a different gesture and
    # must stay out of the batch.
    assert [op["batch_id"] for op in _operations(server)] == [
        "cli-gesture-abc123",
        "cli-gesture-abc123",
        None,
    ]

    # ONE Ctrl+Z: the tag is back and the ledger with it.
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _tags(server, picture_id) == ["sunset"]
    assert _prediction(server, picture_id, "sunset") is None
    assert all(
        op["status"] == "undone"
        for op in _operations(server, batch_id="cli-gesture-abc123")
    )


def test_the_batch_undo_endpoint_reverts_a_client_gesture_in_one_call(client, server):
    """``POST /operations/batches/{batch_id}/undo`` over a client-supplied id.

    Same unit as the implicit undo above, reached the other way - the receipt's
    Undo button knows the batch id it just created and reverts by it.
    """
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})
    headers = {"X-Operation-Batch-Id": "cli-gesture-xyz789"}
    client.post(
        f"{API}/pictures/{picture_id}/tags/remove_all",
        json={"tag": "sunset"},
        headers=headers,
    )
    client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/sunset/reject",
        headers=headers,
    )

    reverted = client.post(f"{API}/operations/batches/cli-gesture-xyz789/undo")
    assert reverted.status_code == 200, reverted.text
    assert len(reverted.json()["operations"]) == 2
    assert _tags(server, picture_id) == ["sunset"]
    assert _prediction(server, picture_id, "sunset") is None


def test_a_malformed_gesture_batch_header_is_ignored_never_a_500(client, server):
    """A header is attacker-controllable: bad values degrade, they do not fail.

    Each rejected value records an unbatched operation - exactly the behaviour
    every caller had before the header existed.
    """
    picture_id = _upload(client)
    for i, bad in enumerate(
        (
            "srv-" + "0" * 32,  # cannot impersonate a server-minted batch
            "no-namespace",
            "cli-" + "a" * 200,  # oversized
            "cli-has spaces",
            "cli-abcd\n",  # G1: a trailing LF must not pass (fullmatch, not match)
        )
    ):
        resp = client.post(
            f"{API}/pictures/{picture_id}/tags",
            json={"tag": f"t{i}"},
            headers={"X-Operation-Batch-Id": bad},
        )
        assert resp.status_code == 200, resp.text

    recorded = _operations(server, op_type="pictures.tags.add")
    assert len(recorded) == 5
    assert all(op["batch_id"] is None for op in recorded), recorded


def test_a_bulk_confirm_gesture_is_one_batch_over_many_pictures(client, server):
    """The tag-panel fan-out: N confirms, one gesture id, one undo unit.

    This is what gives the receipt its ``+N`` count and makes "confirm on all"
    reversible in one press instead of N.
    """
    picture_ids = [_upload(client) for _ in range(3)]
    for picture_id in picture_ids:
        _seed_prediction(server, picture_id, "sunset", confidence=0.99)

    headers = {"X-Operation-Batch-Id": "cli-confirm-all-1"}
    for picture_id in picture_ids:
        assert (
            client.post(
                f"{API}/pictures/{picture_id}/tag_predictions/sunset/confirm",
                headers=headers,
            ).status_code
            == 200
        )
    assert all(_tags(server, pid) == ["sunset"] for pid in picture_ids)

    batch = _operations(server, batch_id="cli-confirm-all-1")
    assert len(batch) == 3

    assert client.post(f"{API}/operations/undo").status_code == 200
    assert all(_tags(server, pid) == [] for pid in picture_ids)
    assert all(
        _prediction(server, pid, "sunset")["label_source"] is None
        for pid in picture_ids
    )


def test_a_reject_on_a_locked_picture_is_refused_and_records_nothing(client, server):
    """The 423 fires inside the recorded task, before anything is written."""
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)
    _lock_picture(server, picture_id)

    rejected = client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/reject"
    )
    assert rejected.status_code == 423, rejected.text
    confirmed = client.post(
        f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/confirm"
    )
    assert confirmed.status_code == 423, confirmed.text

    assert _operations(server, op_type="pictures.tags.reject") == []
    assert _operations(server, op_type="pictures.tags.confirm") == []
    # The decision itself did not land either.
    assert _prediction(server, picture_id, ANOMALY_TAG)["status"] == "PENDING"
    assert _tags(server, picture_id) == []


def test_a_reject_of_an_unpredicted_tag_on_a_confirmed_one_does_not_batch(
    client, server
):
    """Two decisions are two independent steps, as for every other tag op."""
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)
    _seed_prediction(server, picture_id, "noise", confidence=0.98)

    for tag in (ANOMALY_TAG, "noise"):
        assert (
            client.post(
                f"{API}/pictures/{picture_id}/tag_predictions/{tag}/reject"
            ).status_code
            == 200
        )

    operations = _operations(server, op_type="pictures.tags.reject")
    assert len(operations) == 2
    assert all(op["batch_id"] is None for op in operations)

    # One Undo reverts one decision, not both.
    assert client.post(f"{API}/operations/undo").status_code == 200
    assert _prediction(server, picture_id, "noise")["label_source"] is None
    assert _prediction(server, picture_id, ANOMALY_TAG)["label_source"] == "human"


def test_undo_of_a_reject_leaves_a_later_tagger_prediction_alone(client, server):
    """Only the synthetic row a decision invents may be deleted by its undo.

    A prediction the tagger wrote *after* the operation was recorded is model
    output nobody asked to revert; silently deleting it would make undo a data
    loss path.
    """
    picture_id = _upload(client)
    _seed_prediction(server, picture_id, ANOMALY_TAG, confidence=0.99)
    assert (
        client.post(
            f"{API}/pictures/{picture_id}/tag_predictions/{ANOMALY_TAG}/reject"
        ).status_code
        == 200
    )
    # The tagger runs in the background and scores another label.
    _seed_prediction(server, picture_id, "noise", confidence=0.42)

    assert client.post(f"{API}/operations/undo").status_code == 200
    survivor = _prediction(server, picture_id, "noise")
    assert survivor is not None
    assert survivor["confidence"] == 0.42


# ---------------------------------------------------------------------------
# The tagger's write never deletes a tag it did not write (#1357)
# ---------------------------------------------------------------------------

# Outside the anomaly vocabulary, so no human-ledger row protects it.
CONTENT_TAG = "character_x"


def _tagger_write(server, picture_id, model_tags):
    """Land a tag pass's write for *picture_id*, as the queued TagTask would."""
    server.vault.db.run_task(
        TagTask._add_tags_bulk, [{"pic_id": picture_id, "tags": model_tags}]
    )


def _raw_tags(server, picture_id):
    """Every Tag row on the picture, sentinels included."""
    return sorted(
        server.vault.db.run_task(
            lambda session: session.exec(
                select(Tag.tag).where(Tag.picture_id == picture_id)
            ).all()
        )
    )


def test_a_manual_tag_added_while_tagging_is_pending_survives_the_write(client, server):
    """The reported race: the task captured the picture, the owner tagged it,
    and the task's write then deleted the tag the owner had just added."""
    picture_id = _upload(client)
    assert _raw_tags(server, picture_id) == ["__tag"]

    # The task has its batch; the owner adds a tag before the write lands.
    resp = client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": CONTENT_TAG})
    assert resp.status_code == 200, resp.text
    assert _prediction(server, picture_id, CONTENT_TAG) is None

    _tagger_write(server, picture_id, ["woman"])

    assert _raw_tags(server, picture_id) == sorted([CONTENT_TAG, "woman"])


def test_a_retag_replaces_model_tags_and_keeps_tags_other_writers_added(client, server):
    """A legitimate retag still updates the model's tags (the reset clears
    them) but keeps what arrived while it was pending - through the tag route,
    which drops the sentinel, and through a writer that leaves it in place."""
    via_route, beside_sentinel = _upload(client), _upload(client)
    for picture_id in (via_route, beside_sentinel):
        _tagger_write(server, picture_id, ["old-model-tag"])
        assert _raw_tags(server, picture_id) == ["old-model-tag"]
    tag_prediction_service.reset_pictures_tags(
        server.vault, [via_route, beside_sentinel], engine_name="some_plugin"
    )

    client.post(f"{API}/pictures/{via_route}/tags", json={"tag": CONTENT_TAG})

    def _add_beside_sentinel(session):
        session.add(Tag(picture_id=beside_sentinel, tag=CONTENT_TAG))
        session.commit()

    server.vault.db.run_task(_add_beside_sentinel)
    assert _raw_tags(server, beside_sentinel) == sorted(
        ["__tag:some_plugin", CONTENT_TAG]
    )

    for picture_id in (via_route, beside_sentinel):
        _tagger_write(server, picture_id, ["new-model-tag"])
    expected = sorted([CONTENT_TAG, "new-model-tag"])
    assert {
        "via_route": _raw_tags(server, via_route),
        "beside_sentinel": _raw_tags(server, beside_sentinel),
    } == {"via_route": expected, "beside_sentinel": expected}


def test_a_task_that_read_the_picture_before_a_retag_writes_nothing(client, server):
    """#1361: an older task finishing after a retag must neither land its tags
    nor its predictions, or the retag's own task keeps them as another
    writer's and the picture ends up with both models' output."""
    picture_id = _upload(client)
    picture = [SimpleNamespace(id=picture_id)]
    stale = TagTask(database=server.vault.db, tagging_workflow=None, pictures=picture)
    tag_prediction_service.reset_pictures_tags(
        server.vault, [picture_id], engine_name="fresh_plugin"
    )
    fresh = TagTask(database=server.vault.db, tagging_workflow=None, pictures=picture)

    written = server.vault.db.run_task(
        stale._add_tags_unless_reset,
        [{"pic_id": picture_id, "tags": ["stale-model-tag"]}],
    )
    predicted = server.vault.db.run_task(
        stale._write_predictions_unless_reset,
        {picture_id: {"stale-model-tag": 0.9}},
        {picture_id: {"stale-model-tag"}},
        "stale-model",
    )
    assert (written, predicted) == ([], 0)
    assert _raw_tags(server, picture_id) == ["__tag:fresh_plugin"]
    assert _prediction(server, picture_id, "stale-model-tag") is None

    server.vault.db.run_task(
        fresh._add_tags_unless_reset,
        [{"pic_id": picture_id, "tags": ["fresh-model-tag"]}],
    )
    assert _raw_tags(server, picture_id) == ["fresh-model-tag"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_list_filters_and_rejects_a_bad_status(client, server):
    picture_id = _upload(client)
    client.post(f"{API}/pictures/{picture_id}/tags", json={"tag": "sunset"})

    assert (
        client.get(f"{API}/operations", params={"status": "bogus"}).status_code == 400
    )

    applied = client.get(f"{API}/operations", params={"status": "applied"}).json()
    assert len(applied) == 1
    # The list omits the (potentially huge) before/after payloads.
    assert "before" not in applied[0]

    assert client.get(f"{API}/operations", params={"op_type": "nope"}).json() == []


def test_get_unknown_operation_is_404(client, server):
    assert client.get(f"{API}/operations/999999").status_code == 404
    assert client.post(f"{API}/operations/999999/undo").status_code == 409


def test_every_operations_route_is_declared_owner_only():
    """Both-direction authz record: the gate, not a handler, guards these.

    Arithmetic completeness - every mounted /operations route has a declaration
    and every declaration is OWNER_ONLY. The positive direction (the owner
    reaches them) is exercised by every other test in this module.
    """
    from pixlstash.authz.policy import AccessPolicy
    from pixlstash.authz.registry import ROUTE_POLICIES

    declared = {
        (method, path): policy
        for (method, path), policy in ROUTE_POLICIES.items()
        if path.startswith("/api/v1/operations")
    }
    assert set(declared) == {
        ("GET", "/api/v1/operations"),
        ("GET", "/api/v1/operations/undo-state"),
        ("GET", "/api/v1/operations/{operation_id}"),
        ("POST", "/api/v1/operations/undo"),
        ("POST", "/api/v1/operations/redo"),
        ("POST", "/api/v1/operations/{operation_id}/undo"),
        ("POST", "/api/v1/operations/batches/{batch_id}/undo"),
    }
    assert all(policy.policy is AccessPolicy.OWNER_ONLY for policy in declared.values())


def test_operations_routes_have_no_inline_authz_check():
    """§16.1: the gate owns object authorization; handlers carry none."""
    import inspect

    from pixlstash.routes import operations as operations_routes

    source = inspect.getsource(operations_routes)
    for forbidden in (
        "enforce_picture_scope",
        "require_unscoped_owner",
        "fetch_scope_allowed_picture_ids",
        "token_scope",
    ):
        assert forbidden not in source, (
            f"{forbidden} is an inline authz check; the AuthzGate owns "
            "authorization for these routes (docs/backend_architecture.md §16.1)"
        )


# ── post-restore hooks ────────────────────────────────────────────────────────


def _register_recording_hook(op_type):
    """Register a hook that records its calls, and return the call list.

    The registry is module-global, so the hook is removed again by the caller's
    ``finally`` to keep tests independent.
    """
    calls = []

    def _hook(session, operations, direction):
        calls.append((direction, sorted(int(op.id) for op in operations)))

    operation_log_service.register_post_restore_hook(op_type, _hook)
    return calls


def test_a_post_restore_hook_runs_once_per_restore_with_its_whole_batch(client, server):
    """The generic seam a feature uses to reopen what an operation also decided.

    An operation can change state the recorded picture facets do not cover (the
    v1.9 duplicate verdict is the first). Without a hook, undo restored the
    pictures and left that decision standing. The contract asserted here is what
    the feature relies on: called once per restore, with **every** operation of
    its own op_type, with the direction, and never for a foreign op_type.
    """
    try:
        calls = _register_recording_hook("test.hooked")
        other = _register_recording_hook("test.unhooked")
        ids = [_upload(client) for _ in range(2)]
        batch_id = operation_log_service.new_batch_id()

        def _tag_one(session, picture_id, tag):
            session.add(Tag(picture_id=picture_id, tag=tag))
            session.flush()

        for picture_id in ids:
            operation_log_service.run_recorded_metadata_task(
                server.vault,
                _tag_one,
                picture_id,
                "hooked",
                op_type="test.hooked",
                picture_ids=[picture_id],
                batch_id=batch_id,
                summary="Hooked",
            )
        recorded = sorted(op["id"] for op in _operations(server, batch_id=batch_id))
        assert len(recorded) == 2
        assert calls == [], "recording must not fire a restore hook"

        assert client.post(f"{API}/operations/undo").status_code == 200
        assert calls == [("undo", recorded)], calls

        assert client.post(f"{API}/operations/redo").status_code == 200
        assert calls == [("undo", recorded), ("redo", recorded)], calls

        # A hook registered for a different op_type never saw any of it.
        assert other == []
    finally:
        operation_log_service._POST_RESTORE_HOOKS.pop("test.hooked", None)
        operation_log_service._POST_RESTORE_HOOKS.pop("test.unhooked", None)


def test_a_failing_post_restore_hook_aborts_the_whole_undo(client, server):
    """The hook runs inside the restore's transaction, so it fails closed.

    A hook that could fail *after* the state was committed would leave the
    pictures restored and the feature's own state stale - exactly the split the
    hook exists to prevent.
    """
    try:

        def _boom(session, operations, direction):
            raise RuntimeError("hook refused")

        operation_log_service.register_post_restore_hook("test.failing", _boom)
        picture_id = _upload(client)

        def _tag_one(session, pid, tag):
            session.add(Tag(picture_id=pid, tag=tag))
            session.flush()

        operation_log_service.run_recorded_metadata_task(
            server.vault,
            _tag_one,
            picture_id,
            "kept",
            op_type="test.failing",
            picture_ids=[picture_id],
            summary="Failing",
        )
        assert _tags(server, picture_id) == ["kept"]

        # The TestClient re-raises an unhandled server exception rather than
        # turning it into a 500, so the refusal surfaces here.
        with pytest.raises(RuntimeError, match="hook refused"):
            client.post(f"{API}/operations/undo")
        # Nothing was written: the tag is still there and the operation is still
        # applied, so the user can retry rather than being left half-undone.
        assert _tags(server, picture_id) == ["kept"]
        assert [op["status"] for op in _operations(server, op_type="test.failing")] == [
            "applied"
        ]
    finally:
        operation_log_service._POST_RESTORE_HOOKS.pop("test.failing", None)


def test_a_server_minted_batch_id_is_namespaced():
    """M1: an un-namespaced id is indistinguishable from a client-supplied one."""
    assert operation_log_service.new_batch_id().startswith(
        operation_log_service.SERVER_BATCH_ID_PREFIX
    )
    assert operation_log_service.new_batch_id() != operation_log_service.new_batch_id()


# ---------------------------------------------------------------------------
# Read-only listing paths stay off the single writer
# ---------------------------------------------------------------------------

# The table a write statement targets. Reads may still fill a connection-local
# TEMP scope table (``scope_id_subquery``); only a library table counts.
_WRITE_TARGET = re.compile(
    r"^\s*(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?"
    r"|DELETE\s+FROM)\s+[\"`\[]?(\w+)",
    re.IGNORECASE,
)

# Long enough for a cold CPU-only listing, far short of "never": a request that
# waits on the held writer does not return until the hold is released.
_READ_DEADLINE_S = 10


@contextmanager
def _writer_held(server):
    """Occupy the vault's single writer thread for the duration of the block.

    A control task runs on the writer thread itself, so while it waits nothing
    else queued on the writer (``run_task``) can start.
    """
    started = threading.Event()
    release = threading.Event()

    def _hold():
        started.set()
        release.wait()

    held = server.vault.db.submit_control_task(_hold)
    try:
        assert started.wait(timeout=30), "the writer never picked up the hold"
        yield
    finally:
        release.set()
        held.result(timeout=30)


@contextmanager
def _vault_writes(server):
    """Collect every write to a library table the vault engine executes in the block."""
    engine = server.vault.db._engine
    library_tables = set(sa_inspect(engine).get_table_names())
    assert {"picture", "tag", "face"} <= library_tables
    writes: list[str] = []

    def _before_cursor_execute(
        _conn, _cursor, statement, _parameters, _context, _executemany
    ):
        target = _WRITE_TARGET.match(statement)
        if target and target.group(1).lower() in library_tables:
            writes.append(statement)

    event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    try:
        yield writes
    finally:
        event.remove(engine, "before_cursor_execute", _before_cursor_execute)


def test_read_only_listing_paths_neither_wait_on_the_writer_nor_write(
    client, server, monkeypatch
):
    """Search and the special listing modes answer while the writer is busy.

    These paths only read. Queued on the single writer they wait behind every
    pending write and hold back the next one; as immediate reads they answer
    while a write is still running. The writer is held for the whole check, so
    a request routed through it never returns, and no statement may write.
    """
    # Text search encodes its query on the request thread; no model is needed
    # to exercise the read that follows.
    monkeypatch.setattr(server.vault, "generate_text_embedding", lambda _query: None)
    monkeypatch.setattr(
        server.vault, "generate_clip_text_embedding", lambda _query: None
    )
    penalised_tag = next(iter(DEFAULT_SMART_SCORE_PENALIZED_TAGS))
    live_id = _upload(client)
    scrapped_id = _upload(client)
    assert client.delete(f"{API}/pictures/{scrapped_id}").status_code == 200

    def _seed(session):
        character = Character(name="Barrier Person")
        session.add(character)
        session.flush()
        session.add(
            Face(
                picture_id=live_id,
                character_id=character.id,
                bbox_=json.dumps([1, 1, 8, 8]),
            )
        )
        session.add(Tag(picture_id=live_id, tag=penalised_tag))
        session.commit()
        return character.id

    character_id = server.vault.db.run_task(_seed)
    missing_character_id = character_id + 1000

    requests = {
        "text search": lambda: client.get(
            f"{API}/pictures/search",
            params={"query": penalised_tag, "threshold": 0.01},
        ),
        "likeness sort": lambda: client.get(
            f"{API}/pictures",
            params={
                "sort": "CHARACTER_LIKENESS",
                "character_id": missing_character_id,
                "reference_character_id": missing_character_id,
            },
        ),
        "unassigned count": lambda: client.get(
            f"{API}/pictures/count", params={"character_id": "UNASSIGNED"}
        ),
        "unassigned list": lambda: client.get(
            f"{API}/pictures", params={"character_id": "UNASSIGNED"}
        ),
        "scrapheap count": lambda: client.get(
            f"{API}/pictures/count", params={"only_deleted": "true"}
        ),
        # Filtered to a live picture so the page is empty: a non-empty Scrapheap
        # page then reads its retention fields through the writer queue, which
        # would hide whether the listing read itself does.
        "scrapheap list": lambda: client.get(
            f"{API}/pictures", params={"only_deleted": "true", "id": live_id}
        ),
        "no-project list": lambda: client.get(
            f"{API}/pictures", params={"project_id": "UNASSIGNED"}
        ),
    }

    responses = {}
    blocked = []
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        with _writer_held(server), _vault_writes(server) as writes:
            for label, send in requests.items():
                pending = pool.submit(send)
                try:
                    responses[label] = pending.result(timeout=_READ_DEADLINE_S)
                except TimeoutError:
                    blocked.append(label)

    assert blocked == [], f"waited on the held writer: {blocked}"
    assert writes == [], f"a read-only path wrote: {writes}"
    for label, response in responses.items():
        assert response.status_code == 200, f"{label}: {response.text}"

    def _ids(response):
        payload = response.json()
        rows = payload["pictures"] if isinstance(payload, dict) else payload
        return {int(row["id"]) for row in rows}

    # The answers are the library's, not an error path's empty default. The
    # Scrapheap holds only this test's picture (``reset_operation_log``).
    assert responses["scrapheap count"].json()["count"] == 1
    assert live_id in _ids(responses["no-project list"])


def test_thumbnail_faces_and_geometry_come_from_one_committed_state(
    client, server, monkeypatch
):
    """A thumbnail batch never pairs old picture geometry with new face boxes.

    The face boxes come back scaled by the picture's thumbnail and source
    dimensions, which the batch reads in one statement and the faces in the
    next. A rotation rewrites both in one commit. If that commit can land
    between the two statements, the boxes are scaled by geometry that no
    longer belongs to them and the overlay lands in the wrong place.

    The write is forced into that gap: it is submitted when the request's own
    read issues its first face statement, and the response must still carry
    the state from before it.
    """
    picture_id = _upload(client)
    old_bbox = [2, 2, 6, 4]
    new_bbox = [4, 2, 8, 6]

    def _seed(session):
        picture = session.get(Picture, picture_id)
        # A 2x bitmap, so a mapped box differs from the stored one.
        picture.thumbnail_width = picture.width * 2
        picture.thumbnail_height = picture.height * 2
        face = Face(picture_id=picture_id, bbox_=json.dumps(old_bbox))
        session.add(face)
        session.commit()
        return face.id, picture.width

    face_id, width = server.vault.db.run_task(_seed)

    def _rotation_shaped_write(session):
        session.get(Face, face_id).bbox_ = json.dumps(new_bbox)
        picture = session.get(Picture, picture_id)
        picture.thumbnail_width = None
        picture.thumbnail_height = None
        session.commit()

    engine = server.vault.db._engine
    writer = server.vault.db._task_worker
    armed = threading.Event()
    readers = []
    writes = []
    find = Picture.find.__func__

    # The shared server's background work also reads this picture, on task-runner
    # threads and on the writer thread, where it loads the faces too. Only the
    # request's own read may trigger the write, so remember the thread that runs
    # its `Picture.find` and fire on that thread alone.
    def _tracking_find(cls, session, *args, **kwargs):
        if armed.is_set() and picture_id in (kwargs.get("id") or ()):
            readers.append(threading.current_thread())
        return find(cls, session, *args, **kwargs)

    monkeypatch.setattr(Picture, "find", classmethod(_tracking_find))

    def _between_picture_and_faces(
        _conn, _cursor, statement, _parameters, _context, _executemany
    ):
        if not armed.is_set() or "FROM face" not in statement:
            return
        if not readers or threading.current_thread() is not readers[-1]:
            return
        armed.clear()
        writes.append(server.vault.db.submit_task(_rotation_shaped_write))
        # On the writer thread the write can only run after this read; waiting
        # there would deadlock. Off it, let the write commit before the faces load.
        if threading.current_thread() is not writer:
            writes[-1].result(timeout=30)

    event.listen(engine, "before_cursor_execute", _between_picture_and_faces)
    try:
        armed.set()
        response = client.post(f"{API}/pictures/thumbnails", json={"ids": [picture_id]})
    finally:
        armed.clear()
        event.remove(engine, "before_cursor_execute", _between_picture_and_faces)
    assert len(writes) == 1, "the thumbnail request never loaded the picture's faces"
    writes[0].result(timeout=30)
    assert response.status_code == 200, response.text

    thumbnail = response.json()[str(picture_id)]
    boxes = {face["id"]: face["bbox"] for face in thumbnail["faces"]}
    # The picture row was read before the write, so the face must be too.
    assert thumbnail["thumbnail_width"] == width * 2
    assert boxes[face_id] == [c * 2 for c in old_bbox], (
        f"geometry from before the write, face box from after it: {boxes[face_id]}"
    )


# ---------------------------------------------------------------------------
# The event loop
# ---------------------------------------------------------------------------
#
# Every request the server answers shares one asyncio event loop. An
# ``async def`` handler runs ON it, so a call it makes that waits (a writer
# task queued behind a background batch, an immediate read held off while the
# library file is swapped, an upload's disk write) stops every other request
# for as long as it waits. These routes do that kind of work, so it has to run
# on a worker thread.


def _on_event_loop() -> bool:
    """Whether the calling thread is running an asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _record_database_waits(monkeypatch, server):
    """Record every blocking database call and whether it ran on the event loop.

    Wraps ``run_task`` and ``run_immediate_read_task`` on both databases, the
    vault's and the hub's (which owns the user and its tokens), as instance
    attributes so ``monkeypatch`` undoes them. Each entry is ``(database,
    method, the work function's __qualname__, on_loop, thread name)``.
    """
    calls = []

    def _spy(label, method, real):
        def spy(func, *args, **kwargs):
            calls.append(
                (
                    label,
                    method,
                    getattr(func, "__qualname__", repr(func)),
                    _on_event_loop(),
                    threading.current_thread().name,
                )
            )
            return real(func, *args, **kwargs)

        return spy

    for label, database in (("vault", server.vault.db), ("hub", server.hub_engine)):
        for method in ("run_task", "run_immediate_read_task"):
            monkeypatch.setattr(
                database, method, _spy(label, method, getattr(database, method))
            )
    return calls


def _assert_waits_ran_off_the_loop(calls, *expected):
    """Each *expected* work function ran, and no database call ran on the loop.

    *expected* names a work function by the tail of its ``__qualname__``
    (``alter_char``, ``get_user.<locals>.<lambda>``), so it still matches once
    the function moves into a helper. It is the positive control: a request
    that never reached the database would otherwise pass.
    """
    names = [call[2] for call in calls]
    for want in expected:
        assert any(name == want or name.endswith("." + want) for name in names), (
            f"{want} never reached the database; recorded: {names}"
        )
    on_loop = [call for call in calls if call[3]]
    assert on_loop == [], f"database calls ran on the event loop: {on_loop}"


def _answered_while_held(server, client, send, entered, release):
    """Run *send* and a ``GET /version`` on ONE event loop; report whether the
    second was answered while the first was still parked.

    A ``TestClient`` opens a fresh event loop for every request, so it cannot
    show one request blocking another. This drives the app over an in-process
    ASGI transport inside a single ``asyncio.run`` instead. *send* is started,
    the loop waits until it has reached the held resource (*entered*), then
    ``/version`` is requested. If *send* parked the loop itself, ``/version``
    is only answered once *release* has fired. Returns ``(answered while still
    held, the /version response, send's response)``.
    """

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.api),
            base_url="http://testserver",
            cookies={"session_id": client.cookies.get("session_id")},
        ) as loop_client:
            pending = asyncio.ensure_future(send(loop_client))
            while not entered.is_set() and not pending.done():
                await asyncio.sleep(0.01)
            version = await loop_client.get("/version")
            answered_while_held = not release.is_set()
            release.set()
            return answered_while_held, version, await pending

    return asyncio.run(scenario())


def test_a_picture_edit_waiting_on_the_writer_does_not_stall_other_requests(
    monkeypatch, client, server
):
    """While a picture edit waits its turn at the writer, the server still
    answers other requests: the edit waits on a worker thread, not the loop."""
    picture_id = _upload(client)
    started, release, entered = threading.Event(), threading.Event(), threading.Event()

    real_run_task = server.vault.db.run_task

    def spy_run_task(func, *args, **kwargs):
        if getattr(func, "__module__", "") == "pixlstash.routes.pictures._crud":
            entered.set()
        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", spy_run_task)

    def _hold_the_writer(session):
        started.set()
        release.wait(10)

    holder = server.vault.db.submit_task(_hold_the_writer)
    # Bounds a regression: with the edit parked on the loop nothing else runs
    # until this fires.
    safety = threading.Timer(3.0, release.set)
    safety.daemon = True
    try:
        assert started.wait(5), "the writer never picked up the holding task"
        safety.start()
        answered_while_held, version, patched = _answered_while_held(
            server,
            client,
            lambda loop_client: loop_client.patch(
                f"{API}/pictures/{picture_id}", json={"description": "loop check"}
            ),
            entered,
            release,
        )
    finally:
        release.set()
        safety.cancel()
        holder.result(10)

    assert entered.is_set(), "the edit never reached the writer"
    assert patched.status_code == 200, patched.text
    assert version.status_code == 200, version.text
    assert answered_while_held, (
        "GET /version was only answered once the writer was released: the "
        "picture edit waited for it on the event loop"
    )


def test_score_edits_a_moment_apart_end_at_the_later_one(client, server):
    """Two score edits of one picture a moment apart, both waiting on a busy
    writer, leave the later score stored: an edit compares and writes in the
    order it was sent, even while it waits on a worker thread."""
    picture_id = _upload(client)
    start = client.patch(f"{API}/pictures/{picture_id}", json={"score": 3})
    assert start.status_code == 200, start.text

    started = threading.Event()

    def _keep_the_writer_busy(session):
        started.set()
        time.sleep(0.1)

    async def edit_twice():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.api),
            base_url="http://testserver",
            cookies={"session_id": client.cookies.get("session_id")},
        ) as loop_client:
            first = asyncio.ensure_future(
                loop_client.patch(f"{API}/pictures/{picture_id}", json={"score": 5})
            )
            await asyncio.sleep(0.01)
            second = await loop_client.patch(
                f"{API}/pictures/{picture_id}", json={"score": 3}
            )
            return await first, second

    busy = server.vault.db.submit_task(_keep_the_writer_busy, priority=DBPriority.LOW)
    try:
        assert started.wait(5), "the writer never picked up the busy task"
        first, second = asyncio.run(edit_twice())
    finally:
        busy.result(10)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    stored = server.vault.db.run_immediate_read_task(
        lambda session: session.get(Picture, picture_id).score
    )
    assert stored == 3, f"the later edit set 3, but {stored} is stored"


def test_a_thumbnail_waiting_on_a_held_engine_does_not_stall_other_requests(
    monkeypatch, client, server
):
    """While the database file is held (a library restore swaps it under this
    lock), a thumbnail's lookup waits on a worker thread, not the loop."""
    picture_id = _upload(client)
    held, release, entered = threading.Event(), threading.Event(), threading.Event()

    real_read = server.vault.db.run_immediate_read_task

    def spy_read(func, *args, **kwargs):
        if getattr(func, "__qualname__", "").endswith(
            "get_thumbnail.<locals>.fetch_picture"
        ):
            entered.set()
        return real_read(func, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_immediate_read_task", spy_read)

    def _hold_the_engine():
        with server.vault.db.exclusive_engine_access():
            held.set()
            release.wait(10)

    holder = threading.Thread(target=_hold_the_engine, name="test-engine-holder")
    safety = threading.Timer(3.0, release.set)
    safety.daemon = True
    holder.start()
    try:
        assert held.wait(5), "the engine lock was never taken"
        safety.start()
        answered_while_held, version, thumbnail = _answered_while_held(
            server,
            client,
            lambda loop_client: loop_client.get(
                f"{API}/pictures/thumbnails/{picture_id}.webp"
            ),
            entered,
            release,
        )
    finally:
        release.set()
        safety.cancel()
        holder.join(10)

    assert entered.is_set(), "the thumbnail never looked its picture up"
    assert thumbnail.status_code == 200, thumbnail.text
    assert version.status_code == 200, version.text
    assert answered_while_held, (
        "GET /version was only answered once the engine was released: the "
        "thumbnail's lookup waited for it on the event loop"
    )


def test_a_picture_edit_waits_on_the_database_off_the_loop(monkeypatch, client, server):
    picture_id = _upload(client)
    calls = _record_database_waits(monkeypatch, server)

    resp = client.patch(
        f"{API}/pictures/{picture_id}",
        json={"description": "off the loop", "tags": ["loop-check"]},
    )

    assert resp.status_code == 200, resp.text
    _assert_waits_ran_off_the_loop(calls, "run_recorded_metadata_task.<locals>._task")
    # The edit itself landed, through the recorded path.
    assert "loop-check" in _tags(server, picture_id)
    assert sorted(op["op_type"] for op in _operations(server)) == [
        "pictures.fields",
        "pictures.tags.replace",
    ]


def test_a_character_edit_waits_on_the_database_off_the_loop(
    monkeypatch, client, server
):
    _counter[0] += 1
    created = client.post(f"{API}/characters", json={"name": f"loop-{_counter[0]}"})
    assert created.status_code == 200, created.text
    character_id = created.json()["character"]["id"]
    calls = _record_database_waits(monkeypatch, server)

    resp = client.patch(
        f"{API}/characters/{character_id}", json={"description": "off the loop"}
    )

    assert resp.status_code == 200, resp.text
    _assert_waits_ran_off_the_loop(calls, "alter_char")


def test_a_settings_change_waits_on_both_databases_off_the_loop(
    monkeypatch, client, server
):
    calls = _record_database_waits(monkeypatch, server)

    resp = client.patch(f"{API}/users/me/config", json={"theme_mode": "dark"})

    assert resp.status_code == 200, resp.text
    _assert_waits_ran_off_the_loop(calls, "preview_patch", "update_user")


def test_settings_changes_a_moment_apart_all_end_at_the_later_one(client, server):
    """Two PATCHes of one setting a moment apart leave the saved row, both
    owner caches and the running value at the later one: a PATCH applies its
    value in the order it saved it, even while it waits on a worker thread."""

    def saved_cap(session):
        return session.exec(
            select(User.max_vram_gb).where(User.username == "testuser")
        ).one()

    original = server.hub_engine.run_immediate_read_task(saved_cap)
    start = client.patch(f"{API}/users/me/config", json={"max_vram_gb": 4})
    assert start.status_code == 200, start.text

    real_run_task = server.hub_engine.run_task
    real_set_cap = server.vault.set_max_vram_usage_gb
    first_save = threading.Lock()
    first_saved, later_applied = threading.Event(), threading.Event()

    def held_run_task(func, *args, **kwargs):
        result = real_run_task(func, *args, **kwargs)
        if getattr(func, "__name__", "") == "update_user" and (
            first_save.acquire(blocking=False)
        ):
            first_saved.set()
            # Parks the first PATCH between saving its value and applying it
            # until the later one has applied its own, so a PATCH that applies
            # out of order applies the earlier value last every time. Bounded,
            # because one that applies in order keeps the later PATCH waiting.
            later_applied.wait(0.5)
        return result

    def spy_set_cap(max_vram_gb):
        real_set_cap(max_vram_gb)
        if max_vram_gb == 8:
            later_applied.set()

    async def patch_twice():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.api),
            base_url="http://testserver",
            cookies={"session_id": client.cookies.get("session_id")},
        ) as loop_client:
            first = asyncio.ensure_future(
                loop_client.patch(f"{API}/users/me/config", json={"max_vram_gb": 2})
            )
            # The later PATCH is sent once the first has saved, so it never
            # overtakes it. Bounded, in case the first never gets that far.
            for _ in range(500):
                if first_saved.is_set() or first.done():
                    break
                await asyncio.sleep(0.01)
            second = await loop_client.patch(
                f"{API}/users/me/config", json={"max_vram_gb": 8}
            )
            return await first, second

    try:
        server.hub_engine.run_task = held_run_task
        server.vault.set_max_vram_usage_gb = spy_set_cap
        try:
            first, second = asyncio.run(patch_twice())
        finally:
            del server.hub_engine.run_task
            del server.vault.set_max_vram_usage_gb

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first_saved.is_set(), "neither PATCH reached update_user"
        assert later_applied.is_set(), "the later PATCH never applied its value"
        state = {
            "saved": server.hub_engine.run_immediate_read_task(saved_cap),
            "runtime": server.vault._max_vram_gb,
            "auth.user": server.auth.user.max_vram_gb,
            "server._user": server._user.max_vram_gb,
        }
        assert state == dict.fromkeys(state, 8.0), state
    finally:
        restore = client.patch(f"{API}/users/me/config", json={"max_vram_gb": original})
        assert restore.status_code == 200, restore.text


def test_a_watermark_upload_is_decoded_and_saved_off_the_loop(
    monkeypatch, client, server
):
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), color=(200, 30, 30, 128)).save(buf, format="PNG")
    calls = _record_database_waits(monkeypatch, server)
    try:
        resp = client.post(
            f"{API}/users/me/watermark",
            files={"file": ("mark.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        _assert_waits_ran_off_the_loop(calls, "_save")
    finally:
        client.delete(f"{API}/users/me/watermark")


def test_a_project_attachment_upload_waits_on_the_database_off_the_loop(
    monkeypatch, client, server
):
    _counter[0] += 1
    project = client.post(f"{API}/projects", json={"name": f"loop-{_counter[0]}"})
    assert project.status_code == 200, project.text
    project_id = project.json()["id"]
    calls = _record_database_waits(monkeypatch, server)

    resp = client.post(
        f"{API}/projects/{project_id}/attachments",
        files={"file": ("notes.txt", b"test-attachment", "text/plain")},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["original_filename"] == "notes.txt"
    _assert_waits_ran_off_the_loop(calls, "check_project", "insert_record")


def test_an_import_checks_its_project_off_the_loop(monkeypatch, client, server):
    _counter[0] += 1
    n = _counter[0]
    project = client.post(f"{API}/projects", json={"name": f"loop-import-{n}"})
    assert project.status_code == 200, project.text
    buf = io.BytesIO()
    Image.new("RGB", (16 + n, 16 + n), color=(n * 7 % 256, 90, 40)).save(
        buf, format="PNG"
    )
    calls = _record_database_waits(monkeypatch, server)

    status = upload_pictures_and_wait(
        client,
        [("file", (f"loop{n}.png", buf.getvalue(), "image/png"))],
        form_data={"project_id": str(project.json()["id"])},
    )

    assert status["status"] == "completed", status
    _assert_waits_ran_off_the_loop(
        calls, "_validate_association_targets.<locals>._check"
    )


def test_a_plugin_run_resolves_its_token_scope_off_the_loop(
    monkeypatch, client, server
):
    """The scope lookup reads the vault for a set, character or project token."""
    picture_id = _upload(client)
    scope_calls = []
    real_scope = _misc.fetch_scope_allowed_picture_ids

    def spy_scope(*args, **kwargs):
        scope_calls.append(_on_event_loop())
        return real_scope(*args, **kwargs)

    monkeypatch.setattr(_misc, "fetch_scope_allowed_picture_ids", spy_scope)

    resp = client.post(
        f"{API}/pictures/plugins/test-no-such-plugin",
        json={"picture_ids": [picture_id]},
    )

    assert resp.status_code == 404, resp.text
    assert scope_calls == [False], "the scope lookup ran on the event loop"


def test_a_bearer_token_is_looked_up_off_the_loop(monkeypatch, client, server):
    token = mint_read_token(client, "test-loop bearer")
    calls = _record_database_waits(monkeypatch, server)

    resp = bearer_client(server, token).get(f"{API}/sort_mechanisms")

    assert resp.status_code == 200, resp.text
    # A first use misses the token cache, so it reads the candidates.
    _assert_waits_ran_off_the_loop(
        calls, "fetch_candidates", "get_user.<locals>.<lambda>"
    )


def test_a_query_string_token_is_looked_up_off_the_loop(monkeypatch, client, server):
    token = mint_read_token(client, "test-loop query")
    calls = _record_database_waits(monkeypatch, server)

    resp = TestClient(server.api).get(f"{API}/sort_mechanisms", params={"token": token})

    assert resp.status_code == 200, resp.text
    _assert_waits_ran_off_the_loop(
        calls, "fetch_candidates", "get_user.<locals>.<lambda>"
    )


def test_binding_a_pending_link_token_writes_off_the_loop(monkeypatch, client, server):
    """The link check's request binds the pending token: a hub write."""
    r = client.post(
        f"{API}/users/me/token",
        json={"description": "test-loop pending", "scope": "ALL"},
    )
    assert r.status_code == 200, r.text
    token, token_id = r.json()["token"], r.json()["token_id"]

    def make_pending(session):
        session.exec(
            update(UserToken)
            .where(UserToken.id == token_id)
            .values(bound_address=TOKEN_BIND_PENDING)
        )
        session.commit()

    server.auth._db.run_task(make_pending, priority=DBPriority.IMMEDIATE)
    server.auth._flush_token_cache()
    try:
        calls = _record_database_waits(monkeypatch, server)

        resp = bearer_client(server, token).get(f"{API}/sort_mechanisms")

        assert resp.status_code == 200, resp.text
        _assert_waits_ran_off_the_loop(calls, "bind")
    finally:
        client.delete(f"{API}/users/me/token/{token_id}")


def test_a_guest_score_and_its_cookie_are_resolved_off_the_loop(
    monkeypatch, client, server
):
    picture_id = _upload(client)
    _counter[0] += 1
    n = _counter[0]
    created = client.post(f"{API}/picture_sets", json={"name": f"loop-guest-{n}"})
    assert created.status_code == 200, created.text
    set_id = created.json()["picture_set"]["id"]
    assert (
        client.post(f"{API}/picture_sets/{set_id}/members/{picture_id}").status_code
        == 200
    )
    token = mint_read_token(
        client, "test-loop guest", resource_type="picture_set", resource_id=set_id
    )
    guest = bearer_client(server, token)
    calls = _record_database_waits(monkeypatch, server)

    resp = guest.post(
        f"{API}/pictures/guest-scores",
        json={
            "session_id": f"test-session-{n}",
            "set_cookie": True,
            "scores": {str(picture_id): 3},
        },
    )

    assert resp.status_code == 200, resp.text
    _assert_waits_ran_off_the_loop(calls, "fetch_candidates", "_fetch_set", "_run")

    # The cookie it set is resolved by the middleware on the next request.
    calls.clear()
    scores = guest.get(f"{API}/pictures/guest-scores")

    assert scores.status_code == 200, scores.text
    assert scores.json() == {"scores": {str(picture_id): 3}}
    _assert_waits_ran_off_the_loop(calls, "_lookup_by_token")
