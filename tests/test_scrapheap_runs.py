"""Background Scrapheap runs: emptying, deleting and restoring in slices.

``POST /pictures/scrapheap/runs`` records a run and answers 202; a background
task then works the selection a writer slice at a time through the existing
purge or restore path. The synchronous ``DELETE /pictures/scrapheap`` held one
IMMEDIATE writer transaction for the whole selection, which on a 50,000-picture
Scrapheap outlasted the client's 60-second timeout and left the outcome
unknown. These tests cover:

* the request: 202 with ``Location`` and ``Retry-After``, the selection fixed
  when the run starts, the delete-forever consent token, one run at a time, and
  503 when there is no task runner to take the run;
* the delete preview and the Scrapheap listing answering while a long write
  holds the DB writer (they read off the writer queue);
* who may reach a run: the owner only, by session or an ``ALL`` token. No
  credential, a READ token, a share link's token of every resource kind and
  the ComfyUI key are refused on every run route, by the gate and, with the
  gate rolled back, by the auth middleware's rules;
* the slices: their size, their LOW writer priority, and what each reports;
* cancel before the run starts and between slices, the server stopping before
  and during a run, and a slice that fails;
* restore History: one segment per run, a new one when another undoable edit
  lands between slices, the segment cap, and Ctrl+Z undoing one segment;
* the slice size: sized by the writer time the last slice took, within the
  bounds the run shows, and a large run whose slices and segments stay bounded
  and whose per-slice SQL does not grow with the run;
* the coalesced ``pictures_changed`` events, carrying the requesting tab;
* the synchronous routes sending the whole Scrapheap, or more than 1000 ids,
  to a run, without spending the token;
* ``request_id``: a replayed start answers the run it started, a reused id
  with other parameters is refused, and a replay spends no token;
* undoing a restore run: its History segments newest first, at LOW priority,
  stopping on a later edit or a locked set, and refused when there is nothing
  to undo;
* every run event naming its run, and the broadcaster passing that on;
* one run worked end to end by a real task runner.

One server serves the module, with background workers off; a test captures the
task the route submits and runs it on the test thread, which acts on the run
between slices through the same HTTP routes a second tab would use. Every test
starts with no runs, an empty History and an empty Scrapheap, because a run
with no ``picture_ids`` takes the whole Scrapheap.
"""

import asyncio
import gc
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import delete, select, update

import pixlstash.database as database
from pixlstash import auth
from pixlstash.database import DBPriority
from pixlstash.db_models import (
    DedupAutoStackRun,
    Operation,
    Picture,
    PictureSet,
    PictureSetMember,
    ScrapheapRunRecord,
)
from pixlstash.event_types import EventType
from pixlstash.routes.pictures._crud import (
    SCRAPHEAP_MAX_IDS,
    SCRAPHEAP_SYNC_MAX_IDS,
    _preferred_wait,
)
from pixlstash.server import Server
from pixlstash.services import (
    operation_log_service,
    scrapheap_run_service,
    scrapheap_service,
)
from pixlstash.services.scrapheap_run_service import ScrapheapRunRegistry
from pixlstash.task_runner import TaskRunner
from pixlstash.tasks.scrapheap_run_task import ScrapheapRunTask, _SliceSizer
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.ws.broadcaster import WsBroadcasterMixin
from tests.authz_guard import assert_real_route, no_spa_fallback  # noqa: F401
from tests.utils import bearer_client

API = "/api/v1"
RUNS_URL = f"{API}/pictures/scrapheap/runs"
LATEST_URL = f"{RUNS_URL}/latest"
RUN_TEMPLATE = f"{RUNS_URL}/{{run_id}}"
PREVIEW_URL = f"{API}/pictures/scrapheap/delete-preview"
RESTORE_URL = f"{API}/pictures/scrapheap/restore"
UNDO_URL = f"{API}/operations/undo"
REDO_URL = f"{API}/operations/redo"
TOKEN_URL = f"{API}/users/me/token"
OWNER = {"username": "owner", "password": "example-owner-password"}
# What each layer answers a READ token with, so a refusal names its layer.
GATE_REFUSAL = "Owner-level (full, unscoped) access required"
MIDDLEWARE_REFUSAL = "Token is read-only"
# A share link names one picture, picture set, character or project; every
# kind is refused on every run route, so each is minted and tried.
SHARE_RESOURCES = {
    "share": "picture",
    "set_share": "picture_set",
    "character_share": "character",
    "project_share": "project",
}

# The SPA catch-all answers unmatched GETs with 200; see tests/authz_guard.py.
pytestmark = pytest.mark.usefixtures("no_spa_fallback")


@pytest.fixture(scope="module")
def env():
    """One owner-logged-in client and its server, for the whole module."""
    temp_dir = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    force_cpu = Server.DEFAULT_FORCE_CPU
    Server.DEFAULT_FORCE_CPU = True
    server = Server(config_path)
    try:
        client = TestClient(server.api)
        assert client.post(f"{API}/login", json=OWNER).status_code == 200
        yield client, server
    finally:
        server.close()
        Server.DEFAULT_FORCE_CPU = force_cpu
        temp_dir.cleanup()
        gc.collect()


@pytest.fixture(autouse=True)
def _nothing_left_over(env):
    """Start every test with no runs, an empty History and an empty Scrapheap.

    Only one run may be active at once, so a run an earlier test left behind
    would refuse this test's. A run with no ``picture_ids`` takes the whole
    Scrapheap, so earlier pictures left there would join this test's run. And a
    restore run's History segments are judged against the newest undoable
    operation, so an earlier test's operations would decide them. Clearing the
    ``deleted`` flag is the whole Scrapheap reset: every run query is
    ``Picture.deleted is True`` and nothing else.
    """
    _client, server = env
    server.vault.scrapheap_runs = ScrapheapRunRegistry()

    def reset(session):
        session.exec(delete(Operation))
        session.exec(delete(ScrapheapRunRecord))
        session.exec(
            update(Picture)
            .where(Picture.deleted.is_(True))
            .values(deleted=False, deleted_at=None)
        )
        session.commit()

    _run(server, reset)
    assert scrapheap_run_service.latest_run(server.vault) is None
    assert _run(server, _heap_ids_in_session) == []
    assert _run(server, lambda s: s.exec(select(Operation.id)).all()) == []
    yield


@pytest.fixture
def captured(env, monkeypatch):
    """The tasks the routes submit, held for the test to run itself.

    Background workers are off in this module, so the vault reports no task
    runner and the route answers 503; capturing the task instead is the task
    runner taking it, minus the thread.
    """
    _client, server = env
    return _capture(server, monkeypatch)


def _capture(server, monkeypatch) -> list:
    """Stand in for a task runner from here on; see :func:`captured`."""
    tasks: list[ScrapheapRunTask] = []

    def submit(task):
        tasks.append(task)
        return task.id

    monkeypatch.setattr(server.vault, "runs_background_work", lambda: True)
    monkeypatch.setattr(server.vault, "submit_task", submit)
    return tasks


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _token_touched(caplog) -> bool:
    """Whether the confirm-token store accepted or refunded a token."""
    return any(
        record.getMessage().startswith(
            ("Delete-forever: confirmation accepted", "Delete-forever: refunded")
        )
        for record in caplog.records
    )


def _heap_ids_in_session(session) -> list[int]:
    return list(
        session.exec(
            select(Picture.id).where(Picture.deleted.is_(True)).order_by(Picture.id)
        ).all()
    )


def _heap(server, count: int) -> list[int]:
    """Put *count* new pictures straight into the Scrapheap, each with a file."""
    tag = uuid.uuid4().hex[:8]
    folder = os.path.join(server.vault.image_root, tag)
    os.makedirs(folder)
    names = [f"{index}.png" for index in range(count)]
    for name in names:
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(b"example-pixels")
    deleted_at = datetime.now(timezone.utc)

    def insert(session):
        ids = []
        for index, name in enumerate(names):
            picture = Picture(
                file_path=f"{tag}/{name}",
                format="png",
                width=8,
                height=8,
                size_bytes=14,
                pixel_sha=f"{tag}-{index}",
                deleted=True,
                deleted_at=deleted_at,
            )
            session.add(picture)
            session.flush()
            ids.append(int(picture.id))
        session.commit()
        return ids

    return _run(server, insert)


def _states(server, ids: list[int]) -> list:
    """Per id, in order: ``"heap"``, ``"live"``, or ``None`` once destroyed."""

    def read(session):
        rows = dict(
            session.exec(
                select(Picture.id, Picture.deleted).where(Picture.id.in_(ids))
            ).all()
        )
        return [
            None if pid not in rows else ("heap" if rows[pid] else "live")
            for pid in ids
        ]

    return _run(server, read)


def _files(server, ids: list[int]) -> list[str]:
    def read(session):
        rows = dict(
            session.exec(
                select(Picture.id, Picture.file_path).where(Picture.id.in_(ids))
            ).all()
        )
        return [
            ImageUtils.resolve_picture_path(server.vault.image_root, rows[pid])
            for pid in ids
        ]

    return _run(server, read)


def _lock(server, picture_id: int) -> None:
    def create(session):
        picture_set = PictureSet(name=f"frozen-{uuid.uuid4().hex[:6]}", locked=True)
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()

    _run(server, create)


def _token(client, ids) -> str:
    """The confirm_token the delete preview mints for exactly this selection."""
    preview = client.post(PREVIEW_URL, json={"ids": ids} if ids else None)
    assert preview.status_code == 200, preview.text
    return preview.json()["confirm_token"]


def _purge(client, ids=None, **body) -> dict:
    payload = {"kind": "purge", "confirm_token": _token(client, ids), **body}
    if ids:
        payload["picture_ids"] = ids
    response = client.post(RUNS_URL, json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def _purge_with(client, token: str, ids=None) -> dict:
    """Start a purge with a token minted earlier, as a dialog left open would."""
    payload = {"kind": "purge", "confirm_token": token}
    if ids:
        payload["picture_ids"] = ids
    response = client.post(RUNS_URL, json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def _restore(client, ids=None, **headers) -> dict:
    payload = {"kind": "restore"}
    if ids:
        payload["picture_ids"] = ids
    response = client.post(RUNS_URL, json=payload, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


def _varies_on_prefer(response) -> bool:
    """``Vary`` names ``Prefer``, among whatever else the middleware adds."""
    listed = response.headers.get("Vary", "").split(",")
    return "prefer" in {name.strip().lower() for name in listed}


def _get(client, run_id: str) -> dict:
    response = client.get(f"{RUNS_URL}/{run_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _work(tasks: list) -> None:
    """Run every captured task, in order, as the task runner would."""
    while tasks:
        tasks.pop(0).run()


def _between_slices(monkeypatch, run_id: str, callback) -> None:
    """Call ``callback(task, processed)`` after each slice of run *run_id*.

    ``_set_progress`` is the task's own per-slice readout, so this runs on the
    test thread exactly where the writer is free between two slices: an HTTP
    call from here is what a second tab could do at that moment.
    """
    original = ScrapheapRunTask._set_progress

    def wrapped(self, processed, total):
        original(self, processed, total)
        if processed > 0 and self.params["run_id"] == run_id:
            callback(self, processed)

    monkeypatch.setattr(ScrapheapRunTask, "_set_progress", wrapped)


def _announced(monkeypatch, server) -> list[dict]:
    """The ``pictures_changed`` payloads the vault sends from now on."""
    events = []
    original = server.vault.notify

    def notify(event_type, data=None, *args, **kwargs):
        if event_type == EventType.CHANGED_PICTURES:
            events.append(data)
        return original(event_type, data, *args, **kwargs)

    monkeypatch.setattr(server.vault, "notify", notify)
    return events


def _segment_of(server, batch_id: str) -> list[int]:
    """Every picture the operations of History batch *batch_id* name."""

    def read(session):
        rows = session.exec(
            select(Operation.target_ids).where(Operation.batch_id == batch_id)
        ).all()
        return sorted(
            int(pid)
            for target_ids in rows
            for pid in (
                json.loads(target_ids) if isinstance(target_ids, str) else target_ids
            )
        )

    return _run(server, read)


def _shared_resource_id(client, resource_type: str, picture_id: int) -> int:
    """The id a share link of *resource_type* names, made afresh if need be."""
    if resource_type == "picture":
        return picture_id
    route, key = {
        "picture_set": ("picture_sets", "picture_set"),
        "character": ("characters", "character"),
        "project": ("projects", None),
    }[resource_type]
    made = client.post(
        f"{API}/{route}", json={"name": f"example-{uuid.uuid4().hex[:12]}"}
    )
    assert made.status_code in (200, 201), made.text
    return (made.json()[key] if key else made.json())["id"]


def _credential(client, server, kind: str, picture_id: int) -> str:
    """A token of *kind*, minted afresh by the owner for one test.

    A ``*share`` kind is a share link's token, limited to *picture_id* or to a
    new picture set, character or project (``SHARE_RESOURCES``). Each is shown
    to work where it is entitled to, so a refusal it later meets is the route
    refusing it and not a dead credential.
    """
    if kind in SHARE_RESOURCES:
        resource_type = SHARE_RESOURCES[kind]
        body = {
            "scope": "READ",
            "resource_type": resource_type,
            "resource_id": _shared_resource_id(client, resource_type, picture_id),
        }
    else:
        body = {"read": {"scope": "READ"}, "comfyui": {"scope": "COMFYUI"}}[kind]
    minted = client.post(TOKEN_URL, json={"description": f"test-{kind}", **body})
    assert minted.status_code == 200, minted.text
    token = minted.json()["token"]
    alive = bearer_client(server, token).get(f"{API}/pictures")
    assert alive.status_code == 200, f"the {kind} token is dead: {alive.text}"
    return token


def _a_pending_run(client, server) -> SimpleNamespace:
    """A pending restore run, a purge's confirm token, and every run request.

    The restore holds one picture and stays pending, because the test keeps
    its task. The purge request carries a live confirm token for two more
    pictures, so a request that got past a refusal would spend it.
    """
    held, doomed = _heap(server, 1), _heap(server, 2)
    run_id = _restore(client, held)["id"]
    confirm_token = _token(client, doomed)
    run_url = f"{RUNS_URL}/{run_id}"
    purge = {"kind": "purge", "confirm_token": confirm_token, "picture_ids": doomed}
    requests = [
        ("POST", RUNS_URL, RUNS_URL, {"kind": "restore", "picture_ids": held}),
        ("POST", RUNS_URL, RUNS_URL, purge),
        ("GET", LATEST_URL, LATEST_URL, None),
        ("GET", run_url, RUN_TEMPLATE, None),
        ("POST", f"{run_url}/cancel", f"{RUN_TEMPLATE}/cancel", None),
        ("POST", f"{run_url}/undo", f"{RUN_TEMPLATE}/undo", None),
    ]
    for method, path, template, _body in requests:
        assert_real_route(server.api, method, path, template)
    return SimpleNamespace(held=held, run_id=run_id, purge=purge, requests=requests)


def _answers(session, requests, **transport) -> list[tuple]:
    """Each request's method, path, status code and ``detail``, in order."""
    answers = []
    for method, path, _template, body in requests:
        response = session.request(method, path, json=body, **transport)
        detail = response.json().get("detail")
        answers.append((method, path, response.status_code, detail))
    return answers


def _the_owner_still_reaches_every_run_route(client, setup) -> None:
    """The refused requests changed nothing, and the owner reaches all five.

    A test calls this before it reads the refusals' status codes, so a request
    that got through fails on what it did, not only on what it answered. The
    owner's answers also prove each refused path is a live route: the auth
    middleware refuses a READ token before routing, whether or not anything
    serves the path.
    """
    latest = client.get(LATEST_URL)
    assert latest.status_code == 200, latest.text
    assert (latest.json()["run"]["id"], latest.json()["run"]["status"]) == (
        setup.run_id,
        "pending",
    )
    assert _get(client, setup.run_id)["status"] == "pending"
    # A run is still active, so the owner's undo is refused for that, and only
    # for that: the route resolved and the gate let the owner through.
    undo = client.post(f"{RUNS_URL}/{setup.run_id}/undo")
    assert undo.status_code == 409, undo.text
    assert undo.json()["detail"]["code"] == "scrapheap_run_active"
    cancelled = client.post(f"{RUNS_URL}/{setup.run_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    # Nothing refused spent the confirm token: it still starts the purge.
    purge = client.post(RUNS_URL, json=setup.purge)
    assert purge.status_code == 202, purge.text


# ── the request ──────────────────────────────────────────────────────────────


def test_a_purge_run_answers_at_once_and_empties_the_selection_in_low_slices(
    env, captured, monkeypatch
):
    """The request records the run and returns; the slices do the work.

    Each slice goes through the one destruction path at LOW writer priority, so
    every interactive write gets the writer between two of them, and removes
    its files inline: the run is already off the request.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 3)
    ids = _heap(server, 7)
    files = _files(server, ids)
    slices = []
    original = scrapheap_service.purge_scrapheap_pictures

    def recording(vault, chunk, include_protected, **kwargs):
        slices.append((list(chunk), include_protected, kwargs))
        return original(vault, chunk, include_protected, **kwargs)

    monkeypatch.setattr(scrapheap_service, "purge_scrapheap_pictures", recording)

    token = _token(client, ids)
    response = client.post(
        RUNS_URL, json={"kind": "purge", "picture_ids": ids, "confirm_token": token}
    )
    assert response.status_code == 202, response.text
    run = response.json()
    assert response.headers["Location"] == f"{RUNS_URL}/{run['id']}"
    assert response.headers["Retry-After"] == "1"
    assert (run["kind"], run["status"], run["active"]) == ("purge", "pending", True)
    assert (run["total"], run["processed"]) == (7, 0)
    # The request itself destroyed nothing.
    assert _states(server, ids) == ["heap"] * 7
    assert all(os.path.isfile(path) for path in files)

    _work(captured)

    # Every slice carries the confirmation's own bound, the moment its preview
    # was read, so the run can never reach a picture binned after it.
    bound = slices[0][2]["deleted_before"]
    assert bound.tzinfo is not None
    assert slices == [
        (ids[0:3], False, {"priority": DBPriority.LOW, "deleted_before": bound}),
        (ids[3:6], False, {"priority": DBPriority.LOW, "deleted_before": bound}),
        (ids[6:7], False, {"priority": DBPriority.LOW, "deleted_before": bound}),
    ]
    ended = client.get(f"{RUNS_URL}/{run['id']}")
    assert ended.status_code == 200, ended.text
    assert "Retry-After" not in ended.headers
    body = ended.json()
    assert (body["status"], body["active"]) == ("complete", False)
    assert (body["processed"], body["deleted_count"]) == (7, 7)
    assert body["skipped_locked"] == []
    assert body["snapshots_with_deleted"] == []
    assert body["finished_at"] is not None
    assert _states(server, ids) == [None] * 7
    assert not any(os.path.exists(path) for path in files)


def test_the_whole_scrapheap_is_the_one_there_when_the_run_starts(env, captured):
    """A picture scrapheaped after the confirmation is never destroyed by it.

    The owner confirmed the Scrapheap the preview showed. Something landing in
    it afterwards was not in that dialog, so the run fixes its selection when it
    starts rather than re-reading the Scrapheap per slice.
    """
    client, server = env
    ids = _heap(server, 2)
    run = _purge(client)
    assert run["total"] == 2
    late = _heap(server, 1)

    _work(captured)

    assert _get(client, run["id"])["deleted_count"] == 2
    assert _states(server, ids) == [None, None]
    assert _states(server, late) == ["heap"]


def test_a_purge_run_needs_the_previews_token_for_that_selection(env, captured):
    """The run destroys what DELETE would, so it takes the same consent."""
    client, server = env
    ids = _heap(server, 2)

    missing = client.post(RUNS_URL, json={"kind": "purge", "picture_ids": ids})
    assert missing.status_code == 400, missing.text
    assert "confirm_token is required" in missing.json()["detail"]

    other_selection = client.post(
        RUNS_URL,
        json={
            "kind": "purge",
            "picture_ids": ids,
            "confirm_token": _token(client, ids[:1]),
        },
    )
    assert other_selection.status_code == 409, other_selection.text

    whole_heap_token = _token(client, None)
    assert captured == []
    assert client.get(LATEST_URL).json() == {"run": None}

    first = client.post(
        RUNS_URL, json={"kind": "purge", "confirm_token": whole_heap_token}
    )
    assert first.status_code == 202, first.text
    _work(captured)
    spent = client.post(
        RUNS_URL, json={"kind": "purge", "confirm_token": whole_heap_token}
    )
    assert spent.status_code == 409, spent.text
    assert captured == []


@contextmanager
def _writer_held(server):
    """Hold the DB writer with a callback blocked on an event for the block.

    What a long purge or restore transaction does to every request that queues
    on the writer: nothing else gets it until the callback returns.
    """
    started = threading.Event()
    release = threading.Event()

    def hold(_session):
        started.set()
        release.wait(30)

    holder = threading.Thread(
        target=server.vault.db.run_task,
        args=(hold,),
        kwargs={"priority": DBPriority.IMMEDIATE},
        daemon=True,
    )
    holder.start()
    assert started.wait(5), "the holding callback never got the writer"
    try:
        yield
    finally:
        release.set()
        holder.join(10)
        assert not holder.is_alive(), "the holding callback did not let go"


def _answer_within(seconds: float, call):
    """Run *call* on a thread; return its result, or fail if it is still waiting."""
    result = {}
    caller = threading.Thread(target=lambda: result.update(value=call()), daemon=True)
    caller.start()
    caller.join(seconds)
    assert not caller.is_alive(), f"no answer within {seconds}s"
    return result["value"]


def test_the_delete_preview_answers_while_the_writer_is_held(env):
    """The preview reads off the writer, so a long write cannot stall it.

    It used to queue its two reads on the writer at IMMEDIATE, so a preview
    asked for during a 60-second purge waited the whole purge. Nothing relied
    on the serialisation: the purge re-reads everything in its own writer task.
    """
    client, server = env
    ids = _heap(server, 3)

    with _writer_held(server):
        response = _answer_within(
            5, lambda: client.post(PREVIEW_URL, json={"ids": ids})
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["total_count"], body["unprotected_count"]) == (3, 3)
    assert body["confirm_token"]


def test_the_scrapheap_listing_answers_while_the_writer_is_held(env):
    client, server = env
    ids = _heap(server, 2)

    with _writer_held(server):
        response = _answer_within(
            5,
            lambda: client.get(f"{API}/pictures", params={"only_deleted": "true"}),
        )

    assert response.status_code == 200, response.text
    assert {row["id"] for row in response.json()} >= set(ids)


def test_a_restore_run_needs_no_token_and_ignores_include_protected(env, captured):
    client, server = env
    ids = _heap(server, 2)
    response = client.post(
        RUNS_URL,
        json={"kind": "restore", "picture_ids": ids, "include_protected": True},
    )
    assert response.status_code == 202, response.text
    assert response.json()["include_protected"] is False
    _work(captured)
    assert _states(server, ids) == ["live", "live"]


def test_a_purge_leaves_pictures_binned_after_its_preview(env, captured):
    """The dialog's count is what the run may destroy: a later arrival survives."""
    client, server = env
    ids = _heap(server, 2)
    token = _token(client, None)
    late = _heap(server, 1)

    response = client.post(RUNS_URL, json={"kind": "purge", "confirm_token": token})
    assert response.status_code == 202, response.text
    started = response.json()
    assert started["total"] == 2
    assert started["arrived_since"] is None
    _work(captured)

    ended = _get(client, started["id"])
    assert ended["status"] == "complete"
    assert ended["deleted_count"] == 2
    assert ended["arrived_since"] == 1
    assert _states(server, ids) == [None, None]
    assert _states(server, late) == ["heap"]


def test_a_picture_binned_again_after_the_preview_is_left_alone(env, captured):
    """Restored and binned again: the preview never saw this binning."""
    client, server = env
    ids = _heap(server, 2)
    outside = _heap(server, 1)
    token = _token(client, ids)
    rescued = client.post(RESTORE_URL, json={"picture_ids": ids[1:]})
    assert rescued.status_code == 200, rescued.text
    rebinned = client.delete(f"{API}/pictures/{ids[1]}")
    assert rebinned.status_code == 200, rebinned.text
    _heap(server, 1)  # binned after the preview, but outside the selection

    run = _purge_with(client, token, ids)
    _work(captured)

    ended = _get(client, run["id"])
    assert ended["deleted_count"] == 1
    assert ended["arrived_since"] == 1
    assert _states(server, ids) == [None, "heap"]
    assert _states(server, outside) == ["heap"]


def test_a_picture_binned_again_while_the_run_works_is_left_alone(
    env, captured, monkeypatch
):
    """Each slice applies the bound again: the selection fixed at start is not enough."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    ids = _heap(server, 2)
    run = _purge(client, ids)
    answers = []

    def rebin_once(_task, processed):
        if processed == 1:
            answers.append(client.post(RESTORE_URL, json={"picture_ids": ids[1:]}))
            answers.append(client.delete(f"{API}/pictures/{ids[1]}"))

    _between_slices(monkeypatch, run["id"], rebin_once)
    _work(captured)

    assert [answer.status_code for answer in answers] == [200, 200]
    ended = _get(client, run["id"])
    assert ended["status"] == "complete"
    assert ended["deleted_count"] == 1
    assert _states(server, ids) == [None, "heap"]


def test_an_undo_that_bins_a_picture_after_the_preview_leaves_it_alone(env, captured):
    """Undoing a restore bins the picture now, not at its old stamp.

    The confirmation covers what its preview read. The picture was out of the
    Scrapheap then, so spending it must not destroy the picture; written back,
    its old stamp sat under the preview's bound and the purge took it.
    """
    client, server = env
    ids = _heap(server, 2)
    rescued = client.post(RESTORE_URL, json={"picture_ids": ids[1:]})
    assert rescued.status_code == 200, rescued.text
    token = _token(client, None)
    undo = client.post(UNDO_URL)
    assert undo.status_code == 200, undo.text
    assert _states(server, ids) == ["heap", "heap"]

    run = _purge_with(client, token)
    assert run["total"] == 1
    _work(captured)

    ended = _get(client, run["id"])
    assert ended["deleted_count"] == 1
    assert ended["arrived_since"] == 1
    assert _states(server, ids) == [None, "heap"]


def test_a_redo_that_bins_a_picture_after_the_preview_leaves_it_alone(env):
    """Redoing a move to the Scrapheap is a binning too, by any delete path."""
    client, server = env
    ids = _heap(server, 1)
    moved_id = _heap(server, 1)[0]
    assert client.post(RESTORE_URL, json={"picture_ids": [moved_id]}).status_code == (
        200
    )
    assert client.delete(f"{API}/pictures/{moved_id}").status_code == 200
    assert client.post(UNDO_URL).status_code == 200
    selection = ids + [moved_id]
    token = _token(client, selection)
    redo = client.post(REDO_URL)
    assert redo.status_code == 200, redo.text
    assert redo.json()["scrapheaped_picture_ids"] == [moved_id]

    response = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"picture_ids": selection, "confirm_token": token},
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted_count"] == 1
    assert _states(server, selection) == [None, "heap"]


def test_an_undo_keeps_the_stamp_of_a_picture_already_in_the_scrapheap():
    """Only a picture entering the Scrapheap is stamped; one there keeps its clock."""
    binned_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    picture = Picture(file_path="a.png", deleted=True, deleted_at=binned_at)
    session = SimpleNamespace(add=lambda _picture: None)

    operation_log_service._apply_deleted(
        session, picture, {"deleted": True, "deleted_at": "2025-01-01T00:00:00"}
    )
    assert (picture.deleted, picture.deleted_at) == (True, binned_at)

    operation_log_service._apply_deleted(
        session, picture, {"deleted": False, "deleted_at": "2025-01-01T00:00:00"}
    )
    assert (picture.deleted, picture.deleted_at) == (False, None)


def test_a_purge_with_nothing_arrived_reports_zero(env, captured):
    client, server = env
    ids = _heap(server, 1)
    run = _purge(client, ids)
    _work(captured)
    assert _get(client, run["id"])["arrived_since"] == 0
    assert _states(server, ids) == [None]


def test_a_restore_run_reports_no_arrivals(env, captured):
    client, server = env
    run = _restore(client, _heap(server, 1))
    _work(captured)
    assert _get(client, run["id"])["arrived_since"] is None


def _sends_to_a_run(response) -> bool:
    """A 422 that points at the runs route, with the cap it applied."""
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == {
        **response.json()["detail"],
        "code": "use_scrapheap_runs",
        "runs_url": RUNS_URL,
        "max_ids": SCRAPHEAP_SYNC_MAX_IDS,
    }
    return True


def test_the_synchronous_routes_send_the_whole_heap_to_a_run(env, captured):
    """Neither sync route takes the whole Scrapheap, and a refusal costs no token.

    A whole-heap request holds the writer for as long as the heap is large,
    which is what a run exists to avoid. The token the dialog fetched is still
    good afterwards, so the client can start the run with it.
    """
    client, server = env
    ids = _heap(server, 2)
    token = _token(client, None)

    delete = client.request(
        "DELETE", f"{API}/pictures/scrapheap", json={"confirm_token": token}
    )
    restore = client.post(RESTORE_URL)

    assert _sends_to_a_run(delete) and _sends_to_a_run(restore)
    assert _states(server, ids) == ["heap", "heap"]
    _purge_with(client, token)
    _work(captured)
    assert _states(server, ids) == [None, None]


def test_a_synchronous_selection_over_its_cap_is_sent_to_a_run(env, captured):
    """Over 1000 ids is a run; exactly 1000 is still served in the request."""
    client, server = env
    assert SCRAPHEAP_SYNC_MAX_IDS == 1000  # the documented contract
    ids = _heap(server, 2)
    # Ids past the newest picture name nothing, so only *ids* can be touched.
    filler = list(range(ids[-1] + 1, ids[-1] + SCRAPHEAP_SYNC_MAX_IDS))
    over = [ids[0], ids[1], *filler]
    at_cap = [ids[0], *filler]
    assert (len(over), len(at_cap)) == (SCRAPHEAP_SYNC_MAX_IDS + 1, 1000)
    token = _token(client, over)

    delete = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"picture_ids": over, "confirm_token": token},
    )
    restore = client.post(RESTORE_URL, json={"picture_ids": over})

    assert _sends_to_a_run(delete) and _sends_to_a_run(restore)
    assert str(SCRAPHEAP_SYNC_MAX_IDS + 1) in delete.json()["detail"]["message"]
    assert _states(server, ids) == ["heap", "heap"]

    restored = client.post(RESTORE_URL, json={"picture_ids": at_cap})
    assert restored.status_code == 200, restored.text
    assert restored.json()["restored_count"] == 1
    at_cap = [ids[1], *filler]
    deleted = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"picture_ids": at_cap, "confirm_token": _token(client, at_cap)},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted_count"] == 1
    assert _states(server, ids) == ["live", None]
    # The refused DELETE did not spend its token: a run over the same
    # selection still accepts it (a spent one is a 409).
    _purge_with(client, token, over)
    _work(captured)
    assert _states(server, ids) == ["live", None]


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "empty"},
        {"kind": "restore", "picture_ids": []},
        {"kind": "restore", "unknown": 1},
    ],
)
def test_a_malformed_request_is_refused_before_anything_starts(env, captured, body):
    client, _server = env
    response = client.post(RUNS_URL, json=body)
    assert response.status_code == 422, response.text
    assert captured == []
    assert client.get(LATEST_URL).json() == {"run": None}


def test_a_selection_over_the_cap_is_refused_before_anything_starts(env, captured):
    """Every route taking an explicit Scrapheap selection refuses one over the cap.

    A body that size is what the cap is for: the server would hold the list for
    the life of the run. The whole Scrapheap is the omitted selection, which is
    never capped.
    """
    client, _server = env
    assert SCRAPHEAP_MAX_IDS == 200_000  # the documented contract
    over = list(range(1, SCRAPHEAP_MAX_IDS + 2))

    run = client.post(RUNS_URL, json={"kind": "restore", "picture_ids": over})
    preview = client.post(PREVIEW_URL, json={"ids": over})
    delete = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"picture_ids": over, "confirm_token": "test-unused"},
    )

    for response in (run, preview, delete):
        assert response.status_code == 422, response.text
    assert "200000" in json.dumps(run.json()["detail"])
    assert str(SCRAPHEAP_MAX_IDS) in preview.json()["detail"]
    assert str(SCRAPHEAP_MAX_IDS) in delete.json()["detail"]
    assert captured == []
    assert client.get(LATEST_URL).json() == {"run": None}


def test_a_selection_at_the_cap_is_accepted(env, captured):
    """The positive control for the cap: exactly the maximum still works."""
    client, _server = env
    at_cap = list(range(1, SCRAPHEAP_MAX_IDS + 1))

    preview = client.post(PREVIEW_URL, json={"ids": at_cap})
    assert preview.status_code == 200, preview.text
    run = client.post(RUNS_URL, json={"kind": "restore", "picture_ids": at_cap})
    assert run.status_code == 202, run.text
    _work(captured)
    assert _get(client, run.json()["id"])["status"] == "complete"


def test_an_ended_run_holds_no_id_lists(env, captured):
    """The selection and the purged ids serve a run only while it works.

    The registry keeps the last few ended runs for polling; if those kept their
    id lists, each could pin a selection of up to the cap. Every end path
    releases them: worked to the end, and cancelled while still pending.
    """
    client, server = env
    registry = server.vault.scrapheap_runs
    ids = _heap(server, 2)
    worked = _purge(client, ids)
    _work(captured)
    assert _get(client, worked["id"])["deleted_count"] == 2

    pending = _restore(client, _heap(server, 1))
    cancelled = client.post(f"{RUNS_URL}/{pending['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"

    for run_id in (worked["id"], pending["id"]):
        assert registry.mutate(
            run_id, lambda run: (run.picture_ids, run.purged_ids)
        ) == ([], [])
    _work(captured)


def test_only_one_run_may_be_active(env, captured):
    """A second click, or a second tab, attaches to the run already going."""
    client, server = env
    _heap(server, 2)
    first = _restore(client)

    second = client.post(RUNS_URL, json={"kind": "restore"})
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["code"] == "scrapheap_run_active"
    assert second.json()["detail"]["run"]["id"] == first["id"]
    purge = client.post(
        RUNS_URL, json={"kind": "purge", "confirm_token": _token(client, None)}
    )
    assert purge.status_code == 409, purge.text
    assert purge.json()["detail"]["code"] == "scrapheap_run_active"
    assert len(captured) == 1

    _work(captured)
    assert _get(client, first["id"])["status"] == "complete"
    _heap(server, 1)
    assert client.post(RUNS_URL, json={"kind": "restore"}).status_code == 202


def test_a_purge_refused_by_an_active_run_keeps_its_token(env, captured, caplog):
    """A 409 for another run must not cost the user the preview they just read.

    Refused before the token is touched, not spent and refunded: the store
    never accepts it.
    """
    client, server = env
    ids = _heap(server, 3)
    restoring = _restore(client, ids[:1])
    purge_body = {
        "kind": "purge",
        "picture_ids": ids[1:],
        "confirm_token": _token(client, ids[1:]),
    }

    with caplog.at_level(logging.INFO, logger=scrapheap_service.logger.name):
        refused = client.post(RUNS_URL, json=purge_body)
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "scrapheap_run_active"
    assert not _token_touched(caplog)
    _work(captured)
    assert _get(client, restoring["id"])["status"] == "complete"

    retried = client.post(RUNS_URL, json=purge_body)
    assert retried.status_code == 202, retried.text
    _work(captured)
    assert _get(client, retried.json()["id"])["status"] == "complete"
    assert _states(server, ids[1:]) == [None, None]
    spent = client.post(RUNS_URL, json=purge_body)
    assert spent.status_code == 409, spent.text
    assert "no longer valid" in spent.json()["detail"]


def test_a_purge_that_loses_the_race_for_the_run_gets_its_token_back(
    env, captured, monkeypatch
):
    """The pre-check passed, then another run won: the refusal refunds."""
    client, server = env
    ids = _heap(server, 2)
    _restore(client, ids[:1])
    monkeypatch.setattr(scrapheap_run_service, "check_can_start", lambda vault: None)
    purge_body = {
        "kind": "purge",
        "picture_ids": ids[1:],
        "confirm_token": _token(client, ids[1:]),
    }

    lost = client.post(RUNS_URL, json=purge_body)
    assert lost.status_code == 409, lost.text
    assert lost.json()["detail"]["code"] == "scrapheap_run_active"
    _work(captured)

    retried = client.post(RUNS_URL, json=purge_body)
    assert retried.status_code == 202, retried.text
    _work(captured)
    assert _states(server, ids[1:]) == [None]


def test_no_task_runner_is_a_503_that_starts_nothing(env):
    """A server with background workers off has nothing to run the slices.

    The run must not be left pending: it would refuse every later run as
    ``scrapheap_run_active`` until the process ended.
    """
    client, server = env
    ids = _heap(server, 1)
    assert server.vault._task_runner is None

    for body in (
        {"kind": "restore"},
        {"kind": "purge", "confirm_token": _token(client, None)},
    ):
        response = client.post(RUNS_URL, json=body)
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "scrapheap_runs_unavailable"
        assert client.get(LATEST_URL).json() == {"run": None}
    assert _states(server, ids) == ["heap"]


def test_a_purge_refused_for_want_of_a_runner_keeps_its_token(env, monkeypatch, caplog):
    """The 503 comes before the token is spent, so a retry can still use it."""
    client, server = env
    ids = _heap(server, 1)
    purge_body = {"kind": "purge", "confirm_token": _token(client, None)}

    with caplog.at_level(logging.INFO, logger=scrapheap_service.logger.name):
        refused = client.post(RUNS_URL, json=purge_body)
    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"]["code"] == "scrapheap_runs_unavailable"
    assert not _token_touched(caplog)

    tasks = _capture(server, monkeypatch)
    retried = client.post(RUNS_URL, json=purge_body)
    assert retried.status_code == 202, retried.text
    _work(tasks)
    assert _states(server, ids) == [None]


def test_a_stopping_task_runner_is_a_503_that_starts_nothing(env, monkeypatch):
    client, server = env
    ids = _heap(server, 1)
    runner = TaskRunner(name="scrapheap-run-test-stopped")
    runner.stop()
    monkeypatch.setattr(server.vault, "_task_runner", runner)

    purge_body = {"kind": "purge", "confirm_token": _token(client, None)}
    for body in ({"kind": "restore"}, purge_body):
        response = client.post(RUNS_URL, json=body)
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "scrapheap_runs_unavailable"
        assert "stopping" in detail["message"]
        assert client.get(LATEST_URL).json() == {"run": None}
    assert _states(server, ids) == ["heap"]

    # The purge spent its token only after the pre-check passed; the refusal
    # from the stopping runner gave it back.
    tasks = _capture(server, monkeypatch)
    retried = client.post(RUNS_URL, json=purge_body)
    assert retried.status_code == 202, retried.text
    _work(tasks)
    assert _states(server, ids) == [None]


def test_an_unknown_run_is_a_404_on_every_run_route(env):
    client, _server = env
    for response in (
        client.get(f"{RUNS_URL}/{uuid.uuid4().hex}"),
        client.post(f"{RUNS_URL}/{uuid.uuid4().hex}/cancel"),
        client.post(f"{RUNS_URL}/{uuid.uuid4().hex}/undo"),
    ):
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"] == "scrapheap_run_not_found"


def test_an_unknown_run_id_is_logged_and_not_repeated_in_the_404(env, caplog):
    """The run id is the caller's path text: the refusal is a fixed sentence."""
    client, _server = env
    planted = f"example-run-{uuid.uuid4().hex}"
    with caplog.at_level(logging.INFO, logger=scrapheap_run_service.logger.name):
        responses = (
            client.get(f"{RUNS_URL}/{planted}"),
            client.post(f"{RUNS_URL}/{planted}/cancel"),
            client.post(f"{RUNS_URL}/{planted}/undo"),
        )
    for response in responses:
        assert response.status_code == 404, response.text
        assert planted not in response.text
        assert response.json()["detail"]["message"] == "No Scrapheap run has that id."
    logged = [
        record.getMessage()
        for record in caplog.records
        if record.name == scrapheap_run_service.logger.name
    ]
    assert f"Scrapheap run {planted!r} not found (404)." in logged


def test_latest_is_the_newest_run_until_another_starts(env, captured):
    """What the Scrapheap reads when it opens, to pick a run up after a reload."""
    client, server = env
    assert client.get(LATEST_URL).json() == {"run": None}
    _heap(server, 1)
    first = _restore(client)
    assert client.get(LATEST_URL).json()["run"]["id"] == first["id"]
    _work(captured)
    ended = client.get(LATEST_URL).json()["run"]
    assert (ended["id"], ended["status"]) == (first["id"], "complete")

    _heap(server, 1)
    second = _restore(client)
    assert client.get(LATEST_URL).json()["run"]["id"] == second["id"]
    _work(captured)


# ── who may reach a run ──────────────────────────────────────────────────────
#
# Every run route is OWNER_ONLY. Two layers refuse any other credential, and
# the auth middleware's READ-token rules run ahead of the gate, so with both on
# a READ token never reaches a policy: a policy widened by mistake would leave
# the first test green. Each layer is also measured with the other one off.


@pytest.mark.parametrize("kind", ["read", *SHARE_RESOURCES, "comfyui"])
def test_no_token_reaches_a_run_route(env, captured, kind):
    """A READ token, a share link's token and the ComfyUI key, as shipped.

    Each is refused on every run route in the ``Authorization`` header, and the
    two that may travel in a URL are refused as ``?token=`` too.
    """
    client, server = env
    setup = _a_pending_run(client, server)
    token = _credential(client, server, kind, setup.held[0])
    transports = [{"headers": {"Authorization": f"Bearer {token}"}}]
    if kind != "comfyui":
        transports.append({"params": {"token": token}})
    stranger = TestClient(server.api)
    answers = [
        answer
        for transport in transports
        for answer in _answers(stranger, setup.requests, **transport)
    ]

    _the_owner_still_reaches_every_run_route(client, setup)
    assert [answer for answer in answers if answer[2] != 403] == []


@pytest.mark.parametrize("kind", ["read", *SHARE_RESOURCES])
def test_the_gate_alone_refuses_a_read_token_on_every_run_route(
    env, captured, monkeypatch, kind
):
    """The middleware's READ-token rules are switched off, to measure the gate.

    Those rules answer first: a READ token's POST is refused as read-only
    before routing, and both GETs are on the READ-token belt. With them off,
    the five ``OWNER_ONLY`` policies are all that stands between a READ token
    and a run. The rollback test below is the same test the other way round.
    """
    client, server = env
    setup = _a_pending_run(client, server)
    scoped = bearer_client(server, _credential(client, server, kind, setup.held[0]))
    monkeypatch.setattr(auth, "WRITE_ENABLED_SCOPES", frozenset({"WRITE", "READ"}))
    monkeypatch.setattr(auth, "READ_BLOCKED_GET_PATHS", frozenset())
    monkeypatch.setattr(auth, "READ_BLOCKED_GET_PREFIXES", ())
    answers = _answers(scoped, setup.requests)

    _the_owner_still_reaches_every_run_route(client, setup)
    assert [answer for answer in answers if answer[2:] != (403, GATE_REFUSAL)] == []


@pytest.mark.parametrize("kind", ["read", *SHARE_RESOURCES])
def test_the_middleware_alone_refuses_a_read_token_on_every_run_route(
    env, captured, monkeypatch, kind
):
    """``AUTHZ_GATE_ENFORCING = False``, the documented rollback.

    Every POST is refused as read-only. A run by id is on the READ-token belt
    under its ``runs/`` prefix; ``runs/latest`` is under that prefix too, and
    on the belt by its own path besides, so either entry alone holds it.
    """
    client, server = env
    setup = _a_pending_run(client, server)
    scoped = bearer_client(server, _credential(client, server, kind, setup.held[0]))
    monkeypatch.setattr(server.authz, "_enforcing", False)
    answers = _answers(scoped, setup.requests)

    _the_owner_still_reaches_every_run_route(client, setup)
    assert [
        answer for answer in answers if answer[2:] != (403, MIDDLEWARE_REFUSAL)
    ] == []


def test_the_gate_alone_refuses_the_comfyui_key_on_every_run_route(
    env, captured, monkeypatch
):
    """The middleware's ComfyUI path check is switched off, to measure the gate.

    The gate admits the key by the route a request matched, and no run route
    is one its scope lists, so it refuses each one before the ``OWNER_ONLY``
    policy runs. The refusal text is the gate's ComfyUI one, not the owner
    check's, which is what tells the two apart.
    """
    client, server = env
    setup = _a_pending_run(client, server)
    keyed = bearer_client(server, _credential(client, server, "comfyui", setup.held[0]))
    asked = []

    def admits_everything(method: str, path: str) -> bool:
        asked.append((method, path))
        return True

    monkeypatch.setattr(auth, "comfyui_scope_admits", admits_everything)
    answers = _answers(keyed, setup.requests)

    _the_owner_still_reaches_every_run_route(client, setup)
    # The middleware consulted the stand-in for every request, so it let each
    # one through to the gate.
    assert {path for _method, path in asked} >= {
        path for _method, path, _template, _body in setup.requests
    }
    assert [
        answer for answer in answers if answer[2:] != (403, auth.COMFYUI_SCOPE_REFUSAL)
    ] == []


def test_the_middleware_alone_refuses_the_comfyui_key_on_every_run_route(
    env, captured, monkeypatch
):
    """``AUTHZ_GATE_ENFORCING = False``: the middleware's path check stands alone."""
    client, server = env
    setup = _a_pending_run(client, server)
    keyed = bearer_client(server, _credential(client, server, "comfyui", setup.held[0]))
    monkeypatch.setattr(server.authz, "_enforcing", False)
    answers = _answers(keyed, setup.requests)

    _the_owner_still_reaches_every_run_route(client, setup)
    assert [
        answer for answer in answers if answer[2:] != (403, auth.COMFYUI_SCOPE_REFUSAL)
    ] == []


def test_an_anonymous_request_is_refused_on_every_run_route(env, captured):
    client, server = env
    setup = _a_pending_run(client, server)
    answers = _answers(TestClient(server.api), setup.requests)

    _the_owner_still_reaches_every_run_route(client, setup)
    assert [answer for answer in answers if answer[2] != 401] == []


def test_a_full_access_token_reaches_every_run_route(env, captured):
    """Over-blocking is its own regression: an unscoped ``ALL`` token is the owner.

    The positive control beside the refusals above, through a bearer header
    and no owner cookie.
    """
    client, server = env
    setup = _a_pending_run(client, server)
    minted = client.post(TOKEN_URL, json={"description": "test-all", "scope": "ALL"})
    assert minted.status_code == 200, minted.text
    owner = bearer_client(server, minted.json()["token"])

    _the_owner_still_reaches_every_run_route(owner, setup)


# ── cancel, stop and failure ─────────────────────────────────────────────────


def test_cancelling_a_pending_run_stops_it_before_anything_is_destroyed(env, captured):
    client, server = env
    ids = _heap(server, 2)
    run = _purge(client, ids)

    cancelled = client.post(f"{RUNS_URL}/{run['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert (cancelled.json()["status"], cancelled.json()["active"]) == (
        "cancelled",
        False,
    )
    assert "Retry-After" not in cancelled.headers

    _work(captured)
    assert _states(server, ids) == ["heap", "heap"]
    again = client.post(f"{RUNS_URL}/{run['id']}/cancel")
    assert again.status_code == 200, again.text
    assert again.json()["status"] == "cancelled"
    assert again.json()["processed"] == 0


def test_cancelling_a_running_run_stops_it_at_the_next_slice(
    env, captured, monkeypatch
):
    """Stop keeps what has been destroyed: a purge has no undo to offer."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    ids = _heap(server, 3)
    run = _purge(client, ids)
    answers = []

    def cancel_once(_task, processed):
        if processed == 1:
            answers.append(client.post(f"{RUNS_URL}/{run['id']}/cancel"))

    _between_slices(monkeypatch, run["id"], cancel_once)
    _work(captured)

    assert len(answers) == 1
    answer = answers[0]
    assert answer.status_code == 200, answer.text
    assert answer.json()["status"] == "running"
    assert answer.json()["cancel_requested"] is True
    assert answer.headers["Retry-After"] == "1"
    ended = _get(client, run["id"])
    assert ended["status"] == "cancelled"
    assert (ended["total"], ended["processed"], ended["deleted_count"]) == (3, 1, 1)
    assert _states(server, ids) == [None, "heap", "heap"]


def test_a_server_stopping_before_the_run_starts_closes_it_interrupted(env, captured):
    """The runner cancels a queued task without running it when it stops.

    Left pending, the run would refuse every later run until the process ended.
    """
    client, server = env
    ids = _heap(server, 1)
    run = _purge(client, ids)

    captured[0].on_cancel()

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("interrupted", "server_stopped")
    assert ended["active"] is False
    _work(captured)
    assert _states(server, ids) == ["heap"]
    assert _restore(client)["status"] == "pending"
    _work(captured)


def test_a_server_stopping_mid_run_closes_it_interrupted_and_announces_the_rest(
    env, captured, monkeypatch
):
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    monkeypatch.setattr(ScrapheapRunTask, "NOTIFY_INTERVAL_S", 3600.0)
    announced = _announced(monkeypatch, server)
    ids = _heap(server, 3)
    run = _purge(client, ids)

    def stop_once(task, processed):
        if processed == 1:
            # What the task runner does to its running tasks when it stops.
            task.on_cancel()

    _between_slices(monkeypatch, run["id"], stop_once)
    _work(captured)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("interrupted", "server_stopped")
    assert (ended["processed"], ended["deleted_count"]) == (1, 1)
    assert _states(server, ids) == [None, "heap", "heap"]
    # What did commit is still announced, so other tabs drop those cards.
    assert [event["picture_ids"] for event in announced] == [ids[:1]]
    assert announced[0]["change_kind"] == "removed"


def test_a_failing_slice_fails_the_run_and_keeps_what_committed(
    env, captured, monkeypatch, caplog
):
    """The run records a generic message; the exception goes to the log only.

    The exception text can carry a file path, SQL or library content, and the
    run view is served to every owner session, so the view must not echo it.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    ids = _heap(server, 3)
    original = scrapheap_service.purge_scrapheap_pictures
    calls = []
    leak = "example disk failure at /home/me/library/a.png in SELECT picture"

    def second_slice_fails(vault, chunk, include_protected, **kwargs):
        calls.append(list(chunk))
        if len(calls) == 2:
            raise OSError(leak)
        return original(vault, chunk, include_protected, **kwargs)

    monkeypatch.setattr(
        scrapheap_service, "purge_scrapheap_pictures", second_slice_fails
    )
    run = _purge(client, ids)

    with pytest.raises(OSError, match="example disk failure"):
        _work(captured)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("failed", "unexpected_error")
    assert ended["error"] == (
        "The server hit an unexpected error; the server log has the details."
    )
    assert "/home/me" not in json.dumps(ended)
    assert "SELECT" not in json.dumps(ended)
    logged = [r for r in caplog.records if run["id"] in r.getMessage()]
    assert any(r.exc_info and leak in str(r.exc_info[1]) for r in logged)
    assert (ended["processed"], ended["deleted_count"]) == (1, 1)
    assert _states(server, ids) == [None, "heap", "heap"]
    monkeypatch.setattr(scrapheap_service, "purge_scrapheap_pictures", original)
    assert _restore(client)["status"] == "pending"
    _work(captured)


def test_locked_pictures_are_kept_and_named_only_once_the_run_ends(
    env, captured, monkeypatch
):
    """A poll carries the count while the run works, the ids when it ends.

    A Scrapheap full of locked pictures would otherwise put every one of their
    ids in every one-second poll.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    ids = _heap(server, 2)
    _lock(server, ids[0])
    run = _purge(client, ids)
    polls = []
    _between_slices(
        monkeypatch,
        run["id"],
        lambda _task, processed: (
            polls.append(_get(client, run["id"])) if processed == 1 else None
        ),
    )

    _work(captured)

    assert len(polls) == 1
    assert (polls[0]["skipped_locked_count"], polls[0]["skipped_locked"]) == (1, [])
    assert polls[0]["snapshots_with_deleted"] is None
    ended = _get(client, run["id"])
    assert ended["skipped_locked"] == [ids[0]]
    assert ended["deleted_count"] == 1
    assert _states(server, ids) == ["heap", None]


# ── the durable record ───────────────────────────────────────────────────────
#
# A run is written through to a ``scrapheaprun`` row. A restart is a new
# registry plus what start-up does, ``recover_runs``; the module's server is
# not rebuilt for it.


class _HardStop(BaseException):
    """A process dying mid-run: no except clause of the task's catches it."""


def _row(server, run_id: str):
    """The run's row as plain values, or ``None``."""

    def read(session):
        row = session.get(ScrapheapRunRecord, run_id)
        return None if row is None else row.model_dump()

    return _run(server, read)


def _restart(server) -> int:
    """What a new process does with this library's runs: forget, then recover."""
    server.vault.scrapheap_runs = ScrapheapRunRegistry()
    return scrapheap_run_service.recover_runs(server.vault)


def test_a_run_is_on_disk_before_the_start_answers(env, captured):
    client, server = env
    ids = _heap(server, 2)

    run = _purge(client, ids)

    row = _row(server, run["id"])
    assert row is not None
    assert (row["kind"], row["status"], row["total"]) == ("purge", "pending", 2)
    assert row["deleted_before"] is not None
    _work(captured)


def test_a_run_in_flight_at_a_restart_reads_back_interrupted_with_its_counts(
    env, captured, monkeypatch
):
    """The client polling it learns how it ended, instead of a 404.

    Nothing resumes: the pictures it had not reached stay in the Scrapheap.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    ids = _heap(server, 3)
    _lock(server, ids[0])
    run = _purge(client, ids)
    rows = []

    def die_after_two(_task, processed):
        rows.append(_row(server, run["id"]))
        if processed == 2:
            raise _HardStop()

    _between_slices(monkeypatch, run["id"], die_after_two)
    with pytest.raises(_HardStop):
        _work(captured)

    # Written after every slice, so the row is as current as the last one.
    assert [(r["status"], r["processed"]) for r in rows] == [
        ("running", 1),
        ("running", 2),
    ]
    assert _restart(server) == 1

    ended = client.get(LATEST_URL).json()["run"]
    assert ended["id"] == run["id"]
    assert (ended["status"], ended["active"]) == ("interrupted", False)
    assert ended["error_code"] == "server_stopped"
    assert ended["error"] == "The server stopped before the run finished."
    assert (ended["processed"], ended["deleted_count"]) == (2, 1)
    assert (ended["skipped_locked_count"], ended["skipped_locked"]) == (1, [])
    assert ended["finished_at"] is not None
    assert _get(client, run["id"]) == ended
    assert _row(server, run["id"])["status"] == "interrupted"
    assert _states(server, ids) == ["heap", None, "heap"]


def test_a_run_pending_at_a_restart_is_interrupted_and_never_resumes(env, captured):
    client, server = env
    ids = _heap(server, 1)
    run = _purge(client, ids)

    assert _restart(server) == 1

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("interrupted", "server_stopped")
    # The old process's task, were it ever run, finds nothing pending to take.
    _work(captured)
    assert _states(server, ids) == ["heap"]
    assert _get(client, run["id"])["status"] == "interrupted"
    assert _restore(client)["status"] == "pending"
    _work(captured)


def test_an_ended_run_reads_back_after_a_restart_as_it_ended(
    env, captured, monkeypatch
):
    """Every field the view shows survives the row, the lists included."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    restored = _restore(client, _heap(server, 2))
    _work(captured)
    ids = _heap(server, 2)
    _lock(server, ids[0])
    purged = _purge(client, ids)
    _work(captured)
    before = {run_id: _get(client, run_id) for run_id in (restored["id"], purged["id"])}
    assert len(before[restored["id"]]["segment_batch_ids"]) == 1
    assert before[purged["id"]]["skipped_locked"] == [ids[0]]

    assert _restart(server) == 0

    after = {run_id: _get(client, run_id) for run_id in before}
    assert after[restored["id"]] == before[restored["id"]]
    # Only the locked ids are not stored; their count is.
    assert after[purged["id"]] == {**before[purged["id"]], "skipped_locked": []}
    assert client.get(LATEST_URL).json()["run"]["id"] == purged["id"]


def test_a_cancel_is_on_disk_before_it_answers(env, captured):
    client, server = env
    run = _purge(client, _heap(server, 1))

    response = client.post(f"{RUNS_URL}/{run['id']}/cancel")

    assert response.status_code == 200, response.text
    row = _row(server, run["id"])
    assert (row["status"], row["cancel_requested"]) == ("cancelled", True)
    _work(captured)


def test_only_the_last_ended_runs_are_kept_across_a_restart(env, captured, monkeypatch):
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "HISTORY_KEPT", 3)
    run_ids = []
    for _ in range(5):
        run_ids.append(_restore(client, _heap(server, 1))["id"])
        _work(captured)

    # Each start prunes in its own transaction, so the newest start is the one
    # not yet counted: at most HISTORY_KEPT ended rows plus the run itself.
    kept = _run(server, lambda s: list(s.exec(select(ScrapheapRunRecord.id)).all()))
    assert sorted(kept) == sorted(run_ids[1:])
    _restart(server)

    for run_id in run_ids[2:]:
        assert _get(client, run_id)["status"] == "complete"
    for run_id in run_ids[:2]:
        response = client.get(f"{RUNS_URL}/{run_id}")
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"] == "scrapheap_run_not_found"


def test_a_run_the_runner_refuses_leaves_no_row(env, monkeypatch):
    client, server = env
    _heap(server, 1)
    runner = TaskRunner(name="scrapheap-run-test-refuses")
    runner.stop()
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    written = []
    original = scrapheap_run_service._write_record_in_session

    def spy(session, registry, run_id, prune):
        written.append(run_id)
        return original(session, registry, run_id, prune)

    monkeypatch.setattr(scrapheap_run_service, "_write_record_in_session", spy)

    response = client.post(RUNS_URL, json={"kind": "restore"})

    assert response.status_code == 503, response.text
    assert len(written) == 1
    assert _row(server, written[0]) is None


def test_a_closing_database_at_the_start_is_a_503_that_starts_nothing(
    env, captured, monkeypatch
):
    client, server = env
    ids = _heap(server, 1)
    original = server.vault.db.run_task

    def closing(fn, *args, **kwargs):
        if fn is scrapheap_run_service._write_record_in_session:
            raise database.DatabaseUnavailableError("example: the database is closing")
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", closing)

    response = client.post(RUNS_URL, json={"kind": "restore"})

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "scrapheap_runs_unavailable"
    assert "stopping" in detail["message"]
    assert client.get(LATEST_URL).json() == {"run": None}
    assert captured == []
    assert _states(server, ids) == ["heap"]


def test_start_up_closes_orphaned_runs_even_without_background_workers(env):
    """A server with no workers still shows how a run of the last process ended.

    This module's server runs no background workers, the case where
    ``Vault.start`` used to return before closing anything.
    """
    client, server = env
    assert server.vault.runs_background_work() is False
    scrapheap_id = uuid.uuid4().hex
    request_id = f"test-{uuid.uuid4().hex}"

    def orphan(session):
        session.add(
            ScrapheapRunRecord(
                id=scrapheap_id,
                request_id=f"test-{uuid.uuid4().hex}",
                kind="restore",
                status="running",
            )
        )
        session.add(DedupAutoStackRun(request_id=request_id, status="running"))
        session.commit()

    _run(server, orphan)
    server.vault.scrapheap_runs = ScrapheapRunRegistry()
    server.vault._runs_recovered = False

    server.vault.start()

    assert _get(client, scrapheap_id)["status"] == "interrupted"
    stacked = _run(
        server,
        lambda s: s.exec(
            select(DedupAutoStackRun.status).where(
                DedupAutoStackRun.request_id == request_id
            )
        ).one(),
    )
    assert stacked == "interrupted"


def test_a_second_start_does_not_interrupt_this_processes_run(env, captured):
    """``Vault.start`` runs at Server init and again at lifespan start-up.

    The row is read as well as the run: a second recovery would mark the row
    interrupted while the live copy in the registry still read ``pending``.
    """
    client, server = env
    run = _restore(client, _heap(server, 1))

    server.vault.start()

    assert _row(server, run["id"])["status"] == "pending"
    assert _get(client, run["id"])["status"] == "pending"
    _work(captured)
    assert _get(client, run["id"])["status"] == "complete"


# ── restore History ──────────────────────────────────────────────────────────


def test_a_restore_run_records_one_segment_that_one_undo_puts_back(
    env, captured, monkeypatch
):
    """Each slice is the one restore path at LOW priority, announcing nothing.

    The run coalesces the slices' ``restored`` events itself, so a slice that
    announced too would send every picture twice.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    ids = _heap(server, 5)
    slices = []
    original = scrapheap_service.restore_pictures

    def recording(vault, chunk, **kwargs):
        slices.append((list(chunk), kwargs["priority"], kwargs["announce"]))
        return original(vault, chunk, **kwargs)

    monkeypatch.setattr(scrapheap_service, "restore_pictures", recording)
    run = _restore(client, ids)
    _work(captured)

    assert slices == [
        (ids[0:2], DBPriority.LOW, False),
        (ids[2:4], DBPriority.LOW, False),
        (ids[4:5], DBPriority.LOW, False),
    ]
    ended = _get(client, run["id"])
    assert (ended["status"], ended["restored_count"]) == ("complete", 5)
    assert len(ended["segment_batch_ids"]) == 1
    assert _segment_of(server, ended["segment_batch_ids"][0]) == ids
    assert _states(server, ids) == ["live"] * 5

    undo = client.post(UNDO_URL)
    assert undo.status_code == 200, undo.text
    assert _states(server, ids) == ["heap"] * 5


def test_an_edit_between_slices_starts_a_new_segment(env, captured, monkeypatch):
    """A batch must stay contiguous at the top of History to be undone.

    A restore the owner makes in another tab while the run works lands in
    History between two slices. Recording the next slice into the same batch
    would split it around that restore and leave neither undoable, so the run
    starts a new segment, and Ctrl+Z walks back through all three in order.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    ids = _heap(server, 4)
    other = _heap(server, 1)
    run = _restore(client, ids)

    def restore_elsewhere(_task, processed):
        if processed == 2:
            response = client.post(RESTORE_URL, json={"picture_ids": other})
            assert response.status_code == 200, response.text

    _between_slices(monkeypatch, run["id"], restore_elsewhere)
    _work(captured)

    segments = _get(client, run["id"])["segment_batch_ids"]
    assert len(segments) == 2
    assert _segment_of(server, segments[0]) == ids[:2]
    assert _segment_of(server, segments[1]) == ids[2:]

    assert client.post(UNDO_URL).status_code == 200
    assert _states(server, ids + other) == ["live", "live", "heap", "heap", "live"]
    assert client.post(UNDO_URL).status_code == 200
    assert _states(server, ids + other) == ["live", "live", "heap", "heap", "heap"]
    assert client.post(UNDO_URL).status_code == 200
    assert _states(server, ids + other) == ["heap"] * 5


def test_a_segment_that_recorded_nothing_is_not_reported(env, captured, monkeypatch):
    """A run names only History batches that exist, so the client's undo works.

    Here another tab restores the run's remaining pictures between its slices.
    The next slice opens a segment, because that restore is now on top of
    History, and then finds nothing left to bring back.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    ids = _heap(server, 4)
    run = _restore(client, ids)

    def restore_the_rest(_task, processed):
        if processed == 2:
            response = client.post(RESTORE_URL, json={"picture_ids": ids[2:]})
            assert response.status_code == 200, response.text

    _between_slices(monkeypatch, run["id"], restore_the_rest)
    _work(captured)

    ended = _get(client, run["id"])
    assert (ended["restored_count"], ended["processed"]) == (2, 4)
    assert len(ended["segment_batch_ids"]) == 1
    assert _segment_of(server, ended["segment_batch_ids"][0]) == ids[:2]


def test_no_segment_holds_more_pictures_than_the_cap(env, captured, monkeypatch):
    """The cap bounds what one Ctrl+Z, and one undo transaction, has to do."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    monkeypatch.setattr(scrapheap_run_service, "SEGMENT_PICTURE_CAP", 3)
    ids = _heap(server, 5)
    run = _restore(client, ids)
    _work(captured)

    segments = _get(client, run["id"])["segment_batch_ids"]
    assert [_segment_of(server, batch) for batch in segments] == [ids[:2], ids[2:]]


def test_a_restore_run_announces_its_pictures_with_the_requesting_tab(
    env, captured, monkeypatch
):
    """The tab that started it is named, so it can tell its own run's events."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    monkeypatch.setattr(ScrapheapRunTask, "NOTIFY_INTERVAL_S", 3600.0)
    announced = _announced(monkeypatch, server)
    ids = _heap(server, 3)
    run = _restore(client, ids, **{"X-Client-Id": "example-tab"})
    _work(captured)

    # Coalesced: one event for three slices, the final flush.
    assert announced == [
        {
            "picture_ids": ids,
            "origin_client_id": "example-tab",
            "source": "ui",
            "change_kind": "restored",
            "run_id": run["id"],
            "run_kind": "restore",
        }
    ]


def test_a_purge_run_announces_once_per_interval(env, captured, monkeypatch):
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    monkeypatch.setattr(ScrapheapRunTask, "NOTIFY_INTERVAL_S", 0.0)
    announced = _announced(monkeypatch, server)
    ids = _heap(server, 3)
    _purge(client, ids)
    _work(captured)

    assert [event["picture_ids"] for event in announced] == [[pid] for pid in ids]
    assert {event["change_kind"] for event in announced} == {"removed"}
    assert {event["source"] for event in announced} == {"external"}


# ── slice size (time-budgeted) ───────────────────────────────────────────────


def _slice_sizes(monkeypatch, kind: str) -> list[int]:
    """The size of every slice the purge or restore path is handed from now on."""
    sizes = []
    name = "purge_scrapheap_pictures" if kind == "purge" else "restore_pictures"
    original = getattr(scrapheap_service, name)

    def recording(vault, chunk, *args, **kwargs):
        sizes.append(len(chunk))
        return original(vault, chunk, *args, **kwargs)

    monkeypatch.setattr(scrapheap_service, name, recording)
    return sizes


def _fixed_slices(monkeypatch, size: int) -> None:
    """Every slice exactly *size* pictures, whatever the writer time."""
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", size)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_FIRST_PICTURES", size)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_BUDGET_S", 3600.0)


def test_a_run_shows_the_slice_bounds_it_keeps_to(env, captured, monkeypatch):
    """The view's bounds are the ones the run works by, fixed when it starts.

    A Stop takes effect at the next slice boundary, so these are what a client
    can say the run will still do after one.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 3)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_BUDGET_S", 0.125)
    ids = _heap(server, 7)
    sizes = _slice_sizes(monkeypatch, "purge")
    run = _purge(client, ids)
    assert (run["slice_max_pictures"], run["slice_budget_s"]) == (3, 0.125)

    # A change to the setting after the run started is not this run's.
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 1)
    _work(captured)

    assert sizes == [3, 3, 1]
    ended = _get(client, run["id"])
    assert (ended["slice_max_pictures"], ended["slice_budget_s"]) == (3, 0.125)


def test_the_first_slice_is_small_until_a_cost_is_measured(monkeypatch):
    monkeypatch.setattr(scrapheap_run_service, "SLICE_FIRST_PICTURES", 40)
    assert _SliceSizer(500, 0.25).next_size == 40
    assert _SliceSizer(30, 0.25).next_size == 30


def test_a_slice_is_sized_to_the_budget_over_the_averaged_cost(monkeypatch):
    """Budget over cost per picture, the cost a moving average of the slices."""
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MIN_PICTURES", 10)
    sizer = _SliceSizer(500, 0.25)
    sizer.record(50, 0.5)  # 10 ms a picture
    assert sizer.next_size == 25
    # 1 ms a picture now: the average is 5.5 ms, so the next slice grows to 45
    # rather than jumping to 250 on one cheap slice.
    sizer.record(25, 0.025)
    assert sizer.next_size == 45
    sizer.record(0, 1.0)  # an empty slice measures nothing
    assert sizer.next_size == 45


@pytest.mark.parametrize(
    "max_pictures, pictures, writer_s, expected",
    [
        (500, 1, 10.0, 10),  # far over budget: still the minimum, never 0
        (4, 1, 10.0, 4),  # a maximum below the minimum wins
        (500, 100, 0.001, 500),  # far under budget: the maximum
        (500, 100, 0.0, 500),  # nothing measured on the writer
    ],
)
def test_a_slice_size_stays_within_its_bounds(
    monkeypatch, max_pictures, pictures, writer_s, expected
):
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MIN_PICTURES", 10)
    sizer = _SliceSizer(max_pictures, 0.25)
    sizer.record(pictures, writer_s)
    assert sizer.next_size == expected


@pytest.mark.parametrize("held_on", ["writer", "task thread"])
def test_the_next_slice_is_sized_by_the_writer_time_the_last_one_took(
    env, captured, monkeypatch, held_on
):
    """Writer time, metered through the slice's nested writer tasks, not wall time.

    The first slice holds for 0.1 s a picture, as a nested service layer would:
    on the writer, through a task of its own, that is far over the 1 s budget
    and the next slice shrinks to 10 at most; on the task thread (file
    removal, say) it holds nobody up, and the next slice takes the rest.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_FIRST_PICTURES", 4)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MIN_PICTURES", 1)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 100)
    monkeypatch.setattr(scrapheap_run_service, "SLICE_BUDGET_S", 1.0)
    ids = _heap(server, 30)
    sizes = []
    original = scrapheap_service.purge_scrapheap_pictures

    def held(vault, chunk, include_protected, **kwargs):
        sizes.append(len(chunk))
        if len(sizes) == 1:
            seconds = 0.1 * len(chunk)
            if held_on == "writer":
                vault.db.run_task(
                    lambda _session: time.sleep(seconds), priority=DBPriority.LOW
                )
            else:
                time.sleep(seconds)
        return original(vault, chunk, include_protected, **kwargs)

    monkeypatch.setattr(scrapheap_service, "purge_scrapheap_pictures", held)
    run = _purge(client, ids)
    _work(captured)

    assert sizes[0] == 4
    if held_on == "writer":
        assert 1 <= sizes[1] <= 10
    else:
        assert sizes[1:] == [26]
    assert sum(sizes) == 30
    assert _get(client, run["id"])["deleted_count"] == 30


def test_a_restore_slice_meters_the_writer_time_of_its_nested_tasks(
    env, captured, monkeypatch
):
    """A restore slice reaches the writer through History's recorder.

    That layer submits the slice's task itself, so the meter has to follow the
    task's context rather than any one call site.
    """
    client, server = env
    _fixed_slices(monkeypatch, 2)
    measured = []
    original = _SliceSizer.record

    def record(self, pictures, writer_s):
        measured.append((pictures, writer_s))
        original(self, pictures, writer_s)

    monkeypatch.setattr(_SliceSizer, "record", record)
    ids = _heap(server, 5)
    _restore(client, ids)
    _work(captured)

    assert [pictures for pictures, _ in measured] == [2, 2, 1]
    assert all(writer_s > 0 for _, writer_s in measured)


class _SliceStatements:
    """SQL statements per slice: the ones run under each slice's writer meter.

    A run wraps each slice in its own :func:`pixlstash.database.writer_meter`,
    whose context variable is visible wherever the slice's statements run (on
    the writer inside the slice's tasks, or on the task thread), and to nothing
    else, so the count is the slice's own.
    """

    def __init__(self, server):
        self._engine = server.vault.db._engine
        self.per_meter: Counter = Counter()

    def _on_execute(self, conn, cursor, statement, parameters, context, executemany):
        meter = database._WRITER_METER.get()
        if meter is not None:
            self.per_meter[meter] += 1

    def __enter__(self):
        event.listen(self._engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *exc_info):
        event.remove(self._engine, "before_cursor_execute", self._on_execute)
        return False

    @property
    def counts(self) -> list[int]:
        return list(self.per_meter.values())


def _rebin(server, ids: list[int]) -> None:
    """Put *ids* back in the Scrapheap, with an empty History."""

    def rebin(session):
        session.exec(delete(Operation))
        session.exec(
            update(Picture)
            .where(Picture.id.in_(ids))
            .values(deleted=True, deleted_at=datetime.now(timezone.utc))
        )
        session.commit()

    _run(server, rebin)


def test_a_large_run_keeps_its_slices_and_segments_bounded(env, captured, monkeypatch):
    """At scale: slice and segment counts, and a slice's SQL that does not grow.

    A restore of N pictures in slices of 100 records ceil(N / 300) History
    segments of at most 300, each one Ctrl+Z; then a purge of the same
    pictures. A slice issues the same statements whether the run holds 500
    pictures or 2,000: nothing in a slice reads the whole selection.
    """
    client, server = env
    _fixed_slices(monkeypatch, 100)
    monkeypatch.setattr(scrapheap_run_service, "SEGMENT_PICTURE_CAP", 300)
    per_slice = {}
    for total in (500, 2000):
        ids = _heap(server, total)
        restore_sizes = _slice_sizes(monkeypatch, "restore")
        with _SliceStatements(server) as restoring:
            run = _restore(client, ids)
            _work(captured)
        ended = _get(client, run["id"])
        assert (ended["status"], ended["restored_count"]) == ("complete", total)
        segments = [_segment_of(server, batch) for batch in ended["segment_batch_ids"]]
        assert len(segments) == math.ceil(total / 300)
        assert all(len(segment) <= 300 for segment in segments)
        assert [pid for segment in segments for pid in segment] == ids
        assert restore_sizes == [100] * math.ceil(total / 100)

        _rebin(server, ids)
        purge_sizes = _slice_sizes(monkeypatch, "purge")
        with _SliceStatements(server) as purging:
            run = _purge(client, ids)
            _work(captured)
        assert _get(client, run["id"])["deleted_count"] == total
        assert purge_sizes == [100] * math.ceil(total / 100)
        assert len(restoring.counts) == len(purging.counts) == total // 100
        per_slice[total] = (sorted(set(restoring.counts)), sorted(set(purging.counts)))

    assert per_slice[2000] == per_slice[500]


# ── end to end ───────────────────────────────────────────────────────────────


def test_a_real_task_runner_works_a_run_to_the_end(env, monkeypatch):
    """The route, the runner and the task, wired as the server wires them."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    ids = _heap(server, 5)
    runner = TaskRunner(name="scrapheap-run-test", num_workers=1)
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    runner.start()
    try:
        run = _purge(client, ids)
        deadline = time.monotonic() + 30.0
        view = _get(client, run["id"])
        while view["active"] and time.monotonic() < deadline:
            time.sleep(0.05)
            view = _get(client, run["id"])
    finally:
        runner.stop()

    assert (view["status"], view["deleted_count"]) == ("complete", 5)
    assert _states(server, ids) == [None] * 5


# ── Prefer: wait (RFC 7240) ──────────────────────────────────────────────────


def test_a_short_run_answers_with_its_end_when_the_client_waits(env, monkeypatch):
    """The run's end wakes the request: it answers as soon as the run is done."""
    client, server = env
    ids = _heap(server, 2)
    runner = TaskRunner(name="scrapheap-run-wait-test", num_workers=1)
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    runner.start()
    try:
        began = time.monotonic()
        response = client.post(
            RUNS_URL,
            json={"kind": "restore", "picture_ids": ids},
            headers={"Prefer": "wait=2"},
        )
        waited = time.monotonic() - began
    finally:
        runner.stop()

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["active"], body["status"], body["restored_count"]) == (
        False,
        "complete",
        2,
    )
    assert response.headers["Preference-Applied"] == "wait=2"
    assert _varies_on_prefer(response)
    assert response.headers["Location"] == f"{RUNS_URL}/{body['id']}"
    assert "Retry-After" not in response.headers
    # Woken by the end, not by the timeout.
    assert waited < 1.5
    assert _states(server, ids) == ["live", "live"]


def test_a_run_that_outlasts_the_wait_answers_202_after_the_cap(env, captured):
    """A wait longer than the cap is cut to it; then the usual 202 and poll hint."""
    client, server = env
    ids = _heap(server, 1)
    began = time.monotonic()
    response = client.post(
        RUNS_URL,
        json={"kind": "restore", "picture_ids": ids},
        headers={"Prefer": "respond-async, wait=600"},
    )
    waited = time.monotonic() - began

    assert response.status_code == 202, response.text
    assert response.json()["active"] is True
    assert response.headers["Retry-After"] == "1"
    assert _varies_on_prefer(response)
    assert "Preference-Applied" not in response.headers
    assert 1.9 <= waited < 3.5
    _work(captured)


def test_cancelling_a_pending_run_wakes_a_waiting_start(env, captured):
    client, server = env
    ids = _heap(server, 1)
    cancelled = []

    def cancel_when_recorded():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            run = scrapheap_run_service.latest_run(server.vault)
            if run is not None and run["active"]:
                cancelled.append(
                    scrapheap_run_service.cancel_run(server.vault, run["id"])
                )
                return
            time.sleep(0.01)

    canceller = threading.Thread(target=cancel_when_recorded)
    began = time.monotonic()
    canceller.start()
    response = client.post(
        RUNS_URL,
        json={"kind": "restore", "picture_ids": ids},
        headers={"Prefer": "wait=2"},
    )
    waited = time.monotonic() - began
    canceller.join()

    assert len(cancelled) == 1
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert waited < 1.5
    _work(captured)
    assert _states(server, ids) == ["heap"]


@pytest.mark.parametrize(
    "prefer", [None, "wait=abc", "wait=0", "wait=-1", "wait=1.5", "respond-async"]
)
def test_a_start_without_a_usable_wait_answers_at_once(env, captured, prefer):
    client, server = env
    headers = {} if prefer is None else {"Prefer": prefer}
    began = time.monotonic()
    response = client.post(
        RUNS_URL,
        json={"kind": "restore", "picture_ids": _heap(server, 1)},
        headers=headers,
    )
    waited = time.monotonic() - began

    assert response.status_code == 202, response.text
    assert _varies_on_prefer(response)
    assert "Preference-Applied" not in response.headers
    assert waited < 0.9
    _work(captured)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], None),
        (["wait=1"], 1),
        (["WAIT = 1"], 1),
        (['respond-async, wait="2"; extra=x'], 2),
        (["wait=5"], 2),
        (["handling=lenient", "wait=1"], 1),
        (["wait=1, wait=2"], 1),
        (["wait=x, wait=2"], None),
        (["wait=0"], None),
        (["wait=²"], None),
    ],
)
def test_the_prefer_header_is_read_as_rfc_7240_writes_it(values, expected):
    assert _preferred_wait(values) == expected


# ── idempotent start (request_id) ────────────────────────────────────────────


def test_a_run_started_without_a_request_id_carries_one_the_server_minted(
    env, captured
):
    client, server = env
    run = _restore(client, _heap(server, 1))
    assert re.fullmatch(r"srv-[0-9a-f]{32}", run["request_id"]), run["request_id"]
    assert _row(server, run["id"])["request_id"] == run["request_id"]
    _work(captured)


def test_a_replayed_request_id_answers_200_with_the_run_it_started(env, captured):
    """A retry after a lost answer: the same run, not a second one or a 409."""
    client, server = env
    ids = _heap(server, 2)
    body = {"kind": "restore", "picture_ids": ids, "request_id": "example-gesture-1"}
    first = client.post(RUNS_URL, json=body)
    assert first.status_code == 202, first.text

    # While the first is still active: the replay, not the active-run 409.
    again = client.post(RUNS_URL, json=body)
    assert again.status_code == 200, again.text
    assert (again.json()["id"], again.json()["request_id"]) == (
        first.json()["id"],
        "example-gesture-1",
    )
    assert again.headers["Location"] == first.headers["Location"]
    assert again.headers["Retry-After"] == "1"
    assert _varies_on_prefer(again)
    assert len(captured) == 1

    _work(captured)
    ended = client.post(RUNS_URL, json=body)
    assert ended.status_code == 200, ended.text
    assert (ended.json()["id"], ended.json()["status"]) == (
        first.json()["id"],
        "complete",
    )
    assert "Retry-After" not in ended.headers
    assert captured == []


def test_a_request_id_reused_for_other_parameters_is_422_naming_the_original(
    env, captured
):
    """Kind and selection are compared; the selection's order is not, nor a
    restore's ignored ``include_protected``."""
    client, server = env
    ids = _heap(server, 3)
    body = {
        "kind": "restore",
        "picture_ids": ids[:2],
        "request_id": "example-gesture-3",
    }
    first = client.post(RUNS_URL, json=body).json()
    _work(captured)
    whole_heap = {key: value for key, value in body.items() if key != "picture_ids"}

    refused = [
        client.post(RUNS_URL, json=other)
        for other in (
            {**body, "picture_ids": ids},
            whole_heap,
            # A purge without its token: the reuse is found before the token
            # is asked for, so it is the 422 and not the 400.
            {**body, "kind": "purge"},
        )
    ]
    same = client.post(
        RUNS_URL,
        json={**body, "picture_ids": ids[1::-1], "include_protected": True},
    )

    for response in refused:
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "request_id_reused"
        assert detail["run"]["id"] == first["id"]
    assert same.status_code == 200, same.text
    assert same.json()["id"] == first["id"]
    assert captured == []


@pytest.mark.parametrize("request_id", ["", "has space", "x" * 129, "café", "a/b"])
def test_a_malformed_request_id_is_refused_before_anything_starts(
    env, captured, request_id
):
    client, server = env
    response = client.post(
        RUNS_URL,
        json={
            "kind": "restore",
            "picture_ids": _heap(server, 1),
            "request_id": request_id,
        },
    )
    assert response.status_code == 422, response.text
    assert captured == []
    assert client.get(LATEST_URL).json() == {"run": None}


def test_a_request_id_of_the_longest_allowed_length_is_accepted(env, captured):
    """The positive control for the length bound."""
    client, server = env
    request_id = "a" * 128
    response = client.post(
        RUNS_URL,
        json={
            "kind": "restore",
            "picture_ids": _heap(server, 1),
            "request_id": request_id,
        },
    )
    assert response.status_code == 202, response.text
    assert response.json()["request_id"] == request_id
    _work(captured)


def test_a_replayed_purge_does_not_spend_its_token_again(env, captured, caplog):
    """The replay is answered before the token is looked at.

    So the retry of a purge whose 202 was lost gets its run back with the token
    it already spent, and a different ``include_protected`` is a reuse.
    """
    client, server = env
    ids = _heap(server, 2)
    body = {
        "kind": "purge",
        "picture_ids": ids,
        "confirm_token": _token(client, ids),
        "request_id": "example-gesture-2",
    }
    first = client.post(RUNS_URL, json=body)
    assert first.status_code == 202, first.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=scrapheap_service.logger.name):
        active = client.post(RUNS_URL, json=body)
        _work(captured)
        ended = client.post(RUNS_URL, json=body)
        reused = client.post(RUNS_URL, json={**body, "include_protected": True})

    assert (active.status_code, ended.status_code) == (200, 200)
    assert active.json()["id"] == ended.json()["id"] == first.json()["id"]
    assert ended.json()["status"] == "complete"
    assert not _token_touched(caplog)
    assert reused.status_code == 422, reused.text
    assert reused.json()["detail"]["code"] == "request_id_reused"
    assert captured == []
    assert _states(server, ids) == [None, None]


def test_a_purge_whose_request_id_won_the_race_gets_its_token_back(
    env, captured, monkeypatch
):
    """The replay check passed, then the same id started its run first.

    This request started nothing, so the token it redeemed goes back: a later
    purge can still use it.
    """
    client, server = env
    ids = _heap(server, 2)
    body = {"kind": "purge", "picture_ids": ids, "request_id": "example-gesture-4"}
    first = client.post(RUNS_URL, json={**body, "confirm_token": _token(client, ids)})
    assert first.status_code == 202, first.text
    second_token = _token(client, ids)
    monkeypatch.setattr(scrapheap_run_service, "find_replay", lambda *a, **k: None)
    monkeypatch.setattr(scrapheap_run_service, "check_can_start", lambda vault: None)

    raced = client.post(RUNS_URL, json={**body, "confirm_token": second_token})

    assert raced.status_code == 200, raced.text
    assert raced.json()["id"] == first.json()["id"]
    assert len(captured) == 1
    _work(captured)
    monkeypatch.undo()
    _capture(server, monkeypatch)
    retried = client.post(
        RUNS_URL,
        json={"kind": "purge", "picture_ids": ids, "confirm_token": second_token},
    )
    assert retried.status_code == 202, retried.text


def test_a_request_id_replays_after_a_restart(env, captured):
    """The id and what it was sent with are on disk with the run."""
    client, server = env
    ids = _heap(server, 2)
    body = {"kind": "restore", "picture_ids": ids, "request_id": "example-gesture-5"}
    first = client.post(RUNS_URL, json=body).json()
    _work(captured)
    row = _row(server, first["id"])
    assert row["request_id"] == "example-gesture-5"
    assert row["request_fingerprint"] == scrapheap_run_service.request_fingerprint(
        "restore", ids, False
    )

    assert _restart(server) == 0

    again = client.post(RUNS_URL, json=body)
    assert again.status_code == 200, again.text
    assert again.json()["id"] == first["id"]
    reused = client.post(RUNS_URL, json={**body, "picture_ids": ids[:1]})
    assert reused.status_code == 422, reused.text
    assert captured == []


# ── undo of a restore run ────────────────────────────────────────────────────


def _start_undo(client, run_id: str, **body):
    return client.post(f"{RUNS_URL}/{run_id}/undo", json=body or None)


def _undo(client, run_id: str, **body) -> dict:
    response = _start_undo(client, run_id, **body)
    assert response.status_code == 202, response.text
    return response.json()


def _newest_operation_id(server) -> int:
    return _run(
        server,
        lambda session: session.exec(
            select(Operation.id).order_by(Operation.id.desc()).limit(1)
        ).one(),
    )


def test_an_undo_run_puts_a_restore_back_segment_by_segment_newest_first(
    env, captured, monkeypatch
):
    """One LOW writer task per History segment, each the undo Ctrl+Z would do.

    Newest first, because History undoes from the top; after it, Redo brings
    back the segment the undo did last, the oldest, as it would after four
    Ctrl+Z.
    """
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    monkeypatch.setattr(scrapheap_run_service, "SEGMENT_PICTURE_CAP", 2)
    ids = _heap(server, 7)
    restored = _restore(client, ids)
    _work(captured)
    ended_restore = _get(client, restored["id"])
    segments = ended_restore["segment_batch_ids"]
    assert [_segment_of(server, batch) for batch in segments] == [
        ids[0:2],
        ids[2:4],
        ids[4:6],
        ids[6:7],
    ]
    assert ended_restore["can_undo"] is True

    progress, priorities = [], []
    original_run_task = server.vault.db.run_task

    def run_task(fn, *args, **kwargs):
        if fn is scrapheap_run_service._undo_segment_in_session:
            priorities.append(kwargs.get("priority"))
        return original_run_task(fn, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", run_task)
    response = _start_undo(client, restored["id"])
    assert response.status_code == 202, response.text
    undo = response.json()
    _between_slices(
        monkeypatch, undo["id"], lambda _task, _n: progress.append(_states(server, ids))
    )
    assert response.headers["Location"] == f"{RUNS_URL}/{undo['id']}"
    assert response.headers["Retry-After"] == "1"
    assert _varies_on_prefer(response)
    assert (undo["kind"], undo["status"], undo["target_run_id"]) == (
        "undo",
        "pending",
        restored["id"],
    )
    assert (undo["total"], undo["processed"], undo["undone_count"]) == (4, 0, 0)
    assert undo["segment_batch_ids"] == segments
    assert undo["request_id"].startswith("srv-")
    assert _states(server, ids) == ["live"] * 7

    _work(captured)

    live, heap = "live", "heap"
    assert progress == [
        [live] * 6 + [heap],
        [live] * 4 + [heap] * 3,
        [live] * 2 + [heap] * 5,
        [heap] * 7,
    ]
    assert priorities == [DBPriority.LOW] * 4
    ended = _get(client, undo["id"])
    assert (ended["status"], ended["active"], ended["error_code"]) == (
        "complete",
        False,
        None,
    )
    assert (ended["processed"], ended["undone_count"]) == (4, 7)
    assert ended["can_undo"] is False
    assert _get(client, restored["id"])["can_undo"] is False

    assert client.post(REDO_URL).status_code == 200
    assert _states(server, ids) == [live] * 2 + [heap] * 5


def test_an_undo_run_skips_a_segment_ctrl_z_already_took(env, captured, monkeypatch):
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    monkeypatch.setattr(scrapheap_run_service, "SEGMENT_PICTURE_CAP", 2)
    ids = _heap(server, 4)
    restored = _restore(client, ids)
    _work(captured)
    assert client.post(UNDO_URL).status_code == 200
    assert _states(server, ids) == ["live", "live", "heap", "heap"]
    assert _get(client, restored["id"])["can_undo"] is True

    undo = _undo(client, restored["id"])
    _work(captured)

    ended = _get(client, undo["id"])
    assert (ended["status"], ended["processed"], ended["undone_count"]) == (
        "complete",
        2,
        2,
    )
    assert _states(server, ids) == ["heap"] * 4


def test_an_undo_blocked_by_a_later_edit_keeps_what_it_undid_and_finishes_later(
    env, captured, monkeypatch
):
    """History must be undone from the top; an edit landing above the next
    segment stops the run, and the run says which entry to undo first."""
    client, server = env
    monkeypatch.setattr(scrapheap_run_service, "SLICE_MAX_PICTURES", 2)
    monkeypatch.setattr(scrapheap_run_service, "SEGMENT_PICTURE_CAP", 2)
    ids = _heap(server, 4)
    other = _heap(server, 1)
    restored = _restore(client, ids)
    _work(captured)
    undo = _undo(client, restored["id"])
    blocker = []

    def restore_elsewhere(_task, processed):
        if processed == 1:
            response = client.post(RESTORE_URL, json={"picture_ids": other})
            assert response.status_code == 200, response.text
            blocker.append(_newest_operation_id(server))

    _between_slices(monkeypatch, undo["id"], restore_elsewhere)
    _work(captured)

    ended = _get(client, undo["id"])
    assert (ended["status"], ended["active"], ended["error_code"]) == (
        "blocked",
        False,
        "history_moved_on",
    )
    assert (ended["processed"], ended["undone_count"]) == (1, 2)
    assert f"entry {blocker[0]})" in ended["error"]
    assert "OperationLogError" not in ended["error"]
    assert _states(server, ids + other) == ["live", "live", "heap", "heap", "live"]
    assert _get(client, restored["id"])["can_undo"] is True
    # The blocked run reads back as it ended after a restart, and the restore
    # it worked on can still be undone from what the rows hold.
    assert _restart(server) == 0
    assert _get(client, undo["id"]) == ended

    assert client.post(UNDO_URL).status_code == 200
    assert _states(server, other) == ["heap"]
    again = _undo(client, restored["id"])
    _work(captured)

    finished = _get(client, again["id"])
    assert (finished["status"], finished["processed"], finished["undone_count"]) == (
        "complete",
        2,
        2,
    )
    assert _states(server, ids) == ["heap"] * 4


def test_an_undo_refused_by_a_locked_set_stops_without_touching_the_segment(
    env, captured
):
    client, server = env
    ids = _heap(server, 2)
    restored = _restore(client, ids)
    _work(captured)
    _lock(server, ids[0])

    undo = _undo(client, restored["id"])
    _work(captured)

    ended = _get(client, undo["id"])
    assert (ended["status"], ended["error_code"]) == ("blocked", "undo_refused")
    assert ended["error"] == scrapheap_run_service.UNDO_REFUSED_MESSAGE
    assert (ended["processed"], ended["undone_count"]) == (0, 0)
    assert _states(server, ids) == ["live", "live"]


def test_only_a_restore_with_something_left_to_undo_can_be_undone(env, captured):
    client, server = env
    purged = _purge(client, _heap(server, 1))
    _work(captured)
    ctrl_z = _restore(client, _heap(server, 1))
    _work(captured)
    assert client.post(UNDO_URL).status_code == 200
    undone_target = _restore(client, _heap(server, 1))
    _work(captured)
    undo = _undo(client, undone_target["id"])
    _work(captured)

    answers = {
        name: _start_undo(client, run_id)
        for name, run_id in (
            ("purge", purged["id"]),
            ("undo", undo["id"]),
            ("ctrl_z", ctrl_z["id"]),
        )
    }
    # Any active run refuses an undo, its own target included. Asked last:
    # while a run is active that refusal comes before "nothing to undo".
    pending = _restore(client, _heap(server, 1))
    answers["active"] = _start_undo(client, pending["id"])

    def refusal(name):
        response = answers[name]
        detail = response.json()["detail"]
        run = detail["run"]
        return response.status_code, detail["code"], run["id"] if run else None

    assert refusal("purge") == (409, "not_a_restore_run", purged["id"])
    assert refusal("undo") == (409, "not_a_restore_run", undo["id"])
    assert refusal("ctrl_z") == (409, "nothing_to_undo", ctrl_z["id"])
    assert refusal("active") == (409, "scrapheap_run_active", pending["id"])
    assert _get(client, ctrl_z["id"])["can_undo"] is False
    assert len(captured) == 1
    _work(captured)


def test_a_replayed_undo_answers_200_and_a_reused_id_is_422(env, captured):
    client, server = env
    first_target = _restore(client, _heap(server, 1))
    _work(captured)
    undo = _undo(client, first_target["id"], request_id="example-undo-1")

    again = _start_undo(client, first_target["id"], request_id="example-undo-1")
    assert again.status_code == 200, again.text
    assert again.json()["id"] == undo["id"]
    _work(captured)
    second_target = _restore(client, _heap(server, 1))
    _work(captured)
    reused = _start_undo(client, second_target["id"], request_id="example-undo-1")
    assert reused.status_code == 422, reused.text
    assert reused.json()["detail"]["code"] == "request_id_reused"
    assert reused.json()["detail"]["run"]["id"] == undo["id"]
    unknown = _start_undo(client, first_target["id"], unknown=1)
    assert unknown.status_code == 422, unknown.text
    assert captured == []


def test_an_undo_answers_with_its_end_when_the_client_waits(env, monkeypatch):
    """A real task runner works both runs, so Prefer: wait sees each end."""
    client, server = env
    ids = _heap(server, 2)
    runner = TaskRunner(name="scrapheap-undo-wait-test", num_workers=1)
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    runner.start()
    try:
        restored = client.post(
            RUNS_URL,
            json={"kind": "restore", "picture_ids": ids},
            headers={"Prefer": "wait=2"},
        )
        assert restored.status_code == 200, restored.text
        undone = client.post(
            f"{RUNS_URL}/{restored.json()['id']}/undo", headers={"Prefer": "wait=2"}
        )
    finally:
        runner.stop()

    assert undone.status_code == 200, undone.text
    assert (undone.json()["status"], undone.json()["undone_count"]) == ("complete", 2)
    assert undone.headers["Preference-Applied"] == "wait=2"
    assert _varies_on_prefer(undone)
    assert _states(server, ids) == ["heap", "heap"]


# ── which run an event came from ─────────────────────────────────────────────


def test_every_run_event_names_its_run(env, captured, monkeypatch):
    """``run_id`` and ``run_kind`` let a tab tell a run's events from Ctrl+Z's.

    An undo puts pictures back in the Scrapheap, so its events are
    ``removed``, like a purge's.
    """
    client, server = env
    announced = _announced(monkeypatch, server)
    restored_ids, purged_ids = _heap(server, 2), _heap(server, 1)
    restored = _restore(client, restored_ids)
    _work(captured)
    undo = _undo(client, restored["id"])
    _work(captured)
    purged = _purge(client, purged_ids)
    _work(captured)

    assert [
        (event["run_id"], event["run_kind"], event["change_kind"], event["picture_ids"])
        for event in announced
    ] == [
        (restored["id"], "restore", "restored", restored_ids),
        (undo["id"], "undo", "removed", restored_ids),
        (purged["id"], "purge", "removed", purged_ids),
    ]


def test_the_broadcaster_puts_a_runs_identity_on_the_message():
    """Top level, beside ``change_kind``; absent from events no run sent."""
    sent = []

    class _Ws:
        async def send_json(self, payload):
            sent.append(payload)

    hub = WsBroadcasterMixin()
    hub._ws_clients_lock = threading.Lock()
    hub._ws_clients = [{"ws": _Ws(), "owner": True}]
    data = {
        "picture_ids": [7],
        "change_kind": "removed",
        "run_id": "example-run",
        "run_kind": "undo",
    }
    asyncio.run(hub._broadcast_ws_event(EventType.CHANGED_PICTURES, data))
    asyncio.run(
        hub._broadcast_ws_event(
            EventType.CHANGED_PICTURES, {"picture_ids": [7], "change_kind": "removed"}
        )
    )

    assert (sent[0]["run_id"], sent[0]["run_kind"]) == ("example-run", "undo")
    assert "run_id" not in sent[1] and "run_kind" not in sent[1]
