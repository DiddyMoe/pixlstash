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
* who may reach a run: the owner only. A READ token, a share link's token and
  the ComfyUI key are refused on every run route, by the gate and, with the
  gate rolled back, by the auth middleware's READ-token rules;
* the slices: their size, their LOW writer priority, and what each reports;
* cancel before the run starts and between slices, the server stopping before
  and during a run, and a slice that fails;
* restore History: one segment per run, a new one when another undoable edit
  lands between slices, the segment cap, and Ctrl+Z undoing one segment;
* the coalesced ``pictures_changed`` events, carrying the requesting tab;
* one run worked end to end by a real task runner.

One server serves the module, with background workers off; a test captures the
task the route submits and runs it on the test thread, which acts on the run
between slices through the same HTTP routes a second tab would use. Every test
starts with no runs, an empty History and an empty Scrapheap, because a run
with no ``picture_ids`` takes the whole Scrapheap.
"""

import gc
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete, select, update

from pixlstash import auth
from pixlstash.database import DBPriority
from pixlstash.db_models import Operation, Picture, PictureSet, PictureSetMember
from pixlstash.event_types import EventType
from pixlstash.server import Server
from pixlstash.services import scrapheap_run_service, scrapheap_service
from pixlstash.services.scrapheap_run_service import ScrapheapRunRegistry
from pixlstash.task_runner import TaskRunner
from pixlstash.tasks.scrapheap_run_task import ScrapheapRunTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from tests.authz_guard import assert_real_route, no_spa_fallback  # noqa: F401
from tests.utils import bearer_client

API = "/api/v1"
RUNS_URL = f"{API}/pictures/scrapheap/runs"
LATEST_URL = f"{RUNS_URL}/latest"
RUN_TEMPLATE = f"{RUNS_URL}/{{run_id}}"
PREVIEW_URL = f"{API}/pictures/scrapheap/delete-preview"
RESTORE_URL = f"{API}/pictures/scrapheap/restore"
UNDO_URL = f"{API}/operations/undo"
TOKEN_URL = f"{API}/users/me/token"
OWNER = {"username": "owner", "password": "example-owner-password"}
# What each layer answers a READ token with, so a refusal names its layer.
GATE_REFUSAL = "Owner-level (full, unscoped) access required"
MIDDLEWARE_REFUSAL = "Token is read-only"

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

    Background workers are off in this module, so the real ``submit_task``
    returns ``None`` and the route answers 503; capturing the task instead is
    the task runner taking it, minus the thread.
    """
    _client, server = env
    tasks: list[ScrapheapRunTask] = []

    def submit(task):
        tasks.append(task)
        return task.id

    monkeypatch.setattr(server.vault, "submit_task", submit)
    return tasks


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


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


def _restore(client, ids=None, **headers) -> dict:
    payload = {"kind": "restore"}
    if ids:
        payload["picture_ids"] = ids
    response = client.post(RUNS_URL, json=payload, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


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


def _credential(client, server, kind: str, picture_id: int) -> str:
    """A token of *kind*, minted afresh by the owner for one test.

    ``share`` is a share link's token, limited to *picture_id*. Each is shown
    to work where it is entitled to, so a refusal it later meets is the route
    refusing it and not a dead credential.
    """
    body = {
        "read": {"scope": "READ"},
        "share": {
            "scope": "READ",
            "resource_type": "picture",
            "resource_id": picture_id,
        },
        "comfyui": {"scope": "COMFYUI"},
    }[kind]
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
    """The refused requests changed nothing, and the owner reaches all four.

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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 3)
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

    assert [(chunk, protected, kw["priority"]) for chunk, protected, kw in slices] == [
        (ids[0:3], False, DBPriority.LOW),
        (ids[3:6], False, DBPriority.LOW),
        (ids[6:7], False, DBPriority.LOW),
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


def test_delete_forever_spares_a_picture_binned_after_the_preview(env):
    """The dialog's count is what Delete forever may destroy."""
    client, server = env
    ids = _heap(server, 2)
    token = _token(client, None)
    late = _heap(server, 1)

    response = client.request(
        "DELETE", f"{API}/pictures/scrapheap", json={"confirm_token": token}
    )

    assert response.status_code == 200, response.text
    assert response.json()["deleted_count"] == 2
    assert _states(server, ids) == [None, None]
    assert _states(server, late) == ["heap"]


def test_a_purge_run_spares_a_picture_binned_after_the_preview(env, captured):
    """A run started with the token is bound by the same preview."""
    client, server = env
    ids = _heap(server, 2)
    token = _token(client, None)
    late = _heap(server, 1)

    response = client.post(RUNS_URL, json={"kind": "purge", "confirm_token": token})
    assert response.status_code == 202, response.text
    assert response.json()["total"] == 2
    _work(captured)

    assert _get(client, response.json()["id"])["deleted_count"] == 2
    assert _states(server, ids) == [None, None]
    assert _states(server, late) == ["heap"]


def test_empty_run_spares_a_picture_binned_after_it_started(env, captured, monkeypatch):
    """Each slice applies the bound again: a picture restored and binned again
    between two slices was not in the dialog, though its id is in the run."""
    client, server = env
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
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
    assert (ended["status"], ended["deleted_count"]) == ("complete", 1)
    assert _states(server, ids) == [None, "heap"]


def test_an_undo_that_bins_a_picture_after_the_preview_leaves_it_alone(env, captured):
    """Undoing a restore bins the picture now, so the earlier preview misses it."""
    client, server = env
    ids = _heap(server, 2)
    rescued = client.post(RESTORE_URL, json={"picture_ids": ids[1:]})
    assert rescued.status_code == 200, rescued.text
    token = _token(client, None)
    undo = client.post(UNDO_URL)
    assert undo.status_code == 200, undo.text
    assert _states(server, ids) == ["heap", "heap"]

    response = client.post(RUNS_URL, json={"kind": "purge", "confirm_token": token})
    assert response.status_code == 202, response.text
    _work(captured)

    assert _get(client, response.json()["id"])["deleted_count"] == 1
    assert _states(server, ids) == [None, "heap"]


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


def test_a_stopping_task_runner_is_a_503_that_starts_nothing(env, monkeypatch):
    client, server = env
    ids = _heap(server, 1)
    runner = TaskRunner(name="scrapheap-run-test-stopped")
    runner.stop()
    monkeypatch.setattr(server.vault, "_task_runner", runner)

    response = client.post(RUNS_URL, json={"kind": "restore"})
    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "scrapheap_runs_unavailable"
    assert "stopping" in detail["message"]
    assert client.get(LATEST_URL).json() == {"run": None}
    assert _states(server, ids) == ["heap"]


def test_an_unknown_run_is_a_404_on_every_run_route(env):
    client, _server = env
    for response in (
        client.get(f"{RUNS_URL}/{uuid.uuid4().hex}"),
        client.post(f"{RUNS_URL}/{uuid.uuid4().hex}/cancel"),
    ):
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"] == "scrapheap_run_not_found"


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


@pytest.mark.parametrize("kind", ["read", "share", "comfyui"])
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


@pytest.mark.parametrize("kind", ["read", "share"])
def test_the_gate_alone_refuses_a_read_token_on_every_run_route(
    env, captured, monkeypatch, kind
):
    """The middleware's READ-token rules are switched off, to measure the gate.

    Those rules answer first: a READ token's POST is refused as read-only
    before routing, and both GETs are on the READ-token belt. With them off,
    the four ``OWNER_ONLY`` policies are all that stands between a READ token
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


@pytest.mark.parametrize("kind", ["read", "share"])
def test_the_middleware_alone_refuses_a_read_token_on_every_run_route(
    env, captured, monkeypatch, kind
):
    """``AUTHZ_GATE_ENFORCING = False``, the documented rollback.

    Both POSTs are refused as read-only. A run by id is on the READ-token belt
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
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
    env, captured, monkeypatch
):
    client, server = env
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
    ids = _heap(server, 3)
    original = scrapheap_service.purge_scrapheap_pictures
    calls = []

    def second_slice_fails(vault, chunk, include_protected, **kwargs):
        calls.append(list(chunk))
        if len(calls) == 2:
            raise OSError("example disk failure")
        return original(vault, chunk, include_protected, **kwargs)

    monkeypatch.setattr(
        scrapheap_service, "purge_scrapheap_pictures", second_slice_fails
    )
    run = _purge(client, ids)

    with pytest.raises(OSError, match="example disk failure"):
        _work(captured)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("failed", "unexpected_error")
    assert "example disk failure" in ended["error"]
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
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


# ── restore History ──────────────────────────────────────────────────────────


def test_a_restore_run_records_one_segment_that_one_undo_puts_back(
    env, captured, monkeypatch
):
    """Each slice is the one restore path at LOW priority, announcing nothing.

    The run coalesces the slices' ``restored`` events itself, so a slice that
    announced too would send every picture twice.
    """
    client, server = env
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 2)
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 2)
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 2)
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 2)
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
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
    monkeypatch.setattr(ScrapheapRunTask, "NOTIFY_INTERVAL_S", 3600.0)
    announced = _announced(monkeypatch, server)
    ids = _heap(server, 3)
    _restore(client, ids, **{"X-Client-Id": "example-tab"})
    _work(captured)

    # Coalesced: one event for three slices, the final flush.
    assert announced == [
        {
            "picture_ids": ids,
            "origin_client_id": "example-tab",
            "source": "ui",
            "change_kind": "restored",
        }
    ]


def test_a_purge_run_announces_once_per_interval(env, captured, monkeypatch):
    client, server = env
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 1)
    monkeypatch.setattr(ScrapheapRunTask, "NOTIFY_INTERVAL_S", 0.0)
    announced = _announced(monkeypatch, server)
    ids = _heap(server, 3)
    _purge(client, ids)
    _work(captured)

    assert [event["picture_ids"] for event in announced] == [[pid] for pid in ids]
    assert {event["change_kind"] for event in announced} == {"removed"}
    assert {event["source"] for event in announced} == {"external"}


# ── end to end ───────────────────────────────────────────────────────────────


def test_a_real_task_runner_works_a_run_to_the_end(env, monkeypatch):
    """The route, the runner and the task, wired as the server wires them."""
    client, server = env
    monkeypatch.setattr(ScrapheapRunTask, "SLICE_PICTURES", 2)
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
