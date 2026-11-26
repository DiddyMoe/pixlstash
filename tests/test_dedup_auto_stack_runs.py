"""The background auto-stack run's lifecycle, end to end through the API.

``POST /dedup/auto-stack/runs`` records a run and answers 202; a background task
then stacks the exact tier a writer slice at a time and records History in
segments. These tests drive that task the way the planner does (finder, then
``task.run()``) and act on the run *between* slices, from the test thread,
through the same HTTP routes a second tab would use:

* the idempotency key: a replay returns the run it started, a reuse with other
  parameters is refused;
* one run at a time;
* cancel before it starts, and while it is running;
* an unrelated edit landing between slices, and the revert that has to stop at
  it rather than undo through it;
* a revert that stopped short saying whether its run still has anything left
  to undo, so a client offers to send it again only when that would act;
* the segment cap, which bounds what one Ctrl+Z (and one undo transaction) has
  to do;
* a slice that fails, the server stopping mid-run, and a restart finding a run
  its previous process left behind;
* the coalesced ``pictures_changed`` events a long run sends.

One server serves the module. Each test works in its own folder scope, so a
run only ever plans that test's pictures, and the run rows are wiped before
each test because only one run may be active at a time. Background workers are
off, so nothing runs a slice except the test; the server is told it has a
planner (:func:`_works_runs_by_hand`), since a server that really has none
refuses to record a run at all (tested on its own below).
"""

import gc
import json
import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import select

from pixlstash.database import DBPriority
from pixlstash.db_models import Operation, Picture, PictureStack
from pixlstash.db_models.dedup import DedupAutoStackRun
from pixlstash.db_models.operation import STATUS_APPLIED, STATUS_UNDONE
from pixlstash.server import Server
from pixlstash.services import dedup_auto_stack_service
from pixlstash.services import dedup_tier_service as tiers
from pixlstash.services import dedup_verdict_service
from pixlstash.services.dedup_tier_service import DedupScope, ScopeType, TierPolicy
from pixlstash.services.dedup_verdict_service import OP_TYPE_STACK
from pixlstash.tasks import dedup_auto_stack_task
from pixlstash.tasks.dedup_auto_stack_finder import DedupAutoStackFinder
from pixlstash.tasks.dedup_auto_stack_task import DedupAutoStackTask
from tests.authz_guard import no_spa_fallback  # noqa: F401

API = "/api/v1"
GROUPS_URL = f"{API}/dedup/groups"
STACK_URL = f"{API}/dedup/verdicts/stack"
UNDO_URL = f"{API}/operations/undo"
RUNS_URL = f"{API}/dedup/auto-stack/runs"
OWNER = {"username": "owner", "password": "example-owner-password"}
# What a failed run records, and an exception text it must never echo.
GENERIC_ERROR = "The server hit an unexpected error; the server log has the details."
LEAK = "example-injected failure at /home/me/library/a.png in SELECT picture"

# The SPA catch-all answers unmatched GETs with 200; see tests/authz_guard.py.
pytestmark = pytest.mark.usefixtures("no_spa_fallback")


def _run_url(run_id, action=None) -> str:
    url = f"{RUNS_URL}/{run_id}"
    return f"{url}/{action}" if action else url


def _works_runs_by_hand(server) -> None:
    """Let a no-worker server record runs; the tests work them by hand.

    Without a planner a recorded run would never start, so the service refuses
    one (503). These tests stand in for the planner, so they answer its
    question for it on this vault only.
    """
    server.vault.runs_background_work = lambda: True


def _write_config(config_path: str, **overrides) -> None:
    config = {"port": 8000, "disable_background_workers": True}
    if os.path.exists(config_path):
        with open(config_path) as fh:
            config = json.load(fh)
    config.update(overrides)
    with open(config_path, "w") as fh:
        fh.write(json.dumps(config))


@pytest.fixture(scope="module")
def env():
    """One owner-logged-in client and its server, for the whole module."""
    temp_dir = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
    config_path = os.path.join(temp_dir.name, "server-config.json")
    _write_config(config_path)
    force_cpu = Server.DEFAULT_FORCE_CPU
    Server.DEFAULT_FORCE_CPU = True
    server = Server(config_path)
    _works_runs_by_hand(server)
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
def _no_runs_left_over(env):
    """Start every test with no run rows: only one run may be active at once."""
    _client, server = env

    def wipe(session):
        for run in session.exec(select(DedupAutoStackRun)).all():
            session.delete(run)
        session.commit()

    _run(server, wipe)
    assert _run(server, dedup_auto_stack_service.latest_run_in_session) is None
    yield


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _pairs(server, count: int) -> tuple[dict, list[list[int]]]:
    """Insert *count* exact pairs under a folder of their own and scan it.

    Returns the folder scope body for the run routes and the pairs' picture ids,
    in insertion order.
    """
    tag = uuid.uuid4().hex[:8]
    folder = f"/vault/{tag}"

    def insert(session):
        pairs = []
        for index in range(count):
            pair = []
            for member in range(2):
                picture = Picture(
                    file_path=f"{folder}/{index}_{member}.png",
                    format="png",
                    width=100,
                    height=100,
                    size_bytes=500 + index,
                    pixel_sha=f"{tag}-{index}",
                )
                session.add(picture)
                session.flush()
                pair.append(int(picture.id))
            pairs.append(pair)
        session.commit()
        return pairs

    pairs = _run(server, insert)
    _run(
        server,
        tiers.run_scan_now_in_session,
        TierPolicy(),
        DedupScope(ScopeType.FOLDER, folder),
    )
    return {"scope_type": "folder", "scope_id": folder}, pairs


def _stacked(server, pairs: list[list[int]]) -> list[bool]:
    """Per pair, whether both pictures now share one stack."""

    def read(session):
        result = []
        for pair in pairs:
            stack_ids = {session.get(Picture, pid).stack_id for pid in pair}
            result.append(len(stack_ids) == 1 and None not in stack_ids)
        return result

    return _run(server, read)


def _stack_ops(server, batch_id: str, status=None) -> int:
    """How many ``dedup.stack`` operations *batch_id* recorded (one per group)."""

    def count(session):
        query = select(Operation.id).where(
            Operation.batch_id == batch_id, Operation.op_type == OP_TYPE_STACK
        )
        if status is not None:
            query = query.where(Operation.status == status)
        return len(session.exec(query).all())

    return _run(server, count)


def _work(server) -> None:
    """Work every pending run as the planner would: finder, then the task."""
    finder = DedupAutoStackFinder(server.vault)
    while (task := finder.find_task()) is not None:
        task.run()


def _between_slices(monkeypatch, run_id: int, callback) -> None:
    """Call ``callback(task, processed)`` after each slice of run *run_id*.

    ``_set_progress`` is the task's own per-slice readout, so this runs on the
    test thread exactly where the writer is free between two slices: an HTTP call
    from here is what a second tab could do at that moment. Only the named run's
    task calls back; a revert worked later in the same test does not.
    """
    original = DedupAutoStackTask._set_progress

    def wrapped(self, processed, total):
        original(self, processed, total)
        if processed > 0 and self.params["run_id"] == run_id:
            callback(self, processed)

    monkeypatch.setattr(DedupAutoStackTask, "_set_progress", wrapped)


def _start(client, scope: dict, **body) -> dict:
    response = client.post(RUNS_URL, json={"scope": scope, **body})
    assert response.status_code == 202, response.text
    return response.json()


def _get(client, run_id) -> dict:
    response = client.get(_run_url(run_id))
    assert response.status_code == 200, response.text
    return response.json()


def _revert(client, server, run_id) -> dict:
    response = client.post(_run_url(run_id, "revert"), json={})
    assert response.status_code == 202, response.text
    _work(server)
    return _get(client, response.json()["id"])


def _signature_in(client, scope: dict) -> str:
    groups = client.get(GROUPS_URL, params=scope).json()["groups"]
    assert len(groups) == 1, groups
    return groups[0]["signature"]


# ── the request: idempotency and one run at a time ───────────────────────────


def test_a_replayed_request_id_returns_the_run_it_started(env):
    """A retry after a lost response must find the run, never start a second.

    That retry is the gesture the old synchronous route invited: the request
    timed out after 60 s with the outcome unknown, so the only safe move was a
    second click that would have stacked again.
    """
    client, server = env
    scope, pairs = _pairs(server, 2)
    body = {"scope": scope, "limit": 5, "request_id": "example-replay-1"}
    first = client.post(RUNS_URL, json=body)
    assert first.status_code == 202, first.text
    run_id = first.json()["id"]

    again = client.post(RUNS_URL, json=body)
    assert again.status_code == 200, again.text
    assert again.json()["id"] == run_id
    assert again.headers["Location"] == first.headers["Location"] == _run_url(run_id)
    assert again.headers["Retry-After"] == "2"

    for changed in ({**body, "limit": 6}, {**body, "scope": {"scope_type": "global"}}):
        reused = client.post(RUNS_URL, json=changed)
        assert reused.status_code == 422, reused.text
        assert reused.json()["detail"]["code"] == "request_id_reused"
        assert reused.json()["detail"]["run"]["id"] == run_id

    malformed = client.post(RUNS_URL, json={**body, "request_id": "has space"})
    assert malformed.status_code == 422, malformed.text
    assert _run(server, lambda s: s.exec(select(DedupAutoStackRun.id)).all()) == [
        run_id
    ]

    _work(server)
    replayed = client.post(RUNS_URL, json=body)
    assert replayed.status_code == 200, replayed.text
    assert "Retry-After" not in replayed.headers
    assert replayed.json()["status"] == "complete"
    assert replayed.json()["applied"] == 2
    assert _stacked(server, pairs) == [True, True]


def test_a_client_on_another_origin_can_read_the_poll_interval(env):
    """The dev server is another origin, and Retry-After is not safelisted.

    Without `Access-Control-Expose-Headers` the browser hides the header from
    the client, which then polls at a guess instead of at the server's hint.
    """
    client, server = env
    scope, _pairs_made = _pairs(server, 1)
    run_id = _start(client, scope)["id"]

    allowed = client.get(_run_url(run_id), headers={"Origin": "http://localhost:5173"})
    assert allowed.headers["Retry-After"] == "2"
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    exposed = allowed.headers["access-control-expose-headers"].lower().split(",")
    assert "retry-after" in [name.strip() for name in exposed]

    # Exposing a header widens nothing: a refused origin still reads no response.
    refused = client.get(_run_url(run_id), headers={"Origin": "https://evil.example"})
    assert refused.status_code == 200, refused.text
    assert "access-control-allow-origin" not in refused.headers
    _work(server)


def test_only_one_run_may_be_active(env):
    client, server = env
    scope, _ = _pairs(server, 1)
    first = _start(client, scope)

    second = client.post(RUNS_URL, json={"scope": scope})
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["code"] == "auto_stack_run_active"
    assert second.json()["detail"]["run"]["id"] == first["id"]

    # Reverting a run that is still working is refused the same way.
    revert = client.post(_run_url(first["id"], "revert"), json={})
    assert revert.status_code == 409, revert.text
    assert revert.json()["detail"]["code"] == "auto_stack_run_active"

    _work(server)
    assert _get(client, first["id"])["status"] == "complete"
    assert client.post(RUNS_URL, json={"scope": scope}).status_code == 202


def test_a_server_with_no_background_work_records_no_run(env, monkeypatch):
    """Recorded anyway, a run would stay pending and refuse every later one.

    No planner works it on such a server, and the start-up reconcile that
    would close it never runs there either. The real predicate answers here.
    """
    client, server = env
    monkeypatch.delattr(server.vault, "runs_background_work")
    assert server.vault.runs_background_work() is False
    scope, pairs = _pairs(server, 1)

    refused = client.post(RUNS_URL, json={"scope": scope})
    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"]["code"] == "auto_stack_runs_unavailable"
    assert refused.json()["detail"]["run"] is None
    assert "Retry-After" not in refused.headers
    assert _run(server, dedup_auto_stack_service.latest_run_in_session) is None

    monkeypatch.undo()
    run = _start(client, scope)
    _work(server)
    assert _get(client, run["id"])["status"] == "complete"
    monkeypatch.delattr(server.vault, "runs_background_work")

    reverted = client.post(_run_url(run["id"], "revert"), json={})
    assert reverted.status_code == 503, reverted.text
    assert reverted.json()["detail"]["code"] == "auto_stack_runs_unavailable"
    assert (
        _run(server, dedup_auto_stack_service.latest_run_in_session)["id"]
        == (run["id"])
    )
    assert _stacked(server, pairs) == [True]


def _seed_ended_runs(server, count: int) -> list[int]:
    """Insert *count* finished whole-vault runs; their ids, oldest first."""

    def insert(session):
        rows = [
            DedupAutoStackRun(request_id=f"test-seed-{index}", status="complete")
            for index in range(count)
        ]
        session.add_all(rows)
        session.commit()
        return [int(row.id) for row in rows]

    return _run(server, insert)


def _run_ids(server) -> list[int]:
    return _run(
        server,
        lambda s: s.exec(
            select(DedupAutoStackRun.id).order_by(DedupAutoStackRun.id)
        ).all(),
    )


def test_a_new_run_keeps_only_the_newest_ended_rows(env):
    """Run rows were never deleted, so the table grew by a row a click for ever.

    Each new run prunes the ended rows past the newest HISTORY_KEPT in its own
    insert. A refused request prunes nothing, the apply run a revert names is
    kept however old it is, and a pruned run's request_id may be used again.
    """
    client, server = env
    kept = dedup_auto_stack_service.HISTORY_KEPT
    scope, pairs = _pairs(server, 1)
    applied = _start(client, scope)["id"]
    _work(server)
    assert _stacked(server, pairs) == [True]
    seeded = _seed_ended_runs(server, kept + 3)

    reused = client.post(RUNS_URL, json={"scope": scope, "request_id": "test-seed-0"})
    assert reused.status_code == 422, reused.text
    assert reused.json()["detail"]["code"] == "request_id_reused"
    assert _run_ids(server) == [applied, *seeded]

    # The apply run is now the oldest of kept + 4 ended rows, and still revertable.
    reverted = _revert(client, server, applied)
    assert reverted["status"] == "complete", reverted
    assert _stacked(server, pairs) == [False]
    assert _run_ids(server) == [applied, *seeded[-kept:], reverted["id"]]

    # Pruned, the id is free: the same request starts a new run, and this time
    # nothing keeps the apply run.
    again = client.post(RUNS_URL, json={"scope": scope, "request_id": "test-seed-0"})
    assert again.status_code == 202, again.text
    assert _run_ids(server) == [
        *seeded[-(kept - 1) :],
        reverted["id"],
        again.json()["id"],
    ]
    assert client.get(_run_url(applied)).status_code == 404
    _work(server)


# ── cancel ───────────────────────────────────────────────────────────────────


def test_cancelling_a_queued_run_stops_it_before_anything_is_stacked(env):
    client, server = env
    scope, pairs = _pairs(server, 2)
    run = _start(client, scope)

    cancelled = client.post(_run_url(run["id"], "cancel"))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["active"] is False
    assert "Retry-After" not in cancelled.headers

    assert DedupAutoStackFinder(server.vault).find_task() is None
    assert _stacked(server, pairs) == [False, False]
    again = client.post(_run_url(run["id"], "cancel"))
    assert again.status_code == 200, again.text
    assert again.json()["status"] == "cancelled"
    assert again.json()["can_revert"] is False


def test_cancelling_a_running_run_stops_it_at_the_next_slice(env, monkeypatch):
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    scope, pairs = _pairs(server, 3)
    run = _start(client, scope)
    answers = []

    def cancel_once(_task, processed):
        if processed == 1:
            answers.append(client.post(_run_url(run["id"], "cancel")))

    _between_slices(monkeypatch, run["id"], cancel_once)
    _work(server)

    assert len(answers) == 1
    answer = answers[0]
    assert answer.status_code == 200, answer.text
    # Still running when the cancel is accepted: it stops at the slice boundary.
    assert answer.json()["status"] == "running"
    assert answer.json()["cancel_requested"] is True
    assert answer.headers["Retry-After"] == "2"

    ended = _get(client, run["id"])
    assert ended["status"] == "cancelled"
    assert (ended["total"], ended["processed"], ended["applied"]) == (3, 1, 1)
    assert ended["can_revert"] is True
    # Nothing is rolled back: the first slice's group stays stacked.
    assert sum(_stacked(server, pairs)) == 1


# ── History: an edit between slices, and the segment cap ─────────────────────


def test_an_edit_between_slices_starts_a_new_segment_and_blocks_the_revert(
    env, monkeypatch
):
    """A batch must stay contiguous at the top of History to be undone.

    A stack verdict the user makes in another tab while the run works lands in
    History between two slices. Recording the next slice into the same batch
    would split it around that verdict, so the run starts a new segment; the
    revert then undoes the newer segment and stops at the verdict instead of
    undoing through it, until the user has undone the verdict themselves.
    """
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    scope, pairs = _pairs(server, 3)
    other_scope, other_pairs = _pairs(server, 1)
    other_signature = _signature_in(client, other_scope)
    run = _start(client, scope)
    answers = []

    def stack_elsewhere(_task, processed):
        if processed == 1:
            answers.append(client.post(STACK_URL, json={"signature": other_signature}))

    _between_slices(monkeypatch, run["id"], stack_elsewhere)
    _work(server)

    assert [answer.status_code for answer in answers] == [200]
    applied = _get(client, run["id"])
    assert applied["status"] == "complete"
    assert applied["applied"] == 3
    segments = applied["segment_batch_ids"]
    assert len(segments) == 2
    assert [_stack_ops(server, segment) for segment in segments] == [1, 2]
    assert _stacked(server, pairs + other_pairs) == [True, True, True, True]

    blocked = _revert(client, server, run["id"])
    assert blocked["status"] == "blocked"
    assert blocked["error_code"] == "history_moved_on"
    assert (blocked["total"], blocked["processed"], blocked["applied"]) == (2, 1, 2)
    assert _get(client, run["id"])["can_revert"] is True
    # Blocked, the undo can be sent again (it says so before the edit is undone).
    assert blocked["target_can_revert"] is True
    # The verdict and the run's first segment are untouched.
    assert _stack_ops(server, segments[0], STATUS_APPLIED) == 1
    assert _stacked(server, other_pairs) == [True]

    undo = client.post(UNDO_URL)
    assert undo.status_code == 200, undo.text
    assert _stacked(server, other_pairs) == [False]

    finished = _revert(client, server, run["id"])
    assert finished["status"] == "complete"
    assert (finished["superseded"], finished["applied"]) == (1, 1)
    assert _stacked(server, pairs) == [False, False, False]
    assert _get(client, run["id"])["can_revert"] is False
    assert _get(client, blocked["id"])["target_can_revert"] is False


@pytest.mark.parametrize(
    ("cap", "slice_groups", "expected"),
    [
        (2, 1, [2, 2, 1]),
        # Checked before the slice, not after the last one: a slice that would
        # overflow the cap starts a new segment, so none ever holds more than
        # the cap (filling up to it after, the old rule gave [4, 1]).
        (3, 2, [2, 3]),
    ],
)
def test_no_segment_holds_more_groups_than_the_cap(
    env, monkeypatch, cap, slice_groups, expected
):
    """The cap bounds one Ctrl+Z, and the one writer transaction undoing it."""
    client, server = env
    monkeypatch.setattr(dedup_auto_stack_service, "SEGMENT_GROUP_CAP", cap)
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", slice_groups)
    scope, pairs = _pairs(server, 5)
    run = _start(client, scope)
    _work(server)

    ended = _get(client, run["id"])
    assert ended["status"] == "complete"
    segments = ended["segment_batch_ids"]
    assert len(set(segments)) == len(segments)
    assert all(segment.startswith("srv-") for segment in segments)
    assert [_stack_ops(server, segment) for segment in segments] == expected

    # Ctrl+Z takes back exactly the newest segment.
    undo = client.post(UNDO_URL)
    assert undo.status_code == 200, undo.text
    assert _stack_ops(server, segments[-1], STATUS_UNDONE) == expected[-1]
    assert [_stack_ops(server, s, STATUS_APPLIED) for s in segments[:-1]] == (
        expected[:-1]
    )
    assert sum(_stacked(server, pairs)) == sum(expected[:-1])

    reverted = _revert(client, server, run["id"])
    assert reverted["status"] == "complete"
    assert reverted["superseded"] == 1
    assert reverted["applied"] == sum(expected[:-1])
    assert _stacked(server, pairs) == [False] * 5


def _stack_together(server, picture_ids) -> None:
    """Put *picture_ids* into one stack of their own, before any run."""

    def build(session):
        stack = PictureStack(name="example-shared")
        session.add(stack)
        session.flush()
        for position, picture_id in enumerate(picture_ids):
            picture = session.get(Picture, picture_id)
            picture.stack_id = int(stack.id)
            picture.stack_position = position
            session.add(picture)
        session.commit()

    _run(server, build)


def test_a_stack_shared_by_groups_in_two_slices_is_counted_once(env, monkeypatch):
    """Two groups, one per slice, both fold one stack: four pictures, not 3 + 4.

    Each slice counts its own pictures distinctly; the run counts them across
    slices, and its revert across segments.
    """
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    # One segment per group, so the revert undoes the shared stack twice.
    monkeypatch.setattr(dedup_auto_stack_service, "SEGMENT_GROUP_CAP", 1)
    scope, pairs = _pairs(server, 2)
    _stack_together(server, [pairs[0][0], pairs[1][0]])
    run = _start(client, scope)
    _work(server)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["processed"], ended["applied"]) == (
        "complete",
        2,
        2,
    )
    assert len(ended["segment_batch_ids"]) == 2
    assert ended["pictures"] == 4

    reverted = _revert(client, server, run["id"])
    assert (reverted["status"], reverted["applied"]) == ("complete", 2)
    assert reverted["pictures"] == 4


def _announcements(monkeypatch) -> list[list[int]]:
    announced: list[list[int]] = []
    monkeypatch.setattr(
        dedup_auto_stack_task,
        "notify_pictures_changed",
        lambda _vault, picture_ids, *_rest: announced.append(sorted(picture_ids)),
    )
    return announced


@pytest.mark.parametrize("where", ["group", "task"])
def test_a_run_failing_mid_way_announces_what_committed(env, monkeypatch, where):
    """A run that fails in its second slice still sends its change event.

    A failed run ends like a finished or interrupted one: the pictures its
    committed slices stacked go out in a ``pictures_changed`` event, so other
    tabs show them. ``group``: a group fails inside the slice (the service
    closes the row). ``task``: the slice call itself raises (the task closes it).
    """
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    # Never on the interval: only the closing flush announces.
    monkeypatch.setattr(DedupAutoStackTask, "NOTIFY_INTERVAL_S", 1e9)
    announced = _announcements(monkeypatch)
    scope, pairs = _pairs(server, 3)
    calls = []
    if where == "group":
        original = dedup_verdict_service.apply_stack_verdict_in_session

        def second_group_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError(LEAK)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dedup_verdict_service, "apply_stack_verdict_in_session", second_group_fails
        )
    else:
        original = dedup_auto_stack_service.apply_slice_in_session

        def second_slice_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError(LEAK)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dedup_auto_stack_service, "apply_slice_in_session", second_slice_fails
        )
    run = _start(client, scope)
    if where == "group":
        _work(server)
    else:
        with pytest.raises(RuntimeError, match="example-injected"):
            _work(server)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["applied"]) == ("failed", 1)
    stacked = _stacked(server, pairs)
    assert sum(stacked) == 1
    assert announced == [pairs[stacked.index(True)]]


# ── revert ───────────────────────────────────────────────────────────────────


def test_a_revert_undoes_the_whole_run_and_only_once(env):
    client, server = env
    scope, pairs = _pairs(server, 3)
    run = _start(client, scope)
    _work(server)
    assert _stacked(server, pairs) == [True, True, True]

    revert = _revert(client, server, run["id"])
    assert revert["kind"] == "revert"
    assert revert["target_run_id"] == run["id"]
    assert revert["status"] == "complete"
    assert revert["applied"] == 3
    assert _stacked(server, pairs) == [False, False, False]
    assert _get(client, run["id"])["can_revert"] is False
    # Finished, the undo has nothing left to send again.
    assert revert["target_can_revert"] is False

    nothing = client.post(_run_url(run["id"], "revert"), json={})
    assert nothing.status_code == 409, nothing.text
    assert nothing.json()["detail"]["code"] == "nothing_to_revert"

    of_a_revert = client.post(_run_url(revert["id"], "revert"), json={})
    assert of_a_revert.status_code == 409, of_a_revert.text
    assert of_a_revert.json()["detail"]["code"] == "not_an_apply_run"


def _stopped_short_revert(client, server, monkeypatch) -> tuple[dict, dict, list]:
    """An apply run of three one-group segments, and an undo stopped after one.

    Returns the apply run, the revert as its ``202`` answered (still pending)
    and the pairs. The revert undoes the newest segment, then is cancelled
    between slices, so the two older segments stay stacked.
    """
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    monkeypatch.setattr(dedup_auto_stack_service, "SEGMENT_GROUP_CAP", 1)
    scope, pairs = _pairs(server, 3)
    run = _start(client, scope)
    _work(server)
    assert len(_get(client, run["id"])["segment_batch_ids"]) == 3

    response = client.post(_run_url(run["id"], "revert"), json={})
    assert response.status_code == 202, response.text
    revert = response.json()

    def cancel_once(_task, processed):
        if processed == 1:
            cancelled = client.post(_run_url(revert["id"], "cancel"))
            assert cancelled.status_code == 200, cancelled.text

    _between_slices(monkeypatch, revert["id"], cancel_once)
    _work(server)
    return run, revert, pairs


def test_a_stopped_short_revert_says_whether_its_run_has_anything_left(
    env, monkeypatch
):
    """``target_can_revert``: would sending this undo again undo anything?

    The banner holding a revert that stopped short offers to finish it. The
    revert's own ``can_revert`` cannot say so (a revert is not revertable), so
    the revert carries its target's answer, read the same way, on every read.
    """
    client, server = env
    run, pending, pairs = _stopped_short_revert(client, server, monkeypatch)
    # Going, it offers nothing: the run is still undoing.
    assert pending["active"] is True
    assert pending["target_can_revert"] is False

    stopped = _get(client, pending["id"])
    assert (stopped["status"], stopped["processed"]) == ("cancelled", 1)
    assert _stacked(server, pairs) == [True, True, False]
    assert stopped["target_can_revert"] is True
    assert stopped["can_revert"] is False
    assert _get(client, run["id"])["can_revert"] is True

    # The rest undone by hand: nothing is left, and the revert route agrees.
    for _segment in range(2):
        undo = client.post(UNDO_URL)
        assert undo.status_code == 200, undo.text
    assert _stacked(server, pairs) == [False, False, False]
    assert _get(client, pending["id"])["target_can_revert"] is False
    nothing = client.post(_run_url(run["id"], "revert"), json={})
    assert nothing.status_code == 409, nothing.text
    assert nothing.json()["detail"]["code"] == "nothing_to_revert"


def test_a_revert_whose_run_was_pruned_has_nothing_left(env, monkeypatch):
    """Its apply run's row gone, the undo cannot be sent again (404)."""
    client, server = env
    kept = dedup_auto_stack_service.HISTORY_KEPT
    run, pending, _pairs_left = _stopped_short_revert(client, server, monkeypatch)
    assert _get(client, pending["id"])["target_can_revert"] is True

    # kept - 1 newer rows, then a new run's insert prunes past the newest kept:
    # the revert is the oldest row kept, its apply run the newest one pruned.
    _seed_ended_runs(server, kept - 1)
    scope, _pairs_new = _pairs(server, 1)
    _start(client, scope)
    assert client.get(_run_url(run["id"])).status_code == 404

    assert _get(client, pending["id"])["target_can_revert"] is False
    gone = client.post(_run_url(run["id"], "revert"), json={})
    assert gone.status_code == 404, gone.text
    _work(server)


def _statements(server, run_id: int) -> int:
    """SQL statements ``serialize_run`` issues for run *run_id*, read fresh."""

    def count(session):
        run = session.get(DedupAutoStackRun, run_id)
        bind = session.get_bind()
        statements = []

        def record(_conn, _cursor, statement, *_rest):
            statements.append(statement)

        event.listen(bind, "before_cursor_execute", record)
        try:
            dedup_auto_stack_service.serialize_run(session, run)
        finally:
            event.remove(bind, "before_cursor_execute", record)
        return len(statements)

    return _run(server, count)


def test_a_reverts_flag_costs_one_lookup_more_than_its_runs(env, monkeypatch):
    """The revert's answer is its target's, plus one primary-key read.

    Not a query per segment or per run: the target's flag is the same bounded
    probe an apply run makes, and the revert adds the read of the target row.
    """
    client, server = env
    run, pending, _pairs_left = _stopped_short_revert(client, server, monkeypatch)
    assert _statements(server, pending["id"]) == _statements(server, run["id"]) + 1


def test_an_unknown_run_is_a_404_on_every_run_route(env):
    client, _server = env
    for response in (
        client.get(_run_url(999_999)),
        client.post(_run_url(999_999, "cancel")),
        client.post(_run_url(999_999, "revert"), json={}),
    ):
        assert response.status_code == 404, response.text
        # The run's own code, so the route resolved: not a 404 for a bad path.
        assert response.json()["detail"]["code"] == "auto_stack_run_not_found"


# ── failure, shutdown and restart ────────────────────────────────────────────


def _leaks(view: dict) -> bool:
    """Whether a run view carries any part of :data:`LEAK`."""
    text = json.dumps(view)
    return "example-injected" in text or "/home/me" in text or "SELECT" in text


def test_a_failing_group_fails_the_run_and_keeps_what_committed(
    env, monkeypatch, caplog
):
    """The row and its failure sample record a generic message; the log the cause.

    The exception text can carry a file path or SQL, and the run view is served
    to every owner session, so neither ``error`` nor ``failures`` may echo it.
    """
    client, server = env
    scope, pairs = _pairs(server, 3)
    original = dedup_verdict_service.apply_stack_verdict_in_session
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError(LEAK)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dedup_verdict_service, "apply_stack_verdict_in_session", fail_second
    )
    run = _start(client, scope)
    _work(server)

    ended = _get(client, run["id"])
    assert ended["status"] == "failed"
    assert ended["error_code"] == "unexpected_error"
    assert ended["error"] == GENERIC_ERROR
    assert (ended["processed"], ended["applied"], ended["failed"]) == (2, 1, 1)
    assert ended["failures"][0]["outcome"] == "failed"
    assert ended["failures"][0]["signature"] == calls[1]
    assert ended["failures"][0]["error"] == GENERIC_ERROR
    assert not _leaks(ended)
    assert any(LEAK in record.getMessage() for record in caplog.records)
    assert ended["can_revert"] is True
    assert sum(_stacked(server, pairs)) == 1


def test_a_task_failing_outside_a_slice_records_a_generic_error(
    env, monkeypatch, caplog
):
    """A defect outside the group loop (here the plan) closes the row the same way.

    That is the task's own failure path rather than the slice's, and it records
    the same generic message, with the cause in the log only.
    """
    client, server = env
    scope, _pairs_ = _pairs(server, 2)

    def plan_fails(*_args, **_kwargs):
        raise RuntimeError(LEAK)

    monkeypatch.setattr(dedup_auto_stack_task, "plan_auto_stack_in_session", plan_fails)
    run = _start(client, scope)
    with pytest.raises(RuntimeError, match="example-injected"):
        _work(server)

    ended = _get(client, run["id"])
    assert (ended["status"], ended["error_code"]) == ("failed", "unexpected_error")
    assert ended["error"] == GENERIC_ERROR
    assert not _leaks(ended)
    assert any(
        record.exc_info and LEAK in str(record.exc_info[1])
        for record in caplog.records
        if f"run {run['id']} failed" in record.getMessage()
    )


def test_a_server_stopping_mid_run_closes_it_interrupted(env, monkeypatch):
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    announced = []
    monkeypatch.setattr(
        dedup_auto_stack_task,
        "notify_pictures_changed",
        lambda _vault, picture_ids, *_rest: announced.append(sorted(picture_ids)),
    )
    scope, pairs = _pairs(server, 3)
    run = _start(client, scope)

    def stop_once(task, processed):
        if processed == 1:
            # What the task runner does to its tasks when the server stops.
            task.on_cancel()

    _between_slices(monkeypatch, run["id"], stop_once)
    _work(server)

    ended = _get(client, run["id"])
    assert ended["status"] == "interrupted"
    assert ended["error_code"] == "server_restarted"
    assert (ended["processed"], ended["applied"]) == (1, 1)
    assert ended["can_revert"] is True
    stacked = _stacked(server, pairs)
    assert sum(stacked) == 1
    # What did commit is still announced, so other tabs show it.
    assert announced == [pairs[stacked.index(True)]]


def test_a_queued_run_left_by_a_previous_process_is_interrupted(env):
    client, server = env
    scope, pairs = _pairs(server, 1)
    run = _start(client, scope)

    reconciled = _run(
        server, dedup_auto_stack_service.interrupt_orphaned_runs_in_session
    )
    assert reconciled == 1
    ended = _get(client, run["id"])
    assert ended["status"] == "interrupted"
    assert ended["error_code"] == "server_restarted"
    assert ended["active"] is False
    # Never resumed: the user decides whether to run again.
    assert DedupAutoStackFinder(server.vault).find_task() is None
    assert _stacked(server, pairs) == [False]


def test_a_restart_closes_a_run_the_previous_process_was_working():
    """The start-up reconcile is wired into the vault's start, not just written.

    A run the stopped process was working stays ``running`` on disk; with no
    task left to own it, it would read as in progress forever and refuse every
    new run as ``auto_stack_run_active``.
    """
    temp_dir = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
    config_path = os.path.join(temp_dir.name, "server-config.json")
    _write_config(config_path)
    force_cpu = Server.DEFAULT_FORCE_CPU
    Server.DEFAULT_FORCE_CPU = True
    try:
        before = Server(config_path)
        _works_runs_by_hand(before)
        try:
            client = TestClient(before.api)
            assert client.post(f"{API}/login", json=OWNER).status_code == 200
            started = client.post(RUNS_URL, json={})
            assert started.status_code == 202, started.text
            run_id = started.json()["id"]
            assert _run(before, dedup_auto_stack_service.start_run_in_session, run_id)
            assert _get(client, run_id)["status"] == "running"
        finally:
            before.close()

        # Workers on this time: the reconcile runs in Vault.start, which the
        # server's lifespan calls when workers are enabled.
        _write_config(config_path, disable_background_workers=False)
        after = Server(config_path)
        try:
            with TestClient(after.api) as client:
                assert client.post(f"{API}/login", json=OWNER).status_code == 200
                ended = _get(client, run_id)
                assert ended["status"] == "interrupted"
                assert ended["error_code"] == "server_restarted"
                assert client.post(RUNS_URL, json={}).status_code == 202
        finally:
            after.close()
    finally:
        Server.DEFAULT_FORCE_CPU = force_cpu
        temp_dir.cleanup()
        gc.collect()


# ── change events ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("interval", "expected_calls"),
    [
        # A long run announces on an interval, not once per group or slice;
        # here the interval never elapses, so one closing event carries it all.
        (1e9, 1),
        # With no interval every slice announces its own pictures.
        (0.0, 3),
    ],
)
def test_a_run_announces_its_pictures_coalesced_with_the_origin(
    env, monkeypatch, interval, expected_calls
):
    client, server = env
    monkeypatch.setattr(DedupAutoStackTask, "SLICE_GROUPS", 1)
    monkeypatch.setattr(DedupAutoStackTask, "NOTIFY_INTERVAL_S", interval)
    calls = []
    monkeypatch.setattr(
        dedup_auto_stack_task,
        "notify_pictures_changed",
        lambda _vault, picture_ids, origin, source: calls.append(
            (sorted(picture_ids), origin, source)
        ),
    )
    scope, pairs = _pairs(server, 3)
    started = client.post(
        RUNS_URL, json={"scope": scope}, headers={"X-Client-Id": "example-tab"}
    )
    assert started.status_code == 202, started.text
    _work(server)

    assert len(calls) == expected_calls
    assert {(origin, source) for _ids, origin, source in calls} == {
        ("example-tab", "ui")
    }
    announced = sorted(pid for picture_ids, _o, _s in calls for pid in picture_ids)
    assert announced == sorted(pid for pair in pairs for pid in pair)
