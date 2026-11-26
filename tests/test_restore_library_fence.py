"""A full restore closes library admission around the swap.

After a restore swaps the file, no write may land in it from work that began
against the old one, and every writer it paused runs again afterwards. The
writers that are not tasks already route through the
``LibraryGenerationCoordinator`` read lease, so the restore fences them there:

- **The restore** calls ``begin_switch`` (excluding its own request's lease),
  waits within one deadline shared with the task drain, and reopens admission
  on every exit: a new generation once the swap was attempted, the same one
  otherwise. A lease still held at the deadline refuses the restore with
  nothing changed.
- **The edit-copy import and the smart-set refresher** take a lease per pass,
  write through the leased vault, and wait while admission is SWITCHING. While
  no coherent library is open they wait a bounded number of times, then stop.
- **The Scrapheap registry** is reloaded in place, so a reference taken before
  the restore reads the restored runs.
- **A library switch** hands the coordinator to each vault it publishes.
"""

import json
import os
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pixlstash.db_models import Picture
from pixlstash.resources.arbiter import ResourceArbiter
from pixlstash.server import Server
from pixlstash.services import edit_copies
from pixlstash.services.edit_copies import (
    EditCopyWatcher,
    EditSessionDeferred,
    process_session,
)
from pixlstash.services.library_generation_coordinator import (
    LibraryGenerationCoordinator,
    LibraryNotReadyError,
    LibraryReadLease,
    SwitchState,
)
from pixlstash.services.restore import (
    BackgroundWorkStillRunningError,
    LibraryFence,
    LibraryStillInUseError,
    RestoreRefusedError,
)
from pixlstash.services.restore import full_restore
from pixlstash.services.scrapheap_run_service import (
    KIND_PURGE,
    ScrapheapRunRegistry,
)
from pixlstash.task_runner import TaskRunner
from pixlstash.utils.service import smart_set_refresher
from pixlstash.utils.service.library_lease import UNAVAILABLE_RETRIES
from pixlstash.utils.service.smart_set_refresher import SmartSetRefresher

API = "/api/v1"


class _NoCard:
    """A GPU ledger with no card, as ``tests/test_restore_drain.py`` uses."""

    def devices(self, max_age_s=1.0):
        return []

    def device(self, uuid):
        return None

    def pixlstash_device(self):
        return None

    def other_processes(self, limit=3):
        return []


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def admission_is_ready_after(server):
    yield
    assert server.library_coordinator.state is SwitchState.READY


@pytest.fixture
def owner(server):
    client = TestClient(server.api, raise_server_exceptions=False)
    response = client.post(
        f"{API}/login",
        json={"username": "owner", "password": "example-owner-password"},
    )
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def fence(server):
    return LibraryFence(
        coordinator=server.library_coordinator,
        describe_holders=server.library_switch.describe_library_holders,
    )


@pytest.fixture
def short_wait(server, monkeypatch):
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 0.5)


@pytest.fixture
def other_lease(server):
    """A lease held by work other than the restore, as an import thread holds."""
    lease = server.library_coordinator.acquire_read()
    assert lease is not None
    released = threading.Event()

    def release():
        if not released.is_set():
            released.set()
            server.library_coordinator.release_read(lease)

    yield lease, release
    release()


def _add_picture(server, filename: str, description: str) -> int:
    open(os.path.join(server.vault.image_root, filename), "wb").close()

    def _do(session):
        pic = Picture(file_path=filename, filename=filename, description=description)
        session.add(pic)
        session.commit()
        session.refresh(pic)
        return pic.id

    return server.vault.db.run_task(_do)


def _set_description(server, picture_id: int, description: str) -> None:
    def _do(session):
        pic = session.get(Picture, picture_id)
        pic.description = description
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_do)


def _description(server, picture_id: int):
    return server.vault.db.run_immediate_read_task(
        lambda s: s.get(Picture, picture_id).description
    )


def _snapshot_after_a_change(server, name: str) -> tuple[int, int]:
    """A snapshot, then a change to the live library it does not hold."""
    pic_id = _add_picture(server, f"{name}-{uuid.uuid4().hex[:8]}.jpg", "before")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after")
    return cp.id, pic_id


# ---------------------------------------------------------------------------
# The restore
# ---------------------------------------------------------------------------


def test_a_lease_held_elsewhere_refuses_the_restore_with_nothing_changed(
    server, fence, short_wait, other_lease
):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "held")
    generation = server.library_coordinator.generation

    with pytest.raises(LibraryStillInUseError) as raised:
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    assert "Nothing was changed" in str(raised.value)
    assert _description(server, pic_id) == "after"
    assert server.library_coordinator.state is SwitchState.READY
    assert server.library_coordinator.generation == generation
    assert not server.auth.is_auth_closed_for_restore()


def test_the_route_refuses_with_409_while_other_work_holds_the_library(
    server, owner, short_wait, other_lease
):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "route-held")

    response = owner.post(f"{API}/snapshots/{snapshot_id}/restore", json={})

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "other work is still using the library" in detail
    assert "Nothing was changed" in detail
    assert _description(server, pic_id) == "after"


def test_the_route_does_not_wait_for_its_own_lease_and_bumps_the_generation(
    server, owner, short_wait
):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "route-own")
    generation = server.library_coordinator.generation

    response = owner.post(f"{API}/snapshots/{snapshot_id}/restore", json={})

    assert response.status_code == 200, response.text
    assert _description(server, pic_id) == "before"
    assert server.library_coordinator.generation == generation + 1


def test_a_lease_from_before_the_restore_is_of_an_old_generation(server, fence):
    snapshot_id, _pic_id = _snapshot_after_a_change(server, "generation")
    before = server.library_coordinator.acquire_read()
    server.library_coordinator.release_read(before)

    server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    after = server.library_coordinator.acquire_read()
    try:
        assert after.generation == before.generation + 1
    finally:
        server.library_coordinator.release_read(after)


def test_a_failure_before_the_swap_reopens_the_same_generation(
    server, fence, monkeypatch
):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "pre-swap")
    generation = server.library_coordinator.generation

    def _close_fails(*args, **kwargs):
        raise RuntimeError("Timed out draining test- requests")

    monkeypatch.setattr(
        server.vault.auth_service, "close_auth_for_restore", _close_fails
    )
    with pytest.raises(RuntimeError, match="Timed out draining"):
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    assert _description(server, pic_id) == "after"
    assert server.library_coordinator.generation == generation


def test_a_failure_after_the_swap_still_moves_to_a_new_generation(
    server, fence, monkeypatch
):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "post-swap")
    generation = server.library_coordinator.generation

    def _requeue_fails(*args, **kwargs):
        raise RuntimeError("test- requeue failed after the swap")

    monkeypatch.setattr(full_restore, "requeue_library_ghosts", _requeue_fails)
    with pytest.raises(RuntimeError, match="requeue failed"):
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    assert _description(server, pic_id) == "before"
    assert server.library_coordinator.generation == generation + 1


def test_a_switch_already_under_way_refuses_at_once(server, fence):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "not-ready")
    server.library_coordinator.begin_switch(timeout=0)
    try:
        started = time.monotonic()
        with pytest.raises(RestoreRefusedError) as raised:
            server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)
        assert not isinstance(raised.value, LibraryStillInUseError)
        assert "being switched" in str(raised.value)
        assert time.monotonic() - started < 5
        assert server.library_coordinator.state is SwitchState.SWITCHING
    finally:
        server.library_coordinator.restore_ready()
    assert _description(server, pic_id) == "after"


def test_a_dry_run_does_not_close_admission(server, fence, short_wait, other_lease):
    snapshot_id, _pic_id = _snapshot_after_a_change(server, "dry-run")
    generation = server.library_coordinator.generation

    report = server.vault.restore_service.restore_full(
        snapshot_id, dry_run=True, library_fence=fence
    )

    assert report.snapshot_id == snapshot_id
    assert server.library_coordinator.generation == generation


def test_admission_closes_before_the_drain_and_both_share_one_deadline(
    server, fence, monkeypatch, other_lease
):
    """The drain, then the vault writers' quiesce, get what is left of it."""
    runner = TaskRunner(
        name=f"restore-fence-{uuid.uuid4().hex[:8]}",
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 3.0)
    order: list[tuple[str, float]] = []
    coordinator = server.library_coordinator
    real_begin = coordinator.begin_switch
    real_drain = runner.pause_and_drain

    def _begin(timeout, exclude=None, restoring=False):
        order.append(("begin_switch", timeout))
        return real_begin(timeout=timeout, exclude=exclude, restoring=restoring)

    def _drain(timeout, reason):
        order.append(("drain", timeout))
        return real_drain(timeout, reason)

    real_quiesce = server.vault.quiesce_background_writers
    real_resume = server.vault.resume_background_writers

    def _quiesce(timeout_s):
        order.append(("quiesce", timeout_s))
        return real_quiesce(timeout_s)

    def _resume():
        order.append(("resume", 0.0))
        return real_resume()

    monkeypatch.setattr(coordinator, "begin_switch", _begin)
    monkeypatch.setattr(runner, "pause_and_drain", _drain)
    monkeypatch.setattr(server.vault, "quiesce_background_writers", _quiesce)
    monkeypatch.setattr(server.vault, "resume_background_writers", _resume)
    snapshot_id, _pic_id = _snapshot_after_a_change(server, "deadline")
    _lease, release = other_lease
    releaser = threading.Timer(1.0, release)
    runner.start()
    try:
        releaser.start()
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)
    finally:
        releaser.cancel()
        runner.stop()

    assert [name for name, _ in order] == ["begin_switch", "drain", "quiesce", "resume"]
    assert order[0][1] <= 3.0
    # The lease was held for about a second of the three.
    assert order[1][1] <= 2.2
    assert order[2][1] <= order[1][1]


@pytest.fixture
def writers(server, monkeypatch):
    """Records the restore's quiesce and resume of the vault's own writers."""
    calls: list[str] = []
    state = {"still_writing": [], "raise": None}

    def _quiesce(timeout_s):
        calls.append("quiesce")
        if state["raise"] is not None:
            raise state["raise"]
        return list(state["still_writing"])

    def _resume():
        calls.append("resume")

    monkeypatch.setattr(server.vault, "quiesce_background_writers", _quiesce)
    monkeypatch.setattr(server.vault, "resume_background_writers", _resume)
    return calls, state


def test_a_vault_writer_still_running_refuses_the_restore_and_is_resumed(
    server, fence, writers
):
    calls, state = writers
    state["still_writing"] = ["face extraction follow-up"]
    snapshot_id, pic_id = _snapshot_after_a_change(server, "writer-held")
    generation = server.library_coordinator.generation

    with pytest.raises(BackgroundWorkStillRunningError) as raised:
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    assert "face extraction follow-up" in str(raised.value)
    assert "Nothing was changed" in str(raised.value)
    assert calls == ["quiesce", "resume"]
    assert _description(server, pic_id) == "after"
    assert server.library_coordinator.generation == generation


def test_a_quiesce_that_raises_still_resumes_the_writers(server, fence, writers):
    calls, state = writers
    state["raise"] = RuntimeError("test- watcher stop failed")
    snapshot_id, pic_id = _snapshot_after_a_change(server, "writer-raised")

    with pytest.raises(RuntimeError, match="watcher stop failed"):
        server.vault.restore_service.restore_full(snapshot_id, library_fence=fence)

    assert calls == ["quiesce", "resume"]
    assert _description(server, pic_id) == "after"


def test_a_resource_restore_also_quiesces_and_resumes_the_writers(server, writers):
    calls, state = writers
    snapshot_id, pic_id = _snapshot_after_a_change(server, "writer-resource")

    server.vault.restore_service.restore_resource(snapshot_id, "picture", pic_id)
    assert calls == ["quiesce", "resume"]
    assert _description(server, pic_id) == "before"

    calls.clear()
    state["still_writing"] = ["face extraction follow-up"]
    _set_description(server, pic_id, "after")
    with pytest.raises(BackgroundWorkStillRunningError):
        server.vault.restore_service.restore_resource(snapshot_id, "picture", pic_id)
    assert calls == ["quiesce", "resume"]
    assert _description(server, pic_id) == "after"


class _FailingCoordinator:
    """Admission whose reopening fails, as an incoherent runtime's does."""

    state = SwitchState.READY

    def begin_switch(self, timeout, exclude=None, restoring=False):
        self.state = SwitchState.SWITCHING

    def restore_ready(self):
        self.state = SwitchState.UNAVAILABLE
        raise RuntimeError("Previous library runtime is not coherent")

    def publish_ready(self):
        self.state = SwitchState.UNAVAILABLE
        raise RuntimeError("Refusing READY for an incoherent library runtime")


def test_a_failure_to_reopen_does_not_hide_the_restores_own_error(server, monkeypatch):
    snapshot_id, _pic_id = _snapshot_after_a_change(server, "reopen-hidden")
    failing = _FailingCoordinator()

    def _close_fails(*args, **kwargs):
        raise RuntimeError("Timed out draining test- requests")

    monkeypatch.setattr(
        server.vault.auth_service, "close_auth_for_restore", _close_fails
    )
    with pytest.raises(RuntimeError, match="Timed out draining"):
        server.vault.restore_service.restore_full(
            snapshot_id, library_fence=LibraryFence(coordinator=failing)
        )
    assert failing.state is SwitchState.UNAVAILABLE


def test_a_failure_to_reopen_after_a_good_restore_is_raised(server):
    snapshot_id, pic_id = _snapshot_after_a_change(server, "reopen-raised")
    failing = _FailingCoordinator()

    with pytest.raises(RuntimeError, match="incoherent"):
        server.vault.restore_service.restore_full(
            snapshot_id, library_fence=LibraryFence(coordinator=failing)
        )
    assert _description(server, pic_id) == "before"
    assert failing.state is SwitchState.UNAVAILABLE


# ---------------------------------------------------------------------------
# The Scrapheap registry
# ---------------------------------------------------------------------------


def test_a_restore_reloads_the_scrapheap_registry_in_place():
    registry = ScrapheapRunRegistry()
    run_id, created = registry.reserve(KIND_PURGE, False, "test-owner", None, None)
    assert created
    assert registry.latest()["id"] == run_id

    registry.reload([])

    assert registry.latest() is None
    # The pending run was the replaced database's; a new one may start.
    registry.check_idle()


# ---------------------------------------------------------------------------
# The edit-copy import and the smart-set refresher
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Hands out one lease on *vault* while READY, and counts them back."""

    def __init__(self, vault, state=SwitchState.READY):
        self.vault = vault
        self.state = state
        self.held = 0

    def acquire_read_or_raise(self):
        if self.state is not SwitchState.READY:
            raise LibraryNotReadyError(self.state)
        self.held += 1
        return LibraryReadLease(7, "test-library", self.vault, None)

    def release_read(self, lease):
        self.held -= 1


def _fake_server(state=SwitchState.READY):
    leased = SimpleNamespace(name="leased")
    return SimpleNamespace(
        vault=SimpleNamespace(name="current"),
        library_coordinator=_FakeCoordinator(leased, state),
    )


class _Registry:
    """The library registry, with the library open or not."""

    def __init__(self):
        self.library = SimpleNamespace(uuid="test-library", path="test-library-root")
        self.open = False

    def active_library(self):
        return self.library if self.open else None


def _server_with_no_library_open():
    """A server a real coordinator admits only once ``library_registry.open``.

    Admission is READY all along: the coordinator refuses the lease because
    the runtime is not coherent (UNAVAILABLE), not because a switch holds it.
    """
    db = SimpleNamespace(is_open=True)
    server = SimpleNamespace(
        library_registry=_Registry(),
        vault=SimpleNamespace(
            name="current", is_open=True, db=db, image_root="test-library-root"
        ),
        auth=SimpleNamespace(vault_db=db),
    )
    server.library_coordinator = LibraryGenerationCoordinator(server)
    return server


def _counting_refusals(monkeypatch, coordinator) -> list[SwitchState]:
    """Record each state *coordinator* refuses a lease in."""
    refusals: list[SwitchState] = []
    acquire = coordinator.acquire_read_or_raise

    def _acquire():
        try:
            return acquire()
        except LibraryNotReadyError as exc:
            refusals.append(exc.state)
            raise

    monkeypatch.setattr(coordinator, "acquire_read_or_raise", _acquire)
    return refusals


def _until(predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _settles(count, timeout=5.0) -> bool:
    """Wait until *count()* stops changing for a while; True if it did."""
    deadline = time.monotonic() + timeout
    last, quiet_since = count(), time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.02)
        now = count()
        if now != last:
            last, quiet_since = now, time.monotonic()
        elif time.monotonic() - quiet_since >= 0.5:
            return True
    return False


def test_an_edit_copy_pass_runs_on_the_leased_vault_and_releases(tmp_path, monkeypatch):
    server = _fake_server()
    seen = []

    def _leased(srv, vault, session_dir):
        seen.append((vault.name, srv.library_coordinator.held))
        return [1]

    monkeypatch.setattr(edit_copies, "_process_session_leased", _leased)

    assert process_session(server, str(tmp_path)) == [1]
    assert seen == [("leased", 1)]
    assert server.library_coordinator.held == 0


def test_an_edit_copy_pass_waits_while_the_library_is_switching(tmp_path, monkeypatch):
    server = _fake_server(SwitchState.SWITCHING)
    monkeypatch.setattr(
        edit_copies,
        "_process_session_leased",
        lambda *args: pytest.fail("read the session while switching"),
    )

    with pytest.raises(EditSessionDeferred) as deferred:
        process_session(server, str(tmp_path))

    assert deferred.value.state is SwitchState.SWITCHING


def test_an_edit_copy_pass_waits_while_no_library_is_open(tmp_path, monkeypatch):
    """Admission READY with no coherent library refuses the lease; the
    pass defers rather than returning as if there were nothing to import."""
    server = _server_with_no_library_open()
    monkeypatch.setattr(
        edit_copies,
        "_process_session_leased",
        lambda *args: pytest.fail("read the session with no library open"),
    )

    with pytest.raises(EditSessionDeferred) as deferred:
        process_session(server, str(tmp_path))

    assert deferred.value.state is SwitchState.UNAVAILABLE
    assert server.library_coordinator.state is SwitchState.READY


def test_an_edit_saved_while_no_library_is_open_is_read_once_one_is(
    tmp_path, monkeypatch
):
    server = _server_with_no_library_open()
    refusals = _counting_refusals(monkeypatch, server.library_coordinator)
    session = "20260101-000000-abcdef"
    os.makedirs(tmp_path / session)
    read = threading.Event()
    seen = []

    def _leased(srv, vault, session_dir):
        seen.append(vault.name)
        read.set()
        return [1]

    monkeypatch.setattr(edit_copies, "_process_session_leased", _leased)
    watcher = EditCopyWatcher(
        str(tmp_path), lambda path: process_session(server, path), debounce_s=0.05
    )
    try:
        watcher._schedule(session)
        assert _until(lambda: refusals), "the pass was not refused"
        server.library_registry.open = True
        assert read.wait(5), "the session was not read again once a library opened"
    finally:
        watcher.stop()
    assert seen == ["current"]
    assert watcher._unavailable_tries == {}


def test_the_watcher_stops_rereading_a_session_while_no_library_opens(
    tmp_path, monkeypatch
):
    """The re-reads are bounded, one debounce apart, never a busy loop."""
    server = _server_with_no_library_open()
    refusals = _counting_refusals(monkeypatch, server.library_coordinator)
    session = "20260101-000000-abcdef"
    os.makedirs(tmp_path / session)
    watcher = EditCopyWatcher(
        str(tmp_path), lambda path: process_session(server, path), debounce_s=0.02
    )
    try:
        watcher._schedule(session)
        assert _settles(lambda: len(refusals))
        assert refusals == [SwitchState.UNAVAILABLE] * (UNAVAILABLE_RETRIES + 1)
        assert watcher._timers == {}
        assert watcher._unavailable_tries == {}
    finally:
        watcher.stop()


def test_the_watcher_reads_a_deferred_session_again(tmp_path):
    session = "20260101-000000-abcdef"
    os.makedirs(tmp_path / session)
    calls: list[str] = []
    done = threading.Event()

    def _settled(path):
        calls.append(path)
        if len(calls) == 1:
            raise EditSessionDeferred(
                "test- the library is being restored.", SwitchState.SWITCHING
            )
        done.set()

    watcher = EditCopyWatcher(str(tmp_path), _settled, debounce_s=0.05)
    try:
        watcher._schedule(session)
        assert done.wait(5), "a deferred session was not read again"
    finally:
        watcher.stop()
    assert calls == [str(tmp_path / session)] * 2


def test_a_smart_set_refresh_runs_on_the_leased_vault_and_releases(monkeypatch):
    server = _fake_server()
    seen = []

    def _refresh(srv):
        seen.append((srv.vault.name, srv.library_coordinator.held))
        return []

    monkeypatch.setattr(smart_set_refresher, "refresh_smart_sets", _refresh)
    refresher = SmartSetRefresher(server)

    refresher._run()

    assert seen == [("leased", 1)]
    assert server.library_coordinator.held == 0
    assert refresher._timer is None


@pytest.mark.parametrize("state", [SwitchState.SWITCHING, None])
def test_a_smart_set_refresh_waits_while_switching_or_starting(monkeypatch, state):
    server = _fake_server(state or SwitchState.READY)
    if state is None:
        # Start-up: the refresher is built before the coordinator.
        del server.library_coordinator
    monkeypatch.setattr(
        smart_set_refresher,
        "refresh_smart_sets",
        lambda srv: pytest.fail("refreshed while switching"),
    )
    refresher = SmartSetRefresher(server)
    try:
        refresher._run()
        assert refresher._timer is not None, "the refresh was not re-armed"
    finally:
        refresher.close()


def test_a_smart_set_refresh_refused_while_no_library_is_open_runs_once_one_is(
    monkeypatch,
):
    """Admission READY with no coherent library refuses the lease; the
    refresh waits a window rather than being dropped."""
    monkeypatch.setattr(smart_set_refresher, "REFRESH_WINDOW_S", 0.05)
    server = _server_with_no_library_open()
    refreshed = threading.Event()
    seen = []

    def _refresh(srv):
        seen.append(srv.vault.name)
        refreshed.set()
        return []

    monkeypatch.setattr(smart_set_refresher, "refresh_smart_sets", _refresh)
    refresher = SmartSetRefresher(server)
    try:
        refresher._run()
        assert refresher._timer is not None, "the refresh was dropped"
        server.library_registry.open = True
        assert refreshed.wait(5), "the refresh did not run once a library opened"
    finally:
        refresher.close()
    assert seen == ["current"]
    assert refresher._unavailable_tries == 0


def test_a_smart_set_refresh_stops_waiting_while_no_library_opens(monkeypatch):
    """The waits are bounded, one window apart, never a busy loop; the
    next library event starts the count again."""
    monkeypatch.setattr(smart_set_refresher, "REFRESH_WINDOW_S", 0.02)
    server = _server_with_no_library_open()
    refusals = _counting_refusals(monkeypatch, server.library_coordinator)
    monkeypatch.setattr(
        smart_set_refresher,
        "refresh_smart_sets",
        lambda srv: pytest.fail("refreshed with no library open"),
    )
    refresher = SmartSetRefresher(server)
    try:
        refresher._run()
        assert _settles(lambda: len(refusals))
        assert refusals == [SwitchState.UNAVAILABLE] * (UNAVAILABLE_RETRIES + 1)
        assert refresher._timer is None
        assert refresher._unavailable_tries == 0
    finally:
        refresher.close()


# ---------------------------------------------------------------------------
# A library switch hands the coordinator on
# ---------------------------------------------------------------------------


def test_each_vault_a_switch_publishes_holds_the_coordinator(server, tmp_path):
    original = server.library_registry.active_library()
    second = server.library_registry.create(str(tmp_path / "second"), "Second")
    try:
        server.library_switch.switch_to(second.uuid)
        assert server.vault.library_coordinator is server.library_coordinator
    finally:
        server.library_switch.switch_to(original.uuid)
    assert server.vault.library_coordinator is server.library_coordinator
