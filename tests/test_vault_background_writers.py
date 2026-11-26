"""The vault's own background writers stop for a restore and come back after it.

A full restore drains the task runner, but three writers are not tasks: the
changed-tags flush timer, the reference-folder watcher (whose callback writes
``ReferenceFolder.last_scanned`` by an id from its watch map) and face
extraction's follow-ups. ``Vault.quiesce_background_writers`` stops them and
says what it could not stop; ``resume_background_writers`` starts the watcher
again, from the database that is live by then.

The watcher can be restarted only because ``ReferenceFolderWatcher.start``
builds a new watchdog observer after a stop: an observer is a thread, and a
thread starts once.
"""

import threading
import time
from concurrent.futures import Future

import pytest

from pixlstash.event_types import EventType
from pixlstash.utils.reference_folder_watcher import ReferenceFolderWatcher
from pixlstash.vault import Vault


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_stopped_watcher_starts_again_with_no_watches_and_sees_changes(tmp_path):
    seen = []
    watcher = ReferenceFolderWatcher(on_folder_changed=seen.append, debounce_s=0.05)
    watcher.start()
    watcher.watch_folder(1, str(tmp_path))
    assert watcher.watched_ids() == {1}

    watcher.stop()
    assert watcher.watched_ids() == set(), "a stop kept the old watch map"
    # Scheduled while stopped, so on the old observer, which start() replaces:
    # the watch dies with it and must not be reported (or block a re-watch).
    watcher.watch_folder(2, str(tmp_path))
    watcher.start()
    try:
        assert watcher.watched_ids() == set(), "a watch on the old observer survived"
        watcher.watch_folder(1, str(tmp_path))
        assert watcher.watched_ids() == {1}
        (tmp_path / "a.png").write_bytes(b"example-not-an-image")
        assert _until(lambda: seen == [1]), seen
    finally:
        watcher.stop()


def test_starting_a_running_watcher_is_refused_rather_than_hanging(tmp_path):
    """No watch, so the second start cannot reach watchdog's emitter join: what
    is left to tell apart is our refusal from the Thread's own RuntimeError."""
    watcher = ReferenceFolderWatcher(on_folder_changed=lambda folder_id: None)
    watcher.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            watcher.start()
        assert watcher._observer.is_alive()
    finally:
        watcher.stop()


def test_stop_waits_for_a_callback_already_running(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def callback(folder_id):
        entered.set()
        release.wait(5)

    watcher = ReferenceFolderWatcher(on_folder_changed=callback, debounce_s=0.01)
    watcher.start()
    watcher._schedule_rescan(4)
    assert entered.wait(5)
    stopper = threading.Thread(target=watcher.stop, daemon=True)
    stopper.start()
    try:
        stopper.join(0.3)
        assert stopper.is_alive(), "stop() returned with the callback still running"
    finally:
        release.set()
        stopper.join(5)
    assert not stopper.is_alive()


def test_no_callback_starts_once_stop_has_run(tmp_path):
    seen = []
    watcher = ReferenceFolderWatcher(on_folder_changed=seen.append, debounce_s=0.05)
    watcher.start()
    watcher._schedule_rescan(5)
    generation = watcher._generation
    watcher.stop()
    # An event the observer dispatched as it stopped arms nothing.
    watcher._schedule_rescan(6)
    assert watcher._timers == {}
    watcher.start()
    try:
        # A timer that fired as stop() ran, reaching the lock after a restart.
        watcher._trigger_rescan(5, generation)
        time.sleep(0.2)
        assert seen == [], "a callback ran for a timer armed before the stop"
    finally:
        watcher.stop()


@pytest.fixture
def vault(tmp_path):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=True,
        daily_snapshots_enabled=False,
    )
    # A started vault's watcher, without the workers a start() brings up.
    v._ref_folder_watcher = ReferenceFolderWatcher(
        on_folder_changed=v._on_reference_folder_fs_changed
    )
    v._start_reference_folder_watches()
    v._started = True
    try:
        yield v
    finally:
        v._started = False
        v._ref_folder_watcher.stop()
        v._ref_folder_watcher = None
        v.close()


def test_quiesce_stops_the_watcher_and_resume_rewatches_from_the_database(vault):
    watcher = vault._ref_folder_watcher
    assert watcher.watched_ids() == {None}
    observer = watcher._observer

    assert vault.quiesce_background_writers(1.0) == []
    assert watcher.watched_ids() == set(), "the watcher still watches"
    assert not observer.is_alive()

    vault.resume_background_writers()
    assert watcher.watched_ids() == {None}, "the watches were not read again"
    assert watcher._observer is not observer and watcher._observer.is_alive()


def test_resume_without_a_quiesce_leaves_the_watcher_alone(vault):
    observer = vault._ref_folder_watcher._observer
    vault.resume_background_writers()
    assert vault._ref_folder_watcher._observer is observer
    assert observer.is_alive()


def test_quiesce_drops_the_pending_changed_tags_event(vault):
    events = []
    vault.add_event_listener(lambda event, data: events.append((event, data)))
    vault._changed_tags_flush_delay_s = 0.05
    vault._queue_changed_tags_notification([1, 2])

    assert vault.quiesce_background_writers(1.0) == []
    assert vault._changed_tags_flush_timer is None
    assert vault._changed_tags_pending_ids == set()
    time.sleep(0.2)
    assert (EventType.CHANGED_TAGS, [1, 2]) not in events
    vault.resume_background_writers()


def test_quiesce_names_a_face_follow_up_it_could_not_wait_out(vault, caplog):
    tail = Future()
    with vault._face_followups_lock:
        vault._face_followups[tail] = [7]
    try:
        assert vault.quiesce_background_writers(0.05) == ["face extraction follow-up"]
        assert "unfinished after 0.1 s, for pictures [7]" in caplog.text
    finally:
        vault._release_face_followup(tail)
    vault.resume_background_writers()

    assert vault.quiesce_background_writers(0.05) == []
    vault.resume_background_writers()


def test_a_quiesce_whose_stop_raised_is_still_resumed(vault, monkeypatch):
    watcher = vault._ref_folder_watcher
    real_stop = watcher.stop

    def stop_then_fail():
        real_stop()
        raise RuntimeError("example-injected observer failure")

    monkeypatch.setattr(watcher, "stop", stop_then_fail)
    with pytest.raises(RuntimeError):
        vault.quiesce_background_writers(1.0)
    monkeypatch.setattr(watcher, "stop", real_stop)

    vault.resume_background_writers()
    assert watcher.watched_ids() == {None}, "the watcher stayed down"
    assert watcher._observer.is_alive()
