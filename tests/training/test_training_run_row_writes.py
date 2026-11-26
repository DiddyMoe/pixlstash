"""A run's row says what is running, and no write lands on a newer state.

Each write that moves a run on names the states it may move it from
(``training_runs.update_run``'s ``only_if``), so a write that lost a race
changes nothing. A write that must land (the launch's, an end state) is
retried on a busy hub, and never while ``_active_lock`` is held: a cancel and
every poll wait on that lock.

* The launch write records the trainer's pid and folder, which nothing writes
  again; unrecorded, the trainer is stopped.
* A cancel that read the run running writes ``stopping`` only while the run is
  still unfinished, never over an end written since.
* Retries run outside the lock; a poll's own end state is tried once.
* A cancel during the import stops nothing and leaves no skip behind; a skip
  asked while the previews draw ends with them.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from pixlstash.hub import training_runs
from pixlstash.hub.db import now_iso
from pixlstash.services import training_service as training_service_module
from pixlstash.training.backend import TrainingProgress
from tests.training import test_training_service as service_tests
from tests.training.test_training_service import (
    _LIBRARY_UUID,
    _RESOLUTION,
    _SET_ID,
    _CapturingNotify,
    _job_status,
    _make_dataset_builder,
    _make_service,
    _queued_job,
    _RecordingImport,
    _StubLease,
    _write_template,
)

# The fake ai-toolkit checkout the service tests run against.
repo_dir = service_tests.repo_dir
_fast_step_delay = service_tests._fast_step_delay

_UNRECORDED = "The run could not be recorded as started, so it was stopped."


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(training_service_module, "_TERMINAL_WRITE_BACKOFF_S", 0.0)


class _Hub:
    """The hub's run writes, busy for the writes *busy* picks.

    Records every try of a picked write with whether ``_active_lock`` was
    held, and runs *meanwhile* once, before the first try lands or fails.
    """

    def __init__(self, monkeypatch, service, busy, *, failures=0, meanwhile=None):
        self.service = service
        self.busy = busy
        self.failures = failures
        self.meanwhile = meanwhile
        self.tries: list[dict] = []
        self._real = training_runs.update_run
        monkeypatch.setattr(training_service_module.training_runs, "update_run", self)

    def __call__(self, hub, run_uuid, **fields):
        if not self.busy(fields):
            return self._real(hub, run_uuid, **fields)
        self.tries.append(
            {"fields": fields, "locked": self.service._active_lock.locked()}
        )
        if self.meanwhile is not None:
            meanwhile, self.meanwhile = self.meanwhile, None
            meanwhile()
        if len(self.tries) <= self.failures:
            raise sqlite3.OperationalError("database is locked")
        return self._real(hub, run_uuid, **fields)


def _launch_write(fields) -> bool:
    return fields.get("status") == "running" and "pid" in fields


def _trainer_gone(pid) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _a_queued_run(hub, repo_dir, name="launch"):
    lease = _StubLease(grant=True)
    importer = _RecordingImport()
    service = _make_service(
        hub,
        repo_dir,
        lease=lease,
        notify=_CapturingNotify(),
        dataset_builder=_make_dataset_builder(),
        importer=importer,
    )
    job = service.enqueue(
        set_id=_SET_ID,
        name=name,
        base_model="m",
        trigger_word="t",
        template_path=_write_template(repo_dir, steps=4, save_every=2),
        resolution=_RESOLUTION,
    )
    return service, job, lease, importer


def _run(hub, service, job):
    service._run_job(training_runs.get_run(hub, _LIBRARY_UUID, job["id"]))


# ---------------------------------------------------------------------------
# The launch write
# ---------------------------------------------------------------------------


def test_a_launch_that_cannot_be_recorded_stops_its_trainer(hub, repo_dir, monkeypatch):
    """Unrecorded, the row names no pid and no folder: nothing would re-attach
    to the trainer after a restart, and it would train under no lease."""
    # Still loading when its launch is written, so only a stop ends it soon.
    monkeypatch.setenv("FAKE_RUN_LOAD_SECONDS", "30")
    monkeypatch.setattr(
        "pixlstash.training.local_ai_toolkit._CANCEL_GRACE_SECONDS", 0.2
    )
    service, job, lease, importer = _a_queued_run(hub, repo_dir)
    busy = _Hub(monkeypatch, service, _launch_write, failures=99)

    started = time.monotonic()
    _run(hub, service, job)

    assert time.monotonic() - started < 20, "the trainer was left to run"
    attempts = training_service_module._TERMINAL_WRITE_ATTEMPTS
    assert len(busy.tries) == attempts, "the launch write was not tried again"
    pid = busy.tries[0]["fields"]["pid"]
    assert service_tests._poll_until(lambda: _trainer_gone(pid), timeout=10)
    final = _job_status(hub, job["uuid"])
    assert (final.status, final.error, final.pid) == ("error", _UNRECORDED, None)
    assert importer.calls == []
    assert lease.acquire_count == lease.release_count == 1


def test_a_launch_write_that_meets_a_busy_hub_lands_and_holds_no_lock(
    hub, repo_dir, monkeypatch
):
    service, job, lease, importer = _a_queued_run(hub, repo_dir)
    busy = _Hub(monkeypatch, service, _launch_write, failures=2)

    _run(hub, service, job)

    assert len(busy.tries) == 3
    assert [t["locked"] for t in busy.tries] == [False] * 3, (
        "the launch write was retried while holding the lock a cancel waits on"
    )
    final = _job_status(hub, job["uuid"])
    assert final.status == "completed" and final.completed_at
    assert final.pid == busy.tries[-1]["fields"]["pid"]
    assert len(importer.calls) == 1


def test_a_cancel_between_the_submit_and_the_launch_write_keeps_its_stop(
    hub, repo_dir, monkeypatch
):
    """The cancel reads the run queued and writes its stop from queued; the
    launch write then finds it stopped, writes nothing, and stops the trainer."""
    monkeypatch.setenv("FAKE_RUN_LOAD_SECONDS", "30")
    monkeypatch.setattr(
        "pixlstash.training.local_ai_toolkit._CANCEL_GRACE_SECONDS", 0.2
    )
    service, job, lease, importer = _a_queued_run(hub, repo_dir)
    answers = []

    def cancel_meanwhile():
        # On a thread: a launch write held under the lock would hold it up.
        thread = threading.Thread(
            target=lambda: answers.append(service.cancel(job["id"])), daemon=True
        )
        thread.start()
        thread.join(5)
        answers.append("returned" if not thread.is_alive() else "held up")

    busy = _Hub(monkeypatch, service, _launch_write, meanwhile=cancel_meanwhile)

    started = time.monotonic()
    _run(hub, service, job)

    assert answers == [True, "returned"]
    assert time.monotonic() - started < 20, "the trainer was left to run"
    pid = busy.tries[0]["fields"]["pid"]
    assert service_tests._poll_until(lambda: _trainer_gone(pid), timeout=10)
    final = _job_status(hub, job["uuid"])
    assert (final.status, final.error, final.pid) == (
        "stopped",
        "Cancelled before start",
        None,
    )
    assert importer.calls == []
    assert lease.acquire_count == lease.release_count == 1


# ---------------------------------------------------------------------------
# The cancel's stopping flag
# ---------------------------------------------------------------------------


class _Backend:
    def __init__(self):
        self.cancelled = []

    def cancel(self, handle):
        self.cancelled.append(handle.pid)


@pytest.mark.parametrize(
    "ended",
    [
        {"status": "stopped", "completed_at": "2026-11-25T09:00:00+00:00"},
        {"status": "error", "completed_at": "2026-11-25T09:00:00+00:00"},
        # Trained, and being imported.
        {"status": "completed", "completed_at": None},
    ],
    ids=["stopped", "error", "importing"],
)
def test_a_cancel_that_read_running_leaves_a_run_that_ended_meanwhile(
    hub, repo_dir, monkeypatch, ended
):
    service, job = _queued_job(hub, repo_dir)
    training_runs.update_run(hub, job["uuid"], status="running", pid=4242)
    running = training_runs.get_run(hub, _LIBRARY_UUID, job["id"])
    training_runs.update_run(hub, job["uuid"], **ended)
    backend = _Backend()
    service._active_backend = backend
    service._active_job_uuid = job["uuid"]
    monkeypatch.setattr(
        training_service_module.training_runs, "get_run", lambda *args: running
    )

    assert service.cancel(job["id"]) is False

    final = _job_status(hub, job["uuid"])
    assert (final.status, final.completed_at) == (
        ended["status"],
        ended["completed_at"],
    ), "the cancel wrote stopping over a run that had ended"
    assert backend.cancelled == []
    assert service._skip_previews == set()


# ---------------------------------------------------------------------------
# No retry under _active_lock
# ---------------------------------------------------------------------------


def test_a_queued_stop_that_meets_a_busy_hub_is_retried_outside_the_lock(
    hub, repo_dir, monkeypatch
):
    service, job = _queued_job(hub, repo_dir)
    busy = _Hub(
        monkeypatch,
        service,
        lambda fields: fields.get("status") == "stopped",
        failures=2,
    )

    assert service.cancel(job["id"]) is True

    assert len(busy.tries) == 3
    assert [t["locked"] for t in busy.tries] == [False] * 3, (
        "the queued stop was retried while holding the lock the worker waits on"
    )
    assert _job_status(hub, job["uuid"]).status == "stopped"


def test_a_polls_end_state_is_tried_once_under_the_lock(hub, repo_dir, monkeypatch):
    """The poll writes under the lock a cancel waits on, so its end state is
    not retried there: _finalize writes it again, with retries, outside it."""
    service, job = _queued_job(hub, repo_dir)
    training_runs.update_run(hub, job["uuid"], status="running", started_at=now_iso())
    busy = _Hub(
        monkeypatch,
        service,
        lambda fields: fields.get("status") == "completed",
        failures=99,
    )
    finished = TrainingProgress(
        status="completed", info="done", step=4, total_steps=4, speed=""
    )

    service._persist_progress(job["uuid"], finished, SimpleNamespace(log_path=""))

    assert [t["locked"] for t in busy.tries] == [True]
    assert _job_status(hub, job["uuid"]).status == "running"


# ---------------------------------------------------------------------------
# The skip of the remaining previews
# ---------------------------------------------------------------------------


def test_a_cancel_during_the_import_stops_nothing_and_leaves_no_skip(
    hub, repo_dir, caplog
):
    service, job = _queued_job(hub, repo_dir)
    training_runs.update_run(hub, job["uuid"], status="completed", pid=4242)
    backend = _Backend()
    service._active_backend = backend
    service._active_job_uuid = job["uuid"]

    with caplog.at_level("INFO", logger=training_service_module.__name__):
        assert service.cancel(job["id"]) is False

    assert service._skip_previews == set()
    assert backend.cancelled == []
    assert "is completed and being imported; nothing to stop" in caplog.text


def test_a_skip_asked_while_the_previews_draw_ends_with_them(
    hub, repo_dir, monkeypatch
):
    service, job = _queued_job(hub, repo_dir)
    training_runs.update_run(hub, job["uuid"], status="completed")
    info = training_runs.get_run(hub, _LIBRARY_UUID, job["id"])
    monkeypatch.setattr(service, "_resolution", lambda uuid: {"previews_deferred": 1})
    monkeypatch.setattr(service, "_record_run", lambda *args: None)
    seen = []

    def drawing(backend, handle, info, progress, lease):
        seen.append(service.cancel(job["id"]))
        seen.append(set(service._skip_previews))
        return True

    monkeypatch.setattr(service, "_sample_after_training", drawing)
    trained = TrainingProgress(
        status="completed",
        info="done",
        step=4,
        total_steps=4,
        speed="",
        final_checkpoint="final.safetensors",
    )

    assert service._after_training(None, None, info, trained, None, None) is True

    assert seen == [True, {job["uuid"]}]
    assert service._skip_previews == set(), "the skip outlived the previews"
    assert service._drawing_previews == set()
