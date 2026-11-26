"""A finished run gives its GPU lease back before anything waits on the library.

Training needs the card; bringing the LoRA to the shelf does not. The import
may wait out a restore or a library switch (``LibraryNotReadyError`` with
SWITCHING), and while it waits every other GPU job would sit behind a lease
nobody uses. So the worker (``TrainingService._run_job``) and a re-attached
run (``_poll_and_finalize_reattached``) release the lease first, and the
start-up reconcile, which a switch's admission waits on, waits on a thread of
its own.

A run past its release is not claimed again: the worker claims only queued
runs, and the row says running or completed. A failure after the release still
ends the row truthfully.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

import pytest

from pixlstash.hub import training_runs
from pixlstash.services import training_service as training_service_module
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    SwitchState,
)
from pixlstash.services.training_service import TrainingService
from pixlstash.training.backend import TrainingProgress
from pixlstash.training.local_ai_toolkit import LocalAiToolkit
from tests.training import test_training_service as service_tests
from tests.training.test_training_service import (
    _LIBRARY_UUID,
    _RESOLUTION,
    _SET_ID,
    _CapturingNotify,
    _job_status,
    _make_dataset_builder,
    _make_service,
    _poll_until,
    _StubLease,
    _write_template,
)

# The fake ai-toolkit checkout the service tests run against.
repo_dir = service_tests.repo_dir
_fast_step_delay = service_tests._fast_step_delay

_REFUSED = LibraryNotReadyError(SwitchState.SWITCHING)


@pytest.fixture(autouse=True)
def _quick_library_retry(monkeypatch):
    monkeypatch.setattr(training_service_module, "_LIBRARY_NOT_READY_RETRY_S", 0.01)


class _LibraryImport:
    """The vault's importer, refusing while the library switches.

    Each call records what the run's GPU lease had done by then and whether the
    worker could claim the run again; *refusals* calls answer SWITCHING first,
    then *error* is raised or the run lands.
    """

    def __init__(self, hub, lease, *, refusals=1, error=None):
        self.hub = hub
        self.lease = lease
        self.refusals = refusals
        self.error = error
        self.seen: list[dict] = []

    def __call__(self, run_dir: str, info: dict) -> dict:
        claimable = training_runs.next_queued_run(self.hub, _LIBRARY_UUID)
        self.seen.append(
            {
                "lease": [e for e in self.lease.events if e in ("acquire", "release")],
                "claimable": None if claimable is None else claimable["uuid"],
                "status": training_runs.get_run_status(self.hub, info["uuid"]),
            }
        )
        if len(self.seen) <= self.refusals:
            raise _REFUSED
        if self.error is not None:
            raise self.error
        return {"stack_id": 7, "imported": 1, "cover_sha256": "a" * 64}


def _a_queued_run(hub, repo_dir, lease, importer, name):
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
    return service, job


def _run_until_ended(hub, service, job) -> None:
    """Start the worker and stop it once *job* has ended.

    The row reads completed as soon as the trainer is done, before the import,
    so a completed run has ended only once completed_at is set; stopping the
    worker on its status alone would stop it in the middle of the wait.
    """

    def ended():
        row = _job_status(hub, job["uuid"])
        return row.completed_at or row.status in ("error", "stopped")

    service.start()
    try:
        assert _poll_until(ended), (
            f"status={_job_status(hub, job['uuid']).status!r}, never ended"
        )
    finally:
        service.stop()


def test_the_gpu_lease_is_free_while_the_import_waits_for_the_library(hub, repo_dir):
    lease = _StubLease(grant=True)
    importer = _LibraryImport(hub, lease, refusals=2)
    service, job = _a_queued_run(hub, repo_dir, lease, importer, "waits")

    _run_until_ended(hub, service, job)

    assert _job_status(hub, job["uuid"]).status == "completed"
    assert len(importer.seen) == 3, "the import did not wait for the library"
    for seen in importer.seen:
        assert seen["lease"] == ["acquire", "release"], (
            "the GPU lease was held while the import waited for the library"
        )
        # Released, and still not the worker's to claim again.
        assert seen["claimable"] is None
        assert seen["status"] != "queued"
    assert lease.acquire_count == lease.release_count == 1


def test_an_import_failing_after_the_release_ends_the_run_in_error(hub, repo_dir):
    lease = _StubLease(grant=True)
    importer = _LibraryImport(
        hub, lease, refusals=1, error=RuntimeError("example-injected shelf failure")
    )
    service, job = _a_queued_run(hub, repo_dir, lease, importer, "fails")

    _run_until_ended(hub, service, job)

    assert [seen["lease"] for seen in importer.seen] == [["acquire", "release"]] * 2
    row = _job_status(hub, job["uuid"])
    assert row.status == "error"
    assert row.error == "Run completed but the LoRA artifact could not be imported"
    assert row.completed_at
    assert lease.acquire_count == lease.release_count == 1


def test_a_release_check_that_fails_does_not_fail_the_trained_run(
    hub, repo_dir, monkeypatch, caplog
):
    lease = _StubLease(grant=True)
    importer = _LibraryImport(hub, lease, refusals=0)
    service, job = _a_queued_run(hub, repo_dir, lease, importer, "checked")

    def broken_check(job_uuid, lease_, watch):
        raise RuntimeError("example-injected NVML failure")

    monkeypatch.setattr(service, "_check_released", broken_check)

    with caplog.at_level(logging.ERROR, logger=training_service_module.__name__):
        _run_until_ended(hub, service, job)

    assert _job_status(hub, job["uuid"]).status == "completed"
    assert len(importer.seen) == 1
    assert lease.acquire_count == lease.release_count == 1
    assert (
        f"Job {job['uuid']}: could not check what GPU memory its trainer" in caplog.text
    )


class _FinishingBackend:
    """A live trainer that has just finished: one poll, then it is gone."""

    def __init__(self, final):
        self.final = final
        self.alive = True

    def handle_for(self, work_dir, job_id, name, pid, training_folder=None):
        return LocalAiToolkit.handle_for(
            work_dir, job_id, name, pid, training_folder=training_folder
        )

    def is_alive(self, handle):
        return self.alive

    def progress(self, handle):
        self.alive = False
        return TrainingProgress(
            status="completed",
            info="Training completed",
            step=4,
            total_steps=4,
            speed="",
            final_checkpoint=self.final,
        )

    def reconcile(self, handle):
        return self.progress(handle)


def _a_service_on(hub, repo_dir, backend, importer, *, adopted=None):
    return TrainingService(
        hub,
        _LIBRARY_UUID,
        os.path.dirname(hub.path),
        request_gpu=_StubLease().request,
        adopt_gpu=adopted.request if adopted is not None else None,
        import_run=importer,
        notify=_CapturingNotify(),
        config_provider=lambda: {"repo_path": repo_dir, "python": sys.executable},
        dataset_builder=_make_dataset_builder(),
        backend_factory=lambda repo_path, python: backend,
        poll_interval_s=0.02,
    )


def _a_run_left_running(hub, service, repo_dir, pid):
    job = service.enqueue(
        set_id=_SET_ID,
        name="survivor",
        base_model="m",
        trigger_word="t",
        template_path="unused.json",
        resolution=_RESOLUTION,
    )
    training_runs.update_run(
        hub,
        job["uuid"],
        status="running",
        pid=pid,
        training_folder=os.path.join(repo_dir, "output"),
    )
    return job


def test_a_re_attached_run_frees_its_lease_before_the_import_waits(hub, repo_dir):
    adopted = _StubLease()
    adopted.held = True
    importer = _LibraryImport(hub, adopted, refusals=1)
    backend = _FinishingBackend(os.path.join(repo_dir, "output", "run", "run.st"))
    service = _a_service_on(hub, repo_dir, backend, importer, adopted=adopted)
    job = _a_run_left_running(hub, service, repo_dir, os.getpid())

    service.adopt_live_runs()
    service.reconcile_on_startup()

    assert _poll_until(lambda: _job_status(hub, job["uuid"]).completed_at)
    assert _job_status(hub, job["uuid"]).status == "completed"
    assert len(importer.seen) == 2
    assert [seen["lease"] for seen in importer.seen] == [["acquire", "release"]] * 2
    assert adopted.release_count == 1


def test_the_start_up_reconcile_does_not_wait_for_a_switch(hub, repo_dir):
    """A switch opens admission only once the incoming library has started, and
    its start reconciles the runs that finished while it was closed. An import
    waiting there for the switch would wait for itself."""
    switched = threading.Event()
    imports = []

    def importer(run_dir, info):
        imports.append(run_dir)
        if not switched.is_set():
            raise _REFUSED
        return {"stack_id": 7, "imported": 1, "cover_sha256": "a" * 64}

    final = os.path.join(repo_dir, "output", "run", "run.safetensors")
    backend = _FinishingBackend(final)
    backend.alive = False
    service = _a_service_on(hub, repo_dir, backend, importer)
    job = _a_run_left_running(hub, service, repo_dir, 999_999_999)

    starting = threading.Thread(target=service.reconcile_on_startup, daemon=True)
    starting.start()
    starting.join(5)
    returned = not starting.is_alive()
    switched.set()
    starting.join(5)

    assert returned, "the library's start waited on the import"
    assert _poll_until(lambda: _job_status(hub, job["uuid"]).completed_at)
    assert _job_status(hub, job["uuid"]).status == "completed"
    assert imports and all(run_dir == os.path.dirname(final) for run_dir in imports)
