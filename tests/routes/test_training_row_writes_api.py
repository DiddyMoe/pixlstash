"""The training routes' run-row writes never land on a newer state.

Quit ComfyUI writes the owner's yes only on a run still queued and not yet
past the quit; a failed cover choice puts the old step back only while the
row still holds the step it wrote. Both read the row first, so each test
moves the row on between that read and the write.

Runs on the shared Server of ``tests/routes/test_training_api.py``.
"""

from __future__ import annotations

import pytest

from pixlstash.hub import training_runs
from pixlstash.routes import training as training_routes
from tests.routes import test_training_api as api_tests
from tests.routes.test_training_api import _live_run, _seed_job, _setup, _teardown

# The module's Server, built on first use and closed after the last test.
_shared_server = api_tests._shared_server


def _read_before(monkeypatch, hub, job_uuid, moved):
    """The route's first read of the run returns it as it was; *moved* is
    then written, so the route's write meets the row moved on."""
    real = training_runs.get_run
    reads = []

    def get_run(*args):
        row = real(*args)
        if not reads:
            reads.append(row)
            training_runs.update_run(hub, job_uuid, **moved)
        return row

    monkeypatch.setattr(training_routes.training_runs, "get_run", get_run)


def test_a_yes_to_quitting_comfyui_is_not_given_to_a_run_that_started(monkeypatch):
    temp_dir, owner, server, _read_token = _setup()
    try:
        hub = server.vault.hub
        job_id, job_uuid = _seed_job(server, status="queued", completed_at=None)
        _read_before(monkeypatch, hub, job_uuid, {"status": "running"})

        r = owner.post(f"/training/jobs/{job_id}/quit-comfyui")

        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "That run is not waiting."
        assert training_runs.get_run_by_uuid(hub, job_uuid)["comfyui_quit"] is None
    finally:
        _teardown(temp_dir, server)


def test_a_yes_to_quitting_comfyui_after_the_quit_keeps_the_quit(monkeypatch):
    temp_dir, owner, server, _read_token = _setup()
    try:
        hub = server.vault.hub
        job_id, job_uuid = _seed_job(server, status="queued", completed_at=None)
        _read_before(monkeypatch, hub, job_uuid, {"comfyui_quit": "quit"})

        r = owner.post(f"/training/jobs/{job_id}/quit-comfyui")

        assert r.status_code == 200, r.text
        assert r.json() == {"status": "allowed", "job_id": job_id}
        assert training_runs.get_run_by_uuid(hub, job_uuid)["comfyui_quit"] == "quit"
    finally:
        _teardown(temp_dir, server)


@pytest.mark.parametrize(
    ("failure", "status_code"),
    [(LookupError("That step is gone."), 409), (RuntimeError("example-vault"), 500)],
    ids=["refused", "broken"],
)
def test_a_failed_cover_choice_keeps_a_newer_one(failure, status_code):
    temp_dir, owner, server, _read_token = _setup()
    try:
        hub = server.vault.hub
        job_id, job_uuid, _, _ = _live_run(server, temp_dir)

        def newer_choice_then_fail(job, step):
            training_runs.update_run(hub, job_uuid, cover_step=750)
            raise failure

        server.vault.choose_training_cover = newer_choice_then_fail

        r = owner.put(f"/training/jobs/{job_id}/cover", json={"step": 1000})

        assert r.status_code == status_code, r.text
        assert training_runs.get_run_by_uuid(hub, job_uuid)["cover_step"] == 750, (
            "the rollback undid a cover chosen meanwhile"
        )
    finally:
        _teardown(temp_dir, server)


def test_a_failed_cover_choice_puts_the_old_step_back():
    # The positive control: with no newer choice, the rollback lands.
    temp_dir, owner, server, _read_token = _setup()
    try:
        hub = server.vault.hub
        job_id, job_uuid, _, _ = _live_run(server, temp_dir, cover_step=750)

        def fail(job, step):
            raise LookupError("That step is gone.")

        server.vault.choose_training_cover = fail

        r = owner.put(f"/training/jobs/{job_id}/cover", json={"step": 1000})

        assert r.status_code == 409, r.text
        assert training_runs.get_run_by_uuid(hub, job_uuid)["cover_step"] == 750
    finally:
        _teardown(temp_dir, server)
