"""A restore holds background work off the library, and leaves no run active.

Three properties, each against a real ``RestoreService`` on a real ``Server``:

- **The drain.** No full, resource or batch restore swaps or rewrites
  rows while a background task runs. The restore pauses a real ``TaskRunner``
  with ``pause_and_drain``; when a task is still running at the deadline it
  refuses with nothing changed and names the task, and admission resumes and
  the planner restarts on every exit.
- **Run reconciliation.** A snapshot taken while a Scrapheap or Auto-stack
  run was active brings that row back ``running``; after the restore the row is
  ``interrupted`` and the in-memory Scrapheap registry agrees with it.
- **The routes' contract.** An id outside SQLite's integer range is a 422, not a
  500; an unexpected failure's 500, and its ``RESTORE_FAILED`` event, carry a
  fixed sentence and never the exception's text.

The server runs with background workers disabled, so the vault has no runner
or planner of its own; each test puts a real ``TaskRunner`` and a recording
planner in their place.
"""

import ast
import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event as sa_event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import delete, select

from pixlstash.db_models import (
    DeletedFileLog,
    Face,
    Picture,
    PictureSet,
    PictureSetMember,
    Tag,
    User,
)
from pixlstash.db_models.dedup import (
    AUTO_STACK_INTERRUPTED,
    AUTO_STACK_PENDING,
    AUTO_STACK_RUNNING,
    SCAN_PENDING,
    DedupAutoStackRun,
    DedupScan,
)
from pixlstash.db_models.scrapheap_run import ScrapheapRunRecord
from pixlstash.db_models.snapshot import Snapshot
from pixlstash.db_models.tag_health import TagHealth
from pixlstash.db_models.tag import DEFAULT_SMART_SCORE_PENALIZED_TAGS
from pixlstash.event_types import EventType
from pixlstash.middleware.library_admission import person_request
from pixlstash.resources.arbiter import ResourceArbiter
from pixlstash.server import Server
from pixlstash.services import (
    scrapheap_run_service,
    tag_exclusion_service,
    tag_health_service,
)
from pixlstash.services.restore import full_restore as full_restore_module
from pixlstash.services.library_generation_coordinator import (
    LibrarySwitchTimeoutError,
)
from pixlstash.services.restore._models import (
    RECONCILE_RUNS_FAILED_MESSAGE,
    TASK_LABELS,
    task_label,
)
from pixlstash.services.restore import (
    RESOURCE_RESTORE_FAILED_MESSAGE,
    RESTORE_FAILED_MESSAGE,
    BackgroundWorkStillRunningError,
    MissingDependenciesError,
    RestoreRefusedError,
    WorkInProgressError,
)
from pixlstash.services.scrapheap_run_service import (
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    ScrapheapRun,
    ScrapheapRunRegistry,
)
from pixlstash.task_runner import TaskAdmissionRefused, TaskRunner
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority, TaskStatus
from pixlstash.tasks.device_release_task import DeviceReleaseTask
from pixlstash.tasks.task_type import TaskType
from pixlstash.utils.snapshot_compression import materialize_snapshot
from tests.utils import wipe_tables
from tests.package_sources import package_sources

API = "/api/v1"
SECRET_DETAIL = "test-secret-detail /home/me/library/vault.db"
OUT_OF_RANGE = (2**63, -(2**63) - 1)
IN_RANGE_MISSING = 2**63 - 1


class _NoCard:
    """A GPU ledger with no card, as ``tests/test_task_preemption.py`` uses."""

    def devices(self, max_age_s=1.0):
        return []

    def device(self, uuid):
        return None

    def pixlstash_device(self):
        return None

    def other_processes(self, limit=3):
        return []


class _RecordingPlanner:
    """Stands in for the WorkPlanner: counts the restore's stop and start."""

    def __init__(self):
        self.stops = 0
        self.starts = 0

    def stop(self):
        self.stops += 1

    def start(self):
        self.starts += 1

    def is_running(self):
        return self.starts >= self.stops

    def wake(self):
        return None


class _StubbornTask(BaseTask):
    """A batch that ignores cancellation and ends only when released."""

    def __init__(self):
        super().__init__(task_type="DescriptionTask")
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.started.set()
        self.release.wait(30)
        return "done"


class _CooperativeTask(BaseTask):
    """A batch that stops at its next item once asked to."""

    def __init__(self):
        super().__init__(task_type="TagTask")
        self.started = threading.Event()
        self.stopped = threading.Event()

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.started.set()
        self._cancel_event.wait(30)
        self.stopped.set()
        return "stopped"


class _QuickTask(BaseTask):
    """Proves admission is open: it runs to completion when submitted."""

    def __init__(self):
        super().__init__(task_type="QualityTask")
        self.ran = threading.Event()

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.ran.set()
        return "ran"


class _InteractiveTask(_CooperativeTask):
    """A picture-menu action: URGENT. Submitted inside ``person_request()``,
    as the action's request submits it, it is a person's."""

    @property
    def priority(self):
        return TaskPriority.URGENT


class _UserJobTask(_QuickTask):
    """A job a person starts from the UI: an import. Submitted inside
    ``person_request()`` it is theirs; its type alone does not make it so."""

    def __init__(self):
        BaseTask.__init__(self, task_type="PictureImportTask")
        self.ran = threading.Event()


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def clean_db(server):
    server.vault.db.run_task(
        wipe_tables,
        [
            Snapshot,
            ScrapheapRunRecord,
            DedupAutoStackRun,
            DedupScan,
            DeletedFileLog,
            PictureSetMember,
            PictureSet,
            Face,
            Picture,
        ],
    )
    snapshot_dir = os.path.join(server.vault.image_root, "snapshots")
    if os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    server.vault.scrapheap_runs = ScrapheapRunRegistry()
    yield


@pytest.fixture
def background(server, monkeypatch):
    """A real TaskRunner and a recording planner on the vault for one test."""
    runner = TaskRunner(
        name=f"restore-drain-{uuid.uuid4().hex[:8]}",
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    planner = _RecordingPlanner()
    monkeypatch.setattr(server.vault, "_task_runner", runner)
    monkeypatch.setattr(server.vault, "_work_planner", planner)
    runner.start()
    held: list[_StubbornTask] = []
    try:
        yield runner, planner, held
    finally:
        for task in held:
            task.release.set()
        runner.stop()


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
def events(server, monkeypatch):
    """The restore lifecycle events the vault emitted, in order."""
    seen: list[tuple[EventType, dict]] = []
    original = server.vault.emit_event

    def _record(event_type, data=None):
        seen.append((event_type, data))
        return original(event_type, data)

    monkeypatch.setattr(server.vault, "emit_event", _record)
    return seen


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _admission_is_open(runner) -> bool:
    probe = _QuickTask()
    runner.submit(probe)
    return probe.ran.wait(5) and _until(lambda: probe.status == TaskStatus.COMPLETED)


def _hold_a_running_task(runner, held) -> _StubbornTask:
    stuck = _StubbornTask()
    held.append(stuck)
    runner.submit(stuck)
    assert stuck.started.wait(5)
    return stuck


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


def _snapshot_ids(server) -> set[int]:
    return set(
        server.vault.db.run_immediate_read_task(
            lambda s: s.exec(select(Snapshot.id)).all()
        )
    )


def _put_in_a_set(server, picture_id: int, name: str) -> int:
    def _do(session):
        picture_set = PictureSet(name=name)
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()
        return picture_set.id

    return server.vault.db.run_task(_do)


def _delete_set(server, set_id: int) -> None:
    def _do(session):
        session.exec(delete(PictureSetMember).where(PictureSetMember.set_id == set_id))
        session.exec(delete(PictureSet).where(PictureSet.id == set_id))
        session.commit()

    server.vault.db.run_task(_do)


def _set_ids(server) -> set[int]:
    return set(
        server.vault.db.run_immediate_read_task(
            lambda s: s.exec(select(PictureSet.id)).all()
        )
    )


def _busy_with_background_work(runner) -> tuple[_CooperativeTask, _QuickTask]:
    """A planner-style batch running on the GPU worker, and one queued behind it.

    A drain would ask the first to stop and cancel the second.
    """
    busy = _CooperativeTask()
    runner.submit(busy)
    assert busy.started.wait(5)
    queued = _QuickTask()
    runner.submit(queued)
    return busy, queued


def _assert_untouched(busy, queued, planner) -> None:
    assert not busy._cancel_event.is_set(), "a running task was asked to stop"
    assert busy.status == TaskStatus.RUNNING
    assert queued.status == TaskStatus.PENDING, "a queued task was cancelled"
    assert not queued.ran.is_set()
    assert planner.stops == 0, "the planner was stopped"


def _finish(busy, queued) -> None:
    busy._cancel_event.set()
    assert _until(lambda: busy.status == TaskStatus.COMPLETED)
    assert queued.ran.wait(5)


# ---------------------------------------------------------------------------
# The drain
# ---------------------------------------------------------------------------


def test_full_restore_refuses_while_a_task_will_not_stop(
    server, background, owner, events, monkeypatch
):
    runner, planner, held = background
    pic_id = _add_picture(server, "drain-full.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    snapshots_before = _snapshot_ids(server)
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 0.3)
    stuck = _hold_a_running_task(runner, held)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "Descriptions" in detail
    assert "DescriptionTask" not in detail
    assert "Nothing was changed" in detail
    # Refused in the drain, before RESTORE_STARTED: no tab was told a restore
    # began, so none is told it failed either (no reload anywhere).
    assert [t for t, _ in events if t in _LIFECYCLE] == []
    # Nothing changed: no safety snapshot, no swap.
    assert _snapshot_ids(server) == snapshots_before
    assert _description(server, pic_id) == "after the snapshot"
    assert not server.auth.is_auth_closed_for_restore()
    assert (planner.stops, planner.starts) == (1, 1)
    stuck.release.set()
    assert _until(lambda: stuck.status == TaskStatus.COMPLETED)
    assert _admission_is_open(runner), "admission did not resume after a refusal"


def test_full_restore_waits_for_a_cooperative_task_then_restores(
    server, background, owner
):
    runner, planner, _held = background
    pic_id = _add_picture(server, "drain-coop.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    busy = _CooperativeTask()
    runner.submit(busy)
    assert busy.started.wait(5)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 200, response.text
    assert busy.stopped.is_set(), "the restore swapped while the task still ran"
    assert _description(server, pic_id) == "in the snapshot"
    assert (planner.stops, planner.starts) == (1, 1)
    assert _admission_is_open(runner)


def test_a_full_restore_dry_run_does_not_drain(server, background):
    runner, planner, held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    stuck = _hold_a_running_task(runner, held)

    report = server.vault.restore_service.restore_full(cp.id, dry_run=True)

    assert report.snapshot_id == cp.id
    assert stuck.status == TaskStatus.RUNNING
    assert (planner.stops, planner.starts) == (0, 0)


def test_a_failure_past_the_drain_still_resumes_admission(
    server, background, monkeypatch
):
    """The auth gate's close raising (its own drain timing out) is past the
    background drain: admission resumes and the planner restarts anyway."""
    runner, planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _close_fails(*args, **kwargs):
        raise RuntimeError("Timed out draining test- requests")

    monkeypatch.setattr(
        server.vault.auth_service, "close_auth_for_restore", _close_fails
    )
    with pytest.raises(RuntimeError, match="Timed out draining"):
        server.vault.restore_service.restore_full(cp.id)

    assert (planner.stops, planner.starts) == (1, 1)
    assert _admission_is_open(runner)


def test_resource_restore_refuses_while_a_task_will_not_stop(
    server, background, monkeypatch
):
    runner, planner, held = background
    pic_id = _add_picture(server, "drain-resource.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 0.3)
    stuck = _hold_a_running_task(runner, held)

    with pytest.raises(BackgroundWorkStillRunningError) as refused:
        server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert refused.value.still_running == ["Descriptions"]
    assert _description(server, pic_id) == "after the snapshot"
    assert (planner.stops, planner.starts) == (1, 1)
    stuck.release.set()
    assert _until(lambda: stuck.status == TaskStatus.COMPLETED)
    assert _admission_is_open(runner)

    # Positive control: with nothing running the same restore goes through.
    report = server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)
    assert report.upserted_count >= 1
    assert _description(server, pic_id) == "in the snapshot"


def test_batch_restore_refuses_while_a_task_will_not_stop(
    server, background, owner, monkeypatch
):
    runner, planner, held = background
    pic_id = _add_picture(server, "drain-batch.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 0.3)
    stuck = _hold_a_running_task(runner, held)
    body = {"resources": [{"type": "picture", "id": pic_id}]}

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/batch", json=body)

    assert response.status_code == 409, response.text
    assert "Descriptions" in response.json()["detail"]
    assert _description(server, pic_id) == "after the snapshot"
    assert (planner.stops, planner.starts) == (1, 1)
    stuck.release.set()
    assert _until(lambda: stuck.status == TaskStatus.COMPLETED)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/batch", json=body)
    assert response.status_code == 200, response.text
    assert _description(server, pic_id) == "in the snapshot"
    assert _admission_is_open(runner)


# ---------------------------------------------------------------------------
# A refused restore changes nothing, background work included
# ---------------------------------------------------------------------------


def test_a_resource_restore_refused_for_a_missing_parent_leaves_work_alone(
    server, background
):
    """The dependency refusal comes before the drain: the first click of
    a restore whose picture set was deleted asks for confirmation without
    stopping the planner or a single task."""
    runner, planner, _held = background
    pic_id = _add_picture(server, "refused-resource.jpg", "in the snapshot")
    set_id = _put_in_a_set(server, pic_id, "test-refused-resource")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _delete_set(server, set_id)
    _set_description(server, pic_id, "after the snapshot")
    busy, queued = _busy_with_background_work(runner)

    with pytest.raises(MissingDependenciesError) as refused:
        server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert refused.value.missing == {"picture_sets": [set_id]}
    _assert_untouched(busy, queued, planner)
    assert _description(server, pic_id) == "after the snapshot"
    _finish(busy, queued)
    # Positive control: confirmed, it drains and brings the set back.
    report = server.vault.restore_service.restore_resource(
        cp.id, "picture", pic_id, confirm_restore_dependencies=True
    )
    assert report.upserted_count >= 1
    assert set_id in _set_ids(server)
    assert (planner.stops, planner.starts) == (1, 1)


def test_a_batch_restore_refused_for_a_missing_parent_leaves_work_alone(
    server, background, owner
):
    runner, planner, _held = background
    pic_id = _add_picture(server, "refused-batch.jpg", "in the snapshot")
    set_id = _put_in_a_set(server, pic_id, "test-refused-batch")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _delete_set(server, set_id)
    busy, queued = _busy_with_background_work(runner)
    body = {"resources": [{"type": "picture", "id": pic_id}]}

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/batch", json=body)

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "missing_dependencies"
    assert detail["missing"] == {"picture_sets": [set_id]}
    _assert_untouched(busy, queued, planner)
    _finish(busy, queued)


def test_a_parent_deleted_during_a_batch_is_not_brought_back_unasked(
    server, background, monkeypatch
):
    """The batch checked its parents before the drain and found them all.
    One deleted while the drain waits is missing at the write, and nobody
    confirmed bringing it back: that item is refused, not silently given its
    set back."""
    runner, planner, _held = background
    pic_id = _add_picture(server, "batch-race.jpg", "in the snapshot")
    set_id = _put_in_a_set(server, pic_id, "test-batch-race")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    drain = runner.pause_and_drain

    def _set_deleted_while_draining(timeout_s, reason):
        _delete_set(server, set_id)
        return drain(timeout_s, reason)

    monkeypatch.setattr(runner, "pause_and_drain", _set_deleted_while_draining)

    report = server.vault.restore_service.restore_batch(
        cp.id, [{"type": "picture", "id": pic_id}]
    )

    assert set_id not in _set_ids(server), "the deleted set was brought back"
    assert report.upserted_count == 0
    assert report.errors == [
        f"picture/{pic_id}: not restored, because a character, picture set or "
        "project it needs was deleted while the restore ran. Restore it again "
        "to bring that back too."
    ]
    assert _description(server, pic_id) == "after the snapshot"
    assert (planner.stops, planner.starts) == (1, 1)


def _add_face_with_box_text(server, picture_id: int, bbox_text: str) -> int:
    """A face whose stored box is *bbox_text*, written by raw SQL: the only
    way a box the model refuses gets into a row."""

    def _do(session):
        face = Face(picture_id=picture_id, face_index=0, bbox=[1, 2, 3, 4])
        session.add(face)
        session.flush()
        session.execute(
            text("UPDATE face SET bbox = :bbox WHERE id = :id"),
            {"bbox": bbox_text, "id": face.id},
        )
        session.commit()
        return face.id

    return server.vault.db.run_task(_do)


def test_a_batch_item_refused_by_the_restore_says_why(server):
    """A refusal the restore composed reaches the batch report as its own
    sentence, not the generic "could not be restored"; the item is not
    written and the rest of the batch is."""
    refused_id = _add_picture(server, "batch-refused.jpg", "in the snapshot")
    face_id = _add_face_with_box_text(server, refused_id, "0,0,10,10")
    fine_id = _add_picture(server, "batch-fine.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, refused_id, "live")
    _set_description(server, fine_id, "live")

    report = server.vault.restore_service.restore_batch(
        cp.id,
        [{"type": "picture", "id": refused_id}, {"type": "picture", "id": fine_id}],
    )

    assert report.errors == [
        f"picture/{refused_id}: Face {face_id} on picture {refused_id} in the "
        "snapshot has a box that cannot be read, so the restore was refused. "
        "Nothing was changed."
    ]
    assert _description(server, refused_id) == "live"
    # Positive control: the other item of the same batch was restored.
    assert _description(server, fine_id) == "in the snapshot"


def test_a_restore_does_not_start_the_planner_it_did_not_stop(server, background):
    """A planner that was not running (held off by a test, or stopped for
    shutdown) is left as it was."""
    _runner, planner, _held = background
    planner.stops = 1  # not running
    pic_id = _add_picture(server, "planner-off.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert (planner.stops, planner.starts) == (1, 0)


# ---------------------------------------------------------------------------
# A restore never stops work somebody started
# ---------------------------------------------------------------------------


def test_a_resource_restore_refuses_while_an_interactive_task_runs(server, background):
    runner, planner, _held = background
    pic_id = _add_picture(server, "interactive.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    clicked = _InteractiveTask()
    with person_request():
        runner.submit(clicked)
    assert clicked.started.wait(5)
    queued = _QuickTask()
    runner.submit(queued)

    with pytest.raises(RestoreRefusedError) as refused:
        server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert str(refused.value) == (
        "The restore did not start because this work is still queued or "
        "running, and the restore would have stopped it: Tagging. Nothing was "
        "changed. Try again once it has finished."
    )
    _assert_untouched(clicked, queued, planner)
    assert _description(server, pic_id) == "after the snapshot"
    _finish(clicked, queued)
    # Positive control: with the user's task done, the restore goes through.
    server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)
    assert _description(server, pic_id) == "in the snapshot"
    assert (planner.stops, planner.starts) == (1, 1)


def test_a_batch_restore_refuses_while_a_job_somebody_started_is_queued(
    server, background, owner
):
    runner, planner, _held = background
    pic_id = _add_picture(server, "queued-job.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    busy, queued = _busy_with_background_work(runner)
    job = _UserJobTask()
    with person_request():
        runner.submit(job)
    body = {"resources": [{"type": "picture", "id": pic_id}]}

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/batch", json=body)

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "Import" in detail
    assert "PictureImportTask" not in detail
    _assert_untouched(busy, queued, planner)
    assert job.status == TaskStatus.PENDING, "the queued import was cancelled"
    assert _description(server, pic_id) == "after the snapshot"
    _finish(busy, queued)
    assert job.ran.wait(5)


def test_a_full_restore_refuses_while_a_request_waits_on_a_task(
    server, background, owner, monkeypatch
):
    runner, planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    snapshots_before = _snapshot_ids(server)
    switches = []
    begin_switch = server.library_coordinator.begin_switch

    def _recording_begin_switch(*args, **kwargs):
        switches.append(args)
        return begin_switch(*args, **kwargs)

    monkeypatch.setattr(
        server.library_coordinator, "begin_switch", _recording_begin_switch
    )
    awaited = _CooperativeTask()
    waiter = threading.Thread(
        target=runner.submit_and_wait, args=(awaited, 30), daemon=True
    )
    waiter.start()
    assert awaited.started.wait(5)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 409, response.text
    assert "Tagging" in response.json()["detail"]
    assert not awaited._cancel_event.is_set(), "the awaited task was stopped"
    assert planner.stops == 0
    # Refused before library admission closed, so no request was turned away
    # for a restore that never ran.
    assert switches == []
    assert _snapshot_ids(server) == snapshots_before
    assert not server.auth.is_auth_closed_for_restore()
    awaited._cancel_event.set()
    waiter.join(5)
    assert awaited.status == TaskStatus.COMPLETED


def _add_waiting_work(server, runner, blocker: str):
    """Leave work a person asked for waiting; return how a refusal names it
    and a callable that lets it finish."""
    if blocker == "a person's task":
        clicked = _InteractiveTask()
        with person_request():
            runner.submit(clicked)
        assert clicked.started.wait(5)

        def _finish_clicked():
            clicked._cancel_event.set()
            assert _until(lambda: clicked.status == TaskStatus.COMPLETED)

        return "Tagging", _finish_clicked
    if blocker == "an Auto-stack run":
        row = DedupAutoStackRun(
            request_id=f"test-{uuid.uuid4().hex}", status=AUTO_STACK_PENDING
        )
        label = "Auto-stack"
    else:
        row = DedupScan(scope_key=f"test-{uuid.uuid4().hex}", status=SCAN_PENDING)
        label = "Duplicate scan"

    def _insert(session):
        session.add(row)
        session.commit()

    server.vault.db.run_task(_insert)
    return label, lambda: server.vault.db.run_task(
        wipe_tables, [DedupAutoStackRun, DedupScan]
    )


def _restore(server, kind: str, snapshot_id: int, picture_id: int):
    service = server.vault.restore_service
    if kind == "full":
        return service.restore_full(snapshot_id)
    if kind == "resource":
        return service.restore_resource(snapshot_id, "picture", picture_id)
    return service.restore_batch(snapshot_id, [{"type": "picture", "id": picture_id}])


_LIFECYCLE = {
    EventType.RESTORE_STARTED,
    EventType.RESTORE_FAILED,
    EventType.RESTORE_COMPLETED,
}


@pytest.mark.parametrize("kind", ["full", "resource", "batch"])
@pytest.mark.parametrize(
    "blocker", ["a person's task", "an Auto-stack run", "a duplicate scan"]
)
def test_waiting_work_refuses_a_restore_before_it_starts(
    server, background, events, monkeypatch, kind, blocker
):
    """#3, C, D: work a person asked for, queued, running or waiting in its
    row, refuses every kind of restore before ``RESTORE_STARTED``, before
    library admission closes and before the planner stops, so no tab
    reloads and nothing waits for a restore that never ran."""
    runner, planner, _held = background
    pic_id = _add_picture(server, f"waiting-{uuid.uuid4().hex}.jpg", "snapshotted")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "after the snapshot")
    snapshots_before = _snapshot_ids(server)
    switches = []
    begin_switch = server.library_coordinator.begin_switch

    def _recording_begin_switch(*args, **kwargs):
        switches.append(args)
        return begin_switch(*args, **kwargs)

    monkeypatch.setattr(
        server.library_coordinator, "begin_switch", _recording_begin_switch
    )
    label, finish = _add_waiting_work(server, runner, blocker)

    with pytest.raises(WorkInProgressError) as refused:
        _restore(server, kind, cp.id, pic_id)

    assert f"would have stopped it: {label}. Nothing was changed." in str(refused.value)
    assert [t for t, _ in events if t in _LIFECYCLE] == []
    assert planner.stops == 0
    assert switches == []
    assert _snapshot_ids(server) == snapshots_before
    assert _description(server, pic_id) == "after the snapshot"
    assert not server.auth.is_auth_closed_for_restore()
    finish()
    # Positive control: with the work done, the same restore goes through.
    _restore(server, kind, cp.id, pic_id)
    assert _description(server, pic_id) == "snapshotted"


def test_a_full_restore_refused_for_missing_files_stops_nothing(
    server, background, events, monkeypatch
):
    """D: the missing-file check is pre-flight. Most of the snapshot's files
    gone refuses with nothing drained, no admission closed, no safety
    snapshot and no event."""
    runner, planner, _held = background
    names = [f"preflight-{i}.jpg" for i in range(10)]
    for name in names:
        _add_picture(server, name, "snapshotted")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    for name in names[:7]:
        os.remove(os.path.join(server.vault.image_root, name))
    snapshots_before = _snapshot_ids(server)
    switches = []
    begin_switch = server.library_coordinator.begin_switch

    def _recording_begin_switch(*args, **kwargs):
        switches.append(args)
        return begin_switch(*args, **kwargs)

    monkeypatch.setattr(
        server.library_coordinator, "begin_switch", _recording_begin_switch
    )
    busy, queued = _busy_with_background_work(runner)

    with pytest.raises(RestoreRefusedError) as refused:
        server.vault.restore_service.restore_full(cp.id)

    assert str(refused.value) == (
        "Not restored: 7 of 10 pictures in this snapshot (70%) are missing "
        "from the library folder. That usually means the folder is not fully "
        "reachable, such as a network drive that is not connected. Check the "
        "folder and try again."
    )
    _assert_untouched(busy, queued, planner)
    assert switches == []
    assert _snapshot_ids(server) == snapshots_before
    assert [t for t, _ in events if t in _LIFECYCLE] == []
    assert not server.auth.is_auth_closed_for_restore()
    _finish(busy, queued)


class _PlannerThatWillNotStart(_RecordingPlanner):
    def start(self):
        super().start()
        raise RuntimeError("test-planner-start-failure")


def test_a_planner_that_will_not_restart_does_not_hide_the_refusal(
    server, background, monkeypatch, caplog
):
    """The restore's own refusal is the error reported; the planner failing
    to restart after it is logged with its traceback."""
    runner, _planner, held = background
    planner = _PlannerThatWillNotStart()
    monkeypatch.setattr(server.vault, "_work_planner", planner)
    pic_id = _add_picture(server, "planner-fails.jpg", "snapshotted")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    monkeypatch.setattr(server.vault.restore_service, "_DRAIN_TIMEOUT_S", 0.3)
    _hold_a_running_task(runner, held)

    with pytest.raises(BackgroundWorkStillRunningError):
        server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert (planner.stops, planner.starts) == (1, 1)
    logged = [
        r
        for r in caplog.records
        if "WorkPlanner did not restart after a failed restore" in r.getMessage()
    ]
    assert len(logged) == 1
    assert "test-planner-start-failure" in str(logged[0].exc_info[1])


# ---------------------------------------------------------------------------
# Work a person started is known by how it was submitted
# ---------------------------------------------------------------------------

_PIXLSTASH_ROOT = Path(__file__).resolve().parent.parent / "pixlstash"

# Every call that hands a task to the TaskRunner, by file and enclosing
# function, and whose work it is. ``TaskRunner.submit`` decides by how a task
# was submitted (``person_request_active``, ``submit_and_wait``), so a new
# site needs no list to be right; it is listed here so that whoever adds one
# says whose work it is, and a site that writes before it submits says what it
# undoes on ``TaskAdmissionRefused``.
_SUBMIT_SITES = {
    "routes/characters.py::create_router.search_by_character_likeness": (
        "person: waits on the task (submit_and_wait)"
    ),
    "routes/folder_structure.py::create_router._face_detector.detect": (
        "person: waits on the task (submit_and_wait)"
    ),
    "routes/model_folders.py::queue_model_folder_scan": (
        "person from a POST; machine from the config watcher's thread"
    ),
    "routes/pictures/_anomaly.py::register_routes.get_picture_anomaly_region": (
        "person: waits on the task (submit_and_wait)"
    ),
    "routes/pictures/_character_likeness.py::register_routes."
    "score_character_likeness": "person: waits on the task (submit_and_wait)",
    "routes/pictures/_crud.py::register_routes.detect_pictures": "person: POST",
    "routes/pictures/_face_search.py::register_routes.search_by_face_likeness": (
        "person: waits on the task (submit_and_wait)"
    ),
    "routes/pictures/_import.py::register_routes.commit_staging_session": (
        "person: POST"
    ),
    "routes/pictures/_text.py::register_routes.read_picture_text": "person: POST",
    "services/scrapheap_run_service.py::_launch": (
        "person: POST; the row and registry entry are undone on a refusal"
    ),
    "services/tag_exclusion_service.py::start_cleanup": "person: config PATCH",
    "services/tag_health_service.py::start_rebuild": (
        "person from its POST; machine from the auto-rebuild task"
    ),
    "task_runner.py::TaskRunner.submit_and_wait": "the mechanism",
    "vault.py::Vault.retag_picture_interactive": "person: POST",
    "vault.py::Vault.redescribe_picture_interactive": "person: POST",
    "vault.py::Vault.submit_task": "the mechanism",
    "vault.py::Vault._unload_models_on_worker": "never: a DeviceReleaseTask",
    "vault.py::Vault._queue_cuda_suspend": "never: a DeviceReleaseTask",
    "work_planner.py::WorkPlanner._sweep_finder": "machine: the planner's thread",
    "work_planner.py::WorkPlanner._notify_all_tasks_complete": (
        "machine: the planner's thread"
    ),
}

# Calls named like a submit whose receiver is not a TaskRunner: the database
# writer queue, a thread pool, a training backend.
_NOT_THE_TASK_RUNNER = {"self._db", "self.db", "self._database", "pool", "backend"}
_SUBMIT_NAMES = {"submit", "submit_task", "submit_and_wait"}


def _python_sources():
    for path in package_sources(_PIXLSTASH_ROOT):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _task_runner_submit_sites() -> set[str]:
    sites = set()
    for path, tree in _python_sources():
        relative = path.relative_to(_PIXLSTASH_ROOT).as_posix()
        scope: list[str] = []

        def _visit(node):
            named = isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if named:
                scope.append(node.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SUBMIT_NAMES
            ):
                receiver = ast.unparse(node.func.value)
                writer_queue = (
                    relative == "database.py"
                    and receiver == "self"
                    and node.func.attr == "submit_task"
                )
                if receiver not in _NOT_THE_TASK_RUNNER and not writer_queue:
                    sites.add(f"{relative}::{'.'.join(scope)}")
            for child in ast.iter_child_nodes(node):
                _visit(child)
            if named:
                scope.pop()

        _visit(tree)
    return sites


def _declared_task_types() -> set[str]:
    """Every ``task_type`` a task is constructed with anywhere in pixlstash/:
    the ``TaskType`` values, each ``task_type="..."``, and each
    ``DeviceReleaseTask`` label."""
    sources = list(_python_sources())
    constants = {
        node.targets[0].id: node.value.value
        for _path, tree in sources
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    declared = {t.value for t in TaskType}
    for path, tree in sources:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "task_type" and isinstance(
                    keyword.value, ast.Constant
                ):
                    declared.add(keyword.value.value)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "DeviceReleaseTask"
                and node.args
            ):
                label = node.args[0]
                if isinstance(label, ast.Constant):
                    declared.add(label.value)
                elif isinstance(label, ast.Name):
                    declared.add(constants[label.id])
                else:
                    raise AssertionError(
                        f"{path}:{node.lineno}: a DeviceReleaseTask label the "
                        "test cannot read"
                    )
    declared.discard("_stop")  # the runner's own shutdown sentinel
    return declared


def test_every_task_type_has_a_label():
    """A refusal names its work as a person calls it, so every kind of task the
    code constructs has a label, the literal ``task_type`` strings included."""
    declared = _declared_task_types()

    # The scan reaches each kind of declaration.
    assert {"TagHealthRebuildTask", "UnloadModels", "SuspendCuda", "OcrTask"} <= (
        declared
    )
    assert sorted(declared - set(TASK_LABELS)) == []
    assert sorted(set(TASK_LABELS) - declared) == []


def test_a_type_with_no_label_is_an_error_and_never_named_by_its_words():
    with pytest.raises(KeyError):
        task_label("CaptionTask")


def test_a_task_type_added_without_a_label_stops_the_server_at_import():
    """The check runs when the vault module is imported, so a TaskType added
    with no label fails at start-up rather than when a refusal names it. Run
    in a fresh interpreter: the modules this one holds are already checked."""
    script = (
        "from pixlstash.services.restore._models import TASK_LABELS\n"
        "del TASK_LABELS['OcrTask']\n"
        "import pixlstash.vault\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode != 0
    assert (
        "RuntimeError: Every TaskType needs a label in TASK_LABELS "
        "(these have none): ['OcrTask']"
    ) in result.stderr, result.stderr[-2000:]


def test_every_task_runner_submit_site_is_classified():
    """A new place that hands work to the TaskRunner says whose work it is."""
    sites = _task_runner_submit_sites()

    assert sorted(sites - set(_SUBMIT_SITES)) == [], "classify the new site"
    assert sorted(set(_SUBMIT_SITES) - sites) == [], "drop the site that went"


def test_a_task_submitted_inside_a_person_s_request_is_theirs(background):
    runner, _planner, _held = background
    theirs, machine = _QuickTask(), _QuickTask()

    with person_request():
        runner.submit(theirs)
    runner.submit(machine)

    assert runner.is_person_started(theirs)
    assert not runner.is_person_started(machine)
    assert theirs.ran.wait(5) and machine.ran.wait(5)


def test_a_thread_started_from_a_request_does_not_carry_it(background):
    """A new ``threading.Thread`` starts in an empty context on Python 3.12
    (https://docs.python.org/3.12/library/contextvars.html), so work a request
    hands to its own thread is not counted as the person's."""
    runner, _planner, _held = background
    handed_on = _QuickTask()

    with person_request():
        worker = threading.Thread(target=runner.submit, args=(handed_on,))
        worker.start()
        worker.join(5)

    assert not runner.is_person_started(handed_on)
    assert handed_on.ran.wait(5)


def test_a_device_release_submitted_in_a_request_is_not_theirs(background):
    """An idle unload is URGENT and may be queued by a request, but it is
    housekeeping the runtime asks for again: a restore drains it."""
    runner, _planner, _held = background
    release = DeviceReleaseTask("UnloadModels", lambda: None)

    with person_request():
        runner.submit(release)

    assert release.priority == TaskPriority.URGENT
    assert not runner.is_person_started(release)
    assert _until(lambda: release.status == TaskStatus.COMPLETED)


class _CountingJob(_UserJobTask):
    """A person's job that counts its queue hook."""

    def __init__(self):
        super().__init__()
        self.queued = 0

    def on_queued(self):
        self.queued += 1
        return super().on_queued()


class _SlowToQueueJob(_UserJobTask):
    """A person's job whose queue hook waits until it is let go."""

    def __init__(self):
        super().__init__()
        self.queuing = threading.Event()
        self.may_queue = threading.Event()

    def on_queued(self):
        super().on_queued()
        self.queuing.set()
        self.may_queue.wait(5)


class _SubmitAfterTheLook:
    """Stands in for the runner's lock and, once the hold in which the pause
    looked at the unsettled work ends, runs *between*: a submit landing right
    after the look, as a request on another thread can."""

    def __init__(self, lock, between):
        self._lock = lock
        self._between = between
        self.looked = False
        self._ran = False

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc_info):
        self._lock.release()
        if self.looked and not self._ran:
            self._ran = True
            self._between()
        return False


def test_while_admission_is_paused_a_person_s_task_is_refused_untouched(
    background,
):
    runner, _planner, _held = background
    assert (
        runner.pause_admission_unless("library restore", runner.is_person_started) == ()
    )
    theirs, machine = _CountingJob(), _QuickTask()
    try:
        with person_request():
            with pytest.raises(TaskAdmissionRefused) as refused:
                runner.submit(theirs)
        runner.submit(machine)

        assert str(refused.value) == (
            "Not started: a library restore is in progress. Try again once it "
            "has finished."
        )
        assert theirs.status == TaskStatus.PENDING
        assert theirs.queued == 0, "the refused task's queue hook ran"
        assert not runner.unsettled_work(lambda task: task is theirs)
        assert _until(lambda: machine.status == TaskStatus.CANCELLED)
    finally:
        runner.resume_admission("library restore")
    assert _admission_is_open(runner)


def test_a_person_s_task_submitted_just_after_the_look_is_refused(background):
    """m2: the pause looks for a person's work and pauses in one hold of the
    lock, so a person's submit that lands right after the look meets the pause
    and is refused; it is never left queued under it, to be cancelled."""
    runner, _planner, held = background
    _hold_a_running_task(runner, held)  # something for the look to look at
    late = _UserJobTask()
    outcome = []

    def _submit_late():
        with person_request():
            try:
                runner.submit(late)
            except TaskAdmissionRefused:
                outcome.append("refused")
            else:
                outcome.append("queued")

    lock = runner._active_task_lock
    stand_in = _SubmitAfterTheLook(lock, _submit_late)

    def _matching(task):
        stand_in.looked = True
        return runner.is_person_started(task)

    runner._active_task_lock = stand_in
    try:
        blocking = runner.pause_admission_unless("library restore", _matching)
    finally:
        runner._active_task_lock = lock
        runner.resume_admission("library restore")

    assert blocking == ()
    assert outcome == ["refused"]
    assert late.status == TaskStatus.PENDING


def test_a_person_s_task_still_being_queued_holds_the_pause_off(background):
    """A person's task counts as unsettled from the moment ``submit`` takes
    it, before its queue hook runs, so a pause in between sees it."""
    runner, _planner, _held = background
    job = _SlowToQueueJob()

    def _submit():
        with person_request():
            runner.submit(job)

    submitter = threading.Thread(target=_submit, daemon=True)
    submitter.start()
    assert job.queuing.wait(5)
    try:
        blocking = runner.pause_admission_unless(
            "library restore", runner.is_person_started
        )
    finally:
        job.may_queue.set()
        submitter.join(5)
        runner.resume_admission("library restore")

    assert [t.task_id for t in blocking] == [job.id]
    assert job.ran.wait(5), "the person's task was cancelled"
    assert _admission_is_open(runner)


def test_a_person_s_request_during_a_restore_is_a_409_that_creates_nothing(
    server, background, owner
):
    """B, end to end: the request's POST marks what it submits as the
    person's, the paused runner refuses it, and the Scrapheap run it would
    have started leaves no row and no registry entry."""
    runner, _planner, _held = background
    assert (
        runner.pause_admission_unless("library restore", runner.is_person_started) == ()
    )
    try:
        response = owner.post(
            f"{API}/pictures/scrapheap/runs", json={"kind": "restore"}
        )
    finally:
        runner.resume_admission("library restore")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "Not started: a library restore is in progress. Try again once it has finished."
    )
    assert (
        server.vault.db.run_immediate_read_task(
            lambda s: s.exec(select(ScrapheapRunRecord.id)).all()
        )
        == []
    )
    assert server.vault.scrapheap_runs.latest() is None
    # Positive control: with admission open the same request starts its run.
    response = owner.post(f"{API}/pictures/scrapheap/runs", json={"kind": "restore"})
    assert response.status_code == 202, response.text
    assert server.vault.scrapheap_runs.latest() is not None


def test_a_purge_refused_during_a_restore_keeps_its_confirm_token(
    server, background, owner
):
    """B: a purge spends a single-use confirm token before it submits its task,
    so the refusal must come before the spend, or the person has to confirm
    again for a purge that never started."""
    runner, _planner, _held = background
    open(os.path.join(server.vault.image_root, "binned.png"), "wb").close()

    def _bin(session):
        picture = Picture(
            file_path="binned.png",
            filename="binned.png",
            deleted=True,
            deleted_at=datetime.now(timezone.utc),
        )
        session.add(picture)
        session.commit()
        session.refresh(picture)
        return picture.id

    picture_id = server.vault.db.run_task(_bin)
    preview = owner.post(
        f"{API}/pictures/scrapheap/delete-preview", json={"ids": [picture_id]}
    )
    assert preview.status_code == 200, preview.text
    payload = {
        "kind": "purge",
        "confirm_token": preview.json()["confirm_token"],
        "picture_ids": [picture_id],
    }
    assert (
        runner.pause_admission_unless("library restore", runner.is_person_started) == ()
    )
    try:
        response = owner.post(f"{API}/pictures/scrapheap/runs", json=payload)
    finally:
        runner.resume_admission("library restore")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "Not started: a library restore is in progress. Try again once it has finished."
    )
    assert server.vault.scrapheap_runs.latest() is None
    # The same token still starts the purge once the restore is over.
    response = owner.post(f"{API}/pictures/scrapheap/runs", json=payload)
    assert response.status_code == 202, response.text


def test_a_purge_whose_submit_meets_a_restore_gets_its_confirm_token_back(
    server, background, owner, monkeypatch
):
    """B: the pre-check is a look, not a hold. A restore that pauses admission
    after the last check, before the submit, is refused by ``submit`` itself,
    once the token is spent; the route hands the token back, so the purge
    still starts on the same confirmation after the restore."""
    runner, _planner, _held = background
    open(os.path.join(server.vault.image_root, "binned-late.png"), "wb").close()

    def _bin(session):
        picture = Picture(
            file_path="binned-late.png",
            filename="binned-late.png",
            deleted=True,
            deleted_at=datetime.now(timezone.utc),
        )
        session.add(picture)
        session.commit()
        session.refresh(picture)
        return picture.id

    picture_id = server.vault.db.run_task(_bin)
    preview = owner.post(
        f"{API}/pictures/scrapheap/delete-preview", json={"ids": [picture_id]}
    )
    assert preview.status_code == 200, preview.text
    payload = {
        "kind": "purge",
        "confirm_token": preview.json()["confirm_token"],
        "picture_ids": [picture_id],
    }

    real_check = scrapheap_run_service.check_can_start
    checks = []

    def _check_then_restore_begins(vault):
        real_check(vault)
        checks.append(vault)
        # The route checks before it spends the token, and start_run checks
        # again before it reserves. The restore lands after that second
        # look, so only the submit can refuse.
        if len(checks) == 2:
            assert (
                runner.pause_admission_unless(
                    "library restore", runner.is_person_started
                )
                == ()
            )

    monkeypatch.setattr(
        scrapheap_run_service, "check_can_start", _check_then_restore_begins
    )
    try:
        response = owner.post(f"{API}/pictures/scrapheap/runs", json=payload)
    finally:
        runner.resume_admission("library restore")

    assert len(checks) == 2, "the pause landed after both looks"
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "Not started: a library restore is in progress. Try again once it has finished."
    )
    assert server.vault.scrapheap_runs.latest() is None
    monkeypatch.setattr(scrapheap_run_service, "check_can_start", real_check)
    # The same token still starts the purge once the restore is over.
    response = owner.post(f"{API}/pictures/scrapheap/runs", json=payload)
    assert response.status_code == 202, response.text


_REFUSED_DURING_RESTORE = (
    "Not started: a library restore is in progress. Try again once it has finished."
)


@pytest.mark.parametrize("route", ["reset_tags", "reset_description"])
def test_a_picture_reset_during_a_restore_is_refused_before_it_writes(
    server, background, owner, route
):
    """C: a reset writes first (drops the picture's tags, or sets the
    redescribe sentinel) and then queues the pass for it. Refused at the
    submit, the write would outlive the 409, so the route asks first."""
    runner, _planner, _held = background

    def _seed(session):
        picture = Picture(
            file_path="reset.png",
            filename="reset.png",
            description="test-kept description",
        )
        session.add(picture)
        session.flush()
        session.add(Tag(picture_id=picture.id, tag="test-kept-tag"))
        session.commit()
        return picture.id

    picture_id = server.vault.db.run_task(_seed)

    def _read(session):
        picture = session.get(Picture, picture_id)
        tags = session.exec(select(Tag.tag).where(Tag.picture_id == picture_id))
        return picture.description, sorted(tags.all())

    def _drop_tags(session):
        session.exec(delete(Tag).where(Tag.picture_id == picture_id))
        session.commit()

    try:
        assert (
            runner.pause_admission_unless("library restore", runner.is_person_started)
            == ()
        )
        try:
            response = owner.post(f"{API}/pictures/{picture_id}/{route}", json={})
        finally:
            runner.resume_admission("library restore")

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == _REFUSED_DURING_RESTORE
        assert server.vault.db.run_task(_read) == (
            "test-kept description",
            ["test-kept-tag"],
        )
    finally:
        # clean_db wipes pictures with foreign keys off, so tags would stay.
        server.vault.db.run_task(_drop_tags)


def test_a_tag_removal_during_a_restore_is_refused_before_it_writes(
    server, background, owner
):
    """C: removing a penalised tag writes the owner's settings and then starts
    the cleanup job; a retry lists its job before it submits. Both ask first,
    so a 409 leaves the settings and the job list as they were."""
    runner, _planner, _held = background
    user_id = server.auth.user.id

    def _settings(session):
        user = session.get(User, user_id)
        return user.tagger_settings, user.smart_score_penalised_tags

    settings_before = server.hub_engine.run_immediate_read_task(_settings)
    jobs_before = tag_exclusion_service._jobs(server.vault)
    tag = next(iter(DEFAULT_SMART_SCORE_PENALIZED_TAGS))
    assert (
        runner.pause_admission_unless("library restore", runner.is_person_started) == ()
    )
    try:
        removed = owner.delete(f"{API}/users/me/penalised-tags/{tag}")
        retried = owner.post(f"{API}/users/me/penalised-tags/cleanup", json={})
    finally:
        runner.resume_admission("library restore")

    for response in (removed, retried):
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == _REFUSED_DURING_RESTORE
    assert server.hub_engine.run_immediate_read_task(_settings) == settings_before
    assert tag_exclusion_service._jobs(server.vault) == jobs_before


# ---------------------------------------------------------------------------
# The end of a full restore
# ---------------------------------------------------------------------------


def test_restore_completed_is_sent_once_the_library_is_open_again(
    server, background, owner, monkeypatch
):
    """A client reacting to RESTORE_COMPLETED reads the restored library,
    so admission must be READY when it is sent, not still SWITCHING."""
    _runner, _planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    states = []
    original = server.vault.emit_event

    def _record(event_type, data=None):
        if event_type == EventType.RESTORE_COMPLETED:
            states.append(server.library_coordinator.state.value)
        return original(event_type, data)

    monkeypatch.setattr(server.vault, "emit_event", _record)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 200, response.text
    assert states == ["ready"]


def test_sign_in_reopens_when_closing_restored_runs_fails(
    server, background, monkeypatch, caplog
):
    """The swap is done; a failure closing the runs the snapshot recorded
    as active must not leave sign-in closed until a restart."""
    _runner, _planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    # A run recorded in the file the restore replaces.
    server.vault.scrapheap_runs.load([ScrapheapRun(kind="purge", status="complete")])

    def _explode(vault):
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(full_restore_module, "reload_runs", _explode)
    try:
        report = server.vault.restore_service.restore_full(cp.id)
        assert report.snapshot_id == cp.id
        assert RECONCILE_RUNS_FAILED_MESSAGE in report.errors
        assert not server.auth.is_auth_closed_for_restore()
        assert server.vault.scrapheap_runs.latest() is None
        logged = [r for r in caplog.records if r.exc_info is not None]
        assert any(SECRET_DETAIL in str(r.exc_info[1]) for r in logged)
    finally:
        if server.auth.is_auth_closed_for_restore():
            server.vault.auth_service.reopen_auth_after_restore()


def test_a_failure_naming_the_library_holder_is_still_a_409(
    server, background, owner, monkeypatch
):
    """Describing what holds the library is best effort. Its failure
    inside the refusal must not turn the refusal into a 500."""
    _runner, planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _timed_out(*args, **kwargs):
        raise LibrarySwitchTimeoutError("Timed out waiting for active-library readers")

    def _explode():
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(server.library_coordinator, "begin_switch", _timed_out)
    monkeypatch.setattr(server.library_switch, "describe_library_holders", _explode)

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 409, response.text
    assert SECRET_DETAIL not in response.text
    assert "Nothing was changed" in response.json()["detail"]
    assert planner.stops == 0


# ---------------------------------------------------------------------------
# Run reconciliation
# ---------------------------------------------------------------------------


def _insert_active_runs(server) -> tuple[str, int]:
    scrapheap_id = uuid.uuid4().hex

    def _do(session):
        session.add(
            ScrapheapRunRecord(id=scrapheap_id, kind="purge", status=STATUS_RUNNING)
        )
        auto_stack = DedupAutoStackRun(
            request_id=f"test-{uuid.uuid4().hex}", status=AUTO_STACK_RUNNING
        )
        session.add(auto_stack)
        session.commit()
        session.refresh(auto_stack)
        return auto_stack.id

    return scrapheap_id, server.vault.db.run_task(_do)


def _end_runs(server, scrapheap_id: str, auto_stack_id: int) -> None:
    def _do(session):
        session.get(ScrapheapRunRecord, scrapheap_id).status = "complete"
        session.get(DedupAutoStackRun, auto_stack_id).status = "complete"
        session.commit()

    server.vault.db.run_task(_do)


def _run_statuses(server, scrapheap_id: str, auto_stack_id: int):
    return server.vault.db.run_immediate_read_task(
        lambda s: (
            s.get(ScrapheapRunRecord, scrapheap_id).status,
            s.get(DedupAutoStackRun, auto_stack_id).status,
        )
    )


def _run_error_codes(server, scrapheap_id: str, auto_stack_id: int):
    return server.vault.db.run_immediate_read_task(
        lambda s: (
            s.get(ScrapheapRunRecord, scrapheap_id).error_code,
            s.get(DedupAutoStackRun, auto_stack_id).error_code,
        )
    )


def test_a_restore_leaves_no_run_from_the_snapshot_active(server, background):
    _runner, _planner, _held = background
    scrapheap_id, auto_stack_id = _insert_active_runs(server)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    # The runs finish after the snapshot; the swap brings back their rows as
    # the snapshot had them, running, with no task left to own them.
    _end_runs(server, scrapheap_id, auto_stack_id)
    # Taken before the restore, as a request or a task holds it.
    held_registry = server.vault.scrapheap_runs

    server.vault.restore_service.restore_full(cp.id)

    assert _run_statuses(server, scrapheap_id, auto_stack_id) == (
        STATUS_INTERRUPTED,
        AUTO_STACK_INTERRUPTED,
    )
    # Closed as restored, not as a server stop: the run did not fail.
    assert _run_error_codes(server, scrapheap_id, auto_stack_id) == (
        "library_restored",
        "library_restored",
    )
    # The same registry, reloaded: the reference held before the restore
    # reads the restored runs, not the ones the swap replaced.
    assert server.vault.scrapheap_runs is held_registry
    latest = held_registry.latest()
    assert latest is not None
    assert latest["id"] == scrapheap_id
    assert latest["status"] == STATUS_INTERRUPTED
    assert latest["active"] is False
    assert latest["error_code"] == "library_restored"
    # A new run is not refused by a stale active one.
    server.vault.scrapheap_runs.check_idle()


# ---------------------------------------------------------------------------
# The routes' contract
# ---------------------------------------------------------------------------


def _int_bound_requests(value: int) -> list[tuple[str, str, dict | None]]:
    item = [{"type": "picture", "id": 1}]
    return [
        ("PATCH", f"/snapshots/{value}", {"label": "test-label"}),
        ("DELETE", f"/snapshots/{value}", None),
        ("GET", f"/snapshots/{value}/restore/preview", None),
        ("GET", f"/snapshots/{value}/restore/picture/1/preview", None),
        ("GET", f"/snapshots/{IN_RANGE_MISSING}/restore/picture/{value}/preview", None),
        ("POST", f"/snapshots/{value}/restore/preview/batch", {"resources": item}),
        (
            "POST",
            f"/snapshots/{IN_RANGE_MISSING}/restore/preview/batch",
            {"resources": [{"type": "picture", "id": value}]},
        ),
        ("POST", f"/snapshots/{value}/restore", {}),
        ("POST", f"/snapshots/{value}/restore/batch", {"resources": item}),
        (
            "POST",
            f"/snapshots/{IN_RANGE_MISSING}/restore/batch",
            {"resources": [{"type": "picture", "id": value}]},
        ),
        ("POST", f"/snapshots/{value}/hash-compare", {"picture_ids": [1]}),
        (
            "POST",
            f"/snapshots/{IN_RANGE_MISSING}/hash-compare",
            {"picture_ids": [value]},
        ),
        ("POST", f"/snapshots/{value}/restore/picture/1", {}),
        ("POST", f"/snapshots/{IN_RANGE_MISSING}/restore/picture/{value}", {}),
    ]


@pytest.mark.parametrize("value", OUT_OF_RANGE)
def test_an_id_outside_sqlite_integers_is_a_422(server, owner, value):
    for method, path, body in _int_bound_requests(value):
        response = owner.request(method, f"{API}{path}", json=body)
        assert response.status_code == 422, (method, path, response.text)


def test_the_largest_sqlite_integer_is_a_404_not_a_422(server, owner):
    """The positive control: the same requests in range reach the handler."""
    for method, path, body in _int_bound_requests(IN_RANGE_MISSING):
        response = owner.request(method, f"{API}{path}", json=body)
        assert response.status_code == 404, (method, path, response.text)


def test_a_malformed_resource_item_is_a_422(server, owner):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    for body in (
        {"resources": [{"type": "picture"}]},
        {"resources": [{"id": 1}]},
        {"resources": [{"type": "picture", "id": "test-not-an-int"}]},
    ):
        response = owner.post(f"{API}/snapshots/{cp.id}/restore/batch", json=body)
        assert response.status_code == 422, (body, response.text)


def test_an_unexpected_full_restore_failure_shows_the_fixed_sentence(
    server, background, owner, events, monkeypatch
):
    """Past the drain: ``RESTORE_STARTED`` was sent, so ``RESTORE_FAILED``
    follows it, with the fixed sentence."""
    runner, planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    reads = []
    load_deleted_file_index = server.vault.restore_service._load_deleted_file_index

    def _explode_on_the_reread():
        reads.append(None)
        if len(reads) == 2:  # read again with background work drained
            raise RuntimeError(SECRET_DETAIL)
        return load_deleted_file_index()

    monkeypatch.setattr(
        server.vault.restore_service,
        "_load_deleted_file_index",
        _explode_on_the_reread,
    )

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 500
    assert response.json()["detail"] == RESTORE_FAILED_MESSAGE
    assert "test-secret-detail" not in response.text
    assert len(reads) == 2
    lifecycle = [t for t, _ in events if t in _LIFECYCLE]
    assert lifecycle == [EventType.RESTORE_STARTED, EventType.RESTORE_FAILED]
    failed = [p for t, p in events if t == EventType.RESTORE_FAILED]
    assert [p["error"] for p in failed] == [RESTORE_FAILED_MESSAGE]
    assert (planner.stops, planner.starts) == (1, 1)
    assert _admission_is_open(runner)


def test_an_unexpected_preflight_failure_shows_the_fixed_sentence(
    server, background, owner, events, monkeypatch
):
    """Before the drain: nothing was stopped and no ``RESTORE_STARTED`` was
    sent, so no ``RESTORE_FAILED`` follows; the 500 still hides the cause."""
    _runner, planner, _held = background
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _explode(*args, **kwargs):
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(
        server.vault.restore_service, "_find_missing_file_ids", _explode
    )

    response = owner.post(f"{API}/snapshots/{cp.id}/restore", json={})

    assert response.status_code == 500
    assert response.json()["detail"] == RESTORE_FAILED_MESSAGE
    assert "test-secret-detail" not in response.text
    assert [t for t, _ in events if t in _LIFECYCLE] == []
    assert planner.stops == 0
    assert not server.auth.is_auth_closed_for_restore()


def test_an_unexpected_resource_restore_failure_shows_the_fixed_sentence(
    server, owner, events, monkeypatch
):
    pic_id = _add_picture(server, "fail-resource.jpg", "in the snapshot")
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _explode(*args, **kwargs):
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(
        server.vault.restore_service, "_read_resource_from_snapshot", _explode
    )

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/picture/{pic_id}", json={})

    assert response.status_code == 500
    assert response.json()["detail"] == RESOURCE_RESTORE_FAILED_MESSAGE
    assert "test-secret-detail" not in response.text
    failed = [p for t, p in events if t == EventType.RESTORE_FAILED]
    assert [p["error"] for p in failed] == [RESOURCE_RESTORE_FAILED_MESSAGE]


@pytest.mark.parametrize(
    "method, path, body, attribute",
    [
        ("GET", "/snapshots/{id}/restore/preview", None, "preview_full"),
        (
            "POST",
            "/snapshots/{id}/hash-compare",
            {"picture_ids": [1]},
            "compare_hashes",
        ),
        (
            "POST",
            "/snapshots/{id}/restore/preview/batch",
            {"resources": [{"type": "picture", "id": 1}]},
            "preview_batch",
        ),
    ],
)
def test_an_unexpected_preview_failure_hides_the_exception(
    server, owner, monkeypatch, method, path, body, attribute
):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _explode(*args, **kwargs):
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(server.vault.restore_service, attribute, _explode)

    response = owner.request(method, f"{API}{path.format(id=cp.id)}", json=body)

    assert response.status_code == 500
    assert "the server log has the details" in response.json()["detail"]
    assert "test-secret-detail" not in response.text


def test_a_missing_snapshot_file_is_a_404_that_names_only_the_id(server, owner):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    os.remove(os.path.join(server.vault.image_root, cp.relative_path))

    response = owner.get(f"{API}/snapshots/{cp.id}/restore/preview")

    assert response.status_code == 404, response.text
    detail = response.json()["detail"]
    assert detail == f"The file for snapshot {cp.id} is missing on disk."
    assert "snapshots" not in detail


# ---------------------------------------------------------------------------
# Id lists longer than one statement may bind
# ---------------------------------------------------------------------------

#: SQLite's historical floor for bound parameters per statement. python.org's
#: Windows and macOS builds stop at 32,766; pinning 999 makes a plain ``IN``
#: over :data:`_PAST_THE_LIMIT` ids fail here as it would there.
_VARIABLE_FLOOR = 999
_PAST_THE_LIMIT = 1000


@contextlib.contextmanager
def _binding_at_most_the_floor(server):
    """Every SQLite connection opened inside binds at most the floor.

    Listens on ``Engine`` itself, so the engines a restore opens on the
    snapshot and the one the swap reopens are pinned as well as the live one,
    and disposes the live pool so its connections are reopened under the pin.
    The listener comes off and the pool is disposed again on the way out.
    """

    def _pin(dbapi_connection, _record):
        dbapi_connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _VARIABLE_FLOOR)

    sa_event.listen(Engine, "connect", _pin)
    try:
        server.vault.db._engine.dispose()
        # The pin took: a plain IN at this size is what it refuses.
        with pytest.raises(OperationalError, match="too many SQL variables"):
            server.vault.db.run_immediate_read_task(
                lambda s: s.exec(
                    select(Picture.id).where(
                        Picture.id.in_(list(range(_PAST_THE_LIMIT)))
                    )
                ).all()
            )
        yield
    finally:
        sa_event.remove(Engine, "connect", _pin)
        server.vault.db._engine.dispose()


def _add_pictures(server, prefix: str, count: int) -> list[int]:
    names = [f"{prefix}-{index}.jpg" for index in range(count)]
    for name in names:
        open(os.path.join(server.vault.image_root, name), "wb").close()

    def _do(session):
        pictures = [
            Picture(file_path=name, filename=name, description="snapshotted")
            for name in names
        ]
        session.add_all(pictures)
        session.flush()
        ids = [picture.id for picture in pictures]
        session.commit()
        return ids

    return server.vault.db.run_task(_do)


def _picture_ids(server) -> set[int]:
    return set(
        server.vault.db.run_immediate_read_task(
            lambda s: s.exec(select(Picture.id)).all()
        )
    )


def test_a_full_restore_drops_more_ledger_pictures_than_one_statement_binds(
    server, background, caplog
):
    """Every picture the deletion ledger names is dropped after the swap,
    however many: the ledger can name more than one ``IN`` may bind."""
    ids = _add_pictures(server, "purged", _PAST_THE_LIMIT)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    def _purge_after_the_snapshot(session):
        session.exec(delete(Picture))
        now = datetime.now(timezone.utc)
        session.add_all(
            DeletedFileLog(
                path_sha=DeletedFileLog.hash_path(f"purged-{index}.jpg"),
                deleted_at=now,
                file_removed=True,
            )
            for index in range(_PAST_THE_LIMIT)
        )
        session.commit()

    server.vault.db.run_task(_purge_after_the_snapshot)

    with _binding_at_most_the_floor(server), caplog.at_level("INFO"):
        report = server.vault.restore_service.restore_full(cp.id)

    assert report.permanently_deleted_count == _PAST_THE_LIMIT
    assert _picture_ids(server) & set(ids) == set()
    assert f"dropped {_PAST_THE_LIMIT} picture row(s) after restore" in caplog.text


def test_a_set_restore_rewrites_more_members_than_one_statement_binds(
    server, background
):
    """A picture set restores every member's dependents, however many."""
    ids = _add_pictures(server, "member", _PAST_THE_LIMIT)

    def _make_set(session):
        picture_set = PictureSet(name="example-big-set")
        session.add(picture_set)
        session.flush()
        session.add_all(
            PictureSetMember(set_id=picture_set.id, picture_id=pid) for pid in ids
        )
        session.commit()
        return picture_set.id

    set_id = server.vault.db.run_task(_make_set)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    for pid in (ids[0], ids[-1]):
        _set_description(server, pid, "changed after the snapshot")

    with _binding_at_most_the_floor(server):
        report = server.vault.restore_service.restore_resource(
            cp.id, "picture_set", set_id
        )

    assert report.errors == []
    assert _description(server, ids[0]) == "snapshotted"
    assert _description(server, ids[-1]) == "snapshotted"
    members = server.vault.db.run_immediate_read_task(
        lambda s: s.exec(
            select(PictureSetMember.picture_id).where(PictureSetMember.set_id == set_id)
        ).all()
    )
    assert sorted(members) == sorted(ids)


def _register_without_hash_sidecar(server, cp) -> int:
    """Register *cp* again as a plain ``.sqlite`` with no hash sidecar, so
    ``compare_hashes`` reads the hashes from the snapshot file itself."""
    vault_root = server.vault.image_root
    plain_rel = cp.relative_path[: -len(".zst")]
    materialize_snapshot(
        os.path.join(vault_root, cp.relative_path), os.path.join(vault_root, plain_rel)
    )

    def _do(session):
        row = Snapshot(
            kind="MANUAL",
            created_at=cp.created_at,
            relative_path=plain_rel,
            manifest_relative_path=plain_rel + ".no-sidecar.manifest.json",
            byte_size=os.path.getsize(os.path.join(vault_root, plain_rel)),
            picture_count=cp.picture_count,
            schema_version=cp.schema_version,
            label="example-plain",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id

    return server.vault.db.run_task(_do)


def test_a_hash_compare_reads_more_pictures_than_one_statement_binds(server):
    """The compare route allows 1,000 ids; both the live read and the read of
    a snapshot with no hash sidecar must take them all."""
    ids = _add_pictures(server, "compared", _PAST_THE_LIMIT)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    plain_id = _register_without_hash_sidecar(server, cp)

    with _binding_at_most_the_floor(server):
        result = server.vault.restore_service.compare_hashes(plain_id, ids)

    assert result == {"identical_ids": ids, "changed_ids": []}


# A tag-health rebuild the runner cancels before it runs (review SF1). The
# rebuild's start raises ``building``, and only the rebuild's own body took it
# down, so a rebuild cancelled unrun left the board answering "building", and
# refusing every later rebuild, until a restart.


class _HeldCpuTask(BaseTask):
    """Holds the runner's one CPU worker until released; ignores a cancel."""

    def __init__(self):
        super().__init__(task_type="MissingFilePurgeTask")
        self.started = threading.Event()
        self.release = threading.Event()

    def _run_task(self):
        self.started.set()
        self.release.wait(30)
        return "done"


@pytest.fixture
def tag_board(server, monkeypatch):
    """One tagged picture, an empty board and a free ``building`` flag.

    Yields the tasks the vault was handed, in order: the rebuilds
    ``start_rebuild`` submits.
    """
    server.vault.db.run_task(wipe_tables, [TagHealth, Tag])
    picture_id = _add_picture(server, "board.jpg", "tagged")

    def _tag(session):
        session.add(Tag(picture_id=picture_id, tag="test-board-tag"))
        session.commit()

    def _drop_tag(session):
        session.exec(delete(Tag).where(Tag.picture_id == picture_id))
        session.commit()

    server.vault.db.run_task(_tag)
    with tag_health_service._LOCK:
        tag_health_service._STATES.pop(id(server.vault), None)
    submitted: list[BaseTask] = []
    real_submit = server.vault.submit_task

    def _record(task):
        submitted.append(task)
        return real_submit(task)

    monkeypatch.setattr(server.vault, "submit_task", _record)
    try:
        yield submitted
    finally:
        # clean_db wipes pictures with foreign keys off, so the tag would stay.
        server.vault.db.run_task(_drop_tag)


def _board(server) -> list[str]:
    return server.vault.db.run_immediate_read_task(
        lambda session: list(session.exec(select(TagHealth.tag)).all())
    )


def _assert_the_next_rebuild_runs(server, submitted) -> None:
    """A rebuild asked for now is taken, runs, and fills the board."""
    already = len(submitted)
    tag_health_service.start_rebuild(server.vault)
    taken = submitted[already:]
    assert [task.type for task in taken] == ["TagHealthRebuildTask"], (
        "the rebuild was not taken: the board still answered building"
    )
    assert _until(lambda: taken[0].status == TaskStatus.COMPLETED, 10)
    assert tag_health_service.get_status(server.vault) == {
        "building": False,
        "progress": 1.0,
    }
    assert _board(server) == ["test-board-tag"]


def test_a_queued_tag_health_rebuild_the_drain_cancels_takes_building_down(
    server, background, tag_board
):
    """The drain cancels a queued rebuild without running it: the flag its
    start raised goes down with it, and the next rebuild runs."""
    runner, _planner, _held = background
    holder = _HeldCpuTask()
    runner.submit(holder)
    assert holder.started.wait(5)
    try:
        assert tag_health_service.start_rebuild(server.vault) == {
            "building": True,
            "progress": 0.0,
        }
        runner.pause_and_drain(0.2, "library restore")
        [cancelled] = tag_board
        assert cancelled.status == TaskStatus.CANCELLED
        assert tag_health_service.get_status(server.vault) == {
            "building": False,
            "progress": 0.0,
        }
    finally:
        holder.release.set()
        runner.resume_admission("library restore")
    assert _until(lambda: holder.status == TaskStatus.COMPLETED)
    assert _board(server) == []

    _assert_the_next_rebuild_runs(server, tag_board)


def test_a_tag_health_rebuild_submitted_while_admission_is_paused_leaves_building_down(
    server, background, tag_board
):
    """A machine caller (the auto-rebuild task, a tag cleanup) that asks for a
    rebuild during a restore has it cancelled at the submit, which still
    returns its id, so the start's own undo for a refused submit never ran."""
    runner, _planner, _held = background
    assert runner.pause_and_drain(1.0, "library restore").drained
    try:
        status = tag_health_service.start_rebuild(server.vault)
    finally:
        runner.resume_admission("library restore")

    [cancelled] = tag_board
    assert cancelled.status == TaskStatus.CANCELLED
    assert status == {"building": False, "progress": 0.0}
    assert _board(server) == []
    _assert_the_next_rebuild_runs(server, tag_board)


def test_a_tag_health_rebuild_cancelled_as_it_starts_leaves_a_later_one_alone(
    server, tag_board, monkeypatch
):
    """A drain or a stop can cancel a rebuild the runner has just taken, before
    or after ``run()`` reads its cancel event. Before: the flag goes down with
    the cancel. After: the body finds the flag no longer its own and does not
    rebuild, and does not take down the flag of a rebuild started since."""
    taken: list[BaseTask] = []

    def _take_and_hold(task):
        taken.append(task)
        return task.id

    monkeypatch.setattr(server.vault, "submit_task", _take_and_hold)
    tag_health_service.start_rebuild(server.vault)
    [first] = taken

    first.on_cancel()
    assert tag_health_service.get_status(server.vault) == {
        "building": False,
        "progress": 0.0,
    }
    assert first.run() is None
    assert first.status == TaskStatus.CANCELLED

    tag_health_service.start_rebuild(server.vault)
    [_, second] = taken
    # The first rebuild's body had passed run()'s check before the cancel.
    assert first._run_task() is None
    assert tag_health_service.get_status(server.vault) == {
        "building": True,
        "progress": 0.0,
    }
    assert _board(server) == []

    second.run()
    assert second.status == TaskStatus.COMPLETED
    assert tag_health_service.get_status(server.vault) == {
        "building": False,
        "progress": 1.0,
    }
    assert _board(server) == ["test-board-tag"]


def test_a_tag_health_rebuild_holds_building_while_it_runs_and_clears_it_after(
    server, background, tag_board, monkeypatch
):
    """Control: a rebuild nobody cancels holds the flag for its whole run."""
    seen: list[dict] = []
    rebuild = tag_health_service.rebuild_tag_health

    def _observed(vault):
        seen.append(tag_health_service.get_status(vault))
        return rebuild(vault)

    monkeypatch.setattr(tag_health_service, "rebuild_tag_health", _observed)
    _assert_the_next_rebuild_runs(server, tag_board)
    assert seen == [{"building": True, "progress": 0.0}]
