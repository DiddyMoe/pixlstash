"""Whether a worker is reported available agrees with the task runner's lanes.

The server must never report a worker available, or busy with work, while the
runner lane its tasks queue on has no live worker (``lane_health()`` shows
``alive == 0`` with the runner running). The availability answer must not keep
the lane dead either: it replaces the dead workers, as ``submit()`` would.

The lanes are killed the way tests/test_task_preemption.py kills them: an item
on the queue that is not a ``(priority, seq, task)`` triple fails the worker
loop itself, outside any task.
"""

import importlib
import logging
import pkgutil
from types import SimpleNamespace

import pytest

import pixlstash.tasks
from pixlstash.resources.arbiter import ResourceArbiter
from pixlstash.task_runner import TaskRunner
from pixlstash.tasks.base_task import BaseTask, QueueType
from pixlstash.tasks.task_type import TaskType
from pixlstash.vault import GPU_LANE_WORKER_TYPES, Vault


class _NoCard:
    def devices(self, max_age_s=1.0):
        return []

    def device(self, uuid):
        return None

    def pixlstash_device(self):
        return None

    def other_processes(self, limit=3):
        return []


class _Planner:
    """A live planner with a finder for every type and one task in flight."""

    def is_running(self):
        return True

    def has_finder(self, worker_type):
        return True

    def inflight_count(self, finder_name):
        return 1


def _task_classes():
    for info in pkgutil.iter_modules(pixlstash.tasks.__path__):
        importlib.import_module(f"pixlstash.tasks.{info.name}")
    pending = list(BaseTask.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        yield cls


def test_the_gpu_lane_worker_types_are_the_task_classes_on_the_gpu_queue():
    by_name: dict[str, list[type]] = {}
    for cls in _task_classes():
        by_name.setdefault(cls.__name__, []).append(cls)
    on_gpu = set()
    for worker_type in TaskType:
        classes = by_name.get(worker_type.value, [])
        assert len(classes) == 1, f"{worker_type} has task classes {classes}"
        # queue_type is a property returning a constant; no state needed.
        if object.__new__(classes[0]).queue_type == QueueType.GPU:
            on_gpu.add(worker_type)
    assert on_gpu == set(GPU_LANE_WORKER_TYPES)


@pytest.fixture
def vault(tmp_path):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=True,
        daily_snapshots_enabled=False,
    )
    runner = TaskRunner(
        name="lane-test",
        num_workers=2,
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    runner.start()
    # A vault that never started does not stop these in close(); the test does.
    v._task_runner = runner
    v._work_planner = _Planner()
    v._planner_work_finders = {
        worker_type: SimpleNamespace(finder_name=lambda: "finder")
        for worker_type in TaskType
    }
    try:
        yield v
    finally:
        v._task_runner = None
        v._work_planner = None
        runner.stop()
        v.close()


def _kill(runner, lane):
    if lane == "gpu":
        workers, work_queue = [runner._gpu_worker], runner._gpu_queue
    else:
        workers, work_queue = list(runner._cpu_workers), runner._queue
    for _ in workers:
        work_queue.put((0, -1))
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert runner.lane_health()[lane]["alive"] == 0


_ON_LANE = {"gpu": TaskType.FACE_EXTRACTION, "cpu": TaskType.SMART_SCORE}


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_a_worker_on_a_dead_lane_is_reported_idle(vault, lane):
    worker_type = _ON_LANE[lane]
    other = _ON_LANE["cpu" if lane == "gpu" else "gpu"]
    assert vault._is_worker_active(worker_type)
    _kill(vault._task_runner, lane)

    assert not vault._is_worker_active(worker_type), "a dead lane reported busy"
    assert vault._is_worker_active(other), "the live lane was reported idle"
    # The progress read does not revive the lane: it only reports.
    assert vault._task_runner.lane_health()[lane]["alive"] == 0


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_asking_for_a_worker_on_a_dead_lane_replaces_it(vault, lane):
    worker_type = _ON_LANE[lane]
    _kill(vault._task_runner, lane)

    assert vault.worker_unavailable_reason(worker_type) is None
    health = vault._task_runner.lane_health()
    assert health[lane]["alive"] == health[lane]["total"]
    assert health[lane]["restarts"] == health[lane]["total"]
    assert vault.is_worker_running(worker_type)
    assert vault._is_worker_active(worker_type)


def test_a_lane_that_stays_dead_is_named_in_the_reason(vault, monkeypatch):
    runner = vault._task_runner
    _kill(runner, "gpu")
    starts = []
    # A runner whose repair does not bring the lane back.
    monkeypatch.setattr(runner, "replace_dead_workers", lambda: starts.append(1))

    reason = vault.worker_unavailable_reason(TaskType.FACE_EXTRACTION)
    assert starts == [1], "the dead lane was not handed to the runner's repair"
    assert reason is not None and "GPU worker has exited" in reason
    assert not vault.is_worker_running(TaskType.FACE_EXTRACTION)
    # The CPU lane is alive; its workers are available.
    assert vault.worker_unavailable_reason(TaskType.SMART_SCORE) is None


def _replacement_failures_logged(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if "could not start a replacement" in record.getMessage()
    ]


def test_a_replacement_that_cannot_start_is_reported_not_raised(
    vault, monkeypatch, caplog
):
    runner = vault._task_runner
    _kill(runner, "gpu")
    attempts = []

    def cannot_start():
        attempts.append(1)
        # What replace_dead_workers raises when Thread.start fails.
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(runner, "replace_dead_workers", cannot_start)

    with caplog.at_level(logging.ERROR, logger="pixlstash.vault"):
        reason = vault.worker_unavailable_reason(TaskType.FACE_EXTRACTION)

    assert attempts == [1], "the dead lane was not handed to the runner's repair"
    assert reason is not None and "GPU worker has exited" in reason
    logged = _replacement_failures_logged(caplog)
    assert len(logged) == 1, caplog.text
    assert TaskType.FACE_EXTRACTION.value in logged[0]
    assert "RuntimeError: can't start new thread" in logged[0]
    assert "GPU lane still has no live worker" in logged[0]
    # The CPU lane is alive; its workers are available.
    assert vault.worker_unavailable_reason(TaskType.SMART_SCORE) is None


def test_a_lane_a_failed_repair_left_a_worker_on_is_still_available(
    vault, monkeypatch, caplog
):
    runner = vault._task_runner
    _kill(runner, "cpu")
    start_worker = runner._start_worker
    starts = []

    def second_start_fails(work_queue, name):
        starts.append(name)
        if len(starts) > 1:
            raise RuntimeError("can't start new thread")
        return start_worker(work_queue, name)

    # The runner's own repair path: one CPU slot comes back, the next cannot.
    monkeypatch.setattr(runner, "_start_worker", second_start_fails)

    with caplog.at_level(logging.ERROR, logger="pixlstash.vault"):
        reason = vault.worker_unavailable_reason(TaskType.SMART_SCORE)

    assert len(starts) == 2, starts
    assert runner.lane_health()["cpu"]["alive"] == 1
    assert reason is None
    logged = _replacement_failures_logged(caplog)
    assert len(logged) == 1, caplog.text
    assert TaskType.SMART_SCORE.value in logged[0]
    assert "still available" in logged[0]


def test_a_stopped_runner_is_neither_reported_dead_nor_restarted(vault):
    runner = vault._task_runner
    runner.stop()
    assert vault._dead_lane(TaskType.FACE_EXTRACTION) is None
    vault.worker_unavailable_reason(TaskType.FACE_EXTRACTION)
    assert not runner.lane_health()["running"], "a stopped runner was restarted"


def _vault_lines(caplog, level):
    return [
        record
        for record in caplog.records
        if record.name == "pixlstash.vault" and record.levelno == level
    ]


def test_a_cuda_suspend_a_failed_repair_kept_off_the_queue_is_asked_again(
    vault, monkeypatch, caplog
):
    """The runner's real repair path raises; the arbiter's pass carries on.

    Asked every pass while a training run holds the card, the failure is an
    error once in a row and debug after, until a suspend is queued again.
    """
    runner = vault._task_runner
    vault._cuda_suspend_task = None
    vault._cuda_suspend_queue_failures = 0
    # Never suspends: the queued task reaches the GPU worker and is skipped.
    vault._arbiter = SimpleNamespace(wants_suspend=lambda worker_type=None: False)
    start_worker = runner._start_worker
    starts = []

    def cannot_start(work_queue, name):
        starts.append(name)
        raise RuntimeError("can't start new thread")

    _kill(runner, "cpu")
    monkeypatch.setattr(runner, "_start_worker", cannot_start)
    with caplog.at_level(logging.DEBUG, logger="pixlstash.vault"):
        vault._queue_cuda_suspend()
        vault._queue_cuda_suspend()
        assert starts, "the dead CPU lane was not handed to the runner's repair"
        assert vault._cuda_suspend_task is None, "a suspend never queued is pending"
        errors = _vault_lines(caplog, logging.ERROR)
        assert len(errors) == 1, [r.getMessage() for r in errors]
        assert "RuntimeError: can't start new thread" in errors[0].getMessage()
        assert any(
            "(2 in a row)" in r.getMessage()
            for r in _vault_lines(caplog, logging.DEBUG)
        ), caplog.text

        # The worker comes back: the next pass queues the suspend.
        monkeypatch.setattr(runner, "_start_worker", start_worker)
        vault._queue_cuda_suspend()
        queued = vault._cuda_suspend_task
        assert queued is not None, "the suspend was not asked for again"
        assert queued.wait(5)

        # A later failure is a new run of them, logged as an error again.
        _kill(runner, "cpu")
        monkeypatch.setattr(runner, "_start_worker", cannot_start)
        vault._queue_cuda_suspend()
    assert len(_vault_lines(caplog, logging.ERROR)) == 2, caplog.text
