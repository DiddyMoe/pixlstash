"""``TaskRunner.submit_and_wait``: a request that gives up leaves nothing behind.
Nor does a ``submit`` that a stop overtakes.

No ``Server`` here: a bare runner with fake tasks reaches everything under test.
"""

import threading

import pytest

from pixlstash.task_runner import TaskCancelledError, TaskRunner
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskStatus


class _GpuTask(BaseTask):
    """Blocks on *gate* if given, and records whether it ran."""

    def __init__(self, gate=None):
        super().__init__(task_type="GpuTask")
        self._gate = gate
        self.ran = threading.Event()

    @property
    def queue_type(self) -> QueueType:
        return QueueType.GPU

    def _run_task(self):
        self.ran.set()
        if self._gate is not None:
            assert self._gate.wait(timeout=30), "the test never opened the gate"
        return "done"


def test_a_timed_out_task_is_skipped_when_the_worker_reaches_it():
    runner = TaskRunner(name="test-runner")
    runner.start()
    gate = threading.Event()
    blocker = _GpuTask(gate)
    try:
        runner.submit(blocker)
        assert blocker.ran.wait(timeout=30), "the blocker never started"
        late = _GpuTask()
        with pytest.raises(TimeoutError):
            runner.submit_and_wait(late, timeout_s=0.05)

        gate.set()
        assert late._done_event.wait(timeout=30), "the worker never reached it"
        assert not late.ran.is_set(), "a task nobody waits for still ran"
        assert late.status == TaskStatus.CANCELLED
    finally:
        gate.set()
        runner.stop()


class _SlowToQueue(_GpuTask):
    """Its queue hook (a preload, say) is still running when the runner stops."""

    def __init__(self):
        super().__init__()
        self.queuing = threading.Event()
        self.may_finish_queuing = threading.Event()

    def on_queued(self):
        super().on_queued()
        self.queuing.set()
        assert self.may_finish_queuing.wait(timeout=30), "the test never let it on"


def test_a_stop_while_a_task_is_being_queued_cancels_it():
    runner = TaskRunner(name="test-runner")
    endings = []
    runner.add_task_complete_callback(lambda task, error: endings.append(error))
    runner.start()
    task = _SlowToQueue()
    submitted = []
    submitter = threading.Thread(target=lambda: submitted.append(runner.submit(task)))
    try:
        submitter.start()
        assert task.queuing.wait(timeout=30), "submit() never reached the hook"
        runner.stop()
        task.may_finish_queuing.set()
        submitter.join(timeout=30)

        assert submitted == [task.id]
        assert task._done_event.wait(timeout=5), "queued after the stop, never settled"
        assert task.status == TaskStatus.CANCELLED
        assert len(endings) == 1
        assert isinstance(endings[0], TaskCancelledError)
        assert not task.ran.is_set()
        health = runner.lane_health()
        assert (health["cpu"]["restarts"], health["gpu"]["restarts"]) == (0, 0), (
            "a stopped runner replaced its workers"
        )
    finally:
        task.may_finish_queuing.set()
        submitter.join(timeout=30)
        runner.stop()
