"""``TaskRunner``'s lifecycle: a submission settles or is refused untouched, a
stopped runner stays stopped, and the drain and admission state stay honest.

No ``Server``: a bare runner with fake tasks reaches everything under test.
"""

import logging
import threading

import pytest

from pixlstash.resources.arbiter import ResourceArbiter
from pixlstash.services import (
    scrapheap_run_service,
    tag_exclusion_service,
    tag_health_service,
)
from pixlstash.services.scrapheap_run_service import (
    ERROR_UNAVAILABLE,
    KIND_RESTORE,
    ScrapheapRunError,
)
from pixlstash.task_runner import TaskCancelledError, TaskRunner, TaskRunnerStopped
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskStatus
from pixlstash import vault as vault_module
from pixlstash.vault import Vault


class _NoCard:
    def devices(self, max_age_s=1.0):
        return []

    def device(self, uuid):
        return None

    def pixlstash_device(self):
        return None

    def other_processes(self, limit=3):
        return []


class _Task(BaseTask):
    """Counts its hooks; blocks on *gate* while running, if given."""

    def __init__(self, gate=None, queue_type=QueueType.CPU):
        super().__init__(task_type="LifecycleTask")
        self._gate = gate
        self._queue_type = queue_type
        self.queued = 0
        self.cancels = 0
        self.started = threading.Event()

    @property
    def queue_type(self):
        return self._queue_type

    def on_queued(self):
        super().on_queued()
        self.queued += 1

    def on_cancel(self):
        super().on_cancel()
        self.cancels += 1

    def _run_task(self):
        self.started.set()
        if self._gate is not None:
            assert self._gate.wait(timeout=30), "the test never opened the gate"
        return "done"


@pytest.fixture
def runner():
    r = TaskRunner(
        name="lifecycle-test",
        num_workers=1,
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    endings = []
    r.add_task_complete_callback(lambda task, error: endings.append((task, error)))
    r.endings = endings
    r.start()
    try:
        yield r
    finally:
        r.stop()


def _kill_cpu_lane(runner):
    """Fail the CPU worker's loop with an item that is not a queue triple."""
    worker = runner._cpu_workers[0]
    runner._queue.put((0, -1))
    worker.join(5)
    assert not worker.is_alive()
    assert runner.lane_health()["cpu"]["alive"] == 0


def _cannot_start_thread(*args, **kwargs):
    raise RuntimeError("can't start new thread")


# --- A submission whose worker replacement fails ----------------------------


def test_a_submit_whose_dead_worker_cannot_be_replaced_leaves_the_task_untouched(
    runner, monkeypatch
):
    _kill_cpu_lane(runner)
    monkeypatch.setattr(runner, "_start_worker", _cannot_start_thread)
    task = _Task()

    with pytest.raises(RuntimeError, match="can't start new thread") as raised:
        runner.submit(task)

    assert not isinstance(raised.value, TaskRunnerStopped), (
        "a thread-start failure was reported as a stopped runner"
    )
    # Refused before the task was touched: no queue hook ran (nothing to
    # release), nothing was queued, and no settlement is owed.
    assert task.queued == 0, "on_queued ran for a task that was never queued"
    assert runner._queue.qsize() == 0
    assert runner.endings == []

    # The next submission, once threads can start, replaces the worker and runs.
    monkeypatch.undo()
    retry = _Task()
    runner.submit(retry)
    assert retry._done_event.wait(10), "the replaced lane never ran the task"
    assert retry.status == TaskStatus.COMPLETED
    assert runner.lane_health()["cpu"]["alive"] == 1


def test_a_replacement_that_cannot_start_is_not_a_restart_and_logs_once(
    runner, monkeypatch, caplog
):
    _kill_cpu_lane(runner)
    monkeypatch.setattr(runner, "_start_worker", _cannot_start_thread)
    with caplog.at_level(logging.DEBUG, logger="pixlstash.task_runner"):
        for _ in range(3):
            with pytest.raises(RuntimeError, match="can't start new thread"):
                runner.replace_dead_workers()
    health = runner.lane_health()
    assert health["cpu"]["restarts"] == 0, "a replacement that never ran counted"
    assert health["cpu"]["alive"] == 0
    # The dying worker reports its own death (once, from its thread); every
    # other error here is about the replacement.
    errors = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR and "stopped unexpectedly" not in r.getMessage()
    ]
    assert len(errors) == 1, errors
    assert "could not be started (cpu lane)" in errors[0]
    repeats = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and "failed again" in r.getMessage()
    ]
    assert len(repeats) == 2, repeats

    monkeypatch.undo()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pixlstash.task_runner"):
        runner.replace_dead_workers()
    assert runner.lane_health()["cpu"]["restarts"] == 1
    assert runner.lane_health()["cpu"]["alive"] == 1
    assert "(cpu restart 1, after 3 failed attempt(s))" in caplog.text


def _replace_while_threads_fail(runner, monkeypatch, caplog, times):
    """Fail *times* replacements, then let one start; return the records."""
    caplog.clear()
    monkeypatch.setattr(runner, "_start_worker", _cannot_start_thread)
    with caplog.at_level(logging.DEBUG, logger="pixlstash.task_runner"):
        for _ in range(times):
            with pytest.raises(RuntimeError, match="can't start new thread"):
                runner.replace_dead_workers()
        monkeypatch.undo()
        runner.replace_dead_workers()
    return [r for r in caplog.records if "stopped unexpectedly" not in r.getMessage()]


def test_a_lane_whose_replacement_fails_again_after_a_restart_logs_it_again(
    runner, monkeypatch, caplog
):
    """A started replacement ends the run of failures; the next run is new.

    Without the reset a lane that recovered and then hit the thread limit again
    days later would log that second run at debug only, nothing at the default
    level, and its restart would report both runs' failures.
    """
    for restart, failures in ((1, 3), (2, 2)):
        _kill_cpu_lane(runner)
        records = _replace_while_threads_fail(runner, monkeypatch, caplog, failures)
        first = [r for r in records if "could not be started" in r.getMessage()]
        assert [r.levelno for r in first] == [logging.ERROR], (
            f"run {restart}: {[r.getMessage() for r in records]}"
        )
        assert first[0].exc_info is not None, (
            "the run's first failure lost its traceback"
        )
        repeats = [r for r in records if "failed again" in r.getMessage()]
        assert [r.levelno for r in repeats] == [logging.DEBUG] * (failures - 1)
        assert any(
            f"(cpu restart {restart}, after {failures} failed attempt(s))"
            in r.getMessage()
            for r in records
        ), [r.getMessage() for r in records]
        assert runner.lane_health()["cpu"]["alive"] == 1


# --- Only a stopped runner is reported as stopped ----------------------------


class _RefusingRunner:
    """A runner whose GPU worker is alive and whose ``submit`` raises *error*."""

    def __init__(self, error):
        self.error = error

    def gpu_worker_alive(self):
        return True

    def refuse_while_paused(self):
        """Admission is not paused: the refusal under test is the submit's."""
        return None

    def submit(self, task):
        raise self.error


@pytest.fixture
def bare_vault(tmp_path):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=True,
        daily_snapshots_enabled=False,
    )
    try:
        yield v
    finally:
        v._task_runner = None
        v.close()


@pytest.mark.parametrize("queue", ["_unload_models_on_worker", "_queue_cuda_suspend"])
def test_a_device_release_on_a_stopped_runner_is_reported_as_stopped(
    bare_vault, caplog, queue
):
    bare_vault._task_runner = _RefusingRunner(TaskRunnerStopped("stopped"))
    bare_vault._cuda_suspend_task = None
    with caplog.at_level(logging.WARNING, logger="pixlstash.vault"):
        assert getattr(bare_vault, queue)() is None
    assert "the task runner is stopped" in caplog.text


@pytest.mark.parametrize("queue", ["_unload_models_on_worker", "_queue_cuda_suspend"])
def test_a_submit_failure_that_is_not_a_stop_is_not_reported_as_one(
    bare_vault, caplog, queue
):
    """A replacement worker that cannot start is logged, not raised.

    Raised, it failed the request or the arbiter pass that asked for the
    device release; the release is simply not queued, and asked for again.
    """
    bare_vault._task_runner = _RefusingRunner(RuntimeError("can't start new thread"))
    unloads = []
    bare_vault.unload_models = lambda: unloads.append(1)
    # A vault without workers never sets the suspend slot; nothing is pending.
    bare_vault._cuda_suspend_task = None
    bare_vault._cuda_suspend_queue_failures = 0
    with caplog.at_level(logging.WARNING, logger="pixlstash.vault"):
        assert getattr(bare_vault, queue)() is None
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, caplog.text
    assert "could not start a replacement worker" in errors[0].getMessage()
    assert "RuntimeError: can't start new thread" in errors[0].getMessage()
    assert errors[0].exc_info is not None, "the failure lost its traceback"
    assert "stopped" not in caplog.text
    assert unloads == [], "the unload ran beside the live GPU worker"
    assert bare_vault._cuda_suspend_task is None, "a suspend never queued is pending"


def _stopped_runner():
    runner = TaskRunner(name="stopped-test", num_workers=1)
    runner.start()
    runner.stop()
    return runner


def test_submit_task_on_a_stopped_runner_answers_none_and_says_why(bare_vault, caplog):
    bare_vault._task_runner = _stopped_runner()
    task = _Task()
    with caplog.at_level(logging.WARNING, logger="pixlstash.vault"):
        assert bare_vault.submit_task(task) is None
    assert f"Task {task.id} (LifecycleTask) not queued" in caplog.text
    assert task.queued == 0


def test_submit_task_passes_a_failure_that_is_not_a_stop_through(bare_vault):
    bare_vault._task_runner = _RefusingRunner(RuntimeError("can't start new thread"))
    with pytest.raises(RuntimeError, match="can't start new thread"):
        bare_vault.submit_task(_Task())


def test_a_scrapheap_run_on_a_stopped_runner_is_refused_as_stopping(bare_vault):
    bare_vault._task_runner = _stopped_runner()
    with pytest.raises(ScrapheapRunError) as raised:
        scrapheap_run_service.start_run(bare_vault, KIND_RESTORE)
    assert raised.value.code == ERROR_UNAVAILABLE
    # Nothing left reserved in memory or active on disk.
    bare_vault.scrapheap_runs.check_idle()
    assert scrapheap_run_service.latest_run(bare_vault) is None


def test_a_scrapheap_run_whose_submit_fails_otherwise_is_not_called_stopping(
    bare_vault,
):
    bare_vault._task_runner = _RefusingRunner(RuntimeError("can't start new thread"))
    with pytest.raises(RuntimeError, match="can't start new thread") as raised:
        scrapheap_run_service.start_run(bare_vault, KIND_RESTORE)
    assert not isinstance(raised.value, ScrapheapRunError)
    bare_vault.scrapheap_runs.check_idle()
    assert scrapheap_run_service.latest_run(bare_vault) is None


def test_a_tag_health_rebuild_on_a_stopped_runner_is_not_run_inline(
    bare_vault, monkeypatch
):
    bare_vault._task_runner = _stopped_runner()
    inline = []
    monkeypatch.setattr(
        tag_health_service, "_run_rebuild_guarded", lambda vault: inline.append(1)
    )
    assert tag_health_service.start_rebuild(bare_vault) == {
        "building": False,
        "progress": 0.0,
    }
    assert inline == [], "a stopping vault was rebuilt in the request"
    # Not stuck "building": the next request tries again.
    assert tag_health_service.get_status(bare_vault)["building"] is False


def test_a_tag_exclusion_cleanup_on_a_stopped_runner_reads_not_started(bare_vault):
    bare_vault._task_runner = _stopped_runner()
    job = tag_exclusion_service.start_cleanup(bare_vault, ["example-tag"], 1)
    assert job["state"] == "not_started"


def test_a_tag_health_rebuild_whose_submit_raises_is_not_left_building(
    bare_vault, caplog
):
    bare_vault._task_runner = _RefusingRunner(RuntimeError("can't start new thread"))
    with caplog.at_level(logging.ERROR, logger="pixlstash.services"):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            tag_health_service.start_rebuild(bare_vault)
    assert tag_health_service.get_status(bare_vault)["building"] is False, (
        "the board would answer 'building' until a restart"
    )
    assert "tag_health rebuild for library" in caplog.text


def test_a_tag_exclusion_cleanup_whose_submit_raises_reads_not_started(
    bare_vault, monkeypatch
):
    monkeypatch.setitem(tag_exclusion_service._JOBS, bare_vault.library_uuid, [])
    bare_vault._task_runner = _RefusingRunner(RuntimeError("can't start new thread"))
    with pytest.raises(RuntimeError, match="can't start new thread"):
        tag_exclusion_service.start_cleanup(bare_vault, ["example-tag"], 1)
    [job] = tag_exclusion_service._jobs(bare_vault)
    assert job.snapshot()["state"] == "not_started"
    # Not taken for a job still running: a retry tries again rather than
    # answering 409.
    with pytest.raises(RuntimeError, match="can't start new thread"):
        tag_exclusion_service.retry_cleanup(bare_vault)


class _RefusingStartRunner:
    """A runner whose ``start()`` is refused, as beside a lingering worker."""

    def start(self):
        return False


class _PlannerSpy:
    def __init__(self):
        self.starts = 0

    def start(self):
        self.starts += 1


class _TrainingStub:
    def adopt_live_runs(self):
        return None

    def reconcile_on_startup(self):
        return None

    def start(self):
        return None


class _ArbiterStub:
    def add_yielder(self, yielder):
        return None

    def add_listener(self, listener):
        return None

    def set_suspender(self, suspender):
        return None


def test_a_refused_runner_start_leaves_the_planner_unstarted(
    bare_vault, monkeypatch, caplog
):
    bare_vault.start()  # the run recovery, with workers still disabled
    planner = _PlannerSpy()
    monkeypatch.setattr(bare_vault, "_disable_background_workers", False)
    monkeypatch.setattr(bare_vault, "_task_runner", _RefusingStartRunner())
    monkeypatch.setattr(bare_vault, "_work_planner", planner)
    monkeypatch.setattr(bare_vault, "training_service", _TrainingStub())
    # A vault built without workers has no arbiter, and the fixture's close()
    # must find it unstarted, as it was built.
    monkeypatch.setattr(bare_vault, "_arbiter", _ArbiterStub(), raising=False)
    monkeypatch.setattr(bare_vault, "_started", False)
    monkeypatch.setattr(bare_vault, "_start_reference_folder_watches", lambda: None)
    monkeypatch.setattr(bare_vault, "_model_folder_watcher", None)
    monkeypatch.setattr(
        vault_module, "reconcile_pending_people_with_faces", lambda vault: None
    )
    try:
        with caplog.at_level(logging.ERROR, logger="pixlstash.vault"):
            bare_vault.start()
    finally:
        # The fixture's close() runs after this test's patches would otherwise
        # be undone (monkeypatch is set up first), and must meet the vault as
        # it was built.
        monkeypatch.undo()
    assert planner.starts == 0, "the planner cycles against a stopped runner"
    assert "the work planner is not started either" in caplog.text


# --- start() never runs beside a worker that outlived stop() ----------------


def _alive_named(name):
    return [t for t in threading.enumerate() if t.name == name and t.is_alive()]


def test_start_is_refused_beside_a_worker_that_outlived_stop(runner, caplog):
    gate = threading.Event()
    stuck = _Task(gate=gate)
    try:
        runner.submit(stuck)
        assert stuck.started.wait(10)
        # The task ignores its cancel, so its worker outlives stop()'s join.
        runner.STOP_JOIN_TIMEOUT_S = 0.2
        runner.stop()
        assert len(_alive_named("lifecycle-test-cpu-0")) == 1

        with caplog.at_level(logging.WARNING, logger="pixlstash.task_runner"):
            assert runner.start() is False
        assert "start() refused" in caplog.text
        assert len(_alive_named("lifecycle-test-cpu-0")) == 1, (
            "a second worker was started beside the one that outlived stop()"
        )
        assert _alive_named("lifecycle-test-gpu") == []
        with pytest.raises(TaskRunnerStopped):
            runner.submit(_Task())
    finally:
        gate.set()

    assert stuck._done_event.wait(10)
    for worker in _alive_named("lifecycle-test-cpu-0"):
        worker.join(10)
    assert runner.start() is True
    after = _Task()
    runner.submit(after)
    assert after._done_event.wait(10)
    assert after.status == TaskStatus.COMPLETED


def test_replace_dead_workers_repairs_an_open_runner(runner):
    _kill_cpu_lane(runner)
    runner.replace_dead_workers()
    assert runner.lane_health()["cpu"]["alive"] == 1
    assert runner.lane_health()["cpu"]["restarts"] == 1


def test_replace_dead_workers_never_reopens_a_stopped_runner(runner, caplog):
    runner.stop()
    with caplog.at_level(logging.ERROR, logger="pixlstash.task_runner"):
        runner.replace_dead_workers()
    assert _alive_named("lifecycle-test-cpu-0") == []
    assert _alive_named("lifecycle-test-gpu") == []
    # The dead workers are still in their slots after a stop; a repair that
    # ignored the stop would start replacements that exit at once, so only
    # the restart count and the log show it.
    health = runner.lane_health()
    assert health["cpu"]["restarts"] == 0
    assert health["gpu"]["restarts"] == 0
    assert "started a replacement" not in caplog.text
    with pytest.raises(TaskRunnerStopped):
        runner.submit(_Task())


def test_start_during_stop_is_refused_and_stop_returns_promptly(
    runner, monkeypatch, caplog
):
    in_stop = threading.Event()
    let_stop_finish = threading.Event()
    cancel_requeued = runner._cancel_requeued

    def _held_cancel_requeued(reason):
        in_stop.set()
        assert let_stop_finish.wait(30), "the test never let stop() finish"
        return cancel_requeued(reason)

    monkeypatch.setattr(runner, "_cancel_requeued", _held_cancel_requeued)
    stopper = threading.Thread(target=runner.stop, name="lifecycle-stopper")
    stopper.start()
    try:
        assert in_stop.wait(10)
        # The workers see the stop flag and leave on their own (0.1 s poll),
        # so start() finds no live worker: the window it once reopened in.
        for name in ("lifecycle-test-cpu-0", "lifecycle-test-gpu"):
            for worker in _alive_named(name):
                worker.join(5)
        assert _alive_named("lifecycle-test-cpu-0") == []
        with caplog.at_level(logging.WARNING, logger="pixlstash.task_runner"):
            assert runner.start() is False
        assert "stop() is still under way" in caplog.text
    finally:
        let_stop_finish.set()
        stopper.join(10)
    assert not stopper.is_alive(), "stop() blocked joining workers it did not stop"
    assert _alive_named("lifecycle-test-cpu-0") == []
    assert _alive_named("lifecycle-test-gpu") == []
    with pytest.raises(TaskRunnerStopped):
        runner.submit(_Task())
    # Once stop() is done, start() is the explicit restart again.
    assert runner.start() is True
    after = _Task()
    runner.submit(after)
    assert after._done_event.wait(10)
    assert after.status == TaskStatus.COMPLETED


# --- The drain cancels what is queued before it returns ----------------------


class _CooperativeTask(_Task):
    """Runs until asked to stop: ``on_cancel`` opens its own gate."""

    def __init__(self):
        super().__init__(gate=threading.Event())

    def on_cancel(self):
        super().on_cancel()
        self._gate.set()


def test_the_drain_cancels_queued_work_before_it_returns(runner):
    running = _CooperativeTask()
    runner.submit(running)
    assert running.started.wait(10)
    queued = _Task()
    runner.submit(queued)  # one CPU worker, busy: this one waits on the queue
    try:
        result = runner.pause_and_drain(10, "lifecycle test")
        assert result.drained
        # Settled by the drain itself, not refused later by a worker that
        # dequeued it after drained=True was reported.
        assert queued.status == TaskStatus.CANCELLED
        assert result.cancelled == 1
        errors = [error for task, error in runner.endings if task is queued]
        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)
    finally:
        runner.resume_admission("lifecycle test")


# --- An admission pause outlives a stop and start ----------------------------


def test_an_admission_pause_survives_stop_and_start(runner, caplog):
    runner.pause_and_drain(5, "library restore")
    try:
        runner.stop()
        with caplog.at_level(logging.WARNING, logger="pixlstash.task_runner"):
            assert runner.start() is True
        assert "admission still paused (library restore)" in caplog.text
        task = _Task()
        runner.submit(task)
        assert task._done_event.wait(10)
        assert task.status == TaskStatus.CANCELLED
    finally:
        runner.resume_admission("library restore")
    after = _Task()
    runner.submit(after)
    assert after._done_event.wait(10)
    assert after.status == TaskStatus.COMPLETED


# --- A task that never went through submit() is refused by name --------------


def test_a_task_queued_without_submit_is_refused_by_name(runner, caplog):
    task = _Task()
    with caplog.at_level(logging.ERROR, logger="pixlstash.task_runner"):
        runner._queue.put((task.priority, next(runner._queue_seq), task))
        assert task._done_event.wait(10)
    assert task.status == TaskStatus.CANCELLED
    assert "without going through submit()" in caplog.text
    [(_task, error)] = runner.endings
    assert isinstance(error, TaskCancelledError)
    assert "queued without submit()" in str(error)
