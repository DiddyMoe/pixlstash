"""PixlStash's GPU work, stopped to make room for heavy work, loses nothing.

The way this loses data silently is the finders' deferral path: a finder that
sees a task end in any error other than ``TaskCancelledError`` defers its
pictures for the rest of the session (``MissingOcrFinder``,
``MissingPromptMatchFinder``). So every preempted task, running or queued,
must end as exactly ``TaskCancelledError``, whatever it raised on the way out.

A real ``TaskRunner`` with plain fake tasks: no ``Server``, nothing to load.
"""

import logging
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from pixlstash.resources.arbiter import LeaseState, ResourceArbiter
from pixlstash.resources.ledger import DeviceReading
from pixlstash.task_runner import (
    LANE_FAILURE_MESSAGE,
    TASK_HANDLING_FAILURE_MESSAGE,
    TaskCancelledError,
    TaskRunner,
    TaskRunnerStopped,
)
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority, TaskStatus
from pixlstash.tasks.missing_ocr_finder import MissingOcrFinder
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


class _OneCard(_NoCard):
    CARD = "GPU-00000000-0000-0000-0000-000000000001"

    def devices(self, max_age_s=1.0):
        return [
            DeviceReading(
                uuid=self.CARD,
                name="Test GPU",
                total_mb=24576,
                used_mb=0,
                free_mb=24576,
                measurable=True,
            )
        ]

    def device(self, uuid):
        return next((d for d in self.devices() if d.uuid == uuid), None)

    def pixlstash_device(self):
        return self.CARD


class _GpuTask(BaseTask):
    """Runs until cancelled, then raises: the worst case for the deferral path."""

    def __init__(self, name, priority=TaskPriority.MEDIUM):
        super().__init__(task_type=name, params={"picture_ids": [1, 2, 3]})
        self._priority = priority
        self.started = threading.Event()

    @property
    def priority(self):
        return self._priority

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.started.set()
        deadline = time.monotonic() + 5
        while not self._cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._cancel_event.is_set():
            raise RuntimeError("model unloaded under the batch")
        return "done"


class _QuickGpuTask(BaseTask):
    def __init__(self):
        super().__init__(task_type="TagTask")

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        return "done"


@pytest.fixture
def runner():
    runner = TaskRunner(
        name="preempt-test",
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    endings = {}
    runner.add_task_complete_callback(
        lambda task, error: endings.__setitem__(task.type, error)
    )
    runner.start()
    yield SimpleNamespace(runner=runner, endings=endings)
    runner.stop()


# How long a test waits on the runner before calling it hung. A condition that
# holds returns at once, so this bounds a hang rather than timing the runner.
# After every GPU-queue task the worker empties the device cache, runs a full
# gc.collect() and trims the heap, and in a process that has already run a
# shard's worth of tests that costs well over a second a task: a few seconds
# is not enough for a test that waits on three tasks in turn.
_HANG_GUARD_S = 30.0


def _until(condition, seconds=_HANG_GUARD_S):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_preempted_running_and_queued_tasks_end_as_cancelled(runner):
    running = _GpuTask("OcrTask")
    queued = _GpuTask("TagTask")
    runner.runner.submit(running)
    assert running.started.wait(5)
    runner.runner.submit(queued)

    stopped = runner.runner.preempt_gpu("preempted by ComfyUI 'Portrait'")

    assert sorted(stopped) == ["OcrTask", "TagTask"]
    assert _until(lambda: {"OcrTask", "TagTask"} <= set(runner.endings))
    for name in ("OcrTask", "TagTask"):
        error = runner.endings[name]
        assert type(error) is TaskCancelledError, (name, error)
    assert "preempted by ComfyUI 'Portrait'" in str(runner.endings["OcrTask"])
    assert runner.runner.wait_for_gpu_idle(5)


def test_a_preempted_batch_is_found_again_rather_than_deferred(runner):
    """What the finder makes of the ending: its pictures stay eligible."""
    finder = MissingOcrFinder(database=None, engine_getter=lambda: None)
    running = _GpuTask("OcrTask")
    runner.runner.add_task_complete_callback(
        lambda task, error: finder.on_task_complete(task, error)
    )
    runner.runner.submit(running)
    assert running.started.wait(5)
    runner.runner.preempt_gpu("preempted by Training")
    assert _until(lambda: "OcrTask" in runner.endings)
    assert finder._deferred == set(), "a preempted batch was deferred for the session"


def test_a_task_preempted_before_its_body_starts_ends_as_cancelled(runner):
    """The preemption can land after the runner registered the task and before
    ``run()`` reads its cancel: the body never starts, and the ending is the
    same ``TaskCancelledError`` naming the preemption, not a success."""

    class _PreemptedOnTheThreshold(_GpuTask):
        def run(self, on_vram_oom=None):
            # Registered by now: the preemption lands before run()'s check.
            runner.runner.preempt_gpu("preempted by Training")
            return super().run(on_vram_oom=on_vram_oom)

    task = _PreemptedOnTheThreshold("OcrTask")
    runner.runner.submit(task)

    assert _until(lambda: "OcrTask" in runner.endings)
    assert not task.started.is_set()
    assert task.status == TaskStatus.CANCELLED
    error = runner.endings["OcrTask"]
    assert type(error) is TaskCancelledError, error
    assert "preempted by Training" in str(error)


def test_interactive_work_is_never_preempted(runner):
    urgent = _GpuTask("FaceDetectionTask", priority=TaskPriority.URGENT)
    runner.runner.submit(urgent)
    assert urgent.started.wait(5)
    assert runner.runner.preempt_gpu("preempted by Training") == []
    assert not urgent._cancel_event.is_set()
    # Never stopped, but waited for: models are not unloaded under it.
    assert runner.runner.wait_for_gpu_idle(0.2) is False
    urgent.on_cancel()
    assert _until(lambda: "FaceDetectionTask" in runner.endings)


def test_a_task_whose_cancel_hook_fails_is_not_reported_stopped(runner):
    """Nothing stopped, so the snapshot must not say it was paused."""

    class _Stubborn(_GpuTask):
        def on_cancel(self):
            raise RuntimeError("cannot stop")

    task = _Stubborn("OcrTask")
    runner.runner.submit(task)
    assert task.started.wait(5)
    assert runner.runner.preempt_gpu("preempted by Training") == []
    assert task.id not in runner.runner._preempted, "its ending would read as preempted"
    task._cancel_event.set()
    assert _until(lambda: "OcrTask" in runner.endings)


def test_a_cpu_queue_task_granted_gpu_memory_counts_as_gpu_work():
    """``_admit`` leases a CPU-queue task that says it needs VRAM; it then runs
    on the card, so it is stopped to make room and models wait for it."""
    runner = SimpleNamespace(
        runner=TaskRunner(
            name="leased-test",
            resource_arbiter=ResourceArbiter(_OneCard(), usage=lambda: None),
        ),
        endings={},
    )
    runner.runner.add_task_complete_callback(
        lambda task, error: runner.endings.__setitem__(task.type, error)
    )
    runner.runner.start()

    class _CpuTaskOnTheCard(_GpuTask):
        @property
        def queue_type(self):
            return QueueType.CPU

        def estimated_vram_mb(self):
            return 512

    try:
        task = _CpuTaskOnTheCard("PluginTask")
        runner.runner.submit(task)
        assert task.started.wait(5)
        assert runner.runner.has_active_gpu_tasks()
        assert runner.runner.wait_for_gpu_idle(0.2) is False
        assert runner.runner.preempt_gpu("preempted by Training") == ["PluginTask"]
        assert _until(lambda: "PluginTask" in runner.endings)
        assert type(runner.endings["PluginTask"]) is TaskCancelledError
        assert runner.runner.wait_for_gpu_idle(1)
    finally:
        runner.runner.stop()


def test_a_task_refused_gpu_memory_waits_off_the_queue_and_runs_later():
    """Refused and unable to spill, the task is put back after a backoff; the
    worker is free meanwhile."""

    class _Refusing(ResourceArbiter):
        refusals = 1

        def try_light(self, *args, **kwargs):
            if self.refusals:
                self.refusals -= 1
                return None
            return super().try_light(*args, **kwargs)

    runner = TaskRunner(
        name="refused-test", resource_arbiter=_Refusing(_NoCard(), usage=lambda: None)
    )
    runner.VRAM_REQUEUE_BASE_S = 0.05
    done = threading.Event()
    runner.add_task_complete_callback(lambda task, error: done.set())
    runner.start()
    try:
        task = _QuickGpuTask()
        runner.submit(task)
        assert done.wait(5), "the refused task never ran"
        assert task.status.value == "completed"
    finally:
        runner.stop()


def test_only_the_batch_refused_gpu_memory_runs_on_the_cpu():
    """CPU fallback is per batch: one refusal, say during an image generation,
    must not keep the rest of a long tagging run on the CPU."""

    class _RefusingOnce(ResourceArbiter):
        refusals = 1

        def try_light(self, *args, **kwargs):
            if self.refusals:
                self.refusals -= 1
                return None
            return super().try_light(*args, **kwargs)

    class _SpillingTask(_QuickGpuTask):
        def __init__(self):
            super().__init__()
            self.on_cpu = False

        def allow_cpu_spillover(self):
            return True

        def enable_cpu_spillover(self):
            self.on_cpu = True

    runner = TaskRunner(
        name="spill-test",
        resource_arbiter=_RefusingOnce(_OneCard(), usage=lambda: None),
    )
    finished = []
    runner.add_task_complete_callback(lambda task, error: finished.append(task))
    runner.start()
    try:
        batches = [_SpillingTask() for _ in range(3)]
        for batch in batches:
            runner.submit(batch)
        assert _until(lambda: len(finished) == 3)
        assert [batch.on_cpu for batch in batches] == [True, False, False]
    finally:
        runner.stop()


def test_an_admission_that_raises_does_not_end_the_worker():
    class _RaisesOnce(ResourceArbiter):
        raised = False

        def try_light(self, *args, **kwargs):
            if not self.raised:
                self.raised = True
                raise RuntimeError("test-ledger-failure")
            return super().try_light(*args, **kwargs)

    runner = TaskRunner(
        name="admit-raises", resource_arbiter=_RaisesOnce(_NoCard(), usage=lambda: None)
    )
    runner.VRAM_REQUEUE_BASE_S = 0.05
    done = threading.Event()
    runner.add_task_complete_callback(lambda task, error: done.set())
    runner.start()
    try:
        task = _QuickGpuTask()
        runner.submit(task)
        assert done.wait(5), "the worker died on the admission error"
        assert task.status.value == "completed"
    finally:
        runner.stop()


def test_a_requeued_task_cancelled_by_stop_leaves_no_attempt_count():
    class _Refusing(ResourceArbiter):
        def try_light(self, *args, **kwargs):
            return None

    runner = TaskRunner(
        name="refused-stop", resource_arbiter=_Refusing(_NoCard(), usage=lambda: None)
    )
    runner.VRAM_REQUEUE_BASE_S = 30
    runner.start()
    task = _QuickGpuTask()
    try:
        runner.submit(task)
        assert _until(lambda: task.id in runner._requeued)
        assert task.id in runner._requeue_counts
    finally:
        runner.stop()
    assert task.id not in runner._requeue_counts


def test_a_queued_task_cancelled_after_a_requeue_leaves_no_attempt_count():
    """Back on the queue, then cancelled there (preempted, drained)."""
    runner = TaskRunner(
        name="requeued-cancel", resource_arbiter=ResourceArbiter(_NoCard())
    )
    task = _QuickGpuTask()
    runner._requeue_counts[task.id] = 3
    runner._cancel_queued_task(task, "preempted by test")
    assert task.id not in runner._requeue_counts


@pytest.mark.parametrize(
    ("meanwhile", "stopped"),
    [
        (LeaseState.WAITING, ["preempted by Portrait", "unload"]),
        # Granted or withdrawn while the hold was taken: nothing needs the room.
        (LeaseState.GRANTED, []),
        (LeaseState.RELEASED, []),
    ],
)
def test_pixlstash_yields_only_for_a_lease_that_still_waits(meanwhile, stopped):
    done = []
    lease = SimpleNamespace(id="lease-1", label="Portrait", state=LeaseState.WAITING)

    class _Planner:
        def hold(self, owner, seconds):
            lease.state = meanwhile

        def release(self, owner):
            done.append(f"release {owner}")

    vault = _vault_on(
        SimpleNamespace(
            preempt_gpu=lambda reason: done.append(reason) or [],
            gpu_worker_alive=lambda: False,
        ),
        lambda: done.append("unload"),
        _work_planner=_Planner(),
    )
    vault._release_yield_hold = lambda lease_id: Vault._release_yield_hold(
        vault, lease_id
    )
    Vault._yield_gpu(vault, lease)
    assert [step for step in done if not step.startswith("release")] == stopped
    # The hold ends with the yield it was for, or stays for the lease's grant.
    assert (vault._yield_holds == set()) is (meanwhile is not LeaseState.WAITING)


def test_a_grant_landing_before_the_hold_does_not_leave_the_planner_held():
    """The listener ends the hold between ``_yield_holds.add`` and ``hold``."""
    lease = SimpleNamespace(id="lease-1", label="Portrait", state=LeaseState.WAITING)
    holds = set()

    class _Planner:
        def hold(self, owner, seconds):
            lease.state = LeaseState.GRANTED
            Vault._on_resource_lease(vault, {"change": "granted", "lease_id": owner})
            holds.add(owner)

        def release(self, owner):
            holds.discard(owner)

    vault = SimpleNamespace(
        _task_runner=SimpleNamespace(preempt_gpu=lambda reason: []),
        _closed=False,
        _work_planner=_Planner(),
        _yield_holds=set(),
        notify=lambda *args: None,
    )
    vault._release_yield_hold = lambda lease_id: Vault._release_yield_hold(
        vault, lease_id
    )
    assert Vault._yield_gpu(vault, lease) == []
    assert holds == set(), "the planner stays held for the rest of the session"


class _HeldGpuTask(BaseTask):
    """A batch that ignores cancellation and ends only when released."""

    def __init__(self):
        super().__init__(task_type="CaptionTask")
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.started.set()
        self.release.wait(5)
        return "done"


def _vault_on(task_runner, unload_models, **fields):
    """The slice of a ``Vault`` that unloading and making room read."""
    vault = SimpleNamespace(
        **{
            "_task_runner": task_runner,
            "_closed": False,
            "_work_planner": None,
            "_yield_holds": set(),
            "_keep_models_in_memory": False,
            "_engine": object(),
            "_last_aggressive_unload_at": 0.0,
            "AGGRESSIVE_UNLOAD_INTERVAL": Vault.AGGRESSIVE_UNLOAD_INTERVAL,
            "_aggressive_unload_lock": threading.Lock(),
            "unload_models": unload_models,
            **fields,
        }
    )
    vault._unload_models_on_worker = lambda when=None: Vault._unload_models_on_worker(
        vault, when
    )
    return vault


def _record_thread(into):
    return lambda: into.append(threading.current_thread().name)


# ── device-cache flushes run on the GPU worker (#1775) ─────────────────────
# Unloading ends in ``torch.mps.empty_cache()`` on Metal, which kills the
# process when another thread runs a model at that moment. So every unload is
# a task on the GPU worker, behind the batch in flight, never beside it.


def test_making_room_unloads_on_the_gpu_worker_once_the_batch_ends(runner, monkeypatch):
    """The arbiter waits briefly; the unload lands after the batch, not under it."""
    monkeypatch.setattr("pixlstash.vault.YIELD_WAIT_S", 0.2)
    unloaded = []
    batch = _HeldGpuTask()
    runner.runner.submit(batch)
    assert batch.started.wait(5)
    lease = SimpleNamespace(id="lease-1", label="Portrait", state=LeaseState.WAITING)
    vault = _vault_on(runner.runner, _record_thread(unloaded))

    began = time.monotonic()
    Vault._yield_gpu(vault, lease)
    assert time.monotonic() - began < 2, "the arbiter's thread waited out the batch"
    assert unloaded == [], "the models were unloaded under the running batch"

    batch.release.set()
    assert _until(lambda: unloaded), "the models were never unloaded"
    assert unloaded == ["preempt-test-gpu"]


def test_a_lease_granted_during_the_wait_unloads_nothing(runner, monkeypatch):
    """The room is there; unloading now would only cost the next batch a reload."""
    monkeypatch.setattr("pixlstash.vault.YIELD_WAIT_S", 0.1)
    unloaded = []
    batch = _HeldGpuTask()
    runner.runner.submit(batch)
    assert batch.started.wait(5)
    lease = SimpleNamespace(id="lease-1", label="Portrait", state=LeaseState.WAITING)
    vault = _vault_on(runner.runner, _record_thread(unloaded))

    Vault._yield_gpu(vault, lease)
    lease.state = LeaseState.GRANTED
    batch.release.set()
    assert _until(lambda: "UnloadModels" in runner.endings)
    assert unloaded == []


def test_the_idle_sweep_unloads_on_the_gpu_worker_once(runner):
    """Reached from a request thread and the event loop: it queues, never flushes.

    Two progress polls at once used to sweep twice; the second one here enters
    while the first is still reading the snapshot.
    """
    unloaded = []
    vault = _vault_on(runner.runner, _record_thread(unloaded))
    reading = threading.Event()

    class _SlowSnapshot(dict):
        def values(self):
            reading.set()
            time.sleep(0.3)
            return super().values()

    first = threading.Thread(
        target=Vault._maybe_aggressive_unload, args=(vault, _SlowSnapshot())
    )
    first.start()
    assert reading.wait(5)
    Vault._maybe_aggressive_unload(vault, {})
    first.join(5)

    assert _until(lambda: unloaded)
    time.sleep(0.2)
    assert unloaded == ["preempt-test-gpu"]


def test_an_unload_queued_before_the_setting_is_switched_back_on_is_skipped(runner):
    """The sweep queues and returns; the owner's later "keep" wins on the worker."""
    unloaded = []
    vault = _vault_on(runner.runner, _record_thread(unloaded))
    vault._task_runner = SimpleNamespace(
        has_active_gpu_tasks=lambda: False,
        gpu_worker_alive=runner.runner.gpu_worker_alive,
        submit=runner.runner.submit,
    )
    batch = _HeldGpuTask()
    runner.runner.submit(batch)
    assert batch.started.wait(5)

    Vault._maybe_aggressive_unload(vault, {})
    vault._keep_models_in_memory = True
    batch.release.set()
    assert _until(lambda: "UnloadModels" in runner.endings)
    assert unloaded == []


def _cannot_start_a_worker(*args):
    # What TaskRunner.submit raises when a dead worker's replacement cannot start.
    raise RuntimeError("can't start new thread")


def test_making_room_keeps_what_it_paused_when_the_unload_cannot_be_queued():
    """The arbiter records the paused GPU work from what the yield returns."""
    unloaded = []
    lease = SimpleNamespace(id="lease-1", label="Portrait", state=LeaseState.WAITING)
    vault = _vault_on(
        SimpleNamespace(
            preempt_gpu=lambda reason: ["CaptionTask"],
            gpu_worker_alive=lambda: True,
            submit=_cannot_start_a_worker,
        ),
        lambda: unloaded.append("unload"),
    )
    assert Vault._yield_gpu(vault, lease) == ["CaptionTask"]
    assert unloaded == [], "unloaded beside the live GPU worker"


def test_the_idle_sweep_does_not_fail_its_request_when_the_unload_cannot_be_queued():
    """Raised on, it failed the progress poll or the settings save that swept."""
    unloaded = []
    vault = _vault_on(
        SimpleNamespace(
            has_active_gpu_tasks=lambda: False,
            gpu_worker_alive=lambda: True,
            submit=_cannot_start_a_worker,
        ),
        lambda: unloaded.append("unload"),
    )
    Vault._maybe_aggressive_unload(vault, {})
    assert unloaded == [], "unloaded beside the live GPU worker"


def test_stop_leaves_the_models_to_a_worker_that_outlived_the_wait(monkeypatch):
    """``TaskRunner.stop()`` gives up after 60 s; a flush beside it would crash."""
    released = []
    monkeypatch.setattr(
        "pixlstash.vault.FaceExtractionTask.release_detection_models",
        lambda: released.append("faces"),
    )
    for alive in (True, False):
        vault = SimpleNamespace(
            _changed_tags_notify_lock=threading.Lock(),
            _changed_tags_flush_timer=None,
            _changed_tags_pending_ids=set(),
            _cancel_changed_tags_flush=lambda: None,
            _started=False,
            _task_runner=SimpleNamespace(gpu_worker_alive=lambda alive=alive: alive),
            _disable_background_workers=False,
            _engine=SimpleNamespace(close=lambda: released.append("engine")),
            db=None,
            # Stopped ahead of the workers; none is running here.
            snapshot_service=SimpleNamespace(cancel_inflight=lambda: None),
            smart_score_anchor_watch=SimpleNamespace(close=lambda: None),
            _wait_for_face_followups=lambda: None,
        )
        Vault.stop(vault)
        assert vault._engine is None
        if alive:
            assert released == [], "flushed beside a worker that is still running"
    assert released == ["faces", "engine"], "a stopped runner releases as before"


def test_a_task_requeued_as_the_runner_stops_is_cancelled_not_stranded():
    """stop() can drain the queue between _requeue's check and its put."""
    runner = TaskRunner(
        name="requeue-stop",
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    endings = []
    runner.add_task_complete_callback(lambda task, error: endings.append(error))

    class _StopsOnPut(queue.PriorityQueue):
        def put(self, item, *args, **kwargs):
            runner._stop.set()
            super().put(item, *args, **kwargs)

    work_queue = _StopsOnPut()
    task = _QuickGpuTask()
    runner._requeued[task.id] = (threading.Timer(60, lambda: None), task)
    runner._requeue(task.id, work_queue)
    assert work_queue.empty()
    assert task.status.value == "cancelled"
    assert len(endings) == 1


def test_a_task_nobody_waits_for_any_more_leaves_the_gpu_backoff():
    class _Refusing(ResourceArbiter):
        def try_light(self, *args, **kwargs):
            return None

    runner = TaskRunner(
        name="refused-timeout",
        resource_arbiter=_Refusing(_NoCard(), usage=lambda: None),
    )
    runner.VRAM_REQUEUE_BASE_S = 30
    runner.start()
    task = _QuickGpuTask()
    try:
        with pytest.raises(TimeoutError):
            runner.submit_and_wait(task, timeout_s=0.5)
        assert task.id not in runner._requeued
        assert task.status.value == "cancelled"
    finally:
        runner.stop()


def test_a_requeue_count_past_the_float_limit_gets_the_capped_delay():
    """A task waiting out a long GPU run must not overflow the backoff."""

    class _Refusing(ResourceArbiter):
        def try_light(self, *args, **kwargs):
            return None

    runner = TaskRunner(
        name="requeue-cap", resource_arbiter=_Refusing(_NoCard(), usage=lambda: None)
    )
    task = _as_queued(runner, _QuickGpuTask())
    runner._requeue_counts[task.id] = 1100
    try:
        run_now, lease = runner._admit(task, queue.PriorityQueue())
        assert (run_now, lease) == (False, None)
        timer, _ = runner._requeued[task.id]
        assert timer.interval == runner.VRAM_REQUEUE_MAX_S
    finally:
        runner._take_requeued(task.id)


def test_an_error_while_handling_a_task_does_not_end_the_worker():
    """The GPU worker survives a task whose admission raises, and runs the next."""

    class _RefuseFirst(ResourceArbiter):
        refused = False

        def try_light(self, *args, **kwargs):
            if not self.refused:
                self.refused = True
                return None
            return super().try_light(*args, **kwargs)

    class _BrokenHook(_QuickGpuTask):
        def allow_cpu_spillover(self):
            raise RuntimeError("test-broken-hook")

    runner = TaskRunner(
        name="hook-raises", resource_arbiter=_RefuseFirst(_NoCard(), usage=lambda: None)
    )
    endings = {}
    runner.add_task_complete_callback(
        lambda task, error: endings.__setitem__(task.id, error)
    )
    runner.start()
    try:
        broken = _BrokenHook()
        survivor = _QuickGpuTask()
        runner.submit(broken)
        runner.submit(survivor)
        assert _until(lambda: survivor.id in endings), "the worker died on the error"
        assert runner.gpu_worker_alive()
        assert broken.status.value == "failed"
        assert isinstance(endings[broken.id], RuntimeError)
        assert survivor.status.value == "completed"
    finally:
        runner.stop()


def test_a_lease_that_will_not_release_still_settles_the_task():
    """The completion callbacks release the finder's claims, so they must fire."""

    def _boom():
        raise RuntimeError("test-lease-release-failure")

    class _Granting(ResourceArbiter):
        def try_light(self, *args, **kwargs):
            return SimpleNamespace(on_cpu=False, release=_boom)

    runner = TaskRunner(
        name="release-raises", resource_arbiter=_Granting(_NoCard(), usage=lambda: None)
    )
    done = threading.Event()
    runner.add_task_complete_callback(lambda task, error: done.set())
    runner.start()
    try:
        task = _QuickGpuTask()
        runner.submit(task)
        assert done.wait(5), "the callbacks never fired after the release failed"
        assert task.status.value == "completed"
        assert runner.gpu_worker_alive()
    finally:
        runner.stop()


# ── every accepted task settles exactly once ────────────────────────────────
# A task's completion callbacks are what give back the finder's claimed
# pictures and the planner's in-flight slot. Never firing strands them for the
# session; firing twice frees them twice. Every path below ends in exactly one.


class _NeverRoom(ResourceArbiter):
    """Never has GPU memory to give."""

    def try_light(self, *args, **kwargs):
        return None


class _QuickCpuTask(BaseTask):
    def __init__(self):
        super().__init__(task_type="ImportTask")

    def _run_task(self):
        return "done"


class _SpillableGpuTask(_QuickGpuTask):
    def __init__(self):
        super().__init__()
        self.on_cpu = False

    def allow_cpu_spillover(self):
        return True

    def enable_cpu_spillover(self):
        self.on_cpu = True


@pytest.fixture
def fake_timers(monkeypatch):
    """``threading.Timer`` that records each backoff and starts no thread.

    ``fire()`` runs what the timer would have run; ``start_error`` makes
    ``start()`` raise, as it does when no thread can be started.
    """
    timers = SimpleNamespace(made=[], start_error=None)

    class _FakeTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or ()
            self.cancelled = False
            timers.made.append(self)

        def start(self):
            if timers.start_error is not None:
                raise timers.start_error

        def cancel(self):
            self.cancelled = True

        def fire(self):
            self.function(*self.args)

    monkeypatch.setattr(threading, "Timer", _FakeTimer)
    return timers


def _as_queued(runner, task):
    """*task*, stamped as ``submit`` stamps what it queues, for a test that
    drives ``_admit`` or ``_handle`` directly."""
    with runner._active_task_lock:
        runner._stamp_locked(task)
    return task


def _settling_runner(name, arbiter=None):
    """A runner and the endings its callbacks saw, ``task id -> [error, ...]``."""
    runner = TaskRunner(
        name=name,
        resource_arbiter=arbiter or ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    endings: dict[str, list] = {}
    runner.add_task_complete_callback(
        lambda task, error: endings.setdefault(task.id, []).append(error)
    )
    return runner, endings


def test_a_task_refused_over_a_thousand_times_keeps_the_capped_backoff(fake_timers):
    """The real dequeue, admit and requeue cycle, run past where ``2**attempts``
    stops fitting a float: every backoff is still the capped number."""
    runner, _endings = _settling_runner(
        "requeue-cycles", _NeverRoom(_NoCard(), usage=lambda: None)
    )
    work_queue = queue.PriorityQueue()
    task = _as_queued(runner, _QuickGpuTask())
    for _ in range(1100):
        runner._handle(task, work_queue)
        fake_timers.made[-1].fire()
        _priority, _seq, dequeued = work_queue.get_nowait()
        assert dequeued is task
    intervals = [timer.interval for timer in fake_timers.made]
    assert len(intervals) == 1100
    assert intervals[:5] == [1.0, 2.0, 4.0, 8.0, 15.0]
    assert set(intervals[4:]) == {runner.VRAM_REQUEUE_MAX_S}
    assert runner._requeue_counts[task.id] == 1100


def test_a_backoff_that_cannot_start_leaves_no_entry_behind(fake_timers):
    fake_timers.start_error = RuntimeError("test-cannot-start-thread")
    runner, _endings = _settling_runner(
        "timer-fails", _NeverRoom(_NoCard(), usage=lambda: None)
    )
    task = _as_queued(runner, _QuickGpuTask())
    with pytest.raises(RuntimeError, match="test-cannot-start-thread"):
        runner._admit(task, queue.PriorityQueue())
    assert task.id not in runner._requeued


def test_a_task_whose_backoff_cannot_start_settles_once(fake_timers):
    """Failed by its worker, and not found again by the stop that follows."""
    fake_timers.start_error = RuntimeError("test-cannot-start-thread")
    runner, endings = _settling_runner(
        "timer-fails", _NeverRoom(_NoCard(), usage=lambda: None)
    )
    runner.start()
    task = _QuickGpuTask()
    try:
        runner.submit(task)
        assert _until(lambda: task.id in endings), "the task never settled"
        assert task.status == TaskStatus.FAILED
        assert runner.gpu_worker_alive()
    finally:
        runner.stop()
    assert len(endings[task.id]) == 1
    assert isinstance(endings[task.id][0], RuntimeError)


def test_a_task_refused_after_stop_swept_the_backoffs_is_cancelled(fake_timers):
    """A refusal that lands once ``stop()`` has swept the backoffs is cancelled,
    not parked behind a timer the stop will never cancel."""
    runner, endings = _settling_runner(
        "refused-after-stop", _NeverRoom(_NoCard(), usage=lambda: None)
    )
    runner.stop()  # never started: this is the moment after the sweep
    task = _QuickGpuTask()

    assert runner._admit(task, queue.PriorityQueue()) == (False, None)

    assert task.id not in runner._requeued
    assert task.status == TaskStatus.CANCELLED
    assert len(endings[task.id]) == 1
    assert isinstance(endings[task.id][0], TaskCancelledError)
    assert task.id not in runner._requeue_counts


def test_failing_a_task_takes_it_out_of_its_backoff(fake_timers):
    runner, endings = _settling_runner(
        "fail-backoff", _NeverRoom(_NoCard(), usage=lambda: None)
    )
    task = _as_queued(runner, _QuickGpuTask())
    assert runner._admit(task, queue.PriorityQueue()) == (False, None)
    timer = fake_timers.made[-1]

    runner._fail_unhandled(task, RuntimeError("test-handling-failure"))

    assert task.id not in runner._requeued
    assert timer.cancelled
    assert runner.cancel_pending_tasks() == 0
    assert task.status == TaskStatus.FAILED
    assert len(endings[task.id]) == 1


def test_a_task_the_runner_could_not_handle_keeps_the_error_in_the_log(caplog):
    """The error a client reads (an import's status, a folder scan's) is the
    fixed message; the exception, which can name files, goes to the log only."""
    runner, endings = _settling_runner("fail-message")
    task = _QuickGpuTask()
    secret = "test-handling-failure in /home/me/library/private.png"

    with caplog.at_level(logging.ERROR, logger="pixlstash.task_runner"):
        runner._fail_unhandled(task, RuntimeError(secret))

    assert task.status == TaskStatus.FAILED
    assert task.error == TASK_HANDLING_FAILURE_MESSAGE
    assert secret not in task.error
    logged = [r for r in caplog.records if secret in r.getMessage()]
    assert len(logged) == 1 and logged[0].exc_info is not None
    assert len(endings[task.id]) == 1


@pytest.mark.parametrize("submit", ["submit", "submit_and_wait"])
def test_a_stopped_runner_refuses_a_task_with_its_own_error(submit):
    """A caller can tell a stopped runner from any other ``RuntimeError``.

    It is still a ``RuntimeError``, so a caller catching that stays correct.
    """
    runner, endings = _settling_runner("stopped-refuses")
    runner.stop()
    task = _QuickGpuTask()

    with pytest.raises(TaskRunnerStopped, match="stopped-refuses"):
        getattr(runner, submit)(task)

    assert issubclass(TaskRunnerStopped, RuntimeError)
    # Refused, not queued: nothing settles it and nothing ran it.
    assert task.id not in endings
    assert task.status == TaskStatus.PENDING


def test_a_task_that_already_settled_is_not_cancelled_again():
    runner, endings = _settling_runner("settled-twice")
    task = _QuickGpuTask()
    task.run()
    runner._requeued[task.id] = (threading.Timer(60, lambda: None), task)

    runner.stop()

    assert task.status == TaskStatus.COMPLETED
    assert task.id not in endings
    assert not task._cancel_event.is_set()


@pytest.mark.parametrize("cancel", ["cancel_pending_tasks", "preempt_gpu"])
def test_a_cancel_counts_only_the_tasks_it_cancelled(cancel):
    """Settled tasks found on the queue or in a backoff are skipped, and not
    counted among the work the cancel stopped."""
    runner, endings = _settling_runner("count-cancelled")
    settled_queued, settled_waiting, live = (
        _QuickGpuTask(),
        _QuickGpuTask(),
        _QuickGpuTask(),
    )
    settled_queued.run()
    settled_waiting.run()
    runner._gpu_queue.put((live.priority, 0, live))
    runner._gpu_queue.put((settled_queued.priority, 1, settled_queued))
    runner._requeued[settled_waiting.id] = (
        threading.Timer(60, lambda: None),
        settled_waiting,
    )

    if cancel == "preempt_gpu":
        assert runner.preempt_gpu("preempted by test") == [live.type]
    else:
        assert runner.cancel_pending_tasks() == 1

    assert list(endings) == [live.id]


def test_a_task_object_queued_again_settles_again():
    """``on_queued`` makes a re-queued task object a fresh run, so the stop that
    drains it still tells its owner."""
    runner, endings = _settling_runner("queued-again")
    task = _QuickGpuTask()
    task.run()
    runner.submit(task)  # never started: it stays queued

    runner.stop()

    assert task.status == TaskStatus.CANCELLED
    assert len(endings[task.id]) == 1
    assert isinstance(endings[task.id][0], TaskCancelledError)


def test_the_owner_hears_of_a_task_when_the_clean_up_after_it_fails(monkeypatch):
    def _boom():
        raise RuntimeError("test-clean-up-failure")

    monkeypatch.setattr("pixlstash.task_runner.platform", SimpleNamespace(system=_boom))
    runner, endings = _settling_runner("clean-up-fails")
    runner.start()
    try:
        first, second = _QuickGpuTask(), _QuickGpuTask()
        runner.submit(first)
        runner.submit(second)
        assert _until(lambda: second.id in endings), "the callbacks never fired"
        assert endings[first.id] == [None]
        assert first.status == TaskStatus.COMPLETED
        assert runner.gpu_worker_alive()
    finally:
        runner.stop()


def test_a_task_the_cpu_cannot_record_stays_set_for_the_gpu(fake_timers):
    class _CpuLedgerFails(_NeverRoom):
        def note_on_cpu(self, *args, **kwargs):
            raise RuntimeError("test-cpu-ledger-failure")

    runner, _endings = _settling_runner(
        "spill-fails", _CpuLedgerFails(_NoCard(), usage=lambda: None)
    )
    task = _as_queued(runner, _SpillableGpuTask())
    assert runner._admit(task, queue.PriorityQueue()) == (False, None)
    assert task.on_cpu is False, "switched to the CPU, then sent to wait for the GPU"
    assert task.id in runner._requeued


def test_a_recovery_notice_that_fails_leaves_the_task_completed():
    class _OutOfMemoryOnce(_QuickGpuTask):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _run_task(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return "done"

    def _report(task, attempt, final, recovered=False):
        if recovered:
            raise RuntimeError("test-notifier-failure")

    runner, endings = _settling_runner("recovery-notice")
    runner.VRAM_OOM_RETRY_PAUSE_S = 0.01
    runner._report_vram_oom = _report
    runner.start()
    try:
        task = _OutOfMemoryOnce()
        runner.submit(task)
        assert _until(lambda: task.id in endings), "the task never settled"
        assert task.vram_oom_attempts == 1
        assert task.status == TaskStatus.COMPLETED
        assert endings[task.id] == [None]
    finally:
        runner.stop()


class _ExitingTask(_QuickGpuTask):
    """Raises ``SystemExit`` while it works, or while it is being admitted."""

    def __init__(self, where):
        super().__init__()
        self._where = where

    def estimated_vram_mb(self):
        if self._where == "admission":
            raise SystemExit("test-exit")
        return 0

    def _run_task(self):
        if self._where == "work":
            raise SystemExit("test-exit")
        return "done"


@pytest.mark.parametrize("where", ["work", "admission"])
def test_a_task_raising_system_exit_fails_and_its_worker_goes_on(where):
    runner, endings = _settling_runner("system-exit")
    runner.start()
    try:
        exiting = _ExitingTask(where)
        runner.submit(exiting)
        assert _until(lambda: exiting.id in endings), "the task never settled"
        assert exiting.status == TaskStatus.FAILED
        assert len(endings[exiting.id]) == 1
        assert isinstance(endings[exiting.id][0], SystemExit)
        health = runner.lane_health()
        assert health["last_failure"] is None
        assert (health["gpu"]["alive"], health["gpu"]["restarts"]) == (1, 0)

        after = _QuickGpuTask()
        runner.submit(after)
        assert _until(lambda: after.id in endings), "the worker did not go on"
        assert after.status == TaskStatus.COMPLETED
    finally:
        runner.stop()


# ── a worker that dies is seen, reported and replaced ──────────────────────


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_a_worker_that_dies_is_reported_and_replaced(lane):
    runner, endings = _settling_runner("lane-death")
    runner.start()

    def _worker():
        return runner._gpu_worker if lane == "gpu" else runner._cpu_workers[0]

    def _kill():
        # Not a (priority, seq, task) triple: the worker loop itself fails on
        # it, outside any task.
        worker = _worker()
        (runner._gpu_queue if lane == "gpu" else runner._queue).put((0, -1))
        worker.join(5)
        assert not worker.is_alive()
        return worker

    try:
        assert runner.lane_health()["degraded"] is False
        dead = _kill()
        health = runner.lane_health()
        assert (health["running"], health["degraded"]) == (True, True)
        assert health[lane]["alive"] == 0
        assert health["last_failure"]["worker"] == dead.name
        # The cause goes to the log; the health served to the SPA carries
        # the fixed message, never the exception's own text.
        assert health["last_failure"]["error"] == LANE_FAILURE_MESSAGE

        task = _QuickGpuTask() if lane == "gpu" else _QuickCpuTask()
        runner.submit(task)
        assert _until(lambda: task.id in endings), "the lane never ran again"
        assert task.status == TaskStatus.COMPLETED
        health = runner.lane_health()
        assert health["degraded"] is False
        assert (health[lane]["alive"], health[lane]["restarts"]) == (1, 1)
        alive_named = [t for t in threading.enumerate() if t.name == dead.name]
        assert len(alive_named) == 1

        # start() replaces one too.
        _kill()
        runner.start()
        health = runner.lane_health()
        assert health["degraded"] is False
        assert health[lane]["restarts"] == 2
    finally:
        runner.stop()

    health = runner.lane_health()
    assert (health["running"], health["degraded"]) == (False, False)
    with pytest.raises(RuntimeError):
        runner.submit(_QuickGpuTask())
    assert not runner.is_running(), "a stopped runner brought a worker back"
    assert runner.lane_health()[lane]["restarts"] == 2


def test_a_worker_busy_with_a_long_task_is_not_replaced():
    runner, endings = _settling_runner("busy-lane")
    runner.start()
    long_task = _HeldGpuTask()
    try:
        runner.submit(long_task)
        assert long_task.started.wait(5)
        queued = _QuickGpuTask()
        runner.submit(queued)

        gpu_workers = [t for t in threading.enumerate() if t.name == "busy-lane-gpu"]
        assert len(gpu_workers) == 1, "a second GPU worker beside a busy one"
        health = runner.lane_health()
        assert health["degraded"] is False
        assert health["gpu"]["restarts"] == 0

        long_task.release.set()
        assert _until(lambda: queued.id in endings)
    finally:
        long_task.release.set()
        runner.stop()


# ── a backoff that ends while the queue is cancelled is cancelled with it ──


class _CancelsDuringPut(queue.PriorityQueue):
    """Runs a canceller on another thread from inside the first ``put``.

    Lets the canceller run between taking a task off its backoff and putting
    it on the queue, if anything lets it in there.
    """

    def __init__(self, cancel):
        super().__init__()
        self._cancel = cancel
        self.canceller = None

    def put(self, item, *args, **kwargs):
        if self.canceller is None:
            self.canceller = threading.Thread(target=self._cancel)
            self.canceller.start()
            # Long enough for a canceller that is not held off to finish.
            self.canceller.join(0.5)
        super().put(item, *args, **kwargs)


@pytest.mark.parametrize("cancel", ["cancel_pending_tasks", "preempt_gpu"])
def test_a_backoff_ending_during_a_cancel_is_cancelled_too(cancel):
    runner, endings = _settling_runner("backoff-race")

    def canceller():
        if cancel == "preempt_gpu":
            runner.preempt_gpu("preempted by test")
        else:
            runner.cancel_pending_tasks()

    work_queue = _CancelsDuringPut(canceller)
    runner._gpu_queue = work_queue
    task = _QuickGpuTask()
    runner._requeued[task.id] = (threading.Timer(60, lambda: None), task)

    runner._requeue(task.id, work_queue)
    work_queue.canceller.join(5)

    assert task.status == TaskStatus.CANCELLED, "left on a queue already cancelled"
    assert len(endings[task.id]) == 1
    assert work_queue.empty()


class _BackoffEndsOnceDrained(queue.PriorityQueue):
    """Ends a backoff on another thread the first time a drain finds it empty.

    Puts the backoff's end just after a canceller has looked at the queue.
    """

    def __init__(self):
        super().__init__()
        self.end_backoff = None
        self.ended = None

    def get_nowait(self):
        try:
            return super().get_nowait()
        except queue.Empty:
            if self.ended is None and self.end_backoff is not None:
                self.ended = threading.Thread(target=self.end_backoff)
                self.ended.start()
                self.ended.join(5)
            raise


@pytest.mark.parametrize("cancel", ["cancel_pending_tasks", "preempt_gpu"])
def test_a_backoff_ending_just_after_a_cancel_drains_is_cancelled_too(cancel):
    runner, endings = _settling_runner("backoff-after-drain")
    work_queue = _BackoffEndsOnceDrained()
    runner._gpu_queue = work_queue
    task = _QuickGpuTask()
    runner._requeued[task.id] = (threading.Timer(60, lambda: None), task)
    work_queue.end_backoff = lambda: runner._requeue(task.id, work_queue)

    if cancel == "preempt_gpu":
        runner.preempt_gpu("preempted by test")
    else:
        runner.cancel_pending_tasks()

    assert work_queue.ended is not None, "the drain never looked at the queue"
    assert task.status == TaskStatus.CANCELLED, "landed on a queue already drained"
    assert len(endings[task.id]) == 1
    assert work_queue.empty()


# ── cancelling a task held in admission (FU-2) ─────────────────────────────
# A worker takes its task off the queue outside any lock, then asks the
# arbiter for memory. A cancel landing in that window finds the task neither
# queued nor running; the cancel generation stamped at submission catches it
# when it tries to register.


class _HeldAdmission(ResourceArbiter):
    """Holds the worker inside ``try_light`` until released, then grants or refuses."""

    def __init__(self, grant):
        super().__init__(_NoCard(), usage=lambda: None)
        self.grant = grant
        self.entered = threading.Event()
        self.release = threading.Event()

    def try_light(self, *args, **kwargs):
        self.entered.set()
        self.release.wait(5)
        if not self.grant:
            return None
        return super().try_light(*args, **kwargs)


class _RecordingGpuTask(_QuickGpuTask):
    def __init__(self):
        super().__init__()
        self.ran = threading.Event()

    def _run_task(self):
        self.ran.set()
        return "done"


@pytest.mark.parametrize(
    ("cancel", "grant"),
    [
        ("cancel_pending_tasks", True),
        ("cancel_pending_tasks", False),
        ("preempt_gpu", True),
        ("preempt_gpu", False),
        # A refused task's backoff already re-checks the stop.
        ("stop", True),
    ],
)
def test_a_cancel_landing_while_a_task_is_held_in_admission_cancels_it(cancel, grant):
    arbiter = _HeldAdmission(grant)
    runner, endings = _settling_runner(f"held-admission-{cancel}", arbiter)
    runner.start()
    task = _RecordingGpuTask()
    stopper = None
    try:
        runner.submit(task)
        assert arbiter.entered.wait(5)
        if cancel == "stop":
            # stop() joins the worker, which is held: from another thread.
            stopper = threading.Thread(target=runner.stop)
            stopper.start()
            assert _until(lambda: runner._stop.is_set())
        elif cancel == "preempt_gpu":
            runner.preempt_gpu("preempted by test")
        else:
            runner.cancel_pending_tasks()
        arbiter.release.set()
        # A refused task's backoff is 1 s; it would run once that ends.
        assert _until(lambda: task.id in endings, 3), "never settled"
        time.sleep(1.5)
    finally:
        arbiter.release.set()
        if stopper is not None:
            stopper.join(10)
        runner.stop()
    assert not task.ran.is_set(), "ran after it was cancelled"
    assert task.status == TaskStatus.CANCELLED
    assert len(endings[task.id]) == 1
    assert type(endings[task.id][0]) is TaskCancelledError
    assert runner._requeued == {}


def test_a_task_submitted_after_a_cancel_still_runs():
    """The generation is the one in force at submission, not a sticky flag."""
    runner, endings = _settling_runner("after-cancel")
    runner.start()
    try:
        runner.cancel_pending_tasks()
        runner.preempt_gpu("preempted by test")
        task = _RecordingGpuTask()
        runner.submit(task)
        assert _until(lambda: task.id in endings)
    finally:
        runner.stop()
    assert task.status == TaskStatus.COMPLETED
    assert endings[task.id] == [None]


# ── preempt_gpu skips a task that has already finished (FU-4) ──────────────


class _FinishedButHeld(_QuickGpuTask):
    """Ends as if after a GPU out-of-memory retry, so ``_handle`` announces the
    recovery while the task is still registered active: done, not yet gone."""

    def _run_task(self):
        self.vram_oom_attempts = 1
        return "done"


def test_preempt_does_not_stop_a_task_that_has_already_finished():
    announcing = threading.Event()
    release = threading.Event()

    def notifier(event_type, payload):
        if payload.get("recovered"):
            announcing.set()
            release.wait(5)

    runner = TaskRunner(
        name="preempt-finished",
        notifier=notifier,
        resource_arbiter=ResourceArbiter(_NoCard(), usage=lambda: None),
    )
    endings = {}
    runner.add_task_complete_callback(
        lambda task, error: endings.setdefault(task.id, []).append(error)
    )
    runner.start()
    task = _FinishedButHeld()
    try:
        runner.submit(task)
        assert announcing.wait(5)
        assert task._done_event.is_set()
        assert task in runner.get_active_tasks_of_type(task.type)

        assert runner.preempt_gpu("preempted by test") == []
        assert runner._preempted == {}
    finally:
        release.set()
        assert _until(lambda: task.id in endings)
        runner.stop()
    assert task.status == TaskStatus.COMPLETED
    assert endings[task.id] == [None]


# ── draining the runner before the library is swapped ──────────────────────


class _BatchWriter(BaseTask):
    """Writes one item at a time and, when asked to stop, flushes what it has.

    The flush is the write a real batch does on its way out (the tags or
    embeddings it already computed): it must land before the drain returns,
    never after.
    """

    def __init__(self, writes):
        super().__init__(task_type="TagTask")
        self.writes = writes
        self.started = threading.Event()

    @property
    def queue_type(self):
        return QueueType.GPU

    def _run_task(self):
        self.started.set()
        for item in range(500):
            if self._cancel_event.is_set():
                break
            self.writes.append(("item", item))
            time.sleep(0.01)
        time.sleep(0.2)
        self.writes.append(("flush", None))
        return "done"


def test_a_drained_runner_writes_nothing_after_the_drain_succeeds():
    runner, endings = _settling_runner("drain")
    writes: list = []
    heard = []

    def slow_owner(task, error):
        # An owner that takes its time, as a finder writing its outcome does.
        time.sleep(0.3)
        heard.append(task.id)

    runner.add_task_complete_callback(slow_owner)
    runner.start()
    batch = _BatchWriter(writes)
    queued = _RecordingGpuTask()
    try:
        runner.submit(batch)
        assert batch.started.wait(5)
        assert _until(lambda: len(writes) >= 3)
        runner.submit(queued)

        result = runner.pause_and_drain(timeout_s=5, reason="library restore")
        written_at_return = len(writes)

        assert result.drained is True
        assert result.still_running == ()
        assert writes[-1] == ("flush", None), "returned before the batch flushed"
        assert batch.id in heard, "returned before the owner heard"
        time.sleep(0.5)
        assert len(writes) == written_at_return
        assert not queued.ran.is_set()
        assert type(endings[queued.id][0]) is TaskCancelledError

        # Paused: a submission now is cancelled rather than run.
        during = _RecordingGpuTask()
        runner.submit(during)
        assert during.status == TaskStatus.CANCELLED
        assert type(endings[during.id][0]) is TaskCancelledError
        assert "library restore" in str(endings[during.id][0])

        runner.resume_admission("library restore")
        after = _RecordingGpuTask()
        runner.submit(after)
        assert _until(lambda: after.id in endings)
        assert after.status == TaskStatus.COMPLETED
    finally:
        runner.stop()
    assert not during.ran.is_set()


def test_a_drain_that_times_out_names_the_task_still_running():
    runner, endings = _settling_runner("drain-timeout")
    runner.start()
    stuck = _HeldGpuTask()
    try:
        runner.submit(stuck)
        assert stuck.started.wait(5)

        started = time.monotonic()
        result = runner.pause_and_drain(timeout_s=0.3, reason="library restore")
        took = time.monotonic() - started

        assert result.drained is False
        assert [(t.task_id, t.task_type) for t in result.still_running] == [
            (stuck.id, "CaptionTask")
        ]
        assert took < 2.0, "the timeout was not honoured"
        # Still paused after a failed drain: the caller decides.
        refused = _RecordingGpuTask()
        runner.submit(refused)
        assert refused.status == TaskStatus.CANCELLED

        stuck.release.set()
        assert _until(lambda: stuck.id in endings)
        runner.resume_admission("library restore")
        later = _RecordingGpuTask()
        runner.submit(later)
        assert _until(lambda: later.id in endings)
        assert later.status == TaskStatus.COMPLETED
    finally:
        stuck.release.set()
        runner.stop()
