import ctypes
import gc
import itertools
import platform
import queue
import threading
import traceback
import time

from dataclasses import dataclass
from typing import Any, Callable, Optional
from datetime import datetime, UTC

from .event_types import EventType
from .middleware.library_admission import person_request_active
from .pixl_logging import get_logger
from .resources.arbiter import Lease, ResourceArbiter, arbiter
from .resources.ledger import ledger
from .tasks.base_task import BaseTask, QueueType, TaskPriority, TaskStatus
from .tasks.device_release_task import DeviceReleaseTask
from .utils.vram_utils import empty_device_cache


logger = get_logger(__name__)


class TaskCancelledError(RuntimeError):
    """Raised by ``TaskRunner.submit_and_wait`` when a task is cancelled
    before it had a chance to complete (e.g. the runner was stopped)."""


class TaskRunnerStopped(RuntimeError):
    """Raised by ``TaskRunner.submit`` (and so ``submit_and_wait``) when the
    runner is stopped: the task was refused, not queued, and never settles.

    A ``RuntimeError``, so a caller catching that still catches it.
    """


class TaskAdmissionRefused(Exception):
    """Raised by ``TaskRunner.submit`` for work a person started
    (``TaskRunner.is_person_started``) while admission is paused.

    The task was not touched: no ``on_queued``, nothing queued, nothing to
    settle. The caller undoes whatever it set up for the task and answers the
    person; the server maps an uncaught one to 409 with ``str(exc)``.

    Not a ``RuntimeError`` on purpose: a route that maps ``RuntimeError`` (a
    failed or cancelled task) to 503 must not turn a refusal the person can
    act on into "unavailable".

    Attributes:
        reason: Why admission is paused, as ``pause_and_drain`` was given it
            ("library restore").
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(
            f"Not started: a {reason} is in progress. Try again once it has finished."
        )


# What ``lane_health()`` says about a worker that stopped on an error. The
# exception itself goes to the server log only: its text can carry file paths
# and library content, and the lane health is served to the SPA.
LANE_FAILURE_MESSAGE = (
    "The worker stopped on an unexpected error; the server log has the details."
)

# What a task the runner could not handle (``_fail_unhandled``) carries as its
# ``error``. That field is served: an import's status and a model folder scan's
# read it. The exception goes to the server log, for the reason above.
TASK_HANDLING_FAILURE_MESSAGE = (
    "The task stopped on an unexpected error; the server log has the details."
)


@dataclass(frozen=True)
class RunningTask:
    """A task a result names: one still running when a drain gave up waiting
    for it, or one queued or running that a pause would have stopped."""

    task_id: str
    task_type: str


@dataclass(frozen=True)
class DrainResult:
    """What :meth:`TaskRunner.pause_and_drain` achieved.

    Attributes:
        drained: True when no task was running, or settling, at the return.
            Only then has every task stopped writing.
        still_running: The tasks still running at the deadline, when
            ``drained`` is False. They were asked to stop and may still write.
        cancelled: How many queued or backed-off tasks were cancelled.
    """

    drained: bool
    still_running: tuple[RunningTask, ...]
    cancelled: int


class TaskRunner:
    """Multi-thread in-memory task orchestrator.

    Tasks are dequeued by priority and executed concurrently across
    *num_workers* background threads.  A single shared PriorityQueue
    ensures correct ordering while all threads remain busy.
    """

    # Pause between attempts after a GPU out-of-memory failure. Long enough to
    # be worth waiting for - whatever else is holding the card has to give some
    # back - and short enough that the single GPU worker is not parked on it:
    # an interactive ``submit_and_wait`` (face detection, character likeness)
    # queues behind this and has a 60 s budget. Two pauses is the worst case.
    VRAM_OOM_RETRY_PAUSE_S = 5.0

    # A task the arbiter refuses GPU memory, and that cannot spill to the CPU,
    # goes back on its queue after this long, doubling to the cap. The worker
    # never waits on it: it takes the next task meanwhile.
    VRAM_REQUEUE_BASE_S = 1.0
    VRAM_REQUEUE_MAX_S = 15.0

    # How long stop() waits for each worker to exit. A worker still running
    # after it outlives stop(), and start() refuses to run beside it.
    STOP_JOIN_TIMEOUT_S = 60.0

    def __init__(
        self,
        name: str = "TaskRunner",
        num_workers: int = 1,
        notifier: Optional[Callable[[EventType, Any], None]] = None,
        resource_arbiter: Optional[ResourceArbiter] = None,
    ):
        self._name = name
        # Bound ``Vault.notify``, so a GPU out-of-memory retry can reach the
        # user's screen. Passed as the bound method rather than the vault for
        # the same reason the work finders take one: this is all the runner
        # needs from it.
        self._notifier = notifier
        # Every task that may touch the GPU asks it for a LIGHT lease first.
        self._arbiter = resource_arbiter or arbiter()
        self._num_workers = max(1, int(num_workers))
        # CPU queue: serviced by num_workers threads.
        self._queue: queue.PriorityQueue[tuple[int, int, BaseTask]] = (
            queue.PriorityQueue()
        )
        # GPU queue: serviced by exactly ONE dedicated thread so GPU tasks are
        # never concurrent.  Priority ordering ensures high-priority tasks
        # (e.g. face extraction) always run before lower-priority ones.
        self._gpu_queue: queue.PriorityQueue[tuple[int, int, BaseTask]] = (
            queue.PriorityQueue()
        )
        self._queue_seq = itertools.count()
        self._stop = threading.Event()
        # Every worker started and not yet seen to exit, for ``stop()`` to join.
        self._threads: list[threading.Thread] = []
        # One slot per worker, so one that dies is replaced under its own name
        # and one still alive - however long its task runs - never is. Under
        # ``_lock``, like the restart counts and the last failure.
        self._cpu_workers: list[Optional[threading.Thread]] = [None] * self._num_workers
        self._gpu_worker: Optional[threading.Thread] = None
        self._lane_restarts = {"cpu": 0, "gpu": 0}
        # Replacements that failed to start in a row, per lane. Not restarts:
        # only the first is logged at error, and the recovery says how many.
        self._lane_start_failures = {"cpu": 0, "gpu": 0}
        self._last_lane_failure: Optional[dict[str, str]] = None
        self._lock = threading.Lock()
        self._active_task_lock = threading.Lock()
        self._active_tasks: dict[int, BaseTask] = {}  # thread ident -> running task
        # Running CPU-queue tasks that were granted GPU memory. Under
        # ``_active_task_lock``.
        self._leased_on_gpu: set[str] = set()
        # Tasks refused GPU memory, waiting out their backoff off the queue:
        # task id -> (timer, task). Under ``_active_task_lock``.
        self._requeued: dict[str, tuple[threading.Timer, BaseTask]] = {}
        self._requeue_counts: dict[str, int] = {}
        # Task id -> why it was stopped to make room, for the ones preempted.
        self._preempted: dict[str, str] = {}
        # Cancel generations, under ``_active_task_lock``. ``submit`` stamps a
        # task with the ones in force (``_stamp_locked``); every cancel bumps
        # its own. A worker takes a task off its queue outside any lock and
        # then asks the arbiter for memory, so a cancel can land while the
        # task is neither queued nor registered: the stale stamp is what
        # catches it when it tries to register (``_registration_refusal_locked``).
        # ``_cancel_generation`` covers every task (``cancel_pending_tasks``,
        # ``pause_and_drain``); ``_preempt_generation`` the non-interactive
        # GPU-queue tasks ``preempt_gpu`` stops, with its reason.
        self._cancel_generation = 0
        self._preempt_generation = 0
        self._preempt_reason = ""
        # Why admission is paused (``pause_and_drain``), or None. Under
        # ``_active_task_lock``.
        self._admission_paused: Optional[str] = None
        # Thread ident -> the task its worker is handling, from the moment the
        # worker starts on it until its completion callbacks have fired: what
        # a drain waits to see empty. Under ``_active_task_lock``; ``_settled``
        # is notified each time one leaves.
        self._in_flight: dict[int, BaseTask] = {}
        self._settled = threading.Condition(self._active_task_lock)
        # Task id -> every task queued and not yet settled, wherever it is: on
        # a queue, waiting out a GPU backoff, taken off a queue by a worker or
        # running. Entered with the put in ``submit`` and left as it settles
        # (``_fire_task_complete_callbacks``), both under
        # ``_active_task_lock``, so ``pause_admission_unless`` sees all of them
        # in one look.
        self._unsettled: dict[str, BaseTask] = {}
        self._closed = False
        # True from the moment ``stop()`` closes the runner until it has
        # joined the workers it closed it on; ``start()`` refuses meanwhile.
        # Under ``_lock``.
        self._stopping = False
        self._on_task_complete_callbacks: list[
            Callable[[BaseTask, Optional[BaseException]], None]
        ] = []

    def _pause_and_report_vram_oom(
        self, task: BaseTask, attempt: int, error: BaseException
    ) -> None:
        """Give the GPU back, tell the user, and wait before the next attempt.

        Called by :meth:`BaseTask.run` between attempts. Flushing the allocator
        cache first is what makes the retry worth making: the failed attempt's
        partial allocations are still reserved by PyTorch until it is.

        Args:
            task: The task that failed.
            attempt: How many attempts have been used, 1-based.
            error: The out-of-memory error, for the log line.
        """
        try:
            empty_device_cache()
        except Exception:
            logger.warning(
                "Failed to flush CUDA cache before retrying task %s (%s): %s",
                task.id,
                task.type,
                traceback.format_exc(),
            )
        self._report_vram_oom(task, attempt, final=False)
        # ``_stop.wait`` rather than ``sleep`` so a shutdown does not have to sit
        # out the pause.
        self._stop.wait(self.VRAM_OOM_RETRY_PAUSE_S)
        if self._stop.is_set():
            # The runner is shutting down: raising abandons the remaining
            # attempts rather than starting an inference pass nobody is waiting
            # for. ``run()`` records it as the task's failure, as it would have
            # done had the last attempt failed.
            raise error

    def _report_vram_oom(
        self, task: BaseTask, attempt: int, final: bool, recovered: bool = False
    ) -> None:
        """Emit the VRAM_OOM event the SPA turns into a toast.

        Every retry sequence ends with one closing frame - ``recovered`` or
        ``gave_up`` - because the SPA coalesces them all onto one card, and a
        card whose last word is "retrying…" describes a state that is over.
        """
        if self._notifier is None:
            return
        try:
            self._notifier(
                EventType.VRAM_OOM,
                {
                    # Diagnostic only, like the envelope's ``event`` field: the
                    # SPA's sentence is about the GPU, not about a task class.
                    "task_type": task.type,
                    "attempt": attempt,
                    "max_attempts": task.VRAM_OOM_ATTEMPTS,
                    "gave_up": final,
                    "recovered": recovered,
                    # Who else is on the card, so the toast can say "LM Studio
                    # is holding 18 GB" instead of guessing. Not on the
                    # recovery frame: the contention is over.
                    "other_processes": [] if recovered else ledger().other_processes(),
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to announce the GPU out-of-memory retry for task %s (%s): %s",
                task.id,
                task.type,
                exc,
            )

    def add_task_complete_callback(
        self, callback: Callable[[BaseTask, Optional[BaseException]], None]
    ):
        self._on_task_complete_callbacks.append(callback)

    def cancel_pending_tasks(self) -> int:
        """Drain and cancel all tasks waiting in both CPU and GPU queues.

        Tasks that are already executing are not interrupted. A task a worker
        has taken off its queue but not yet started is cancelled too, when it
        tries to start.

        Returns:
            Number of tasks cancelled.
        """
        return self._cancel_pending("drained from the queue")

    def _cancel_pending(self, reason: str) -> int:
        """Cancel every task queued, backed off or about to start; see above."""
        with self._active_task_lock:
            self._cancel_generation += 1
        # The tasks waiting out a GPU backoff are taken first, then the queues
        # drained: a backoff that ends meanwhile moves its task onto a queue
        # under the same lock (see ``_requeue``), so it is caught by the drain
        # rather than landing on a queue already drained.
        cancelled = self._cancel_requeued(reason)
        for q in (self._queue, self._gpu_queue):
            while True:
                try:
                    _priority, _seq, queued_task = q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(queued_task, _StopTask):
                    continue
                if self._cancel_queued_task(queued_task, reason):
                    cancelled += 1
        logger.debug(
            "TaskRunner %s: cancelled %d pending task(s).", self._name, cancelled
        )
        return cancelled

    def pause_and_drain(self, timeout_s: float, reason: str) -> DrainResult:
        """Stop admitting tasks and wait, up to *timeout_s*, for none to run.

        For a caller about to replace what the tasks write to (a library
        restore swapping the database). In order:

        1. Admission is paused: from now on ``submit`` cancels what it is
           given, and a worker cancels what it takes off a queue, each with
           ``TaskCancelledError`` naming *reason*. Nothing starts.
        2. Every queued task, every task waiting out a GPU backoff and every
           task taken off a queue but not yet started is cancelled, as
           ``cancel_pending_tasks`` does.
        3. Every running task is asked to stop (``on_cancel``). Stopping is
           cooperative: a task that checks its cancel event between items
           stops at the next one, the rest finish their batch.
        4. It waits until no task is running or settling - a task counts until
           its completion callbacks have fired, since they may write too - or
           until the deadline.

        Admission stays paused whatever the result, until
        ``resume_admission``. Python threads cannot be interrupted, so a task
        still running at the deadline is not stopped by this: it is named in
        the result, and the caller must not assume it has stopped writing.

        Args:
            timeout_s: How long to wait for the running tasks, in seconds.
            reason: Why, for the cancellations and the log ("library restore").

        Returns:
            A :class:`DrainResult`. ``drained`` True means no task ran or
            settled at the return, and none can start until
            ``resume_admission``.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._active_task_lock:
            self._admission_paused = reason
        cancelled = self._cancel_pending(reason)
        with self._active_task_lock:
            running = [
                task
                for task in self._active_tasks.values()
                if not task._done_event.is_set()
            ]
        for task in running:
            try:
                task.on_cancel()
            except Exception as exc:
                logger.warning(
                    "Task %s (%s) cancel hook failed while draining (%s): %s",
                    task.id,
                    task.type,
                    reason,
                    exc,
                    exc_info=True,
                )
        with self._settled:
            self._settled.wait_for(
                lambda: not self._in_flight,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            still_running = tuple(
                RunningTask(task_id=task.id, task_type=task.type)
                for task in self._in_flight.values()
            )
        if still_running:
            logger.error(
                "TaskRunner %s: %d task(s) still running %.1fs after the drain "
                "for %s began: %s",
                self._name,
                len(still_running),
                timeout_s,
                reason,
                ", ".join(f"{t.task_type} {t.task_id}" for t in still_running),
            )
        else:
            logger.info(
                "TaskRunner %s: drained for %s; %d pending task(s) cancelled.",
                self._name,
                reason,
                cancelled,
            )
        return DrainResult(
            drained=not still_running,
            still_running=still_running,
            cancelled=cancelled,
        )

    def resume_admission(self, reason: str) -> None:
        """Admit tasks again after a pause for *reason*; safe to repeat.

        Only the pause it names is lifted: a pause made for another reason
        belongs to its own caller, so it is left in place and logged.

        Args:
            reason: The reason the pause was made with.
        """
        with self._active_task_lock:
            paused = self._admission_paused
            if paused == reason:
                self._admission_paused = None
        if paused is not None and paused != reason:
            logger.error(
                "TaskRunner %s: not resuming admission for %s; it is paused for %s.",
                self._name,
                reason,
                paused,
            )

    def unsettled_work(
        self, matching: Callable[[BaseTask], bool]
    ) -> tuple[RunningTask, ...]:
        """The tasks queued or running, not yet settled, that *matching* picks.

        A look, not a promise: a task can be submitted the moment after. A
        caller that must act on the answer uses ``pause_admission_unless``.

        Args:
            matching: Called under the runner's lock with each such task; it
                must not call back into the runner.

        Returns:
            The matching tasks, in submission order.
        """
        with self._active_task_lock:
            return self._unsettled_matching_locked(matching)

    def pause_admission_unless(
        self, reason: str, matching: Callable[[BaseTask], bool]
    ) -> tuple[RunningTask, ...]:
        """Pause admission, as ``pause_and_drain`` does, unless a task *matching*
        picks is queued or running.

        The look and the pause are one hold of the runner's lock: a task
        submitted after the look is cancelled by the pause, and one submitted
        before it is seen. Nothing is cancelled or asked to stop here.

        Args:
            reason: Why, as ``pause_and_drain`` takes it.
            matching: As ``unsettled_work`` takes it.

        Returns:
            The matching tasks; when there are any, admission was left as it
            was. Empty when admission is now paused, until
            ``resume_admission``.
        """
        with self._active_task_lock:
            blocking = self._unsettled_matching_locked(matching)
            if not blocking:
                self._admission_paused = reason
        return blocking

    @staticmethod
    def is_interactive(task: BaseTask) -> bool:
        """True for a task somebody is waiting on: URGENT, or a
        ``submit_and_wait``'s. ``preempt_gpu`` never stops one."""
        return task.priority == TaskPriority.URGENT or getattr(task, "_awaited", False)

    @staticmethod
    def is_person_started(task: BaseTask) -> bool:
        """True for work a person asked for, decided by how it was submitted:
        inside a person's request (``person_request_active``, which ``submit``
        records on the task), or by ``submit_and_wait``, whose caller waits
        for it.

        Not decided by the task's type: one type can be a person's (Detect
        faces again) or the planner's (the faces finder). Such work is not
        asked for again if it is cancelled, so while admission is paused it is
        refused (``TaskAdmissionRefused``) rather than cancelled, and a pause
        that must not stop it waits for it (``pause_admission_unless``).
        """
        return getattr(task, "_person_started", False) or getattr(
            task, "_awaited", False
        )

    def refuse_while_paused(self) -> None:
        """Raise ``TaskAdmissionRefused`` now if admission is paused.

        For a request that writes something before it submits the task the
        write is for (a reset that clears tags, then queues the retag): called
        first, a refusal comes before the write. A look only, not a hold: a
        pause that lands between it and ``submit`` is still refused by
        ``submit``.

        Raises:
            TaskAdmissionRefused: Admission is paused.
        """
        with self._active_task_lock:
            paused = self._admission_paused
        if paused is not None:
            raise TaskAdmissionRefused(paused)

    def get_active_tasks_of_type(self, task_type: str) -> list:
        """Return a list of currently executing task instances of the given type."""
        with self._active_task_lock:
            return [t for t in self._active_tasks.values() if t.type == task_type]

    def has_active_gpu_tasks(self) -> bool:
        """True while the GPU worker is executing a task."""
        with self._active_task_lock:
            return any(self._on_gpu(t) for t in self._active_tasks.values())

    def preempt_gpu(self, reason: str) -> list[str]:
        """Stop PixlStash's GPU work to make room, at a task boundary.

        Running GPU tasks are asked to stop (``on_cancel``): the ones that can
        stop early do, the rest finish their batch. Queued GPU tasks are
        cancelled; the CPU queue, imports included, is not touched. Interactive
        tasks - URGENT, or waited on by ``submit_and_wait`` - are never stopped.

        Everything stopped ends as ``TaskCancelledError``, the one ending the
        finders do not defer pictures over, so the pictures it did not process
        are found again once the planner runs.

        Args:
            reason: Said in the cancellation and the log ("preempted by ...").

        Returns:
            The worker types that were stopped, for the snapshot's
            ``preempted`` entries.
        """
        stopped: list[str] = []
        with self._active_task_lock:
            # Bumped first: a GPU task taken off the queue and still in
            # admission is cancelled when it tries to start.
            self._preempt_generation += 1
            self._preempt_reason = reason
            # A task whose run has ended is only waiting to be unregistered:
            # there is nothing left to stop, and marking it would turn the
            # COMPLETED it already is into a cancellation nowhere else sees.
            running = [
                task
                for task in self._active_tasks.values()
                if self._on_gpu(task)
                and not self._interactive(task)
                and not task._done_event.is_set()
            ]
            for task in running:
                self._preempted[task.id] = reason
        for task in running:
            try:
                task.on_cancel()
            except Exception as exc:
                logger.warning(
                    "Task %s (%s) cancel hook failed while making room: %s",
                    task.id,
                    task.type,
                    exc,
                )
                # Not asked to stop, so its ending is not a preemption.
                with self._active_task_lock:
                    self._preempted.pop(task.id, None)
                continue
            stopped.append(task.type)
        # Backoffs before the queue, for the reason ``cancel_pending_tasks``
        # gives.
        with self._active_task_lock:
            waiting = [
                task_id
                for task_id, (_timer, task) in self._requeued.items()
                if task.queue_type == QueueType.GPU and not self._interactive(task)
            ]
        for task_id in waiting:
            task = self._take_requeued(task_id)
            if task is not None and self._cancel_queued_task(task, reason):
                stopped.append(task.type)
        kept = []
        while True:
            try:
                item = self._gpu_queue.get_nowait()
            except queue.Empty:
                break
            queued_task = item[2]
            if isinstance(queued_task, _StopTask) or self._interactive(queued_task):
                kept.append(item)
                continue
            if self._cancel_queued_task(queued_task, reason):
                stopped.append(queued_task.type)
        for item in kept:
            self._gpu_queue.put(item)
        if stopped:
            logger.warning(
                "TaskRunner %s: stopped %d GPU task(s) (%s): %s",
                self._name,
                len(stopped),
                ", ".join(sorted(set(stopped))),
                reason,
            )
        return stopped

    def wait_for_gpu_idle(self, timeout_s: float = 60.0) -> bool:
        """Block until no GPU task runs, interactive ones included; True if so.

        Interactive tasks are never stopped, but they are short, and unloading
        a model under one makes it return nothing: what this waits for is the
        moment models can be unloaded.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._active_task_lock:
                busy = any(self._on_gpu(t) for t in self._active_tasks.values())
            if not busy:
                return True
            if self._stop.wait(0.1):
                return False
        return False

    def _on_gpu(self, task: BaseTask) -> bool:
        """*task* is running on the card; call under ``_active_task_lock``.

        Every GPU-queue task counts, even one running on the CPU: it holds a
        model, and ``unload_models`` under its batch makes it return nothing.
        """
        return task.queue_type == QueueType.GPU or task.id in self._leased_on_gpu

    @staticmethod
    def _interactive(task: BaseTask) -> bool:
        return TaskRunner.is_interactive(task)

    def _admit(
        self, task: BaseTask, work_queue: queue.PriorityQueue
    ) -> tuple[bool, Optional[Lease]]:
        """Ask the arbiter for the GPU memory *task* needs, without blocking.

        Every GPU-queue task asks, even one with no estimate, so an exclusive
        lease elsewhere can keep it off the card; a CPU-queue task asks only
        when it says it needs VRAM. Refused, the task spills to the CPU where
        it can, or goes back on its queue after a backoff while this worker
        takes the next one.

        Returns:
            ``(run_now, lease)``. ``run_now`` False means the task was put
            aside and must not run on this pass.
        """
        try:
            need_mb = max(0, int(task.estimated_vram_mb()))
        except Exception as exc:
            logger.warning(
                "Task %s (%s) VRAM estimate failed; asking for none: %s",
                task.id,
                task.type,
                exc,
            )
            need_mb = 0
        if task.queue_type != QueueType.GPU and need_mb <= 0:
            return True, None
        try:
            lease = self._arbiter.try_light(
                task.type,
                need_mb=need_mb,
                worker_type=task.type,
                urgent=self._interactive(task),
            )
        except Exception as exc:
            # Raised here, it would end the worker thread and every task
            # queued behind this one. Taken as a refusal instead: it spills
            # to the CPU or is tried again after the backoff.
            logger.error(
                "Task %s (%s) could not be checked against the GPU's memory: %s",
                task.id,
                task.type,
                exc,
                exc_info=True,
            )
            lease = None
        if lease is not None:
            self._requeue_counts.pop(task.id, None)
            return True, lease
        if task.allow_cpu_spillover():
            # Recorded before the task is switched over: a task switched and
            # then not recorded would wait out the backoff below still set for
            # the CPU, and run there on the GPU lease it is granted next time.
            cpu_lease: Optional[Lease] = None
            try:
                cpu_lease = self._arbiter.note_on_cpu(task.type, task.type)
                task.enable_cpu_spillover()
                logger.info(
                    "Task %s (%s) moved to the CPU: the GPU has no room for %s MB",
                    task.id,
                    task.type,
                    need_mb,
                )
                self._requeue_counts.pop(task.id, None)
                return True, cpu_lease
            except Exception as exc:
                logger.warning(
                    "Task %s (%s) could not move to the CPU, so it waits for "
                    "the GPU instead: %s",
                    task.id,
                    task.type,
                    exc,
                )
                if cpu_lease is not None:
                    self._give_back(task, cpu_lease)
        attempts = self._requeue_counts.get(task.id, 0)
        self._requeue_counts[task.id] = attempts + 1
        # The exponent is capped: a task that waits out a long GPU run is requeued
        # thousands of times, and ``2**attempts`` overflows the float conversion
        # past ~1024. 64 doublings is far past the cap for any real base.
        delay = min(
            self.VRAM_REQUEUE_MAX_S,
            self.VRAM_REQUEUE_BASE_S * (2 ** min(attempts, 64)),
        )
        logger.debug(
            "Task %s (%s) waits %.0fs for GPU memory (%s MB)",
            task.id,
            task.type,
            delay,
            need_mb,
        )
        timer = threading.Timer(delay, self._requeue, args=(task.id, work_queue))
        timer.daemon = True
        # Entered before the timer starts, so a timer that fires at once finds
        # its task. A timer that never started must not leave the entry behind:
        # the task is failed by the caller, and a later cancel taking the entry
        # would settle it a second time. The stop is read under the lock
        # ``stop()``'s backoff sweep takes: an entry made first is found by the
        # sweep, and a refusal that lands after it is cancelled here rather than
        # left waiting behind a timer nobody will cancel. The same goes for a
        # cancel, preemption or drain (``_registration_refusal_locked``).
        with self._active_task_lock:
            refusal = self._registration_refusal_locked(task)
            if refusal is None:
                self._requeued[task.id] = (timer, task)
        if refusal is not None:
            self._cancel_queued_task(task, refusal)
            return False, None
        try:
            timer.start()
        except BaseException:
            self._take_requeued(task.id)
            raise
        return False, None

    def _stamp_locked(self, task: BaseTask) -> None:
        """Record the cancel generations in force as *task* is queued.

        Call under ``_active_task_lock``, with the put that queues it.
        """
        task._runner_generations = (self._cancel_generation, self._preempt_generation)

    def _registration_refusal_locked(self, task: BaseTask) -> Optional[str]:
        """Why *task*, about to start or enter a backoff, must be cancelled.

        None when it may go on. Call under ``_active_task_lock``: the cancels
        bump their generation under it, so either they find the task
        registered or the task finds the bumped generation here.
        """
        if self._stop.is_set():
            return "runner stopped"
        if self._admission_paused is not None:
            return self._admission_paused
        generations = getattr(task, "_runner_generations", None)
        if generations is None:
            # Only ``submit`` stamps a task. One put on a queue any other way
            # cannot be checked against the cancels, so it is refused by name
            # rather than failing on the missing attribute.
            logger.error(
                "TaskRunner %s: task %s (%s) reached a worker without going "
                "through submit(); refusing it.",
                self._name,
                task.id,
                task.type,
            )
            return "queued without submit()"
        cancel_generation, preempt_generation = generations
        if cancel_generation != self._cancel_generation:
            return "drained from the queue"
        if (
            preempt_generation != self._preempt_generation
            and task.queue_type == QueueType.GPU
            and not self._interactive(task)
        ):
            return self._preempt_reason
        return None

    def _unsettled_matching_locked(
        self, matching: Callable[[BaseTask], bool]
    ) -> tuple[RunningTask, ...]:
        """See ``unsettled_work``. Call under ``_active_task_lock``."""
        return tuple(
            RunningTask(task_id=task.id, task_type=task.type)
            for task in self._unsettled.values()
            if not task._done_event.is_set() and matching(task)
        )

    def _take_requeued(self, task_id: str) -> Optional[BaseTask]:
        with self._active_task_lock:
            entry = self._requeued.pop(task_id, None)
        self._requeue_counts.pop(task_id, None)
        if entry is None:
            return None
        entry[0].cancel()
        return entry[1]

    def _requeue(self, task_id: str, work_queue: queue.PriorityQueue) -> None:
        # Taken off the backoff and put on the queue under one hold of the lock,
        # so a canceller that takes the backoffs and then drains the queues
        # finds the task in one place or the other, never between the two.
        with self._active_task_lock:
            entry = self._requeued.pop(task_id, None)
            if entry is None:
                return
            task = entry[1]
            stopped = self._stop.is_set()
            if not stopped:
                work_queue.put((task.priority, next(self._queue_seq), task))
        if stopped:
            self._cancel_queued_task(task, "runner stopped")
            return
        if self._stop.is_set():
            # stop() may have drained the queue between the check and the put.
            self._drain(work_queue, "runner stopped")

    def _drain(self, work_queue: queue.PriorityQueue, reason: str) -> None:
        """Cancel every task still waiting in *work_queue*."""
        while True:
            try:
                _priority, _seq, queued_task = work_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(queued_task, _StopTask):
                self._cancel_queued_task(queued_task, reason)

    def _cancel_requeued(self, reason: str) -> int:
        with self._active_task_lock:
            task_ids = list(self._requeued)
        cancelled = 0
        for task_id in task_ids:
            task = self._take_requeued(task_id)
            if task is not None and self._cancel_queued_task(task, reason):
                cancelled += 1
        return cancelled

    def start(self) -> bool:
        """Start the workers, or replace any that died since; safe to repeat.

        A stopped runner starts afresh once its old workers have exited, and
        not before: started beside a worker that outlived ``stop()``, a new
        GPU worker would run a second GPU task at once. To repair a running
        runner without ever reopening a stopped one, use
        ``replace_dead_workers``.

        An admission pause (``pause_and_drain``) is not lifted here: it belongs
        to whoever paused, and only ``resume_admission`` ends it. A restore
        fence lifted by a stop and start inside the restore would let work run
        against the database being replaced. Starting while paused is logged,
        since every submission is then cancelled.

        A ``start()`` that lands while ``stop()`` is still running is refused
        too: reopened there, ``stop()`` would return with the runner running,
        after joining the new workers rather than its own.

        Returns:
            True when the runner is running; False when it was refused because
            a ``stop()`` is still under way or a worker outlived one (logged as
            a warning), so it stays stopped and every ``submit`` raises
            ``TaskRunnerStopped``.
        """
        with self._lock:
            if self._stopping:
                logger.warning(
                    "TaskRunner %s: start() refused; stop() is still under way. "
                    "The runner stays stopped.",
                    self._name,
                )
                return False
            if self._closed or self._stop.is_set():
                self._threads = [t for t in self._threads if t.is_alive()]
                if self._threads:
                    logger.warning(
                        "TaskRunner %s: start() refused; %d worker(s) from the "
                        "last stop() still running (%s). The runner stays "
                        "stopped and refuses every submission until they exit.",
                        self._name,
                        len(self._threads),
                        ", ".join(t.name for t in self._threads),
                    )
                    return False
                self._closed = False
                self._stop.clear()
                self._cpu_workers = [None] * self._num_workers
                self._gpu_worker = None
                with self._active_task_lock:
                    paused = self._admission_paused
                if paused is not None:
                    logger.warning(
                        "TaskRunner %s: started with admission still paused "
                        "(%s); every task is cancelled until resume_admission().",
                        self._name,
                        paused,
                    )
            for slot, worker in enumerate(self._cpu_workers):
                if worker is None:
                    self._cpu_workers[slot] = self._start_worker(
                        self._queue, f"{self._name}-cpu-{slot}"
                    )
            # Single dedicated GPU worker - one task at a time, priority-ordered.
            if self._gpu_worker is None:
                self._gpu_worker = self._start_worker(
                    self._gpu_queue, f"{self._name}-gpu"
                )
            self._replace_dead_workers_locked()
        return True

    def replace_dead_workers(self) -> None:
        """Replace each worker that has exited, while the runner is open.

        The repair path for a lane whose worker died (``lane_health`` shows it
        degraded). Unlike ``start`` it never reopens a runner: once stopped or
        closed it does nothing, so a repair racing a shutdown cannot bring the
        workers back up behind it.

        Raises:
            RuntimeError: A replacement thread could not be started
                (``Thread.start``); the dead worker stays in its slot, it is
                not counted as a restart, and the next repair tries again.
        """
        with self._lock:
            if self._closed or self._stop.is_set():
                return
            self._replace_dead_workers_locked()

    def _start_worker(
        self, work_queue: queue.PriorityQueue, name: str
    ) -> threading.Thread:
        """Start one worker on *work_queue*; call under ``_lock``."""
        worker = threading.Thread(
            target=self._run, args=(work_queue,), name=name, daemon=True
        )
        worker.start()
        self._threads.append(worker)
        return worker

    def _replace_dead_workers_locked(self) -> None:
        """Start a worker in place of each one that has exited; under ``_lock``.

        Only a worker that has *exited* is replaced. One still alive is left
        alone however long its task runs: a second worker on the GPU queue
        would run two GPU tasks at once. A slot never started stays empty -
        a runner nobody started is not brought up by this.
        """
        for slot, worker in enumerate(self._cpu_workers):
            if worker is not None and not worker.is_alive():
                self._cpu_workers[slot] = self._replace_worker(
                    worker, "cpu", self._queue
                )
        if self._gpu_worker is not None and not self._gpu_worker.is_alive():
            self._gpu_worker = self._replace_worker(
                self._gpu_worker, "gpu", self._gpu_queue
            )

    def _replace_worker(
        self, dead: threading.Thread, lane: str, work_queue: queue.PriorityQueue
    ) -> threading.Thread:
        """Start *dead*'s replacement; a restart counts only once it started.

        A replacement that cannot start is retried on every ``submit`` and
        every planner repair (``LANE_REPAIR_INTERVAL_S``), so a failure that
        persists, a process at its thread limit, would log on each of them.
        Only the first failure in a row is an error, with its traceback; the
        rest are debug, and the start that finally works says how many failed
        before it.
        """
        try:
            worker = self._start_worker(work_queue, dead.name)
        except Exception as exc:
            self._lane_start_failures[lane] += 1
            failures = self._lane_start_failures[lane]
            if failures == 1:
                logger.error(
                    "TaskRunner %s: worker %s had stopped and its replacement "
                    "could not be started (%s lane): %s: %s. Retried on each "
                    "submit and repair; only its recovery is logged from here.",
                    self._name,
                    dead.name,
                    lane,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "TaskRunner %s: replacement for %s failed again (%d in a "
                    "row): %s: %s",
                    self._name,
                    dead.name,
                    failures,
                    type(exc).__name__,
                    exc,
                )
            raise
        self._threads = [t for t in self._threads if t is not dead]
        self._lane_restarts[lane] += 1
        failures = self._lane_start_failures[lane]
        self._lane_start_failures[lane] = 0
        logger.error(
            "TaskRunner %s: worker %s had stopped; started a replacement "
            "(%s restart %d%s).",
            self._name,
            dead.name,
            lane,
            self._lane_restarts[lane],
            f", after {failures} failed attempt(s)" if failures else "",
        )
        return worker

    def stop(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            # The workers this stop closed the runner on, and the only ones it
            # joins: ``start()`` refuses until it is done (``_stopping``), and a
            # closed runner starts no replacement, so none is added meanwhile.
            self._stopping = True
            threads = list(self._threads)
        try:
            # Cancel tasks still waiting in both queues so task-specific background
            # resources (such as preload threads) can be released deterministically.
            for q in (self._queue, self._gpu_queue):
                self._drain(q, "runner stopped")
            self._cancel_requeued("runner stopped")

            # Cancel tasks that are currently executing so their loops can exit early.
            with self._active_task_lock:
                active = list(self._active_tasks.values())
            for active_task in active:
                try:
                    active_task.on_cancel()
                except Exception as exc:
                    logger.warning(
                        "Task %s (%s) cancel hook failed (active): %s",
                        active_task.id,
                        active_task.type,
                        exc,
                    )

            # Unblock CPU workers + the single GPU worker with stop sentinels.
            for _ in range(self._num_workers):
                self._queue.put((TaskPriority.HIGH, next(self._queue_seq), _StopTask()))
            self._gpu_queue.put((TaskPriority.HIGH, next(self._queue_seq), _StopTask()))
            for t in threads:
                t.join(timeout=self.STOP_JOIN_TIMEOUT_S)
                if t.is_alive():
                    logger.warning(
                        "TaskRunner %s worker %s did not stop within timeout.",
                        self._name,
                        t.name,
                    )
        finally:
            with self._lock:
                self._stopping = False

    def submit(self, task: BaseTask) -> str:
        """Queue *task*; it settles exactly once, through the callbacks.

        Also replaces any worker that has died, so work queued behind a lost
        worker runs again from the next submission on.

        Submitted inside a person's request (``person_request_active``), the
        task is recorded as theirs (``is_person_started``); a
        ``DeviceReleaseTask`` never is, as it is housekeeping the runtime asks
        for again. While admission is paused (``pause_and_drain``) such a task
        is refused untouched, with ``TaskAdmissionRefused``; any other task is
        cancelled at once, with ``TaskCancelledError``, rather than queued.
        The pause check and the task's entry in the unsettled work are one
        hold of the runner's lock, so ``pause_admission_unless`` either sees
        a person's task or the task sees the pause.

        Raises:
            TaskAdmissionRefused: Admission is paused and the task is a
                person's. Raised before the task is touched.
            TaskRunnerStopped: The runner is stopped. A stop that lands while the
                task is being queued does not raise: the task is cancelled
                instead, as the stop would have cancelled it had it been
                queued a moment earlier.
            RuntimeError: A dead worker's replacement thread could not start.
                Raised before the task is touched (no ``on_queued``, nothing
                queued, nothing to settle); it is not a stop, so callers that
                mean "stopped" catch ``TaskRunnerStopped`` only.
        """
        if self._closed or self._stop.is_set():
            raise TaskRunnerStopped(f"TaskRunner {self._name} is stopped.")
        # Before the task is touched: a replacement that cannot start raises
        # here, with no queue hook run and so no settlement owed. Run after
        # ``on_queued`` it left a task that had started its preload, was never
        # queued, and never settled.
        self.replace_dead_workers()
        if person_request_active() and not isinstance(task, DeviceReleaseTask):
            task._person_started = True
        with self._active_task_lock:
            refused = self._admission_paused
            if refused is not None and not self.is_person_started(task):
                refused = None
            if refused is None:
                # Queued again, a task object is a fresh run (see
                # ``on_queued``), and its settlement is a fresh one too.
                task._done_event.clear()
                if self._admission_paused is None:
                    # Unsettled from here on, before ``on_queued`` runs: a
                    # ``pause_admission_unless`` that comes now sees it.
                    self._unsettled[task.id] = task
        if refused is not None:
            logger.info(
                "TaskRunner %s: refused task %s (%s) a person started; "
                "admission is paused for %s.",
                self._name,
                task.id,
                task.type,
                refused,
            )
            raise TaskAdmissionRefused(refused)
        try:
            task.on_queued()
        except Exception as exc:
            logger.warning(
                "Task %s (%s) queue hook failed: %s",
                task.id,
                task.type,
                exc,
            )
        target_queue = (
            self._gpu_queue if task.queue_type == QueueType.GPU else self._queue
        )
        paused: Optional[str] = None
        with self._lock:
            # Checked again under the lock ``stop()`` closes the runner under:
            # a stop during ``on_queued()`` has drained the queues already, and
            # a task put on one now would never run and never settle.
            closed = self._closed or self._stop.is_set()
            if not closed:
                # Stamped and queued under the lock the cancels bump their
                # generation under: a cancel either drains it from the queue
                # or, if a worker has taken it, finds it stale at the start.
                with self._active_task_lock:
                    paused = self._admission_paused
                    if paused is None:
                        self._stamp_locked(task)
                        self._unsettled[task.id] = task
                        target_queue.put((task.priority, next(self._queue_seq), task))
        if closed:
            self._cancel_queued_task(task, "runner stopped")
            return task.id
        if paused is not None:
            self._cancel_queued_task(task, paused)
            return task.id
        qsize = target_queue.qsize()
        logger.debug(
            "TaskRunner %s: submitted task id=%s type=%s queue=%s queue_depth=%s",
            self._name,
            task.id,
            task.type,
            task.queue_type,
            qsize,
        )
        return task.id

    def submit_and_wait(self, task: BaseTask, timeout_s: float = 60.0) -> Any:
        """Submit *task* and block until it completes, then return its result.

        This is intended for interactive, user-triggered tasks that need a
        result before the caller can continue (e.g. face detection during a
        search request).  It should be called from a background thread (e.g.
        via ``asyncio.run_in_executor``) so the event loop is not blocked.

        Args:
            task: The task to submit and wait for.
            timeout_s: Maximum seconds to wait before raising ``TimeoutError``.

        Returns:
            The value stored in ``task.result`` after successful completion.

        Raises:
            TimeoutError: The task did not complete within *timeout_s* seconds.
                It is cancelled, so if it had not started it never runs.
            TaskCancelledError: The task was cancelled before completing
                (e.g. the runner was stopped).
            TaskRunnerStopped: The runner was already stopped; the task was
                not queued.
            TaskAdmissionRefused: Admission is paused; a task somebody waits
                for is a person's, so it was refused untouched.
            RuntimeError: The task failed; the original error message is included.
        """
        # Somebody is waiting on it: it is never stopped to make room for
        # other work, and a waiting heavy lease does not hold it back.
        task._awaited = True
        self.submit(task)
        if not task._done_event.wait(timeout=timeout_s):
            # Nobody is waiting any more: a task still queued is skipped when the
            # worker reaches it, rather than running for no one, and one
            # waiting out a GPU backoff is not put back at all.
            task.on_cancel()
            requeued = self._take_requeued(task.id)
            if requeued is not None:
                self._cancel_queued_task(requeued, "nobody is waiting for it any more")
            raise TimeoutError(
                f"Task {task.id} ({task.type}) did not complete within {timeout_s}s"
            )
        if task.status == TaskStatus.CANCELLED:
            raise TaskCancelledError(
                f"Task {task.id} ({task.type}) was cancelled before completion"
            )
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task {task.id} ({task.type}) failed: {task.error}")
        return task.result

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def gpu_worker_alive(self) -> bool:
        """True while the GPU worker thread runs, even one that outlived ``stop()``."""
        return self._gpu_worker is not None and self._gpu_worker.is_alive()

    def lane_health(self) -> dict[str, Any]:
        """Whether the workers are there to run what is queued, for the owner.

        ``degraded`` is True while the runner is meant to be running and a
        worker has exited: the work queued for its lane waits until the next
        submission replaces it (``submit``). A runner that was never started,
        or was stopped, is not degraded; nothing is meant to run.

        Returns:
            ``running``, ``degraded``, ``last_failure`` (the latest worker
            that exited, ``{"worker", "error", "at"}``, or None),
            ``waiting_for_gpu_memory`` (tasks in a GPU backoff), and per lane
            (``cpu``, ``gpu``) ``alive``, ``total``, ``restarts`` and
            ``queued``.
        """
        with self._lock:
            running = self._gpu_worker is not None and not (
                self._closed or self._stop.is_set()
            )
            cpu_alive = sum(
                1 for w in self._cpu_workers if w is not None and w.is_alive()
            )
            gpu_alive = int(self.gpu_worker_alive())
            restarts = dict(self._lane_restarts)
            last_failure = (
                dict(self._last_lane_failure) if self._last_lane_failure else None
            )
        with self._active_task_lock:
            waiting = len(self._requeued)
        return {
            "running": running,
            "degraded": running and (cpu_alive < self._num_workers or not gpu_alive),
            "last_failure": last_failure,
            "waiting_for_gpu_memory": waiting,
            "cpu": {
                "alive": cpu_alive,
                "total": self._num_workers,
                "restarts": restarts["cpu"],
                "queued": self._queue.qsize(),
            },
            "gpu": {
                "alive": gpu_alive,
                "total": 1,
                "restarts": restarts["gpu"],
                "queued": self._gpu_queue.qsize(),
            },
        }

    def _run(self, work_queue: queue.PriorityQueue):
        logger.debug("TaskRunner %s worker started.", self._name)
        try:
            while not self._stop.is_set():
                try:
                    _priority, _seq, task = work_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(task, _StopTask):
                    continue

                # In flight from here until it has settled, however it
                # settles: what a drain waits to see empty.
                thread_ident = threading.current_thread().ident
                with self._active_task_lock:
                    self._in_flight[thread_ident] = task
                try:
                    try:
                        self._handle(task, work_queue)
                    except BaseException as exc:
                        self._fail_unhandled(task, exc)
                finally:
                    with self._active_task_lock:
                        self._in_flight.pop(thread_ident, None)
                        self._settled.notify_all()
        except BaseException as exc:
            # Whatever a task raises is caught above. What reaches here is a
            # fault in the loop itself - a malformed queue item, a failure
            # while failing a task - and it ends this worker, with the work
            # queued behind it waiting. Recorded for ``lane_health()``; the
            # next ``submit()`` or ``start()`` replaces the worker.
            worker = threading.current_thread().name
            with self._lock:
                self._last_lane_failure = {
                    "worker": worker,
                    "error": LANE_FAILURE_MESSAGE,
                    "at": datetime.now(UTC).isoformat(),
                }
            logger.error(
                "TaskRunner %s worker %s stopped unexpectedly; the work queued "
                "for it waits until a replacement starts (the next submission, or "
                "the planner within seconds): %r",
                self._name,
                worker,
                exc,
                exc_info=True,
            )
            return

        logger.debug("TaskRunner %s stopped.", self._name)

    def _handle(self, task: BaseTask, work_queue: queue.PriorityQueue) -> None:
        """Admit, run and settle one dequeued task. See _run."""
        if self._stop.is_set():
            self._cancel_queued_task(task, "runner stopped before the task ran")
            return

        logger.debug(
            "TaskRunner %s: dequeued task id=%s type=%s queue=%s queue_depth=%s.",
            self._name,
            task.id,
            task.type,
            task.queue_type,
            work_queue.qsize(),
        )

        run_now, lease = self._admit(task, work_queue)
        if not run_now:
            return

        task_start = time.perf_counter()
        logger.debug(
            "TaskRunner %s: starting task id=%s type=%s.",
            self._name,
            task.id,
            task.type,
        )
        error: Optional[BaseException] = None
        thread_ident = threading.current_thread().ident
        # Registered under the lock every cancel bumps its generation under:
        # a cancel that landed while the task was in admission is caught
        # here, and one that lands after finds it registered and asks it to
        # stop. The lease is given back before the cancellation is reported,
        # as it is after a run.
        with self._active_task_lock:
            refusal = self._registration_refusal_locked(task)
            if refusal is None:
                self._active_tasks[thread_ident] = task
                if lease is not None and not lease.on_cpu:
                    self._leased_on_gpu.add(task.id)
        if refusal is not None:
            if lease is not None:
                self._give_back(task, lease)
            self._cancel_queued_task(task, refusal)
            return
        try:
            task.run(on_vram_oom=self._pause_and_report_vram_oom)
            if task.status == TaskStatus.CANCELLED:
                # run() returns CANCELLED only when the cancel reached it before
                # it started: one set while the task was queued (a
                # submit_and_wait that timed out) or one that landed after it
                # was registered (a stop, a drain, a preemption). Settled as
                # every other cancellation is, so an owner reads one shape: with
                # no error it would take the task for a success that wrote
                # nothing.
                reason = self._preempted.get(task.id, "cancelled before it started")
                error = TaskCancelledError(f"Task {task.id} ({task.type}) {reason}")
            # ``vram_oom_attempts`` decides whether the user has a card open
            # about this task; ``attempts_used`` is what that card counts -
            # the attempt that actually finished the work, not the last one
            # that OOMed.
            if task.vram_oom_attempts:
                # It got there in the end - say so, rather than leaving the
                # user's last card reading "retrying". Guarded on its own: the
                # work is done, and a failed announcement must not reach the
                # handler below and settle a COMPLETED task with an error.
                try:
                    self._report_vram_oom(
                        task, task.attempts_used, final=False, recovered=True
                    )
                except Exception as exc:
                    logger.warning(
                        "Task %s (%s) completed, but its GPU out-of-memory "
                        "recovery could not be announced: %s",
                        task.id,
                        task.type,
                        exc,
                        exc_info=True,
                    )
        except BaseException as exc:
            # BaseException, as concurrent.futures settles a work item: a task
            # raising SystemExit is that task's failure, not this worker's
            # end. Let through, it would end the worker silently, with the
            # task never settled and everything queued behind it stranded.
            error = exc
            if not isinstance(exc, Exception):
                logger.error(
                    "Task %s (%s) raised %s; it is failed and the worker goes on.",
                    task.id,
                    task.type,
                    type(exc).__name__,
                    exc_info=True,
                )
            preempted_by = self._preempted.get(task.id)
            if preempted_by is not None:
                # Stopped to make room, whatever the task raised on the
                # way out. A finder defers a batch for the session on any
                # other error; this one says nothing about the pictures.
                task.status = TaskStatus.CANCELLED
                error = TaskCancelledError(
                    f"Task {task.id} ({task.type}) {preempted_by}"
                )
            # Same split: a task that OOMed twice and then died of something
            # else still has a card open, and it closes naming the attempt
            # that died - as does one abandoned at shutdown, which used one
            # attempt, not three.
            if task.vram_oom_attempts:
                self._report_vram_oom(task, task.attempts_used, final=True)
            tb = traceback.extract_tb(exc.__traceback__)
            if tb:
                last = tb[-1]
                logger.warning(
                    "Task %s (%s) failed at %s:%s in %s: %s | code=%s",
                    task.id,
                    task.type,
                    last.filename,
                    last.lineno,
                    last.name,
                    exc,
                    (last.line or "").strip(),
                )
            else:
                logger.warning("Task %s (%s) failed: %s", task.id, task.type, exc)
        finally:
            try:
                with self._active_task_lock:
                    self._active_tasks.pop(thread_ident, None)
                    self._leased_on_gpu.discard(task.id)
                    self._preempted.pop(task.id, None)
                if lease is not None:
                    self._give_back(task, lease)
                # Always flush PyTorch's CUDA allocator cache after a GPU-queue
                # task so that activation tensors (data, not models) are returned
                # promptly.  CPU-queue tasks only flush when they held a VRAM
                # reservation.
                if task.queue_type == QueueType.GPU:
                    try:
                        empty_device_cache()
                    except Exception:
                        logger.warning(
                            "Failed to flush CUDA cache after task %s (%s): %s",
                            task.id,
                            task.type,
                            traceback.format_exc(),
                        )
                    # Collect Python objects freed during inference (e.g. preloaded
                    # image dicts) and trim glibc's malloc arena so that resident
                    # set size drops back towards the true working set.
                    gc.collect()
                    if platform.system().lower().startswith("linux"):
                        try:
                            trim = getattr(
                                ctypes.CDLL("libc.so.6"), "malloc_trim", None
                            )
                            if trim is not None:
                                trim(0)
                        except Exception:
                            logger.warning(
                                "Failed to trim malloc arena after task %s (%s): %s",
                                task.id,
                                task.type,
                                traceback.format_exc(),
                            )
                elif lease is not None and not lease.on_cpu:
                    try:
                        empty_device_cache()
                    except Exception:
                        logger.warning(
                            "Failed to flush CUDA cache after task %s (%s): %s",
                            task.id,
                            task.type,
                            traceback.format_exc(),
                        )
                elapsed_s = time.perf_counter() - task_start
                logger.debug(
                    "TaskRunner %s: finished task id=%s type=%s status=%s elapsed=%.3fs.",
                    self._name,
                    task.id,
                    task.type,
                    task.status,
                    elapsed_s,
                )
            finally:
                # Whatever the clean-up above raised, the task has settled and
                # its owner must hear so: the finder's claims and the
                # planner's in-flight slot are given back only here. What was
                # raised still reaches ``_run``, which logs it.
                self._fire_task_complete_callbacks(task, error)

    @staticmethod
    def _give_back(task: BaseTask, lease: Lease) -> None:
        """Release *lease*, logging rather than raising.

        Whoever calls this still has the task to settle: the completion
        callbacks must fire after it, or the finder never releases its claims.
        """
        try:
            lease.release()
        except Exception as exc:
            logger.error(
                "Task %s (%s) could not give back its GPU lease: %s",
                task.id,
                task.type,
                exc,
                exc_info=True,
            )

    def _fail_unhandled(self, task: BaseTask, exc: BaseException) -> None:
        """Fail a task the worker could not finish, and keep the worker going.

        Reached only when something raised while the runner was handling a
        task outside its own ``run()`` - admission, or the bookkeeping around
        the run. The task is failed, as a task that raised from ``run()`` is,
        and its completion callbacks fire so the finder releases its pictures.
        It is not requeued: a fault in the runner or a task hook would recur on
        every pass and never end. A task that already settled is only logged.

        Args:
            task: The task being handled when the error was raised.
            exc: The error.
        """
        logger.error(
            "TaskRunner %s: task id=%s type=%s could not be handled; failing it "
            "and moving on: %s",
            self._name,
            task.id,
            task.type,
            exc,
            exc_info=True,
        )
        # A settled task holds no pending retry: one left in the backoff would
        # be taken by a later cancel and settled a second time. Also clears
        # its attempt count.
        self._take_requeued(task.id)
        if task._done_event.is_set():
            return
        # The exception is in the log above; ``error`` is served to clients.
        task.error = TASK_HANDLING_FAILURE_MESSAGE
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now(UTC)
        task._done_event.set()
        self._fire_task_complete_callbacks(task, exc)

    def _fire_task_complete_callbacks(
        self, task: BaseTask, error: Optional[BaseException]
    ):
        # Settled: left before the callbacks run, so one that queues the task
        # again enters it afresh. Every caller is outside the lock.
        with self._active_task_lock:
            if self._unsettled.get(task.id) is task:
                del self._unsettled[task.id]
        for callback in list(self._on_task_complete_callbacks):
            try:
                callback(task, error)
            except Exception as callback_exc:
                logger.warning(
                    "Task completion callback failed for %s (%s): %s",
                    task.id,
                    task.type,
                    callback_exc,
                )

    def _cancel_queued_task(self, task: BaseTask, reason: str) -> bool:
        """Cancel a task that will never run and release everything it holds.

        The completion callbacks are the only path that gives a task's resources
        back: ``BaseTaskFinder.on_task_complete`` discards its claimed picture
        ids and ``WorkPlanner.on_task_complete`` frees its in-flight slot. They
        used to fire only from the worker's ``finally``, so a task cancelled off
        the queue kept both for the life of the process - the finder then sat at
        max in-flight for ever, every finder that ``depends_on()`` it starved,
        and the claimed pictures could never be selected again.

        Args:
            task: The task being cancelled; it has been taken off its queue.
            reason: Why it was cancelled, for the log and the callback error.

        Returns:
            True if this call cancelled it; False if it had already settled, so
            a caller counting what it cancelled does not count it.
        """
        self._requeue_counts.pop(task.id, None)
        if task._done_event.is_set():
            # Settled already, and its owner told: a second settlement would
            # free the planner's slot and the finder's claims twice.
            logger.debug(
                "Task %s (%s) already settled as %s; not cancelling it (%s).",
                task.id,
                task.type,
                task.status,
                reason,
            )
            return False
        try:
            task.on_cancel()
        except Exception as exc:
            logger.warning(
                "Task %s (%s) cancel hook failed (%s): %s",
                task.id,
                task.type,
                reason,
                exc,
            )
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(UTC)
        task._done_event.set()
        self._fire_task_complete_callbacks(
            task,
            TaskCancelledError(f"Task {task.id} ({task.type}) cancelled: {reason}"),
        )
        return True


class _StopTask(BaseTask):
    def __init__(self):
        super().__init__(task_type="_stop")

    def _run_task(self) -> Any:
        return None
