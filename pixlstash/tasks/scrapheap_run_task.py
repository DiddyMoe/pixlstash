"""Work one background Scrapheap run, a short writer slice at a time.

The run (:class:`~pixlstash.services.scrapheap_run_service.ScrapheapRun`) is both
the request and the progress readout, and the service
(:mod:`pixlstash.services.scrapheap_run_service`) owns every rule. This task
only sequences the slices, each one LOW-priority writer callback through the
existing purge or restore path, so every other write gets the writer in
between.

**A slice is sized by time, not by count.** A picture's writer cost varies
several-fold with the library (a 50,000-picture purge measured 1.2 ms a picture
against 0.7 ms on a small heap, and stacks and hub ghosts add more), so a fixed
count made a slice's writer hold, and the Stop overshoot, whatever the library
made them. :class:`_SliceSizer` meters each slice's writer time
(:func:`pixlstash.database.writer_meter`) and sizes the next to
:data:`~pixlstash.services.scrapheap_run_service.SLICE_BUDGET_S`, between
:data:`~pixlstash.services.scrapheap_run_service.SLICE_MIN_PICTURES` and
:data:`~pixlstash.services.scrapheap_run_service.SLICE_MAX_PICTURES`. The run
shows both bounds, so a client can say what Stop will still do.

An undo run's slice is one History segment of its target, newest first, so its
size is the segment's (:data:`~pixlstash.services.scrapheap_run_service.SEGMENT_PICTURE_CAP`
at most) rather than the sizer's.

A cancel the user asks for is a flag on the run, read between slices. The runner
cancelling the task means the server is stopping (or a full restore is
replacing the library): the run is closed ``interrupted`` at the next slice
boundary, or straight away if it never started.

**Every event a run sends names it.** ``run_id`` and ``run_kind`` ride on each
``pictures_changed`` message, so a tab can tell a run's slices from a single
gesture's change. An undo slice does not go through
:func:`operation_log_service.announce_restore`, which a Ctrl+Z uses, for that
reason: it sends ``source: "ui"`` and no run. What it would add is a stats
cache clear for a non-picture History entry, and a restore segment holds only
picture entries.
"""

import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from pixlstash.database import writer_meter
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services import scrapheap_run_service
from pixlstash.services.scrapheap_run_service import (
    KIND_PURGE,
    KIND_RESTORE,
    KIND_UNDO,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
)
from pixlstash.tasks.base_task import BaseTask, TaskPriority, TaskStatus
from pixlstash.tasks.task_type import TaskType

if TYPE_CHECKING:
    from pixlstash.vault import Vault

logger = get_logger(__name__)


class _ScrapheapRunInterrupted(RuntimeError):
    """Internal control flow: the runner cancelled the task (shutdown)."""


class _SliceSizer:
    """Choose each slice's size from the writer time the slices before it took.

    The cost per picture is a moving average (weight :attr:`COST_WEIGHT` on the
    newest slice), so one slow slice (a checkpoint, a burst of other writes
    measured into it) halves the next rather than collapsing it, and a run
    that turns cheaper grows back within a slice or two. The size is the
    budget over that cost, clamped to ``[min, max]``. With no cost measured
    yet, the first slice takes ``SLICE_FIRST_PICTURES``, small so an
    expensive library never pays a full-size first slice.

    Attributes:
        COST_WEIGHT: The newest slice's weight in the moving average.
        next_size: The size of the slice to run next.
    """

    COST_WEIGHT = 0.5

    def __init__(self, max_pictures: int, budget_s: float) -> None:
        self._max = max(1, int(max_pictures))
        self._min = min(scrapheap_run_service.SLICE_MIN_PICTURES, self._max)
        self._budget_s = float(budget_s)
        self._cost_s: Optional[float] = None
        self.next_size = min(scrapheap_run_service.SLICE_FIRST_PICTURES, self._max)

    def record(self, pictures: int, writer_s: float) -> None:
        """Fold in one slice of *pictures* that held the writer *writer_s*."""
        if pictures <= 0:
            return
        cost = writer_s / pictures
        if self._cost_s is None:
            self._cost_s = cost
        else:
            weight = self.COST_WEIGHT
            self._cost_s = weight * cost + (1 - weight) * self._cost_s
        if self._cost_s <= 0:
            self.next_size = self._max
            return
        fit = int(self._budget_s / self._cost_s)
        self.next_size = max(self._min, min(self._max, fit))


class ScrapheapRunTask(BaseTask):
    """Execute one :class:`~pixlstash.services.scrapheap_run_service.ScrapheapRun`.

    Attributes:
        NOTIFY_INTERVAL_S: How often the coalesced ``pictures_changed`` event
            goes out while a run works, so other tabs follow it without one
            event per slice.
    """

    NOTIFY_INTERVAL_S = 2.0

    def __init__(self, vault: "Vault", run_id: str) -> None:
        super().__init__(
            task_type=TaskType.SCRAPHEAP_RUN.value,
            # ``picture_ids`` is the key BaseTaskFinder releases its claims from;
            # no finder made this task, so it is deliberately empty.
            params={"picture_ids": [], "run_id": str(run_id)},
        )
        self._vault = vault
        self._run_id = str(run_id)
        self._progress_lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 1
        self._kind: Optional[str] = None
        self._pending_picture_ids: set[int] = set()
        self._last_notify = time.monotonic()
        self._origin_client_id: Optional[str] = None
        self._source = "external"

    @property
    def priority(self) -> TaskPriority:
        # Asked for by the user, who is watching it; the writer slices are LOW,
        # so this orders it ahead of background sweeps without starving writes.
        return TaskPriority.HIGH

    def task_progress(self) -> tuple[int, int]:
        """Return ``(processed, total)`` for Task Manager: pictures, or an
        undo run's History segments."""
        with self._progress_lock:
            return self._processed_count, self._total_count

    def on_cancel(self) -> None:
        """Close a run whose task the runner cancels before it ever ran.

        The runner cancels a queued task without running it when it stops, or
        when the task is submitted while a restore holds admission paused (a
        restore refuses, rather than drains, a run already queued or running).
        Left pending, the run would block every later run until the process
        ended. A running task only has its event set; it closes the run itself
        at the next slice boundary.
        """
        never_ran = self.status == TaskStatus.PENDING
        super().on_cancel()
        if never_ran:
            scrapheap_run_service.close_run(
                self._vault,
                self._run_id,
                STATUS_INTERRUPTED,
                "The run was stopped before it started.",
            )

    def _run_task(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            params = scrapheap_run_service.begin_run(self._vault, self._run_id)
            if params is None:
                logger.info(
                    "[scrapheap-run] %s is no longer pending; nothing to do",
                    self._run_id,
                )
                return {"run_id": self._run_id, "status": None}
            self._kind = params["kind"]
            self._origin_client_id = params["origin_client_id"]
            self._source = params["source"]
            error_code: Optional[str] = None
            error: Optional[str] = None
            if self._kind == KIND_UNDO:
                status, error_code, error = self._undo(params["segment_batch_ids"])
            else:
                status = self._work(
                    params["picture_ids"],
                    _SliceSizer(params["slice_max_pictures"], params["slice_budget_s"]),
                )
            self._flush_notifications(force=True)
            final = scrapheap_run_service.finish_run(
                self._vault, self._run_id, status, error_code, error
            )
        except _ScrapheapRunInterrupted:
            self._flush_notifications(force=True)
            scrapheap_run_service.close_run(
                self._vault,
                self._run_id,
                STATUS_INTERRUPTED,
                scrapheap_run_service.SERVER_STOPPED_MESSAGE,
            )
            logger.info(
                "[scrapheap-run] %s interrupted at a slice boundary after %.1fs "
                "(%d of %d processed)",
                self._run_id,
                time.monotonic() - start,
                self._processed_count,
                self._total_count,
            )
            return {"run_id": self._run_id, "status": STATUS_INTERRUPTED}
        except Exception:
            logger.exception(
                "[scrapheap-run] %s (%s) failed after %.1fs (%d of %d processed)",
                self._run_id,
                self._kind,
                time.monotonic() - start,
                self._processed_count,
                self._total_count,
            )
            # Announce what did commit before the failure, then close the run.
            # The run records a generic message: the exception, logged above
            # with its traceback, can carry paths or SQL the view must not serve.
            self._flush_notifications(force=True)
            scrapheap_run_service.close_run(
                self._vault,
                self._run_id,
                STATUS_FAILED,
                scrapheap_run_service.UNEXPECTED_ERROR_MESSAGE,
            )
            raise
        logger.info(
            "[scrapheap-run] %s finished %s in %.1fs",
            self._run_id,
            final["status"],
            time.monotonic() - start,
        )
        return {"run_id": self._run_id, "status": final["status"]}

    def _work(self, picture_ids: list[int], sizer: _SliceSizer) -> str:
        """Run every slice, each sized by *sizer*; return how the run ended."""
        if self._kind == KIND_PURGE:
            run_slice = scrapheap_run_service.purge_slice
        else:
            run_slice = scrapheap_run_service.restore_slice
        total = len(picture_ids)
        self._set_progress(0, total)
        offset = 0
        while offset < total:
            if self._cancel_event.is_set():
                raise _ScrapheapRunInterrupted
            if scrapheap_run_service.cancel_requested(self._vault, self._run_id):
                logger.info(
                    "[scrapheap-run] %s stopping on request at %d of %d",
                    self._run_id,
                    offset,
                    total,
                )
                return STATUS_CANCELLED
            chunk = picture_ids[offset : offset + sizer.next_size]
            with writer_meter() as meter:
                changed = run_slice(self._vault, self._run_id, chunk)
            sizer.record(len(chunk), meter.seconds)
            logger.debug(
                "[scrapheap-run] %s slice of %d: %.3fs writer over %d task(s); next %d",
                self._run_id,
                len(chunk),
                meter.seconds,
                meter.tasks,
                sizer.next_size,
            )
            offset += len(chunk)
            self._pending_picture_ids.update(changed)
            self._flush_notifications()
            self._set_progress(offset, total)
        return STATUS_COMPLETE

    def _undo(
        self, segment_batch_ids: list[str]
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Undo the target's segments newest first, one writer slice each.

        Returns:
            How the run ended, and for ``blocked`` its ``error_code`` and
            ``error``.
        """
        total = len(segment_batch_ids)
        self._set_progress(0, total)
        for done, batch_id in enumerate(reversed(segment_batch_ids)):
            if self._cancel_event.is_set():
                raise _ScrapheapRunInterrupted
            if scrapheap_run_service.cancel_requested(self._vault, self._run_id):
                logger.info(
                    "[scrapheap-run] %s stopping on request at segment %d of %d",
                    self._run_id,
                    done,
                    total,
                )
                return STATUS_CANCELLED, None, None
            outcome = scrapheap_run_service.undo_slice(
                self._vault, self._run_id, batch_id
            )
            if outcome["outcome"] == "blocked":
                return STATUS_BLOCKED, outcome["error_code"], outcome["error"]
            if outcome["outcome"] == "undone":
                # The removals first, then what they changed around them, so a
                # tab re-reads a stack's count after its members have left.
                self._pending_picture_ids.update(outcome["removed"])
                self._flush_notifications(force=True)
                self._notify(outcome["restored"], "restored")
                self._notify(outcome["updated"], "updated")
                self._notify(outcome["stack_siblings"], "updated", ["stack_count"])
            self._set_progress(done + 1, total)
        return STATUS_COMPLETE, None, None

    def _flush_notifications(self, force: bool = False) -> None:
        """Announce the pictures this run changed since the last flush, coalesced.

        ``removed`` for a purge and an undo (which puts pictures back in the
        Scrapheap), ``restored`` for a restore: the kinds the synchronous
        routes and Ctrl+Z send, so a grid treats a run's slices exactly as it
        treats those.
        """
        if not self._pending_picture_ids:
            return
        now = time.monotonic()
        if not force and now - self._last_notify < self.NOTIFY_INTERVAL_S:
            return
        picture_ids = sorted(self._pending_picture_ids)
        self._pending_picture_ids.clear()
        self._last_notify = now
        change_kind = "restored" if self._kind == KIND_RESTORE else "removed"
        self._notify(picture_ids, change_kind)

    def _notify(
        self,
        picture_ids: list[int],
        change_kind: str,
        fields: Optional[list[str]] = None,
    ) -> None:
        """Send one ``CHANGED_PICTURES`` event carrying the run's identity."""
        if not picture_ids:
            return
        data: dict[str, Any] = {
            "picture_ids": list(picture_ids),
            "origin_client_id": self._origin_client_id,
            "source": self._source,
            "change_kind": change_kind,
            "run_id": self._run_id,
            "run_kind": self._kind,
        }
        if fields:
            data["fields"] = list(fields)
        self._vault.notify(EventType.CHANGED_PICTURES, data)

    def _set_progress(self, processed: int, total: int) -> None:
        with self._progress_lock:
            self._total_count = max(1, int(total))
            self._processed_count = min(max(0, int(processed)), self._total_count)
