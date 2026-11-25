"""Work one background Scrapheap run, a short writer slice at a time.

The run (:class:`~pixlstash.services.scrapheap_run_service.ScrapheapRun`) is both
the request and the progress readout, and the service
(:mod:`pixlstash.services.scrapheap_run_service`) owns every rule. This task
only sequences the slices: :attr:`ScrapheapRunTask.SLICE_PICTURES` pictures at a
time, each one LOW-priority writer callback through the existing purge or
restore path. A purge slice costs about 0.7 ms a picture, so it holds the
writer for well under a second and every other write gets it in between.

A cancel the user asks for is a flag on the run, read between slices. The runner
cancelling the task means the server is stopping (or a full restore is
replacing the library): the run is closed ``interrupted`` at the next slice
boundary, or straight away if it never started.
"""

import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services import scrapheap_run_service
from pixlstash.services.scrapheap_run_service import (
    KIND_PURGE,
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


class ScrapheapRunTask(BaseTask):
    """Execute one :class:`~pixlstash.services.scrapheap_run_service.ScrapheapRun`.

    Attributes:
        SLICE_PICTURES: Pictures per writer slice. At the 0.63-0.85 ms a purged
            picture measured, about a third of a second of writer time; a
            restore slice costs about the same.
        NOTIFY_INTERVAL_S: How often the coalesced ``pictures_changed`` event
            goes out while a run works, so other tabs follow it without one
            event per slice.
    """

    SLICE_PICTURES = 500
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
        """Return ``(processed, total)`` pictures for Task Manager."""
        with self._progress_lock:
            return self._processed_count, self._total_count

    def on_cancel(self) -> None:
        """Close a run whose task the runner cancels before it ever ran.

        The runner cancels a queued task without running it when it stops, or
        when a full restore drains the queue. Left pending, the run would block
        every later run until the process ended. A running task only has its
        event set; it closes the run itself at the next slice boundary.
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
            status = self._work(params["picture_ids"])
            self._flush_notifications(force=True)
            final = scrapheap_run_service.finish_run(self._vault, self._run_id, status)
        except _ScrapheapRunInterrupted:
            self._flush_notifications(force=True)
            scrapheap_run_service.close_run(
                self._vault,
                self._run_id,
                STATUS_INTERRUPTED,
                "The server stopped before the run finished.",
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
        except Exception as exc:
            logger.exception(
                "[scrapheap-run] %s (%s) failed after %.1fs (%d of %d processed)",
                self._run_id,
                self._kind,
                time.monotonic() - start,
                self._processed_count,
                self._total_count,
            )
            # Announce what did commit before the failure, then close the run.
            self._flush_notifications(force=True)
            scrapheap_run_service.close_run(
                self._vault,
                self._run_id,
                STATUS_FAILED,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        logger.info(
            "[scrapheap-run] %s finished %s in %.1fs",
            self._run_id,
            final["status"],
            time.monotonic() - start,
        )
        return {"run_id": self._run_id, "status": final["status"]}

    def _work(self, picture_ids: list[int]) -> str:
        """Run every slice; return how the run ended."""
        if self._kind == KIND_PURGE:
            run_slice = scrapheap_run_service.purge_slice
        else:
            run_slice = scrapheap_run_service.restore_slice
        total = len(picture_ids)
        self._set_progress(0, total)
        for offset in range(0, total, self.SLICE_PICTURES):
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
            chunk = picture_ids[offset : offset + self.SLICE_PICTURES]
            changed = run_slice(self._vault, self._run_id, chunk)
            self._pending_picture_ids.update(changed)
            self._flush_notifications()
            self._set_progress(offset + len(chunk), total)
        return STATUS_COMPLETE

    def _flush_notifications(self, force: bool = False) -> None:
        """Announce the pictures this run changed since the last flush, coalesced.

        ``removed`` for a purge and ``restored`` for a restore, the kinds the
        synchronous routes send, so a grid treats a run's slices exactly as it
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
        self._vault.notify(
            EventType.CHANGED_PICTURES,
            {
                "picture_ids": picture_ids,
                "origin_client_id": self._origin_client_id,
                "source": self._source,
                "change_kind": "removed" if self._kind == KIND_PURGE else "restored",
            },
        )

    def _set_progress(self, processed: int, total: int) -> None:
        with self._progress_lock:
            self._total_count = max(1, int(total))
            self._processed_count = min(max(0, int(processed)), self._total_count)
