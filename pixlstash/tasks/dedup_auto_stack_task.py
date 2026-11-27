"""Work one background auto-stack run, a short writer slice at a time.

The run row (:class:`~pixlstash.db_models.dedup.DedupAutoStackRun`) is both the
request and the progress readout; the service
(:mod:`pixlstash.services.dedup_auto_stack_service`) owns every rule. This task
only sequences the slices:

* **Apply.** Plan the scope's exact groups once (a read, off the writer), then
  stack them :attr:`DedupAutoStackTask.SLICE_GROUPS` at a time, each slice one
  LOW-priority writer callback. Stacking a group costs tens of milliseconds, so
  a slice holds the writer for about a second and every other write gets it in
  between.
* **Revert.** Undo the target run's History segments newest first, one segment
  per slice, announcing each restore as Ctrl+Z would.

A cancel the user asks for is durable (``cancel_requested`` on the row) and the
service stops at the next slice. The runner cancelling the task in memory means
the server is stopping: the run is closed ``interrupted`` straight away, and the
start-up reconcile catches the case where even that could not run.
"""

import threading
import time
from typing import Any, Optional

from pixlstash.database import DBPriority
from pixlstash.db_models.dedup import (
    AUTO_STACK_COMPLETE,
    AUTO_STACK_FAILED,
    AUTO_STACK_INTERRUPTED,
    AUTO_STACK_KIND_APPLY,
    AUTO_STACK_RUNNING,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.services import dedup_auto_stack_service, operation_log_service
from pixlstash.services.dedup_tier_service import DedupScope, ScopeType
from pixlstash.services.dedup_verdict_service import (
    notify_pictures_changed,
    plan_auto_stack_in_session,
)
from pixlstash.services.folder_structure_commit_service import (
    resolve_pending_people_with_faces,
)
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.tasks.task_type import TaskType

logger = get_logger(__name__)


class _AutoStackInterrupted(RuntimeError):
    """Internal control flow: the runner cancelled the task (shutdown)."""


class DedupAutoStackTask(BaseTask):
    """Execute one :class:`~pixlstash.db_models.dedup.DedupAutoStackRun`.

    Attributes:
        SLICE_GROUPS: Groups stacked per writer slice. At the 14-24 ms a group
            measured, about half a second to a second of writer time.
        NOTIFY_INTERVAL_S: How often the coalesced ``pictures_changed`` event
            goes out while a run works, so other tabs refresh without one
            event per group.
    """

    SLICE_GROUPS = 40
    NOTIFY_INTERVAL_S = 2.0

    def __init__(self, vault, run_id: int):
        super().__init__(
            task_type=TaskType.DEDUP_AUTO_STACK.value,
            # ``picture_ids`` is the key BaseTaskFinder releases its claims from;
            # this task claims no pictures, so it is deliberately empty.
            params={"picture_ids": [], "run_id": int(run_id)},
        )
        self._vault = vault
        self._run_id = int(run_id)
        self._progress_lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 1
        self._pending_picture_ids: set[int] = set()
        self._last_notify = time.monotonic()
        self._origin_client_id: Optional[str] = None
        self._source = "external"

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    def task_progress(self) -> tuple[int, int]:
        """Return ``(processed, total)`` for Task Manager: groups, or segments."""
        with self._progress_lock:
            return self._processed_count, self._total_count

    def _run_task(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            params = self._run_slice(
                dedup_auto_stack_service.start_run_in_session, self._run_id
            )
            if params is None:
                logger.info(
                    "[auto-stack] run %s is no longer pending; nothing to do",
                    self._run_id,
                )
                return {"run_id": self._run_id, "status": None}
            self._origin_client_id = params["origin_client_id"]
            self._source = params["source"]
            if params["kind"] == AUTO_STACK_KIND_APPLY:
                status = self._apply(params)
            else:
                status = self._revert(params)
            self._flush_notifications(force=True)
            final = self._run_slice(
                dedup_auto_stack_service.finish_run_in_session, self._run_id, status
            )
        except _AutoStackInterrupted:
            self._flush_notifications(force=True)
            self._vault.db.run_task(
                dedup_auto_stack_service.close_run_in_session,
                self._run_id,
                AUTO_STACK_INTERRUPTED,
                "The server stopped before the run finished.",
                priority=DBPriority.IMMEDIATE,
            )
            logger.info(
                "[auto-stack] run %s interrupted at a slice boundary after %.1fs",
                self._run_id,
                time.monotonic() - start,
            )
            return {"run_id": self._run_id, "status": AUTO_STACK_INTERRUPTED}
        except Exception as exc:
            logger.exception(
                "[auto-stack] run %s failed after %.1fs (%d of %d processed)",
                self._run_id,
                time.monotonic() - start,
                self._processed_count,
                self._total_count,
            )
            # Announce what did commit before the failure, then close the row.
            self._flush_notifications(force=True)
            self._vault.db.run_task(
                dedup_auto_stack_service.close_run_in_session,
                self._run_id,
                AUTO_STACK_FAILED,
                f"{type(exc).__name__}: {exc}",
                priority=DBPriority.IMMEDIATE,
            )
            raise
        logger.info(
            "[auto-stack] run %s finished %s in %.1fs",
            self._run_id,
            final["status"],
            time.monotonic() - start,
        )
        return {"run_id": self._run_id, "status": final["status"]}

    def _apply(self, params: dict[str, Any]) -> str:
        scope = DedupScope(
            scope_type=ScopeType(params["scope_type"]), scope_id=params["scope_id"]
        )
        # A plain read: it runs on its own connection and never takes the writer.
        group_ids = self._vault.db.run_immediate_read_task(
            plan_auto_stack_in_session, scope, params["limit"]
        )
        self._run_slice(
            dedup_auto_stack_service.record_plan_in_session,
            self._run_id,
            len(group_ids),
        )
        self._set_progress(0, len(group_ids))
        for start in range(0, len(group_ids), self.SLICE_GROUPS):
            chunk = group_ids[start : start + self.SLICE_GROUPS]
            result = self._run_slice(
                dedup_auto_stack_service.apply_slice_in_session, self._run_id, chunk
            )
            resolve_pending_people_with_faces(
                self._vault, result["event_picture_ids"], origin="Auto-stack"
            )
            self._pending_picture_ids.update(result["event_picture_ids"])
            self._flush_notifications()
            if result["status"] != AUTO_STACK_RUNNING:
                return result["status"]
            self._set_progress(result["processed"], result["total"])
        return AUTO_STACK_COMPLETE

    def _revert(self, params: dict[str, Any]) -> str:
        segments = list(reversed(params["segment_batch_ids"]))
        self._set_progress(0, len(segments))
        for index, batch_id in enumerate(segments, start=1):
            result = self._run_slice(
                dedup_auto_stack_service.revert_segment_in_session,
                self._run_id,
                batch_id,
                self._vault.interactive_rescore_registry,
                self._vault.image_root,
            )
            restore = result.get("restore")
            if restore is not None:
                operation_log_service.announce_restore(self._vault, **restore)
            if result["status"] != AUTO_STACK_RUNNING:
                return result["status"]
            self._set_progress(index, len(segments))
        return AUTO_STACK_COMPLETE

    def _run_slice(self, func, *args):
        """Run one LOW writer slice, unless the runner has cancelled the task."""
        if self._cancel_event.is_set():
            raise _AutoStackInterrupted
        return self._vault.db.run_task(func, *args, priority=DBPriority.LOW)

    def _flush_notifications(self, force: bool = False) -> None:
        """Announce the pictures stacked since the last flush, coalesced."""
        now = time.monotonic()
        if not self._pending_picture_ids:
            return
        if not force and now - self._last_notify < self.NOTIFY_INTERVAL_S:
            return
        picture_ids = sorted(self._pending_picture_ids)
        self._pending_picture_ids.clear()
        self._last_notify = now
        notify_pictures_changed(
            self._vault, picture_ids, self._origin_client_id, self._source
        )

    def _set_progress(self, processed: int, total: int) -> None:
        with self._progress_lock:
            self._total_count = max(1, int(total))
            self._processed_count = min(max(0, int(processed)), self._total_count)
