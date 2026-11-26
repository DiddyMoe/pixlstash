"""RestoreService facade for the restore package.

Assembles the restore behaviour split across :mod:`schema_upgrade`,
:mod:`full_restore`, :mod:`resource_restore`, and :mod:`preview` into the
single :class:`RestoreService` class, and re-exports the public data models
and exceptions.  This package is the import path for all of them (see backend
refactor plan §4.4).
"""

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

from sqlmodel import Session, select

from pixlstash.db_models.dedup import SCAN_PENDING, SCAN_RUNNING, DedupScan
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services.dedup_auto_stack_service import active_run_id
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    LibrarySwitchTimeoutError,
)

from ._models import (
    RESOURCE_RESTORE_FAILED_MESSAGE,
    RESTORE_FAILED_MESSAGE,
    BackgroundWorkStillRunningError,
    LibraryFence,
    LibraryStillInUseError,
    LibrarySwitch,
    MissingDependenciesError,
    ResourcePreview,
    RestoreInProgressError,
    RestorePreview,
    RestoreRefusedError,
    RestoreReport,
    SafetySnapshotFailedError,
    UNSUPPORTED_RESOURCE_TYPE_MESSAGE,
    SnapshotNotFoundError,
    UnsupportedResourceTypeError,
    WorkInProgressError,
    task_label,
)
from .full_restore import FullRestoreMixin
from .preview import PreviewMixin
from .resource_restore import ResourceRestoreMixin
from .schema_upgrade import SchemaUpgradeMixin

if TYPE_CHECKING:
    from pixlstash.task_runner import RunningTask
    from pixlstash.vault import Vault

logger = get_logger(__name__)

# The statuses ``DedupScanTask.find_pending_scan`` picks a scan up in: a
# pending row is one a person asked for that has no task yet, and a running
# one is picked up again after a stop. Not imported from the task: the tasks
# package imports the snapshot service, which imports this package.
_DEDUP_SCAN_ACTIVE = (SCAN_PENDING, SCAN_RUNNING)


def _runs_waiting_in_session(session: Session) -> list[str]:
    """The ``task_type`` of each kind of run a person asked for whose row is
    pending or running: work a finder submits later, so no task shows it."""
    waiting = []
    if active_run_id(session) is not None:
        waiting.append("DedupAutoStackTask")
    scan = session.exec(
        select(DedupScan.id).where(DedupScan.status.in_(_DEDUP_SCAN_ACTIVE)).limit(1)
    ).first()
    if scan is not None:
        waiting.append("DedupScanTask")
    return waiting


class RestoreService(
    SchemaUpgradeMixin,
    FullRestoreMixin,
    ResourceRestoreMixin,
    PreviewMixin,
):
    """Restores vault metadata from a snapshot snapshot.

    Attributes:
        _vault: Back-reference to the owning Vault.
    """

    # How long a restore waits for other work to let go of the library before
    # it refuses with nothing changed: first for read leases to be released
    # (LibraryStillInUseError), then, within what is left of the same
    # deadline, for running background tasks to stop
    # (BackgroundWorkStillRunningError). The same bound as the auth gate's
    # HTTP drain (``AuthService._RESTORE_DRAIN_TIMEOUT_SECONDS``): a
    # cooperative task stops at its next item, so 30 s covers one batch that
    # cannot be interrupted. The auth gate waits its own 30 s after this one,
    # so a full restore adds at most 60 s before the swap, which is nginx's
    # default ``proxy_read_timeout``
    # (http://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_read_timeout)
    # for the synchronous restore request.
    _DRAIN_TIMEOUT_S: float = 30.0

    # Why admission is paused; ``TaskCancelledError`` and the log name it.
    _DRAIN_REASON = "library restore"

    # Exceptions whose message the restore composed itself from fixed text and
    # validated values, so a client may be shown it. Any other exception is
    # logged and reported with a fixed sentence.
    _CLIENT_VISIBLE_ERRORS = (
        RestoreRefusedError,
        SafetySnapshotFailedError,
        MissingDependenciesError,
    )

    def __init__(self, vault: "Vault") -> None:
        """Initialise the service.

        Args:
            vault: The owning Vault instance.
        """
        self._vault = vault
        # Tracks the currently executing restore job for /snapshots/status.
        # Read via ``get_active_job()`` from outside this module.
        self._active_job: Optional[dict] = None
        # Mutual exclusion for restore_full / restore_resource / restore_batch.
        # Held for the entire duration of a restore (including the queued
        # _do_swap + _post_restore_cleanup tasks); any concurrent restore call
        # short-circuits with RestoreInProgressError → 409.
        self._restore_lock = threading.Lock()
        # Per-snapshot-file locks. compare_hashes can rewrite an old snapshot
        # in place to backfill metadata_hash; concurrent compare/preview/
        # restore on the same path would otherwise race on disk (corrupt
        # copy, partial read). Reentrant so a caller can hold the file lock
        # across multiple helpers (e.g. compare_hashes → _backfill_snapshot)
        # without self-deadlocking. The meta-lock guards the dict itself.
        self._snapshot_file_locks: dict[str, threading.RLock] = {}
        self._snapshot_file_locks_meta: threading.Lock = threading.Lock()

    def get_active_job(self) -> Optional[dict]:
        """Return the in-flight restore job descriptor, or ``None`` if idle.

        The shape is ``{"kind": "RESTORE", "snapshot_id": int,
        "started_at": isoformat, "progress": float}`` while a restore runs.
        Used by ``GET /snapshots/status``.
        """
        return self._active_job

    @contextmanager
    def _library_switched(
        self, fence: Optional[LibraryFence], deadline: float
    ) -> Iterator[LibrarySwitch]:
        """Close library admission for a full restore, and open it on every exit.

        ``begin_switch`` refuses new read leases and waits, until *deadline*,
        for every lease of the current generation but the restore request's
        own. Those are the writers that are not tasks: an import thread, a
        folder mapping commit, a ComfyUI output import, the edit-copy import,
        the smart-set refresher and every other request. None of them can
        then write into the file the restore swaps in.

        On the way out, admission is READY again: with a new generation
        (``publish_ready``) once the swap was attempted, so work that began
        against the old file is turned away; with the same one
        (``restore_ready``) otherwise. When the runtime is incoherent the
        coordinator marks itself UNAVAILABLE instead and raises; that is
        raised on a successful restore and logged when the restore is already
        raising, so the restore's own error is the one reported.

        Without a fence (the CLI, a direct service call) nothing is switched.

        Raises:
            RestoreRefusedError: Admission was not READY (a library switch is
                under way, or the runtime is unavailable); nothing was changed.
            LibraryStillInUseError: A lease was still held at *deadline*;
                admission is READY again and nothing was changed.
        """
        switch = LibrarySwitch()
        if fence is None:
            yield switch
            return
        coordinator = fence.coordinator
        try:
            coordinator.begin_switch(
                timeout=max(0.0, deadline - time.monotonic()),
                exclude=fence.own_lease,
                restoring=True,
            )
        except LibraryNotReadyError as exc:
            # A library switch holds admission (it is waiting for this very
            # request's lease to end), or the runtime is unavailable. Waiting
            # here would only make the two wait on each other.
            logger.error(
                "RestoreService: refusing the restore; library admission is %s.",
                exc.state.value,
            )
            raise RestoreRefusedError(
                "The restore did not start because the library is being "
                "switched or is unavailable. Nothing was changed. Try again "
                "once the switch has finished."
            ) from exc
        except LibrarySwitchTimeoutError as exc:
            holder = self._describe_holders(fence)
            logger.error(
                "RestoreService: refusing the restore; the library was still in "
                "use after %.0f s (%s): %s",
                self._DRAIN_TIMEOUT_S,
                holder or "holder unknown",
                exc,
            )
            raise LibraryStillInUseError(holder, self._DRAIN_TIMEOUT_S) from exc
        logger.info("RestoreService: library admission closed for the restore.")

        def end() -> None:
            if switch.swap_attempted:
                coordinator.publish_ready()
            else:
                coordinator.restore_ready()
            logger.info(
                "RestoreService: library admission reopened (%s generation).",
                "new" if switch.swap_attempted else "same",
            )

        try:
            yield switch
        except BaseException:
            try:
                end()
            except RuntimeError:
                logger.critical(
                    "RestoreService: library admission could not reopen after a "
                    "failed restore; the library is unavailable until a restart.",
                    exc_info=True,
                )
            raise
        end()

    @contextmanager
    def _background_work_paused(
        self, deadline: Optional[float] = None
    ) -> Iterator[None]:
        """Hold every background task off the library for the block.

        First, while work a person started is queued or running
        (``TaskRunner.is_person_started``), the restore is refused with
        nothing paused, stopped or cancelled. The look and the admission pause
        are one step (``TaskRunner.pause_admission_unless``), so such work
        submitted after the look is refused by the pause
        (``TaskAdmissionRefused``) rather than missed.

        Then it stops the WorkPlanner, if it was running, so no finder submits
        new work, and refuses while a run a person asked for waits in its row
        (an Auto-stack run or a duplicate scan, pending or running): a finder
        submits that work, so no task shows it, and with the planner stopped
        and admission paused no task can take it up now. Then it drains the
        TaskRunner with ``pause_and_drain``: queued tasks are cancelled,
        running ones are asked to stop, and nothing is admitted until the
        block ends. Their finders find that work again. Then it stops the
        vault's own writers that no task or lease fences
        (``Vault.quiesce_background_writers``: the changed-tags flush, the
        reference-folder watcher, face extraction's follow-ups). If a task or a
        follow-up is still running at the deadline, the restore is refused
        before it has changed anything. Those writers restart, admission
        resumes and the planner restarts, if it was running, on every exit,
        the refusals included; when the block is already raising, a failure to
        restart them is logged so the block's own error is the one reported.

        Args:
            deadline: A ``time.monotonic()`` deadline shared with an earlier
                wait (a full restore's ``_library_switched``), so the two
                together stay within ``_DRAIN_TIMEOUT_S``. ``None`` gives the
                drain the whole ``_DRAIN_TIMEOUT_S``.

        Raises:
            WorkInProgressError: Work a person started was queued or running,
                or waited in its row; nothing was cancelled. It names the work.
            BackgroundWorkStillRunningError: A task, or a vault writer, was
                still running after ``_DRAIN_TIMEOUT_S``. It names them.
        """
        if deadline is None:
            deadline = time.monotonic() + self._DRAIN_TIMEOUT_S
        planner = self._vault._work_planner
        task_runner = self._vault._task_runner
        planner_was_running = planner is not None and planner.is_running()
        if task_runner is not None:
            started = task_runner.pause_admission_unless(
                self._DRAIN_REASON, task_runner.is_person_started
            )
            if started:
                self._refuse_for_work_in_progress(started)
        failed = True
        try:
            if planner_was_running:
                planner.stop()
                logger.info("RestoreService: WorkPlanner stopped for the restore.")
            self._refuse_for_waiting_runs()
            if task_runner is not None:
                result = task_runner.pause_and_drain(
                    max(0.0, deadline - time.monotonic()), self._DRAIN_REASON
                )
                if not result.drained:
                    still_running = sorted(
                        {task_label(t.task_type) for t in result.still_running}
                    )
                    logger.error(
                        "RestoreService: refusing the restore; %d background "
                        "task(s) still running after %.0f s: %s",
                        len(result.still_running),
                        self._DRAIN_TIMEOUT_S,
                        ", ".join(
                            f"{t.task_type} {t.task_id}" for t in result.still_running
                        ),
                    )
                    raise BackgroundWorkStillRunningError(
                        still_running, self._DRAIN_TIMEOUT_S
                    )
                logger.info(
                    "RestoreService: background work drained; %d pending "
                    "task(s) cancelled.",
                    result.cancelled,
                )
            # After the drain: a face follow-up is queued by a task that has
            # just ended, so it can only be waited for once none runs.
            still_writing = self._vault.quiesce_background_writers(
                max(0.0, deadline - time.monotonic())
            )
            if still_writing:
                logger.error(
                    "RestoreService: refusing the restore; vault writer(s) "
                    "still running after %.0f s: %s",
                    self._DRAIN_TIMEOUT_S,
                    ", ".join(still_writing),
                )
                raise BackgroundWorkStillRunningError(
                    still_writing, self._DRAIN_TIMEOUT_S
                )
            yield
            failed = False
        finally:
            self._resume_background_work(
                task_runner, planner if planner_was_running else None, failed
            )

    def _resume_background_work(self, task_runner, planner, failed: bool) -> None:
        """Restart the vault writers, resume admission and restart *planner*.

        Each step runs whatever the one before did. When the restore is
        already failing (*failed*), a step that raises is logged with its
        traceback and the restore's own error is the one raised; otherwise
        the step's error is raised.
        """
        try:
            # A no-op for whatever the quiesce did not stop, so it is safe
            # after a refusal or a quiesce that raised partway.
            self._vault.resume_background_writers()
        except Exception:
            if not failed:
                raise
            logger.error(
                "RestoreService: the vault writers did not restart after a "
                "failed restore.",
                exc_info=True,
            )
        finally:
            if task_runner is not None:
                task_runner.resume_admission(self._DRAIN_REASON)
            if planner is not None:
                try:
                    planner.start()
                except Exception:
                    if not failed:
                        raise
                    logger.error(
                        "RestoreService: the WorkPlanner did not restart after "
                        "a failed restore; background work waits for a "
                        "restart.",
                        exc_info=True,
                    )
                else:
                    logger.info(
                        "RestoreService: WorkPlanner restarted after the restore."
                    )

    def _refuse_while_somebody_started_work(self) -> None:
        """Refuse a restore up front while work a person started is waiting.

        A look only, made before anything costly: ``_background_work_paused``
        looks again and pauses in the same step. Looking here as well keeps a
        restore from emitting ``RESTORE_STARTED``, upgrading the snapshot or,
        for a full restore, closing library admission and so turning every
        request away, for a restore that would refuse anyway.

        Raises:
            WorkInProgressError: A task a person started was queued or
                running, or a run they asked for waited in its row.
        """
        task_runner = self._vault._task_runner
        if task_runner is not None:
            started = task_runner.unsettled_work(task_runner.is_person_started)
            if started:
                self._refuse_for_work_in_progress(started)
        self._refuse_for_waiting_runs()

    def _refuse_for_waiting_runs(self) -> None:
        """Refuse while a run a person asked for is pending or running in its
        row (``_runs_waiting_in_session``).

        Raises:
            WorkInProgressError: Such a run exists; it names the work.
        """
        waiting = self._vault.db.run_immediate_read_task(_runs_waiting_in_session)
        if waiting:
            logger.warning(
                "RestoreService: refusing the restore; run(s) a person asked "
                "for are still pending or running: %s",
                ", ".join(waiting),
            )
            raise WorkInProgressError(waiting)

    def _refuse_for_work_in_progress(self, started: "tuple[RunningTask, ...]") -> None:
        """Log the work that blocks a restore, and raise WorkInProgressError."""
        logger.warning(
            "RestoreService: refusing the restore; %d task(s) a person started "
            "still queued or running: %s",
            len(started),
            ", ".join(f"{t.task_type} {t.task_id}" for t in started),
        )
        raise WorkInProgressError([t.task_type for t in started])

    def _describe_holders(self, fence: LibraryFence) -> Optional[str]:
        """Name what holds the library, for a refusal already decided.

        ``None`` when the fence has no ``describe_holders`` or it raised. The
        name only adorns the refusal: a failure to compute it is logged, and
        the restore still refuses with nothing changed (409), not 500.
        """
        if fence.describe_holders is None:
            return None
        try:
            return fence.describe_holders()
        except Exception:
            logger.error(
                "RestoreService: could not name what holds the library; "
                "refusing the restore without naming it.",
                exc_info=True,
            )
            return None

    def _emit_restore_failed(
        self, started_payload: dict, exc: Exception, failed_message: str
    ) -> None:
        """Emit RESTORE_FAILED with text a client may be shown.

        A message the restore composed itself (``_CLIENT_VISIBLE_ERRORS``) is
        passed on; any other exception is logged with its traceback and the
        event carries *failed_message* instead.
        """
        if isinstance(exc, self._CLIENT_VISIBLE_ERRORS):
            error = str(exc)
        else:
            logger.error(
                "RestoreService: restore failed (%s)",
                started_payload,
                exc_info=exc,
            )
            error = failed_message
        self._emit_lifecycle(
            EventType.RESTORE_FAILED, {**started_payload, "error": error}
        )

    def _emit_lifecycle(self, event_type: EventType, payload: dict) -> None:
        """Emit a RESTORE_STARTED / _COMPLETED / _FAILED event, swallowing
        emit failures so a flaky event bus cannot derail the restore itself.
        """
        try:
            self._vault.emit_event(event_type, payload)
        except Exception as exc:
            logger.warning(
                "RestoreService: failed to emit %s: %s", event_type.name, exc
            )


__all__ = [
    "RestoreService",
    "RestoreReport",
    "RestorePreview",
    "ResourcePreview",
    "RestoreInProgressError",
    "SafetySnapshotFailedError",
    "MissingDependenciesError",
    "RestoreRefusedError",
    "BackgroundWorkStillRunningError",
    "LibraryFence",
    "LibraryStillInUseError",
    "SnapshotNotFoundError",
    "UnsupportedResourceTypeError",
    "WorkInProgressError",
    "RESTORE_FAILED_MESSAGE",
    "RESOURCE_RESTORE_FAILED_MESSAGE",
    "UNSUPPORTED_RESOURCE_TYPE_MESSAGE",
]
