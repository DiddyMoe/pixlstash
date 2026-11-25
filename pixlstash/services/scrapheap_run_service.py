"""Background Scrapheap runs: empty, delete forever and restore, a slice at a time.

Emptying the Scrapheap used to be one request: one IMMEDIATE writer callback
that planned, ledgered and deleted every picture, then removed the files.
Destroying a picture costs about 0.7 ms, so a 50,000-picture heap held the
single SQLite writer for over half a minute and outlived the client's 60 s
timeout. The user saw "Couldn't empty the scrapheap" over a purge that was
still running, with no progress and no way to stop it. "Restore all" had the
same shape.

A run follows the asynchronous request-reply pattern, as auto-stack runs do
(:mod:`pixlstash.services.dedup_auto_stack_service`): the request reserves the
run, answers 202 with its address, and the client polls it. A background task
(:class:`~pixlstash.tasks.scrapheap_run_task.ScrapheapRunTask`) works through
the pictures in short LOW-priority writer slices, so every other write gets the
writer between them, and the run is the progress readout.

**Every slice is the existing path.** A purge slice is
:func:`~pixlstash.services.scrapheap_service.purge_scrapheap_pictures` over that
slice's ids, so the locked-set freeze, the protected-original skip, the re-check
of ``deleted`` inside the DELETE, the ledger-before-file order and the hub reach
all hold per slice exactly as they did for one request. A restore slice is
:func:`~pixlstash.services.scrapheap_service.restore_pictures`. Nothing here
decides what may be destroyed.

**The selection is fixed when the run starts.** "The whole Scrapheap" resolves
to the ids in it at that moment, so a picture scrapheaped after the user
confirmed is not destroyed by that confirmation. A picture restored meanwhile is
skipped by the slice that reaches it, as before.

**History in segments.** A purge is not recorded (it cannot be undone). A
restore is, and one operation per slice would make "restore all" a hundred
Ctrl+Z presses, while one batch for the whole run would make one Ctrl+Z restore
50,000 pictures in a single writer callback, which is the problem this module
removes. So slices record into bounded batches ("segments") of at most
:data:`SEGMENT_PICTURE_CAP` pictures. A segment also ends whenever some other
undoable operation lands on top of it between slices: a batch must stay
contiguous at the top of History to be undone
(``operation_log_service._enforce_latest_undo_unit``), and appending to a buried
batch would make it and the operation above it both un-undoable. The decision is
made on the slice's own writer session, so no write can land between the check
and the record.

**One run at a time.** Purge and restore share the rule; a second request gets
409 with the active run.

**In memory.** A run lives in the vault's :class:`ScrapheapRunRegistry`, not in
a table: a purge leaves nothing to revert and a restore's undo is History's, so
nothing needs the run once it ends. It ends with the process or the library. A
restart or a library switch interrupts it at the next slice boundary;
everything committed stays committed, and the pictures it had not reached are
still in the Scrapheap, so starting again finishes the job.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.pixl_logging import get_logger
from pixlstash.services import operation_log_service, scrapheap_service
from pixlstash.utils.service.scope_table import scope_id_subquery

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pixlstash.vault import Vault

logger = get_logger(__name__)

_T = TypeVar("_T")

KIND_PURGE = "purge"
KIND_RESTORE = "restore"
KINDS = (KIND_PURGE, KIND_RESTORE)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_RUNNING})

SEGMENT_PICTURE_CAP = 5000
"""Pictures per History segment of a restore run. Undoing 5,000 restored
pictures measured about 2 s, so one Ctrl+Z stays a few seconds of writer time."""

HISTORY_KEPT = 20
"""Ended runs the registry keeps, newest first, so a client that polls a run
after it ended (another tab, a slow poll) still finds it."""

ERROR_ACTIVE = "scrapheap_run_active"
ERROR_NOT_FOUND = "scrapheap_run_not_found"
ERROR_UNAVAILABLE = "scrapheap_runs_unavailable"
ERROR_SERVER_STOPPED = "server_stopped"
ERROR_UNEXPECTED = "unexpected_error"


class ScrapheapRunError(Exception):
    """A run request the service refuses, with a stable code for the client.

    The route maps :attr:`code` to an HTTP status; the service names no status
    so the rule lives in one place.

    Attributes:
        code: :data:`ERROR_ACTIVE`, :data:`ERROR_NOT_FOUND` or
            :data:`ERROR_UNAVAILABLE`.
        run: The run the refusal is about (the active one for
            :data:`ERROR_ACTIVE`), as :meth:`ScrapheapRun.view` shapes it, or
            ``None``.
    """

    def __init__(self, code: str, message: str, run: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.run = run


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


@dataclass
class ScrapheapRun:
    """One background Scrapheap run. Mutated only under the registry's lock.

    Attributes:
        id: The run's id, a UUID string.
        kind: :data:`KIND_PURGE` or :data:`KIND_RESTORE`.
        include_protected: A purge's choice to destroy protected
            reference-folder originals too. Always ``False`` for a restore.
        status: One of the ``STATUS_*`` values.
        total: Pictures the run set out to handle, fixed when it starts.
        processed: Pictures handled so far, whatever the outcome.
        cancel_requested: The user asked it to stop; it does at the next slice.
        deleted_count: Pictures destroyed (purge).
        skipped_count: Protected originals left in the Scrapheap (purge).
        skipped_locked: Ids a locked picture set froze (purge).
        skipped_restored: Pictures that had left the Scrapheap before their
            slice reached them (purge).
        ghosts_kept: Picture ghosts the hub kept (purge).
        ghosts_cascaded: Picture ghosts the hub destroyed (purge).
        restored_count: Pictures brought back (restore).
        snapshots_with_deleted: Snapshots still holding metadata for the
            destroyed pictures, set when a purge ends.
        segment_batch_ids: The History batches a restore recorded into.
        segment_pictures: Pictures recorded into the newest segment.
        purged_ids: Every id the purge destroyed, for the snapshot check.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.
        picture_ids: The resolved selection, in the order slices take it.
        error_code: Why it ended early, when it did.
        error: The detail of a failure, for the log and the client.
        created_at: When it was requested.
        started_at: When the task took it.
        updated_at: When anything about it last changed.
        finished_at: When it ended.
    """

    kind: str
    include_protected: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = STATUS_PENDING
    total: int = 0
    processed: int = 0
    cancel_requested: bool = False
    deleted_count: int = 0
    skipped_count: int = 0
    skipped_locked: list[int] = field(default_factory=list)
    skipped_restored: int = 0
    ghosts_kept: int = 0
    ghosts_cascaded: int = 0
    restored_count: int = 0
    snapshots_with_deleted: Optional[list[dict]] = None
    segment_batch_ids: list[str] = field(default_factory=list)
    segment_pictures: int = 0
    purged_ids: list[int] = field(default_factory=list)
    actor: Optional[str] = None
    source: str = "external"
    origin_client_id: Optional[str] = None
    picture_ids: list[int] = field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=_now)
    finished_at: Optional[datetime] = None

    @property
    def active(self) -> bool:
        """Whether the run is still pending or running."""
        return self.status in ACTIVE_STATUSES

    def view(self) -> dict[str, Any]:
        """Shape the run for the API.

        ``skipped_locked`` and ``snapshots_with_deleted`` are reported once the
        run has ended; while it runs a poll carries the count only, so a heap
        full of locked pictures does not make every poll carry every id.
        """
        active = self.active
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "active": active,
            "include_protected": self.include_protected,
            "total": self.total,
            "processed": self.processed,
            "cancel_requested": self.cancel_requested,
            "deleted_count": self.deleted_count,
            "skipped_count": self.skipped_count,
            "skipped_locked_count": len(self.skipped_locked),
            "skipped_locked": [] if active else sorted(self.skipped_locked),
            "skipped_restored": self.skipped_restored,
            "ghosts_kept": self.ghosts_kept,
            "ghosts_cascaded": self.ghosts_cascaded,
            "restored_count": self.restored_count,
            "snapshots_with_deleted": None if active else self.snapshots_with_deleted,
            "segment_batch_ids": list(self.segment_batch_ids),
            "error_code": self.error_code,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "updated_at": _iso(self.updated_at),
            "finished_at": _iso(self.finished_at),
        }


class ScrapheapRunRegistry:
    """The vault's Scrapheap runs: the active one and the last few that ended.

    Request threads read and cancel while the task thread folds slice results
    in, so every access takes one lock, and every read returns a
    :meth:`ScrapheapRun.view` copy rather than the live run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, ScrapheapRun] = OrderedDict()

    def reserve(
        self,
        kind: str,
        include_protected: bool,
        actor: Optional[str],
        source: Optional[str],
        origin_client_id: Optional[str],
    ) -> str:
        """Record a pending run, unless one is already active.

        Returns:
            The new run's id.

        Raises:
            ScrapheapRunError: :data:`ERROR_ACTIVE`, carrying the active run.
        """
        with self._lock:
            active = self._active_locked()
            if active is not None:
                raise ScrapheapRunError(
                    ERROR_ACTIVE,
                    "A Scrapheap run is already in progress.",
                    active.view(),
                )
            run = ScrapheapRun(
                kind=kind,
                include_protected=bool(include_protected) and kind == KIND_PURGE,
                actor=actor,
                source=source or "external",
                origin_client_id=origin_client_id,
            )
            self._runs[run.id] = run
            self._trim_locked()
            return run.id

    def discard(self, run_id: str) -> None:
        """Forget a run that never reached the task runner."""
        with self._lock:
            self._runs.pop(run_id, None)

    def get(self, run_id: str) -> dict[str, Any]:
        """Return one run.

        Raises:
            ScrapheapRunError: :data:`ERROR_NOT_FOUND`.
        """
        with self._lock:
            return self._get_locked(run_id).view()

    def latest(self) -> Optional[dict[str, Any]]:
        """Return the newest run, or ``None`` before the first."""
        with self._lock:
            if not self._runs:
                return None
            return next(reversed(self._runs.values())).view()

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Ask a run to stop.

        A pending run is cancelled at once. A running one is flagged and stops
        at its next slice boundary, keeping what it has committed. An ended run
        is returned unchanged, so a repeated cancel is harmless.

        Raises:
            ScrapheapRunError: :data:`ERROR_NOT_FOUND`.
        """
        with self._lock:
            run = self._get_locked(run_id)
            if not run.active:
                return run.view()
            now = _now()
            run.cancel_requested = True
            if run.status == STATUS_PENDING:
                run.status = STATUS_CANCELLED
                run.finished_at = now
            run.updated_at = now
            logger.info("[scrapheap-run] %s: cancel requested (%s)", run_id, run.status)
            return run.view()

    def mutate(self, run_id: str, change: Callable[[ScrapheapRun], _T]) -> _T:
        """Apply *change* to the live run under the lock and return its result.

        For the service's own slice bookkeeping; *change* must not block.

        Raises:
            LookupError: The run is gone (trimmed, or never recorded).
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise LookupError(f"Scrapheap run {run_id} no longer exists")
            return change(run)

    def finish(
        self,
        run_id: str,
        status: str,
        error_code: Optional[str] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """Close a still-active run with *status*; an ended run is left as is.

        Raises:
            LookupError: The run is gone.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise LookupError(f"Scrapheap run {run_id} no longer exists")
            if run.active:
                now = _now()
                run.status = status
                run.error_code = error_code
                run.error = (error or "")[:2000] or None
                run.finished_at = now
                run.updated_at = now
                # A restore segment the last slice opened but recorded nothing
                # into is not a History batch; reporting it would name an undo
                # that does not exist.
                if run.segment_batch_ids and run.segment_pictures == 0:
                    run.segment_batch_ids.pop()
            return run.view()

    def _active_locked(self) -> Optional[ScrapheapRun]:
        for run in self._runs.values():
            if run.active:
                return run
        return None

    def _get_locked(self, run_id: str) -> ScrapheapRun:
        run = self._runs.get(str(run_id))
        if run is None:
            raise ScrapheapRunError(ERROR_NOT_FOUND, f"No Scrapheap run {run_id}")
        return run

    def _trim_locked(self) -> None:
        ended = [run_id for run_id, run in self._runs.items() if not run.active]
        for run_id in ended[: max(0, len(ended) - HISTORY_KEPT)]:
            del self._runs[run_id]


def scrapheap_ids_in_session(
    session: Session, picture_ids: Optional[list[int]]
) -> list[int]:
    """The ids of *picture_ids* still in the Scrapheap, oldest first.

    ``None`` means the whole Scrapheap. A temp-table scope, so a selection of
    any size binds no per-id parameters.
    """
    query = select(Picture.id).where(Picture.deleted.is_(True))
    if picture_ids is not None:
        scope = scope_id_subquery(
            session, picture_ids, name="_pixlstash_scrapheap_run_ids"
        )
        query = query.where(Picture.id.in_(scope))
    rows = session.exec(query.order_by(Picture.id)).all()
    return [int(pid) for pid in rows if pid is not None]


def start_run(
    vault: "Vault",
    kind: str,
    picture_ids: Optional[list[int]] = None,
    include_protected: bool = False,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> dict[str, Any]:
    """Reserve a run, fix its selection and hand it to the task runner.

    A purge's consent check (the delete-preview ``confirm_token``) belongs to
    the caller and must already have passed: this function destroys nothing
    itself, but the run it starts does.

    Args:
        vault: The vault whose Scrapheap it works on.
        kind: :data:`KIND_PURGE` or :data:`KIND_RESTORE`.
        picture_ids: The selection, or ``None`` for the whole Scrapheap.
        include_protected: A purge's choice to destroy protected originals.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.

    Returns:
        The run as the API shows it.

    Raises:
        ScrapheapRunError: :data:`ERROR_ACTIVE`, or :data:`ERROR_UNAVAILABLE`
            when this server runs no background tasks or is stopping.
    """
    # Imported here, not at the top: the task module imports this one for its
    # slices, and every task module is loaded through ``pixlstash.tasks``.
    from pixlstash.tasks.scrapheap_run_task import ScrapheapRunTask

    if kind not in KINDS:
        raise ValueError(f"Unknown Scrapheap run kind {kind!r}")
    registry = vault.scrapheap_runs
    run_id = registry.reserve(kind, include_protected, actor, source, origin_client_id)
    try:
        ids = vault.db.run_immediate_read_task(scrapheap_ids_in_session, picture_ids)
        registry.mutate(run_id, lambda run: _record_selection(run, ids))
    except BaseException:
        registry.discard(run_id)
        raise
    try:
        submitted = vault.submit_task(ScrapheapRunTask(vault, run_id))
    except RuntimeError as exc:
        # ``TaskRunner.submit`` raises only this, and only once it has stopped.
        registry.discard(run_id)
        logger.warning(
            "[scrapheap-run] %s run not started: the task runner is stopping (%s)",
            kind,
            exc,
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "The server is stopping, so nothing was started.",
        ) from exc
    except BaseException:
        registry.discard(run_id)
        raise
    if submitted is None:
        registry.discard(run_id)
        logger.warning(
            "[scrapheap-run] %s run not started: this server runs no background "
            "tasks (disable_background_workers)",
            kind,
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "This server runs no background tasks, so nothing was started.",
        )
    logger.info(
        "[scrapheap-run] %s: %s of %d picture(s) requested (include_protected=%s, "
        "actor=%s)",
        run_id,
        kind,
        len(ids),
        include_protected,
        actor,
    )
    return registry.get(run_id)


def get_run(vault: "Vault", run_id: str) -> dict[str, Any]:
    """Return one run.

    Raises:
        ScrapheapRunError: :data:`ERROR_NOT_FOUND`.
    """
    return vault.scrapheap_runs.get(run_id)


def latest_run(vault: "Vault") -> Optional[dict[str, Any]]:
    """Return the newest run, or ``None``.

    What a client reads when the Scrapheap opens, so a run started before a
    reload, or in another tab, is picked up again.
    """
    return vault.scrapheap_runs.latest()


def cancel_run(vault: "Vault", run_id: str) -> dict[str, Any]:
    """Ask a run to stop; see :meth:`ScrapheapRunRegistry.cancel`."""
    return vault.scrapheap_runs.cancel(run_id)


# --- Task slices --------------------------------------------------------------
# Called from the task thread. Each slice is one existing Scrapheap path at LOW
# writer priority; the registry is updated after it, under its lock.


def begin_run(vault: "Vault", run_id: str) -> Optional[dict[str, Any]]:
    """Mark a pending run running and return what the task needs to work it.

    Returns:
        ``None`` when the run is no longer pending (cancelled before the task
        took it), otherwise its ``kind``, ``picture_ids``, ``origin_client_id``
        and ``source``.
    """

    def _begin(run: ScrapheapRun) -> Optional[dict[str, Any]]:
        if run.status != STATUS_PENDING:
            return None
        now = _now()
        run.status = STATUS_RUNNING
        run.started_at = now
        run.updated_at = now
        return {
            "kind": run.kind,
            "picture_ids": list(run.picture_ids),
            "origin_client_id": run.origin_client_id,
            "source": run.source,
        }

    return vault.scrapheap_runs.mutate(run_id, _begin)


def cancel_requested(vault: "Vault", run_id: str) -> bool:
    """Whether the user has asked the run to stop."""
    return vault.scrapheap_runs.mutate(run_id, lambda run: run.cancel_requested)


def purge_slice(vault: "Vault", run_id: str, chunk: list[int]) -> list[int]:
    """Destroy one slice of a purge run's selection.

    Files are removed inline, on the task thread: this is already off the
    request, and the next slice must not start ahead of this one's reconcile.

    Returns:
        The ids this slice destroyed.
    """
    include_protected = vault.scrapheap_runs.mutate(
        run_id, lambda run: run.include_protected
    )
    outcome = scrapheap_service.purge_scrapheap_pictures(
        vault, list(chunk), include_protected, priority=DBPriority.LOW
    )
    vault.scrapheap_runs.mutate(
        run_id, lambda run: _absorb_purge(run, outcome, len(chunk))
    )
    return list(outcome.purged_ids)


def restore_slice(vault: "Vault", run_id: str, chunk: list[int]) -> list[int]:
    """Restore one slice of a restore run's selection into its History segment.

    Returns:
        The ids this slice brought back.
    """
    registry = vault.scrapheap_runs
    actor, source, origin = registry.mutate(
        run_id, lambda run: (run.actor, run.source, run.origin_client_id)
    )
    slice_size = len(chunk)

    def _choose_segment(session: Session) -> str:
        newest = operation_log_service.newest_undo_batch_id_in_session(session)
        return registry.mutate(
            run_id, lambda run: _segment_for_slice(run, newest, slice_size)
        )

    restored = scrapheap_service.restore_pictures(
        vault,
        list(chunk),
        actor=actor,
        source=source,
        origin_client_id=origin,
        batch_id=_choose_segment,
        priority=DBPriority.LOW,
        announce=False,
    )
    registry.mutate(run_id, lambda run: _absorb_restore(run, restored, slice_size))
    return list(restored)


def finish_run(vault: "Vault", run_id: str, status: str) -> dict[str, Any]:
    """Close a run the task worked to the end of, or stopped on request.

    A purge that destroyed anything names the snapshots still holding those
    pictures' metadata, as the synchronous delete does. That reads only the
    snapshot manifests.
    """
    registry = vault.scrapheap_runs
    kind, purged_ids = registry.mutate(
        run_id, lambda run: (run.kind, list(run.purged_ids))
    )
    if kind == KIND_PURGE:
        snapshots = (
            vault.snapshot_service.snapshots_containing(purged_ids)
            if purged_ids
            else []
        )

        def _record(run: ScrapheapRun) -> None:
            run.snapshots_with_deleted = snapshots

        registry.mutate(run_id, _record)
    view = registry.finish(run_id, status)
    logger.info(
        "[scrapheap-run] %s: %s %s, %d of %d processed, %d deleted, %d restored, "
        "%d protected kept, %d locked kept, %d already out of the Scrapheap",
        run_id,
        view["kind"],
        view["status"],
        view["processed"],
        view["total"],
        view["deleted_count"],
        view["restored_count"],
        view["skipped_count"],
        view["skipped_locked_count"],
        view["skipped_restored"],
    )
    return view


def close_run(
    vault: "Vault", run_id: str, status: str, error: Optional[str] = None
) -> None:
    """Mark a still-active run interrupted or failed, from the task's own paths.

    A run that is already gone is logged, not raised: this runs while a task is
    failing or the server is stopping, and must not mask why.
    """
    if status == STATUS_INTERRUPTED:
        error_code = ERROR_SERVER_STOPPED
    else:
        error_code = ERROR_UNEXPECTED
    try:
        vault.scrapheap_runs.finish(run_id, status, error_code, error)
    except LookupError:
        logger.warning(
            "[scrapheap-run] cannot mark run %s %s: it is no longer registered",
            run_id,
            status,
        )


def _record_selection(run: ScrapheapRun, ids: list[int]) -> None:
    run.picture_ids = list(ids)
    run.total = len(ids)
    run.updated_at = _now()


def _absorb_purge(
    run: ScrapheapRun, outcome: scrapheap_service.ScrapheapPurgeOutcome, size: int
) -> None:
    run.processed = min(run.total, run.processed + size)
    run.deleted_count += int(outcome.deleted_count)
    run.skipped_count += int(outcome.skipped_count)
    run.skipped_locked.extend(int(pid) for pid in outcome.skipped_locked)
    run.skipped_restored += len(outcome.skipped_restored)
    run.ghosts_kept += int(outcome.ghosts_kept)
    run.ghosts_cascaded += int(outcome.ghosts_cascaded)
    run.purged_ids.extend(int(pid) for pid in outcome.purged_ids)
    run.updated_at = _now()


def _absorb_restore(run: ScrapheapRun, restored: list[int], size: int) -> None:
    run.processed = min(run.total, run.processed + size)
    run.restored_count += len(restored)
    run.segment_pictures += len(restored)
    run.updated_at = _now()


def _segment_for_slice(run: ScrapheapRun, newest: Optional[str], size: int) -> str:
    """Return the batch id this slice records into, starting a segment if due.

    The current segment is kept while nothing has been recorded into it yet, or
    while this slice still fits under :data:`SEGMENT_PICTURE_CAP` AND the
    segment is still the newest applied undoable batch in History (*newest*,
    read on the slice's own session). Checking room before the slice, rather
    than the count after the last one, keeps a segment at or under the cap.
    """
    current = run.segment_batch_ids[-1] if run.segment_batch_ids else None
    if current is not None:
        if run.segment_pictures == 0:
            return current
        if run.segment_pictures + size <= SEGMENT_PICTURE_CAP and newest == current:
            return current
    segment = operation_log_service.new_batch_id()
    run.segment_batch_ids.append(segment)
    run.segment_pictures = 0
    run.updated_at = _now()
    return segment


__all__ = [
    "ACTIVE_STATUSES",
    "ERROR_ACTIVE",
    "ERROR_NOT_FOUND",
    "ERROR_SERVER_STOPPED",
    "ERROR_UNAVAILABLE",
    "ERROR_UNEXPECTED",
    "HISTORY_KEPT",
    "KINDS",
    "KIND_PURGE",
    "KIND_RESTORE",
    "SEGMENT_PICTURE_CAP",
    "STATUS_CANCELLED",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "ScrapheapRun",
    "ScrapheapRunError",
    "ScrapheapRunRegistry",
    "begin_run",
    "cancel_requested",
    "cancel_run",
    "close_run",
    "finish_run",
    "get_run",
    "latest_run",
    "purge_slice",
    "restore_slice",
    "scrapheap_ids_in_session",
    "start_run",
]
