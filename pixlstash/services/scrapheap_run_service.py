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

**Undoing a restore run is a run too.** One segment is one Ctrl+Z, but undoing
a whole "restore all" by hand is one press per segment, and undoing it as one
transaction would hold the writer for the whole run. An undo run
(:data:`KIND_UNDO`, :func:`start_undo_run`) takes the target's segments newest
first, one LOW writer slice each, through
:func:`operation_log_service.undo_batch_in_session`, as the Auto-stack revert
run does (:mod:`pixlstash.services.dedup_auto_stack_service`). History must
stay contiguous: when something unrelated sits above the next segment, the run
stops ``blocked`` (:data:`ERROR_HISTORY_MOVED_ON`) with what it undid kept, and
a second undo of the same target finishes once that edit is undone.

**A start is idempotent.** Every run carries a ``request_id``, the client's
(one per gesture) or one the server mints. The same id with the same
parameters returns the run it already started, whatever its state, without
starting another or spending a purge's confirm token again; the same id with
different parameters is refused (:data:`ERROR_REQUEST_ID_REUSED`). The
parameters are compared as a hash (:func:`request_fingerprint`), so a 200,000-id
selection costs 64 characters on the run and its row. The replay window is the
runs kept (:data:`HISTORY_KEPT` ended ones plus the active one).

**One run at a time.** Every kind shares the rule; a second request gets 409
with the active run.

**Worked in memory, written through to a row.** The live run is the vault's
:class:`ScrapheapRunRegistry` entry, which request threads read and the task
folds each slice into under one lock. Every state change is then written
through to its :class:`~pixlstash.db_models.ScrapheapRunRecord` row: when the
run is reserved (IMMEDIATE, before the 202, so a run the client was told about
is on disk), when the task takes it, after each slice, on cancel and when it
ends (LOW, beside the slices). The write reads the run under the lock when the
writer runs it, not when it was asked for, so the row is never older than the
newest write and two writes cannot land out of order.

A restart or a library switch interrupts a run at the next slice boundary:
everything committed stays committed, and the pictures it had not reached are
still in the Scrapheap, so starting again finishes the job. At start-up
:func:`recover_runs` closes every row still active as ``interrupted`` (nothing
resumes: a purge's consent was a single-use token of the process that
stopped) and reads the last :data:`HISTORY_KEPT` ended runs back, so the client
polling the run learns how it ended instead of getting 404. The row is written
after a slice, not inside it, so after a hard crash its counts can trail what
committed by at most one slice.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from fastapi import HTTPException
from sqlalchemy import delete, func, or_
from sqlmodel import Session, select

from pixlstash.database import DatabaseUnavailableError, DBPriority
from pixlstash.db_models import Picture, ScrapheapRunRecord
from pixlstash.db_models.operation import STATUS_APPLIED, Operation
from pixlstash.pixl_logging import get_logger
from pixlstash.services import operation_log_service, scrapheap_service
from pixlstash.services.operation_log_service import (
    PURGED_STATUS_CODE,
    OperationLogError,
)
from pixlstash.utils.service.scope_table import scope_id_subquery

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pixlstash.vault import Vault

logger = get_logger(__name__)

_T = TypeVar("_T")

KIND_PURGE = "purge"
KIND_RESTORE = "restore"
KIND_UNDO = "undo"
KINDS = (KIND_PURGE, KIND_RESTORE, KIND_UNDO)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_BLOCKED = "blocked"
"""An undo run stopped because History moved on above its next segment."""
ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_RUNNING})

SEGMENT_PICTURE_CAP = 1000
"""Pictures per History segment of a restore run, so one Ctrl+Z, and one slice
of an undo run, is one bounded writer transaction. Undoing a 5,000-picture
segment measured 4.2 s of writer time in a 50,000-picture library (1.9 s for
3,000), and every other write waits behind it. At 1,000, an undo slice in a
20,000-picture library held the writer 0.41 s on average and 0.57 s at most
(``scripts/bench_scrapheap_run.py --segment-cap 1000``)."""

SLICE_MAX_PICTURES = 250
"""The most pictures one slice takes, whatever the measured cost. It is also
what Stop can overshoot by: a cancel takes effect at the next slice boundary.
On a 20,000-picture bench the budget alone sized slices to a median of about
260 (about 1 ms of writer time a picture), so 250 bounds a Stop without
shrinking the typical slice; at 500 single slices reached 411 pictures and
0.64 s of writer time (``scripts/bench_scrapheap_run.py``)."""

SLICE_BUDGET_S = 0.25
"""Writer time one slice aims to hold the writer for. The next slice's size is
this budget over the measured writer time per picture, so a library where a
picture costs more (big stacks, many hub ghosts, a slow disk) gets smaller
slices instead of longer ones."""

SLICE_FIRST_PICTURES = 50
"""The first slice's size, before anything has been measured."""

SLICE_MIN_PICTURES = 10
"""The fewest pictures a slice takes, so a slow library still makes progress
rather than paying a slice's fixed overhead per picture."""

HISTORY_KEPT = 20
"""Ended runs kept, in the registry and as rows, so a client that polls a run
after it ended (another tab, a slow poll, a restart) still finds it."""

ERROR_ACTIVE = "scrapheap_run_active"
ERROR_NOT_FOUND = "scrapheap_run_not_found"
ERROR_UNAVAILABLE = "scrapheap_runs_unavailable"
ERROR_REQUEST_ID_REUSED = "request_id_reused"
ERROR_NOT_A_RESTORE_RUN = "not_a_restore_run"
ERROR_NOTHING_TO_UNDO = "nothing_to_undo"
ERROR_SERVER_STOPPED = "server_stopped"
ERROR_LIBRARY_RESTORED = "library_restored"
"""A run's ``error_code`` when a full restore brought its row back active: the
snapshot was taken while it ran, and no task of this process owns it."""
ERROR_UNEXPECTED = "unexpected_error"
ERROR_HISTORY_MOVED_ON = "history_moved_on"
"""An undo run's ``error_code`` when an unrelated History entry sits above the
segment it reached."""
ERROR_UNDO_REFUSED = "undo_refused"
"""An undo run's ``error_code`` when the undo itself refused the segment: a
locked picture set holds one of its pictures, or one was deleted forever."""

REQUEST_ID_MAX_LENGTH = 128
"""The longest ``request_id`` a client may send, as Auto-stack runs allow."""

SERVER_REQUEST_ID_PREFIX = "srv-"
"""Marks a ``request_id`` the server minted because the client sent none."""

SERVER_STOPPED_MESSAGE = "The server stopped before the run finished."
"""What a run the server stopped under records as its ``error``, whether the
task saw the stop or start-up found the row still active."""

LIBRARY_RESTORED_MESSAGE = (
    "The library was restored to a snapshot taken before this run finished."
)
"""What a run a full restore brought back active records as its ``error``."""

UNEXPECTED_ERROR_MESSAGE = (
    "The server hit an unexpected error; the server log has the details."
)
"""What a failed run records as its ``error``. The exception text is logged, not
stored: it can carry file paths, SQL or library content, and the run view is
served to every owner session (OWASP Logging Cheat Sheet, "Data to exclude")."""

HISTORY_MOVED_ON_MESSAGE = (
    "Undo stopped: a later change in History must be undone first. What this "
    "undo already put back in the Scrapheap stays there; undo that change, "
    "then undo this restore again to finish."
)
"""A blocked undo's ``error`` when the blocking entry cannot be read back; the
usual message names it (:func:`_history_moved_on_message`)."""

HISTORY_LABEL_MAX_LENGTH = 120
"""The longest History summary a blocked undo's ``error`` quotes."""

UNDO_REFUSED_MESSAGE = (
    "Undo stopped: a picture of this restore is now in a locked picture set or "
    "was deleted forever, so this part cannot be put back in the Scrapheap. "
    "What this undo already put back stays there."
)
"""A blocked undo's ``error`` for :data:`ERROR_UNDO_REFUSED`. Fixed text: the
refusal's own detail is logged, not stored."""


class ScrapheapRunError(Exception):
    """A run request the service refuses, with a stable code for the client.

    The route maps :attr:`code` to an HTTP status; the service names no status
    so the rule lives in one place.

    Attributes:
        code: :data:`ERROR_ACTIVE`, :data:`ERROR_NOT_FOUND`,
            :data:`ERROR_UNAVAILABLE`, :data:`ERROR_REQUEST_ID_REUSED`,
            :data:`ERROR_NOT_A_RESTORE_RUN` or :data:`ERROR_NOTHING_TO_UNDO`.
        run: The run the refusal is about (the active one for
            :data:`ERROR_ACTIVE`, the original for
            :data:`ERROR_REQUEST_ID_REUSED`, the target for the two undo
            refusals), as :meth:`ScrapheapRun.view` shapes it, or ``None``.
    """

    def __init__(self, code: str, message: str, run: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.run = run


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def request_fingerprint(
    kind: str,
    picture_ids: Optional[Iterable[int]] = None,
    include_protected: bool = False,
    target_run_id: Optional[str] = None,
) -> str:
    """Hash the parameters a replayed ``request_id`` must repeat.

    The kind, a purge's ``include_protected``, the selection as a sorted set
    (or the whole Scrapheap) and an undo's target. A purge's confirm token and
    its ``deleted_before`` are left out: a retry after a lost answer carries
    the token the first attempt spent.
    """
    selection = None if picture_ids is None else sorted({int(p) for p in picture_ids})
    body = json.dumps(
        [
            kind,
            bool(include_protected) and kind == KIND_PURGE,
            selection,
            target_run_id,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _mint_request_id() -> str:
    return f"{SERVER_REQUEST_ID_PREFIX}{uuid.uuid4().hex}"


@dataclass
class ScrapheapRun:
    """One background Scrapheap run. Mutated only under the registry's lock.

    Attributes:
        id: The run's id, a UUID string.
        kind: :data:`KIND_PURGE`, :data:`KIND_RESTORE` or :data:`KIND_UNDO`.
        request_id: The idempotency key it was started with, the client's or
            one the server minted (:data:`SERVER_REQUEST_ID_PREFIX`).
        request_fingerprint: :func:`request_fingerprint` of its parameters,
            which a replay of ``request_id`` must match.
        target_run_id: The restore run an undo run works on; ``None`` for the
            other kinds.
        include_protected: A purge's choice to destroy protected
            reference-folder originals too. Always ``False`` for a restore.
        status: One of the ``STATUS_*`` values.
        total: Pictures the run set out to handle, fixed when it starts; for
            an undo run, the History segments.
        processed: Pictures (segments, for an undo run) handled so far,
            whatever the outcome.
        cancel_requested: The user asked it to stop; it does at the next slice.
        deleted_count: Pictures destroyed (purge).
        skipped_count: Protected originals left in the Scrapheap (purge).
        skipped_locked: Ids a locked picture set froze (purge). Not written
            to the row, so a run read back after a restart has none.
        skipped_locked_count: How many there were, which the row keeps.
        skipped_restored: Pictures that had left the Scrapheap before their
            slice reached them (purge).
        deleted_before: A purge's consent bound, the confirming preview's
            time; ``None`` for a restore.
        arrived_since: Pictures in the purge's scope binned after the preview,
            which the run therefore left in the Scrapheap; counted when the
            selection is fixed, shown once the run has ended.
        ghosts_kept: Picture ghosts the hub kept (purge).
        ghosts_cascaded: Picture ghosts the hub destroyed (purge).
        restored_count: Pictures brought back (restore).
        undone_count: Pictures put back in the Scrapheap (undo).
        snapshots_with_deleted: Snapshots still holding metadata for the
            destroyed pictures, set when a purge ends.
        segment_batch_ids: The History batches a restore recorded into, or
            the target's batches an undo works through.
        segment_pictures: Pictures recorded into the newest segment.
        purged_ids: Every id the purge destroyed, for the snapshot check;
            cleared when the run ends.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.
        picture_ids: The resolved selection, in the order slices take it;
            cleared when the run ends.
        error_code: Why it ended early, when it did.
        error: The detail of a failure, for the log and the client.
        slice_max_pictures: The most pictures a slice of this run takes, so
            the most a Stop can overshoot by (:data:`SLICE_MAX_PICTURES` when
            the run was created).
        slice_budget_s: The writer time a slice of this run aims for
            (:data:`SLICE_BUDGET_S` when the run was created).
        created_at: When it was requested.
        started_at: When the task took it.
        updated_at: When anything about it last changed.
        finished_at: When it ended.
    """

    kind: str
    include_protected: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = field(default_factory=_mint_request_id)
    request_fingerprint: Optional[str] = None
    target_run_id: Optional[str] = None
    status: str = STATUS_PENDING
    total: int = 0
    processed: int = 0
    cancel_requested: bool = False
    deleted_count: int = 0
    skipped_count: int = 0
    skipped_locked: list[int] = field(default_factory=list)
    skipped_locked_count: int = 0
    skipped_restored: int = 0
    deleted_before: Optional[datetime] = None
    arrived_since: int = 0
    ghosts_kept: int = 0
    ghosts_cascaded: int = 0
    restored_count: int = 0
    undone_count: int = 0
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
    # Read when the run is created, not when the module is, so the module
    # values are the single setting.
    slice_max_pictures: int = field(default_factory=lambda: SLICE_MAX_PICTURES)
    slice_budget_s: float = field(default_factory=lambda: SLICE_BUDGET_S)
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
        ``can_undo`` needs History, which the registry does not hold, so it is
        ``False`` here and :func:`_with_can_undo` fills it in.
        """
        active = self.active
        return {
            "id": self.id,
            "request_id": self.request_id,
            "kind": self.kind,
            "status": self.status,
            "active": active,
            "target_run_id": self.target_run_id,
            "include_protected": self.include_protected,
            "total": self.total,
            "processed": self.processed,
            "cancel_requested": self.cancel_requested,
            "deleted_count": self.deleted_count,
            "skipped_count": self.skipped_count,
            "skipped_locked_count": self.skipped_locked_count,
            "skipped_locked": [] if active else sorted(self.skipped_locked),
            "skipped_restored": self.skipped_restored,
            "arrived_since": (
                self.arrived_since if self.kind == KIND_PURGE and not active else None
            ),
            "ghosts_kept": self.ghosts_kept,
            "ghosts_cascaded": self.ghosts_cascaded,
            "restored_count": self.restored_count,
            "undone_count": self.undone_count,
            "can_undo": False,
            "snapshots_with_deleted": None if active else self.snapshots_with_deleted,
            "segment_batch_ids": list(self.segment_batch_ids),
            "error_code": self.error_code,
            "error": self.error,
            "slice_max_pictures": self.slice_max_pictures,
            "slice_budget_s": self.slice_budget_s,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "updated_at": _iso(self.updated_at),
            "finished_at": _iso(self.finished_at),
        }


class ScrapheapRunRegistry:
    """The vault's Scrapheap runs: the active one and the last few that ended.

    Request threads read and cancel while the task thread folds slice results
    in, so every access takes one lock, and every read returns a
    :meth:`ScrapheapRun.view` copy rather than the live run. A condition on
    that lock is notified whenever a run ends, for :meth:`wait_ended`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ended = threading.Condition(self._lock)
        self._runs: OrderedDict[str, ScrapheapRun] = OrderedDict()

    def reserve(
        self,
        kind: str,
        include_protected: bool,
        actor: Optional[str],
        source: Optional[str],
        origin_client_id: Optional[str],
        request_id: Optional[str] = None,
        fingerprint: Optional[str] = None,
        target_run_id: Optional[str] = None,
        segment_batch_ids: Optional[list[str]] = None,
    ) -> tuple[str, bool]:
        """Record a pending run, unless *request_id* already started one.

        The replay check and the reservation share the lock, so two requests
        with the same id cannot both start a run.

        Args:
            kind: The run's kind.
            include_protected: A purge's choice to destroy protected originals.
            actor: Who asked.
            source: WS-envelope source.
            origin_client_id: WS-envelope per-tab origin.
            request_id: The client's idempotency key; one is minted if absent.
            fingerprint: :func:`request_fingerprint` of the parameters.
            target_run_id: An undo's restore run.
            segment_batch_ids: An undo's work, its target's segments; the
                run's ``total`` is their count.

        Returns:
            The run's id, and whether this call created it (``False`` when
            *request_id* names a run already kept).

        Raises:
            ScrapheapRunError: :data:`ERROR_REQUEST_ID_REUSED`, carrying the
                original run, or :data:`ERROR_ACTIVE`, carrying the active one.
        """
        with self._lock:
            if request_id is not None:
                original = self._find_request_locked(request_id, fingerprint)
                if original is not None:
                    return original.id, False
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
                request_fingerprint=fingerprint,
                target_run_id=target_run_id,
            )
            if request_id is not None:
                run.request_id = request_id
            if segment_batch_ids is not None:
                run.segment_batch_ids = list(segment_batch_ids)
                run.total = len(run.segment_batch_ids)
            self._runs[run.id] = run
            self._trim_locked()
            return run.id, True

    def find_request(
        self, request_id: str, fingerprint: Optional[str]
    ) -> Optional[dict[str, Any]]:
        """Return the run *request_id* started, or ``None`` if none is kept.

        Raises:
            ScrapheapRunError: :data:`ERROR_REQUEST_ID_REUSED` when that run
                was started with other parameters, carrying it.
        """
        with self._lock:
            run = self._find_request_locked(request_id, fingerprint)
            return None if run is None else run.view()

    def check_idle(self) -> None:
        """Refuse while a run is active; :meth:`reserve` makes the same check.

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
                _release_ids(run)
                self._ended.notify_all()
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
                # that does not exist. An undo run's segments are its target's,
                # all real, and it records into none of them.
                if (
                    run.kind == KIND_RESTORE
                    and run.segment_batch_ids
                    and run.segment_pictures == 0
                ):
                    run.segment_batch_ids.pop()
                _release_ids(run)
                self._ended.notify_all()
            return run.view()

    def wait_ended(self, run_id: str, timeout: float) -> dict[str, Any]:
        """Return the run once it has ended, or as it is after *timeout* seconds.

        Raises:
            ScrapheapRunError: :data:`ERROR_NOT_FOUND`.
        """
        with self._ended:
            run = self._get_locked(run_id)
            self._ended.wait_for(lambda: not run.active, timeout)
            return run.view()

    def load(self, runs: Iterable[ScrapheapRun]) -> int:
        """Put ended runs read back from their rows ahead of the ones in memory.

        For start-up (:func:`recover_runs`): *runs* are older than anything
        this process started, oldest first. A run already registered is kept as
        it is, since the live copy is the newer one.

        Returns:
            How many runs were added.
        """
        with self._lock:
            loaded = [run for run in runs if run.id not in self._runs]
            merged: OrderedDict[str, ScrapheapRun] = OrderedDict(
                (run.id, run) for run in loaded
            )
            merged.update(self._runs)
            self._runs = merged
            self._trim_locked()
            return len(loaded)

    def reload(self, runs: Iterable[ScrapheapRun]) -> None:
        """Hold exactly *runs*, read back from a database that replaced the old.

        For a full restore (:func:`reload_runs`): every run held was recorded
        in the file the swap replaced, so none is kept. The registry object
        stays the same, so a caller that took ``vault.scrapheap_runs`` before
        the restore reads the restored runs too, not a stale copy. A
        :meth:`wait_ended` caller is woken; it waits on the run it already
        found.
        """
        with self._lock:
            self._runs = OrderedDict((run.id, run) for run in runs)
            self._trim_locked()
            self._ended.notify_all()

    def _active_locked(self) -> Optional[ScrapheapRun]:
        for run in self._runs.values():
            if run.active:
                return run
        return None

    def _find_request_locked(
        self, request_id: str, fingerprint: Optional[str]
    ) -> Optional[ScrapheapRun]:
        # Newest first: past the kept window an id may start a second run, and
        # a row read back at start-up may share it with a newer one.
        for run in reversed(self._runs.values()):
            if run.request_id != request_id:
                continue
            if run.request_fingerprint != fingerprint:
                raise ScrapheapRunError(
                    ERROR_REQUEST_ID_REUSED,
                    "This request_id already started a Scrapheap run with "
                    "different parameters. Send a new request_id for a new run.",
                    run.view(),
                )
            return run
        return None

    def _get_locked(self, run_id: str) -> ScrapheapRun:
        run = self._runs.get(str(run_id))
        if run is None:
            # The id is the caller's path text: it goes to the log, not the
            # refusal the route answers as its message.
            logger.info("Scrapheap run %r not found (404).", run_id)
            raise ScrapheapRunError(ERROR_NOT_FOUND, "No Scrapheap run has that id.")
        return run

    def _trim_locked(self) -> None:
        ended = [run_id for run_id, run in self._runs.items() if not run.active]
        for run_id in ended[: max(0, len(ended) - HISTORY_KEPT)]:
            del self._runs[run_id]


def scrapheap_ids_in_session(
    session: Session,
    picture_ids: Optional[list[int]],
    deleted_before: Optional[datetime] = None,
) -> list[int]:
    """The ids of *picture_ids* still in the Scrapheap, oldest first.

    ``None`` means the whole Scrapheap. A temp-table scope, so a selection of
    any size binds no per-id parameters. ``deleted_before`` (a purge's consent
    bound) keeps only pictures binned by then; see
    :func:`scrapheap_service.binned_by`.
    """
    query = select(Picture.id).where(Picture.deleted.is_(True))
    if picture_ids is not None:
        scope = scope_id_subquery(
            session, picture_ids, name="_pixlstash_scrapheap_run_ids"
        )
        query = query.where(Picture.id.in_(scope))
    if deleted_before is not None:
        query = query.where(scrapheap_service.binned_by(deleted_before))
    rows = session.exec(query.order_by(Picture.id)).all()
    return [int(pid) for pid in rows if pid is not None]


def arrived_since_in_session(
    session: Session, picture_ids: Optional[list[int]], deleted_before: datetime
) -> int:
    """How many of *picture_ids* in the Scrapheap were binned after the bound.

    The pictures :func:`scrapheap_ids_in_session` leaves out for
    ``deleted_before``, so a purge can say how many it did not touch.
    """
    query = select(func.count(Picture.id)).where(
        Picture.deleted.is_(True),
        Picture.deleted_at > deleted_before,
    )
    if picture_ids is not None:
        scope = scope_id_subquery(
            session, picture_ids, name="_pixlstash_scrapheap_run_arrived_ids"
        )
        query = query.where(Picture.id.in_(scope))
    return int(session.exec(query).one())


def _selection_in_session(
    session: Session,
    picture_ids: Optional[list[int]],
    deleted_before: Optional[datetime],
) -> tuple[list[int], int]:
    """A run's selection and, for a purge, the pictures its bound left out."""
    ids = scrapheap_ids_in_session(session, picture_ids, deleted_before)
    if deleted_before is None:
        return ids, 0
    return ids, arrived_since_in_session(session, picture_ids, deleted_before)


def check_can_start(vault: "Vault") -> None:
    """Refuse a run that could not start now, before the caller spends anything.

    A purge's caller redeems a single-use confirm token; asking this first means
    the common refusals (a run already going, a server with no task runner)
    leave the token unspent. It is a pre-check, not a reservation: a second
    request can still win the race, and :func:`start_run` refuses that one
    atomically, so a caller that spent something restores it on a
    :class:`ScrapheapRunError` from :func:`start_run` as well.

    Raises:
        ScrapheapRunError: :data:`ERROR_ACTIVE`, carrying the active run, or
            :data:`ERROR_UNAVAILABLE` when this server runs no background tasks.
        TaskAdmissionRefused: A library restore has paused admission, so the
            task would be refused (the server answers 409).
    """
    vault.scrapheap_runs.check_idle()
    vault.refuse_while_admission_paused()
    if not vault.runs_background_work():
        logger.warning(
            "[scrapheap-run] run not started: this server runs no background "
            "tasks (disable_background_workers)"
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "This server runs no background work, so the Scrapheap cannot be "
            "emptied or restored here. Nothing was started.",
        )


def find_replay(
    vault: "Vault",
    request_id: Optional[str],
    kind: str,
    picture_ids: Optional[list[int]] = None,
    include_protected: bool = False,
    target_run_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Return the run *request_id* already started with these parameters.

    The route asks this before anything else, so a retry of a purge whose
    answer was lost gets its run back without redeeming the spent confirm
    token again, and without the active-run refusal its own run would cause.
    ``None`` when the client sent no id or the id started nothing kept.

    Raises:
        ScrapheapRunError: :data:`ERROR_REQUEST_ID_REUSED`, carrying the
            original run.
    """
    if request_id is None:
        return None
    fingerprint = request_fingerprint(
        kind, picture_ids, include_protected, target_run_id
    )
    view = vault.scrapheap_runs.find_request(request_id, fingerprint)
    return None if view is None else _with_can_undo(vault, view)


def start_run(
    vault: "Vault",
    kind: str,
    picture_ids: Optional[list[int]] = None,
    include_protected: bool = False,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
    deleted_before: Optional[datetime] = None,
    request_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Reserve a purge or restore run, fix its selection and hand it to the
    task runner.

    A purge's consent check (the delete-preview ``confirm_token``) belongs to
    the caller and must already have passed: this function destroys nothing
    itself, but the run it starts does. The caller hands on that
    confirmation's ``deleted_before``, and the selection keeps only pictures
    binned by then.

    Args:
        vault: The vault whose Scrapheap it works on.
        kind: :data:`KIND_PURGE` or :data:`KIND_RESTORE`.
        picture_ids: The selection, or ``None`` for the whole Scrapheap.
        include_protected: A purge's choice to destroy protected originals.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.
        deleted_before: A purge's consent bound, the confirming preview's
            time (required for a purge, ignored for a restore).
        request_id: The client's idempotency key, or ``None`` to mint one.

    Returns:
        The run as the API shows it, and whether this call started it
        (``False`` for a replay of *request_id*).

    Raises:
        ScrapheapRunError: :data:`ERROR_REQUEST_ID_REUSED`,
            :data:`ERROR_ACTIVE`, or :data:`ERROR_UNAVAILABLE` when this
            server runs no background tasks or is stopping.
        ValueError: An unknown kind, an undo (see :func:`start_undo_run`), or
            a purge with no ``deleted_before``.
    """
    if kind not in (KIND_PURGE, KIND_RESTORE):
        raise ValueError(f"Unknown Scrapheap run kind {kind!r}")
    if kind == KIND_PURGE and deleted_before is None:
        raise ValueError("A purge run needs its confirmation's deleted_before")
    if kind != KIND_PURGE:
        deleted_before = None
    fingerprint = request_fingerprint(kind, picture_ids, include_protected)
    replay = find_replay(vault, request_id, kind, picture_ids, include_protected)
    if replay is not None:
        return replay, False
    check_can_start(vault)
    registry = vault.scrapheap_runs
    run_id, created = registry.reserve(
        kind,
        include_protected,
        actor,
        source,
        origin_client_id,
        request_id=request_id,
        fingerprint=fingerprint,
    )
    if not created:
        # The same request_id won the race since find_replay looked.
        return _with_can_undo(vault, registry.get(run_id)), False
    selected: list[int] = []

    def _prepare() -> None:
        ids, arrived = vault.db.run_immediate_read_task(
            _selection_in_session, picture_ids, deleted_before
        )
        selected.append(len(ids))
        registry.mutate(
            run_id,
            lambda run: _record_selection(run, ids, deleted_before, arrived),
        )

    _launch(vault, run_id, kind, _prepare)
    logger.info(
        "[scrapheap-run] %s: %s of %d picture(s) requested (include_protected=%s, "
        "actor=%s, request_id=%s)",
        run_id,
        kind,
        selected[0],
        include_protected,
        actor,
        request_id,
    )
    return _with_can_undo(vault, registry.get(run_id)), True


def start_undo_run(
    vault: "Vault",
    target_run_id: str,
    request_id: Optional[str] = None,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Start a run that undoes a restore run's History segments, newest first.

    Args:
        vault: The vault.
        target_run_id: The restore run to undo.
        request_id: The client's idempotency key, or ``None`` to mint one.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.

    Returns:
        The undo run as the API shows it, and whether this call started it.

    Raises:
        ScrapheapRunError: :data:`ERROR_REQUEST_ID_REUSED`,
            :data:`ERROR_NOT_FOUND`, :data:`ERROR_NOT_A_RESTORE_RUN` or
            :data:`ERROR_NOTHING_TO_UNDO` (both carrying the target),
            :data:`ERROR_ACTIVE`, or :data:`ERROR_UNAVAILABLE`.
    """
    target_run_id = str(target_run_id)
    fingerprint = request_fingerprint(KIND_UNDO, target_run_id=target_run_id)
    replay = find_replay(vault, request_id, KIND_UNDO, target_run_id=target_run_id)
    if replay is not None:
        return replay, False
    registry = vault.scrapheap_runs
    target = registry.get(target_run_id)
    if target["kind"] != KIND_RESTORE:
        raise ScrapheapRunError(
            ERROR_NOT_A_RESTORE_RUN,
            "Only a restore run can be undone. Pictures a purge deleted are "
            "gone for good.",
            target,
        )
    check_can_start(vault)
    segments = list(target["segment_batch_ids"])
    try:
        undoable = bool(segments) and vault.db.run_immediate_read_task(
            _has_applied_operations_in_session, segments
        )
    except DatabaseUnavailableError as exc:
        logger.warning(
            "[scrapheap-run] undo of %s not started: the database is closing (%s)",
            target_run_id,
            exc,
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "The server is stopping, so nothing was started.",
        ) from exc
    if not undoable:
        raise ScrapheapRunError(
            ERROR_NOTHING_TO_UNDO,
            "Nothing this restore brought back is left to undo: every part of "
            "it has been undone already.",
            target,
        )
    run_id, created = registry.reserve(
        KIND_UNDO,
        False,
        actor,
        source,
        origin_client_id,
        request_id=request_id,
        fingerprint=fingerprint,
        target_run_id=target_run_id,
        segment_batch_ids=segments,
    )
    if not created:
        return _with_can_undo(vault, registry.get(run_id)), False
    _launch(vault, run_id, KIND_UNDO, None)
    logger.info(
        "[scrapheap-run] %s: undo of restore run %s, %d segment(s) (actor=%s, "
        "request_id=%s)",
        run_id,
        target_run_id,
        len(segments),
        actor,
        request_id,
    )
    return _with_can_undo(vault, registry.get(run_id)), True


def _launch(
    vault: "Vault",
    run_id: str,
    kind: str,
    prepare: Optional[Callable[[], None]],
) -> None:
    """Prepare a reserved run, write its row and hand it to the task runner.

    The row is on disk before the 202, so a run the client was told about
    survives a restart; the same transaction drops ended rows past
    :data:`HISTORY_KEPT`. A run that fails to launch leaves the registry and
    the table.

    Raises:
        ScrapheapRunError: :data:`ERROR_UNAVAILABLE` when the database or the
            task runner is stopping.
    """
    # Imported here, not at the top: the task module imports this one for its
    # slices, and every task module is loaded through ``pixlstash.tasks``.
    from pixlstash.tasks.scrapheap_run_task import ScrapheapRunTask

    registry = vault.scrapheap_runs
    try:
        if prepare is not None:
            prepare()
        vault.db.run_task(
            _write_record_in_session,
            registry,
            run_id,
            True,
            priority=DBPriority.IMMEDIATE,
        )
    except DatabaseUnavailableError as exc:
        registry.discard(run_id)
        logger.warning(
            "[scrapheap-run] %s run not started: the database is closing (%s)",
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
    try:
        task_id = vault.submit_task(ScrapheapRunTask(vault, run_id))
    except BaseException:
        # Not a stop (``Vault.submit_task`` answers that with None): a worker
        # thread that cannot start, say. Not "stopping", but nothing started.
        registry.discard(run_id)
        _delete_record(vault, run_id)
        raise
    if task_id is None:
        # The runner has stopped since the caller's ``runs_background_work``
        # check; ``Vault.submit_task`` logged the refusal.
        registry.discard(run_id)
        _delete_record(vault, run_id)
        logger.warning(
            "[scrapheap-run] %s run %s not started: the task runner is stopping",
            kind,
            run_id,
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "The server is stopping, so nothing was started.",
        )


def _with_can_undo(vault: "Vault", view: dict[str, Any]) -> dict[str, Any]:
    """Fill in ``can_undo``: an ended restore with a segment left to undo.

    Asked of History on every read of such a run, since any Ctrl+Z, here or in
    another tab, can change the answer.

    Raises:
        ScrapheapRunError: :data:`ERROR_UNAVAILABLE` when the database is
            closing.
    """
    if view["kind"] != KIND_RESTORE or view["active"] or not view["segment_batch_ids"]:
        return view
    try:
        view["can_undo"] = vault.db.run_immediate_read_task(
            _has_applied_operations_in_session, list(view["segment_batch_ids"])
        )
    except DatabaseUnavailableError as exc:
        logger.warning(
            "[scrapheap-run] %s: can_undo not read, the database is closing (%s)",
            view["id"],
            exc,
        )
        raise ScrapheapRunError(
            ERROR_UNAVAILABLE,
            "The server is stopping, so the run cannot be read now.",
        ) from exc
    return view


def _has_applied_operations_in_session(session: Session, batch_ids: list[str]) -> bool:
    """Whether any of *batch_ids* still has an applied, undoable operation.

    As :func:`dedup_auto_stack_service._has_applied_operations`, in chunks of
    500 so a run of many segments stays under SQLite's bound-parameter limit.
    """
    for start in range(0, len(batch_ids), 500):
        chunk = batch_ids[start : start + 500]
        row = session.exec(
            select(Operation.id)
            .where(
                Operation.batch_id.in_(chunk),
                Operation.status == STATUS_APPLIED,
                Operation.undoable.is_(True),
            )
            .limit(1)
        ).first()
        if row is not None:
            return True
    return False


def get_run(vault: "Vault", run_id: str) -> dict[str, Any]:
    """Return one run.

    Raises:
        ScrapheapRunError: :data:`ERROR_NOT_FOUND`, or
            :data:`ERROR_UNAVAILABLE` when the database is closing.
    """
    return _with_can_undo(vault, vault.scrapheap_runs.get(run_id))


def latest_run(vault: "Vault") -> Optional[dict[str, Any]]:
    """Return the newest run, or ``None``.

    What a client reads when the Scrapheap opens, so a run started before a
    reload, or in another tab, is picked up again.
    """
    view = vault.scrapheap_runs.latest()
    return None if view is None else _with_can_undo(vault, view)


def wait_for_run(vault: "Vault", run_id: str, timeout: float) -> dict[str, Any]:
    """Wait up to *timeout* seconds for a run to end; see
    :meth:`ScrapheapRunRegistry.wait_ended`. For ``Prefer: wait``.

    Raises:
        ScrapheapRunError: :data:`ERROR_NOT_FOUND`.
    """
    return _with_can_undo(vault, vault.scrapheap_runs.wait_ended(run_id, timeout))


def cancel_run(vault: "Vault", run_id: str) -> dict[str, Any]:
    """Ask a run to stop; see :meth:`ScrapheapRunRegistry.cancel`.

    The request is written through at IMMEDIATE priority, so a pending run's
    cancellation is on disk before the reply.
    """
    view = vault.scrapheap_runs.cancel(run_id)
    _write_record(vault, run_id, DBPriority.IMMEDIATE)
    return _with_can_undo(vault, view)


def recover_runs(vault: "Vault") -> int:
    """Close the runs a previous process left active and read the last few back.

    Called once at start-up, before any task of this process can own a run.
    A row still ``pending`` or ``running`` is marked ``interrupted`` with
    :data:`ERROR_SERVER_STOPPED`; it is not resumed, and what it committed
    stays. The newest :data:`HISTORY_KEPT` ended rows are then loaded into the
    registry, so ``runs/latest`` and ``runs/{id}`` answer across the restart.

    Returns:
        How many runs were marked interrupted.
    """
    interrupted, runs = vault.db.run_task(
        _recover_in_session,
        ERROR_SERVER_STOPPED,
        SERVER_STOPPED_MESSAGE,
        priority=DBPriority.IMMEDIATE,
    )
    vault.scrapheap_runs.load(runs)
    return interrupted


def reload_runs(vault: "Vault") -> int:
    """After a full restore: close the restored active rows and reload runs.

    What :func:`recover_runs` does at start-up, except that the runs held in
    memory are dropped (:meth:`ScrapheapRunRegistry.reload`) rather than kept:
    they belong to the database the restore replaced, and that a restored
    row is closed with :data:`ERROR_LIBRARY_RESTORED`, not as a server stop.
    Called with every task drained, so no task of this process owns a run.

    Returns:
        How many runs were marked interrupted.
    """
    interrupted, runs = vault.db.run_task(
        _recover_in_session,
        ERROR_LIBRARY_RESTORED,
        LIBRARY_RESTORED_MESSAGE,
        priority=DBPriority.IMMEDIATE,
    )
    vault.scrapheap_runs.reload(runs)
    return interrupted


# --- The row ------------------------------------------------------------------
# Written from the live run, whole, at every state change. The ids a run works
# from are not stored: an ended run does not need them, and an interrupted one
# is never resumed.

_RECORD_FIELDS = (
    "id",
    "kind",
    "status",
    "include_protected",
    "total",
    "processed",
    "cancel_requested",
    "deleted_count",
    "skipped_count",
    "skipped_locked_count",
    "skipped_restored",
    "deleted_before",
    "arrived_since",
    "ghosts_kept",
    "ghosts_cascaded",
    "restored_count",
    "undone_count",
    "request_id",
    "request_fingerprint",
    "target_run_id",
    "segment_pictures",
    "actor",
    "source",
    "origin_client_id",
    "slice_max_pictures",
    "slice_budget_s",
    "error_code",
    "error",
    "created_at",
    "started_at",
    "updated_at",
    "finished_at",
)
"""The run fields stored as they are; the two lists are stored as JSON."""


def _record_values(run: ScrapheapRun) -> dict[str, Any]:
    values = {name: getattr(run, name) for name in _RECORD_FIELDS}
    values["segment_batch_ids"] = json.dumps(list(run.segment_batch_ids))
    values["snapshots_with_deleted"] = (
        None
        if run.snapshots_with_deleted is None
        else json.dumps(run.snapshots_with_deleted)
    )
    return values


def _run_from_record(row: ScrapheapRunRecord) -> ScrapheapRun:
    values = {name: getattr(row, name) for name in _RECORD_FIELDS}
    values["segment_batch_ids"] = list(json.loads(row.segment_batch_ids or "[]"))
    values["snapshots_with_deleted"] = (
        None
        if row.snapshots_with_deleted is None
        else json.loads(row.snapshots_with_deleted)
    )
    return ScrapheapRun(**values)


def _write_record(vault: "Vault", run_id: str, priority: DBPriority) -> None:
    """Write the run's row from the live run, waiting for the writer.

    A database that is closing is logged, not raised: the live run has already
    changed, and start-up closes a row left active.
    """
    try:
        vault.db.run_task(
            _write_record_in_session,
            vault.scrapheap_runs,
            run_id,
            False,
            priority=priority,
        )
    except DatabaseUnavailableError as exc:
        logger.warning(
            "[scrapheap-run] %s: its row was not updated, the database is "
            "closing (%s); start-up marks a row left active interrupted",
            run_id,
            exc,
        )


def _write_record_in_session(
    session: Session, registry: ScrapheapRunRegistry, run_id: str, prune: bool
) -> bool:
    """Upsert the run's row from the registry as it is now.

    Read under the registry's lock when the writer runs this, not when the
    write was asked for, so the row never goes back to an older state. With
    *prune*, ended rows past :data:`HISTORY_KEPT` go in the same transaction,
    first: the registry has already let those runs go, so a new run may carry
    a ``request_id`` one of them had, and the unique index must not see both.

    Returns:
        Whether a row was written (``False`` once the run left the registry).
    """
    try:
        values = registry.mutate(run_id, _record_values)
    except LookupError:
        logger.info(
            "[scrapheap-run] %s: row not written, the run is no longer registered",
            run_id,
        )
        return False
    if prune:
        _prune_in_session(session)
    row = session.get(ScrapheapRunRecord, run_id)
    if row is None:
        row = ScrapheapRunRecord(**values)
    else:
        for name, value in values.items():
            setattr(row, name, value)
    session.add(row)
    session.commit()
    return True


def _prune_in_session(session: Session) -> int:
    """Delete the ended rows past the newest :data:`HISTORY_KEPT`."""
    stale = session.exec(
        select(ScrapheapRunRecord.id)
        .where(ScrapheapRunRecord.status.not_in(ACTIVE_STATUSES))
        .order_by(ScrapheapRunRecord.created_at.desc(), ScrapheapRunRecord.id.desc())
        .offset(HISTORY_KEPT)
    ).all()
    if stale:
        session.exec(
            delete(ScrapheapRunRecord).where(ScrapheapRunRecord.id.in_(list(stale)))
        )
    return len(stale)


def _delete_record(vault: "Vault", run_id: str) -> None:
    """Remove the row of a run that never reached the task runner."""
    try:
        vault.db.run_task(
            _delete_record_in_session, run_id, priority=DBPriority.IMMEDIATE
        )
    except DatabaseUnavailableError as exc:
        logger.warning(
            "[scrapheap-run] %s: the row of a run that never started was not "
            "removed, the database is closing (%s); start-up marks it interrupted",
            run_id,
            exc,
        )


def _delete_record_in_session(session: Session, run_id: str) -> None:
    session.exec(delete(ScrapheapRunRecord).where(ScrapheapRunRecord.id == run_id))
    session.commit()


def _recover_in_session(
    session: Session, error_code: str, error: str
) -> tuple[int, list[ScrapheapRun]]:
    rows = session.exec(
        select(ScrapheapRunRecord).where(ScrapheapRunRecord.status.in_(ACTIVE_STATUSES))
    ).all()
    now = _now()
    for row in rows:
        row.status = STATUS_INTERRUPTED
        row.error_code = error_code
        row.error = error
        row.finished_at = now
        row.updated_at = now
        # As ScrapheapRunRegistry.finish: a segment the last slice opened but
        # recorded nothing into is not a History batch. Only a restore opens
        # segments; an undo's are its target's.
        segments = list(json.loads(row.segment_batch_ids or "[]"))
        if row.kind == KIND_RESTORE and segments and row.segment_pictures == 0:
            segments.pop()
            row.segment_batch_ids = json.dumps(segments)
        session.add(row)
    if rows:
        session.commit()
        logger.warning(
            "[scrapheap-run] marked %d run(s) interrupted (%s): %s",
            len(rows),
            error_code,
            [row.id for row in rows],
        )
    kept = session.exec(
        select(ScrapheapRunRecord)
        .where(ScrapheapRunRecord.status.not_in(ACTIVE_STATUSES))
        .order_by(ScrapheapRunRecord.created_at.desc(), ScrapheapRunRecord.id.desc())
        .limit(HISTORY_KEPT)
    ).all()
    return len(rows), [_run_from_record(row) for row in reversed(kept)]


# --- Task slices --------------------------------------------------------------
# Called from the task thread. Each slice is one existing Scrapheap path at LOW
# writer priority; the registry is updated after it, under its lock.


def begin_run(vault: "Vault", run_id: str) -> Optional[dict[str, Any]]:
    """Mark a pending run running and return what the task needs to work it.

    Returns:
        ``None`` when the run is no longer pending (cancelled before the task
        took it), otherwise its ``kind``, ``picture_ids``,
        ``segment_batch_ids`` (an undo's work), ``origin_client_id``,
        ``source``, ``slice_max_pictures`` and ``slice_budget_s``.
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
            "segment_batch_ids": list(run.segment_batch_ids),
            "origin_client_id": run.origin_client_id,
            "source": run.source,
            "slice_max_pictures": run.slice_max_pictures,
            "slice_budget_s": run.slice_budget_s,
        }

    params = vault.scrapheap_runs.mutate(run_id, _begin)
    if params is not None:
        _write_record(vault, run_id, DBPriority.LOW)
    return params


def cancel_requested(vault: "Vault", run_id: str) -> bool:
    """Whether the user has asked the run to stop."""
    return vault.scrapheap_runs.mutate(run_id, lambda run: run.cancel_requested)


def purge_slice(vault: "Vault", run_id: str, chunk: list[int]) -> list[int]:
    """Destroy one slice of a purge run's selection.

    Files are removed inline, on the task thread: this is already off the
    request, and the next slice must not start ahead of this one's reconcile.

    The slice applies the run's consent bound again, so a picture of the
    selection restored and binned anew while the run works is left alone.

    Returns:
        The ids this slice destroyed.
    """
    include_protected, deleted_before = vault.scrapheap_runs.mutate(
        run_id, lambda run: (run.include_protected, run.deleted_before)
    )
    outcome = scrapheap_service.purge_scrapheap_pictures(
        vault,
        list(chunk),
        include_protected,
        priority=DBPriority.LOW,
        deleted_before=deleted_before,
    )
    vault.scrapheap_runs.mutate(
        run_id, lambda run: _absorb_purge(run, outcome, len(chunk))
    )
    _write_record(vault, run_id, DBPriority.LOW)
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
    _write_record(vault, run_id, DBPriority.LOW)
    return list(restored)


def undo_slice(vault: "Vault", run_id: str, batch_id: str) -> dict[str, Any]:
    """Undo one History segment of an undo run's target, as one LOW write.

    Returns:
        ``outcome``: ``"undone"``, ``"skipped"`` (the segment has nothing left
        to undo, e.g. a Ctrl+Z took it already) or ``"blocked"``. An undone
        segment also carries ``removed`` (the pictures put back in the
        Scrapheap), ``updated`` (touched, not moved) and ``stack_siblings``
        (members whose stack count changed), for the task's events. A
        blocked one carries ``error_code`` and ``error`` for the run.
    """
    registry = vault.scrapheap_runs
    origin = registry.mutate(run_id, lambda run: run.origin_client_id)
    outcome = vault.db.run_task(
        _undo_segment_in_session,
        batch_id,
        vault.interactive_rescore_registry,
        origin,
        vault.image_root,
        priority=DBPriority.LOW,
    )
    if outcome["outcome"] == "blocked":
        logger.info(
            "[scrapheap-run] %s: undo stopped at segment %s (%s): %s",
            run_id,
            batch_id,
            outcome["error_code"],
            outcome["detail"],
        )
    else:
        registry.mutate(run_id, lambda run: _absorb_undo(run, outcome))
        _write_record(vault, run_id, DBPriority.LOW)
    return outcome


def _undo_segment_in_session(
    session: Session,
    batch_id: str,
    rescore_registry,
    origin_client_id: Optional[str],
    image_root: Optional[str],
) -> dict[str, Any]:
    """Undo one segment through the History path Ctrl+Z on it would take.

    :func:`operation_log_service.undo_batch_in_session` refuses a segment that
    is not contiguous at the top of History, and the undo refuses a picture a
    locked set holds (423) or one deleted forever (410). Either stops the run
    with its work so far kept; nothing of this segment is applied.
    """
    if not _has_applied_operations_in_session(session, [batch_id]):
        return {"outcome": "skipped"}
    try:
        _members, touched, _facets, lifecycle = (
            operation_log_service.undo_batch_in_session(
                session, batch_id, rescore_registry, origin_client_id, image_root
            )
        )
    except OperationLogError as exc:
        session.rollback()
        blocker = _history_blocker_in_session(session, batch_id)
        return {
            "outcome": "blocked",
            "error_code": ERROR_HISTORY_MOVED_ON,
            "error": _history_moved_on_message(blocker),
            "detail": str(exc),
        }
    except HTTPException as exc:
        if exc.status_code not in (423, PURGED_STATUS_CODE):
            raise
        session.rollback()
        return {
            "outcome": "blocked",
            "error_code": ERROR_UNDO_REFUSED,
            "error": UNDO_REFUSED_MESSAGE,
            "detail": f"HTTP {exc.status_code}: {exc.detail}",
        }
    scrapheaped = [int(pid) for pid in (lifecycle or {}).get("scrapheaped") or []]
    restored = [int(pid) for pid in (lifecycle or {}).get("restored") or []]
    moved = set(scrapheaped) | set(restored)
    return {
        "outcome": "undone",
        "removed": scrapheaped,
        "restored": restored,
        "updated": [int(pid) for pid in touched if int(pid) not in moved],
        "stack_siblings": [
            int(pid) for pid in (lifecycle or {}).get("stack_siblings") or []
        ],
    }


def _history_blocker_in_session(session: Session, batch_id: str) -> Optional[Operation]:
    """The History entry that must be undone before *batch_id* can be.

    The same rule as ``operation_log_service._enforce_latest_undo_unit``: the
    newest applied, undoable entry from the segment's oldest member up that
    belongs to another batch, or else the newest one in History.
    """
    oldest = session.exec(
        select(func.min(Operation.id)).where(
            Operation.batch_id == batch_id,
            Operation.status == STATUS_APPLIED,
            Operation.undoable.is_(True),
        )
    ).one()
    if oldest is not None:
        conflict = session.exec(
            select(Operation)
            .where(
                Operation.status == STATUS_APPLIED,
                Operation.undoable.is_(True),
                Operation.id >= int(oldest),
                or_(Operation.batch_id.is_(None), Operation.batch_id != batch_id),
            )
            .order_by(Operation.id.desc())
            .limit(1)
        ).first()
        if conflict is not None:
            return conflict
    latest = session.exec(
        select(Operation)
        .where(Operation.status == STATUS_APPLIED, Operation.undoable.is_(True))
        .order_by(Operation.id.desc())
        .limit(1)
    ).first()
    if latest is not None and latest.batch_id != batch_id:
        return latest
    return None


def _history_moved_on_message(blocker: Optional[Operation]) -> str:
    """The run's ``error`` for :data:`ERROR_HISTORY_MOVED_ON`, built from the
    blocking entry's own History summary, never from the exception."""
    if blocker is None:
        return HISTORY_MOVED_ON_MESSAGE
    label = (blocker.summary or blocker.op_type or "").strip()
    if len(label) > HISTORY_LABEL_MAX_LENGTH:
        label = label[: HISTORY_LABEL_MAX_LENGTH - 1].rstrip() + "…"
    return (
        f"Undo stopped: a later change in History (“{label}”, entry "
        f"{blocker.id}) must be undone first. What this undo already put back "
        "in the Scrapheap stays there; undo that change, then undo this "
        "restore again to finish."
    )


def finish_run(
    vault: "Vault",
    run_id: str,
    status: str,
    error_code: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Close a run the task worked to the end of, stopped on request, or that
    an undo found blocked (*error_code* and *error* then say why).

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
    view = registry.finish(run_id, status, error_code, error)
    _write_record(vault, run_id, DBPriority.LOW)
    logger.info(
        "[scrapheap-run] %s: %s %s, %d of %d processed, %d deleted, %d restored, "
        "%d undone, %d protected kept, %d locked kept, %d already out of the "
        "Scrapheap (error_code=%s)",
        run_id,
        view["kind"],
        view["status"],
        view["processed"],
        view["total"],
        view["deleted_count"],
        view["restored_count"],
        view["undone_count"],
        view["skipped_count"],
        view["skipped_locked_count"],
        view["skipped_restored"],
        view["error_code"],
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
        return
    # IMMEDIATE, not LOW: this runs while a runner stops or drains its queue,
    # and must not wait behind a backlog of LOW writes to let it.
    _write_record(vault, run_id, DBPriority.IMMEDIATE)


def _release_ids(run: ScrapheapRun) -> None:
    """Drop an ended run's id lists.

    They only serve the run while it works (the slices, and the snapshot check
    :func:`finish_run` makes before it closes the run). Kept on the
    :data:`HISTORY_KEPT` ended runs, they would pin that many selections of up
    to 200,000 ids each for the life of the process.
    """
    run.picture_ids = []
    run.purged_ids = []


def _record_selection(
    run: ScrapheapRun,
    ids: list[int],
    deleted_before: Optional[datetime],
    arrived: int,
) -> None:
    run.picture_ids = list(ids)
    run.total = len(ids)
    run.deleted_before = deleted_before
    run.arrived_since = int(arrived)
    run.updated_at = _now()


def _absorb_purge(
    run: ScrapheapRun, outcome: scrapheap_service.ScrapheapPurgeOutcome, size: int
) -> None:
    run.processed = min(run.total, run.processed + size)
    run.deleted_count += int(outcome.deleted_count)
    run.skipped_count += int(outcome.skipped_count)
    run.skipped_locked.extend(int(pid) for pid in outcome.skipped_locked)
    run.skipped_locked_count += len(outcome.skipped_locked)
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


def _absorb_undo(run: ScrapheapRun, outcome: dict[str, Any]) -> None:
    run.processed = min(run.total, run.processed + 1)
    run.undone_count += len(outcome.get("removed") or [])
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
    "ERROR_HISTORY_MOVED_ON",
    "ERROR_LIBRARY_RESTORED",
    "ERROR_NOTHING_TO_UNDO",
    "ERROR_NOT_A_RESTORE_RUN",
    "ERROR_NOT_FOUND",
    "ERROR_REQUEST_ID_REUSED",
    "ERROR_SERVER_STOPPED",
    "ERROR_UNAVAILABLE",
    "ERROR_UNDO_REFUSED",
    "ERROR_UNEXPECTED",
    "HISTORY_KEPT",
    "KINDS",
    "KIND_PURGE",
    "KIND_RESTORE",
    "KIND_UNDO",
    "LIBRARY_RESTORED_MESSAGE",
    "REQUEST_ID_MAX_LENGTH",
    "SEGMENT_PICTURE_CAP",
    "STATUS_BLOCKED",
    "STATUS_CANCELLED",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "SERVER_STOPPED_MESSAGE",
    "UNEXPECTED_ERROR_MESSAGE",
    "ScrapheapRun",
    "ScrapheapRunError",
    "ScrapheapRunRegistry",
    "begin_run",
    "cancel_requested",
    "cancel_run",
    "check_can_start",
    "close_run",
    "find_replay",
    "wait_for_run",
    "finish_run",
    "get_run",
    "latest_run",
    "purge_slice",
    "recover_runs",
    "reload_runs",
    "request_fingerprint",
    "restore_slice",
    "scrapheap_ids_in_session",
    "arrived_since_in_session",
    "start_run",
    "start_undo_run",
    "undo_slice",
]
