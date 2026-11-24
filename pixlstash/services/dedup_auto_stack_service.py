"""Background auto-stack runs: the request, its progress, cancel and revert.

Auto-stack used to stack the whole exact tier inside one request and one DB
writer callback. Stacking a group costs tens of milliseconds, so on a library
with a hundred thousand exact groups the request outlived the client's timeout
by about an hour, held the single SQLite writer the whole time, and left the
user with "did not answer" and no way to see progress, stop it or find out what
it had done.

A run is now a :class:`~pixlstash.db_models.dedup.DedupAutoStackRun` row (the
asynchronous request-reply pattern: the request writes the row, answers 202 with
its address, and the client polls it). A background task
(:class:`~pixlstash.tasks.dedup_auto_stack_task.DedupAutoStackTask`) works
through it in short LOW-priority writer slices, so every other write still gets
the writer between them, and the row is the progress readout.

**History in segments.** A run records into bounded batches ("segments"), each
its own operation-log batch id. A new segment starts when the next slice could
take the current one past :data:`SEGMENT_GROUP_CAP` groups, and whenever some
other undoable operation has landed on top of it since the last slice: a batch
must stay contiguous at the top of History to be undone
(``operation_log_service._enforce_latest_undo_unit``), and appending to a buried
batch would make both it and the operation above it un-undoable. Ctrl+Z
therefore undoes one segment (a few seconds of work), and the run's own revert
undoes all of them, newest first, one segment per slice. If an unrelated edit
sits between two segments, the revert stops there with ``blocked`` rather than
undo around it.

**One run at a time.** Apply and revert runs share the rule; a second request
gets 409 with the active run. A request may carry a client ``request_id``: the
same id with the same parameters returns the run it already created, so a retry
after a timeout cannot start a second run, and the same id with different
parameters is refused (Google AIP-155, IETF Idempotency-Key draft).

**A restart interrupts.** A run is not resumed after the server stops; it is
marked ``interrupted`` at start-up and the user decides whether to run again.
Everything it committed stays committed and revertable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlmodel import Session, select

from pixlstash.db_models.dedup import (
    AUTO_STACK_ACTIVE_STATUSES,
    AUTO_STACK_BLOCKED,
    AUTO_STACK_CANCELLED,
    AUTO_STACK_FAILED,
    AUTO_STACK_INTERRUPTED,
    AUTO_STACK_KIND_APPLY,
    AUTO_STACK_KIND_REVERT,
    AUTO_STACK_PENDING,
    AUTO_STACK_RUNNING,
    DedupAutoStackRun,
)
from pixlstash.db_models.operation import STATUS_APPLIED, Operation
from pixlstash.pixl_logging import get_logger
from pixlstash.services import operation_log_service
from pixlstash.services.dedup_tier_service import (
    DedupScope,
    ScopeType,
    prune_stale_groups_in_session,
)
from pixlstash.services.dedup_verdict_service import (
    BULK_REASON_FAILED,
    AutoStackSliceError,
    auto_stack_groups_in_session,
    new_batch_id,
)
from pixlstash.services.operation_log_service import OperationLogError
from pixlstash.utils.service.smart_score_invalidation import (
    InteractiveRescoreRegistry,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pixlstash.vault import Vault

logger = get_logger(__name__)

SEGMENT_GROUP_CAP = 250
"""Groups per History segment. Undoing 250 stacked groups measured 1.3-2.3 s on
a CPU-only box, so one Ctrl+Z, and one revert slice, stays a few seconds."""

FAILURES_KEPT = 50
"""Per-group failures kept on the run row. The counters stay exact; the list is
for showing the user a sample of why, not an audit log (the log has every one)."""

REQUEST_ID_MAX_LENGTH = 128

ERROR_ACTIVE = "auto_stack_run_active"
ERROR_REQUEST_ID_REUSED = "request_id_reused"
ERROR_NOT_FOUND = "auto_stack_run_not_found"
ERROR_NOT_AN_APPLY_RUN = "not_an_apply_run"
ERROR_NOTHING_TO_REVERT = "nothing_to_revert"
ERROR_HISTORY_MOVED_ON = "history_moved_on"
ERROR_SERVER_RESTARTED = "server_restarted"
ERROR_UNEXPECTED = "unexpected_error"


class AutoStackRunError(Exception):
    """A run request the service refuses, with a stable code for the client.

    The route maps :attr:`code` to an HTTP status; the service names no status
    so the rule lives in one place.

    Attributes:
        code: One of the ``ERROR_*`` codes above.
        run: The run the refusal is about (the active one for
            :data:`ERROR_ACTIVE`, the original for
            :data:`ERROR_REQUEST_ID_REUSED`), serialised, or ``None``.
    """

    def __init__(self, code: str, message: str, run: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.run = run


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _segments(run: DedupAutoStackRun) -> list[str]:
    return list(json.loads(run.segment_batch_ids or "[]"))


def _failures(run: DedupAutoStackRun) -> list[dict[str, Any]]:
    return list(json.loads(run.failures or "[]"))


def _scope(run: DedupAutoStackRun) -> DedupScope:
    return DedupScope(scope_type=ScopeType(run.scope_type), scope_id=run.scope_id)


def _has_applied_operations(session: Session, batch_ids: list[str]) -> bool:
    """Whether any of *batch_ids* still has an applied, undoable operation."""
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


def serialize_run(session: Session, run: DedupAutoStackRun) -> dict[str, Any]:
    """Shape a run for the API.

    For an apply run, ``total`` / ``processed`` count planned groups, and
    ``applied`` / ``superseded`` / ``blocked`` / ``failed`` split the processed
    ones by outcome. For a revert run, ``total`` / ``processed`` count History
    segments (the unit one slice undoes), ``applied`` counts the stacked groups
    unstacked so far and ``superseded`` the segments that had nothing left to
    undo. ``pictures`` is pictures stacked, or restored, so far.

    ``can_revert`` is true for a finished apply run with a segment that still
    has something to undo.
    """
    segments = _segments(run)
    active = run.status in AUTO_STACK_ACTIVE_STATUSES
    can_revert = (
        run.kind == AUTO_STACK_KIND_APPLY
        and not active
        and bool(segments)
        and _has_applied_operations(session, segments)
    )
    scope = _scope(run)
    return {
        "id": int(run.id),
        "request_id": run.request_id,
        "kind": run.kind,
        "status": run.status,
        "active": active,
        "target_run_id": run.target_run_id,
        "scope": scope.as_dict(),
        "limit": run.group_limit,
        "total": int(run.total),
        "processed": int(run.processed),
        "applied": int(run.applied),
        "superseded": int(run.superseded),
        "blocked": int(run.blocked),
        "failed": int(run.failed),
        "pictures": int(run.pictures),
        "failures": _failures(run),
        "segment_batch_ids": segments,
        "cancel_requested": bool(run.cancel_requested),
        "can_revert": can_revert,
        "error_code": run.error_code,
        "error": run.error,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "updated_at": _iso(run.updated_at),
        "finished_at": _iso(run.finished_at),
    }


def _active_run(session: Session) -> Optional[DedupAutoStackRun]:
    return session.exec(
        select(DedupAutoStackRun)
        .where(DedupAutoStackRun.status.in_(AUTO_STACK_ACTIVE_STATUSES))
        .order_by(DedupAutoStackRun.id.asc())
        .limit(1)
    ).first()


def _get_or_raise(session: Session, run_id: int) -> DedupAutoStackRun:
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        raise AutoStackRunError(ERROR_NOT_FOUND, f"No auto-stack run {run_id}")
    return run


def _replay(
    session: Session, request_id: str, params: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Return the run a repeated ``request_id`` created, or ``None`` if new.

    Raises:
        AutoStackRunError: :data:`ERROR_REQUEST_ID_REUSED` when the id was used
            for a request with different parameters.
    """
    existing = session.exec(
        select(DedupAutoStackRun).where(DedupAutoStackRun.request_id == request_id)
    ).first()
    if existing is None:
        return None
    recorded = {
        "kind": existing.kind,
        "scope_type": existing.scope_type,
        "scope_id": existing.scope_id,
        "group_limit": existing.group_limit,
        "target_run_id": existing.target_run_id,
    }
    if recorded != params:
        logger.info(
            "[auto-stack] request_id %s reused with different parameters "
            "(recorded %s, sent %s); refused",
            request_id,
            recorded,
            params,
        )
        raise AutoStackRunError(
            ERROR_REQUEST_ID_REUSED,
            "This request id already started a different auto-stack run.",
            serialize_run(session, existing),
        )
    return serialize_run(session, existing)


def _insert_run(
    session: Session,
    params: dict[str, Any],
    request_id: Optional[str],
    actor: Optional[str],
    source: Optional[str],
    origin_client_id: Optional[str],
    total: int = 0,
    segment_batch_ids: Optional[list[str]] = None,
) -> tuple[dict[str, Any], bool]:
    """Insert a pending run after the replay and one-active checks."""
    if request_id:
        replayed = _replay(session, request_id, params)
        if replayed is not None:
            return replayed, False
    active = _active_run(session)
    if active is not None:
        raise AutoStackRunError(
            ERROR_ACTIVE,
            "An auto-stack run is already in progress.",
            serialize_run(session, active),
        )
    now = _now()
    run = DedupAutoStackRun(
        request_id=request_id or f"srv-{uuid.uuid4().hex}",
        status=AUTO_STACK_PENDING,
        total=total,
        segment_batch_ids=json.dumps(segment_batch_ids or []),
        actor=actor,
        source=source,
        origin_client_id=origin_client_id,
        created_at=now,
        updated_at=now,
        **params,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    logger.info(
        "[auto-stack] run %s requested: %s (request_id=%s, actor=%s)",
        run.id,
        params,
        run.request_id,
        actor,
    )
    return serialize_run(session, run), True


def create_apply_run_in_session(
    session: Session,
    scope: Optional[DedupScope] = None,
    limit: Optional[int] = None,
    request_id: Optional[str] = None,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Request a run that stacks the scope's exact groups.

    Args:
        session: A writer session.
        scope: The scope; the whole vault when omitted.
        limit: Cap the number of groups the run plans.
        request_id: The client's idempotency key, if it sent one.
        actor: Who asked, from ``request_context``.
        source: WS-envelope source, likewise.
        origin_client_id: WS-envelope per-tab origin, likewise.

    Returns:
        The serialised run and whether this call created it (``False`` for a
        replayed ``request_id``).

    Raises:
        AutoStackRunError: :data:`ERROR_ACTIVE` or
            :data:`ERROR_REQUEST_ID_REUSED`.
    """
    scope = scope or DedupScope()
    params = {
        "kind": AUTO_STACK_KIND_APPLY,
        "scope_type": scope.scope_type.value,
        "scope_id": scope.scope_id,
        "group_limit": int(limit) if limit is not None else None,
        "target_run_id": None,
    }
    return _insert_run(session, params, request_id, actor, source, origin_client_id)


def create_revert_run_in_session(
    session: Session,
    target_run_id: int,
    request_id: Optional[str] = None,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Request a run that undoes everything an apply run stacked.

    The revert copies the target's segment list, so it works through exactly
    the batches the run recorded. Reverting a run again is allowed: segments
    with nothing left to undo are skipped, which is how a revert that stopped
    at an unrelated edit is finished once that edit has been undone.

    Raises:
        AutoStackRunError: :data:`ERROR_NOT_FOUND`,
            :data:`ERROR_NOT_AN_APPLY_RUN`, :data:`ERROR_NOTHING_TO_REVERT`,
            :data:`ERROR_ACTIVE` or :data:`ERROR_REQUEST_ID_REUSED`.
    """
    params = {
        "kind": AUTO_STACK_KIND_REVERT,
        "scope_type": ScopeType.GLOBAL.value,
        "scope_id": None,
        "group_limit": None,
        "target_run_id": int(target_run_id),
    }
    if request_id:
        replayed = _replay(session, request_id, params)
        if replayed is not None:
            return replayed, False
    target = _get_or_raise(session, target_run_id)
    if target.kind != AUTO_STACK_KIND_APPLY:
        raise AutoStackRunError(
            ERROR_NOT_AN_APPLY_RUN,
            f"Run {target_run_id} is a revert; only an auto-stack run can be reverted.",
        )
    if target.status in AUTO_STACK_ACTIVE_STATUSES:
        raise AutoStackRunError(
            ERROR_ACTIVE,
            "That auto-stack run is still in progress; stop it before reverting.",
            serialize_run(session, target),
        )
    segments = _segments(target)
    if not _has_applied_operations(session, segments):
        raise AutoStackRunError(
            ERROR_NOTHING_TO_REVERT,
            f"Run {target_run_id} has nothing left to undo.",
            serialize_run(session, target),
        )
    return _insert_run(
        session,
        params,
        request_id,
        actor,
        source,
        origin_client_id,
        total=len(segments),
        segment_batch_ids=segments,
    )


def get_run_in_session(session: Session, run_id: int) -> dict[str, Any]:
    """Return one run.

    Raises:
        AutoStackRunError: :data:`ERROR_NOT_FOUND`.
    """
    return serialize_run(session, _get_or_raise(session, run_id))


def latest_run_in_session(session: Session) -> Optional[dict[str, Any]]:
    """Return the newest run of either kind, or ``None`` before the first.

    This is what a client reads when the Duplicates screen opens, so a run
    started in another tab, or before a reload, is picked up again.
    """
    run = session.exec(
        select(DedupAutoStackRun).order_by(DedupAutoStackRun.id.desc()).limit(1)
    ).first()
    return serialize_run(session, run) if run is not None else None


def cancel_run_in_session(session: Session, run_id: int) -> dict[str, Any]:
    """Ask a run to stop.

    A pending run is cancelled at once. A running one is flagged and stops at
    its next slice boundary, keeping what it has committed. A finished run is
    returned unchanged, so a repeated cancel is harmless.

    Raises:
        AutoStackRunError: :data:`ERROR_NOT_FOUND`.
    """
    run = _get_or_raise(session, run_id)
    now = _now()
    if run.status == AUTO_STACK_PENDING:
        run.status = AUTO_STACK_CANCELLED
        run.cancel_requested = True
        run.finished_at = now
    elif run.status == AUTO_STACK_RUNNING:
        run.cancel_requested = True
    else:
        return serialize_run(session, run)
    run.updated_at = now
    session.add(run)
    session.commit()
    logger.info("[auto-stack] run %s: cancel requested (%s)", run.id, run.status)
    return serialize_run(session, run)


def find_pending_run_id(session: Session) -> Optional[int]:
    """Return the oldest pending run's id, for the finder."""
    row = session.exec(
        select(DedupAutoStackRun.id)
        .where(DedupAutoStackRun.status == AUTO_STACK_PENDING)
        .order_by(DedupAutoStackRun.id.asc())
        .limit(1)
    ).first()
    return int(row) if row is not None else None


def interrupt_orphaned_runs_in_session(session: Session) -> int:
    """Mark every run left pending or running by a previous process interrupted.

    Called once at start-up, before the planner starts: no task of this process
    can own such a run yet. A run is not resumed; what it committed stays.
    """
    rows = session.exec(
        select(DedupAutoStackRun).where(
            DedupAutoStackRun.status.in_(AUTO_STACK_ACTIVE_STATUSES)
        )
    ).all()
    now = _now()
    for run in rows:
        run.status = AUTO_STACK_INTERRUPTED
        run.error_code = ERROR_SERVER_RESTARTED
        run.error = "The server stopped before the run finished."
        run.finished_at = now
        run.updated_at = now
        session.add(run)
    if rows:
        session.commit()
        logger.warning(
            "[auto-stack] marked %d run(s) interrupted at start-up: %s",
            len(rows),
            [int(run.id) for run in rows],
        )
    return len(rows)


# --- Task slices --------------------------------------------------------------
# Each runs as one short callback on the DB writer. They take ids and plain data
# and return plain data: no Session or ORM row crosses a slice boundary.


def start_run_in_session(session: Session, run_id: int) -> Optional[dict[str, Any]]:
    """Mark a pending run running and return what the task needs to work it.

    Returns:
        ``None`` when the run is no longer pending (cancelled before it
        started, or already taken), otherwise its kind, scope, limit, segment
        list and origin.
    """
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None or run.status != AUTO_STACK_PENDING:
        return None
    now = _now()
    run.status = AUTO_STACK_RUNNING
    run.started_at = now
    run.updated_at = now
    session.add(run)
    session.commit()
    return {
        "kind": run.kind,
        "scope_type": run.scope_type,
        "scope_id": run.scope_id,
        "limit": run.group_limit,
        "segment_batch_ids": _segments(run),
        "origin_client_id": run.origin_client_id,
        "source": run.source or "external",
    }


def record_plan_in_session(session: Session, run_id: int, total: int) -> None:
    """Store how many groups an apply run planned."""
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        raise LookupError(f"auto-stack run {run_id} no longer exists")
    run.total = int(total)
    run.updated_at = _now()
    session.add(run)
    session.commit()


def _segment_for_slice(
    session: Session, run: DedupAutoStackRun, slice_groups: int
) -> str:
    """Return the batch id this slice records into, starting a segment if due.

    The current segment is kept while it is empty, or while this slice's
    *slice_groups* still fit under the cap AND it is still the newest applied
    undoable unit in History. Checking the room before the slice, rather than
    the count after the last one, is what keeps a segment at or under
    :data:`SEGMENT_GROUP_CAP`; a slice only ever adds to one segment. Anything
    else that recorded an undoable operation since the last slice sits on top of
    the current segment, and appending would split the batch around that
    operation, so a new segment starts. A new segment is committed before any
    group is stacked into it, so a revert finds it even if the process dies
    inside the slice.
    """
    segments = _segments(run)
    current = segments[-1] if segments else None
    if current is not None:
        if run.segment_groups == 0:
            return current
        if run.segment_groups + slice_groups <= SEGMENT_GROUP_CAP:
            newest = session.exec(
                select(Operation.batch_id)
                .where(
                    Operation.status == STATUS_APPLIED,
                    Operation.undoable.is_(True),
                )
                .order_by(Operation.id.desc())
                .limit(1)
            ).first()
            if newest == current:
                return current
    segment = new_batch_id()
    segments.append(segment)
    run.segment_batch_ids = json.dumps(segments)
    run.segment_groups = 0
    run.updated_at = _now()
    session.add(run)
    session.commit()
    return segment


def _absorb(
    run: DedupAutoStackRun, report: dict[str, Any], extra_failures: list[dict]
) -> None:
    """Add one slice report's counts to the run row (not committed)."""
    run.applied += int(report["groups"])
    run.superseded += int(report["superseded"])
    run.blocked += int(report["blocked"])
    run.failed += int(report["failed"]) + len(extra_failures)
    run.pictures += int(report["pictures"])
    run.segment_groups += int(report["groups"])
    failures = _failures(run)
    room = FAILURES_KEPT - len(failures)
    if room > 0:
        failures.extend((list(report["failures"]) + extra_failures)[:room])
        run.failures = json.dumps(failures, default=str)
    run.updated_at = _now()


def apply_slice_in_session(
    session: Session, run_id: int, group_ids: list[int]
) -> dict[str, Any]:
    """Stack one slice of planned groups for a running apply run.

    Returns:
        ``status`` (``running``, or ``cancelled`` when a cancel was requested
        before the slice began, or ``failed``), the run's ``processed`` /
        ``total``, and ``event_picture_ids`` for the task's coalesced
        ``pictures_changed`` announcement.
    """
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        raise LookupError(f"auto-stack run {run_id} no longer exists")
    if run.cancel_requested:
        return {"status": AUTO_STACK_CANCELLED, "event_picture_ids": []}
    segment = _segment_for_slice(session, run, len(group_ids))
    scope = _scope(run)
    actor, source, origin = run.actor, run.source or "external", run.origin_client_id
    try:
        report = auto_stack_groups_in_session(
            session, group_ids, scope, segment, actor, source, origin
        )
    except AutoStackSliceError as exc:
        # The failing group was rolled back; the groups before it committed.
        run = session.get(DedupAutoStackRun, int(run_id))
        cause = exc.__cause__ or exc
        failure = {
            "signature": exc.signature,
            "outcome": BULK_REASON_FAILED,
            "status_code": None,
            "error": f"{type(cause).__name__}: {cause}",
        }
        report = exc.report
        handled = (
            int(report["groups"])
            + int(report["superseded"])
            + int(report["blocked"])
            + int(report["failed"])
            + 1
        )
        _absorb(run, report, [failure])
        run.processed += handled
        run.status = AUTO_STACK_FAILED
        run.error_code = ERROR_UNEXPECTED
        run.error = failure["error"][:2000]
        run.finished_at = _now()
        session.add(run)
        session.commit()
        logger.error(
            "[auto-stack] run %s failed at group %s after %d of %d group(s): %s",
            run_id,
            exc.signature,
            run.processed,
            run.total,
            run.error,
        )
        return {
            "status": AUTO_STACK_FAILED,
            "processed": int(run.processed),
            "total": int(run.total),
            "event_picture_ids": list(report["event_picture_ids"]),
        }
    run = session.get(DedupAutoStackRun, int(run_id))
    _absorb(run, report, [])
    run.processed += len(group_ids)
    session.add(run)
    session.commit()
    return {
        "status": AUTO_STACK_RUNNING,
        "processed": int(run.processed),
        "total": int(run.total),
        "event_picture_ids": list(report["event_picture_ids"]),
    }


def revert_segment_in_session(
    session: Session,
    run_id: int,
    batch_id: str,
    registry: Optional[InteractiveRescoreRegistry] = None,
    image_root: Optional[str] = None,
) -> dict[str, Any]:
    """Undo one segment of the target run for a running revert run.

    Returns:
        ``status``: ``running`` after undoing or skipping the segment,
        ``cancelled`` when a cancel was requested before it began, or
        ``blocked`` when History has moved on above the segment. When the
        segment was undone, ``restore`` holds what
        :func:`operation_log_service.announce_restore` needs.
    """
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        raise LookupError(f"auto-stack run {run_id} no longer exists")
    if run.cancel_requested:
        return {"status": AUTO_STACK_CANCELLED}
    origin = run.origin_client_id
    if not _has_applied_operations(session, [batch_id]):
        run.processed += 1
        run.superseded += 1
        run.updated_at = _now()
        session.add(run)
        session.commit()
        return {"status": AUTO_STACK_RUNNING}
    try:
        members, touched, facets, lifecycle = (
            operation_log_service.undo_batch_in_session(
                session, batch_id, registry, origin, image_root
            )
        )
    except OperationLogError as exc:
        session.rollback()
        run = session.get(DedupAutoStackRun, int(run_id))
        run.status = AUTO_STACK_BLOCKED
        run.error_code = ERROR_HISTORY_MOVED_ON
        run.error = str(exc)[:2000]
        run.finished_at = _now()
        run.updated_at = run.finished_at
        session.add(run)
        session.commit()
        logger.info(
            "[auto-stack] revert run %s stopped at segment %s: %s",
            run_id,
            batch_id,
            exc,
        )
        return {"status": AUTO_STACK_BLOCKED}
    run = session.get(DedupAutoStackRun, int(run_id))
    run.processed += 1
    run.applied += len(members)
    run.pictures += len(touched)
    run.updated_at = _now()
    session.add(run)
    session.commit()
    return {
        "status": AUTO_STACK_RUNNING,
        "restore": {
            "members": members,
            "touched": touched,
            "facets": facets,
            "lifecycle": lifecycle,
            "origin_client_id": origin,
        },
    }


def finish_run_in_session(session: Session, run_id: int, status: str) -> dict:
    """Close a run with *status* unless something already closed it.

    An apply run prunes the groups its stacks left with fewer than two live
    members first, as every verdict path does.
    """
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        raise LookupError(f"auto-stack run {run_id} no longer exists")
    if run.kind == AUTO_STACK_KIND_APPLY and run.applied:
        prune_stale_groups_in_session(session)
        run = session.get(DedupAutoStackRun, int(run_id))
    if run.status in AUTO_STACK_ACTIVE_STATUSES:
        now = _now()
        run.status = status
        run.finished_at = now
        run.updated_at = now
        session.add(run)
        session.commit()
    logger.info(
        "[auto-stack] %s run %s %s: %d of %d processed, %d applied, %d superseded, "
        "%d blocked, %d failed",
        run.kind,
        run.id,
        run.status,
        run.processed,
        run.total,
        run.applied,
        run.superseded,
        run.blocked,
        run.failed,
    )
    return {"status": run.status}


def close_run_in_session(
    session: Session, run_id: int, status: str, error: Optional[str] = None
) -> None:
    """Mark a still-active run *status* (interrupted or failed) with *error*.

    Used from the task's own failure paths at IMMEDIATE priority; a run some
    slice already closed is left as it is.
    """
    run = session.get(DedupAutoStackRun, int(run_id))
    if run is None:
        logger.warning(
            "[auto-stack] cannot mark run %s %s: the row is gone", run_id, status
        )
        return
    if run.status not in AUTO_STACK_ACTIVE_STATUSES:
        return
    now = _now()
    run.status = status
    if status == AUTO_STACK_FAILED:
        run.error_code = ERROR_UNEXPECTED
    elif status == AUTO_STACK_INTERRUPTED:
        run.error_code = ERROR_SERVER_RESTARTED
    run.error = (error or "")[:2000] or None
    run.finished_at = now
    run.updated_at = now
    session.add(run)
    session.commit()


# --- Vault wrappers -----------------------------------------------------------


def start_apply_run(
    vault: "Vault",
    scope: Optional[DedupScope] = None,
    limit: Optional[int] = None,
    request_id: Optional[str] = None,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Record an apply run and wake the planner so it starts promptly."""
    run, created = vault.db.run_task(
        create_apply_run_in_session,
        scope,
        limit,
        request_id,
        actor,
        source,
        origin_client_id,
    )
    if created:
        vault.wake()
    return run, created


def start_revert_run(
    vault: "Vault",
    target_run_id: int,
    request_id: Optional[str] = None,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    origin_client_id: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Record a revert run and wake the planner so it starts promptly."""
    run, created = vault.db.run_task(
        create_revert_run_in_session,
        target_run_id,
        request_id,
        actor,
        source,
        origin_client_id,
    )
    if created:
        vault.wake()
    return run, created


def get_run(vault: "Vault", run_id: int) -> dict[str, Any]:
    """Read one run off the writer queue."""
    return vault.db.run_immediate_read_task(get_run_in_session, run_id)


def latest_run(vault: "Vault") -> Optional[dict[str, Any]]:
    """Read the newest run off the writer queue."""
    return vault.db.run_immediate_read_task(latest_run_in_session)


def cancel_run(vault: "Vault", run_id: int) -> dict[str, Any]:
    """Request a cancel. Ahead of the run's own LOW slices on the writer."""
    return vault.db.run_task(cancel_run_in_session, run_id)


__all__ = [
    "ERROR_ACTIVE",
    "ERROR_HISTORY_MOVED_ON",
    "ERROR_NOTHING_TO_REVERT",
    "ERROR_NOT_AN_APPLY_RUN",
    "ERROR_NOT_FOUND",
    "ERROR_REQUEST_ID_REUSED",
    "ERROR_SERVER_RESTARTED",
    "ERROR_UNEXPECTED",
    "FAILURES_KEPT",
    "REQUEST_ID_MAX_LENGTH",
    "SEGMENT_GROUP_CAP",
    "AutoStackRunError",
    "apply_slice_in_session",
    "cancel_run",
    "cancel_run_in_session",
    "close_run_in_session",
    "create_apply_run_in_session",
    "create_revert_run_in_session",
    "find_pending_run_id",
    "finish_run_in_session",
    "get_run",
    "get_run_in_session",
    "interrupt_orphaned_runs_in_session",
    "latest_run",
    "latest_run_in_session",
    "record_plan_in_session",
    "revert_segment_in_session",
    "serialize_run",
    "start_apply_run",
    "start_revert_run",
    "start_run_in_session",
]
