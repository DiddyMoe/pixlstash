"""The durable record of a background Scrapheap run.

A run (:mod:`pixlstash.services.scrapheap_run_service`) is worked from memory,
where the task folds each slice in under a lock, and written through to this
table at every state change: when it is reserved, when the task takes it, after
each slice, on cancel and when it ends. So a restart neither loses a run nor
leaves one ``running`` with no task to own it: start-up closes every row still
active as ``interrupted`` and reads the last few ended rows back, so
``GET /pictures/scrapheap/runs/latest`` still names the run that was in flight.
Nothing is resumed: a purge's consent was a single-use token of the process
that stopped, so finishing the job is the user's new request.

The selection itself (up to 200,000 ids) is not stored. An ended run does not
need it, and an interrupted one is never resumed.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String
from sqlmodel import Field, SQLModel, UTCDateTime


class ScrapheapRunRecord(SQLModel, table=True):
    """One Scrapheap run as it last stood: the request, its counts and outcome.

    The service's in-memory ``ScrapheapRun`` is the live copy; this row is
    rewritten from it whole, so the two carry the same fields under the same
    names. Named ``…Record`` so it does not shadow that class.

    Attributes:
        id: The run's id, a UUID hex string.
        request_id: The idempotency key, the client's or one the server
            minted. Unique: the service keeps the ids of the runs it holds
            distinct and prunes a row past its window before a new row can
            reuse the id.
        request_fingerprint: A hash of the parameters a replay of
            ``request_id`` must repeat (kind, selection, include_protected,
            target), so a reused id with other parameters is refused across a
            restart too.
        kind: ``purge``, ``restore`` or ``undo``.
        status: ``pending``, ``running`` or an end state.
        target_run_id: For an undo, the restore run it undoes.
        include_protected: A purge's choice to destroy protected originals.
        total / processed: Pictures the run set out to handle, and handled.
        cancel_requested: The user asked it to stop.
        deleted_count / skipped_count / skipped_locked_count /
            skipped_restored / arrived_since / ghosts_kept / ghosts_cascaded:
            A purge's counts. The locked ids themselves are not stored.
        deleted_before: A purge's consent bound, the confirming preview's time.
        restored_count: Pictures a restore brought back.
        undone_count: Pictures an undo put back in the Scrapheap.
        snapshots_with_deleted: JSON list naming the snapshots that still hold
            a finished purge's pictures, or null.
        segment_batch_ids: JSON list of a restore's History batches, oldest
            first (an undo's: its target's).
        segment_pictures: Pictures recorded into the newest segment, so a
            segment the run opened but recorded nothing into can be dropped.
        actor / source / origin_client_id: Who asked, for History and events.
        slice_max_pictures / slice_budget_s: The slice bounds the run kept to.
        error_code / error: Why it ended early, as the run view shows them.
    """

    __tablename__ = "scrapheaprun"

    id: str = Field(sa_column=Column("id", String, primary_key=True))
    request_id: Optional[str] = Field(
        default=None,
        sa_column=Column("request_id", String, nullable=True, unique=True, index=True),
    )
    request_fingerprint: Optional[str] = Field(
        default=None, sa_column=Column("request_fingerprint", String, nullable=True)
    )
    kind: str = Field(sa_column=Column("kind", String, nullable=False))
    status: str = Field(sa_column=Column("status", String, nullable=False, index=True))
    target_run_id: Optional[str] = Field(
        default=None, sa_column=Column("target_run_id", String, nullable=True)
    )
    include_protected: bool = Field(default=False)
    total: int = Field(default=0)
    processed: int = Field(default=0)
    cancel_requested: bool = Field(default=False)
    deleted_count: int = Field(default=0)
    skipped_count: int = Field(default=0)
    skipped_locked_count: int = Field(default=0)
    skipped_restored: int = Field(default=0)
    deleted_before: Optional[datetime] = Field(
        default=None, sa_column=Column("deleted_before", UTCDateTime(), nullable=True)
    )
    arrived_since: int = Field(default=0)
    ghosts_kept: int = Field(default=0)
    ghosts_cascaded: int = Field(default=0)
    restored_count: int = Field(default=0)
    undone_count: int = Field(default=0)
    snapshots_with_deleted: Optional[str] = Field(
        default=None,
        sa_column=Column("snapshots_with_deleted", String, nullable=True),
    )
    segment_batch_ids: str = Field(
        default="[]", sa_column=Column("segment_batch_ids", String, nullable=False)
    )
    segment_pictures: int = Field(default=0)
    actor: Optional[str] = Field(
        default=None, sa_column=Column("actor", String, nullable=True)
    )
    source: str = Field(
        default="external", sa_column=Column("source", String, nullable=False)
    )
    origin_client_id: Optional[str] = Field(
        default=None, sa_column=Column("origin_client_id", String, nullable=True)
    )
    slice_max_pictures: int = Field(default=0)
    slice_budget_s: float = Field(default=0.0)
    error_code: Optional[str] = Field(
        default=None, sa_column=Column("error_code", String, nullable=True)
    )
    error: Optional[str] = Field(
        default=None, sa_column=Column("error", String, nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column("created_at", UTCDateTime(), nullable=False),
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column("started_at", UTCDateTime(), nullable=True)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column("updated_at", UTCDateTime(), nullable=False),
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column("finished_at", UTCDateTime(), nullable=True)
    )


__all__ = ["ScrapheapRunRecord"]
