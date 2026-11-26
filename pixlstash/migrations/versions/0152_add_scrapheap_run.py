"""Add ``scrapheaprun``: the durable record of background Scrapheap runs.

Emptying or restoring the Scrapheap is a background run, which until now lived
only in memory: a restart forgot it, so a client polling it got 404 and could
not tell what had committed, and nothing recorded a restore run's History
segments for an undo after the restart. Each run is now written through to a
row. Start-up closes a row left active as ``interrupted``; nothing resumes.

Schema-only and additive.

Revision ID: 0152_add_scrapheap_run
Revises: 0151_add_dedup_scan_phase
Create Date: 2026-11-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0152_add_scrapheap_run"
down_revision: Union[str, None] = "0151_add_dedup_scan_phase"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

__all__ = ["revision", "down_revision", "branch_labels", "depends_on"]

TABLE = "scrapheaprun"
_REQUEST_ID_INDEX = "ix_scrapheaprun_request_id"
_STATUS_INDEX = "ix_scrapheaprun_status"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("request_id", sa.String(), nullable=True),
            sa.Column("request_fingerprint", sa.String(), nullable=True),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("target_run_id", sa.String(), nullable=True),
            sa.Column("include_protected", sa.Boolean(), nullable=False),
            sa.Column("total", sa.Integer(), nullable=False),
            sa.Column("processed", sa.Integer(), nullable=False),
            sa.Column("cancel_requested", sa.Boolean(), nullable=False),
            sa.Column("deleted_count", sa.Integer(), nullable=False),
            sa.Column("skipped_count", sa.Integer(), nullable=False),
            sa.Column("skipped_locked_count", sa.Integer(), nullable=False),
            sa.Column("skipped_restored", sa.Integer(), nullable=False),
            sa.Column("deleted_before", sa.DateTime(), nullable=True),
            sa.Column("arrived_since", sa.Integer(), nullable=False),
            sa.Column("ghosts_kept", sa.Integer(), nullable=False),
            sa.Column("ghosts_cascaded", sa.Integer(), nullable=False),
            sa.Column("restored_count", sa.Integer(), nullable=False),
            sa.Column("undone_count", sa.Integer(), nullable=False),
            sa.Column("snapshots_with_deleted", sa.String(), nullable=True),
            sa.Column("segment_batch_ids", sa.String(), nullable=False),
            sa.Column("segment_pictures", sa.Integer(), nullable=False),
            sa.Column("actor", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("origin_client_id", sa.String(), nullable=True),
            sa.Column("slice_max_pictures", sa.Integer(), nullable=False),
            sa.Column("slice_budget_s", sa.Float(), nullable=False),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("error", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    # Re-inspected: the inspector above cached the table list before the CREATE.
    # Both indexes are declared on the model too, so a fresh database already
    # has them from the baseline; guarded by name so both paths converge.
    inspector = sa.inspect(bind)
    existing = {idx["name"] for idx in inspector.get_indexes(TABLE)}
    if _REQUEST_ID_INDEX not in existing:
        op.create_index(_REQUEST_ID_INDEX, TABLE, ["request_id"], unique=True)
    if _STATUS_INDEX not in existing:
        op.create_index(_STATUS_INDEX, TABLE, ["status"])


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        op.drop_table(TABLE)
