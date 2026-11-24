"""Add ``dedupautostackrun``: background auto-stack runs and their progress.

Auto-stack used to stack the whole exact tier inside one request, which on a
large library outlived the client's timeout by an hour. A run is now a row the
request writes and a background task works through, so progress, cancel and
revert have a durable place to live.

Schema-only and additive.

Revision ID: 0150_add_dedup_auto_stack_run
Revises: 0149_add_description_and_prediction_time_indexes
Create Date: 2026-11-23
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0150_add_dedup_auto_stack_run"
down_revision: Union[str, None] = "0149_add_description_and_prediction_time_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

__all__ = ["revision", "down_revision", "branch_labels", "depends_on"]

TABLE = "dedupautostackrun"
_REQUEST_ID_INDEX = "ix_dedupautostackrun_request_id"
_STATUS_INDEX = "ix_dedupautostackrun_status"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("request_id", sa.String(), nullable=False),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("target_run_id", sa.Integer(), nullable=True),
            sa.Column("scope_type", sa.String(), nullable=False),
            sa.Column("scope_id", sa.String(), nullable=True),
            sa.Column("group_limit", sa.Integer(), nullable=True),
            sa.Column("total", sa.Integer(), nullable=False),
            sa.Column("processed", sa.Integer(), nullable=False),
            sa.Column("applied", sa.Integer(), nullable=False),
            sa.Column("superseded", sa.Integer(), nullable=False),
            sa.Column("blocked", sa.Integer(), nullable=False),
            sa.Column("failed", sa.Integer(), nullable=False),
            sa.Column("pictures", sa.Integer(), nullable=False),
            sa.Column("failures", sa.String(), nullable=False),
            sa.Column("segment_batch_ids", sa.String(), nullable=False),
            sa.Column("segment_groups", sa.Integer(), nullable=False),
            sa.Column("cancel_requested", sa.Boolean(), nullable=False),
            sa.Column("actor", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=True),
            sa.Column("origin_client_id", sa.String(), nullable=True),
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
