"""Add ``dedupscan.phase``: which tier a running duplicate scan is in.

A scan runs up to three tiers, and each reports progress on its own scale
(pictures for the exact tier, candidate buckets for the near tier, none for the
embedding tier). Without the tier on the row the banner could not tell which
scale its counters were on. Null for every existing row: none is running
across an upgrade, and a scan that resumes sets it again when it starts.

Schema-only and additive.

Revision ID: 0151_add_dedup_scan_phase
Revises: 0150_add_dedup_auto_stack_run
Create Date: 2026-11-24
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0151_add_dedup_scan_phase"
down_revision: Union[str, None] = "0150_add_dedup_auto_stack_run"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

__all__ = ["revision", "down_revision", "branch_labels", "depends_on"]

TABLE = "dedupscan"
COLUMN = "phase"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(TABLE):
        return
    existing_cols = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN not in existing_cols:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(TABLE):
        return
    existing_cols = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN in existing_cols:
        with op.batch_alter_table(TABLE) as batch:
            batch.drop_column(COLUMN)
