"""Add ``unprocessableimage``: the durable copy of the unprocessable-image registry.

The registry of pictures the background pipeline holds back (an undecodable
file, or a file on a location that is not mounted) lived only in memory, so a
restart forgot it and every such picture was decoded and failed once more. The
registry now writes each entry through to a row and reads the rows back on its
first use. A row is deleted with its picture.

Schema-only and additive.

Revision ID: 0153_add_unprocessable_image
Revises: 0152_add_scrapheap_run
Create Date: 2026-11-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0153_add_unprocessable_image"
down_revision: Union[str, None] = "0152_add_scrapheap_run"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

__all__ = ["revision", "down_revision", "branch_labels", "depends_on"]

TABLE = "unprocessableimage"


def upgrade() -> None:
    # A fresh database already has the table from the model baseline.
    if sa.inspect(op.get_bind()).has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("picture_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("mtime_ns", sa.Integer(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["picture_id"], ["picture.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("picture_id"),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        op.drop_table(TABLE)
