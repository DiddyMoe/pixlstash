"""Index the description-pending probe and the latest-prediction lookup.

``ix_picture_description_pending`` on ``picture (id)`` is partial, ``WHERE
description IS NULL OR description LIKE '\\_\\_description::%' ESCAPE '\\'``:
the rows ``MissingDescriptionFinder`` asks for on every planner sweep. Without
it the probe scanned the whole picture table even with nothing pending. The
finder renders the LIKE pattern inline, because SQLite only uses a partial
index whose LIKE pattern is the query's own literal.

``ix_tag_prediction_predicted_at`` on ``tag_prediction (predicted_at DESC,
id)`` serves ``tag_health_service.latest_model_version`` (newest non-manual
prediction, ties to the lowest id) without sorting the table.

Both are declared on the models, and the baseline builds a fresh vault from
the models, so each create and drop is guarded on the index name.

Revision ID: 0149_add_description_and_prediction_time_indexes
Revises: 0148_add_picture_set_smart
Create Date: 2026-11-23
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0149_add_description_and_prediction_time_indexes"
down_revision: Union[str, None] = "0148_add_picture_set_smart"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

__all__ = ["revision", "down_revision", "branch_labels", "depends_on"]

_DESCRIPTION_PENDING = "ix_picture_description_pending"
_PREDICTED_AT = "ix_tag_prediction_predicted_at"


def _indexes(inspector, table: str) -> set | None:
    """Index names on *table*, or ``None`` when the table does not exist."""
    if table not in inspector.get_table_names():
        return None
    return {ix["name"] for ix in inspector.get_indexes(table)}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    existing = _indexes(inspector, "picture")
    if existing is not None and _DESCRIPTION_PENDING not in existing:
        op.create_index(
            _DESCRIPTION_PENDING,
            "picture",
            ["id"],
            sqlite_where=sa.text(
                "description IS NULL OR description LIKE '\\_\\_description::%' "
                "ESCAPE '\\'"
            ),
        )

    existing = _indexes(inspector, "tag_prediction")
    if existing is not None and _PREDICTED_AT not in existing:
        op.create_index(
            _PREDICTED_AT,
            "tag_prediction",
            [sa.desc(sa.column("predicted_at")), "id"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    for table, name in (
        ("picture", _DESCRIPTION_PENDING),
        ("tag_prediction", _PREDICTED_AT),
    ):
        existing = _indexes(inspector, table)
        if existing is not None and name in existing:
            op.drop_index(name, table_name=table)
