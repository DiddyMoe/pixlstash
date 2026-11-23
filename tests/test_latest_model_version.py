"""The "current tagger version" lookup: one query, index-served, same answer.

Tag health, the tag scan's confidence fallback and the boundary scan each pin
their prediction queries to the model version of the most recently written
non-``manual`` prediction row. They share :func:`latest_model_version`, which
reads ``ix_tag_prediction_predicted_at`` instead of sorting the whole table.

The answer must not change with the index. Without it SQLite scans in rowid
order and its top-1 sort keeps the first row it saw, so on equal timestamps the
LOWEST id wins and NULL timestamps sort last. The index is built
``(predicted_at DESC, id)`` and the query orders ``predicted_at DESC, id ASC``,
so the index walk produces that same answer without a sort. A plain
``(predicted_at)`` index would be read backwards and hand ties to the HIGHEST
id instead.

The vault is constructed but never ``start()``ed, so nothing races the seed.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import event
from sqlmodel import Session, select

from pixlstash.db_models import Picture
from pixlstash.db_models.tag_prediction import TagPrediction
from pixlstash.services import tag_health_service, tag_scan_service
from pixlstash.services.tag_health_service import latest_model_version
from pixlstash.utils.near_neighbor import EMBEDDING_DIM
from pixlstash.vault import Vault

_INDEX = "ix_tag_prediction_predicted_at"
_LOOKUP = "order by tag_prediction.predicted_at desc"
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Recorder:
    """Record every statement whose collapsed SQL contains *needle*."""

    def __init__(self, engine, needle: str):
        self._engine = engine
        self._needle = needle
        self.statements: list[str] = []
        self.parameters: list = []

    def _on_execute(self, conn, cursor, statement, parameters, context, executemany):
        collapsed = " ".join(statement.split()).lower()
        if self._needle in collapsed:
            self.statements.append(collapsed)
            self.parameters.append(parameters)

    def __enter__(self):
        event.listen(self._engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *exc_info):
        event.remove(self._engine, "before_cursor_execute", self._on_execute)
        return False


def _lookup_before_the_index(session: Session) -> str | None:
    """The lookup as each caller wrote it before they shared one helper."""
    return session.exec(
        select(TagPrediction.model_version)
        .where(TagPrediction.model_version != "manual")
        .order_by(TagPrediction.predicted_at.desc())
        .limit(1)
    ).first()


def _seed(vault, rows, *, embeddings: bool = False) -> list[int]:
    """Insert one picture per ``(model_version, predicted_at)`` row, in order.

    Returns the prediction ids, which follow insertion order.
    """

    def seed(session: Session):
        ids = []
        for index, (version, predicted_at) in enumerate(rows):
            picture = Picture(
                file_path=f"p{index}.png",
                format="png",
                image_embedding=(
                    np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes()
                    if embeddings
                    else None
                ),
            )
            session.add(picture)
            session.flush()
            prediction = TagPrediction(
                picture_id=picture.id,
                tag="stand-in tag",
                confidence=0.95,
                model_version=version,
                predicted_at=predicted_at,
            )
            session.add(prediction)
            session.flush()
            if predicted_at is None:
                # The column is nullable, but the ORM replaces a None with the
                # column default (now), so a NULL time is written directly.
                session.connection().exec_driver_sql(
                    "UPDATE tag_prediction SET predicted_at = NULL WHERE id = ?",
                    (prediction.id,),
                )
            ids.append(prediction.id)
        session.commit()
        return ids

    return vault.db.run_task(seed)


def _answer_without_the_index(vault) -> str | None:
    """Today's answer: the old lookup on a schema that lacks the new index."""

    def run(session: Session):
        connection = session.connection()
        sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (_INDEX,),
        ).scalar()
        connection.exec_driver_sql(f"DROP INDEX IF EXISTS {_INDEX}")
        try:
            return _lookup_before_the_index(session)
        finally:
            if sql:
                connection.exec_driver_sql(sql)
            session.commit()

    return vault.db.run_task(run)


_CASES = {
    "empty": ([], None),
    "only manual rows": ([("manual", _T0)], None),
    "single version": ([("v1", _T0), ("v1", _T0 + timedelta(seconds=1))], "v1"),
    "newest version wins": (
        [("v1", _T0), ("v2", _T0 + timedelta(hours=1)), ("v1", _T0)],
        "v2",
    ),
    "a newer manual row is skipped": (
        [
            ("v1", _T0),
            ("v2", _T0 + timedelta(hours=1)),
            ("manual", _T0 + timedelta(hours=2)),
        ],
        "v2",
    ),
    "a NULL time sorts last": ([("v1", None), ("v2", _T0), ("v3", None)], "v2"),
    "all NULL times: lowest id": ([("v1", None), ("v2", None), ("v3", None)], "v1"),
    "tied times: lowest id": (
        [
            ("v0", _T0),
            ("v1", _T0 + timedelta(hours=1)),
            ("v2", _T0 + timedelta(hours=1)),
        ],
        "v1",
    ),
    "tie after a newer manual row": (
        [
            ("v2", _T0 + timedelta(hours=1)),
            ("v1", _T0 + timedelta(hours=1)),
            ("manual", _T0 + timedelta(hours=3)),
        ],
        "v2",
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_the_lookup_gives_the_answer_it_gave_before_the_index(tmp_path, case):
    rows, expected = _CASES[case]
    with Vault(image_root=str(tmp_path)) as vault:
        _seed(vault, rows)

        assert vault.db.run_immediate_read_task(latest_model_version) == expected
        assert _answer_without_the_index(vault) == expected


def test_the_lookup_is_served_by_the_prediction_time_index(tmp_path):
    """No full scan of ``tag_prediction`` and no sort, however many rows."""
    with Vault(image_root=str(tmp_path)) as vault:
        _seed(vault, [("v1", _T0), ("manual", _T0 + timedelta(hours=1))])

        with _Recorder(vault.db._engine, _LOOKUP) as rec:
            vault.db.run_immediate_read_task(latest_model_version)
        assert len(rec.statements) == 1, rec.statements

        def explain(session: Session):
            rows = session.connection().exec_driver_sql(
                f"EXPLAIN QUERY PLAN {rec.statements[0]}", rec.parameters[0]
            )
            return [row[-1] for row in rows]

        plan = vault.db.run_immediate_read_task(explain)
        assert any(_INDEX in line for line in plan), plan
        assert not any("TEMP B-TREE" in line.upper() for line in plan), plan


def test_every_caller_issues_the_shared_lookup(tmp_path):
    """Tag health, the scan fallback and the boundary scan read the version
    through the one helper, so they cannot disagree about which version is
    current."""
    with Vault(image_root=str(tmp_path)) as vault:
        _seed(vault, [("v1", _T0), ("v2", _T0 + timedelta(hours=1))], embeddings=True)

        with _Recorder(vault.db._engine, _LOOKUP) as rec:
            vault.db.run_immediate_read_task(latest_model_version)
        shared = rec.statements[0]

        callers = {
            "tag health": lambda: vault.db.run_immediate_read_task(
                tag_health_service.compute_tag_health_rows
            ),
            # Two pictures and no tagged ground truth reach the confidence
            # fallback, which is where the scan pins its version.
            "tag scan fallback": lambda: tag_scan_service.scan_tag(
                vault, "stand-in tag", project=None
            ),
            "boundary scan": lambda: tag_scan_service.scan_boundary(
                vault, "stand-in tag"
            ),
        }
        for name, call in callers.items():
            with _Recorder(vault.db._engine, _LOOKUP) as rec:
                call()
            assert rec.statements, f"{name} issued no version lookup"
            assert set(rec.statements) == {shared}, (name, rec.statements)
