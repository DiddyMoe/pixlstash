"""The unprocessable-image registry outlasts a restart.

Each test builds a real ``VaultDatabase`` on a ``tmp_path`` library, so the
registry is the one the database constructs for itself; a restart is the
database closed and a new one opened on the same file. The registry's queued
writes are LOW, and the writer runs equal priorities first come, first served,
so a LOW no-op queued after them returns only once they have landed.
"""

import logging
import threading

import pytest
from sqlmodel import select

from pixlstash.database import DBPriority, VaultDatabase
from pixlstash.db_models import Picture
from pixlstash.db_models.unprocessable_image import UnprocessableImage


@pytest.fixture
def open_library(tmp_path):
    """Open (or re-open) the library at ``tmp_path``; every one is closed at the end."""
    opened = []

    def open_db():
        db = VaultDatabase(str(tmp_path / "vault.db"))
        opened.append(db)
        return db

    yield open_db
    for db in opened:
        db.close()


def _add_pictures(db, folder, count):
    """*count* pictures, each with an undecodable file. Returns ``[(id, path)]``."""
    folder.mkdir(exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"broken-{index}.png"
        path.write_bytes(b"example-not-an-image-%d" % index)
        paths.append(str(path))

    def insert(session):
        pictures = [Picture(file_path=path) for path in paths]
        session.add_all(pictures)
        session.commit()
        return [picture.id for picture in pictures]

    ids = db.run_task(insert, priority=DBPriority.LOW)
    return list(zip(ids, paths))


def _settle(db):
    db.run_task(lambda session: None, priority=DBPriority.LOW)


def _rows(db):
    def read(session):
        return {
            row.picture_id: (row.kind, row.file_path, row.mtime_ns, row.size)
            for row in session.exec(select(UnprocessableImage)).all()
        }

    return db.run_immediate_read_task(read)


def test_a_picture_held_before_a_restart_is_still_held_after_it(open_library, tmp_path):
    """Both kinds come back as they were marked, and the undecodable one is
    not marked a second time: the restart does not cost it another decode."""
    db = open_library()
    (broken_id, broken_path), (away_id, _) = _add_pictures(db, tmp_path / "pics", 2)
    away_path = str(tmp_path / "unmounted" / "away.png")
    registry = db.unprocessable_images
    assert registry.mark_unprocessable(broken_id, broken_path, reason="test")
    assert registry.mark_unreachable(away_id, away_path, reason="test")
    before = registry.snapshot()
    _settle(db)
    db.close()

    reopened = open_library().unprocessable_images

    assert reopened.active_suppressed_ids() == {broken_id, away_id}
    assert reopened.snapshot() == before
    assert reopened.is_suppressed(broken_id)
    assert not reopened.mark_unprocessable(broken_id, broken_path)


def test_a_lifted_forgotten_or_deleted_picture_leaves_no_row(open_library, tmp_path):
    db = open_library()
    pictures = _add_pictures(db, tmp_path / "pics", 3)
    registry = db.unprocessable_images
    for picture_id, path in pictures:
        assert registry.mark_unprocessable(picture_id, path)
    _settle(db)
    (rewritten_id, rewritten_path), (forgotten_id, _), (deleted_id, _) = pictures
    assert set(_rows(db)) == {rewritten_id, forgotten_id, deleted_id}

    with open(rewritten_path, "ab") as handle:
        handle.write(b"example-repair")
    assert not registry.is_suppressed(rewritten_id)
    registry.discard(forgotten_id)

    def delete_picture(session):
        session.delete(session.get(Picture, deleted_id))
        session.commit()

    db.run_task(delete_picture, priority=DBPriority.LOW)
    _settle(db)

    assert _rows(db) == {}


def test_a_mark_for_a_picture_already_gone_writes_nothing_and_logs_no_failure(
    open_library, tmp_path, caplog
):
    db = open_library()
    [(picture_id, path)] = _add_pictures(db, tmp_path / "pics", 1)
    missing_id = picture_id + 1000

    with caplog.at_level(logging.WARNING):
        assert db.unprocessable_images.mark_unprocessable(missing_id, path)
        _settle(db)

    assert _rows(db) == {}
    assert db.unprocessable_images.is_suppressed(missing_id)
    assert not [r for r in caplog.records if "failed" in r.getMessage()]


def test_the_first_read_of_the_rows_never_waits_on_the_writer(open_library, tmp_path):
    """The rows are read on first use, which can come while the writer is busy
    or from inside a writer task; neither may wait on the writer."""
    db = open_library()
    [(picture_id, path)] = _add_pictures(db, tmp_path / "pics", 1)
    assert db.unprocessable_images.mark_unprocessable(picture_id, path)
    _settle(db)
    db.close()

    busy = open_library()
    started, release = threading.Event(), threading.Event()
    held = busy.submit_task(
        lambda session: (started.set(), release.wait(30)),
        priority=DBPriority.IMMEDIATE,
    )
    assert started.wait(10)
    answers = []
    reader = threading.Thread(
        target=lambda: answers.append(busy.unprocessable_images.active_suppressed_ids())
    )
    reader.start()
    reader.join(10)
    release.set()
    held.result(30)
    assert answers == [{picture_id}], "the first read waited on the busy writer"
    busy.close()

    inside = open_library()
    on_writer = inside.submit_task(
        lambda session: inside.unprocessable_images.is_suppressed(picture_id),
        priority=DBPriority.IMMEDIATE,
    )
    assert on_writer.result(30) is True
