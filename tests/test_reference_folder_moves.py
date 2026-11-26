"""The reference-folder scan follows a file moved inside the folder.

A move is one path disappearing and another appearing in the same scan pass.
Handled as a removal plus an import it costs the picture id, and with it the
tags, score, faces and set/stack membership hanging off it, so reorganizing
your own folders used to wipe everything PixlStash had added. The scan now
matches the two halves by pixel hash and updates ``file_path`` instead.

Ambiguous matches are deliberately not followed: identical pixels at several
paths are copies whose rows can differ, and guessing would move one picture's
work onto another picture's file.
"""

import os
import sys
import tempfile
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, inspect
from sqlmodel import Session, select

from pixlstash.db_models import (
    DeletedFileLog,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    Project,
    ReferenceFolder,
    Tag,
)
from pixlstash.db_models.external_move_review import ExternalMoveReview
from pixlstash.db_models.operation import Operation
from pixlstash.db_models.picture_move import PictureMove
from pixlstash.server import Server
from pixlstash.services import move_reconciliation_service
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.library_layout import DEFAULT_LAYOUT, format_layout
from pixlstash.tasks import TaskType
from pixlstash.tasks import missing_file_purge_task
from pixlstash.tasks.missing_file_purge_task import MissingFilePurgeTask
from pixlstash.database import DBPriority
from pixlstash.tasks import reference_folder_scan_task
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask

# ``Server.__init__`` calls ``vault.start()``, so the planner is live for the
# whole life of a module-scoped server and its sweeps land *inside* these tests
# rather than sitting in the long backoff a freshly built per-test vault had.
# Each of these owns something a test here writes by hand or asserts on:
_CONFLICTING_FINDERS = (
    # ThumbnailGenerationTask selects on ``thumbnail_width IS NULL`` and writes
    # the regenerated bitmap's dimensions back, so it can both repopulate a row
    # this module expects to stay blank and re-render a bitmap whose absence is
    # being asserted. Same reasoning as tests/test_inline_rotate.py.
    TaskType.THUMBNAIL_GENERATION,
    # ReferenceFolderScanFinder scans the same folders these tests scan by hand.
    # A sweep between ``_make_folder`` and ``_run_scan`` imports the files first,
    # and the manual scan then imports them again: duplicate rows for one path,
    # which breaks ``len(...) == 1`` and strands the curated row at the dead path
    # because ``existing_by_path`` keeps only the last row per file_path.
    TaskType.REFERENCE_FOLDER_SCAN,
    # MissingFilePurgeTask deletes any picture whose file is absent. Between the
    # ``os.rename`` and the scan that follows it, the moved picture is exactly
    # that, so the sweep destroys the row under test.
    TaskType.MISSING_FILE_PURGE,
    # TagTask deletes a picture's existing Tag rows before writing its own, so a
    # sweep landing after ``_tag`` removes the seeded label ``_tags`` asserts on.
    TaskType.TAGGER,
)


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "server-config.json")
        with Server(config_path) as srv:
            for task_type in _CONFLICTING_FINDERS:
                srv.vault._planner_work_finders.pop(task_type)
            # detach_finders() edits the planner's structures under its own lock,
            # so this is safe against the loop thread running right now.
            srv.vault._work_planner.detach_finders(_CONFLICTING_FINDERS)
            yield srv


def _make_folder(server, folder_dir, layout=None, **fields):
    """Register a reference folder that the background scanner will leave alone.

    ``ReferenceFolderScanFinder`` selects on ``last_scanned``, not on status: a
    row with NULL there is due for a scan *immediately*, and every folder in the
    database is a candidate. So between this insert and the first ``_run_scan``
    the planner is free to scan the same folder concurrently, and both passes
    import the same files - the CI gate failed on exactly that, twice, with
    ``assert 2 == 1`` and ``assert 4 == 2``: every row doubled.

    Stamping ``last_scanned`` now puts the folder outside the finder's rescan
    interval from the moment it exists, which closes the window by construction
    rather than by relying on the finder staying detached. That matters because
    the detachment is done once against the vault the fixture saw, and a vault
    can be rebuilt underneath it (``library_switch_service`` assigns a freshly
    built one, with every finder re-attached). The scan task itself re-stamps
    the column when it completes, so later scans stay outside the interval too.
    """
    os.makedirs(folder_dir, exist_ok=True)

    def _insert(session: Session):
        folder = ReferenceFolder(
            folder=folder_dir,
            label="refs",
            status="active",
            last_scanned=time.time(),
            layout=layout,
            **fields,
        )
        session.add(folder)
        session.commit()
        session.refresh(folder)
        return folder.id

    return server.vault.db.run_task(_insert)


def _make_image(path, color, size=8):
    """A flat image at ``path``. ``size`` is the square edge in pixels.

    A large flat BMP is how a sampled-hash collision is produced on purpose: the
    format is uncompressed, so the file is far past the 128 KiB sampling
    threshold and every sampled window is the same repeated pixel.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = "BMP" if path.lower().endswith(".bmp") else "PNG"
    Image.new("RGB", (size, size), color=color).save(path, format=fmt)
    return path


def _run_scan(server, folder_id, folder_dir, database=None):
    task = ReferenceFolderScanTask(
        database=database or server.vault.db,
        folder_id=folder_id,
        folder_path=folder_dir,
        resolved_path=folder_dir,
    )
    return task._run_task()


def _pictures(server, folder_dir):
    """Every picture indexed under this test's folder, newest id last.

    Scoped by path rather than by ``reference_folder_id`` because each test
    makes its own folder row against the shared module server, so a query by
    folder id would still have to be told which id, and the path each test
    already owns says it directly. ``tmp_path`` is per-test, so the prefix
    cannot collide with another test in this module.
    """
    return server.vault.db.run_task(
        lambda s: list(
            s.exec(
                select(Picture)
                .where(Picture.file_path.startswith(folder_dir))
                .order_by(Picture.id)
            ).all()
        )
    )


def _tag(server, picture_id, tag):
    def _add(session: Session):
        session.add(Tag(picture_id=picture_id, tag=tag))
        session.commit()

    server.vault.db.run_task(_add)


def _mark(server, picture_id, marker):
    """Stamp a field the scan never writes, so a re-imported row is detectable.

    SQLite reuses free rowids, so a deleted-and-re-added picture can come back
    with the same id: an id assertion alone would pass for the very behaviour
    these tests exist to catch. A marker only the test writes cannot survive a
    re-import.
    """

    def _update(session: Session):
        pic = session.get(Picture, picture_id)
        pic.description = marker
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_update)


def _tags(server, picture_id):
    rows = server.vault.db.run_task(
        lambda s: s.exec(select(Tag).where(Tag.picture_id == picture_id)).all()
    )
    return sorted(t.tag for t in rows if not t.tag.startswith("__"))


def _curate(server, picture_id, set_name):
    """Give a picture what losing its row would cost: a tag, a score, a set."""
    _tag(server, picture_id, "mira")
    _mark(server, picture_id, "keepme")

    def _add(session: Session):
        pic = session.get(Picture, picture_id)
        pic.score = 4
        session.add(pic)
        picture_set = PictureSet(name=set_name)
        session.add(picture_set)
        session.commit()
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()
        return picture_set.id

    return server.vault.db.run_task(_add)


def _curation(server, picture_id):
    """The row's path and what ``_curate`` gave it; ``None`` once the row is gone."""

    def _read(session: Session):
        pic = session.get(Picture, picture_id)
        if pic is None:
            return None
        members = session.exec(
            select(PictureSetMember).where(PictureSetMember.picture_id == picture_id)
        ).all()
        return {
            "file_path": pic.file_path,
            "description": pic.description,
            "score": pic.score,
            "set_ids": sorted(m.set_id for m in members),
        }

    state = server.vault.db.run_task(_read)
    if state is not None:
        state["tags"] = _tags(server, picture_id)
    return state


def _repoint(server, picture_id, file_path):
    """Point a row at a new file, as the explicit move route does."""

    def _update(session: Session):
        pic = session.get(Picture, picture_id)
        pic.file_path = file_path
        pic.original_file_name = os.path.basename(file_path)
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_update)


def _ledger_rows(server, file_path):
    """Deletion-ledger rows recorded for ``file_path``."""
    path_sha = DeletedFileLog.hash_path(file_path)
    return server.vault.db.run_task(
        lambda s: s.exec(
            select(DeletedFileLog).where(DeletedFileLog.path_sha == path_sha)
        ).all()
    )


class _WriteBetween:
    """Lands another committed write between a sweep's read and its own write.

    The scan and the purge sweep both read rows, judge them against the disk,
    and only then queue the write that acts on the judgement. Whatever the
    library commits in between - the explicit move route, a second sweep -
    lands on a row that has already been judged. Handed to the task as its
    database, this makes that window deterministic: when the task submits the
    writer callback named ``callback_name``, ``before`` runs first, as a write
    of its own. A read the task makes through ``run_immediate_read_task`` is
    matched by name too, for a window that has to open before the task's own
    look at the disk rather than before its write.
    """

    def __init__(self, db, callback_name, before):
        self._db = db
        self._callback_name = callback_name
        self._before = before
        self.fired = False

    def __getattr__(self, name):
        return getattr(self._db, name)

    def _fire_before(self, func):
        if not self.fired and getattr(func, "__name__", None) == self._callback_name:
            self.fired = True
            self._before()

    def run_task(self, func, *args, **kwargs):
        self._fire_before(func)
        return self._db.run_task(func, *args, **kwargs)

    def run_immediate_read_task(self, func, *args, **kwargs):
        self._fire_before(func)
        return self._db.run_immediate_read_task(func, *args, **kwargs)


class _StatementsOf:
    """Records the SQL one named callback emits, and the rows it returns.

    ``before_cursor_execute`` fires for every thread on the engine, and the
    module server's planner keeps querying ``picture`` throughout, so a listener
    left open for the whole scan would also catch the planner's statements.
    This one listens only while the callback runs, and only on the thread
    running it.
    """

    def __init__(self, db, callback_name):
        self._db = db
        self._callback_name = callback_name
        self.statements: list[str] = []
        self.returned = None

    def __getattr__(self, name):
        return getattr(self._db, name)

    def run_task(self, func, *args, **kwargs):
        if getattr(func, "__name__", None) != self._callback_name:
            return self._db.run_task(func, *args, **kwargs)

        def recorded(session, *call_args, **call_kwargs):
            engine = session.get_bind()
            caller = threading.get_ident()

            def _record(conn, cursor, statement, parameters, context, executemany):
                if threading.get_ident() == caller:
                    self.statements.append(" ".join(statement.lower().split()))

            event.listen(engine, "before_cursor_execute", _record)
            try:
                return func(session, *call_args, **call_kwargs)
            finally:
                event.remove(engine, "before_cursor_execute", _record)

        self.returned = self._db.run_task(recorded, *args, **kwargs)
        return self.returned


def test_moved_file_keeps_its_picture(server, tmp_path):
    """Moving a file into a subfolder must not cost the row or its tags."""
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))

    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    assert [os.path.basename(p.file_path) for p in indexed] == ["mira_042.png"]
    picture_id = indexed[0].id
    thumb_width = indexed[0].thumbnail_width
    assert thumb_width is not None, "the import must have rendered a thumbnail"
    _tag(server, picture_id, "mira")
    _mark(server, picture_id, "keepme")

    # Renamed as well as relocated, so the basename genuinely changes: the
    # download name is taken from it, and the explicit move route resets it too.
    moved_to = os.path.join(folder_dir, "Characters", "Mira", "final", "final_042.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)

    _run_scan(server, folder_id, folder_dir)

    after = _pictures(server, folder_dir)
    assert len(after) == 1, "a move must not leave a second row behind"
    assert after[0].description == "keepme", "the row itself must survive the move"
    assert after[0].id == picture_id
    assert after[0].file_path == moved_to
    assert _tags(server, picture_id) == ["mira"]
    # The thumbnail lives at sha256(file_path), so it has to travel with the
    # file. Nothing sweeps .ref_thumbs by anything but a row's current
    # file_path, so a bitmap left at the old name would never be collected.
    image_root = server.vault.db.image_root
    assert not os.path.exists(ImageUtils.get_thumbnail_path(image_root, original)), (
        "the bitmap at the old path-derived name would be an unreachable orphan"
    )
    assert os.path.exists(ImageUtils.get_thumbnail_path(image_root, moved_to))
    assert after[0].thumbnail_width == thumb_width, (
        "the bitmap was carried, so it must not be blanked for re-rendering"
    )
    assert after[0].original_file_name == "final_042.png", (
        "a renamed file must not keep downloading under its old name"
    )


def test_a_bitmap_that_cannot_be_carried_is_regenerated(server, tmp_path, monkeypatch):
    """A followed move whose bitmap rename fails is left for the thumbnail sweep.

    The bitmap is carried after the move commits, so a failed rename leaves a
    row at its new path with the old dimensions and no bitmap under its new
    path-derived name. Blanking the dimensions is what puts it back in front of
    ``MissingThumbnailFinder``; kept, the grid asks for a thumbnail that is not
    there for good.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    picture_id = indexed[0].id
    assert indexed[0].thumbnail_width is not None, "the import rendered a bitmap"
    image_root = server.vault.db.image_root
    old_thumb = ImageUtils.find_thumbnail(image_root, original)
    assert old_thumb is not None

    real_replace = os.replace

    def _refuse_the_bitmap(src, dst, *args, **kwargs):
        if os.fspath(src) == old_thumb:
            raise OSError(5, "Input/output error", src)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _refuse_the_bitmap)
    moved_to = os.path.join(folder_dir, "final", "final_042.png")
    os.makedirs(os.path.dirname(moved_to))
    os.rename(original, moved_to)

    result = _run_scan(server, folder_id, folder_dir)

    assert result["moved_picture_ids"] == [picture_id], result
    after = _pictures(server, folder_dir)
    assert [(p.id, p.file_path) for p in after] == [(picture_id, moved_to)]
    assert not os.path.exists(ImageUtils.get_thumbnail_path(image_root, moved_to))
    assert (after[0].thumbnail_width, after[0].thumbnail_height) == (None, None), (
        "a row with no bitmap at its new name must be left for regeneration"
    )


def test_deleted_file_is_still_removed(server, tmp_path):
    """The rescue must not turn a genuine deletion into a kept row.

    The deletion is paired with an unrelated *addition* so the matcher is
    actually entered. Without one, ``new_paths`` is empty and
    ``_match_moved_paths`` returns on its first line, which would test the
    early return rather than the pairing this is about.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "gone.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))

    _run_scan(server, folder_id, folder_dir)
    assert sorted(
        os.path.basename(p.file_path) for p in _pictures(server, folder_dir)
    ) == ["gone.png", "kept.png"]

    os.remove(doomed)
    # Different pixels, so this is not the deleted file arriving elsewhere.
    _make_image(os.path.join(folder_dir, "unrelated.png"), (7, 200, 90))
    _run_scan(server, folder_id, folder_dir)

    remaining = _pictures(server, folder_dir)
    assert sorted(os.path.basename(p.file_path) for p in remaining) == [
        "kept.png",
        "unrelated.png",
    ]


def test_a_removal_overtaken_by_a_repoint_keeps_the_picture(server, tmp_path):
    """A row repointed after the scan judged it missing must not be deleted.

    The scan decides from the rows it read and deletes later, by id. If the
    row was pointed at a live file in between, deleting it takes the
    picture's tags, score and set membership with it for a file that is right
    there on disk. The decision is stale, so it is dropped and the next scan
    judges the row as it is now.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == original
    )
    set_id = _curate(server, picture_id, "Stale removal set")

    # Out of the folder, so the scan sees a removal with nothing to pair it with.
    parked = str(tmp_path / "parked" / "mira_042.png")
    os.makedirs(os.path.dirname(parked))
    os.rename(original, parked)
    landed = os.path.join(folder_dir, "final", "final_042.png")

    def _move_back_in():
        os.makedirs(os.path.dirname(landed))
        os.rename(parked, landed)
        _repoint(server, picture_id, landed)

    database = _WriteBetween(server.vault.db, "delete_removed", _move_back_in)
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired, "the repoint must land inside the window under test"
    expected = {
        "file_path": landed,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert _curation(server, picture_id) == expected, (
        "a row pointed at a live file after the scan looked must not be deleted"
    )
    assert os.path.isfile(landed)
    assert result["removed_count"] == 0, result

    # The next scan sees the row where it now is: nothing to remove or import.
    _run_scan(server, folder_id, folder_dir)
    assert _curation(server, picture_id) == expected
    assert sorted(p.file_path for p in _pictures(server, folder_dir)) == sorted(
        [landed, os.path.join(folder_dir, "kept.png")]
    )


def test_a_removal_is_not_saved_by_an_unrelated_write(server, tmp_path):
    """Only a change to what the decision rested on makes it stale.

    A tag landing between the read and the delete leaves the row's path,
    owner and deletion state as the scan saw them, so the row whose file is
    gone is still deleted, exactly as before. A check that skipped every row
    touched in the window would pass the test above and keep dead rows for
    ever.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "gone.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == doomed
    )
    os.remove(doomed)

    database = _WriteBetween(
        server.vault.db, "delete_removed", lambda: _tag(server, picture_id, "late")
    )
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired
    assert result["removed_count"] == 1, result
    assert server.vault.db.run_task(lambda s: s.get(Picture, picture_id)) is None
    assert [os.path.basename(p.file_path) for p in _pictures(server, folder_dir)] == [
        "kept.png"
    ]


def test_a_removal_is_not_made_on_a_file_that_came_back(server, tmp_path):
    """A row whose file is back at its path after the walk is kept.

    The removal rests on the file being absent as well as on the row. A file
    put back at the same path leaves the row exactly as the scan read it, so
    only a fresh look at the disk after the walk shows the decision no longer
    holds. That look is made before the delete is queued, off the writer (see
    ``test_a_slow_disk_never_holds_the_writer``), so the file comes back while
    the scan consults the move journal: after the walk, before the look.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == original
    )
    set_id = _curate(server, picture_id, "Returned removal set")

    # Out of the folder, so the scan sees a removal with nothing to pair it with.
    parked = str(tmp_path / "parked" / "mira_042.png")
    os.makedirs(os.path.dirname(parked))
    os.rename(original, parked)

    database = _WriteBetween(
        server.vault.db, "_load_move_journal", lambda: os.rename(parked, original)
    )
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired, "the file must come back inside the window under test"
    assert _curation(server, picture_id) == {
        "file_path": original,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }, "a row whose file is back must not be deleted"
    assert result["removed_count"] == 0, result

    rescan = _run_scan(server, folder_id, folder_dir)
    assert rescan["new_count"] == 0, rescan
    assert rescan["removed_count"] == 0, rescan


def _hold_the_writer(server, seconds: float):
    """Queue a write that keeps the writer busy for *seconds*, ahead of the scan's."""

    def busy(session: Session):
        time.sleep(seconds)

    return server.vault.db.submit_task(busy, priority=DBPriority.IMMEDIATE)


def test_a_delete_kept_waiting_is_not_made_on_a_file_that_came_back(
    server, tmp_path, monkeypatch
):
    """A look at the disk counts only while it is fresh when the delete runs.

    The scan looks for the vanished file off the writer, then queues the
    delete. A writer busy with other work keeps the delete waiting, and a file
    put back in that wait used to lose its row to a look made before it
    returned. The writer now refuses a look older than the limit, and the scan
    looks again and finds the file.
    """
    monkeypatch.setattr(reference_folder_scan_task, "_REMOVAL_CHECK_MAX_AGE_S", 0.2)
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == original
    )
    set_id = _curate(server, picture_id, "Kept waiting removal set")
    parked = str(tmp_path / "parked" / "mira_042.png")
    os.makedirs(os.path.dirname(parked))
    os.rename(original, parked)
    busy = []

    def _back_while_the_writer_is_busy():
        busy.append(_hold_the_writer(server, 0.6))
        os.rename(parked, original)

    database = _WriteBetween(
        server.vault.db, "delete_removed", _back_while_the_writer_is_busy
    )
    result = _run_scan(server, folder_id, folder_dir, database=database)
    busy[0].result(timeout=10)

    assert database.fired, "the file must come back after the scan looked"
    assert _curation(server, picture_id) == {
        "file_path": original,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }, "a row whose file is back when the delete runs must not be deleted"
    assert result["removed_count"] == 0, result


def test_a_delete_kept_waiting_is_still_made_when_the_file_stays_gone(
    server, tmp_path, monkeypatch
):
    """The refusal above costs a second look, not the removal.

    A busy writer makes the first look stale; the scan looks again, the file
    is still gone, and the row is deleted in the same scan.
    """
    monkeypatch.setattr(reference_folder_scan_task, "_REMOVAL_CHECK_MAX_AGE_S", 0.2)
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "gone.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == doomed
    )
    os.remove(doomed)
    busy = []
    database = _WriteBetween(
        server.vault.db,
        "delete_removed",
        lambda: busy.append(_hold_the_writer(server, 0.6)),
    )

    result = _run_scan(server, folder_id, folder_dir, database=database)
    busy[0].result(timeout=10)

    assert database.fired
    assert result["removed_count"] == 1, result
    assert server.vault.db.run_task(lambda s: s.get(Picture, picture_id)) is None


def test_a_slow_disk_never_holds_the_writer(server, tmp_path, monkeypatch):
    """The scan's look at the disk is made off the writer.

    Whether a vanished file came back is a stat, and a stat can block for as
    long as the filesystem likes: a network share that dropped, a disk spinning
    up. Made on the writer, it holds every other write in the library - an
    upload, a tag, a score - until the disk answers. Here the stat blocks until
    released, and another write has to complete in the meantime.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "gone.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, folder_id, folder_dir)
    picture_id = next(
        p.id for p in _pictures(server, folder_dir) if p.file_path == doomed
    )
    os.remove(doomed)

    entered = threading.Event()
    release = threading.Event()
    real_lexists = os.path.lexists

    def _slow_lexists(path):
        if os.fspath(path) == doomed:
            entered.set()
            release.wait(timeout=60)
        return real_lexists(path)

    monkeypatch.setattr(os.path, "lexists", _slow_lexists)
    outcome: dict = {}

    def _scan():
        outcome["result"] = _run_scan(server, folder_id, folder_dir)

    def _relabel(session: Session):
        folder = session.get(ReferenceFolder, folder_id)
        folder.label = "relabelled while the disk was slow"
        session.add(folder)
        session.commit()

    scanner = threading.Thread(target=_scan, name="slow-disk-scan")
    writer_client = threading.Thread(
        target=lambda: server.vault.db.run_task(_relabel), name="other-write"
    )
    scanner.start()
    try:
        assert entered.wait(timeout=60), "the scan never looked at the file"
        writer_client.start()
        writer_client.join(timeout=10)
        held = writer_client.is_alive()
    finally:
        release.set()
        scanner.join(timeout=60)
        writer_client.join(timeout=60)

    assert not held, "another write waited on the scan's stat"
    assert not scanner.is_alive()
    assert outcome["result"]["removed_count"] == 1, outcome
    assert server.vault.db.run_task(lambda s: s.get(Picture, picture_id)) is None
    label = server.vault.db.run_task(lambda s: s.get(ReferenceFolder, folder_id).label)
    assert label == "relabelled while the disk was slow"


def test_a_move_overtaken_by_a_newer_one_keeps_the_newer_path(server, tmp_path):
    """A followed move must not overwrite a path written after the scan looked.

    The scan pairs the vanished path with the arrived one and repoints the row
    later, by id. If the file moved on again and the row followed it in
    between, assigning the scan's path would point the picture at a file that
    is no longer there. The newer path wins; the skipped pair claims no move
    journal row and is not reported as a move.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, "Stale move set")

    renamed = os.path.join(folder_dir, "renamed_042.png")
    os.rename(original, renamed)

    # Journalled as PixlStash's own move, so applying the pair would claim it.
    def _journal(session: Session):
        session.add(
            PictureMove(picture_id=picture_id, old_path=original, new_path=renamed)
        )
        session.commit()

    server.vault.db.run_task(_journal)
    newest = os.path.join(folder_dir, "final", "final_042.png")

    def _move_on():
        os.makedirs(os.path.dirname(newest))
        os.rename(renamed, newest)
        _repoint(server, picture_id, newest)

    database = _WriteBetween(server.vault.db, "apply_moves", _move_on)
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired, "the repoint must land inside the window under test"
    expected = {
        "file_path": newest,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert _curation(server, picture_id) == expected, (
        "the scan's older path must not overwrite the row's newer one"
    )
    assert os.path.isfile(newest)
    assert picture_id not in result["moved_picture_ids"], result
    image_root = server.vault.db.image_root
    assert not os.path.exists(ImageUtils.get_thumbnail_path(image_root, renamed)), (
        "a move the writer skipped must not carry the bitmap to a path no row names"
    )
    assert ImageUtils.find_thumbnail(image_root, original) is not None, (
        "the bitmap stays where the writer that moved the row last left it"
    )
    journal = server.vault.db.run_task(
        lambda s: s.exec(
            select(PictureMove).where(PictureMove.picture_id == picture_id)
        ).all()
    )
    assert [row.consumed for row in journal] == [False], (
        "a pair that was not applied must leave the journal unclaimed"
    )

    _run_scan(server, folder_id, folder_dir)
    assert _curation(server, picture_id) == expected
    assert [p.id for p in _pictures(server, folder_dir)] == [picture_id]


def test_a_move_overtaken_by_a_scrapheap_still_follows_the_file(server, tmp_path):
    """A picture sent to the Scrapheap while its move is pending still moves.

    The Scrapheap flag says nothing about where the file is, so it is no reason
    to drop the move. Dropped, the row stays at a path that no longer exists,
    and the next scan deletes it outright - tags, score and sets with it - and
    imports the file again as a new picture with none of them.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, "Scrapheaped move set")

    renamed = os.path.join(folder_dir, "renamed_042.png")
    os.rename(original, renamed)

    def _scrapheap(session: Session):
        pic = session.get(Picture, picture_id)
        pic.deleted = True
        session.add(pic)
        session.commit()

    database = _WriteBetween(
        server.vault.db, "apply_moves", lambda: server.vault.db.run_task(_scrapheap)
    )
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired, "the soft-delete must land inside the window under test"
    expected = {
        "file_path": renamed,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert picture_id in result["moved_picture_ids"], result
    assert _curation(server, picture_id) == expected, (
        "a row sent to the Scrapheap must still follow its file"
    )
    assert (
        server.vault.db.run_task(lambda s: s.get(Picture, picture_id).deleted) is True
    )

    # The row is where its file is, so the next scan has nothing to do.
    rescan = _run_scan(server, folder_id, folder_dir)
    assert rescan["new_count"] == 0, rescan
    assert rescan["removed_count"] == 0, rescan
    assert _curation(server, picture_id) == expected


def test_the_purge_sweep_keeps_a_row_repointed_after_it_looked(server, tmp_path):
    """The purge sweep reads a batch and deletes by id later: the same window.

    Deleting there also writes a ``file_removed=True`` ledger row for the old
    path, which stops any file that later appears at that path from being
    imported. A stale decision writes neither.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, "Stale purge set")

    parked = str(tmp_path / "parked" / "mira_042.png")
    os.makedirs(os.path.dirname(parked))
    os.rename(original, parked)
    snapshot = server.vault.db.run_immediate_read_task(
        lambda s: [s.get(Picture, picture_id)]
    )
    landed = os.path.join(folder_dir, "final", "final_042.png")

    def _move_back_in():
        os.makedirs(os.path.dirname(landed))
        os.rename(parked, landed)
        _repoint(server, picture_id, landed)

    database = _WriteBetween(server.vault.db, "_purge_pictures", _move_back_in)
    result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()

    assert database.fired, "the repoint must land inside the window under test"
    assert _curation(server, picture_id) == {
        "file_path": landed,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert result == {"purged": 0, "repaired": 0, "deferred": 0}, result
    assert _ledger_rows(server, original) == [], (
        "a purge that did not happen must not bar the old path from import"
    )


def test_the_purge_sweep_keeps_a_row_whose_file_came_back(server, tmp_path):
    """A file back at its path by the time of the purge keeps its row.

    The file returns after the sweep's first look, while it reads the move
    journal. The row is untouched, so only the fresh look the purge takes, off
    the writer and just before queueing the delete, shows the purge no longer
    holds. Purging anyway would also write a ``file_removed=True`` ledger row
    that keeps the live file out of the library for good.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, "Returned purge set")

    snapshot = server.vault.db.run_immediate_read_task(
        lambda s: [s.get(Picture, picture_id)]
    )
    parked = str(tmp_path / "parked" / "mira_042.png")
    os.makedirs(os.path.dirname(parked))
    os.rename(original, parked)

    database = _WriteBetween(
        server.vault.db, "_load_move_journal", lambda: os.rename(parked, original)
    )
    result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()

    assert database.fired, "the file must come back inside the window under test"
    assert _curation(server, picture_id) == {
        "file_path": original,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert result == {"purged": 0, "repaired": 0, "deferred": 0}, result
    assert _ledger_rows(server, original) == [], (
        "a purge that did not happen must not bar the path from import"
    )


def test_the_purge_sweep_does_not_repair_over_a_newer_path(server, tmp_path):
    """A journal repair decided on the row's old path must not overwrite a newer one.

    The sweep repoints a row PixlStash moved but did not finish recording. If
    the row was repointed again before that write, the repair is stale.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, "Stale repair set")

    # A PixlStash move whose rename landed and whose row update did not.
    filed = os.path.join(folder_dir, "filed", "mira_042.png")
    os.makedirs(os.path.dirname(filed))
    os.rename(original, filed)

    def _journal(session: Session):
        session.add(
            PictureMove(picture_id=picture_id, old_path=original, new_path=filed)
        )
        session.commit()

    server.vault.db.run_task(_journal)
    snapshot = server.vault.db.run_immediate_read_task(
        lambda s: [s.get(Picture, picture_id)]
    )
    newest = os.path.join(folder_dir, "final", "final_042.png")

    def _move_on():
        os.makedirs(os.path.dirname(newest))
        os.rename(filed, newest)
        _repoint(server, picture_id, newest)

    database = _WriteBetween(server.vault.db, "_repair_moved_pictures", _move_on)
    result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()

    assert database.fired, "the repoint must land inside the window under test"
    assert _curation(server, picture_id) == {
        "file_path": newest,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert result == {"purged": 0, "repaired": 0, "deferred": 0}, result


def _purge_one(server, tmp_path, name):
    """A curated reference-folder picture whose file is then removed.

    Returns ``(picture_id, path, snapshot, set_id)``; the snapshot is the row as
    the sweep's finder would have read it, after the file went.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    path = _make_image(os.path.join(folder_dir, name), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, f"Purge set {name}")
    os.remove(path)
    snapshot = server.vault.db.run_immediate_read_task(
        lambda s: [s.get(Picture, picture_id)]
    )
    return picture_id, path, snapshot, set_id


@pytest.mark.parametrize(
    "left_behind",
    [
        pytest.param(
            "dangling-link",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="creating a symlink needs privileges on Windows",
            ),
        ),
        "nothing",
    ],
)
def test_the_purge_sweep_and_the_scan_agree_on_what_is_present(
    server, tmp_path, left_behind
):
    """One meaning of "present" for the purge sweep and the folder scan.

    The scan's walk lists a symlinked file, a dangling one included, so it
    keeps the row of a picture whose path is now a dangling link (a link into a
    drive that is not mounted, typically). The purge used to follow the link,
    find no file, and purge the row with a ``file_removed=True`` ledger entry
    the scan's next pass then honoured. Both sides now ask ``lexists``. With
    nothing at the path, the purge still purges.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    path = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, f"Present set {left_behind}")
    os.remove(path)
    if left_behind == "dangling-link":
        os.symlink(str(tmp_path / "unmounted" / "mira_042.png"), path)
        assert os.path.lexists(path) and not os.path.exists(path)
    snapshot = server.vault.db.run_immediate_read_task(
        lambda s: [s.get(Picture, picture_id)]
    )

    result = MissingFilePurgeTask(
        database=server.vault.db, pictures=snapshot
    )._run_task()

    if left_behind == "nothing":
        assert result == {"purged": 1, "repaired": 0, "deferred": 0}, result
        assert _curation(server, picture_id) is None
        assert [row.file_removed for row in _ledger_rows(server, path)] == [True]
        return
    assert result == {"purged": 0, "repaired": 0, "deferred": 0}, result
    assert _ledger_rows(server, path) == []
    scan = _run_scan(server, folder_id, folder_dir)
    assert (scan["removed_count"], scan["new_count"]) == (0, 0), scan
    assert _curation(server, picture_id) == {
        "file_path": path,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }, "both sweeps keep the row of a dangling link"


def test_a_purge_kept_waiting_is_not_made_on_a_file_that_came_back(
    server, tmp_path, monkeypatch
):
    """The purge's look at the disk counts only while it is fresh at the delete.

    The sweep looks off the writer, then queues the delete. A writer busy with
    other work keeps it waiting, and a file put back in that wait would lose its
    row (and be barred from re-import by the ledger) to a look made before it
    returned. The writer refuses a look older than the limit; the sweep looks
    again and finds the file.
    """
    monkeypatch.setattr(missing_file_purge_task, "REMOVAL_CHECK_MAX_AGE_S", 0.2)
    picture_id, path, snapshot, set_id = _purge_one(server, tmp_path, "back.png")
    parked = str(tmp_path / "parked.png")
    _make_image(parked, (10, 20, 30))
    busy = []

    def _back_while_the_writer_is_busy():
        busy.append(_hold_the_writer(server, 0.6))
        os.rename(parked, path)

    database = _WriteBetween(
        server.vault.db, "_purge_pictures", _back_while_the_writer_is_busy
    )
    result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()
    busy[0].result(timeout=10)

    assert database.fired, "the file must come back after the sweep looked"
    assert result == {"purged": 0, "repaired": 0, "deferred": 0}, result
    assert _curation(server, picture_id) == {
        "file_path": path,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }
    assert _ledger_rows(server, path) == []


def test_a_purge_kept_waiting_is_still_made_when_the_file_stays_gone(
    server, tmp_path, monkeypatch
):
    """The refusal above costs a second look, not the purge."""
    monkeypatch.setattr(missing_file_purge_task, "REMOVAL_CHECK_MAX_AGE_S", 0.2)
    picture_id, path, snapshot, _ = _purge_one(server, tmp_path, "gone.png")
    busy = []
    database = _WriteBetween(
        server.vault.db,
        "_purge_pictures",
        lambda: busy.append(_hold_the_writer(server, 0.6)),
    )

    result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()
    busy[0].result(timeout=10)

    assert database.fired
    assert result == {"purged": 1, "repaired": 0, "deferred": 0}, result
    assert _curation(server, picture_id) is None


def test_the_purge_sweep_never_looks_at_the_disk_on_the_writer(
    server, tmp_path, monkeypatch
):
    """Every look the purge takes at a file is made on the sweep's own thread.

    A stat can block as long as a dropped share likes, and the writer holds
    every other write in the library while it waits. The purge used to look
    again on the writer, inside the delete's transaction.
    """
    picture_id, path, snapshot, _ = _purge_one(server, tmp_path, "slow.png")
    lookers = []
    real_lexists = os.path.lexists

    def _recording_lexists(candidate):
        if os.fspath(candidate) == path:
            lookers.append(threading.current_thread())
        return real_lexists(candidate)

    monkeypatch.setattr(os.path, "lexists", _recording_lexists)
    result = MissingFilePurgeTask(
        database=server.vault.db, pictures=snapshot
    )._run_task()

    assert result == {"purged": 1, "repaired": 0, "deferred": 0}, result
    assert lookers, "the purge never looked at the file"
    assert set(lookers) == {threading.current_thread()}, lookers


def test_a_file_back_after_the_purge_looked_is_reported(server, tmp_path, caplog):
    """The residual race: a file put back between the fresh look and the commit.

    The filesystem and the database share no transaction, so this window is
    bounded (by ``REMOVAL_CHECK_MAX_AGE_S``), not closed. The row is purged on
    a look that was true when it was taken, and the look after the delete says
    so, naming the path the ledger now bars from routine re-import.
    """
    picture_id, path, snapshot, _ = _purge_one(server, tmp_path, "late.png")
    parked = str(tmp_path / "parked-late.png")
    _make_image(parked, (10, 20, 30))
    database = _WriteBetween(
        server.vault.db, "_purge_pictures", lambda: os.rename(parked, path)
    )

    with caplog.at_level("WARNING", logger=missing_file_purge_task.__name__):
        result = MissingFilePurgeTask(database=database, pictures=snapshot)._run_task()

    assert database.fired
    assert result == {"purged": 1, "repaired": 0, "deferred": 0}, result
    assert _curation(server, picture_id) is None
    assert any(
        record.levelname == "WARNING"
        and path in record.getMessage()
        and "came back" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# The blob columns on ``picture``. The scan compares paths, hashes and sidecar
# bookkeeping and never needs one of these, so none may be read off disk by it.
_BLOB_COLUMNS = ("image_embedding", "text_embedding", "likeness_parameters")


def test_the_scan_reads_no_blob_column_of_the_pictures_it_already_has(server, tmp_path):
    """The scan's read of the folder's indexed rows selects no embedding.

    A root scan reads every indexed row on every pass, so a full-row read pulls
    each picture's embeddings off disk only for the scan to drop them.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id

    recorder = _StatementsOf(server.vault.db, "fetch_existing")
    _run_scan(server, folder_id, folder_dir, database=recorder)

    picture_reads = [s for s in recorder.statements if " from picture " in s]
    assert len(picture_reads) == 1, recorder.statements
    selected = picture_reads[0].split(" from picture ")[0]
    assert "picture.file_path" in selected, selected
    for column in _BLOB_COLUMNS:
        assert column not in selected, f"the scan selects {column}: {selected}"
    # The same narrowing, seen from the rows the rest of the scan works on.
    [row] = recorder.returned
    assert row.id == picture_id
    assert set(_BLOB_COLUMNS) <= inspect(row).unloaded


def test_one_scan_over_the_narrowed_rows_reconciles_everything(server, tmp_path):
    """A move, a removal, an import and a sidecar edit, all in one pass.

    The rows the scan works on carry only the columns it reads, and they are
    detached by the time it reads them, so a column the read left out raises
    rather than reading as empty. Between them these cases read every column
    it keeps: path and scrapheap flag, pixel hash and size, and the sidecar
    bookkeeping with the description.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    moving = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    doomed = _make_image(os.path.join(folder_dir, "gone.png"), (200, 100, 50))
    captioned = _make_image(os.path.join(folder_dir, "kept.png"), (7, 200, 90))
    binned = _make_image(os.path.join(folder_dir, "binned.png"), (90, 90, 200))
    tags_file = os.path.join(folder_dir, "kept_tags.txt")
    description_file = os.path.join(folder_dir, "kept_description.txt")
    with open(tags_file, "w", encoding="utf-8") as fh:
        fh.write("mira, coat")
    with open(description_file, "w", encoding="utf-8") as fh:
        fh.write("Mira by the window.")

    assert _run_scan(server, folder_id, folder_dir)["new_count"] == 4
    ids = {os.path.basename(p.file_path): p.id for p in _pictures(server, folder_dir)}
    _tag(server, ids["mira_042.png"], "mira")
    _mark(server, ids["mira_042.png"], "keepme")

    def _bin(session: Session):
        pic = session.get(Picture, ids["binned.png"])
        pic.deleted = True
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_bin)

    moved_to = os.path.join(folder_dir, "final", "final_042.png")
    os.makedirs(os.path.dirname(moved_to))
    os.rename(moving, moved_to)
    os.remove(doomed)
    arrival = _make_image(os.path.join(folder_dir, "new.png"), (1, 150, 60))
    with open(tags_file, "w", encoding="utf-8") as fh:
        fh.write("mira, scarf")
    with open(description_file, "w", encoding="utf-8") as fh:
        fh.write("Mira by the door.")
    # A later mtime, whatever the filesystem's timestamp resolution.
    later = time.time() + 60
    os.utime(tags_file, (later, later))
    os.utime(description_file, (later, later))

    result = _run_scan(server, folder_id, folder_dir)

    assert result["moved_picture_ids"] == [ids["mira_042.png"]]
    assert result["removed_count"] == 1
    assert result["new_count"] == 1
    assert result["caption_updated_picture_ids"] == [ids["kept.png"]]
    after = {p.file_path: p for p in _pictures(server, folder_dir)}
    assert sorted(after) == sorted([moved_to, captioned, binned, arrival])
    assert after[moved_to].id == ids["mira_042.png"]
    assert after[moved_to].description == "keepme"
    assert _tags(server, ids["mira_042.png"]) == ["mira"]
    assert after[captioned].description == "Mira by the door."
    assert _tags(server, ids["kept.png"]) == ["mira", "scarf"]
    assert after[binned].id == ids["binned.png"]
    assert after[binned].deleted is True


def test_a_sampled_hash_collision_of_a_different_size_is_not_a_move(server, tmp_path):
    """The size is part of the key, so an equal digest alone cannot pair two files.

    ``calculate_hash_from_file_path`` samples 8 x 8 KiB windows of anything over
    128 KiB and does not mix the size into the digest, which its own docstring
    says makes it a candidate key rather than an identity. Driving the matcher
    directly is what pins that: a collision between two real image files is
    awkward to construct on purpose, and the guard has to hold regardless of how
    the stored ``pixel_sha`` came to be written.
    """
    folder_dir = str(tmp_path / "refs")
    os.makedirs(folder_dir, exist_ok=True)
    arrived = _make_image(os.path.join(folder_dir, "arrived.png"), (10, 20, 30))
    task = ReferenceFolderScanTask(
        database=server.vault.db,
        folder_id=0,
        folder_path=folder_dir,
        resolved_path=folder_dir,
    )
    gone = os.path.join(folder_dir, "gone.png")
    sha = ImageUtils.calculate_hash_from_file_path(arrived)
    real_size = os.path.getsize(arrived)

    def _match(size_bytes):
        return task._match_moved_paths(
            {gone: Picture(file_path=gone, pixel_sha=sha, size_bytes=size_bytes)},
            {arrived},
            {gone},
        )

    assert _match(real_size) == {gone: arrived}, (
        "the same digest and the same size is the move this feature follows"
    )
    assert _match(real_size + 1) == {}, (
        "the same digest at a different size is a different file; pairing them "
        "would rebind one picture's tags and memberships onto another's file"
    )


@pytest.mark.parametrize("scrapheap", [False, True], ids=["live", "scrapheap"])
def test_a_moved_file_keeps_its_picture_in_or_out_of_the_scrapheap(
    server, tmp_path, scrapheap
):
    """A picture in the Scrapheap follows its file as a live one does.

    ``fetch_existing`` loads ``deleted=True`` rows on purpose, and the flag
    says nothing about where the file is. Left out of the matching, a binned
    picture whose file moved is deleted outright as a removal - its tags,
    description, score and sets with it - and the file comes back as a new,
    live picture with none of them: emptying the Scrapheap by renaming a file.
    The move keeps the flag, so a binned picture stays binned.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(os.path.join(folder_dir, "binned.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    set_id = _curate(server, picture_id, f"Moved set ({scrapheap})")

    def _scrapheap(session: Session):
        pic = session.get(Picture, picture_id)
        pic.deleted = True
        session.add(pic)
        session.commit()

    if scrapheap:
        server.vault.db.run_task(_scrapheap)
    moved_to = os.path.join(folder_dir, "sorted", "renamed.png")
    os.makedirs(os.path.dirname(moved_to))
    os.rename(original, moved_to)

    result = _run_scan(server, folder_id, folder_dir)

    assert result["moved_picture_ids"] == [picture_id], result
    assert result["new_count"] == 0, result
    assert result["removed_count"] == 0, result
    assert [(p.id, p.deleted) for p in _pictures(server, folder_dir)] == [
        (picture_id, scrapheap)
    ], "one row, the same one, with its Scrapheap flag as it was"
    assert _curation(server, picture_id) == {
        "file_path": moved_to,
        "description": "keepme",
        "score": 4,
        "set_ids": [set_id],
        "tags": ["mira"],
    }


def test_an_unhashed_unchanged_file_blocks_matching(server, tmp_path):
    """A stable row with no pixel_sha means 'unknown collision', not 'no collision'.

    ``pixel_sha`` is nullable and backfilled in the background, so a present
    unchanged file can be invisible to the ambiguity count - and it is exactly
    the file whose existence would have refused the match.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "a.png"), (10, 20, 30))
    twin = _make_image(os.path.join(folder_dir, "b.png"), (10, 20, 30))

    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    assert sorted(os.path.basename(p.file_path) for p in indexed) == ["a.png", "b.png"]
    doomed_id = next(p.id for p in indexed if p.file_path == doomed)
    _mark(server, doomed_id, "keepme")

    # The twin loses its hash the way an un-backfilled row would have it.
    def _blank_sha(session: Session):
        pic = session.exec(select(Picture).where(Picture.file_path == twin)).one()
        pic.pixel_sha = None
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_blank_sha)

    os.remove(doomed)
    _make_image(os.path.join(folder_dir, "c.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)

    after = _pictures(server, folder_dir)
    assert not any(p.description == "keepme" for p in after), (
        "the unchanged twin is invisible to the count while its hash is NULL, "
        "so the match cannot be shown to be 1:1 and must not be followed"
    )


def test_an_unchanged_twin_makes_the_match_ambiguous(server, tmp_path):
    """A stable file sharing the key blocks the pairing, not just a new one.

    Deleting one copy and separately adding another looks 1:1 when only the
    removed and new sets are counted, but an untouched identical file in the
    folder means there is no telling which row the arrival belongs to.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "a.png"), (10, 20, 30))
    _make_image(os.path.join(folder_dir, "b.png"), (10, 20, 30))  # the stable twin

    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    assert sorted(os.path.basename(p.file_path) for p in indexed) == ["a.png", "b.png"]
    doomed_id = next(p.id for p in indexed if p.file_path == doomed)
    _mark(server, doomed_id, "keepme")

    os.remove(doomed)
    _make_image(os.path.join(folder_dir, "c.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)

    after = _pictures(server, folder_dir)
    assert sorted(os.path.basename(p.file_path) for p in after) == ["b.png", "c.png"]
    assert not any(p.description == "keepme" for p in after), (
        "an unchanged file shares the key, so the arrival cannot be attributed "
        "to the removal; re-import rather than guess"
    )


def test_ambiguous_pixel_match_is_not_followed(server, tmp_path):
    """Two copies moved at once must not have their rows paired by guess."""
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    # Same colour, so both files carry the same pixel hash.
    first = _make_image(os.path.join(folder_dir, "copy_a.png"), (10, 20, 30))
    second = _make_image(os.path.join(folder_dir, "copy_b.png"), (10, 20, 30))

    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    assert sorted(os.path.basename(p.file_path) for p in indexed) == [
        "copy_a.png",
        "copy_b.png",
    ]
    _mark(server, indexed[0].id, "keepme")

    sub_dir = os.path.join(folder_dir, "sub")
    os.makedirs(sub_dir, exist_ok=True)
    os.rename(first, os.path.join(sub_dir, "copy_a.png"))
    os.rename(second, os.path.join(sub_dir, "copy_b.png"))
    _run_scan(server, folder_id, folder_dir)

    after = _pictures(server, folder_dir)
    assert len(after) == 2
    assert not any(p.description == "keepme" for p in after), (
        "a 2:2 pixel match is ambiguous; the scan must re-import rather than "
        "bind one picture's row to the other's file"
    )


def test_a_move_in_a_laid_out_folder_is_queued_for_reconciliation(server, tmp_path):
    """v1.11 Phase 5: a laid-out root's move is a fact worth reconciling.

    The queue only needs the file to have moved and the root to have a
    layout - classifying what it means happens later, live, in
    move_reconciliation_service (see tests/test_library_layout.py).
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, layout=format_layout(DEFAULT_LAYOUT))
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id

    moved_to = os.path.join(folder_dir, "Client Nordvik", "mira.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    result = _run_scan(server, folder_id, folder_dir)

    assert result["external_moved_picture_ids"] == [picture_id]
    assert result["external_moves_queued_for_review"] == [picture_id]

    rows = server.vault.db.run_task(
        lambda s: s.exec(
            select(ExternalMoveReview).where(
                ExternalMoveReview.picture_id == picture_id
            )
        ).all()
    )
    assert len(rows) == 1
    assert rows[0].old_path == original
    assert rows[0].new_path == moved_to


def test_a_move_in_a_folder_without_a_layout_is_not_queued(server, tmp_path):
    """The move is still followed; there is just nothing to reconcile it against."""
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id

    moved_to = os.path.join(folder_dir, "Client Nordvik", "mira.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    result = _run_scan(server, folder_id, folder_dir)

    assert result["external_moved_picture_ids"] == [picture_id]
    assert result["external_moves_queued_for_review"] == []

    rows = server.vault.db.run_task(
        lambda s: s.exec(
            select(ExternalMoveReview).where(
                ExternalMoveReview.picture_id == picture_id
            )
        ).all()
    )
    assert rows == []


def _make_project(server, name):
    def _insert(session: Session):
        project = Project(name=name)
        session.add(project)
        session.commit()
        session.refresh(project)
        return project.id

    return server.vault.db.run_task(_insert)


def _assign_project(server, picture_id, project_id):
    def _update(session: Session):
        pic = session.get(Picture, picture_id)
        pic.project_id = project_id
        session.add(pic)
        session.add(PictureProjectMember(picture_id=picture_id, project_id=project_id))
        session.commit()

    server.vault.db.run_task(_update)


def _pending_review(server, picture_id):
    return server.vault.db.run_task(
        lambda s: s.exec(
            select(ExternalMoveReview).where(
                ExternalMoveReview.picture_id == picture_id
            )
        ).first()
    )


def test_applying_an_unambiguous_move_swaps_the_project_and_clears_the_queue(
    server, tmp_path
):
    """The end-to-end path: scan queues it, apply reconciles it, undo is recorded."""
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, layout=format_layout(DEFAULT_LAYOUT))
    old_project = _make_project(server, "2024 Shoots")
    new_project = _make_project(server, "Client Nordvik")
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _assign_project(server, picture_id, old_project)

    moved_to = os.path.join(folder_dir, "Client Nordvik", "mira.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    _run_scan(server, folder_id, folder_dir)

    review = _pending_review(server, picture_id)
    assert review is not None

    result = move_reconciliation_service.apply_reviews(server.vault, [review.id])
    assert result["applied_picture_ids"] == [picture_id]

    picture = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert picture.project_id == new_project

    memberships = server.vault.db.run_task(
        lambda s: sorted(
            m.project_id
            for m in s.exec(
                select(PictureProjectMember).where(
                    PictureProjectMember.picture_id == picture_id
                )
            ).all()
        )
    )
    assert memberships == [new_project]
    assert _pending_review(server, picture_id) is None, "the row must clear either way"

    operations = server.vault.db.run_task(
        lambda s: s.exec(
            select(Operation).where(
                Operation.op_type
                == move_reconciliation_service.OP_EXTERNAL_MOVE_RECONCILE
            )
        ).all()
    )
    assert len(operations) == 1, (
        "an applied reconciliation must be one undoable operation"
    )


def test_dismissing_a_move_clears_the_queue_without_changing_membership(
    server, tmp_path
):
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, layout=format_layout(DEFAULT_LAYOUT))
    old_project = _make_project(server, "2024 Shoots (dismiss)")
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots (dismiss)", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _assign_project(server, picture_id, old_project)

    moved_to = os.path.join(folder_dir, "Client Nordvik (dismiss)", "mira.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    _run_scan(server, folder_id, folder_dir)

    review = _pending_review(server, picture_id)
    assert review is not None

    result = move_reconciliation_service.dismiss_reviews(server.vault, [review.id])
    assert result["dismissed_review_ids"] == [review.id]

    picture = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert picture.project_id == old_project, "dismissing must not touch any assignment"
    assert _pending_review(server, picture_id) is None


def test_the_http_routes_round_trip_an_unambiguous_move(server, tmp_path):
    """GET /moves/pending, POST /moves/apply - the wiring, not the classification.

    Everything about *what* gets classified is tested at the service level
    above; this is the one place that proves the route table, the response
    models and the authz gate actually agree on the shape.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, layout=format_layout(DEFAULT_LAYOUT))
    old_project = _make_project(server, "2024 Shoots (http)")
    new_project = _make_project(server, "Client Nordvik (http)")
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots (http)", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _assign_project(server, picture_id, old_project)

    moved_to = os.path.join(folder_dir, "Client Nordvik (http)", "mira.png")
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    _run_scan(server, folder_id, folder_dir)

    client = TestClient(server.api)
    login = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert login.status_code == 200, login.text

    pending = client.get("/api/v1/moves/pending")
    assert pending.status_code == 200, pending.text
    body = pending.json()
    review_ids = [
        item["review_id"]
        for item in body["unambiguous"]
        if item["picture_id"] == picture_id
    ]
    assert review_ids, body

    applied = client.post("/api/v1/moves/apply", json={"review_ids": review_ids})
    assert applied.status_code == 200, applied.text
    assert applied.json()["applied_picture_ids"] == [picture_id]

    picture = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert picture.project_id == new_project

    dismiss_empty = client.post("/api/v1/moves/dismiss", json={"review_ids": []})
    assert dismiss_empty.status_code == 200
    assert dismiss_empty.json()["dismissed_review_ids"] == []


def test_applying_a_move_to_a_non_unique_set_name_is_skipped_not_guessed(
    server, tmp_path
):
    """Set names are not DB-unique - a collision must be refused, not applied.

    The refusal happens at entity resolution, after the picture already left
    the queue by review_id, so the caller has to be able to tell "applied
    nothing because it was refused" apart from "applied nothing because there
    was nothing to apply".
    """

    def _make_duplicate_sets(session):
        first = PictureSet(name="Summer (skip)")
        second = PictureSet(name="Summer (skip)")
        session.add_all([first, second])
        session.commit()

    server.vault.db.run_task(_make_duplicate_sets)

    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, layout=format_layout(DEFAULT_LAYOUT))
    project_id = _make_project(server, "2024 Shoots (skip)")
    original = _make_image(
        os.path.join(folder_dir, "2024 Shoots (skip)", "mira.png"), (10, 20, 30)
    )
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _assign_project(server, picture_id, project_id)

    moved_to = os.path.join(
        folder_dir, "2024 Shoots (skip)", "Summer (skip)", "mira.png"
    )
    os.makedirs(os.path.dirname(moved_to), exist_ok=True)
    os.rename(original, moved_to)
    _run_scan(server, folder_id, folder_dir)

    review = _pending_review(server, picture_id)
    assert review is not None

    result = move_reconciliation_service.apply_reviews(server.vault, [review.id])
    assert result["applied_picture_ids"] == []
    assert result["skipped_review_ids"] == [review.id]

    members = server.vault.db.run_task(
        lambda s: s.exec(
            select(PictureSetMember).where(PictureSetMember.picture_id == picture_id)
        ).all()
    )
    assert members == [], "a non-unique name must never be guessed"
    assert _pending_review(server, picture_id) is None, "the row must clear either way"


def test_a_reference_folder_cannot_be_given_a_layout_over_http(server, tmp_path):
    """v1.11 ships the field disarmed.

    Arming it turns the automatic mover loose inside a tree the owner curates
    by hand, with no UI, no preview and no migration route in front of it. The
    Phase 5 machinery behind it is still exercised at the service level above -
    those tests insert the layout directly - so refusing here disarms the
    reachable surface without giving up the coverage.
    """
    folder_dir = str(tmp_path / "refs-layout")
    folder_id = _make_folder(server, folder_dir)

    client = TestClient(server.api)
    login = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert login.status_code == 200, login.text

    armed = client.patch(
        f"/api/v1/reference-folders/{folder_id}",
        json={"layout": format_layout(DEFAULT_LAYOUT)},
    )
    assert armed.status_code == 400, armed.text
    assert "curate yourself" in armed.json()["detail"]

    # Turning it off stays available - that is the default and the way back.
    disarmed = client.patch(
        f"/api/v1/reference-folders/{folder_id}", json={"layout": None}
    )
    assert disarmed.status_code == 200, disarmed.text

    row = server.vault.db.run_task(lambda s: s.get(ReferenceFolder, folder_id))
    assert row.layout is None

    # An unrelated edit is not collateral damage.
    relabelled = client.patch(
        f"/api/v1/reference-folders/{folder_id}", json={"label": "still editable"}
    )
    assert relabelled.status_code == 200, relabelled.text


def test_removing_a_folder_deletes_its_pictures_and_keeps_its_files(server, tmp_path):
    """DELETE /reference-folders/{id} deletes the rows, never the files (#2355).

    The pictures go for good, curation included, not to the Scrapheap. A second
    folder's picture is the control: removal is scoped to the folder named.
    """
    doomed_dir = str(tmp_path / "refs-removed")
    kept_dir = str(tmp_path / "refs-kept")
    doomed_folder = _make_folder(server, doomed_dir)
    kept_folder = _make_folder(server, kept_dir)
    doomed_file = _make_image(os.path.join(doomed_dir, "gone.png"), (10, 20, 30))
    _make_image(os.path.join(kept_dir, "kept.png"), (200, 100, 50))
    _run_scan(server, doomed_folder, doomed_dir)
    _run_scan(server, kept_folder, kept_dir)
    doomed_id = _pictures(server, doomed_dir)[0].id
    kept_id = _pictures(server, kept_dir)[0].id
    _tag(server, doomed_id, "curated")

    def _add_to_set(session: Session):
        picture_set = PictureSet(name="Removed folder set")
        session.add(picture_set)
        session.commit()
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=doomed_id))
        session.commit()

    server.vault.db.run_task(_add_to_set)

    client = TestClient(server.api)
    login = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert login.status_code == 200, login.text
    removed = client.delete(f"/api/v1/reference-folders/{doomed_folder}")
    assert removed.status_code == 200, removed.text

    assert server.vault.db.run_task(lambda s: s.get(Picture, doomed_id)) is None, (
        "the row must be deleted outright, not kept or sent to the Scrapheap"
    )
    assert _tags(server, doomed_id) == []
    members = server.vault.db.run_task(
        lambda s: s.exec(
            select(PictureSetMember).where(PictureSetMember.picture_id == doomed_id)
        ).all()
    )
    assert members == []
    assert (
        server.vault.db.run_task(lambda s: s.get(ReferenceFolder, doomed_folder))
        is None
    )
    assert os.path.exists(doomed_file), "the owner's file must stay on disk"

    assert server.vault.db.run_task(lambda s: s.get(Picture, kept_id)) is not None


def test_a_sidecar_export_is_not_recorded_on_a_row_that_moved_since(server, tmp_path):
    """Sidecar bookkeeping decided at one path is not written onto a moved row.

    The scan exports a tags sidecar beside the file it read and records the
    sidecar's path on the row later, by id. If the row was moved in between,
    recording it would tie the picture at its new path to a sidecar beside a
    file it no longer is - and the next scan would read that file's tags back
    onto it. A row that changed is left for the next scan, and is not reported
    as updated either.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, sync_tags=True)
    original = _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _tag(server, picture_id, "mira")
    newest = os.path.join(folder_dir, "final", "final_042.png")

    def _move_on():
        os.makedirs(os.path.dirname(newest))
        os.rename(original, newest)
        _repoint(server, picture_id, newest)

    database = _WriteBetween(server.vault.db, "apply_caption_updates", _move_on)
    result = _run_scan(server, folder_id, folder_dir, database=database)

    assert database.fired, "the move must land inside the window under test"
    row = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert row.file_path == newest
    assert (row.tags_file, row.tags_file_mtime) == (None, None), (
        "the sidecar beside the old path must not be recorded on the moved row"
    )
    assert _tags(server, picture_id) == ["mira"]
    assert result["caption_updated_picture_ids"] == [], result
    assert result["caption_updated_count"] == 0, result


def test_a_moved_file_gets_its_sidecar_recorded_in_the_same_scan(server, tmp_path):
    """The scan that follows a move also records the sidecar it exports there.

    The sidecar pass runs after the move is applied and checks each row against
    what it read. The row it walks has to carry the path the move stored, or
    the check sees its own move as another writer's and leaves the row for the
    next scan: the sidecar is on disk beside the file, the row does not know it.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, sync_tags=True)
    original = _make_image(os.path.join(folder_dir, "lune_007.png"), (91, 92, 93))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _tag(server, picture_id, "lune")
    moved_to = os.path.join(folder_dir, "final", "final_007.png")
    os.makedirs(os.path.dirname(moved_to))
    os.rename(original, moved_to)

    result = _run_scan(server, folder_id, folder_dir)

    row = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert row.file_path == moved_to
    assert row.tags_file is not None, "the sidecar exported beside it is recorded"
    assert os.path.dirname(row.tags_file) == os.path.dirname(moved_to)
    assert os.path.isfile(row.tags_file)
    assert result["caption_updated_picture_ids"] == [picture_id], result


def test_a_locked_picture_is_not_counted_as_updated(server, tmp_path):
    """A picture a locked set freezes keeps its data and is not reported.

    The sidecar pass skips a locked picture rather than write to it. Reporting
    it anyway tells an open grid to refresh a picture that did not change, and
    the count in the scan's result overstates what the scan did.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir, sync_tags=True)
    _make_image(os.path.join(folder_dir, "mira_042.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)
    picture_id = _pictures(server, folder_dir)[0].id
    _tag(server, picture_id, "mira")

    def _lock(session: Session):
        picture_set = PictureSet(name="Locked sidecar set", locked=True)
        session.add(picture_set)
        session.commit()
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()
        return picture_set.id

    def _unlock(session: Session, set_id: int):
        session.get(PictureSet, set_id).locked = False
        session.commit()

    set_id = server.vault.db.run_task(_lock)
    try:
        result = _run_scan(server, folder_id, folder_dir)
    finally:
        server.vault.db.run_task(_unlock, set_id)

    row = server.vault.db.run_task(lambda s: s.get(Picture, picture_id))
    assert row.tags_file is None, "a locked picture's row must not be written"
    assert result["caption_updated_picture_ids"] == [], result
    assert result["caption_updated_count"] == 0, result


def test_a_junction_in_a_reference_folder_is_not_walked_and_not_read_as_emptied(
    server, tmp_path, monkeypatch
):
    """A Windows junction is a link: the scan neither indexes what it leads to
    nor reads it as a subtree the owner emptied.

    ``os.walk`` does not descend a directory symlink but does descend a
    junction, and ``islink`` says False for one. Simulated on Linux by giving
    the directory's ``lstat`` the junction reparse tag. Wrong if the file added
    behind the junction is indexed, or if the picture indexed there before it
    became one loses its row.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    inner = os.path.join(folder_dir, "linked")
    _make_image(os.path.join(inner, "before.png"), (31, 41, 59))
    _run_scan(server, folder_id, folder_dir)
    assert [os.path.basename(p.file_path) for p in _pictures(server, folder_dir)] == [
        "before.png"
    ]

    real_lstat = os.lstat

    class _JunctionStat:
        def __init__(self, info):
            self._info = info
            self.st_reparse_tag = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT

        def __getattr__(self, name):
            return getattr(self._info, name)

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        return _JunctionStat(info) if os.fspath(path) == inner else info

    monkeypatch.setattr(os, "lstat", lstat)
    _make_image(os.path.join(inner, "behind.png"), (26, 53, 58))
    _run_scan(server, folder_id, folder_dir)

    assert [os.path.basename(p.file_path) for p in _pictures(server, folder_dir)] == [
        "before.png"
    ]
