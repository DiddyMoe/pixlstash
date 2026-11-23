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
from pixlstash.tasks.missing_file_purge_task import MissingFilePurgeTask
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


def _make_folder(server, folder_dir, layout=None):
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
    of its own.
    """

    def __init__(self, db, callback_name, before):
        self._db = db
        self._callback_name = callback_name
        self._before = before
        self.fired = False

    def __getattr__(self, name):
        return getattr(self._db, name)

    def run_task(self, func, *args, **kwargs):
        if not self.fired and getattr(func, "__name__", None) == self._callback_name:
            self.fired = True
            self._before()
        return self._db.run_task(func, *args, **kwargs)


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
    """A row whose file is back at its path by the time of the delete is kept.

    The removal rests on the file being absent as well as on the row. A file
    put back at the same path leaves the row exactly as the scan read it, so
    only a fresh look at the disk on the writer shows the decision no longer
    holds.
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
        server.vault.db, "delete_removed", lambda: os.rename(parked, original)
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

    The row is untouched, so only a fresh look at the disk on the writer shows
    the purge no longer holds. Purging anyway would also write a
    ``file_removed=True`` ledger row that keeps the live file out of the
    library for good.
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
        server.vault.db, "_purge_pictures", lambda: os.rename(parked, original)
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


def test_a_scrapheap_row_does_not_swallow_an_unrelated_new_file(server, tmp_path):
    """A soft-deleted row must not claim a new file of the same content.

    ``fetch_existing`` loads ``deleted=True`` rows on purpose, so without the
    exclusion a hidden scrapheap picture whose file really was deleted pairs
    with an unrelated arrival: the new path is taken out of ``new_paths`` as a
    move, and what the user gets is not a new picture but a soft-deleted one
    they cannot see.
    """
    folder_dir = str(tmp_path / "refs")
    folder_id = _make_folder(server, folder_dir)
    doomed = _make_image(os.path.join(folder_dir, "binned.png"), (10, 20, 30))

    _run_scan(server, folder_id, folder_dir)
    indexed = _pictures(server, folder_dir)
    assert [os.path.basename(p.file_path) for p in indexed] == ["binned.png"]

    def _scrapheap(session: Session):
        pic = session.get(Picture, indexed[0].id)
        pic.deleted = True
        session.add(pic)
        session.commit()

    server.vault.db.run_task(_scrapheap)

    # The binned file goes, and an unrelated file of the same content arrives.
    os.remove(doomed)
    _make_image(os.path.join(folder_dir, "fresh.png"), (10, 20, 30))
    _run_scan(server, folder_id, folder_dir)

    live = [p for p in _pictures(server, folder_dir) if not p.deleted]
    assert [os.path.basename(p.file_path) for p in live] == ["fresh.png"], (
        "the arrival is its own picture, not a scrapheap row quietly relabelled"
    )


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
