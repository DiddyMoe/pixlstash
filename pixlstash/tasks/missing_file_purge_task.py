import os
import time
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models import DeletedFileLog, Picture
from pixlstash.db_models.picture_move import RETENTION_S, PictureMove
from pixlstash.pixl_logging import get_logger
from pixlstash.services.move_reconciliation_service import (
    record_pending_reviews_in_laid_out_roots,
)
from pixlstash.services.scrapheap_service import file_location_is_unreachable
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.sql_chunking import chunked

logger = get_logger(__name__)

# A vanished file's row is deleted only on a look at the disk at most this old
# when the delete runs. The look cannot be made on the writer (a stat can block
# as long as a dropped share likes, and the writer holds every write in the
# library), and the filesystem and the database share no transaction, so the
# check and the delete cannot be one atomic step; what can be bounded is the
# time between them (CWE-367's mitigation). The writer refuses a row whose look
# is older, and the caller looks again. A file put back within this window of
# the commit is the residual race, caught by the look after the delete. Shared
# by this sweep and the reference-folder scan's ``_delete_vanished``.
REMOVAL_CHECK_MAX_AGE_S = 1.0

# Looks (and delete attempts) per vanished row in one pass. A row the writer was
# too busy to delete in time on every round is kept for the next pass.
REMOVAL_ROUNDS = 3


def file_is_present(path: str) -> bool:
    """Whether a picture's file is present, as the reference-folder scan sees it.

    The one meaning of "present" for every check that decides whether a row is
    deleted over a missing file: this sweep's and the scan's. ``lexists``, not
    ``isfile``, because the scan's walk lists a symlinked *file* in ``files``,
    a dangling one included, so the scan keeps the row of a picture whose path
    is a dangling link. A dangling link usually points into a drive or share
    that is not mounted right now, and a purge writes ``file_removed=True``,
    which bars the path from restore and from routine re-import: following the
    link here would purge exactly the rows the scan keeps.
    """
    return os.path.lexists(path)


class MissingFilePurgeTask(BaseTask):
    """Low-priority task that removes picture records whose files no longer exist.

    For each picture in the batch whose file is absent from the filesystem
    (:func:`file_is_present`) the task:

    1. Inserts a row into ``deleted_file_log`` (if one does not already exist
       for that ``file_path``).
    2. Deletes the picture row from the ``picture`` table.

    The task operates at ``TaskPriority.LOW`` so it never starves active work.
    The delete itself follows the reference-folder scan's rule: the disk is
    looked at off the writer, and the delete is conditional on that look being
    fresh (see :meth:`_purge`).
    """

    BATCH_SIZE = 250

    def __init__(self, database, pictures: list):
        """Initialise the task.

        Args:
            database: The application database instance.
            pictures: Sequence of picture rows returned by the finder (must
                expose ``id``, ``file_path``, and ``pixel_sha`` attributes).
        """
        picture_ids = [p.id for p in (pictures or []) if getattr(p, "id", None)]
        super().__init__(
            task_type="MissingFilePurgeTask",
            params={"picture_ids": picture_ids, "batch_size": len(picture_ids)},
        )
        self._db = database
        self._pictures = pictures or []

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    def _run_task(self):
        image_root = self._db.image_root
        missing = []
        on_disk: dict = {}
        unreachable = 0
        for pic in self._pictures:
            if not pic.file_path:
                continue
            try:
                resolved = ImageUtils.resolve_picture_path(image_root, pic.file_path)
                if file_is_present(resolved):
                    continue
                if file_location_is_unreachable(resolved):
                    # The file is absent because its LOCATION is absent - an
                    # unmounted reference folder or a dropped network share -
                    # not because it was deleted. Purging here would delete a
                    # live picture's row AND write file_removed=True, so a later
                    # restore would refuse to bring it back: permanent loss of a
                    # file that never went anywhere. Skip until the volume is
                    # back; the next pass will re-examine it.
                    unreachable += 1
                    logger.warning(
                        "MissingFilePurgeTask: SKIPPING picture %s - its "
                        "location is unreachable (parent directory missing at "
                        "%s; unmounted reference folder or network vault?). Not "
                        "purging, because the file may still exist.",
                        pic.id,
                        resolved,
                    )
                    continue
                missing.append(pic)
                on_disk[pic.id] = resolved
            except Exception as exc:
                logger.debug(
                    "MissingFilePurgeTask: could not resolve path for picture %s: %s",
                    pic.id,
                    exc,
                )

        if unreachable:
            logger.warning(
                "MissingFilePurgeTask: %s picture(s) in this batch live at "
                "unreachable locations and were left untouched.",
                unreachable,
            )

        if not missing:
            return {"purged": 0, "repaired": 0, "deferred": 0}

        repairs, deferred, missing = self._separate_our_own_moves(missing)

        repaired = 0
        if repairs:
            repaired = self._db.run_task(
                self._repair_moved_pictures,
                repairs,
                image_root,
                priority=DBPriority.LOW,
            )
            logger.info(
                "MissingFilePurgeTask: repaired %s picture(s) the layout engine "
                "had moved but not finished recording; none were purged.",
                repaired,
            )

        if not missing:
            return {"purged": 0, "repaired": repaired, "deferred": deferred}

        logger.info(
            "MissingFilePurgeTask: found %s missing file(s) in batch of %s - purging.",
            len(missing),
            len(self._pictures),
        )

        purged = self._purge(missing, on_disk)
        logger.info("MissingFilePurgeTask: purged %s picture record(s).", purged)
        return {"purged": purged, "repaired": repaired, "deferred": deferred}

    def _separate_our_own_moves(self, candidates: list):
        """Split off pictures PixlStash moved itself but never finished recording.

        The move engine commits a ``picture_move`` row naming both paths
        **before** it renames anything, precisely so this sweep can tell "the
        owner deleted it" from "we moved it and the process died before the row
        was repointed". Purging the latter is unrecoverable: the picture row
        goes and its tags, sets, faces and score go with it, and the
        ``DeletedFileLog`` written alongside carries ``file_removed=True``,
        which :mod:`full_restore` reads as *never resurrect* - so not even a
        snapshot brings it back.

        **The journal is read in both directions**, because a move has two ends
        and a crash can strand a file at either. A row is a record that this
        file belongs at one of two paths: if the row's *other* end holds a file,
        that is where it is, whether the interrupted operation was the move
        (``old_path`` is the row's, the file is at ``new_path``) or the undo of
        one (``new_path`` is the row's, the file is back at ``old_path``).

        Three outcomes per candidate:

        * a journal row whose other end **is** a file on disk - repaired here,
          the row repointed, nothing lost;
        * a journal row whose other end is not - left alone for a later pass,
          because an operation still in flight must not be read as a deletion.
          The journal expires after ``RETENTION_S``, so this defers rather than
          exempts;
        * no journal row at all - a genuine deletion, purged as before.

        Returns ``(repairs, deferred_count, still_missing)``, each repair a
        ``(picture_id, landed_path, came_back_to_our_source, seen_path)``
        tuple - the flag says which end of the journal row the file turned up
        at, which is what :meth:`_repair_moved_pictures` needs to know whether
        Phase 5 is owed a review as well as a repoint, and ``seen_path`` is the
        row's path as read here, the end the journal row was matched against.
        """
        by_id: dict = {pic.id: pic for pic in candidates if pic.file_path}
        if not by_id:
            return [], 0, candidates

        rows = self._db.run_immediate_read_task(self._load_move_journal, list(by_id))
        if not rows:
            return [], 0, candidates

        # The far end of the newest move recorded for each picture. Matched on
        # the picture id *and* the path: rows outlive their move by
        # ``RETENTION_S``, and a later picture that reuses the path must not be
        # repointed at wherever the earlier one went.
        other_end: dict = {}
        came_back: set = set()
        stamped: dict = {}
        for row in rows:
            pic = by_id.get(row.picture_id)
            if pic is None:
                continue
            for here, there in (
                (row.old_path, row.new_path),
                (row.new_path, row.old_path),
            ):
                if here != pic.file_path:
                    continue
                if pic.id in stamped and stamped[pic.id] >= row.moved_at:
                    continue
                stamped[pic.id] = row.moved_at
                other_end[pic.id] = there
                # Which end the file is at decides who moved it. At new_path it
                # is our own move finishing; at old_path the file has come back
                # to where we took it from, which the scan would have read as
                # the owner's doing. See _repair_moved_pictures.
                came_back.discard(pic.id)
                if there == row.old_path:
                    came_back.add(pic.id)

        image_root = self._db.image_root
        repairs: list = []
        deferred = 0
        still_missing: list = []
        for pic in candidates:
            landed_path = other_end.get(pic.id)
            if landed_path is None:
                still_missing.append(pic)
                continue
            try:
                landed = ImageUtils.resolve_picture_path(image_root, landed_path)
            except Exception as exc:
                logger.warning(
                    "MissingFilePurgeTask: picture %s at %s has a move journal "
                    "row pointing at %s, which could not be resolved (%s). "
                    "Leaving it alone rather than purging.",
                    pic.id,
                    pic.file_path,
                    landed_path,
                    exc,
                )
                deferred += 1
                continue
            if file_is_present(landed):
                repairs.append(
                    (pic.id, landed_path, pic.id in came_back, pic.file_path)
                )
                continue
            deferred += 1
            logger.warning(
                "MissingFilePurgeTask: SKIPPING picture %s at %s - PixlStash "
                "recorded a move between that path and %s, and neither holds a "
                "file. Not purging: an operation still in flight must not be "
                "read as a deletion. The journal row expires on its own if it "
                "never completes.",
                pic.id,
                pic.file_path,
                landed_path,
            )
        return repairs, deferred, still_missing

    @staticmethod
    def _load_move_journal(session: Session, picture_ids: list) -> list:
        """Load unexpired journal rows for any of *picture_ids*."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=RETENTION_S)
        rows: list = []
        for chunk in chunked(picture_ids):
            rows.extend(
                session.exec(
                    select(PictureMove)
                    .where(PictureMove.picture_id.in_(list(chunk)))
                    .where(PictureMove.moved_at >= cutoff)
                ).all()
            )
        return rows

    @staticmethod
    def _repair_moved_pictures(session: Session, repairs: list, image_root=None) -> int:
        """Repoint each row at the path its file actually reached.

        **Repointing is only half of what the discovery is worth**, and which
        half depends on the direction. A file found at the journal row's
        ``new_path`` is our own move landing, and the reference-folder scan
        would have claimed that pair as ours and queued nothing - repointing is
        the whole of it.

        A file found back at the row's ``old_path`` is the other case, and the
        journal cannot tell its two causes apart: an undo whose rename landed
        and whose transaction did not, or the owner dragging the file back out
        of the folder PixlStash filed it in. The scan can: ``claim_own_moves``
        matches a pair in one direction only, so it would find the reversed pair
        unclaimed, read it as the owner's, and queue an ``ExternalMoveReview``
        (v1.11 Phase 5). Whichever of the two sweeps notices first used to
        decide whether that ever happened - and once this one has repointed the
        row, the file is where the row says it is, so the next scan sees no move
        to follow and the reconciliation is lost for good. So it is queued here
        too, from the same fact and through the same recorder. A crashed undo
        queues one as well, which is the safe direction: the row is a question
        for the owner, never an applied change, and finishing the undo is what
        it would have asked anyway.

        **A repair is only applied to a row still at the path it was decided
        on.** The journal row was matched against the path read before this
        write was queued; a row repointed since by another writer is somewhere
        newer, and writing the journal's end over it would point the picture at
        a file that may have moved on. It is left for the next pass.
        """
        repaired = 0
        came_back: list = []
        for picture_id, new_path, came_back_to_our_source, seen_path in repairs:
            picture = session.get(Picture, picture_id)
            if picture is None:
                continue
            if picture.file_path != seen_path:
                logger.info(
                    "MissingFilePurgeTask: not repointing picture %s to %s - it "
                    "moved from %s to %s after this pass read it.",
                    picture_id,
                    new_path,
                    seen_path,
                    picture.file_path,
                )
                continue
            logger.info(
                "MissingFilePurgeTask: repointing picture %s from %s to %s - "
                "PixlStash moved this file and did not finish recording it.",
                picture_id,
                picture.file_path,
                new_path,
            )
            if came_back_to_our_source and picture.file_path:
                came_back.append((picture_id, picture.file_path, new_path))
            picture.file_path = new_path
            session.add(picture)
            repaired += 1
        if came_back:
            queued = record_pending_reviews_in_laid_out_roots(
                session, came_back, image_root
            )
            if queued:
                logger.info(
                    "MissingFilePurgeTask: queued %s move(s) for reconciliation - "
                    "the file is back at the path PixlStash moved it from, which "
                    "the folder scan would have read as the owner's own move.",
                    len(queued),
                )
        session.commit()
        return repaired

    def _purge(self, pictures: list, on_disk: dict) -> int:
        """Purge the rows of *pictures* whose files are still gone; return the count.

        Check-then-act across two systems with no shared transaction, as the
        reference-folder scan's ``_delete_vanished``: the disk must never be
        looked at on the writer, and the delete must not rest on an old look.
        So each file is looked at here, off the writer, and the look stamped;
        :meth:`_purge_pictures` refuses a row whose look is older than
        :data:`REMOVAL_CHECK_MAX_AGE_S` when it runs, and such a row is looked
        at again, up to :data:`REMOVAL_ROUNDS` times, then kept for the next
        pass. Looks are sent in groups flushed before their oldest look is
        half the limit old.

        The delete is HIGH, although the task is LOW: the condition is a
        deadline, and LOW waits behind every import and thumbnail write. Each
        is one short transaction over at most :attr:`BATCH_SIZE` rows.

        After the delete each path is looked at once more (CWE-367's "recheck
        after use"): a file put back inside the window has lost its row, which
        is said in a warning.

        Args:
            pictures: The rows as read, each found missing by :meth:`_run_task`.
            on_disk: Picture id to the path :meth:`_run_task` resolved for it.
        """
        purged: list[int] = []
        pending = [pic for pic in pictures if pic.id is not None]
        by_id = {pic.id: pic for pic in pending}

        def flush(batch: list[tuple]) -> list[int]:
            deleted, stale = self._db.run_task(
                self._purge_pictures, batch, priority=DBPriority.HIGH
            )
            purged.extend(deleted)
            for pic_id in deleted:
                if file_is_present(on_disk[pic_id]):
                    logger.warning(
                        "MissingFilePurgeTask: %s came back while its record "
                        "(picture %s) was being purged. The deletion ledger now "
                        "lists the path, so a routine folder scan will not "
                        "import it again; re-adding the folder does.",
                        on_disk[pic_id],
                        pic_id,
                    )
            return stale

        for _ in range(REMOVAL_ROUNDS):
            if not pending:
                break
            stale_ids: list[int] = []
            batch: list[tuple] = []
            for pic in pending:
                if file_is_present(on_disk[pic.id]):
                    logger.info(
                        "MissingFilePurgeTask: not purging picture %s - its file "
                        "is back at %s.",
                        pic.id,
                        pic.file_path,
                    )
                    continue
                batch.append((pic, time.monotonic()))
                if time.monotonic() - batch[0][1] >= REMOVAL_CHECK_MAX_AGE_S / 2:
                    stale_ids.extend(flush(batch))
                    batch = []
            if batch:
                stale_ids.extend(flush(batch))
            pending = [by_id[pic_id] for pic_id in stale_ids]
        if pending:
            logger.info(
                "MissingFilePurgeTask: %d missing picture(s) were not purged: the "
                "database was too busy to act on a look at the disk within %.1f s, "
                "%d times. Keeping their records for the next pass.",
                len(pending),
                REMOVAL_CHECK_MAX_AGE_S,
                REMOVAL_ROUNDS,
            )
        return len(purged)

    @staticmethod
    def _purge_pictures(
        session: Session, decisions: list[tuple]
    ) -> tuple[list[int], list[int]]:
        """Delete picture rows and log them to deleted_file_log.

        Each decision is ``(picture, checked_at)``: the row as read before this
        write was queued, and the ``time.monotonic()`` of the look that found
        its file gone. One that another writer has since repointed, re-owned or
        scrapheaped may name a live file now, so it is neither deleted nor
        logged - the ledger entry would bar its old path from ever being
        imported again - and is left for the next pass to judge as it is. One
        whose look is older than :data:`REMOVAL_CHECK_MAX_AGE_S` is returned as
        stale, for :meth:`_purge` to look again: the disk is never touched from
        here. Returns ``(purged_ids, stale_ids)``.
        """
        purged: list[int] = []
        stale: list[int] = []
        written: list[int] = []
        oldest = None
        now = datetime.now(timezone.utc)

        for pic, checked_at in decisions:
            db_pic = session.get(Picture, pic.id)
            if db_pic is not None and (
                db_pic.file_path,
                db_pic.reference_folder_id,
                db_pic.deleted,
            ) != (pic.file_path, pic.reference_folder_id, pic.deleted):
                logger.info(
                    "MissingFilePurgeTask: not purging picture %s - it changed "
                    "after this pass read it (now at %s).",
                    pic.id,
                    db_pic.file_path,
                )
                continue
            if time.monotonic() - checked_at > REMOVAL_CHECK_MAX_AGE_S:
                stale.append(pic.id)
                continue
            written.append(pic.id)
            oldest = checked_at if oldest is None else min(oldest, checked_at)
            # Log the deletion only if not already recorded. The path is stored
            # as an opaque hash (path_sha), never in cleartext.
            if pic.file_path:
                path_sha = DeletedFileLog.hash_path(pic.file_path)
                already_logged = session.exec(
                    select(DeletedFileLog).where(DeletedFileLog.path_sha == path_sha)
                ).first()
                if not already_logged:
                    session.add(
                        DeletedFileLog(
                            path_sha=path_sha,
                            pixel_sha=pic.pixel_sha,
                            deleted_at=now,
                            # The file is genuinely gone from disk (that is why
                            # it is being purged), so this is a real permanent
                            # deletion restore must never resurrect.
                            file_removed=True,
                        )
                    )
                elif not already_logged.file_removed:
                    # The path was first logged as file_removed=False (removed
                    # from the library but the file was deliberately kept). The
                    # file has now genuinely vanished, so upgrade the stale flag
                    # to True instead of skipping - the ledger must be truthful,
                    # not merely rely on restore's separate missing-file net.
                    # Only ever raise False -> True; never downgrade.
                    already_logged.file_removed = True
                    session.add(already_logged)

            if db_pic is not None:
                session.delete(db_pic)
                purged.append(pic.id)

        if written:
            # Flushed first, so the age is judged with the writes (and their
            # cascades) already run: only the COMMIT follows.
            session.flush()
            if time.monotonic() - oldest > REMOVAL_CHECK_MAX_AGE_S:
                session.rollback()
                return [], stale + written
        session.commit()
        return purged, stale
