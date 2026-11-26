"""Finder for pictures whose thumbnail has not been generated.

Keyed on ``Picture.thumbnail_width IS NULL``. New imports populate the thumbnail
columns at import time, so this finder normally only picks up rows an upgrade
reset to NULL, driving a one-time regeneration of the whole-frame AR bitmap
through :class:`ThumbnailGenerationTask`.

A picture the task could not make a thumbnail for keeps its NULL and is selected
again. The registry holds the ones whose cause outlasts a sweep (undecodable,
unreachable, unreadable). One more cause is held here: a file that decodes but
whose EXIF turn raises (see
``ThumbnailGenerationTask._note_orientation_failure``), which no registry kind
fits, because only the stages that transpose fail on it.
"""

import threading
from typing import Collection

from sqlmodel import Session, select
from sqlalchemy.orm import load_only, selectinload

from pixlstash.db_models import Face, Picture
from pixlstash.pixl_logging import get_logger
from pixlstash.utils.sql_chunking import not_in_json
from pixlstash.utils.unprocessable_image_registry import file_signature

from .base_task_finder import BaseTaskFinder
from .pipeline_scope import pipeline_picture_clauses
from .task_outcome import RetryHold, id_sample
from .thumbnail_generation_task import ThumbnailGenerationTask

logger = get_logger(__name__)


class MissingThumbnailFinder(BaseTaskFinder):
    """Find pictures missing a generated thumbnail and create a generation task.

    A picture the task reports under
    :attr:`ThumbnailGenerationTask.ORIENTATION_FAILED` is held in a
    :class:`RetryHold` (capped, jittered backoff, never permanent) together with
    the file signature the task took before decoding it. The hold is released
    the moment that signature moves, by the registry's own rule
    (:func:`file_signature`): a rewritten or repaired file is retried on the next
    sweep, an unchanged one at most once per backoff interval.
    """

    def __init__(self, database, notifier=None, *, retry_hold: RetryHold | None = None):
        """
        Args:
            database: The vault database.
            notifier: Announces regenerated thumbnails to open clients.
            retry_hold: Per-picture backoff; a default one when omitted.
        """
        super().__init__()
        self._db = database
        self._notifier = notifier
        self._retry_hold = retry_hold if retry_hold is not None else RetryHold()
        # picture id -> (path, mtime_ns, size) the hold is pinned to.
        self._held_files: dict[int, tuple[str, int, int]] = {}
        self._held_lock = threading.Lock()

    def finder_name(self) -> str:
        return "MissingThumbnailFinder"

    def max_inflight_tasks(self) -> int:
        return 1

    def find_task(self):
        limit = ThumbnailGenerationTask.BATCH_SIZE * (
            max(1, self.max_inflight_tasks()) + 1
        )
        # Exclude undecodable and held pictures (issue #585) at the query so
        # they cannot crowd the candidate window and stall real work behind
        # them. Both re-stat files, so both run before the read is opened.
        suppressed_ids = self._held_picture_ids() | self._suppressed_picture_ids()
        pictures = self._db.run_immediate_read_task(
            lambda session: self._fetch_missing(session, limit, suppressed_ids)
        )
        if not pictures:
            return None

        selected = self._drop_held(
            self._filter_and_claim(pictures, ThumbnailGenerationTask.BATCH_SIZE)
        )
        if not selected:
            return None

        return ThumbnailGenerationTask(
            database=self._db, pictures=selected, notifier=self._notifier
        )

    def on_task_complete(self, task, error) -> None:
        """Hold what the task could not orient, then release the batch.

        The holds are recorded before the claims are released: the planner may
        run ``find_task`` the moment a claim is let go, and a picture released
        before its hold exists would be selected straight back. In a
        ``finally``, so a malformed result cannot strand the batch's claims.
        """
        try:
            self._settle(task, error)
        finally:
            super().on_task_complete(task, error)

    def _held_picture_ids(self) -> set[int]:
        """The held pictures whose file has not changed since the hold.

        A held picture whose file's signature moved (rewritten, repaired,
        replaced, or gone) is released here, so it is selected on this very
        sweep. One ``stat`` per active hold, and only for those.
        """
        active = self._retry_hold.active()
        if not active:
            return set()
        with self._held_lock:
            watched = {
                pid: self._held_files[pid] for pid in active if pid in self._held_files
            }
        changed = [
            pid
            for pid, (path, mtime_ns, size) in watched.items()
            if file_signature(path) != (mtime_ns, size)
        ]
        for pid in changed:
            self._release(pid)
        if changed:
            logger.info(
                "MissingThumbnailFinder: %d held pictures changed on disk and are "
                "retried: %r",
                len(changed),
                id_sample(changed),
            )
        return active - set(changed)

    def _drop_held(self, selected: list) -> list:
        """*selected* without the pictures held since the candidates were read.

        A task completing between the read and the claim holds its pictures and
        then releases their claims, so the claim check alone lets them through.
        Their new claims are released here.
        """
        held = self._retry_hold.active()
        raced = {pic.id for pic in selected if pic.id in held}
        if not raced:
            return selected
        with self._claim_lock:
            self._claimed_picture_ids.difference_update(raced)
        return [pic for pic in selected if pic.id not in raced]

    def _release(self, picture_id: int) -> None:
        """Forget *picture_id*'s hold and the file signature it was pinned to."""
        self._retry_hold.clear(picture_id)
        with self._held_lock:
            self._held_files.pop(picture_id, None)

    def _settle(self, task, error) -> None:
        """Hold the pictures *task* could not orient; release the rest of its batch."""
        if error is not None:
            # It failed as a whole, or was stopped or never ran (the runner
            # settles both as a cancellation): nothing was learnt about the
            # pictures, and their holds stay as they were.
            return
        result = getattr(task, "result", None)
        if not isinstance(result, dict):
            logger.error(
                "MissingThumbnailFinder: task %s returned %r, not a result dict; "
                "the holds of its pictures are left as they were",
                getattr(task, "id", None),
                result,
            )
            return
        # The key is present only when a picture failed this way.
        failed = {
            int(pid): (str(path), int(mtime_ns), int(size))
            for pid, path, mtime_ns, size in result.get(
                ThumbnailGenerationTask.ORIENTATION_FAILED, ()
            )
        }
        picture_ids = (getattr(task, "params", None) or {}).get("picture_ids") or []
        for pid in picture_ids:
            if pid not in failed:
                self._release(pid)
        if not failed:
            return
        waits = []
        for pid, pinned in failed.items():
            # The signature first: `_held_picture_ids` reads it for every
            # active hold.
            with self._held_lock:
                self._held_files[pid] = pinned
            waits.append(self._retry_hold.hold(pid))
        logger.warning(
            "MissingThumbnailFinder: %d pictures decode but their EXIF "
            "orientation cannot be read; held for %.0f-%.0f s, or until the "
            "file changes: %r",
            len(waits),
            min(waits),
            max(waits),
            id_sample(failed),
        )

    @staticmethod
    def missing_thumbnail_clauses() -> tuple:
        """WHERE terms for the pictures still owed a thumbnail: this finder's
        and the progress count's (``Vault._count_missing_thumbnails``), so the
        two count the same pictures."""
        return (
            Picture.thumbnail_width.is_(None),
            *pipeline_picture_clauses(deleted_index=True),
        )

    @staticmethod
    def _fetch_missing(
        session: Session, limit: int, suppressed_ids: Collection[int] = frozenset()
    ):
        stmt = (
            select(Picture)
            .where(*MissingThumbnailFinder.missing_thumbnail_clauses())
            .where(not_in_json(Picture.id, suppressed_ids))
        )
        return session.exec(
            stmt.options(
                # This probe runs on every planning sweep, so it must never drag
                # the LargeBinary columns (image_embedding / text_embedding /
                # likeness_parameters) or Face.features off disk. Narrow it to
                # exactly what ThumbnailGenerationTask reads from a DETACHED
                # picture: ``pic.id`` (task params + claim filter), ``pic.file_path``
                # (source resolution + thumbnail write path), and each face's
                # ``bbox`` (the square-crop weighting), which is a Python property
                # over the ``bbox_`` column. ``Face.picture_id`` is the selectin
                # correlation column and must not be deferred. Anything else read
                # downstream would raise DetachedInstanceError, because
                # ``run_immediate_read_task`` closes the session before the task runs.
                load_only(Picture.id, Picture.file_path),
                selectinload(Picture.faces).load_only(Face.bbox_, Face.picture_id),
            )
            .order_by(Picture.id)
            .limit(limit)
        ).all()
