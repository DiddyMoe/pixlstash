"""Backfill ``Picture.pixel_sha`` - the tier-1 exact-duplicate key.

Every current import path computes ``pixel_sha`` as the file is written, so this
task exists for the rows that predate it (and for rows whose file was replaced
under a reference folder). Tier 1 of the duplicate queue is a ``GROUP BY`` on
that indexed column, so a NULL there is a duplicate the queue can never see -
which is the one failure mode that makes the sidebar count untrustworthy.

The digest is ``ImageUtils.calculate_hash_from_file_path``, i.e. exactly what the
import paths write, so a backfilled row and a freshly imported row of the same
file agree.
"""

import time
from typing import Collection

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models.picture import Picture
from pixlstash.pixl_logging import get_logger
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.sql_chunking import not_in_json

logger = get_logger(__name__)


class PixelShaTask(BaseTask):
    """Compute ``pixel_sha`` for a batch of pictures that have none."""

    BATCH_SIZE = 64
    SCAN_LIMIT = 512

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    def __init__(self, database, pictures: list):
        picture_ids = [pic.id for pic in (pictures or []) if getattr(pic, "id", None)]
        super().__init__(
            task_type="PixelShaTask",
            params={"picture_ids": picture_ids, "batch_size": len(picture_ids)},
        )
        self._db = database
        self._pictures = pictures or []

    def _run_task(self):
        start = time.time()
        updates: list[tuple[int, str]] = []
        unreadable: list[int] = []
        for pic in self._pictures:
            file_path = ImageUtils.resolve_picture_path(
                self._db.image_root, pic.file_path
            )
            try:
                digest = ImageUtils.calculate_hash_from_file_path(str(file_path))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "pixel_sha computation failed for picture_id=%s path=%s: %s. "
                    "It is not tried again until PixlStash restarts, and stays "
                    "invisible to tier-1 exact duplicate detection until then.",
                    pic.id,
                    file_path,
                    exc,
                )
                unreadable.append(int(pic.id))
                continue
            updates.append((int(pic.id), digest))

        changed = []
        if updates:
            changed = (
                self._db.run_task(
                    PixelShaTask._persist_pixel_shas, updates, priority=DBPriority.LOW
                )
                or []
            )
        logger.debug(
            "PixelShaTask completed in %.2fs with %s update(s)",
            time.time() - start,
            len(changed),
        )
        # ``deferred`` even when nothing was hashed: a batch where no file
        # could be read is the one the finder must not hand out again.
        return {
            "changed_count": len(changed),
            "changed": changed,
            "deferred": unreadable,
        }

    @staticmethod
    def _persist_pixel_shas(session: Session, updates: list[tuple[int, str]]) -> list:
        """Write the digests in one transaction."""
        changed = []
        for picture_id, digest in updates:
            pic = session.get(Picture, picture_id)
            if pic is None:
                continue
            pic.pixel_sha = digest
            session.add(pic)
            changed.append((Picture, picture_id, "pixel_sha", digest))
        session.commit()
        return changed

    @staticmethod
    def find_pictures_missing_pixel_sha(
        session: Session, limit: int, exclude_ids: Collection[int] = frozenset()
    ) -> list:
        """Return non-deleted pictures with a file but no ``pixel_sha``.

        *exclude_ids* are left out ahead of the ``LIMIT``.
        """
        return session.exec(
            select(Picture)
            .where(Picture.pixel_sha.is_(None))
            .where(Picture.deleted.is_(False))
            .where(Picture.file_path.is_not(None))
            .where(not_in_json(Picture.id, exclude_ids))
            .limit(limit)
        ).all()
