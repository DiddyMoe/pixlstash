"""Finder for pictures missing the tier-1 exact-duplicate hash."""

from typing import Collection

from pixlstash.task_runner import TaskCancelledError
from pixlstash.tasks.base_task_finder import BaseDeferringFinder, SimpleMissingFinder
from pixlstash.tasks.pixel_sha_task import PixelShaTask


class MissingPixelShaFinder(SimpleMissingFinder, BaseDeferringFinder):
    """Queue :class:`PixelShaTask` for pictures whose ``pixel_sha`` is NULL.

    Tier 1 of the duplicate queue groups on ``picture.pixel_sha``; a NULL there
    is a duplicate nobody can find. Reading a file to digest it is cheap next to
    the embedding pipeline, so this runs at one task in flight and gets out of
    the way.

    A picture whose file cannot be read is deferred for the rest of the session
    (:class:`BaseDeferringFinder`): a duplicate scan waits until this finder
    runs dry, so one picture handed out on every sweep would keep it waiting.
    """

    _IDS_PARAM = "picture_ids"
    _TRANSIENT = (TaskCancelledError,)
    _WHAT = "Hashing pictures for exact duplicates"

    def finder_name(self) -> str:
        return "MissingPixelShaFinder"

    def max_inflight_tasks(self) -> int:
        return 1

    def _batch_size(self) -> int:
        return PixelShaTask.BATCH_SIZE

    def _fetch_candidates(
        self, session, limit: int, suppressed_ids: Collection[int] = frozenset()
    ) -> list:
        return PixelShaTask.find_pictures_missing_pixel_sha(
            session, limit, set(suppressed_ids) | self._deferred
        )

    def _create_task(self, pictures: list):
        return PixelShaTask(database=self._db, pictures=pictures)
