from sqlmodel import Session

from .base_task_finder import BaseTaskFinder
from .likeness_parameters_task import LikenessParametersTask
from .likeness_task import LikenessTask
from pixlstash.database import DBPriority
from pixlstash.utils.likeness.likeness_utils import BulkCandidateArrays
from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)


class MissingLikenessFinder(BaseTaskFinder):
    """Find pending likeness work and create a LikenessTask.

    Caches the decoded bulk candidate arrays (``BulkCandidateArrays``) for the
    duration of each sweep.  With 28k pictures × 2 KB embeddings the cache
    eliminates ~57 MB of repeated SQLite reads and blob deserialization from
    every task cycle.  The cache is invalidated when the queue drains to zero
    (sweep complete) so a fresh DB state is used for the next sweep.
    """

    def __init__(self, database):
        super().__init__()
        self._db = database
        self._bulk_cache: BulkCandidateArrays | None = None
        # Whether this finder has already offered the first-sweep seed. See
        # ``find_task``: without it a library whose candidates yield no pair
        # would be "pending" forever.
        self._seed_offered = False

    def finder_name(self) -> str:
        return "MissingLikenessFinder"

    def find_task(self):
        # Block pairs until *all* findable parameter work is done: the count
        # the parameter stage's progress row shows, so a picture that stage
        # will never take (in the Scrapheap, or unprocessable and without a
        # quality score) cannot hold this one back.
        suppressed_ids = self._suppressed_picture_ids()
        pending = self._db.run_immediate_read_task(
            LikenessParametersTask.count_pending, suppressed_ids
        )
        if pending:
            return None

        queue_count, candidate_count, pair_count = self._db.run_immediate_read_task(
            self._likeness_state
        )
        logger.debug(
            "MissingLikenessFinder: queue=%d candidates=%d pairs=%d",
            int(queue_count or 0),
            int(candidate_count or 0),
            int(pair_count or 0),
        )

        # Require at least one fully-ready candidate (embedding + parameters +
        # phash all set).  Queue count alone is not sufficient because
        # ImageEmbeddingTask enqueues pictures immediately after writing their
        # embeddings, before likeness_parameters or perceptual_hash are ready.
        # Without this guard, the finder spams "building bulk candidate cache…"
        # every planning cycle whenever newly-enqueued pictures lack the other
        # two fields (build_bulk_arrays returns None → cache stays None → loop).
        #
        # Work is a queued candidate. An empty likeness table with nothing
        # queued is ALSO work, once: it is a library's first sweep, which
        # ``seed_queue`` fills. Only once per finder, because a sweep may write
        # no pair at all (a single candidate has nothing to pair with, and the
        # gap thresholds can reject every pair of a few dissimilar ones), and
        # "no pairs" would then read as "not swept yet" on every cycle: a
        # LikenessTask over an empty queue, forever. Every later change that
        # needs likeness (new embeddings, a parameter reset) enqueues its
        # pictures, so the queue alone carries the rest.
        first_sweep = int(pair_count or 0) == 0 and not self._seed_offered
        has_work = int(candidate_count or 0) > 0 and (
            int(queue_count or 0) > 0 or first_sweep
        )
        if not has_work:
            # Sweep done - drop the cache so the next sweep gets fresh data.
            self._bulk_cache = None
            return None

        # Build (or reuse) the bulk candidate array cache.
        if self._bulk_cache is None:
            # Seed the queue once at the start of a new sweep.
            self._db.result_or_throw(
                self._db.submit_task(LikenessTask.seed_queue, priority=DBPriority.LOW)
            )
            self._seed_offered = True
            logger.info("MissingLikenessFinder: building bulk candidate cache…")
            self._bulk_cache = self._db.run_immediate_read_task(
                LikenessTask.build_bulk_arrays
            )
            if self._bulk_cache is None:
                return None
            logger.info(
                "MissingLikenessFinder: cache built (%d candidates)",
                len(self._bulk_cache.ids),
            )

        return LikenessTask(database=self._db, bulk_arrays=self._bulk_cache)

    @staticmethod
    def _likeness_state(session: Session):
        return (
            LikenessTask.count_queue(session),
            LikenessTask.count_total_candidates(session),
            LikenessTask.count_total_pairs(session),
        )
