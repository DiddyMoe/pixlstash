from collections.abc import Collection

from sqlalchemy import func, or_
from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models.picture import (
    LikenessParameter,
    Picture,
)
from pixlstash.db_models.quality import Quality
from pixlstash.utils.likeness.likeness_parameter_utils import LikenessParameterUtils
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.tasks.pipeline_scope import pipeline_picture_clauses
from pixlstash.utils.sql_chunking import not_in_json
from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)


class LikenessParametersTask(BaseTask):
    """Write all pre-fetched likeness parameter values to the database in a single pass.

    The finder is responsible for discovering work and fetching all required
    data (quality metrics, picture metadata, size-bin indices).  This task
    computes the final blobs and writes them atomically: one DB update covers
    SIZE_BIN, quality params, and picture params for the entire batch.
    """

    BATCH_SIZE = 1024
    SCAN_LIMIT = 4096

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.MEDIUM

    def __init__(self, database, ids: list, payload: dict):
        """
        Args:
            database: Database instance.
            ids: Picture IDs in this batch.
            payload: Pre-fetched data dict with keys:
                - ``size_bin_by_id``: mapping of id → size_bin_index integer
                - ``quality_by_id``: mapping of id → quality field dict
                - ``picture_by_id``: mapping of id → picture param field dict
                - ``picture_updates``: mapping of id → metadata fields to write back
        """
        super().__init__(
            task_type="LikenessParametersTask",
            params={"picture_ids": ids},
        )
        self._db = database
        self._ids = ids
        self._payload = payload

    @staticmethod
    def pending_parameter_clauses() -> tuple:
        """WHERE terms for the pictures still owed likeness parameters: the
        finder's, the progress count's and the likeness stage's gate, so the
        three count the same pictures.

        All parameters are written at once, so ``size_bin_index IS NULL`` is
        the one pending marker. A picture without width and height can never
        get a size bin and is not pending. ``size_bin_index`` has a plain
        index, hence the ``IS NOT`` spelling of the Scrapheap term.
        """
        return (
            Picture.size_bin_index.is_(None),
            Picture.width.is_not(None),
            Picture.height.is_not(None),
            *pipeline_picture_clauses(),
        )

    @staticmethod
    def quality_done_clause():
        """The picture's own quality row is done: a ``Quality`` row with its
        sharpness set (a failed calculation writes -1.0 and is done too), the
        inverse of ``QualityTask.missing_quality_clauses``. Parameters are
        written once, with real quality values, so the finder offers only such
        pictures; a pending picture still waiting on its score is counted but
        not offered. ``ix_quality_picture_id`` serves the EXISTS."""
        return (
            select(Quality.id)
            .where(Quality.picture_id == Picture.id, Quality.sharpness.is_not(None))
            .exists()
        )

    @staticmethod
    def count_pending(
        session: Session, suppressed_ids: Collection[int] = frozenset()
    ) -> int:
        """How many pictures are still owed likeness parameters that the
        finder will get to, those waiting on their quality score included.

        A picture in *suppressed_ids* (marked unprocessable) whose quality is
        not done never gets it, since the quality finder skips it, so the
        likeness-parameters finder never offers it: it is not counted. One
        whose quality was done before it was marked is still offered and still
        counted."""
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                *LikenessParametersTask.pending_parameter_clauses(),
                or_(
                    LikenessParametersTask.quality_done_clause(),
                    not_in_json(Picture.id, suppressed_ids),
                ),
            )
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _run_task(self):
        def submit_low(func, *args, **kwargs):
            return self._db.result_or_throw(
                self._db.submit_task(func, *args, priority=DBPriority.LOW, **kwargs)
            )

        ids = self._ids
        vector_length = len(LikenessParameter)

        logger.info(
            "LikenessParametersTask: writing all params for %d pictures",
            len(ids),
        )

        # Fetch existing blobs in a concurrent read session - outside the
        # serialised write queue so all worker threads can do this in parallel.
        blobs_by_id = self._db.run_immediate_read_task(
            LikenessParameterUtils.fetch_blobs_for_ids, ids
        )

        # Compute all parameter blobs in the worker thread (parallel CPU).
        updates = LikenessParameterUtils.compute_all_param_updates(
            ids,
            blobs_by_id,
            self._payload["size_bin_by_id"],
            self._payload["quality_by_id"],
            self._payload["picture_by_id"],
            vector_length,
        )

        picture_updates = self._payload.get("picture_updates", {})
        if picture_updates:
            submit_low(LikenessParameterUtils.update_picture_metadata, picture_updates)

        # Single write covering likeness_parameters + size_bin_index for all pictures.
        submit_low(LikenessParameterUtils.write_blob_updates, updates)
        submit_low(LikenessParameterUtils.reset_likeness_for_pictures, ids)

        return {
            "changed_count": len(ids),
            "changed": [(Picture, pid, "likeness_parameters", None) for pid in ids],
        }
