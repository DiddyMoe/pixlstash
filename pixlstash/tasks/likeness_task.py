from sqlalchemy import func
from sqlmodel import Session, delete, select

from pixlstash.database import DBPriority
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_likeness import PictureLikeness, PictureLikenessQueue
from pixlstash.utils.likeness.likeness_utils import LikenessUtils, BulkCandidateArrays
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.tasks.pipeline_scope import pipeline_picture_clauses
from pixlstash.pixl_logging import get_logger
from pixlstash.utils.service.scope_table import scope_id_subquery

logger = get_logger(__name__)


class LikenessTask(BaseTask):
    """Task that processes one likeness queue scoring cycle."""

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    def __init__(self, database, bulk_arrays: BulkCandidateArrays | None = None):
        super().__init__(
            task_type="LikenessTask",
            params={},
        )
        self._db = database
        self._bulk_arrays = bulk_arrays

    def _run_task(self):
        helper = LikenessUtils(self._db)

        def submit_low(func, *args, **kwargs):
            return self._db.result_or_throw(
                self._db.submit_task(func, *args, priority=DBPriority.LOW, **kwargs)
            )

        work_items = submit_low(
            LikenessTask.fetch_work_batch,
            helper.MAX_A_PER_CYCLE,
        )
        if not work_items:
            return {"changed_count": 0, "changed": [], "pairs_written": 0}

        queued_ids = [int(item[0]) for item in work_items]

        # Use the cached pre-decoded arrays when available (set by the finder),
        # falling back to a fresh fetch only if the task was created without them.
        bulk = self._bulk_arrays
        if bulk is None:
            bulk = self._db.run_immediate_read_task(LikenessTask.build_bulk_arrays)

        if bulk is None:
            return {"changed_count": 0, "changed": [], "pairs_written": 0}

        logger.info(
            "LikenessTask: processing %d queued pictures against %d candidates",
            len(queued_ids),
            len(bulk.ids),
        )
        likeness_results = helper.compute_bulk_likeness(queued_ids, bulk)
        logger.info(
            "LikenessTask: computed %d likeness pairs from %d queued pictures",
            len(likeness_results),
            len(queued_ids),
        )

        if likeness_results:
            submit_low(
                LikenessUtils.write_results,
                likeness_results,
                helper.TOP_K,
            )

        changed = [(PictureLikenessQueue, pid, "queue", None) for pid in queued_ids]
        return {
            "changed_count": len(changed),
            "changed": changed,
            "pairs_written": len(likeness_results),
        }

    @staticmethod
    def candidate_clauses() -> tuple:
        """WHERE terms for the pictures likeness compares: embedding, likeness
        parameters and perceptual hash all set, and the picture one the
        pipeline works on. The progress row's total counts these, and its
        remaining counts the queued ones among them."""
        return (
            Picture.image_embedding.is_not(None),
            Picture.likeness_parameters.is_not(None),
            Picture.perceptual_hash.is_not(None),
            *pipeline_picture_clauses(),
        )

    @staticmethod
    def fetch_work_batch(session: Session, max_a: int) -> list[tuple]:
        """Take up to *max_a* queued candidates, oldest first, and delete
        their queue rows.

        Only candidates are taken (``candidate_clauses``): a queued picture in
        the Scrapheap, without a file or not yet ready keeps its queue row and
        is not compared, the same pictures ``count_queue`` leaves out of
        "remaining". Each item is ``(picture_id, None, None, phash, width,
        height, size_bytes, likeness_parameters, created_at, embedding)``.
        """
        rows = session.exec(
            select(
                PictureLikenessQueue.picture_id,
                Picture.perceptual_hash,
                Picture.width,
                Picture.height,
                Picture.size_bytes,
                Picture.likeness_parameters,
                Picture.created_at,
                Picture.image_embedding,
            )
            .join(Picture, Picture.id == PictureLikenessQueue.picture_id)
            .where(*LikenessTask.candidate_clauses())
            .order_by(PictureLikenessQueue.queued_at)
            .limit(max_a)
        ).all()
        if not rows:
            return []

        queued_ids = [int(row[0]) for row in rows]
        session.exec(
            delete(PictureLikenessQueue).where(
                PictureLikenessQueue.picture_id.in_(
                    scope_id_subquery(
                        session, queued_ids, name="_pixlstash_likeness_taken_ids"
                    )
                )
            )
        )
        session.commit()

        return [
            (
                int(pic_id),
                None,
                None,
                phash_a,
                width_a,
                height_a,
                size_a,
                params_blob,
                created_at,
                emb_blob,
            )
            for (
                pic_id,
                phash_a,
                width_a,
                height_a,
                size_a,
                params_blob,
                created_at,
                emb_blob,
            ) in rows
        ]

    @staticmethod
    def build_bulk_arrays(session: Session) -> BulkCandidateArrays | None:
        """Fetch and decode every candidate (``candidate_clauses``) into the
        arrays each queued picture is compared against.

        Called once per sweep by the finder and cached: decoding the 28k
        embedding blobs (2 KB each) takes ~200-500 ms.
        """
        rows = session.exec(
            select(
                Picture.id,
                Picture.likeness_parameters,
                Picture.image_embedding,
                Picture.perceptual_hash,
            ).where(*LikenessTask.candidate_clauses())
        ).all()
        return LikenessUtils.decode_bulk_rows(rows)

    @staticmethod
    def seed_queue(session: Session) -> None:
        """Queue every picture the pipeline works on (live, with a file) when
        the queue and the likeness table are both empty: a library's first
        sweep. A Scrapheap picture is not queued."""
        queued_count = session.exec(
            select(func.count()).select_from(PictureLikenessQueue)
        ).one()
        if queued_count and int(queued_count) > 0:
            return
        likeness_count = session.exec(
            select(func.count()).select_from(PictureLikeness)
        ).one()
        if likeness_count and int(likeness_count) > 0:
            return
        ids = [
            int(pid)
            for pid in session.exec(
                select(Picture.id).where(*pipeline_picture_clauses())
            ).all()
        ]
        logger.debug("LikenessTask: seeding queue with %d pictures", len(ids))
        PictureLikenessQueue.enqueue(session, ids)
        session.commit()

    @staticmethod
    def count_queue(session: Session) -> int:
        """Queued pictures that are candidates: a queued picture in the
        Scrapheap, or not yet ready, is not counted as remaining."""
        result = session.exec(
            select(func.count())
            .select_from(PictureLikenessQueue)
            .join(Picture, Picture.id == PictureLikenessQueue.picture_id)
            .where(*LikenessTask.candidate_clauses())
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    @staticmethod
    def count_total_candidates(session: Session) -> int:
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(*LikenessTask.candidate_clauses())
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    @staticmethod
    def count_total_pairs(session: Session) -> int:
        result = session.exec(select(func.count()).select_from(PictureLikeness)).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0
