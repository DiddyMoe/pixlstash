from sqlmodel import Session, select

from pixlstash.db_models import Picture
from pixlstash.utils.likeness.likeness_parameter_utils import LikenessParameterUtils
from pixlstash.worker_config import LIKENESS_PARAMETERS_MAX_INFLIGHT

from .base_task_finder import BaseTaskFinder
from .likeness_parameters_task import LikenessParametersTask
from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)


class MissingLikenessParametersFinder(BaseTaskFinder):
    """Discover pending likeness-parameter work and create pre-fetched tasks.

    The finder does all the work: SQL scan, claim-filtering, and data
    pre-fetch (quality metrics / picture metadata).  Worker tasks receive
    ready-to-write payloads and never touch the serialised DB queue for reads.
    """

    def __init__(self, database):
        super().__init__()
        self._db = database

    def finder_name(self) -> str:
        return "MissingLikenessParametersFinder"

    def max_inflight_tasks(self) -> int:
        return LIKENESS_PARAMETERS_MAX_INFLIGHT

    def find_task(self):
        # Quality metrics are likeness parameters (brightness, contrast, ...),
        # and each picture's row is written once, with its real values: the
        # candidate query only offers a picture whose own Quality row is done
        # (_fetch_pending), so no stage-wide gate is needed and a picture flows
        # here the moment its quality is in.
        #
        # Discover work and pre-fetch all required data inside a single
        # immediate read session - no write-queue involvement.
        work = self._db.run_immediate_read_task(
            self._find_and_prefetch,
            self._db.image_root,
        )
        if work is None:
            return None

        ids, payload = work
        if not ids:
            return None

        return LikenessParametersTask(
            database=self._db,
            ids=ids,
            payload=payload,
        )

    @staticmethod
    def _fetch_pending(session: Session, limit: int) -> list[tuple[int, int, int]]:
        """Up to *limit* ``(id, width, height)`` of the pending pictures whose
        quality is done, in id order: the task's pending clauses
        (``LikenessParametersTask.pending_parameter_clauses``, the ones the
        progress row counts) and the quality wait. ``ix_picture_size_bin_index``
        drives the probe.

        Each picture gets its own size bin from its dimensions, so a flat batch
        needs no grouping by resolution.
        """
        rows = session.exec(
            select(Picture.id, Picture.width, Picture.height)
            .where(
                *LikenessParametersTask.pending_parameter_clauses(),
                LikenessParametersTask.quality_done_clause(),
            )
            .order_by(Picture.id)
            .limit(limit)
        ).all()
        return [(int(pid), int(w), int(h)) for pid, w, h in rows]

    def _find_and_prefetch(self, session: Session, image_root: str):
        """Run inside a read session: find next work batch and pre-fetch all data."""
        id_wh_list = self._fetch_pending(session, LikenessParametersTask.SCAN_LIMIT)
        if not id_wh_list:
            return None

        all_ids = [t[0] for t in id_wh_list]
        # Claim-filter: exclude IDs already in-flight with another task.
        ids = self._claim_ids(all_ids[: LikenessParametersTask.BATCH_SIZE])
        if not ids:
            return None

        id_set = set(ids)
        size_bin_by_id = {
            pid: LikenessParameterUtils.size_bin_index(w, h)
            for pid, w, h in id_wh_list
            if pid in id_set
        }
        quality_by_id = LikenessParameterUtils.fetch_quality_for_ids(session, ids)
        picture_by_id, picture_updates = (
            LikenessParameterUtils.fetch_picture_params_for_ids(
                session, ids, image_root
            )
        )
        return ids, {
            "size_bin_by_id": size_bin_by_id,
            "quality_by_id": quality_by_id,
            "picture_by_id": picture_by_id,
            "picture_updates": picture_updates,
        }

    def _claim_ids(self, ids: list) -> list:
        """Claim IDs atomically; return only unclaimed ones."""
        claimed = []
        with self._claim_lock:
            for pid in ids:
                if pid not in self._claimed_picture_ids:
                    self._claimed_picture_ids.add(pid)
                    claimed.append(pid)
        return claimed
