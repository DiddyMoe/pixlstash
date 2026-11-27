from typing import Callable, Collection

from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

from pixlstash.db_models import (
    Picture,
    is_tag_sentinel,
    parse_tag_engine_from_sentinel,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.task_runner import TaskCancelledError
from pixlstash.utils.query.predicate_filter import awaiting_tagger_clauses
from pixlstash.worker_config import IMAGE_TASK_PIXEL_BUDGET, TAGGER_MAX_INFLIGHT
from .base_task import TaskStatus
from .base_task_finder import BaseTaskFinder
from .tag_task import TagTask
from .task_outcome import RetryHold

logger = get_logger(__name__)


class MissingTagFinder(BaseTaskFinder):
    """Find a batch of pictures with a pending-retag sentinel and create a TagTask.

    A task that fails outright, such as one whose tagger did not load, leaves
    every picture's sentinel, so the next sweep would build the same batch and
    load the tagger again at once. Such a failure holds the finder back on
    :class:`RetryHold`, capped at :attr:`BACKEND_HOLD_CAP_S`; the next batch
    after the hold is the probe, and a task that ends without error clears it.
    A cancelled task holds nothing. In memory: a restart retries at once.
    """

    #: The backend hold's only key.
    _BACKEND = "backend"
    #: Longest wait between two attempts after a failed task, seconds.
    BACKEND_HOLD_CAP_S = 60.0

    def __init__(
        self,
        database,
        engine_getter: Callable,
        *,
        backend_hold: RetryHold | None = None,
    ):
        """
        Args:
            database: The vault database.
            engine_getter: Returns the inference engine, or ``None`` while
                there is none.
            backend_hold: Backoff after a task fails outright; a default one
                capped at :attr:`BACKEND_HOLD_CAP_S` when omitted.
        """
        super().__init__()
        self._db = database
        self._engine_getter = engine_getter
        self._backend_hold = (
            backend_hold
            if backend_hold is not None
            else RetryHold(cap_s=self.BACKEND_HOLD_CAP_S)
        )

    def finder_name(self) -> str:
        return "MissingTagFinder"

    def max_inflight_tasks(self) -> int:
        return TAGGER_MAX_INFLIGHT

    # No `on_all_tasks_complete`: the drain used to tear the WD14 session down,
    # and with per-row dependencies a finder drains and refills many times per
    # pass, so that was a reload storm. The session is bounded by its own
    # `gpu_mem_limit` now, and `Vault._maybe_aggressive_unload` frees it once
    # every worker has been idle - only under `keep_models_in_memory=False`.

    def find_task(self):
        engine = self._engine_getter()
        if engine is None:
            return None
        # Only queue tagging when an active tag plugin is configured.
        tagger_settings = getattr(engine, "tagger_settings", None)
        active_tag_plugin = (tagger_settings or {}).get("active_tag_plugin")
        if not active_tag_plugin:
            return None
        if self._BACKEND in self._backend_hold.active():
            return None

        batch_limit = max(
            1,
            int(engine.tagging_workflow.suggested_task_size()),
        )
        # Fetch enough candidates that _filter_and_claim can always fill one
        # additional task even when all max_inflight slots are already in-flight.
        max_inflight = max(1, self.max_inflight_tasks())
        # Exclude undecodable pictures at the query, as the face, thumbnail and
        # embedding finders already do. `_filter_and_claim` drops them anyway,
        # but only after they have taken up room in the candidate window - and
        # an undecodable picture keeps its pending-tag sentinel for ever
        # (TagTask marks it unprocessable and writes nothing, so nothing
        # clears the sentinel). They therefore accumulate permanently, and
        # this window is ordered by `Picture.id`: once as few as
        # `batch_limit * (max_inflight + 1)` of them sit below the pictures
        # that still need tagging, every sweep reads the same corrupt rows,
        # claims none of them and returns None - which the planner reads as
        # "no work" and answers with a growing backoff. That is tagging
        # starving library-wide on a handful of bad files, the same shape
        # `MissingFaceExtractionFinder` documents on its own query.
        suppressed_ids = self._suppressed_picture_ids()
        # Before the read: a reset landing between the two must make the task
        # older than the reset, never newer (#1361).
        reset_generation = self._db.tag_resets.current()
        pictures = self._db.run_immediate_read_task(
            lambda session: self._fetch_missing_tags(
                session, batch_limit * (max_inflight + 1), suppressed_ids
            )
        )
        if not pictures:
            return None

        # Group pictures by their requested engine (None = use active_tag_plugin).
        groups: dict[str | None, list] = {}
        for pic in pictures:
            sentinel_tag = next(
                (t.tag for t in pic.tags if is_tag_sentinel(t.tag)),
                None,
            )
            engine_name = parse_tag_engine_from_sentinel(sentinel_tag)
            groups.setdefault(engine_name, []).append(pic)

        # Process the first group only (subsequent calls will handle the rest).
        first_engine, first_pics = next(iter(groups.items()))
        selected = self._filter_and_claim(
            first_pics, batch_limit, pixel_budget=IMAGE_TASK_PIXEL_BUDGET
        )
        if not selected:
            return None

        return TagTask(
            database=self._db,
            tagging_workflow=engine.tagging_workflow,
            pictures=selected,
            engine_override=first_engine,
            reset_generation=reset_generation,
        )

    def on_task_complete(self, task, error) -> None:
        """Hold the finder back if *task* failed, then release the batch.

        The hold is recorded before the claims are released: the planner may
        run ``find_task`` the moment a claim is let go, and a batch built
        before the hold exists would load the failing tagger straight away.
        """
        try:
            self._settle_backend(task, error)
        finally:
            super().on_task_complete(task, error)

    def _settle_backend(self, task, error) -> None:
        """Hold or clear the backend hold from how *task* ended."""
        if isinstance(error, TaskCancelledError) or (
            error is None and getattr(task, "status", None) == TaskStatus.CANCELLED
        ):
            # Stopped, or cancelled before it started (which reports no
            # error, only its status): nothing was learnt about the tagger.
            return
        if error is None:
            self._backend_hold.clear(self._BACKEND)
            return
        # Up to TAGGER_MAX_INFLIGHT tasks fail together when the tagger
        # cannot load; only the first of them lengthens the hold.
        if self._BACKEND in self._backend_hold.active():
            return
        wait = self._backend_hold.hold(self._BACKEND)
        logger.warning(
            "MissingTagFinder: a tagging task failed: %s. No new batch for "
            "%.0f s; its pictures keep their place in the queue.",
            error,
            wait,
        )

    @staticmethod
    def _fetch_missing_tags(
        session: Session, limit: int, suppressed_ids: Collection[int] = frozenset()
    ):
        # Per-row stage dependency: tag a picture once ITS faces are known, not
        # once the whole face stage has drained. Face extraction always writes
        # at least one Face row (a ``face_index=-1`` sentinel when there is no
        # face), so ``faces.any()`` is exactly "face extraction has run". The
        # quality crop in TagTask only *prefers* a face (it centre-crops
        # without one), but cropping on the face is the shipped behaviour, so
        # the picture waits for it.
        #
        # The wait is bounded because `MissingFaceExtractionFinder` selects on
        # the exact complement, `~Picture.faces.any()`: a picture with no row
        # is by definition still face-stage work, so a cancelled or failed
        # batch is re-offered rather than stranded.
        faces_known = Picture.faces.any()
        stmt = select(Picture).where(
            *awaiting_tagger_clauses(suppressed_ids),
            faces_known,
        )
        return session.exec(
            stmt.options(
                selectinload(Picture.tags),
            )
            .order_by(Picture.id)
            .limit(limit)
        ).all()
