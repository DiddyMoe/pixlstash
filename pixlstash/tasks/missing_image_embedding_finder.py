import threading
from collections import Counter
from typing import Callable

from .base_task_finder import BaseTaskFinder
from .image_embedding_task import ImageEmbeddingTask
from .task_outcome import Reason, RetryHold, id_sample, read_outcomes
from pixlstash.task_runner import TaskCancelledError
from pixlstash.worker_config import (
    IMAGE_EMBEDDING_MAX_INFLIGHT,
    IMAGE_TASK_PIXEL_BUDGET,
)

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)


class MissingImageEmbeddingFinder(BaseTaskFinder):
    """Find pending image embedding work and create an ImageEmbeddingTask.

    Reads every task's ``outcomes`` (see :mod:`pixlstash.tasks.task_outcome`)
    so that a picture that failed is not selected again on the next sweep:

    - ``FAILED_RETRYABLE`` holds the picture back on a capped, jittered
      exponential backoff (:class:`RetryHold`), so one that keeps failing is
      retried more and more rarely, never marked broken.
    - ``FAILED_RETRYABLE`` because CLIP itself is unavailable holds the whole
      finder instead: no batch is built and nothing is decoded until the hold
      expires, and the next batch is the probe.
    - ``FAILED_PERMANENT`` is the registry's to suppress; ``DEFERRED`` was not
      attempted and stays eligible.
    - ``SUCCEEDED`` clears the picture's hold and the backend hold.

    A task that failed outright holds all of its pictures; a cancelled one
    holds none. The holds live in memory: a restart retries everything.

    A task can complete while some of its decodes are still running (it waits
    for its preload only so long, and a decode cannot be interrupted). Until
    they return no new batch is built, so a file that hangs its decode costs
    at most one task's decode threads, never a new set per sweep.
    """

    #: The backend hold's only key.
    _BACKEND = "backend"
    #: Longest wait between two probes of an unavailable CLIP backend, seconds.
    BACKEND_HOLD_CAP_S = 60.0

    def __init__(
        self,
        database,
        engine_getter: Callable,
        *,
        retry_hold: RetryHold | None = None,
        backend_hold: RetryHold | None = None,
    ):
        """
        Args:
            database: The vault database.
            engine_getter: Returns the inference engine, or ``None`` while
                there is none.
            retry_hold: Per-picture backoff; a default one when omitted.
            backend_hold: Backoff for an unavailable CLIP backend; a default
                one capped at :attr:`BACKEND_HOLD_CAP_S` when omitted.
        """
        super().__init__()
        self._db = database
        self._engine_getter = engine_getter
        self._retry_hold = retry_hold if retry_hold is not None else RetryHold()
        self._backend_hold = (
            backend_hold
            if backend_hold is not None
            else RetryHold(cap_s=self.BACKEND_HOLD_CAP_S)
        )
        # {task id: picture ids} of the decodes that outlived their task.
        self._decoding: dict[str, frozenset] = {}
        self._decoding_lock = threading.Lock()

    def finder_name(self) -> str:
        return "MissingImageEmbeddingFinder"

    def max_inflight_tasks(self) -> int:
        return IMAGE_EMBEDDING_MAX_INFLIGHT

    def find_task(self):
        engine = self._engine_getter()
        if engine is None:
            return None
        if self._BACKEND in self._backend_hold.active():
            return None
        with self._decoding_lock:
            if self._decoding:
                return None

        batch_size = ImageEmbeddingTask.BATCH_SIZE
        try:
            batch_size = max(
                1, int(engine.clip_embedding_workflow.suggested_batch_size())
            )
        except Exception:
            logger.warning(
                "clip_embedding_workflow.suggested_batch_size() failed, using default batch size",
                exc_info=True,
            )

        # Fetch more than one task worth so _filter_and_claim can skip claimed IDs.
        # Exclude undecodable pictures (issue #585) at the query so they cannot
        # crowd the candidate window and stall real work, and the pictures held
        # back after a retryable failure for the same reason.
        excluded_ids = (
            set(self._db.unprocessable_images.active_suppressed_ids())
            | self._retry_hold.active()
        )
        candidates = self._db.run_immediate_read_task(
            lambda session: ImageEmbeddingTask.fetch_work(
                session=session,
                limit=batch_size * IMAGE_EMBEDDING_MAX_INFLIGHT,
                suppressed_ids=excluded_ids,
            )
        )
        if not candidates:
            return None

        selected = self._filter_and_claim(
            candidates, batch_size, pixel_budget=IMAGE_TASK_PIXEL_BUDGET
        )
        selected = self._drop_held(selected)
        if not selected:
            return None
        # A task that completed during the query may have left a decode
        # running; a batch built now would start more decodes behind it.
        with self._decoding_lock:
            decoding = bool(self._decoding)
        if decoding:
            with self._claim_lock:
                self._claimed_picture_ids.difference_update(row.id for row in selected)
            return None

        return ImageEmbeddingTask(
            database=self._db,
            clip_workflow=engine.clip_embedding_workflow,
            batch=[(row.id, row.file_path) for row in selected],
        )

    def on_task_complete(self, task, error) -> None:
        """Hold back what failed and may be retried, then release the batch.

        The holds, and the wait for any decode still running, are recorded
        before the claims are released: the planner may run ``find_task`` the
        moment a claim is let go, and a picture released before its hold
        exists would be selected straight back. In a ``finally``, so a
        malformed result cannot strand the batch's claims.
        """
        try:
            self._wait_for_running_decodes(task)
            self._settle(task, error)
        finally:
            super().on_task_complete(task, error)

    def _wait_for_running_decodes(self, task) -> None:
        """Build no batch until the decodes *task* left running have returned."""
        task_id = task.id
        # Held across finish_preload, so the preload thread's callback cannot
        # run before the decodes it reports are recorded.
        with self._decoding_lock:
            running = task.finish_preload(lambda: self._decodes_returned(task_id))
            if running:
                self._decoding[task_id] = frozenset(running)
        if running:
            logger.warning(
                "MissingImageEmbeddingFinder: task %s completed with %d decodes "
                "still running; no new batch until they return: %r",
                task_id,
                len(running),
                id_sample(running),
            )

    def _decodes_returned(self, task_id: str) -> None:
        """Called from task *task_id*'s preload thread when it has stopped."""
        with self._decoding_lock:
            returned = self._decoding.pop(task_id, frozenset())
            waiting = len(self._decoding)
        logger.info(
            "MissingImageEmbeddingFinder: the %d decodes task %s left running "
            "have returned; %d tasks still have decodes running",
            len(returned),
            task_id,
            waiting,
        )

    def _drop_held(self, selected: list) -> list:
        """*selected* without the pictures held since ``fetch_work`` ran.

        A task completing between the query and the claim holds its failed
        pictures and then releases their claims, so the claim check alone lets
        them through. Their new claims are released here.
        """
        held = self._retry_hold.active()
        raced = {row.id for row in selected if row.id in held}
        if not raced:
            return selected
        with self._claim_lock:
            self._claimed_picture_ids.difference_update(raced)
        return [row for row in selected if row.id not in raced]

    def _settle(self, task, error) -> None:
        """Record what *task*'s outcomes say about its pictures' holds."""
        picture_ids = (getattr(task, "params", None) or {}).get("picture_ids") or []
        if isinstance(error, TaskCancelledError):
            # It never ran, or was stopped (the runner settles both as a
            # cancellation): nothing was learnt about the pictures, and they
            # stay eligible.
            return
        if error is not None:
            self._hold(picture_ids, f"the task failed: {error}")
            return

        try:
            summary = read_outcomes(getattr(task, "result", None))
        except ValueError as exc:
            summary = None
            logger.error(
                "MissingImageEmbeddingFinder: task %s reported an outcome that "
                "is not one: %s",
                getattr(task, "id", None),
                exc,
            )
        if summary is None:
            # The task always reports outcomes, so this is a bug in it. Its
            # pictures are not known to have succeeded; holding them keeps the
            # bug from turning into a loop.
            logger.error(
                "MissingImageEmbeddingFinder: task %s returned no outcomes for "
                "its %d pictures",
                getattr(task, "id", None),
                len(picture_ids),
            )
            self._hold(picture_ids, "the task reported no outcomes")
            return

        for pid in summary.succeeded | summary.failed_permanent:
            self._retry_hold.clear(pid)
        if summary.succeeded:
            self._backend_hold.clear(self._BACKEND)

        backend_down = {
            pid
            for pid in summary.failed_retryable
            if summary.reasons.get(pid) is Reason.BACKEND_UNAVAILABLE
        }
        # Up to IMAGE_EMBEDDING_MAX_INFLIGHT tasks fail together when the
        # backend goes; one hold covers them all, and only the first of them
        # lengthens it.
        if backend_down and self._BACKEND not in self._backend_hold.active():
            wait = self._backend_hold.hold(self._BACKEND)
            logger.warning(
                "MissingImageEmbeddingFinder: CLIP is unavailable; no new batch "
                "for %.0f s (%d pictures wait for it)",
                wait,
                len(backend_down),
            )

        failed = summary.failed_retryable - backend_down
        if failed:
            reasons = Counter(
                summary.reasons[pid].value if pid in summary.reasons else "unspecified"
                for pid in failed
            )
            self._hold(
                failed, ", ".join(f"{r}={n}" for r, n in sorted(reasons.items()))
            )

    def _hold(self, picture_ids, why: str) -> None:
        """Hold *picture_ids* back once more, and say so in one line."""
        if not picture_ids:
            return
        waits = [self._retry_hold.hold(pid) for pid in picture_ids]
        logger.warning(
            "MissingImageEmbeddingFinder: %d pictures are held back for "
            "%.0f-%.0f s before their next attempt (%s): %r",
            len(waits),
            min(waits),
            max(waits),
            why,
            id_sample(picture_ids),
        )
