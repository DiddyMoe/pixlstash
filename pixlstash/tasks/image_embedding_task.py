import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import requests
from PIL import Image
from sqlalchemy import func, or_
from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture, PictureLikenessQueue
from pixlstash.services.builtin_models import builtin_model_dir
from pixlstash.services.scrapheap_service import file_location_is_unreachable
from pixlstash.tagger_plugins.clip_service import CLIP_MODEL_NAME

from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.media_load import (
    LoadFailure,
    load_pil_classified,
)
from pixlstash.utils.sql_chunking import not_in_json
from pixlstash.pixl_logging import get_logger
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority
from pixlstash.tasks.task_outcome import Outcome, Reason, id_sample, outcomes_payload
from pixlstash.utils.vram_utils import is_vram_oom


logger = get_logger(__name__)


class ImageEmbeddingTask(BaseTask):
    """Task for generating image embeddings and aesthetic scores for one batch.

    The result reports every picture of the batch exactly once, in
    ``outcomes`` (see :mod:`pixlstash.tasks.task_outcome`): what was written,
    what failed and may be retried, what cannot be decoded, and what was not
    attempted. ``changed`` lists only the pictures whose row was written.
    """

    BATCH_SIZE = 128
    #: A batch slower than this logs its [EMBED_TIMING] line at INFO instead of
    #: DEBUG. Same threshold as FaceExtractionTask.SLOW_BATCH_LOG_S.
    SLOW_BATCH_LOG_S = 5.0
    #: Longest the task waits for its preload, seconds. A decode runs in native
    #: code and cannot be interrupted, so this bounds the wait, not the decode:
    #: past it the batch goes on with what was decoded (see _wait_for_preload).
    PRELOAD_WAIT_S = 300.0
    #: How long on_cancel waits for the preload to stop, seconds.
    _CANCEL_JOIN_S = 10.0
    #: Decode threads per preload.
    _PRELOAD_WORKERS = 4
    #: The outcome of each load failure the preload can report, apart from an
    #: I/O error, whose reason depends on whether the file's location is there.
    _LOAD_FAILURE_OUTCOMES = {
        LoadFailure.UNDECODABLE: (Outcome.FAILED_PERMANENT, Reason.UNDECODABLE),
        LoadFailure.NO_FRAMES: (Outcome.FAILED_PERMANENT, Reason.NO_FRAMES),
        LoadFailure.RESOURCE: (Outcome.FAILED_RETRYABLE, Reason.RESOURCE),
        LoadFailure.UNCLASSIFIED: (Outcome.FAILED_RETRYABLE, Reason.LOAD_ERROR),
    }

    # `filename`, not `path`. This table used to hold absolute paths built at
    # import time from a second copy of the download folder's location, which is
    # what stopped the folder from being relocatable at all: the shelf would have
    # declared the new location while this table kept naming the old one, so a
    # scorer that had just been moved would be downloaded again. The folder is
    # asked for at use time instead - see `_aesthetic_config`.
    AESTHETIC_MODELS = {
        "ViT-L-14": {
            "url": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
            "filename": "sac+logos+ava1-l14-linearMSE.pth",
            "dim": 768,
        },
        "ViT-B-32": {
            "url": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_b_32_linear.pth",
            "filename": "sa_0_4_vit_b_32_linear.pth",
            "dim": 512,
        },
    }
    AESTHETIC_SUPPORTED_CLIP = set(AESTHETIC_MODELS.keys())

    _aesthetic_model = None
    _aesthetic_disabled: Optional[bool] = None

    def __init__(self, database, clip_workflow, batch: list):
        """
        Args:
            clip_workflow: A :class:`~pixlstash.inference.workflows.clip_embedding.ClipEmbeddingWorkflow`
                instance used for CLIP inference, or ``None`` when no tagger is
                available.
            batch: List of ``(picture_id, file_path)`` pairs pre-fetched by the
                finder.  Images are loaded from disk in ``on_queued()`` so that
                I/O overlaps with the previous task's GPU inference.
        """
        picture_ids = [pid for pid, _ in (batch or [])]
        super().__init__(
            task_type="ImageEmbeddingTask",
            params={
                "picture_ids": picture_ids,
                "batch_size": len(picture_ids),
            },
        )
        self._db = database
        self._clip_workflow = clip_workflow
        self._batch = batch or []

        # Preloading state - images loaded from disk in on_queued() so I/O
        # overlaps with the previous task's GPU inference. All of it is read
        # and written under _preload_lock, and reset by on_queued().
        # {batch index: entries}, published by each decode as it finishes. An
        # entry is (pid, file_path, image, dhash, tensor): the image is already
        # reduced by CLIP's own first resize, never the decoded original (#2748).
        self._preload_entries: dict[int, list] = {}
        # Batch indices whose decode has started, and those whose decode has
        # returned (published, or failed).
        self._preload_started: set[int] = set()
        self._preload_done: set[int] = set()
        # {pid: (Outcome, Reason)} for the pictures the preload has no entry
        # for: cancelled before their turn, failed outside the loader, left
        # behind when the pool failed or when the task stopped waiting.
        self._preload_unattempted: dict = {}
        # Set when the preload thread has finished; then _preload_on_stopped,
        # if finish_preload() registered one, is called once.
        self._preload_finished = False
        self._preload_on_stopped: Callable[[], None] | None = None
        self._preload_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._preload_cancel = threading.Event()
        # Set by on_cancel() only, never by the task's own timeout: the
        # decoded batch is then not embedded either.
        self._cancel_requested = threading.Event()

        if ImageEmbeddingTask._aesthetic_disabled is None:
            ImageEmbeddingTask._aesthetic_disabled = self._aesthetic_config() is None

    def on_queued(self) -> None:
        # A requeue is a fresh run: an earlier cancel no longer applies.
        self._cancel_requested.clear()
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return
        self._preload_cancel.clear()
        with self._preload_lock:
            self._preload_entries = {}
            self._preload_started = set()
            self._preload_done = set()
            self._preload_unattempted = {}
            self._preload_finished = False
            self._preload_on_stopped = None
        self._preload_thread = threading.Thread(
            target=self._preload_images_task,
            name=f"EmbedPreload-{self.id[:8]}",
            daemon=True,
        )
        self._preload_thread.start()

    def on_cancel(self) -> None:
        self._cancel_requested.set()
        self._preload_cancel.set()
        if self._preload_thread is None:
            return
        self._preload_thread.join(timeout=self._CANCEL_JOIN_S)
        if self._preload_thread.is_alive():
            logger.warning(
                "ImageEmbeddingTask preload thread did not stop in time for task "
                "%s; a decode is still running",
                self.id,
            )

    def finish_preload(self, on_stopped: Callable[[], None]) -> set:
        """Stop the preload; name the pictures whose decode is still running.

        Called by the finder when the task has completed. The preload may
        outlive the task: the wait for it is bounded and a decode cannot be
        interrupted. No new decode starts after this call.

        Args:
            on_stopped: Called once, from the preload thread, when the last of
                the running decodes has returned. Never called when the
                returned set is empty.

        Returns:
            The ids of the pictures still being decoded; empty when none is.
        """
        self._preload_cancel.set()
        with self._preload_lock:
            if self._preload_finished:
                return set()
            running = {
                self._batch[index][0]
                for index in self._preload_started - self._preload_done
            }
            if running:
                self._preload_on_stopped = on_stopped
            return running

    def _preload_images_task(self) -> None:
        """Decode, hash and preprocess the batch, off the GPU worker.

        Everything per-image that is CPU work happens here, in a small pool:
        the decode, the perceptual hash (a LANCZOS resample of the full
        frame), the reduction to CLIP's first resize, and CLIP's own
        preprocessing when the model is already loaded.
        Measured on the GPU worker instead, those were 4 s of a 4.2 s batch of
        128 - the forward pass itself is a tenth of a second - and the single
        GPU worker sat on CPU work while every other stage waited for it.

        The pool is up to :attr:`_PRELOAD_WORKERS` daemon threads draining a
        queue, so a decode stuck in native code holds neither the GPU worker
        (see :meth:`_wait_for_preload`) nor the interpreter's exit. Each
        picture's entries are published as soon as its decode returns, and
        every picture that gets none is named in ``_preload_unattempted`` with
        the reason: cancelled before its turn, failed outside the loader, or
        left behind when the pool itself failed. Absence is never read as a
        broken file.
        """

        def _one(pid, file_path):
            full_path = ImageUtils.resolve_picture_path(self._db.image_root, file_path)
            if not full_path:
                logger.error("EmbedPreload: picture %s has no file path", pid)
                return [(pid, file_path, None, None, None, LoadFailure.UNCLASSIFIED)]
            loaded = load_pil_classified(full_path, multiframe_count=3)
            if loaded.failure is not None:
                logger.log(
                    logging.ERROR
                    if loaded.failure is LoadFailure.UNCLASSIFIED
                    else logging.WARNING,
                    "EmbedPreload: could not load picture %s (%s): %s, %s",
                    pid,
                    file_path,
                    loaded.failure.value,
                    loaded.error,
                )
                return [(pid, file_path, None, None, None, loaded.failure)]
            images = loaded.frames
            # Hashed from the full frame, then dropped (#2748): the original
            # held to the end of the batch, beside its tensor, is what kept
            # 6.6 GB per batch of 64 24 MP photos.
            hashes = [self._compute_dhash(img) for img in images]
            tensors = None
            if self._clip_workflow is not None:
                try:
                    images = self._clip_workflow.reduce_for_preprocess(images)
                except Exception as exc:
                    # Embedded from the original instead; only the memory
                    # is lost.
                    logger.warning(
                        "EmbedPreload: could not reduce %s (%s) for CLIP; "
                        "holding the decoded original: %s",
                        file_path,
                        [img.size for img in images],
                        exc,
                    )
                try:
                    tensors = self._clip_workflow.preprocess_images(images)
                except Exception as exc:
                    # The worker preprocesses anything not done here.
                    logger.debug(
                        "EmbedPreload: preprocess failed for %s, deferring to the "
                        "worker: %s",
                        file_path,
                        exc,
                    )
                    tensors = None
            return [
                (
                    pid,
                    file_path,
                    img,
                    hashes[i],
                    tensors[i] if tensors is not None else None,
                )
                for i, img in enumerate(images)
            ]

        def _drain() -> None:
            while True:
                with self._preload_lock:
                    # Taken and marked started in one step, so that a picture
                    # is never being decoded without finish_preload knowing.
                    if self._preload_cancel.is_set():
                        return
                    try:
                        index, (pid, file_path) = todo.get_nowait()
                    except queue.Empty:
                        return
                    self._preload_started.add(index)
                try:
                    entries = _one(pid, file_path)
                except Exception as exc:
                    reason = self._preload_failure_reason(exc)
                    logger.error(
                        "EmbedPreload: preloading picture %s failed (task %s, %s); "
                        "it is retried later: %s",
                        pid,
                        self.id,
                        reason.value,
                        exc,
                        exc_info=True,
                    )
                    with self._preload_lock:
                        self._preload_unattempted.setdefault(
                            pid, (Outcome.FAILED_RETRYABLE, reason)
                        )
                        self._preload_done.add(index)
                    continue
                with self._preload_lock:
                    self._preload_entries[index] = entries
                    self._preload_done.add(index)

        batch = list(self._batch)
        todo: queue.SimpleQueue = queue.SimpleQueue()
        for index, item in enumerate(batch):
            todo.put((index, item))
        workers: list[threading.Thread] = []
        pool_failure: Reason | None = None
        try:
            try:
                for n in range(min(self._PRELOAD_WORKERS, max(1, len(batch)))):
                    worker = threading.Thread(
                        target=_drain,
                        name=f"EmbedDecode-{self.id[:8]}-{n}",
                        daemon=True,
                    )
                    worker.start()
                    workers.append(worker)
            except Exception as exc:
                # "can't start new thread", out of memory, out of file
                # descriptors: the process is short of a resource, not the
                # files of their bytes. The threads already running still
                # drain the whole queue.
                pool_failure = self._preload_failure_reason(exc)
                logger.error(
                    "EmbedPreload: the preload pool failed (task %s, %s) with %d "
                    "decode threads running; the pictures it did not load are "
                    "retried later: %s",
                    self.id,
                    pool_failure.value,
                    len(workers),
                    exc,
                    exc_info=True,
                )
            for worker in workers:
                worker.join()
        finally:
            with self._preload_lock:
                for index, (pid, _) in enumerate(batch):
                    if index in self._preload_done or pid in self._preload_unattempted:
                        continue
                    if self._preload_cancel.is_set():
                        self._preload_unattempted[pid] = (
                            Outcome.DEFERRED,
                            Reason.CANCELLED,
                        )
                    elif pool_failure is not None:
                        self._preload_unattempted[pid] = (
                            Outcome.FAILED_RETRYABLE,
                            pool_failure,
                        )
                self._preload_finished = True
                on_stopped, self._preload_on_stopped = self._preload_on_stopped, None
                loaded = sum(
                    1
                    for entries in self._preload_entries.values()
                    for entry in entries[:1]
                    if entry[2] is not None
                )
            if on_stopped is not None:
                try:
                    on_stopped()
                except Exception as exc:
                    logger.error(
                        "EmbedPreload: telling the finder that task %s's decodes "
                        "stopped failed; it may build no new batch: %s",
                        self.id,
                        exc,
                        exc_info=True,
                    )
        logger.debug(
            "[EMBED_PRELOAD] task_id=%s preloaded=%d/%d", self.id, loaded, len(batch)
        )

    @staticmethod
    def _preload_failure_reason(exc: BaseException) -> Reason:
        """``RESOURCE`` when the process is short of threads, memory or file
        descriptors; ``LOAD_ERROR``, a bug, for anything else."""
        if isinstance(exc, (MemoryError, OSError, RuntimeError)):
            return Reason.RESOURCE
        return Reason.LOAD_ERROR

    def _wait_for_preload(self, timeout_s: float | None = None) -> list:
        """The preloaded entries in batch order, waiting at most *timeout_s*.

        *timeout_s* defaults to :attr:`PRELOAD_WAIT_S`. A decode stuck on a
        file (a stalled network share, a pathological image) cannot be
        interrupted, and waiting on it would hold the GPU worker for as long as
        it is stuck. Past the deadline the preload is stopped, the entries it
        published are returned, and each picture it did not finish is named in
        ``_preload_unattempted``: ``FAILED_RETRYABLE`` (``DECODE_TIMEOUT``)
        when its decode was still running, so the finder backs off from it
        rather than handing it to a new decode at once, and ``DEFERRED``
        (``DECODE_TIMEOUT``) when it was never started.
        """
        thread = self._preload_thread
        waited_s = 0.0
        if thread is not None:
            started_at = time.perf_counter()
            thread.join(self.PRELOAD_WAIT_S if timeout_s is None else timeout_s)
            waited_s = time.perf_counter() - started_at
        stuck: list = []
        skipped: list = []
        with self._preload_lock:
            # Decided under the lock, so the entries returned and the pictures
            # given up on are one consistent picture of the preload.
            if thread is not None and thread.is_alive():
                self._preload_cancel.set()
                for index, (pid, _) in enumerate(self._batch):
                    if index in self._preload_done or pid in self._preload_unattempted:
                        continue
                    if index in self._preload_started:
                        stuck.append(pid)
                        self._preload_unattempted[pid] = (
                            Outcome.FAILED_RETRYABLE,
                            Reason.DECODE_TIMEOUT,
                        )
                    else:
                        skipped.append(pid)
                        self._preload_unattempted[pid] = (
                            Outcome.DEFERRED,
                            Reason.DECODE_TIMEOUT,
                        )
            entries = [
                entry
                for index in sorted(self._preload_entries)
                for entry in self._preload_entries[index]
            ]
        if stuck or skipped:
            logger.warning(
                "ImageEmbeddingTask: task %s stopped waiting for its preload "
                "after %.1f s; %d decodes still running are retried later and "
                "%d pictures not started stay eligible. Still decoding: %r",
                self.id,
                waited_s,
                len(stuck),
                len(skipped),
                id_sample(stuck),
            )
        return entries

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.MEDIUM

    @property
    def queue_type(self) -> QueueType:
        return QueueType.GPU

    def estimated_vram_mb(self) -> int:
        if self._clip_workflow is None:
            return 0
        try:
            return max(0, self._clip_workflow.estimated_vram_mb(len(self._batch)))
        except Exception as exc:
            logger.debug(
                "ImageEmbeddingTask: VRAM estimate failed for %d image(s); "
                "assuming 0: %s",
                len(self._batch),
                exc,
            )
            return 0

    @classmethod
    def _aesthetic_config(cls):
        """The scorer for the active CLIP model, with its path resolved now.

        ``path`` is joined here rather than stored in the table so it follows a
        relocation of the download folder: every reader of the folder asks
        :func:`builtin_model_dir`, which is what makes the folder movable.
        """
        config = cls.AESTHETIC_MODELS.get(CLIP_MODEL_NAME)
        if config is None:
            return None
        return {
            **config,
            "path": os.path.join(builtin_model_dir(), config["filename"]),
        }

    @classmethod
    def _is_aesthetic_disabled(cls):
        if cls._aesthetic_disabled is None:
            cls._aesthetic_disabled = cls._aesthetic_config() is None
        return bool(cls._aesthetic_disabled)

    @classmethod
    def count_remaining(
        cls,
        session: Session,
        aesthetic_disabled: Optional[bool] = None,
        suppressed_ids: Optional[set] = None,
    ) -> int:
        """Count pictures needing image embedding or aesthetic score work.

        *suppressed_ids* - pictures whose file cannot be decoded (issue #585) -
        are excluded so progress does not stall at a non-zero "remaining" that can
        never drain. Pictures in the scrapheap are not counted, the same rows
        :meth:`fetch_work` leaves out.
        """
        stmt = (
            select(func.count())
            .select_from(Picture)
            .where(cls._needs_work_condition(aesthetic_disabled))
            .where(*cls._eligible_picture_clauses())
        )
        if suppressed_ids:
            stmt = stmt.where(not_in_json(Picture.id, suppressed_ids))
        result = session.exec(stmt).one()
        if isinstance(result, tuple):
            return result[0]
        return result or 0

    @classmethod
    def fetch_work(
        cls,
        session: Session,
        aesthetic_disabled: Optional[bool] = None,
        limit: Optional[int] = None,
        suppressed_ids: Optional[set] = None,
    ):
        """Fetch pictures needing image embedding or aesthetic score work.

        *suppressed_ids* - undecodable pictures (issue #585) - are excluded from
        the candidate window so a handful of corrupt files cannot crowd out real
        work and stall the finder. The finder adds the pictures it is holding
        back after a retryable failure, for the same reason.
        """
        # width and height size the batch (#2750); the finder hands the task
        # (id, file_path) pairs.
        stmt = (
            select(Picture.id, Picture.file_path, Picture.width, Picture.height)
            .where(cls._needs_work_condition(aesthetic_disabled))
            .where(*cls._eligible_picture_clauses())
        )
        if suppressed_ids:
            stmt = stmt.where(not_in_json(Picture.id, suppressed_ids))
        stmt = stmt.limit(int(limit or cls.BATCH_SIZE))
        return session.exec(stmt).all()

    @classmethod
    def _needs_work_condition(cls, aesthetic_disabled: Optional[bool]):
        """No embedding yet, or (while a scorer exists) no aesthetic score."""
        if aesthetic_disabled is None:
            aesthetic_disabled = cls._is_aesthetic_disabled()
        # Each arm has its own partial index (ix_picture_image_embedding_missing,
        # ix_picture_aesthetic_score_missing); SQLite serves the OR from both.
        missing_embedding = Picture.image_embedding.is_(None)
        if aesthetic_disabled:
            return missing_embedding
        return or_(missing_embedding, Picture.aesthetic_score.is_(None))

    @staticmethod
    def _eligible_picture_clauses() -> tuple:
        """The rows this stage may touch: not in the scrapheap, and with a file.

        The rule tagging applies (``tag_task.taggable_picture_clauses``). A
        picture restored from the scrapheap still has no embedding, so the next
        sweep picks it up; nothing is lost by leaving it out while it is
        deleted.

        ``deleted IS NOT 1`` rather than ``deleted IS 0``: as an equality term
        ``deleted`` lets the planner serve the probe from ``ix_picture_deleted``
        and walk every live row, instead of the two partial missing-column
        indexes, which have no ``deleted`` column to outbid it with (see
        ``Picture.__table_args__``). ``IS NOT`` is not an indexable term, and
        ``deleted`` is never NULL, so the rows are the same.
        """
        return (Picture.deleted.is_not(True), Picture.file_path.is_not(None))

    @classmethod
    def release_models(cls):
        cls._aesthetic_model = None

    def _settle_load_failures(
        self, failures: dict[int, LoadFailure], batch_files: dict
    ) -> dict:
        """The outcome of each picture that could not be loaded.

        A failed picture keeps a NULL embedding, which ``fetch_work`` treats as
        still missing. The registry keeps two kinds out of every finder:

        - a permanent failure (the bytes read fine and the decoder rejects
          them), suppressed until the file changes;
        - an I/O failure whose whole location is gone (an unmounted drive),
          held until the directory is back.

        Everything else (no file descriptor left, an I/O error on a mounted
        disk, an exception nobody classified) is retryable, and the finder
        holds it back for a while instead.

        Returns:
            ``{pid: (Outcome, Reason)}`` for every picture in *failures*.
        """
        registry = getattr(self._db, "unprocessable_images", None)
        image_root = getattr(self._db, "image_root", "") or ""
        settled = {}
        for pid, failure in failures.items():
            full_path = ImageUtils.resolve_picture_path(
                image_root, batch_files.get(pid)
            )
            if failure is LoadFailure.IO_ERROR:
                unreachable = bool(full_path) and file_location_is_unreachable(
                    full_path
                )
                settled[pid] = (
                    Outcome.FAILED_RETRYABLE,
                    Reason.UNREACHABLE if unreachable else Reason.IO_TRANSIENT,
                )
                if unreachable and registry is not None:
                    registry.mark_unreachable(
                        pid, full_path, reason="image location is not mounted"
                    )
                continue
            settled[pid] = self._LOAD_FAILURE_OUTCOMES[failure]
            if failure.permanent and full_path and registry is not None:
                registry.mark_unprocessable(
                    pid,
                    full_path,
                    reason=f"image could not be decoded ({failure.value})",
                )
        return settled

    @staticmethod
    def _compute_dhash(image: Image.Image, hash_size: int = 8) -> Optional[str]:
        try:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img = image.convert("L").resize((hash_size + 1, hash_size), resample)
            pixels = np.asarray(img, dtype=np.int16)
            diff = pixels[:, 1:] > pixels[:, :-1]
            bits = diff.flatten()
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            return f"{value:0{hash_size * hash_size // 4}x}"
        except Exception as exc:
            logger.debug("ImageEmbeddingTask: dhash computation failed: %s", exc)
            return None

    def _ensure_clip_ready(self) -> bool:
        if self._clip_workflow is None:
            logger.error(
                "ImageEmbeddingTask: ClipEmbeddingWorkflow not available for CLIP embeddings."
            )
            return False

        for attempt in range(1, 4):
            try:
                self._clip_workflow.ensure_ready()
                if self._clip_workflow.is_ready():
                    return True
            except Exception as exc:
                logger.warning(
                    "ImageEmbeddingTask: CLIP init attempt %s/3 failed: %s",
                    attempt,
                    exc,
                )
                if attempt < 3:
                    time.sleep(1.0)

        logger.error(
            "ImageEmbeddingTask: CLIP model unavailable after retries; embeddings cannot be generated."
        )
        return False

    def _ensure_model(self):
        if ImageEmbeddingTask._aesthetic_model is not None:
            return
        if self._is_aesthetic_disabled():
            return

        if CLIP_MODEL_NAME not in self.AESTHETIC_SUPPORTED_CLIP:
            logger.info(
                "ImageEmbeddingTask: Aesthetic model disabled for CLIP model '%s'.",
                CLIP_MODEL_NAME,
            )
            ImageEmbeddingTask._aesthetic_disabled = True
            return

        config = self._aesthetic_config()
        if not config:
            logger.info(
                "ImageEmbeddingTask: No aesthetic model config for CLIP model '%s'.",
                CLIP_MODEL_NAME,
            )
            ImageEmbeddingTask._aesthetic_disabled = True
            return

        # Local import: torch costs seconds to import and this module is on the
        # server's import path. By the time an aesthetic model is being built,
        # the CLIP workflow has already loaded torch anyway.
        import torch
        import torch.nn as nn

        try:
            model_path = config["path"]
            model_url = config["url"]
            model_dim = config["dim"]

            if not os.path.exists(model_path):
                logger.info("Downloading aesthetic model from %s...", model_url)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                response = requests.get(model_url, timeout=30)
                response.raise_for_status()
                with open(model_path, "wb") as file_handle:
                    file_handle.write(response.content)

            state_dict = torch.load(model_path, map_location="cpu")
            model = nn.Linear(model_dim, 1)
            model.load_state_dict(state_dict)
            model.eval()

            if self._clip_workflow is not None:
                model = model.to(self._clip_workflow.device)

            ImageEmbeddingTask._aesthetic_model = model
            logger.info("ImageEmbeddingTask: Aesthetic model loaded.")

        except Exception as exc:
            logger.error("ImageEmbeddingTask: Failed to load aesthetic model: %s", exc)
            ImageEmbeddingTask._aesthetic_model = None
            ImageEmbeddingTask._aesthetic_disabled = True

    def _ensure_embedding_backend(self) -> bool:
        """Whether CLIP can embed now; says why not every time it cannot.

        Not rate-limited: while the backend is down the finder holds every new
        batch back on a growing backoff, so each refusal here is one of a few
        widely spaced probes and each is worth its log line.
        """
        if self._clip_workflow is None:
            logger.error(
                "ImageEmbeddingTask: no CLIP workflow; %d pictures are retried "
                "once one is available.",
                len(self._batch),
            )
            return False
        if self._clip_workflow.is_ready():
            return True
        try:
            self._clip_workflow.ensure_ready()
        except Exception as exc:
            logger.error(
                "ImageEmbeddingTask: Failed to initialise CLIP backend; %d "
                "pictures are retried later: %s",
                len(self._batch),
                exc,
            )
            return False
        if self._clip_workflow.is_ready():
            return True
        logger.error(
            "ImageEmbeddingTask: CLIP backend is not ready after initialising; "
            "%d pictures are retried later.",
            len(self._batch),
        )
        return False

    def _run_task(self):
        self._ensure_model()
        if not self._batch:
            return {"changed_count": 0, "changed": [], **outcomes_payload([])}
        if not self._ensure_embedding_backend():
            # Nothing will use what the preload decodes.
            self._preload_cancel.set()
            return {
                "changed_count": 0,
                "changed": [],
                **outcomes_payload(
                    (pid, Outcome.FAILED_RETRYABLE, Reason.BACKEND_UNAVAILABLE)
                    for pid, _ in self._batch
                ),
            }

        started_at = time.perf_counter()
        preloaded = self._wait_for_preload()
        preload_wait_s = time.perf_counter() - started_at
        return self._process_preloaded(
            preloaded, started_at=started_at, preload_wait_s=preload_wait_s
        )

    def _process_preloaded(
        self,
        preloaded: list,
        *,
        batch: list | None = None,
        started_at: float | None = None,
        preload_wait_s: float = 0.0,
    ) -> dict:
        """Embed, score and persist the preloaded batch; settle every picture.

        After :meth:`on_cancel` nothing is embedded or written: the decoded
        pictures are ``DEFERRED`` (``CANCELLED``).

        Args:
            preloaded: One entry per picture (per frame for a video):
                ``(pid, file_path, PIL.Image | None)``, the image reduced by
                :meth:`ClipEmbeddingWorkflow.reduce_for_preprocess` (the hash
                is of the original), optionally followed by
                the perceptual hash and the CLIP-preprocessed tensor the
                preload pool computed. Anything missing is computed here. A
                picture that could not be loaded is one entry,
                ``(pid, file_path, None, None, None, LoadFailure)``.
            batch: The ``(pid, file_path)`` pairs the task was given; defaults
                to the task's own. The batch, not ``preloaded``, says which
                pictures need an outcome: a picture the preload produced no
                entry for still gets one.
            started_at: ``time.perf_counter()`` when the task started running;
                ``total_s`` in the timing line is measured from it.
            preload_wait_s: How long the task blocked on the preload thread.

        Returns:
            The task result: ``changed`` holds a ``(model, pic_id, field,
            value)`` change tuple per picture whose row was written,
            ``changed_count`` its length, and ``outcomes`` one ``(pid,
            Outcome, Reason | None)`` per picture of the batch.
        """
        if started_at is None:
            started_at = time.perf_counter()
        batch_files = dict(self._batch if batch is None else batch)
        for entry in preloaded:
            batch_files.setdefault(entry[0], entry[1])
        with self._preload_lock:
            unattempted = dict(self._preload_unattempted)
        settled: dict[int, tuple[Outcome, Reason | None]] = {}
        flat_images = []
        flat_pids = []
        flat_hashes = []
        flat_tensors = []
        load_failures: dict[int, LoadFailure] = {}

        for entry in preloaded:
            pid, file_path, img = entry[:3]
            if img is None:
                load_failures[pid] = entry[5]
                continue
            # A preloaded image is reduced, so its hash comes from the preload
            # (None when that failed); hashing it here would differ.
            dhash = entry[3] if len(entry) > 3 else self._compute_dhash(img)
            tensor = entry[4] if len(entry) > 4 else None
            flat_images.append(img)
            flat_hashes.append(dhash)
            flat_tensors.append(tensor)
            flat_pids.append(pid)
        # All or nothing: a batch is one forward pass, and the service
        # preprocesses the whole list itself when any tensor is missing.
        if any(t is None for t in flat_tensors):
            flat_tensors = None

        # A None image means the file could not be loaded; the preload says why.
        frame_pids = set(flat_pids)
        settled.update(self._settle_load_failures(load_failures, batch_files))
        for pid, outcome in unattempted.items():
            if pid not in settled and pid not in frame_pids:
                settled[pid] = outcome

        embeddings = None
        scores = None
        inference_start = time.perf_counter()
        if flat_images and self._cancel_requested.is_set():
            # Pre-empted or shut down after the decode: the GPU is wanted
            # elsewhere, so nothing is embedded or written, and the pictures
            # stay eligible.
            logger.info(
                "ImageEmbeddingTask: task %s was cancelled before inference; "
                "%d decoded pictures are left for a later batch: %r",
                self.id,
                len(frame_pids),
                id_sample(frame_pids),
            )
            for pid in frame_pids:
                settled[pid] = (Outcome.DEFERRED, Reason.CANCELLED)
        elif flat_images:
            embeddings, failure = self._encode(flat_images, flat_tensors, frame_pids)
            if embeddings is None:
                for pid in frame_pids:
                    settled[pid] = (Outcome.FAILED_RETRYABLE, failure)
            else:
                scores, scorer_failed = self._score(embeddings, frame_pids)
                if scorer_failed:
                    # The embedding and hash are still written; the score is
                    # left as it is and the picture comes back for it later.
                    for pid in frame_pids:
                        settled[pid] = (Outcome.FAILED_RETRYABLE, Reason.SCORER_ERROR)
        inference_s = time.perf_counter() - inference_start

        updates = []
        if embeddings is not None:
            updates = self._aggregate(flat_pids, embeddings, scores, flat_hashes)

        db_s = 0.0
        changed = []
        if updates:
            db_start = time.perf_counter()
            updated_ids = self._db.run_task(
                self._save_results, updates, priority=DBPriority.LOW
            )
            db_s = time.perf_counter() - db_start
            changed = [(Picture, pid, "image_embedding", None) for pid in updated_ids]
            written = set(updated_ids)
            not_written = {pid for pid, *_ in updates if pid not in written}
            for pid, *_ in updates:
                if pid in written:
                    settled.setdefault(pid, (Outcome.SUCCEEDED, None))
                else:
                    settled[pid] = (Outcome.FAILED_RETRYABLE, Reason.PERSIST_ERROR)
            if not_written:
                logger.warning(
                    "ImageEmbeddingTask: %d embedded pictures were not written "
                    "(their rows are gone): %s",
                    len(not_written),
                    id_sample(not_written),
                )

        unsettled = [pid for pid in batch_files if pid not in settled]
        if unsettled:
            # Every path above settles what it touched, so this is a bug in
            # the preload's bookkeeping; the pictures are retried, not guessed.
            logger.error(
                "ImageEmbeddingTask: %d pictures got no preload result and no "
                "failure; retrying them later: %s",
                len(unsettled),
                id_sample(unsettled),
            )
            for pid in unsettled:
                settled[pid] = (Outcome.FAILED_RETRYABLE, Reason.LOAD_ERROR)

        total_s = time.perf_counter() - started_at
        n = len(updates)
        logger.log(
            logging.INFO if total_s >= self.SLOW_BATCH_LOG_S else logging.DEBUG,
            "[EMBED_TIMING] task_id=%s n=%d device=%s preload_wait_s=%.3f "
            "inference_s=%.3f db_s=%.3f total_s=%.3f throughput=%.1f/s",
            self.id,
            n,
            getattr(self._clip_workflow, "device", "unknown"),
            preload_wait_s,
            inference_s,
            db_s,
            total_s,
            n / total_s if total_s > 0 else 0.0,
        )
        return {
            "changed_count": len(changed),
            "changed": changed,
            **outcomes_payload((pid, *settled[pid]) for pid in batch_files),
        }

    def _encode(self, images: list, tensors: list | None, pids: set) -> tuple:
        """CLIP embeddings for *images*, one row per image, or why there are none.

        Returns:
            ``(embeddings, None)``, or ``(None, Reason)`` when CLIP is not
            available, raised, or returned a different number of rows than it
            was given images (rows cannot be matched to pictures then).
        """
        if not self._ensure_clip_ready():
            return None, Reason.BACKEND_UNAVAILABLE
        try:
            embeddings = (
                self._clip_workflow.encode_images(images, tensors=tensors)
                if tensors is not None
                else self._clip_workflow.encode_images(images)
            )
        except Exception as exc:
            reason = (
                Reason.RESOURCE
                if isinstance(exc, MemoryError) or is_vram_oom(exc)
                else Reason.INFERENCE_ERROR
            )
            logger.error(
                "ImageEmbeddingTask: CLIP failed on %d images (%s); %d pictures "
                "are retried later: %s; pictures %s",
                len(images),
                reason.value,
                len(pids),
                exc,
                id_sample(pids),
                exc_info=True,
            )
            return None, reason
        rows = None if embeddings is None else len(embeddings)
        if rows != len(images):
            logger.error(
                "ImageEmbeddingTask: CLIP returned %s embeddings for %d images; "
                "%d pictures are retried later: %s",
                rows,
                len(images),
                len(pids),
                id_sample(pids),
            )
            return None, Reason.INFERENCE_ERROR
        return embeddings, None

    def _score(self, embeddings, pids: set) -> tuple:
        """Aesthetic scores for *embeddings*, one per row.

        Returns:
            ``(scores, False)``; ``(None, False)`` when no scorer is loaded; or
            ``(None, True)`` when the scorer raised or returned a different
            number of scores than rows.
        """
        model = ImageEmbeddingTask._aesthetic_model
        if model is None:
            return None, False
        # Local import (see _ensure_model): a loaded scorer means torch is
        # already resident.
        import torch

        try:
            with torch.no_grad():
                model_param = next(model.parameters())
                emb_tensor = torch.from_numpy(embeddings).to(
                    dtype=model_param.dtype,
                    device=model_param.device,
                )
                scores = model(emb_tensor).squeeze()
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)
                scores = scores.cpu().numpy()
        except Exception as exc:
            logger.error(
                "ImageEmbeddingTask: aesthetic scoring failed on %d embeddings; "
                "%d pictures keep their score and are retried later: %s; "
                "pictures %s",
                len(embeddings),
                len(pids),
                exc,
                id_sample(pids),
                exc_info=True,
            )
            return None, True
        if len(scores) != len(embeddings):
            logger.error(
                "ImageEmbeddingTask: the aesthetic scorer returned %d scores for "
                "%d embeddings; %d pictures keep their score and are retried "
                "later: %s",
                len(scores),
                len(embeddings),
                len(pids),
                id_sample(pids),
            )
            return None, True
        return scores, False

    @staticmethod
    def _aggregate(pids: list, embeddings, scores, hashes: list) -> list:
        """One ``(pid, embedding bytes, score, hash)`` update per picture.

        A video's frames are averaged into one unit-length embedding and one
        score; its hash is the first frame's that has one. *scores* is
        ``None`` when nothing was scored, and the update's score is then
        ``None``, which leaves the stored score alone.
        """
        per_pid = defaultdict(lambda: {"embs": [], "scores": [], "phash": None})
        for i, pid in enumerate(pids):
            data = per_pid[pid]
            data["embs"].append(embeddings[i])
            if scores is not None:
                data["scores"].append(scores[i])
            if data["phash"] is None and hashes[i]:
                data["phash"] = hashes[i]

        updates = []
        for pid, data in per_pid.items():
            embs = data["embs"]
            final_emb = embs[0] if len(embs) == 1 else np.mean(embs, axis=0)
            norm = np.linalg.norm(final_emb)
            if norm > 0:
                final_emb = final_emb / norm
            final_score = float(np.mean(data["scores"])) if data["scores"] else None
            emb_bytes = np.asarray(final_emb, dtype=np.float32).tobytes()
            updates.append((pid, emb_bytes, final_score, data["phash"]))
        return updates

    @staticmethod
    def _save_results(session: Session, updates):
        updated_ids = []
        for pid, emb_bytes, score, phash in updates:
            pic = session.get(Picture, pid)
            if pic:
                pic.image_embedding = emb_bytes
                # Scored against the embedding just replaced; re-queue it.
                pic.prompt_match = None
                if score is not None:
                    pic.aesthetic_score = score
                pic.perceptual_hash = phash
                updated_ids.append(pid)
        session.commit()
        if updated_ids:
            PictureLikenessQueue.enqueue(session, updated_ids)
            session.commit()
        return updated_ids
