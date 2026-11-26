from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from sqlmodel import Session

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.inference.workflows.description import DescriptionWorkflow
from pixlstash.pixl_logging import get_logger
from pixlstash.services.scrapheap_service import file_location_is_unreachable
from pixlstash.services.set_lock_service import locked_picture_ids
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority
from pixlstash.tasks.pipeline_scope import pipeline_picture_clauses
from pixlstash.tasks.task_outcome import Outcome, Reason, id_sample, outcomes_payload
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.media_load import LoadFailure
from pixlstash.utils.unprocessable_image_registry import load_signed
from pixlstash.utils.vram_utils import is_vram_oom

if TYPE_CHECKING:
    from pixlstash.inference.engine import InferenceEngine


logger = get_logger(__name__)


def describable_picture_clauses() -> tuple:
    """The picture rows captioning may touch: not in the Scrapheap, and with a file.

    One definition, because two queries have to answer identically: the
    selection the work planner runs
    (``MissingDescriptionFinder._fetch_missing_descriptions``) and the
    "awaiting captions" number (``Vault._count_missing_descriptions``). A
    picture one leaves out and the other keeps would sit in the count for ever.
    A picture restored from the Scrapheap keeps its NULL or sentinel
    description and is picked up again on the next sweep, so nothing is lost by
    holding it out while it is deleted.

    The Scrapheap term is ``deleted IS NOT 1``, which no index serves, so the
    finder's probe still reads ``ix_picture_description_pending``: beside
    ``deleted IS 0`` SQLite takes ``ix_picture_deleted`` instead. ``deleted`` is
    never NULL, so the two select the same rows. This is the pipeline-wide
    rule, :func:`pipeline_picture_clauses`.
    """
    return pipeline_picture_clauses()


class DescriptionTask(BaseTask):
    """Task for generating and persisting description batches.

    The result reports an outcome for every picture it was handed (see
    :mod:`pixlstash.tasks.task_outcome`):

    - ``SUCCEEDED``: a caption was written.
    - ``FAILED_PERMANENT`` (``INFERENCE_ERROR``): the batch captioned other
      pictures, this one's file decodes, and the model still gave it nothing.
      Its description is written as ``""``, which the finder never selects.
    - ``FAILED_PERMANENT`` (``UNDECODABLE``, ``NO_FRAMES``): the file reads
      and does not decode. Nothing is written; the unprocessable-image
      registry holds it out of the finders until the file changes.
    - ``FAILED_RETRYABLE``: nothing is written. ``UNREACHABLE``,
      ``IO_TRANSIENT``, ``RESOURCE`` or ``LOAD_ERROR`` when the picture's own
      file cannot be loaded for now; ``BACKEND_UNAVAILABLE`` when the batch
      captioned nobody whose file loads (the workflow raised, returned
      nothing, or its model failed to load), which says the backend is down,
      not that these pictures cannot be captioned.
    - ``DEFERRED``: not written: cancelled before it was captioned
      (``CANCELLED``), frozen by a locked set, or, while the batch ran, moved
      to the Scrapheap or given a description by someone else (see
      :meth:`_write_descriptions`).

    Args:
        database: Vault database instance.
        workflow: :class:`~pixlstash.inference.workflows.description.DescriptionWorkflow`
            used to generate captions.
        pictures: Pictures to process in this batch.
    """

    CPU_SPILLOVER_REUSE_GRACE_S = 8.0
    #: The outcome of each load failure of a picture the workflow left
    #: uncaptioned, apart from an I/O error, whose reason depends on whether
    #: the file's location is there. The image-embedding stage's table.
    _LOAD_FAILURE_OUTCOMES = {
        LoadFailure.UNDECODABLE: (Outcome.FAILED_PERMANENT, Reason.UNDECODABLE),
        LoadFailure.NO_FRAMES: (Outcome.FAILED_PERMANENT, Reason.NO_FRAMES),
        LoadFailure.RESOURCE: (Outcome.FAILED_RETRYABLE, Reason.RESOURCE),
        LoadFailure.UNCLASSIFIED: (Outcome.FAILED_RETRYABLE, Reason.LOAD_ERROR),
    }
    _cpu_spillover_engine: "InferenceEngine | None" = None
    _cpu_spillover_last_used_at: float = 0.0
    _cpu_spillover_lock = threading.Lock()

    def __init__(
        self,
        database,
        workflow: DescriptionWorkflow,
        pictures: list[Picture],
        engine_override: str | None = None,
        interactive: bool = False,
    ):
        picture_ids = [pic.id for pic in (pictures or []) if getattr(pic, "id", None)]
        super().__init__(
            task_type="DescriptionTask",
            params={
                "picture_ids": picture_ids,
                "batch_size": len(picture_ids),
            },
        )
        self._db = database
        self._workflow = workflow
        self._pictures = pictures or []
        self._cpu_spillover_enabled = False
        self._engine_override = engine_override
        self._interactive = interactive

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.URGENT if self._interactive else TaskPriority.LOW

    @property
    def queue_type(self) -> QueueType:
        return QueueType.GPU

    def allow_cpu_spillover(self) -> bool:
        return True

    def enable_cpu_spillover(self) -> None:
        self._cpu_spillover_enabled = True

    @classmethod
    def _acquire_cpu_spillover_engine(
        cls, image_root: str, tagger_settings: dict
    ) -> "InferenceEngine":
        with cls._cpu_spillover_lock:
            if cls._cpu_spillover_engine is None:
                from pixlstash.inference.engine import InferenceEngine

                logger.debug("DescriptionTask: creating CPU spillover InferenceEngine.")
                cls._cpu_spillover_engine = InferenceEngine.create(
                    device="cpu",
                    image_root=image_root,
                )
            # The configured captioner and its settings, not the defaults.
            cls._cpu_spillover_engine.set_tagger_settings(tagger_settings)
            cls._cpu_spillover_last_used_at = time.perf_counter()
            return cls._cpu_spillover_engine

    @classmethod
    def _release_idle_cpu_spillover_engine(cls, force: bool = False) -> None:
        with cls._cpu_spillover_lock:
            engine = cls._cpu_spillover_engine
            if engine is None:
                return
            if not force:
                idle_s = time.perf_counter() - cls._cpu_spillover_last_used_at
                if idle_s < cls.CPU_SPILLOVER_REUSE_GRACE_S:
                    return
            cls._cpu_spillover_engine = None
        try:
            engine.close()
        except Exception as exc:
            logger.debug("DescriptionTask CPU spillover engine close failed: %s", exc)

    def estimated_vram_mb(self) -> int:
        try:
            return max(
                0,
                self._workflow.estimate_vram_mb(
                    len(self._pictures), plugin_name=self._engine_override
                ),
            )
        except Exception as exc:
            logger.debug(
                "DescriptionTask: VRAM estimate failed for %d picture(s); "
                "assuming 0: %s",
                len(self._pictures),
                exc,
            )
            return 0

    def _run_task(self):
        if not self._pictures:
            return {"changed_count": 0, "changed": [], **outcomes_payload([])}

        generated, settled = self._generate_descriptions_batch(self._pictures)
        changed = []
        if generated:
            changed = self._db.run_task(
                self._write_descriptions,
                generated,
                {pic.id: pic.description for pic in self._pictures},
                priority=DBPriority.LOW,
            )

        written = {pid for _, pid, _, _ in changed}
        outcomes = []
        for pic in self._pictures:
            if pic.id in written:
                outcome = (
                    (Outcome.SUCCEEDED, None)
                    if generated[pic.id]
                    else (Outcome.FAILED_PERMANENT, Reason.INFERENCE_ERROR)
                )
            else:
                # Generated but not written: locked since the batch started,
                # given a description while it ran, or the row is gone or in
                # the Scrapheap.
                outcome = settled.get(pic.id, (Outcome.DEFERRED, None))
            outcomes.append((pic.id, *outcome))
        return {
            "changed_count": len(changed),
            "changed": changed,
            **outcomes_payload(outcomes),
        }

    @staticmethod
    def _write_descriptions(
        session: Session, generated: dict[int, str], read: dict[int, str | None]
    ) -> list:
        """Write *generated* (``{pid: description}``) on the writer thread.

        A picture is written only while its description is still the one in
        *read*, the value the batch was computed from. Inference takes minutes
        on a slow model, and a person may type a caption, or ask for one from
        another engine, in the meantime. Their edit is a write on this same
        thread, so here it is either already in the row or lands after this
        write; deciding when the batch was queued would let it be overwritten.
        A picture moved to the Scrapheap while the batch ran is not written
        either; the finder does not select it again until it is restored.

        Returns the ``changed`` entries of the pictures actually written.
        """
        # Defense in depth: a picture frozen by a locked set has a read-only
        # description (rule 3) - never persist a machine-regenerated caption
        # onto it, even if one was generated for an in-flight task.
        locked = locked_picture_ids(session, list(generated))
        changed = []
        edited = []
        scrapped = []
        for pid, description in generated.items():
            if pid in locked:
                continue
            db_pic = session.get(Picture, pid)
            if db_pic is None:
                continue
            if db_pic.deleted:
                scrapped.append(pid)
                continue
            if db_pic.description != read[pid]:
                edited.append(pid)
                continue
            db_pic.description = description
            session.add(db_pic)
            changed.append((Picture, pid, "description", description))
        if edited:
            logger.info(
                "DescriptionTask: %d picture(s) were given a description while "
                "the batch ran; theirs is kept: %r",
                len(edited),
                id_sample(edited),
            )
        if scrapped:
            logger.info(
                "DescriptionTask: %d picture(s) were moved to the Scrapheap while "
                "the batch ran; their description is left as it was: %r",
                len(scrapped),
                id_sample(scrapped),
            )
        session.commit()
        return changed

    def _generate_descriptions_batch(
        self, pictures: list[Picture]
    ) -> tuple[dict[int, str], dict]:
        """Caption *pictures* and decide what to write for each.

        Returns:
            ``({pid: description}, {pid: (Outcome, Reason)})``: the
            descriptions to write (``""`` for a picture the model cannot
            caption), and the outcome of every picture that gets no write.
        """
        picture_ids = [pic.id for pic in pictures]
        settled: dict = {}
        # Drop pictures frozen by a locked set: their description is read-only
        # (rule 3), so never machine-regenerate it. MissingDescriptionFinder also
        # excludes them; this covers the interactive/detection dispatch path too.
        locked = self._db.run_immediate_read_task(locked_picture_ids, picture_ids)
        if locked:
            logger.info(
                "DescriptionTask: skipping %d locked picture(s) %s "
                "(frozen description)",
                len(locked),
                sorted(locked),
            )
            settled.update((pid, (Outcome.DEFERRED, None)) for pid in locked)
            pictures = [pic for pic in pictures if pic.id not in locked]
            picture_ids = [pid for pid in picture_ids if pid not in locked]
            if not pictures:
                return {}, settled
        logger.debug(
            "DescriptionTask: Generating descriptions for batch_size=%s ids=%s",
            len(pictures),
            picture_ids,
        )

        self._release_idle_cpu_spillover_engine(force=False)
        active_workflow = self._workflow
        cpu_spillover_engine = None
        if self._cpu_spillover_enabled:
            logger.debug(
                "DescriptionTask %s: using CPU spillover for ids=%s",
                self.id,
                picture_ids,
            )
            cpu_spillover_engine = self._acquire_cpu_spillover_engine(
                self._db.image_root, self._workflow.tagger_settings
            )
            active_workflow = cpu_spillover_engine.description_workflow

        try:
            # The event is the task's own, not the workflow's: with CPU
            # spillover the batch runs on a workflow object built fresh on
            # access (InferenceEngine.description_workflow), so a cancel stored
            # on the workflow would be set on something nobody is running.
            batch_results = active_workflow.generate_batch(
                pictures,
                engine_override=self._engine_override,
                stop_event=self._cancel_event,
            )
        except Exception as exc:
            if is_vram_oom(exc):
                # The GPU was full, so nothing is known about these pictures.
                # Raising leaves every description exactly as it was and hands
                # the batch to BaseTask.run's retry; clearing them here would
                # destroy captions over a transient condition.
                logger.warning(
                    "DescriptionTask ran out of GPU memory for ids=%s; "
                    "descriptions left unchanged: %s",
                    picture_ids,
                    exc,
                )
                raise
            logger.error(
                "DescriptionTask failed for ids=%s (plugin=%r); no description "
                "is changed and the batch is retried later: %s",
                id_sample(picture_ids),
                self._engine_override,
                exc,
                exc_info=True,
            )
            batch_results = None
        finally:
            if cpu_spillover_engine is not None:
                with self._cpu_spillover_lock:
                    self._cpu_spillover_last_used_at = time.perf_counter()
                self._release_idle_cpu_spillover_engine(force=False)

        results = batch_results or {}
        generated = {pid: results[pid] for pid in picture_ids if results.get(pid)}
        uncaptioned = [pic for pic in pictures if pic.id not in generated]
        if not uncaptioned:
            return generated, settled

        # A cancel is not a failure: it says nothing about these pictures, so
        # leave them exactly as they were and let the finder pick them up on
        # the next run.
        if self._cancel_event.is_set():
            settled.update(
                (pic.id, (Outcome.DEFERRED, Reason.CANCELLED)) for pic in uncaptioned
            )
            return generated, settled

        # The picture's own file first: one that cannot be loaded says nothing
        # about the model.
        settled.update(self._settle_file_failures(uncaptioned))
        loadable = [pic for pic in uncaptioned if pic.id not in settled]
        if not loadable:
            return generated, settled

        if not generated:
            # Nothing in the batch was captioned: the workflow raised, returned
            # nothing, or its model failed to load (the Florence-2 service logs
            # that and answers None for every picture). That is the backend,
            # not these pictures, and blanking them would lose them for good.
            logger.warning(
                "DescriptionTask: the batch captioned none of its %d loadable "
                "pictures (plugin=%r); no description is changed and the batch "
                "is retried later: %r",
                len(loadable),
                self._engine_override,
                id_sample(pic.id for pic in loadable),
            )
            settled.update(
                (pic.id, (Outcome.FAILED_RETRYABLE, Reason.BACKEND_UNAVAILABLE))
                for pic in loadable
            )
            return generated, settled

        # The model captioned the rest of the batch and gave these nothing.
        # Blanking is how a picture it genuinely cannot caption stops being
        # retried for ever - MissingDescriptionFinder selects only NULL or a
        # ``__description::`` sentinel, so an empty string is a permanent
        # exclusion.
        logger.error(
            "DescriptionTask: no description could be generated for %d "
            "picture(s) whose file decodes (plugin=%r); clearing them: %r",
            len(loadable),
            self._engine_override,
            id_sample(pic.id for pic in loadable),
        )
        generated.update((pic.id, "") for pic in loadable)
        return generated, settled

    def _settle_file_failures(self, pictures: list[Picture]) -> dict:
        """The outcome of each picture in *pictures* whose file cannot be loaded.

        Each file is loaded with the image-embedding stage's loader and its
        failure classified the same way: a file whose whole location is gone
        (an unmounted drive) is held out of every finder by the registry until
        the directory is back, as is a file that reads and does not decode
        until it changes; any other failure is retried later. Only the
        pictures the workflow left uncaptioned get here, so a batch that
        captions everyone loads nothing twice.

        Returns:
            ``{pid: (Outcome, Reason)}`` for the pictures that do not load.
        """
        registry = getattr(self._db, "unprocessable_images", None)
        image_root = getattr(self._db, "image_root", "") or ""
        settled = {}
        for pic in pictures:
            file_path = getattr(pic, "file_path", None)
            full_path = ImageUtils.resolve_picture_path(image_root, file_path)
            if not full_path:
                logger.error("DescriptionTask: picture %s has no file path", pic.id)
                settled[pic.id] = (Outcome.FAILED_RETRYABLE, Reason.LOAD_ERROR)
                continue
            # The registry's evidence too, so the gate does not decode the
            # file a second time.
            loaded = load_signed(full_path, multiframe_count=1)
            failure = loaded.result.failure
            if failure is None:
                continue
            if failure is LoadFailure.IO_ERROR:
                unreachable = file_location_is_unreachable(full_path)
                if unreachable and registry is not None:
                    registry.mark_unreachable(
                        pic.id, full_path, reason="image location is not mounted"
                    )
                settled[pic.id] = (
                    Outcome.FAILED_RETRYABLE,
                    Reason.UNREACHABLE if unreachable else Reason.IO_TRANSIENT,
                )
                continue
            settled[pic.id] = self._LOAD_FAILURE_OUTCOMES[failure]
            if failure.permanent and registry is not None:
                registry.mark_unprocessable(
                    pic.id,
                    full_path,
                    reason=f"image could not be decoded ({failure.value})",
                    load=loaded,
                )
        if settled:
            logger.warning(
                "DescriptionTask: %d picture(s) got no description because "
                "their file cannot be loaded (%s); no description is changed: %r",
                len(settled),
                ", ".join(sorted({reason.value for _, reason in settled.values()})),
                id_sample(settled),
            )
        return settled
