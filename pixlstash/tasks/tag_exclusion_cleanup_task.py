"""Take tags the owner removed from tagging off this library's pictures.

Removing a penalised tag stops the tagger producing it; this task removes what
is already there: the picture's tags, the tagger's predictions and labels for
it, and its pending suggestions. Started by ``DELETE
/users/me/penalised-tags/{tag}`` and by a retry, through
:mod:`pixlstash.services.tag_exclusion_service`.

The work is bound to the decision that started it: the names, the library, and
the exclusion generation. Each batch checks that decision again inside its own
writer turn, so a tag added back (or removed again with another tag) stops a
job at its next batch rather than letting it delete on an old decision.

Rows are removed in id order, a bounded batch per writer turn, so interactive
writes interleave. Locked pictures are left out of the query itself, before the
batch limit, so a run of locked rows cannot stall the job. Rows are deleted one
by one through the ORM so every picture's metadata hash follows.

Nothing is kept for undo: the receipt in the activity log carries the names and
the counts. A batch of tags rewrites the tag sidecars of its pictures in its own
writer turn, from the tags they keep. A picture whose sidecar cannot be written
keeps its rows, so the status still counts it and a retry takes it up again.
"""

import threading
from typing import Iterable

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models import TARGET_TAG_EXCLUSION, Picture, Tag, TagPrediction
from pixlstash.db_models.library_settings import LibrarySettings
from pixlstash.db_models.reference_folder import ReferenceFolder
from pixlstash.db_models.tag import is_tag_sentinel
from pixlstash.db_models.tag_suggestion import TagSuggestion
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services import tag_health_service
from pixlstash.services.operation_log_service import (
    record_payload_operation_in_session,
)
from pixlstash.services.set_lock_service import locked_picture_id_subquery
from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority, TaskStatus
from pixlstash.utils.caption_file_utils import (
    DEFAULT_TAGS_SUFFIX,
    SIDECAR_TYPE_TAGS,
    get_sidecar_mtime,
    write_sidecar,
    writeback_target,
)
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.quality.tag_exclusion import (
    excluded_spellings,
    filter_tag_names,
    session_tag_exclusions,
)
from pixlstash.utils.service.tag_prediction_utils import (
    recompute_anomaly_tag_uncertainty,
)
from pixlstash.utils.sql_chunking import in_json

logger = get_logger(__name__)

#: Rows removed per writer turn.
BATCH_SIZE = 200

#: The job states that mean it has not finished.
ACTIVE_STATES = frozenset({"queued", "running"})

# (count key, model) in the order the rows are removed.
_PHASES = (
    ("tags", Tag),
    ("predictions", TagPrediction),
    ("suggestions", TagSuggestion),
)


def matching_rows(
    session: Session, model, names: Iterable[str], *columns, locked: bool = False
):
    """Select *columns* (default: the rows) of *model* carrying one of *names*.

    *names* are normalised (stripped, lower case); the stored spellings that
    normalise to one of them are read first (:func:`excluded_spellings`). Rows
    on locked pictures are left out in the query itself, or with *locked* are
    the only ones returned. Only pending suggestions count: a reviewed one is a
    verdict.
    """
    lock = model.picture_id.in_ if locked else model.picture_id.not_in
    spellings = excluded_spellings(session, model.tag, frozenset(names))
    query = select(*(columns or (model,))).where(
        in_json(model.tag, spellings),
        lock(locked_picture_id_subquery()),
    )
    if model is TagSuggestion:
        query = query.where(TagSuggestion.status == "PENDING")
    return query


def pictures_carrying(
    session: Session, names: Iterable[str], *, locked: bool = False
) -> set[int]:
    """Ids of the pictures that still carry one of *names* in any form."""
    names = sorted(names)
    if not names:
        return set()
    carrying: set[int] = set()
    for _key, model in _PHASES:
        query = matching_rows(session, model, names, model.picture_id, locked=locked)
        carrying.update(session.exec(query.distinct()).all())
    return carrying


class TagExclusionCleanupTask(BaseTask):
    """Remove the named tags from every unlocked picture in one library.

    The Scrapheap is included and counted on its own; locked pictures are left
    as they are and counted. ``job`` is what the status route reports.
    """

    def __init__(self, vault, names: Iterable[str], generation: int):
        names = sorted(set(names))
        super().__init__(
            task_type="TagExclusionCleanupTask",
            params={"names": names, "generation": generation},
        )
        self._vault = vault
        self._names = frozenset(names)
        self._generation = generation
        self._library_uuid = vault.library_uuid
        self._lock = threading.Lock()
        self._job = {
            "id": self.id,
            "names": names,
            "state": "queued",
            "pictures": 0,
            "scrapheap": 0,
            "locked": 0,
            "removed": {key: 0 for key, _model in _PHASES},
            "sidecar_failed": 0,
        }
        self._changed: set[int] = set()
        self._scrapheap: set[int] = set()

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    @property
    def queue_type(self) -> QueueType:
        return QueueType.CPU

    @property
    def library_uuid(self) -> str:
        return self._library_uuid

    def snapshot(self) -> dict:
        """A copy of the job's progress, safe to hand to a response."""
        with self._lock:
            job = dict(self._job)
            job["names"] = list(job["names"])
            job["removed"] = dict(job["removed"])
        # Cancelled or failed before ``_run_task`` could say so itself.
        if job["state"] in ACTIVE_STATES and self.status is TaskStatus.CANCELLED:
            job["state"] = "cancelled"
        elif job["state"] in ACTIVE_STATES and self.status is TaskStatus.FAILED:
            job["state"] = "failed"
        return job

    def mark_not_started(self) -> None:
        """Record that no task runner took the job."""
        self._set(state="not_started")

    def _set(self, **values) -> None:
        with self._lock:
            self._job.update(values)

    def _run_task(self) -> dict:
        self._set(state="running")
        try:
            state = self._remove_rows()
            if state == "completed" and self.snapshot()["sidecar_failed"]:
                # Pictures whose sidecar was not written still carry the tags.
                state = "failed"
            self._finish(state)
        except Exception:
            logger.exception(
                "Removing %s from library %s failed after %s.",
                sorted(self._names),
                self._library_uuid,
                self.snapshot(),
            )
            self._set(state="failed")
            raise
        return self.snapshot()

    def _remove_rows(self) -> str:
        """Run the three phases. Returns ``completed``, ``superseded`` or ``cancelled``."""
        for key, model in _PHASES:
            last_id = 0
            while True:
                if self._cancel_event.is_set():
                    return "cancelled"
                result = self._vault.db.run_task(
                    self._remove_batch, key, model, last_id, priority=DBPriority.LOW
                )
                if result is None:
                    return "superseded"
                last_id, picture_ids = result
                if last_id is None:
                    break
                if picture_ids:
                    self._vault.notify(
                        EventType.CHANGED_TAGS,
                        {"picture_ids": sorted(picture_ids), "change_kind": "updated"},
                    )
        return "completed"

    def _still_current(self, session: Session) -> bool:
        return (
            self._names <= session_tag_exclusions(session)
            and self._vault.db.tag_exclusion_generation == self._generation
            and self._vault.library_uuid == self._library_uuid
        )

    def _remove_batch(self, session: Session, key: str, model, last_id: int):
        """Remove one batch of *model* rows after *last_id*.

        Tag rows go only with the picture's tag sidecar rewritten to match; a
        picture whose sidecar cannot be written keeps them.

        Returns ``None`` when the decision the job was started on no longer
        holds, ``(None, set())`` when the phase has nothing left, and
        otherwise ``(the last row id, the picture ids changed)``.
        """
        if not self._still_current(session):
            return None
        rows = session.exec(
            matching_rows(session, model, self._names)
            .where(model.id > last_id)
            .order_by(model.id)
            .limit(BATCH_SIZE)
        ).all()
        if not rows:
            return None, set()
        failed = self._write_tag_sidecars(session, rows) if model is Tag else set()
        removed = [row for row in rows if row.picture_id not in failed]
        picture_ids = {row.picture_id for row in removed}
        # One by one, so the unit of work sees each removal and the metadata
        # hash of every picture is recomputed. Not a verdict: no label is left.
        for row in removed:
            session.delete(row)
        session.flush()
        if model is not TagSuggestion:
            for picture in session.exec(
                select(Picture).where(Picture.id.in_(picture_ids))
            ).all():
                # The score was computed over the old tags; the finder
                # computes it again.
                picture.smart_score = None
                session.add(picture)
                recompute_anomaly_tag_uncertainty(session, picture.id)
        deleted = set(
            session.exec(
                select(Picture.id).where(
                    Picture.id.in_(picture_ids), Picture.deleted.is_(True)
                )
            ).all()
        )
        session.commit()
        with self._lock:
            self._job["removed"][key] += len(removed)
            self._job["sidecar_failed"] += len(failed)
            self._changed |= picture_ids
            self._scrapheap |= deleted
            self._job["pictures"] = len(self._changed)
            self._job["scrapheap"] = len(self._scrapheap)
        return rows[-1].id, picture_ids

    def _write_tag_sidecars(self, session: Session, rows: list[Tag]) -> set[int]:
        """Rewrite the tag sidecars of the pictures *rows* are taken off.

        Written in this writer turn, as ``sync_picture_sidecar`` writes in its
        own, so nothing records a sidecar for the picture in between. A sidecar
        edited outside PixlStash since it was last read in is left for the
        folder scan to import.

        Returns:
            The ids of the pictures whose sidecar could not be written.
        """
        failed = set()
        for plan in self._plan_sidecars(session, rows):
            picture = plan["picture"]
            target = writeback_target(
                plan["image_path"],
                SIDECAR_TYPE_TAGS,
                plan["suffix"],
                picture.tags_file,
                bool(plan["content"]),
            )
            if target is None:
                continue
            if target == picture.tags_file and picture.tags_file_mtime is not None:
                on_disk = get_sidecar_mtime(target)
                if on_disk is not None and on_disk != picture.tags_file_mtime:
                    logger.info(
                        "Not rewriting %s for picture %s: it changed on disk "
                        "since it was read in; the folder scan imports it.",
                        target,
                        picture.id,
                    )
                    continue
            mtime = write_sidecar(target, plan["content"])
            if mtime is None:
                failed.add(picture.id)
                logger.warning(
                    "Could not rewrite the tag sidecar %s of picture %s; the "
                    "picture keeps its tags until a retry.",
                    target,
                    picture.id,
                )
                continue
            picture.tags_file = target
            picture.tags_file_mtime = mtime
            session.add(picture)
        return failed

    def _plan_sidecars(self, session: Session, rows: list[Tag]) -> list[dict]:
        """What each picture's tag sidecar should hold once *rows* are gone.

        Every one of *rows* names an excluded tag, so the picture's permitted
        tags are the ones it keeps.
        """
        exclusions = session_tag_exclusions(session)
        settings = session.exec(select(LibrarySettings)).first()
        picture_ids = {row.picture_id for row in rows}
        tags: dict[int, list[str]] = {}
        for picture_id, tag in session.exec(
            select(Tag.picture_id, Tag.tag)
            .where(Tag.picture_id.in_(picture_ids))
            .order_by(Tag.id)
        ).all():
            if tag and not is_tag_sentinel(tag):
                tags.setdefault(picture_id, []).append(tag)
        plans = []
        for picture in session.exec(
            select(Picture).where(Picture.id.in_(picture_ids))
        ).all():
            if not picture.file_path:
                continue
            # A reference picture follows its folder's settings; a managed
            # one follows the library's, by its root-relative path.
            if picture.reference_folder_id:
                owner = session.get(ReferenceFolder, picture.reference_folder_id)
                image_path = picture.file_path
                suffix = owner.tags_suffix if owner else None
            else:
                owner = settings
                image_path = ImageUtils.resolve_picture_path(
                    self._vault.image_root, picture.file_path
                )
                suffix = (owner.tags_suffix if owner else None) or DEFAULT_TAGS_SUFFIX
            if owner is None or not owner.sync_tags or not image_path:
                continue
            plans.append(
                {
                    "picture": picture,
                    "image_path": image_path,
                    "suffix": suffix,
                    "content": ", ".join(
                        filter_tag_names(tags.get(picture.id, []), exclusions)
                    ),
                }
            )
        return plans

    def _finish(self, state: str) -> None:
        locked = len(
            self._vault.db.run_immediate_read_task(
                pictures_carrying, self._names, locked=True
            )
        )
        self._set(locked=locked)
        if self._changed:
            # Tag health does not see tag removals on its own.
            tag_health_service.start_rebuild(self._vault)
            self._vault.db.run_task(
                self._record_receipt, state, priority=DBPriority.LOW
            )
        self._set(state=state)
        logger.info(
            "Removed %s from library %s: %s",
            sorted(self._names),
            self._library_uuid,
            self.snapshot(),
        )

    def _record_receipt(self, session: Session, state: str) -> None:
        job = self.snapshot()
        names = ", ".join(f"'{name}'" for name in job["names"])
        record_payload_operation_in_session(
            session,
            op_type="tags.exclusion_cleanup",
            target_type=TARGET_TAG_EXCLUSION,
            before={"names": job["names"]},
            after={
                "job": job["id"],
                "state": state,
                "counts": {
                    "pictures": job["pictures"],
                    "scrapheap": job["scrapheap"],
                    "locked": job["locked"],
                    "removed": job["removed"],
                },
            },
            summary=f"Removed {names} from {job['pictures']} pictures.",
            undoable=False,
        )
        session.commit()
