import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models.character import Character
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_project import PictureProjectMember
from pixlstash.db_models.picture_set import PictureSet, PictureSetMember
from pixlstash.db_models.tag import Tag, TAG_PENDING_SENTINEL
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services import import_dedup_service, scrapheap_service
from pixlstash.services.import_dedup_service import ContentFingerprint
from pixlstash.services.layout_move_service import resolve_placement
from pixlstash.services.project_membership_service import (
    character_project_ids,
    picture_set_project_ids,
    reconcile_entity_projects_change,
)
from pixlstash.tasks.base_task import BaseTask, TaskPriority
from pixlstash.utils.image_processing.image_utils import ImageUtils


logger = get_logger(__name__)


def _sidecar_stem(name: str) -> str:
    """Basename stem for sidecar matching.

    Mirrors ``routes.pictures._helpers._normalise_sidecar_stem`` verbatim;
    duplicated as a one-liner to avoid a routes→tasks import cycle.
    """
    return os.path.splitext(os.path.basename(name or ""))[0].strip().lower()


# Staged files are looked up, decided and inserted this many at a time. One
# batch is one writer-queued lookup and reads each matching library file at
# most once, and the next batch sees the pictures this one added.
_DEDUP_BATCH = 64


@dataclass
class _ImportTally:
    """What a staged import has decided so far, across its batches."""

    imported_ids: list[int] = field(default_factory=list)
    duplicate_count: int = 0
    scrapheaped_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    # Distinct scrapheaped pictures this import matched and restores. Per
    # PICTURE, while ``scrapheaped_count`` is per FILE (the bucket arithmetic):
    # several staged copies of one content name one id once.
    scrapheaped_picture_ids: list[int] = field(default_factory=list)
    seen_scrapheaped_ids: set[int] = field(default_factory=set)
    # ``(size, full hash)`` of every file this import accepted as new. The
    # lookup only sees pictures already inserted, and a batch is inserted after
    # its files are decided, so two byte-identical files in one batch would both
    # pass it. Track them here so the repeat is counted, not imported.
    seen_fingerprints: set[tuple[int, str]] = field(default_factory=set)
    built_count: int = 0


class PictureImportTask(BaseTask):
    """Finish an async import server-side, in the safe (background) window.

    The client first streams uploaded files into a per-session staging directory
    during the *unsafe* window (the browser tab must stay open). Once staging is
    complete the route hands off to this task, which runs on the shared
    ``TaskRunner`` so the import completes even after the tab closes (the *safe*
    window). Staged files are handled in batches of ``_DEDUP_BATCH``: each file
    is hashed, the batch's matches are looked up and confirmed on full content
    together, and the new files are ingested into the vault. Each batch's new
    ``Picture`` rows (each with a pending-tag sentinel so the finders pick it up
    for downstream processing) are inserted together, followed by their
    project, set and character membership, before the next batch is looked up;
    ``phase`` says which of these steps is running. The lookup and the insert
    run under ``import_dedup_service.serialised_import``, so an overlapping
    import of the same content finds this one's pictures instead of adding them
    again. The staging directory is removed when the task ends, whether it
    completed or failed.

    The commit route hands the task a library read lease. It is released
    exactly once, by the task runner's completion callback (see
    ``Vault._release_task_library_lease``), so a library switch cannot close
    the vault while the import is still queued or running.

    **Every staged file lands in exactly one of five disjoint buckets**,
    imported, duplicate (live content already in the vault, including a repeat
    inside this batch), **scrapheaped** (content matches a soft-deleted picture:
    not imported again, the picture is restored instead), failed, cancelled and
    they sum
    to the staged total. A scrapheaped match used to be reported as an ordinary
    duplicate, which told the user nothing about why their file did not appear.
    See :mod:`pixlstash.services.import_dedup_service`.

    Live progress is exposed via ``_total_count`` / ``_processed_count`` and read
    by ``Vault._build_worker_progress_snapshot`` so the import surfaces as a task
    row in the frontend task manager, mirroring the detection / watch-folder
    workers. Completion is announced by ``Vault._on_task_completed`` (a
    ``CHANGED_PICTURES`` + ``PICTURE_IMPORTED`` broadcast). That callback sends
    nothing for a failed task, so when a batch fails after earlier batches
    added pictures, the task sends the same two events for those itself.

    Attributes:
        _vault: The ``Vault``; restoring a Scrapheap match records an undoable
            operation and announces it, which needs more than the database.
        _db: The application ``VaultDatabase``.
        _staged_files: List of ``{"file_path", "original_file_name"}`` staged
            entries to ingest.
        _project_id: Optional project the imported pictures are added to.
        _staging_id: The staging-session id (carried into the result/logs).
        _staging_dir: Absolute path of the staging directory to remove when done.
        _phase: What ``_run_task`` is doing, read through ``phase``.
        _library_lease_release: Releases the commit's library read lease;
            ``None`` once released, or when the task holds no lease.
        _origin_client_id: The originating tab's client id (for origin-stamped
            WebSocket events), or ``None`` for an external caller.
        outcomes: Per-file outcomes in staged order, read live by the staging
            status route so the grid can mark each arriving tile (#2512).
            A batch's outcomes are appended once its pictures are in the
            library, so ``new`` is never reported for a row that does not
            exist yet. Append-only while the task runs.
    """

    def __init__(
        self,
        vault,
        staged_files: list[dict],
        *,
        project_id: Optional[int] = None,
        set_id: Optional[int] = None,
        character_id: Optional[int] = None,
        sidecar_tags_by_stem: Optional[dict[str, list[str]]] = None,
        staging_id: str,
        staging_dir: Optional[str] = None,
        origin_client_id: Optional[str] = None,
        restore_context: Optional[dict] = None,
        library_lease_release: Optional[Callable[[], None]] = None,
    ):
        """Initialise the task.

        Args:
            vault: The application ``Vault``.
            staged_files: Sequence of ``{"file_path", "original_file_name"}``
                dicts for each staged media file on disk.
            project_id: Optional project id to add every imported picture to.
            set_id: Optional picture-set id every imported picture is added to
                (a drop-to-set target).
            character_id: Optional character id every imported picture is
                associated with (a drop-to-character target; deferred via
                ``Picture.pending_character_id`` until face extraction runs).
            sidecar_tags_by_stem: Optional map of ``basename stem -> [tags]``
                parsed from ``.txt`` caption sidecars, applied to the matching
                imported picture (mirrors the one-shot import).
            staging_id: The staging-session id this task finishes.
            staging_dir: Absolute path of the staging directory to remove on
                completion (best-effort cleanup).
            origin_client_id: Originating tab's client id for origin-stamped
                events, or ``None`` for an external client.
            restore_context: ``operation_log_service.request_context`` of the
                commit request, recorded on the restore of any Scrapheap match.
            library_lease_release: Releases the library read lease the commit
                took for this import; called once, when the task settles.
        """
        staged_files = staged_files or []
        super().__init__(
            task_type="PictureImportTask",
            params={
                "staging_id": staging_id,
                "candidate_count": len(staged_files),
                "project_id": project_id,
                "set_id": set_id,
                "character_id": character_id,
                "origin_client_id": origin_client_id,
            },
        )
        self._vault = vault
        self._db = vault.db
        self._staged_files = staged_files
        self._project_id = project_id
        self._set_id = set_id
        self._character_id = character_id
        self._sidecar_tags_by_stem = sidecar_tags_by_stem or {}
        self._staging_id = staging_id
        self._staging_dir = staging_dir
        self._origin_client_id = origin_client_id
        self._restore_context = restore_context or {}
        # Live progress, read by Vault.get_worker_progress for the task manager.
        self._total_count = len(staged_files)
        self._processed_count = 0
        self.outcomes: list[dict] = []
        self._stop_event = threading.Event()
        self._phase: Optional[str] = None
        self._library_lease_release = library_lease_release
        self._lease_lock = threading.Lock()

    @property
    def priority(self) -> TaskPriority:
        # User-initiated, latency-sensitive ingest - ahead of the finder-driven
        # background lanes, mirroring DetectionTask.
        return TaskPriority.HIGH

    @property
    def phase(self) -> Optional[str]:
        """``processing`` while a batch of files is checked and read in,
        ``persisting`` while that batch's new pictures and their memberships
        are written (the two alternate, once per batch), and ``finalizing``
        while Scrapheap matches are restored and the staged files removed.
        ``None`` before the task starts."""
        return self._phase

    def on_cancel(self) -> None:
        self._stop_event.set()

    def release_library_lease(self) -> None:
        """Release the commit's library read lease, at most once.

        Called by the task runner's completion callback on every settlement
        (completed, failed, or cancelled while queued), and by the commit route
        when the task never reached the runner. A second call does nothing:
        releasing a read lease twice would undercount the library's readers.
        """
        with self._lease_lock:
            release, self._library_lease_release = self._library_lease_release, None
        if release is not None:
            release()

    def _record_outcome(
        self, entry: dict, outcome: str, picture_id: Optional[int] = None
    ) -> None:
        """Note one staged file's outcome for the live status (#2512).

        Args:
            entry: The staged entry; its ``client_key`` names the file to the
                client that streamed it.
            outcome: ``new``, ``duplicate``, ``scrapheaped`` or ``failed``.
            picture_id: The picture a duplicate or scrapheaped file matched.
        """
        self.outcomes.append(
            {
                "key": entry.get("client_key"),
                "outcome": outcome,
                "picture_id": picture_id,
            }
        )

    def _run_task(self):
        self._phase = "processing"
        try:
            return self._import_staged_files()
        finally:
            # On failure too: the files belong to this task from commit on, and
            # nothing else removes them while its session is kept for polling.
            self._cleanup_staging_dir()

    def _import_staged_files(self) -> dict:
        """Ingest every staged file, one batch at a time.

        Returns:
            The task result read by the staging status route.
        """
        tally = _ImportTally()

        # Placement on write (v1.11 Phase 4b): where the library's layout says
        # a picture with these assignments belongs. Resolved once for the whole
        # import, because every picture in it gets the same project and set, and
        # empty for a library with no layout - which is every library until its
        # owner chooses one.
        subfolder = resolve_placement(
            self._db, project_id=self._project_id, set_id=self._set_id
        )
        if subfolder:
            logger.info(
                "PictureImportTask: the library layout places this import in "
                "%s/ (staging_id=%s).",
                subfolder,
                self._staging_id,
            )

        try:
            batch: list[tuple[dict, Optional[ContentFingerprint]]] = []
            for index, entry in enumerate(self._staged_files):
                if self._stop_event.is_set():
                    # Count the untouched tail explicitly. The buckets have to
                    # sum to the staged total on a cancelled run too, and
                    # deriving that remainder by subtracting the other buckets
                    # would report a number nobody measured.
                    tally.cancelled_count = len(self._staged_files) - index
                    break
                batch.append((entry, self._fingerprint(index, entry)))
                if len(batch) >= _DEDUP_BATCH:
                    self._settle_batch(batch, tally, subfolder)
                    batch = []
            if batch:
                self._settle_batch(batch, tally, subfolder)
        except Exception as exc:
            if not tally.imported_ids:
                raise
            # Earlier batches are already in the library. Say so: a failed
            # import otherwise reads as one that added nothing.
            logger.error(
                "PictureImportTask %s failed after adding %d new picture(s) "
                "(staging_id=%s): %s",
                self.id,
                len(tally.imported_ids),
                self._staging_id,
                exc,
            )
            # The completion broadcast is not sent for a failed task, so
            # announce the pictures that are in the library from here.
            source = "ui" if self._origin_client_id else "external"
            added = list(tally.imported_ids)
            self._vault.notify(
                EventType.CHANGED_PICTURES,
                {
                    "picture_ids": added,
                    "source": source,
                    "origin_client_id": self._origin_client_id,
                    "change_kind": "added",
                },
            )
            self._vault.notify(
                EventType.PICTURE_IMPORTED,
                {
                    "ids": added,
                    "source": source,
                    "origin_client_id": self._origin_client_id,
                    "change_kind": "added",
                },
            )
            raise RuntimeError(
                f"The import stopped after {len(tally.imported_ids)} new "
                f"picture(s) were added: {exc}"
            ) from exc

        if tally.cancelled_count:
            logger.info(
                "PictureImportTask %s cancelled; stopped early "
                "(staging_id=%s, processed=%d/%d, %d staged file(s) never "
                "reached)",
                self.id,
                self._staging_id,
                self._processed_count,
                self._total_count,
                tally.cancelled_count,
            )

        self._phase = "finalizing"
        scrapheaped_picture_ids = tally.scrapheaped_picture_ids
        if scrapheaped_picture_ids:
            matched_ids = scrapheaped_picture_ids
            # Report what was restored, not what matched: a match the retention
            # sweep purged meanwhile did not come back.
            scrapheaped_picture_ids = scrapheap_service.restore_pictures(
                self._vault, matched_ids, **self._restore_context
            )
            logger.info(
                "PictureImportTask: restored %s of the scrapheaped picture(s) %s "
                "matched by this import (staging_id=%s)",
                scrapheaped_picture_ids,
                matched_ids,
                self._staging_id,
            )

        imported_ids = tally.imported_ids
        logger.info(
            "PictureImportTask completed: staging_id=%s imported=%d duplicates=%d "
            "scrapheaped=%d (%d picture(s)) failed=%d cancelled=%d of %d staged",
            self._staging_id,
            len(imported_ids),
            tally.duplicate_count,
            tally.scrapheaped_count,
            len(scrapheaped_picture_ids),
            tally.failed_count,
            tally.cancelled_count,
            self._total_count,
        )
        # The five buckets are disjoint and describe every staged file. Each is
        # counted where it happens; none is derived by subtracting the others
        # from the total. If they ever disagree with the staged total the
        # summary the user reads is wrong, so say so instead of shipping it
        # quietly.
        bucket_total = (
            len(imported_ids)
            + tally.duplicate_count
            + tally.scrapheaped_count
            + tally.failed_count
            + tally.cancelled_count
        )
        if bucket_total != self._total_count:
            logger.error(
                "PictureImportTask %s bucket arithmetic is inconsistent "
                "(staging_id=%s): imported=%d + duplicate=%d + scrapheaped=%d + "
                "failed=%d + cancelled=%d = %d, but %d file(s) were staged. "
                "(%d picture(s) were built for insert.)",
                self.id,
                self._staging_id,
                len(imported_ids),
                tally.duplicate_count,
                tally.scrapheaped_count,
                tally.failed_count,
                tally.cancelled_count,
                bucket_total,
                self._total_count,
                tally.built_count,
            )
        return {
            "staging_id": self._staging_id,
            "imported_picture_ids": imported_ids,
            "imported_count": len(imported_ids),
            "duplicate_count": tally.duplicate_count,
            "scrapheaped_count": tally.scrapheaped_count,
            "scrapheaped_picture_ids": scrapheaped_picture_ids,
            "failed_count": tally.failed_count,
            "cancelled_count": tally.cancelled_count,
            "candidate_count": self._total_count,
            "origin_client_id": self._origin_client_id,
        }

    def _fingerprint(self, index: int, entry: dict) -> Optional[ContentFingerprint]:
        """Hash one staged file, or return None when it cannot be read."""
        file_path = entry.get("file_path")
        if not file_path:
            # A staged entry with no path can never be ingested; it belongs in
            # the failed bucket, not in no bucket at all.
            logger.warning(
                "PictureImportTask: staged entry %d has no file_path "
                "(staging_id=%s); counting it as failed.",
                index,
                self._staging_id,
            )
            return None
        try:
            return (
                ImageUtils.calculate_hash_from_file_path(file_path),
                os.path.getsize(file_path),
                ImageUtils.calculate_full_hash_from_file_path(file_path),
            )
        except Exception as exc:
            logger.warning(
                "PictureImportTask: failed to hash staged file %s (staging_id=%s): %s",
                file_path,
                self._staging_id,
                exc,
            )
            return None

    def _settle_batch(
        self,
        batch: list[tuple[dict, Optional[ContentFingerprint]]],
        tally: _ImportTally,
        subfolder: Optional[str],
    ) -> None:
        """Decide one batch of staged files and add its new pictures.

        The lookup, the decisions and the insert hold the import lock, so an
        overlapping import looks up only after this batch's pictures exist. The
        batch's outcomes are recorded once its pictures are in the library.
        """
        fingerprints = [fp for _entry, fp in batch if fp is not None]
        decided: list[tuple[dict, str, Optional[int]]] = []
        new_pictures: list[Picture] = []
        with import_dedup_service.serialised_import():
            live: dict = {}
            scrapheaped: dict = {}
            if fingerprints:
                # Deliberately matches soft-deleted rows too: a scrapheaped
                # picture used to be invisible to import dedup on the one-shot
                # path, which re-imported its file as a second row. Here the
                # match was already found (this query never had a ``deleted``
                # filter) but was reported as an ordinary duplicate, which hid
                # from the user that their file is sitting in the Scrapheap.
                # The service classifies it instead of silently collapsing it.
                # Every insert an earlier holder of the import lock made, this
                # import's earlier batches included, committed before the lock
                # was released, so this lookup sees those pictures.
                candidates = self._db.run_task(
                    import_dedup_service.load_match_candidates_in_session,
                    sorted({(sha, size) for sha, size, _full in fingerprints}),
                )
                live, scrapheaped = import_dedup_service.partition_confirmed_matches(
                    candidates, fingerprints, self._db.image_root
                )
            for entry, fingerprint in batch:
                match = None
                if fingerprint is not None:
                    match = live.get(fingerprint) or scrapheaped.get(fingerprint)
                outcome, picture_id, pic = self._decide(
                    entry, fingerprint, match, tally, subfolder
                )
                if pic is not None:
                    new_pictures.append(pic)
                decided.append((entry, outcome, picture_id))
                self._processed_count += 1

            self._phase = "persisting"
            imported_ids: list[int] = []
            if new_pictures:
                tally.built_count += len(new_pictures)
                sidecar_tags_by_stem = self._sidecar_tags_by_stem

                def insert_pictures(session: Session, pictures: list[Picture]):
                    session.add_all(pictures)
                    session.flush()
                    for pic in pictures:
                        # Apply sidecar caption tags when the picture matches
                        # one by basename stem; otherwise seed the pending
                        # sentinel so the tagger picks it up (mirrors
                        # WatchFolderImportTask).
                        tags = sidecar_tags_by_stem.get(
                            _sidecar_stem(pic.original_file_name or "")
                        )
                        if tags:
                            for tag_str in tags:
                                session.add(Tag(picture_id=pic.id, tag=tag_str))
                        else:
                            session.add(
                                Tag(picture_id=pic.id, tag=TAG_PENDING_SENTINEL)
                            )
                    # The flush assigned the ids. Read them before the commit
                    # so nothing after it can fail an insert that is already
                    # in the library.
                    ids = [pic.id for pic in pictures]
                    session.commit()
                    return ids

                try:
                    imported_ids = self._db.run_task(
                        insert_pictures,
                        new_pictures,
                        priority=DBPriority.IMMEDIATE,
                    )
                except Exception:
                    # None of this batch's new files reached the library. A
                    # duplicate without a picture id matched a file this import
                    # decided was new: earlier batches are committed before
                    # this lookup, so their matches carry an id. Here it is a
                    # copy of a file that was rolled back.
                    for entry, outcome, picture_id in decided:
                        self._record_outcome(
                            entry,
                            "failed"
                            if outcome == "new"
                            or (outcome == "duplicate" and picture_id is None)
                            else outcome,
                            picture_id,
                        )
                    raise

        tally.imported_ids.extend(imported_ids)
        for entry, outcome, picture_id in decided:
            self._record_outcome(entry, outcome, picture_id)
        if imported_ids:
            self._apply_project(imported_ids)
            self._apply_set(imported_ids)
            self._apply_character(imported_ids)
        self._phase = "processing"

    def _decide(
        self,
        entry: dict,
        fingerprint: Optional[ContentFingerprint],
        match: Optional[import_dedup_service.ShaMatch],
        tally: _ImportTally,
        subfolder: Optional[str],
    ) -> tuple[str, Optional[int], Optional[Picture]]:
        """Decide one staged file and count it in its bucket.

        Returns:
            The outcome, the picture a duplicate or scrapheaped file matched,
            and for a new file the picture built for insert.
        """
        if fingerprint is None:
            tally.failed_count += 1
            return "failed", None, None
        pixel_sha, size_bytes, full_sha = fingerprint
        file_path = entry["file_path"]
        if match is not None and match.deleted:
            logger.info(
                "PictureImportTask: staged file %s matches scrapheaped "
                "picture %d (sha %s); not importing a second copy, "
                "restoring it instead (staging_id=%s)",
                file_path,
                match.id,
                pixel_sha,
                self._staging_id,
            )
            tally.scrapheaped_count += 1
            if match.id not in tally.seen_scrapheaped_ids:
                tally.seen_scrapheaped_ids.add(match.id)
                tally.scrapheaped_picture_ids.append(match.id)
            return "scrapheaped", match.id, None
        if match is not None or (size_bytes, full_sha) in tally.seen_fingerprints:
            logger.debug(
                "PictureImportTask: duplicate sha %s already imported; "
                "skipping staged file %s",
                pixel_sha,
                file_path,
            )
            tally.duplicate_count += 1
            return "duplicate", match.id if match is not None else None, None

        try:
            pic = ImageUtils.create_picture_from_file(
                image_root_path=self._db.image_root,
                source_file_path=file_path,
                pixel_sha=pixel_sha,
                subfolder=subfolder,
            )
        except Exception as exc:
            logger.warning(
                "PictureImportTask: failed to ingest staged file %s "
                "(staging_id=%s): %s",
                file_path,
                self._staging_id,
                exc,
            )
            tally.failed_count += 1
            return "failed", None, None
        pic.imported_at = datetime.now(timezone.utc)
        original_file_name = entry.get("original_file_name") or os.path.basename(
            file_path
        )
        if original_file_name:
            pic.original_file_name = original_file_name
        tally.seen_fingerprints.add((size_bytes, full_sha))
        return "new", None, pic

    def _apply_project(self, imported_ids: list[int]) -> None:
        """Add imported pictures to the drop target project (mirrors the
        one-shot import's ``apply_import_context``)."""
        if self._project_id is None:
            return

        def apply_project(session: Session, ids: list[int], project_id_value: int):
            for pic in session.exec(select(Picture).where(Picture.id.in_(ids))).all():
                member = session.exec(
                    select(PictureProjectMember).where(
                        PictureProjectMember.picture_id == pic.id,
                        PictureProjectMember.project_id == project_id_value,
                    )
                ).first()
                if member is None:
                    session.add(
                        PictureProjectMember(
                            picture_id=pic.id, project_id=project_id_value
                        )
                    )
                pic.project_id = project_id_value
                session.add(pic)
            session.commit()

        self._db.run_task(
            apply_project,
            imported_ids,
            self._project_id,
            priority=DBPriority.IMMEDIATE,
        )

    def _apply_set(self, imported_ids: list[int]) -> None:
        """Add imported pictures to the drop target set (mirrors
        ``picture_sets.add_picture_to_set``: ``PictureSetMember`` + project
        reconciliation when the set belongs to a project)."""
        if self._set_id is None:
            return

        def apply_set(session: Session, ids: list[int], set_id_value: int):
            picture_set = session.get(PictureSet, set_id_value)
            if picture_set is None:
                # Validated at commit; a race that deletes it here is logged, not
                # silently ignored, and does not fail the (already-done) import.
                logger.error(
                    "PictureImportTask: set %s vanished before association "
                    "(staging_id=%s)",
                    set_id_value,
                    self._staging_id,
                )
                return
            member_ids: list[int] = []
            for pic in session.exec(select(Picture).where(Picture.id.in_(ids))).all():
                exists = session.exec(
                    select(PictureSetMember).where(
                        PictureSetMember.set_id == set_id_value,
                        PictureSetMember.picture_id == pic.id,
                    )
                ).first()
                if exists is None:
                    session.add(
                        PictureSetMember(set_id=set_id_value, picture_id=pic.id)
                    )
                member_ids.append(int(pic.id))
            # Issue #125: the set may belong to several projects, so the imported
            # pictures join *all* of them, not just the primary FK.
            reconcile_entity_projects_change(
                session,
                picture_ids=member_ids,
                ensure_project_ids=picture_set_project_ids(session, set_id_value),
                remove_project_ids=[],
            )
            session.commit()

        self._db.run_task(
            apply_set,
            imported_ids,
            self._set_id,
            priority=DBPriority.IMMEDIATE,
        )

    def _apply_character(self, imported_ids: list[int]) -> None:
        """Associate imported pictures with the drop target character.

        Fresh imports have no faces yet, so this uses the same deferral the
        ``POST /characters/{id}/faces`` handler uses for face-less pictures:
        set ``Picture.pending_character_id`` (consumed by
        ``Vault._process_pending_character_assignments`` once face extraction
        runs), plus project reconciliation when the character has a project."""
        if self._character_id is None:
            return

        def apply_character(session: Session, ids: list[int], character_id_value: int):
            character = session.get(Character, character_id_value)
            if character is None:
                logger.error(
                    "PictureImportTask: character %s vanished before association "
                    "(staging_id=%s)",
                    character_id_value,
                    self._staging_id,
                )
                return
            pending_ids: list[int] = []
            for pic in session.exec(select(Picture).where(Picture.id.in_(ids))).all():
                pic.pending_character_id = character_id_value
                session.add(pic)
                pending_ids.append(int(pic.id))
            # Issue #125: the character may belong to several projects, so the
            # imported pictures join *all* of them, not just the primary FK.
            reconcile_entity_projects_change(
                session,
                picture_ids=pending_ids,
                ensure_project_ids=character_project_ids(session, character_id_value),
                remove_project_ids=[],
            )
            session.commit()

        self._db.run_task(
            apply_character,
            imported_ids,
            self._character_id,
            priority=DBPriority.IMMEDIATE,
        )

    def _cleanup_staging_dir(self) -> None:
        """Remove the staging directory once the import has finished."""
        if not self._staging_dir:
            return
        if not os.path.isdir(self._staging_dir):
            return
        try:
            shutil.rmtree(self._staging_dir)
            logger.debug(
                "PictureImportTask: removed staging dir %s (staging_id=%s)",
                self._staging_dir,
                self._staging_id,
            )
        except Exception as exc:
            logger.warning(
                "PictureImportTask: failed to remove staging dir %s "
                "(staging_id=%s): %s",
                self._staging_dir,
                self._staging_id,
                exc,
            )
