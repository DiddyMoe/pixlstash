import os
import stat
import threading
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models.import_folder import ImportFolder
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.tag import Tag, TAG_PENDING_SENTINEL
from pixlstash.utils.caption_file_utils import attach_sidecars
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.links import is_link_or_missing
from pixlstash.utils.quality.tag_exclusion import (
    filter_tag_names,
    session_tag_exclusions,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.stacking import (
    assign_picture_to_stack,
    get_or_create_stack_for_picture,
    parse_stack_tags_from_filename,
    parse_workflow_tag_from_filename,
)
from pixlstash.services import import_dedup_service
from pixlstash.services.layout_move_service import resolve_placement
from pixlstash.tasks.base_task import BaseTask


logger = get_logger(__name__)

# Delete-after-import reaches the file through directory descriptors where the
# platform has them: every folder from "/" down is opened with O_NOFOLLOW, so a
# link anywhere on the path fails the walk instead of steering the unlink
# elsewhere. Windows has no dir_fd; there each folder on the path is checked
# for a link, a junction included, before the remove.
_UNLINK_BY_DIR_FD = (
    os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
)


def _is_file_read(read: os.stat_result, now: os.stat_result) -> bool:
    """Whether *now* is the regular file *read* was taken from, unchanged."""
    return (
        stat.S_ISREG(now.st_mode)
        and now.st_dev == read.st_dev
        and now.st_ino == read.st_ino
        and now.st_size == read.st_size
        and now.st_mtime_ns == read.st_mtime_ns
    )


def _open_parent_by_dir_fd(path: str) -> tuple[int, str]:
    """Open the folder holding absolute *path* without following any link.

    Returns its descriptor, which the caller closes, and *path*'s last name.
    O_NOFOLLOW checks only the last component of a path, so each folder is
    opened in turn from "/"; one that is a link fails with ELOOP or ENOTDIR.
    """
    parent, name = os.path.split(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parent.split(os.sep):
            if part:
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
    except BaseException:
        os.close(fd)
        raise
    return fd, name


def _delete_imported_source(file_path: str, read: os.stat_result) -> str | None:
    """Remove a watched file the library now holds a copy of.

    Returns why the file was kept, or ``None`` once it is removed. It goes
    only when its path still leads, through no link, to the file the import
    read: a link on the way can lead outside the watch folder, and a file
    replaced or rewritten since the read is not the one the library holds.

    Raises:
        OSError: the path could not be walked, checked or removed.
    """
    path = os.path.abspath(file_path)
    if _UNLINK_BY_DIR_FD:
        fd, name = _open_parent_by_dir_fd(path)
        try:
            if not _is_file_read(read, os.stat(name, dir_fd=fd, follow_symlinks=False)):
                return "it is no longer the file that was imported"
            os.unlink(name, dir_fd=fd)
        finally:
            os.close(fd)
        return None
    # A folder swapped for a link between this walk and the remove is not
    # ruled out here.
    folder = os.path.dirname(path)
    while True:
        if is_link_or_missing(folder):
            return f"the folder {folder} on its path is a link"
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    if not _is_file_read(read, os.lstat(path)):
        return "it is no longer the file that was imported"
    os.remove(path)
    return None


class WatchFolderImportTask(BaseTask):
    """Task that imports discovered files from watch folders."""

    def __init__(
        self,
        database,
        candidate_files: list[dict],
        total_candidates: int,
        last_checked_updates: Optional[dict[int, float]] = None,
        finder=None,
    ):
        super().__init__(
            task_type="WatchFolderImportTask",
            params={
                "candidate_count": len(candidate_files or []),
                "total_candidates": int(total_candidates or 0),
            },
        )
        self._db = database
        self._candidate_files = candidate_files or []
        # The finder that produced this task. Used to report back candidate
        # paths that failed to import so it can drop them from its seen-paths
        # set and re-discover them on a later scan (transient-failure retry).
        self._finder = finder
        self._last_checked_updates = {
            int(folder_id): float(last_checked)
            for folder_id, last_checked in (last_checked_updates or {}).items()
        }
        self._total_candidates = int(total_candidates or 0)
        self._processed_count = 0
        self._stop_event = threading.Event()

    def on_cancel(self) -> None:
        self._stop_event.set()

    def _run_task(self):
        new_pictures = []
        stack_assignments = []
        delete_paths = []
        # Candidate paths that failed to process this pass. They are reported to
        # the finder at the end so it forgets them and retries on a later scan,
        # which keeps a transient hash/import error (e.g. a momentary
        # PermissionError on a just-copied file) from dropping the file forever.
        failed_paths: list[str] = []

        for candidate in self._candidate_files:
            if self._stop_event.is_set():
                logger.debug(
                    "WatchFolderImportTask cancelled, stopping early at task %s",
                    self.id,
                )
                break
            file_path = candidate.get("file_path")
            if not file_path:
                continue

            try:
                # Taken before the reads, so a change while they run keeps
                # the file from delete-after-import.
                read_stat = os.lstat(file_path)
                pixel_sha = ImageUtils.calculate_hash_from_file_path(file_path)
                size_bytes = os.path.getsize(file_path)
            except Exception as exc:
                logger.warning(
                    "Failed to hash watched file %s: %s (will retry on next scan)",
                    file_path,
                    exc,
                )
                failed_paths.append(file_path)
                self._processed_count += 1
                continue

            # The sampled hash only narrows the search: a file is skipped only
            # when a picture has the same size and the same full content. A
            # match in the Scrapheap is skipped too and left there, because a
            # file kept in a watch folder is offered again after every restart
            # and restoring it each time would undo the owner's cleanup.
            candidates = self._db.run_task(
                import_dedup_service.load_match_candidates_in_session,
                [(pixel_sha, size_bytes)],
            )
            existing = None
            if candidates:
                try:
                    full_sha = ImageUtils.calculate_full_hash_from_file_path(file_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to read watched file %s in full: %s "
                        "(will retry on next scan)",
                        file_path,
                        exc,
                    )
                    failed_paths.append(file_path)
                    self._processed_count += 1
                    continue
                existing = import_dedup_service.confirmed_match(
                    candidates,
                    (pixel_sha, size_bytes, full_sha),
                    self._db.image_root,
                )
            if existing is not None:
                logger.debug(
                    "Already have picture %d with the content of %s, skipping",
                    existing.id,
                    file_path,
                )
                self._processed_count += 1
                continue

            try:
                pic = ImageUtils.create_picture_from_file(
                    image_root_path=self._db.image_root,
                    source_file_path=file_path,
                    pixel_sha=pixel_sha,
                    # Placement on write (v1.11 Phase 4b). A watch folder knows
                    # nothing about the picture's projects or people, so this is
                    # the unfiled folder in a laid-out library and nothing at
                    # all in one without a layout. The picture leaves the moment
                    # something files it.
                    subfolder=resolve_placement(self._db),
                )
                pic.imported_at = datetime.now(timezone.utc)
                import_source_folder = candidate.get("import_source_folder")
                if isinstance(import_source_folder, str) and import_source_folder:
                    pic.import_source_folder = import_source_folder

                # Detect tags + description sidecars next to the source image.
                # Import folders have no per-folder suffix config, so probe the
                # known conventions; any tags are stashed on the row for the
                # inserter below.
                attach_sidecars(pic, file_path)

                new_pictures.append(pic)

                # A manual workflow's run tags its output's name (#1688).
                pic.run_workflow_id = parse_workflow_tag_from_filename(file_path)

                stack_id, source_id = parse_stack_tags_from_filename(file_path)
                if stack_id or source_id:
                    stack_assignments.append((pic, stack_id, source_id))
                if bool(candidate.get("delete_after_import", False)):
                    delete_paths.append((file_path, read_stat))
            except Exception as exc:
                logger.warning(
                    "Failed to import watched file %s: %s (will retry on next scan)",
                    file_path,
                    exc,
                )
                failed_paths.append(file_path)
            finally:
                self._processed_count += 1

        changed = []
        imported_ids = []
        if new_pictures:

            def insert_pictures(
                session: Session,
                pictures: list[Picture],
                published: import_dedup_service.PublishedPictures,
            ):
                session.add_all(pictures)
                session.flush()
                exclusions = session_tag_exclusions(session)
                left_out = 0
                for pic in pictures:
                    # A sidecar naming only tags removed from tagging still
                    # tagged the picture: no sentinel, so the tagger leaves it.
                    sidecar_tags = getattr(pic, "_sidecar_tags", None)
                    if sidecar_tags and pic.id is not None:
                        permitted = filter_tag_names(sidecar_tags, exclusions)
                        left_out += len(sidecar_tags) - len(permitted)
                        for tag_str in permitted:
                            session.add(Tag(picture_id=pic.id, tag=tag_str))
                    else:
                        session.add(Tag(picture_id=pic.id, tag=TAG_PENDING_SENTINEL))
                if left_out:
                    logger.debug(
                        "WatchFolderImportTask: left out %d sidecar tag(s) "
                        "removed from tagging.",
                        left_out,
                    )
                session.commit()
                published.mark_committed()
                for pic in pictures:
                    session.refresh(pic)
                return pictures

            with import_dedup_service.PublishedPictures(
                self._db.image_root, "Watch-folder import"
            ) as published:
                for pic in new_pictures:
                    published.add(pic)
                inserted = self._db.run_task(
                    insert_pictures,
                    new_pictures,
                    published,
                    priority=DBPriority.IMMEDIATE,
                )

            if stack_assignments:

                def apply_stack_assignments(session: Session, assignments: list[tuple]):
                    for pic, stack_id, source_id in assignments:
                        if pic.id is None:
                            continue
                        if stack_id:
                            assign_picture_to_stack(session, pic.id, stack_id)
                            continue
                        if source_id:
                            resolved_stack_id = get_or_create_stack_for_picture(
                                session, source_id
                            )
                            if resolved_stack_id:
                                assign_picture_to_stack(
                                    session,
                                    pic.id,
                                    resolved_stack_id,
                                )

                self._db.run_task(
                    apply_stack_assignments,
                    stack_assignments,
                    priority=DBPriority.IMMEDIATE,
                )

            for pic in inserted or []:
                if getattr(pic, "id", None) is None:
                    continue
                imported_ids.append(pic.id)
                changed.append((Picture, pic.id, "imported_at", pic.imported_at))

            logger.info("Added %d new pictures from watch folders.", len(imported_ids))

            for file_path, read_stat in delete_paths:
                try:
                    kept = _delete_imported_source(file_path, read_stat)
                except OSError as exc:
                    kept = str(exc)
                if kept is not None:
                    logger.warning(
                        "Kept watched file %s after importing it, not deleting it: %s",
                        file_path,
                        kept,
                    )

        if self._last_checked_updates:

            def update_last_checked(session: Session, updates: dict[int, float]):
                folders = session.exec(
                    select(ImportFolder).where(
                        ImportFolder.id.in_(list(updates.keys()))
                    )
                ).all()
                for folder in folders:
                    next_ts = updates.get(int(folder.id))
                    if next_ts is None:
                        continue
                    folder.last_checked = float(next_ts)
                    session.add(folder)
                session.commit()

            self._db.run_task(
                update_last_checked,
                self._last_checked_updates,
                priority=DBPriority.IMMEDIATE,
            )

        if failed_paths and self._finder is not None:
            self._finder.discard_seen_paths(failed_paths)

        return {
            "changed_count": len(changed),
            "changed": changed,
            "imported_picture_ids": imported_ids,
            "candidate_count": self._total_candidates,
        }
