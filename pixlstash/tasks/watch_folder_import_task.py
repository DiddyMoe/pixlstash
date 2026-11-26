import io
import os
import stat
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from sqlmodel import Session, select

from pixlstash.database import DBPriority
from pixlstash.db_models.import_folder import ImportFolder
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.tag import Tag, TAG_PENDING_SENTINEL
from pixlstash.utils.caption_file_utils import (
    attach_sidecars,
    is_recorded_sidecar_shape,
)
from pixlstash.utils.image_processing.image_utils import ImageUtils
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
from pixlstash.services import import_dedup_service, import_publish_service
from pixlstash.services.layout_move_service import resolve_placement
from pixlstash.tasks.base_task import BaseTask
from pixlstash.utils.links import is_link_or_missing
from pixlstash.utils.reference_folder_validator import (
    canonical_path,
    validate_reference_folder_path,
)
from pixlstash.utils.sql_chunking import in_json


logger = get_logger(__name__)

# Candidates are decided and inserted in batches of this many, each batch's
# look again and insert under the import lock, so the rows of a long pass reach
# the library as it goes.
_IMPORT_BATCH = 64

# Reading a watched file and delete-after-import both go through directory
# descriptors where the platform has them (Linux, macOS): every directory from
# "/" down to the file is opened with O_NOFOLLOW, and so is the file itself, so
# a link swapped into the path after the checks makes the open fail instead of
# steering the read or the unlink outside the watch folder. Windows has no
# dir_fd support; there the file is opened by its checked path and then
# confirmed to be the file that path names (see `_open_checked_by_path`).
_UNLINK_BY_DIR_FD = (
    os.unlink in os.supports_dir_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NONBLOCK")
)


class _NotTheCheckedFile(OSError):
    """The open file is not the regular file the checks passed."""


@dataclass(frozen=True)
class _ReadIdentity:
    """The file an import read, as it stood when it was read.

    Delete-after-import removes a file only when the name still leads, inside
    the same watch folder, to this exact file: same device, inode, size and
    modification time. A file replaced or rewritten after the read is not the
    one the library holds a copy of.
    """

    root_real: str
    real_path: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int

    def matches(self, st: os.stat_result) -> bool:
        return (
            stat.S_ISREG(st.st_mode)
            and st.st_dev == self.st_dev
            and st.st_ino == self.st_ino
            and st.st_size == self.st_size
            and st.st_mtime_ns == self.st_mtime_ns
        )


def _is_within(root: str, path: str) -> bool:
    """Whether resolved *path* is *root* or under it, by path components."""
    root_c = os.path.normcase(root)
    path_c = os.path.normcase(path)
    try:
        return os.path.commonpath([root_c, path_c]) == root_c
    except ValueError:
        # Different drives (Windows) or a relative path: not inside.
        return False


def _link_on_the_way_to(root: str) -> str | None:
    """The first directory on the path to *root* that is a link, or ``None``.

    A link's target can be changed at any time without PixlStash seeing it,
    so a watch folder reached through one leads wherever the link points at
    the moment of the scan: an import is harmless, a delete is not.
    A Windows junction counts as a link: ``os.path.islink`` says False for
    one, and making one needs only write access to the parent directory.
    """
    path = os.path.abspath(root)
    while True:
        if is_link_or_missing(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _resolved_source(
    file_path: str, root: str | None, image_root: str | None
) -> tuple[str, str | None]:
    """The resolved path of a watched file and why it may not be used, or
    ``None`` for the reason when it may.

    Resolved on every call: a watch folder that is a link, or a link inside
    one, can be re-pointed at any time, so the answer at discovery says
    nothing about the answer at read or delete time. A reader opens the
    resolved path returned here with `_open_checked`, never *file_path*.
    """
    real_path = os.path.realpath(file_path)
    if root:
        root_real = os.path.realpath(root)
        if not _is_within(root_real, real_path):
            return (
                real_path,
                f"it resolves to {real_path}, outside the watch folder {root_real}",
            )
        refusal = validate_reference_folder_path(root_real)
        if refusal:
            return real_path, refusal
    refusal = validate_reference_folder_path(real_path)
    if refusal:
        return real_path, refusal
    if image_root and _is_within(canonical_path(image_root), real_path):
        return real_path, f"it resolves to {real_path}, inside the library's own folder"
    return real_path, None


def _source_refusal(file_path: str, root: str | None, image_root: str | None):
    """Why a watched file may not be read or removed now, or ``None``."""
    return _resolved_source(file_path, root, image_root)[1]


def _caption_link_refusal(
    file_path: str,
    root: str | None,
    image_root: str | None,
    links_by_dir: dict[str, list[str]],
) -> str | None:
    """Why the caption files beside *file_path* may not be read, or ``None``.

    The caption reader opens the files named after the picture by path, so a
    caption-shaped link leading out of the watch folder would bring that
    file's text into the library as tags or a description. The picture is
    still imported; its captions are not read. *links_by_dir* caches each
    directory's links for one pass.
    """
    directory = os.path.dirname(file_path)
    links = links_by_dir.get(directory)
    if links is None:
        with os.scandir(directory) as entries:
            links = [entry.path for entry in entries if entry.is_symlink()]
        links_by_dir[directory] = links
    # Lower-cased: the reader finds a caption by name, which matches in any
    # case on Windows and macOS, so this must not miss a link that differs
    # from the picture's name only in case.
    for link in links:
        if not is_recorded_sidecar_shape(file_path.lower(), link.lower()):
            continue
        refusal = _source_refusal(link, root, image_root)
        if refusal is not None:
            return f"the caption file {link} is refused: {refusal}"
    return None


def _open_parent_by_dir_fd(real_path: str) -> tuple[int, str]:
    """Open the directory holding *real_path* without following any link.

    Returns the directory's descriptor, which the caller closes, and the last
    name of *real_path*. The walk starts at "/" and opens each component with
    O_NOFOLLOW | O_DIRECTORY, so a component swapped for a link fails with
    ELOOP or ENOTDIR instead of leading elsewhere (open(2): O_NOFOLLOW checks
    only the last component, which is why every directory is opened in turn).
    """
    parent, name = os.path.split(real_path)
    components = [part for part in parent.split(os.sep) if part]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in components:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except BaseException:
        os.close(fd)
        raise
    return fd, name


def _open_checked_by_dir_fd(real_path: str) -> BinaryIO:
    """Open *real_path* for reading through O_NOFOLLOW descriptors.

    The file itself is opened with O_NOFOLLOW too, and O_NONBLOCK so a FIFO
    put in its place cannot hang the import; the open file must then be a
    regular file.
    """
    dir_fd, name = _open_parent_by_dir_fd(real_path)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    handle = os.fdopen(fd, "rb")
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _NotTheCheckedFile(f"{real_path} is not a regular file")
    except BaseException:
        handle.close()
        raise
    return handle


def _open_checked_by_path(real_path: str) -> BinaryIO:
    """Open *real_path* for reading where there is no dir_fd (Windows).

    Opened by path, then confirmed: the open file is a regular file, the
    path's own ``lstat`` names that same file (device and file index; a link
    at the name has its own index, so it never matches), and the path still
    resolves to itself, so no link or junction lies on it. A link swapped in
    for the open and out again before the ``lstat`` leaves the indexes
    unequal; what is not ruled out is a parent link in place for the open and
    the ``lstat`` and gone before the realpath. Windows opens without
    FILE_SHARE_DELETE, so the open file cannot be renamed or removed while it
    is held.
    """
    handle = open(real_path, "rb")
    try:
        opened = os.fstat(handle.fileno())
        named = os.lstat(real_path)
        if not (
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            and os.path.realpath(real_path) == real_path
        ):
            raise _NotTheCheckedFile(f"{real_path} changed while it was opened")
    except BaseException:
        handle.close()
        raise
    return handle


def _open_checked(real_path: str) -> BinaryIO:
    """Open the resolved path a check passed, as the file it named.

    Every byte an import hashes, copies or reads as a caption comes from the
    handle returned here, never from the name again: a link swapped into the
    path after the check makes the open fail rather than read elsewhere.
    """
    if _UNLINK_BY_DIR_FD:
        return _open_checked_by_dir_fd(real_path)
    return _open_checked_by_path(real_path)


def _caption_reader(root: str | None, image_root: str | None):
    """A caption reader for `attach_sidecars` that checks each file it reads.

    The caption file is resolved and checked as the picture was, and its text
    is read from the file the check passed. ``None`` (not recorded, tried
    again next scan) when it is refused or cannot be read.
    """

    def read_text(path: str) -> str | None:
        real_path, refusal = _resolved_source(path, root, image_root)
        if refusal is not None:
            logger.warning("Not reading caption file %s: %s", path, refusal)
            return None
        try:
            with _open_checked(real_path) as handle:
                with io.TextIOWrapper(
                    handle, encoding="utf-8-sig", errors="replace"
                ) as text:
                    return text.read()
        except OSError as exc:
            logger.warning("Could not read caption file %s: %s", path, exc)
            return None

    return read_text


def _read_identity(source: BinaryIO, real_path: str, root: str) -> _ReadIdentity:
    """Capture what delete-after-import must find again before it unlinks:
    the open file's own identity, not whatever its name leads to now."""
    st = os.fstat(source.fileno())
    return _ReadIdentity(
        root_real=os.path.realpath(root),
        real_path=real_path,
        st_dev=st.st_dev,
        st_ino=st.st_ino,
        st_size=st.st_size,
        st_mtime_ns=st.st_mtime_ns,
    )


def _unchanged_since_read(identity: _ReadIdentity) -> bool:
    """Whether the read file is still there, unchanged, after the copy."""
    try:
        return identity.matches(os.stat(identity.real_path, follow_symlinks=False))
    except FileNotFoundError:
        return False


def _unlink_by_dir_fd(identity: _ReadIdentity) -> str | None:
    """Unlink the read file through O_NOFOLLOW directory descriptors.

    Returns why the file was kept, or ``None`` once it is removed. The walk
    starts at "/" and follows the resolved path captured at read time, so no
    component can be a link: a component swapped for one fails with ELOOP or
    ENOTDIR. The last name is checked with ``fstatat(AT_SYMLINK_NOFOLLOW)``
    against the read identity and removed with ``unlinkat`` on the same
    directory descriptor.
    """
    fd, name = _open_parent_by_dir_fd(identity.real_path)
    try:
        st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not identity.matches(st):
            return "it is no longer the file that was imported"
        os.unlink(name, dir_fd=fd)
        return None
    finally:
        os.close(fd)


def _unlink_by_path(identity: _ReadIdentity) -> str | None:
    """The same checks by path, for platforms without dir_fd (Windows).

    A rename between the ``lstat`` and the ``remove`` is not ruled out here;
    the checks above it have just confirmed the path stays inside the watch
    folder and allowed by the path rules.
    """
    st = os.lstat(identity.real_path)
    if not identity.matches(st):
        return "it is no longer the file that was imported"
    os.remove(identity.real_path)
    return None


def _delete_imported_source(
    file_path: str,
    root: str,
    identity: _ReadIdentity,
    image_root: str | None,
) -> bool:
    """Remove a watched file the library now holds a copy of.

    Never through a watch folder reached by a link: where it leads was not
    checked when the folder was registered, only where the link pointed then.
    Every other check runs again here, at delete time: the resolved path is
    still inside the watch folder, the folder still resolves where it did at
    read time, the path rules still allow both, and the file is the exact one
    that was read. Anything else keeps the file and says why.
    """
    link = _link_on_the_way_to(root)
    refusal = (
        f"the watch folder is reached through the link {link}, which can be "
        "re-pointed at any time; files are only deleted from a folder that is "
        "not"
        if link is not None
        else _source_refusal(file_path, root, image_root)
    )
    if refusal is None and os.path.realpath(root) != identity.root_real:
        refusal = (
            f"the watch folder now resolves to {os.path.realpath(root)}, "
            f"not {identity.root_real} where it was read"
        )
    if refusal is None and os.path.realpath(file_path) != identity.real_path:
        refusal = f"it now resolves to {os.path.realpath(file_path)}"
    if refusal is None:
        try:
            if _UNLINK_BY_DIR_FD:
                refusal = _unlink_by_dir_fd(identity)
            else:
                refusal = _unlink_by_path(identity)
        except OSError as exc:
            refusal = f"it could not be checked or removed: {exc}"
    if refusal is not None:
        logger.warning(
            "Kept watched file %s after importing it, not deleting it: %s",
            file_path,
            refusal,
        )
        return False
    return True


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
        # Whether ``_run_task`` has begun, under ``_run_lock``: a task cancelled
        # off the queue never runs, so ``on_cancel`` hands its paths back.
        self._run_lock = threading.Lock()
        self._run_started = False

    def on_cancel(self) -> None:
        self._stop_event.set()
        with self._run_lock:
            if self._run_started:
                # The pass hands back what it did not decide when it stops.
                return
        # Cancelled before it ran. Every candidate was recorded as dispatched
        # when this task was made; left so, the finder would not offer one
        # again until the file changes on disk.
        self._hand_back(
            [c.get("file_path") for c in self._candidate_files],
            "the task was cancelled before it ran",
        )

    def _hand_back(self, paths: list, reason: str) -> None:
        """Tell the finder to forget *paths*, so a later scan offers them again.

        For the candidates this pass did not import, skip or refuse: a file
        whose read or insert failed, and every candidate a failure or a cancel
        left undecided. A file handed back that is in the library by then is
        skipped as a match on the next pass, so handing back too many costs a
        hash, while handing back too few loses the file until it changes.
        """
        paths = [path for path in paths if path]
        if not paths or self._finder is None:
            return
        self._finder.discard_seen_paths(paths)
        logger.info(
            "WatchFolderImportTask %s: %d watched file(s) will be offered again "
            "on a later scan: %s",
            self.id,
            len(set(paths)),
            reason,
        )

    def _build_batch(
        self,
        batch: list[dict],
        failed_paths: list[str],
        links_by_dir: dict[str, list[str]],
        published: import_publish_service.PublishedPictures,
        compared: set[int],
    ):
        """Hash, match and copy one batch of candidates into the library.

        Runs before the import lock is taken (see :meth:`_run_task`), so no
        other import waits on these reads and copies. Returns the new
        pictures, their stack assignments, the files to delete once the
        batch's rows are in the library (each with its picture), the watched
        file each new picture was copied from (by ``id`` of the picture), and
        how many candidates it reached (fewer than the batch when cancelled).
        *links_by_dir* is the pass's cache for `_caption_link_refusal`. Every
        file copied into the library is recorded in *published*, and the id of
        every candidate a file was compared with in *compared*.
        """
        new_pictures = []
        stack_assignments = []
        delete_paths = []
        source_of: dict[int, str] = {}
        # The pictures this batch copied into the library, by sampled hash and
        # size, each with its full hash once read. The library lookup cannot
        # see them until the batch's insert, so a second file with the same
        # content is matched against them here.
        batch_copies: dict[import_dedup_service.ContentKey, list[list]] = {}
        walked = len(batch)
        for index, candidate in enumerate(batch):
            if self._stop_event.is_set():
                logger.debug(
                    "WatchFolderImportTask cancelled, stopping early at task %s",
                    self.id,
                )
                walked = index
                break
            file_path = candidate.get("file_path")
            if not file_path:
                continue
            watch_root = candidate.get("import_source_folder")
            if not isinstance(watch_root, str) or not watch_root:
                watch_root = None

            real_path, refusal = _resolved_source(
                file_path, watch_root, self._db.image_root
            )
            if refusal is not None:
                logger.warning(
                    "Not importing watched file %s: %s",
                    file_path,
                    refusal,
                )
                self._processed_count += 1
                continue

            # Everything below reads the file the check above passed, through
            # one handle opened without following links: the hash, the size,
            # the copy into the library and the identity delete-after-import
            # looks for. The name is not opened again: a link swapped in after
            # the check would lead the read anywhere.
            with ExitStack() as held:
                # Delete-after-import needs the folder the file was read from and
                # the identity of what was read. A candidate without its folder,
                # or a file that is itself a link (unlinking it would not remove
                # what was imported), is imported and kept.
                identity = None
                wants_delete = bool(candidate.get("delete_after_import", False))
                try:
                    source = held.enter_context(_open_checked(real_path))
                    if wants_delete and watch_root is not None:
                        if os.path.islink(file_path):
                            logger.info(
                                "Watched file %s is a link; importing it without "
                                "deleting it.",
                                file_path,
                            )
                        else:
                            identity = _read_identity(source, real_path, watch_root)
                    caption_refusal = _caption_link_refusal(
                        file_path, watch_root, self._db.image_root, links_by_dir
                    )
                    pixel_sha = ImageUtils.calculate_hash_from_open_file(
                        source, source_label=file_path
                    )
                    size_bytes = os.fstat(source.fileno()).st_size
                except Exception as exc:
                    logger.warning(
                        "Failed to open or hash watched file %s: %s "
                        "(will retry on next scan)",
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
                # A read on the read path, not the writer queue. Earlier batches
                # and other lock holders committed before the lock was
                # released, and a WAL read that starts after a commit observes
                # it; one that commits after this read is found by the look
                # under the lock before the insert
                # (:meth:`_without_imported_meanwhile`).
                candidates = self._db.run_immediate_read_task(
                    import_dedup_service.load_match_candidates_in_session,
                    [(pixel_sha, size_bytes)],
                )
                compared.update(row.id for rows in candidates.values() for row in rows)
                existing = None
                earlier_copies = batch_copies.get((pixel_sha, size_bytes), [])
                full_sha = None
                if candidates or earlier_copies:
                    try:
                        full_sha = ImageUtils.calculate_full_hash_from_open_file(source)
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
                    if candidates:
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
                earlier = self._earlier_copy(earlier_copies, full_sha)
                if earlier is not None:
                    # Kept on disk, as a file whose content the library holds
                    # already is: the copy made from the earlier file is the
                    # one the library will have.
                    logger.info(
                        "Watched file %s has the content of a file imported "
                        "earlier in this pass (%s); skipping it",
                        file_path,
                        earlier.file_path,
                    )
                    self._processed_count += 1
                    continue

                pic = None
                try:
                    pic = ImageUtils.create_picture_from_file(
                        image_root_path=self._db.image_root,
                        source_file_path=file_path,
                        pixel_sha=pixel_sha,
                        source_file=source,
                        # Placement on write (v1.11 Phase 4b). A watch folder knows
                        # nothing about the picture's projects or people, so this is
                        # the unfiled folder in a laid-out library and nothing at
                        # all in one without a layout. The picture leaves the moment
                        # something files it.
                        subfolder=resolve_placement(self._db),
                    )
                    published.add(pic)
                    pic.imported_at = datetime.now(timezone.utc)
                    import_source_folder = candidate.get("import_source_folder")
                    if isinstance(import_source_folder, str) and import_source_folder:
                        pic.import_source_folder = import_source_folder

                    # Detect tags + description sidecars next to the source image.
                    # Import folders have no per-folder suffix config, so probe the
                    # known conventions; any tags are stashed on the row for the
                    # inserter below.
                    if caption_refusal is None:
                        attach_sidecars(
                            pic,
                            file_path,
                            read_text=_caption_reader(watch_root, self._db.image_root),
                        )
                    else:
                        logger.warning(
                            "Importing watched file %s without its captions: %s",
                            file_path,
                            caption_refusal,
                        )

                    # A manual workflow's run tags its output's name (#1688).
                    pic.run_workflow_id = parse_workflow_tag_from_filename(file_path)

                    stack_id, source_id = parse_stack_tags_from_filename(file_path)
                    to_delete = None
                    if identity is not None:
                        # The file must not have changed while it was copied: a
                        # rewrite during the read means the copy may not hold it.
                        if _unchanged_since_read(identity):
                            to_delete = (file_path, watch_root, identity)
                        else:
                            logger.warning(
                                "Watched file %s changed or vanished while it was "
                                "imported; keeping it.",
                                file_path,
                            )
                    elif wants_delete and watch_root is None:
                        logger.warning(
                            "Watched file %s has no watch folder to check it "
                            "against; keeping it.",
                            file_path,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to import watched file %s: %s (will retry on next scan)",
                        file_path,
                        exc,
                    )
                    failed_paths.append(file_path)
                    if pic is not None:
                        # Copied into the library but never to be inserted.
                        published.withdraw(pic, exc)
                    continue
                finally:
                    self._processed_count += 1

                # Recorded only once nothing above can fail, so a file that
                # failed is in neither the batch nor its copies.
                new_pictures.append(pic)
                if stack_id or source_id:
                    stack_assignments.append((pic, stack_id, source_id))
                if to_delete is not None:
                    delete_paths.append((pic, *to_delete))
                source_of[id(pic)] = file_path
                batch_copies.setdefault((pixel_sha, size_bytes), []).append(
                    [full_sha, pic]
                )

        return new_pictures, stack_assignments, delete_paths, source_of, walked

    def _without_imported_meanwhile(
        self,
        new_pictures: list[Picture],
        stack_assignments: list[tuple],
        delete_paths: list[tuple],
        source_of: dict[int, str],
        published: import_publish_service.PublishedPictures,
        compared: set[int],
    ) -> tuple[list[Picture], list[tuple], list[tuple]]:
        """Drop the built pictures whose content an import added meanwhile.

        Call under :func:`import_dedup_service.serialised_import`, right before
        :meth:`_insert_batch`. :meth:`_build_batch` looked each file up before
        it took the lock, so an import that inserted the same content since
        is found here (``import_publish_service.withdraw_imported_meanwhile``)
        and its copy withdrawn. The watched file is kept, as a file whose
        content the library already holds is. The candidates the batch was
        compared with before the lock (*compared*) are not read again.
        """
        matches = import_publish_service.withdraw_imported_meanwhile(
            self._db, published, [(pic, None) for pic in new_pictures], compared
        )
        dropped: set[int] = set()
        for pic, match in zip(new_pictures, matches):
            if match is None:
                continue
            dropped.add(id(pic))
            logger.info(
                "Not importing watched file %s: picture %d%s with its content "
                "was imported while this pass read the file; keeping the file.",
                source_of.get(id(pic), pic.file_path),
                match.id,
                " (in the Scrapheap)" if match.deleted else "",
            )
        if not dropped:
            return new_pictures, stack_assignments, delete_paths
        return (
            [pic for pic in new_pictures if id(pic) not in dropped],
            [entry for entry in stack_assignments if id(entry[0]) not in dropped],
            [entry for entry in delete_paths if id(entry[0]) not in dropped],
        )

    def _earlier_copy(self, earlier_copies: list[list], full_sha) -> Picture | None:
        """The picture this batch copied with the full hash *full_sha*, if any.

        *earlier_copies* are ``[full hash or None, picture]`` for the pictures
        the batch copied with the same sampled hash and size. A full hash not
        read yet is read from the picture's copy in the library, which is the
        file as it was read. A copy that cannot be read is not a match: an
        extra picture can be removed later, a skipped file is lost.
        """
        for entry in earlier_copies:
            if entry[0] is None:
                path = ImageUtils.resolve_picture_path(
                    self._db.image_root, entry[1].file_path
                )
                try:
                    entry[0] = ImageUtils.calculate_full_hash_from_file_path(path)
                except OSError as exc:
                    logger.warning(
                        "Could not read %s, copied into the library earlier in "
                        "this pass, to compare a file with it: %s",
                        path,
                        exc,
                    )
                    continue
            if entry[0] == full_sha:
                return entry[1]
        return None

    def _insert_batch(
        self,
        new_pictures,
        stack_assignments,
        published: import_publish_service.PublishedPictures,
    ) -> list[Picture]:
        """Insert one batch's new pictures and apply their stack assignments.

        *published* is told the moment the rows commit, so a later failure
        (the stack assignments) does not remove the files of saved pictures.
        """
        if not new_pictures:
            return []

        def insert_pictures(session: Session, pictures: list[Picture]):
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

        inserted = self._db.run_task(
            insert_pictures,
            new_pictures,
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
        return inserted or []

    def _run_task(self):
        with self._run_lock:
            self._run_started = True
        # Candidate paths that failed to process this pass. They are handed
        # back to the finder at the end so it forgets them and retries on a
        # later scan, which keeps a transient hash/import error (e.g. a
        # momentary PermissionError on a just-copied file) from dropping the
        # file forever.
        failed_paths: list[str] = []
        changed = []
        imported_ids = []
        links_by_dir: dict[str, list[str]] = {}
        # Candidates before this index were imported, skipped, refused or put
        # in ``failed_paths``. Those from it on are handed back when the pass
        # ends early: a cancel, or a batch whose insert failed.
        decided = 0

        try:
            for start in range(0, len(self._candidate_files), _IMPORT_BATCH):
                if self._stop_event.is_set():
                    break
                batch = self._candidate_files[start : start + _IMPORT_BATCH]
                # The files are read, hashed and copied before the import lock
                # is taken; the lock covers the look again and the insert of
                # each batch, as on every other import path: an overlapping
                # import of the same content (another watch folder, an upload,
                # a staged import) looks up only after this batch's pictures
                # are in the library, and none waits on this batch's
                # disk reads. A file copied into the library whose row does
                # not commit is removed again.
                with import_publish_service.PublishedPictures(
                    self._db.image_root, "Watch-folder import"
                ) as published:
                    compared: set[int] = set()
                    (
                        new_pictures,
                        stack_assignments,
                        delete_paths,
                        source_of,
                        walked,
                    ) = self._build_batch(
                        batch, failed_paths, links_by_dir, published, compared
                    )
                    with import_dedup_service.serialised_import():
                        new_pictures, stack_assignments, delete_paths = (
                            self._without_imported_meanwhile(
                                new_pictures,
                                stack_assignments,
                                delete_paths,
                                source_of,
                                published,
                                compared,
                            )
                        )
                        inserted = self._insert_batch(
                            new_pictures, stack_assignments, published
                        )
                decided = start + walked

                for pic in inserted:
                    if getattr(pic, "id", None) is None:
                        continue
                    imported_ids.append(pic.id)
                    changed.append((Picture, pic.id, "imported_at", pic.imported_at))

                for _pic, file_path, watch_root, identity in delete_paths:
                    _delete_imported_source(
                        file_path, watch_root, identity, self._db.image_root
                    )

            if imported_ids:
                logger.info(
                    "Added %d new pictures from watch folders.", len(imported_ids)
                )

            if self._last_checked_updates:

                def update_last_checked(session: Session, updates: dict[int, float]):
                    folders = session.exec(
                        select(ImportFolder).where(
                            in_json(ImportFolder.id, updates.keys())
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
        finally:
            self._hand_back(failed_paths, "they could not be read or imported")
            self._hand_back(
                [c.get("file_path") for c in self._candidate_files[decided:]],
                "the pass stopped before it decided them",
            )

        return {
            "changed_count": len(changed),
            "changed": changed,
            "imported_picture_ids": imported_ids,
            "candidate_count": self._total_candidates,
        }
