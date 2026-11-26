"""Registry of pictures whose image file cannot be decoded or reached.

GitHub issue #585 ("Invalid Images break tasks"): the background pipeline is
data-driven - every ``Missing*Finder`` re-selects any picture whose target
column is still unset. When a picture points at a corrupt/undecodable file the
task can never produce a value, nothing durably marks the row as done (the
image-embedding task leaves the embedding NULL on failure, which ``fetch_work``
treats as missing), so the same picture is picked up on every sweep forever.

The reporter accepts that such a file may be "ignored for the remaining
duration of the server process lifetime, or until it is modified". This
registry does that and outlasts the process: a thread-safe set of picture ids
that failed to decode, pinned to the file's ``(mtime, size)`` at the moment of
failure. A finder consults :meth:`is_suppressed` to skip the picture; when the
file is rewritten (the "still being written" / repaired case) its signature moves
and suppression lifts automatically, so the picture is retried.

**Only bytes that do not decode are pinned.** :meth:`mark_unprocessable`
decodes the file itself before it records anything, and records only a
permanent failure (``LoadFailure.permanent``). A read the operating system
refused (no permission, an I/O error, a missing file, a dropped share) or a
process that ran out of memory is not the file's fault, and pinning it to a
signature would keep a good picture out until somebody touched the file. Two
other kinds are held while a condition lasts instead, and lift the moment it
ends: *unreachable* (:meth:`mark_unreachable`, the file's location is not
mounted) and *unreadable* (:meth:`hold_unreadable`, the file is there and
refuses to be read).

The map in memory is the live copy, and every change to it is written through
to the ``unprocessableimage`` table (:class:`UnprocessableImage`) of the
database that owns the registry. The rows are read back on the registry's
first use, so a restart no longer decodes and fails every such picture once
more. Writes are queued, never waited on: the callers are task and finder
threads in the middle of other work, and the in-memory answer does not depend
on the write having landed. A registry built without a database (the unit
tests' stand-alone ones) keeps the map in memory only.
"""

import os
import threading
import time
from dataclasses import dataclass

from sqlmodel import delete, select

from pixlstash.db_models import Picture
from pixlstash.db_models.unprocessable_image import (
    KIND_UNDECODABLE,
    KIND_UNREACHABLE,
    KIND_UNREADABLE,
    UnprocessableImage,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.utils.image_processing.media_load import (
    LoadResult,
    file_read_error,
    load_pil_classified,
)

logger = get_logger(__name__)

#: Stands in for the (mtime, size) of an entry that has no signature because
#: its file could not be stat'd at all. Its own object, not ``None``: ``None``
#: already means "stat failed just now" in :meth:`_stat_signature`, and the two
#: must not be confused.
_UNREACHABLE = object()

#: Stands in for the (mtime, size) of an entry held because its file exists
#: and cannot be read. Distinct from `_UNREACHABLE`: the two are lifted by
#: different conditions (see `_entry_is_live`).
_UNREADABLE = object()

#: The row kind for each sentinel signature; anything else is undecodable.
_SENTINEL_KINDS = {_UNREACHABLE: KIND_UNREACHABLE, _UNREADABLE: KIND_UNREADABLE}
_KIND_SENTINELS = {kind: sentinel for sentinel, kind in _SENTINEL_KINDS.items()}


def _location_is_unreachable(file_path: str) -> bool:
    """``scrapheap_service.file_location_is_unreachable``, imported locally.

    Local because the import is a cycle: ``database`` imports this registry and
    ``scrapheap_service`` imports ``database``. One definition of "the parent
    directory is missing, so say nothing about the file" is worth the indirect
    import - the two must never drift, because between them they decide whether
    a picture is held or purged.
    """
    from pixlstash.services.scrapheap_service import file_location_is_unreachable

    return file_location_is_unreachable(file_path)


def file_signature(file_path: str) -> "tuple[int, int] | None":
    """``(mtime_ns, size)`` for *file_path*, or ``None`` if it cannot be stat'd.

    What an undecodable entry is pinned to, and the one definition of "the file
    changed" for any hold that is lifted the same way (``MissingThumbnailFinder``
    holds a file whose EXIF orientation cannot be read until it moves).
    """
    try:
        stat = os.stat(file_path)
    except OSError as exc:
        logger.debug("UnprocessableImageRegistry: cannot stat %s: %s", file_path, exc)
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _write_priority():
    """``DBPriority.LOW``, imported locally for the same cycle as above.

    ``database`` imports this module at its top. Every registry write is one
    priority so the writer applies them in the order they were queued: a mark,
    its lift and a re-mark of one picture must land in that order.
    """
    from pixlstash.database import DBPriority

    return DBPriority.LOW


# Ids per ``IN (...)`` in one DELETE, well under SQLite's bound-variable limit.
_DELETE_CHUNK = 500


def _read_rows(session, limit: int) -> list:
    return session.exec(
        select(
            UnprocessableImage.picture_id,
            UnprocessableImage.kind,
            UnprocessableImage.file_path,
            UnprocessableImage.mtime_ns,
            UnprocessableImage.size,
        )
        .order_by(UnprocessableImage.picture_id)
        .limit(limit)
    ).all()


def _write_row(session, picture_id: int, kind: str, file_path: str, mtime_ns, size):
    # The picture can be deleted between the mark and this write; its row
    # would then fail the foreign key, and there is nothing left to hold.
    if session.get(Picture, picture_id) is None:
        return
    session.merge(
        UnprocessableImage(
            picture_id=picture_id,
            kind=kind,
            file_path=file_path,
            mtime_ns=mtime_ns,
            size=size,
        )
    )
    session.commit()


def _delete_rows(session, picture_ids: list[int]) -> None:
    for start in range(0, len(picture_ids), _DELETE_CHUNK):
        chunk = picture_ids[start : start + _DELETE_CHUNK]
        session.exec(
            delete(UnprocessableImage).where(UnprocessableImage.picture_id.in_(chunk))
        )
    session.commit()


class UnprocessableImageRegistry:
    """Thread-safe ``picture_id -> (file_path, mtime_ns, size)`` map of held pictures.

    An *undecodable* picture is suppressed only while it is recorded here
    **and** its file's current ``(mtime_ns, size)`` still match what was
    captured when it was marked. Any change to the file (a rewrite, a repair,
    or the file being replaced) moves the signature and lifts suppression, so
    the picture flows back into the normal finder queue and is retried. An
    *unreachable* or *unreadable* picture is held while its condition lasts
    (see :meth:`_entry_is_live`).

    The map is bounded to *max_entries*, and written through to *database*
    when one is given (see the module docstring). Both the marking side
    (background task threads, on a decode failure) and the suppression side
    (finder threads, on every planning sweep) touch this map, so every method is
    guarded by a single lock. ``os.stat`` is performed under the lock: it is a
    fast syscall and contention is negligible (marks are rare, sweeps are cheap),
    and holding it removes any check-then-act race between marking and pruning.
    """

    # Corrupt images are pathological; this cap only bounds a runaway (e.g. a
    # whole reference tree of unreadable files) while staying far above any
    # realistic count.
    MAX_ENTRIES = 100_000

    # How long an `active_suppressed_ids()` answer is reused before every entry
    # is re-stat'd. The tag, face, thumbnail and embedding finders each rebuild
    # the set on every planner sweep, and `Vault` counts with it twice more -
    # six full walks of the map, back to back, on the planner thread. That is
    # free while the files are local and it is not free at all for an
    # *unreachable* entry, whose `_entry_is_live` waits on a dead mount for
    # every stat (#1206 item 9b): exactly the case the map fills up with. The
    # window is short enough that a repaired or remounted file is retried
    # within seconds, and any change to the map clears the cache outright, so
    # the staleness only ever delays a *retry* - never a suppression.
    ACTIVE_CACHE_TTL_S = 5.0

    def __init__(self, database=None, max_entries: int = MAX_ENTRIES) -> None:
        """Build an empty registry.

        Args:
            database: The ``VaultDatabase`` whose ``unprocessableimage`` rows
                back this registry, or ``None`` to keep the map in memory
                only. Its engine does not exist yet when it builds the
                registry, so the rows are read on first use, not here.
            max_entries: Cap on the number of entries held.
        """
        self._database = database
        self._max_entries = int(max_entries)
        self._lock = threading.Lock()
        # picture_id -> (file_path, mtime_ns, size)
        self._entries: dict[int, tuple[str, int, int]] = {}
        # (monotonic time, ids) of the last full re-validation, or None.
        self._active_cache: "tuple[float, frozenset[int]] | None" = None
        # Held across the one read of the stored rows, so every caller that
        # arrives meanwhile waits for it instead of answering without them.
        self._load_lock = threading.Lock()
        self._loaded = database is None

    def _ensure_loaded(self) -> None:
        """Read the stored rows into the map, once, on the registry's first use.

        Runs on whichever thread asks first: a finder on the planner thread,
        a task worker marking a picture, or a synchronous route handler in
        the server's thread pool. None of the callers is the event loop or
        the database writer. The read goes through
        ``run_immediate_read_task``, which takes only the engine lock's read
        side and never queues on the writer, so it cannot wait on a writer
        task. A caller already inside an immediate read re-enters that lock.

        An entry already in the map is kept over its row: the map is the live
        copy. A failed read is raised to the caller and retried on the next
        use.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            rows = self._database.run_immediate_read_task(_read_rows, self._max_entries)
            with self._lock:
                for picture_id, kind, file_path, mtime_ns, size in rows:
                    sentinel = _KIND_SENTINELS.get(kind)
                    if sentinel is not None:
                        entry = (file_path, sentinel, sentinel)
                    else:
                        entry = (file_path, mtime_ns, size)
                    self._entries.setdefault(int(picture_id), entry)
                self._active_cache = None
            self._loaded = True
        if rows:
            logger.info(
                "UnprocessableImageRegistry: %d held picture(s) read back from "
                "the database.",
                len(rows),
            )

    def _store(self, picture_id: int, entry: tuple) -> None:
        """Queue the row for *entry*. Call with ``self._lock`` held, so the
        writer applies changes to one picture in the order the map made them."""
        if self._database is None:
            return
        file_path, mtime_ns, size = entry
        sentinel_kind = _SENTINEL_KINDS.get(mtime_ns)
        future = self._database.submit_task(
            _write_row,
            picture_id,
            sentinel_kind or KIND_UNDECODABLE,
            file_path,
            None if sentinel_kind else mtime_ns,
            None if sentinel_kind else size,
            priority=_write_priority(),
        )
        future.add_done_callback(
            lambda done: self._log_write_failure(done, "store", [picture_id])
        )

    def _unstore(self, picture_ids: list[int]) -> None:
        """Queue the deletion of *picture_ids*' rows. Call with ``self._lock`` held."""
        if self._database is None or not picture_ids:
            return
        future = self._database.submit_task(
            _delete_rows, picture_ids, priority=_write_priority()
        )
        future.add_done_callback(
            lambda done: self._log_write_failure(done, "delete", picture_ids)
        )

    @staticmethod
    def _log_write_failure(future, action: str, picture_ids: list[int]) -> None:
        """Done-callback: log a failed write. Runs on the writer thread, so it
        only logs. The map is unaffected; the row is out of step until the
        entry next changes, and a restart then reads the old state."""
        if future.cancelled():
            logger.warning(
                "UnprocessableImageRegistry: %s of %d row(s) (first id=%s) was "
                "cancelled before it ran.",
                action,
                len(picture_ids),
                picture_ids[0],
            )
            return
        error = future.exception()
        if error is not None:
            logger.warning(
                "UnprocessableImageRegistry: %s of %d row(s) (first id=%s) failed.",
                action,
                len(picture_ids),
                picture_ids[0],
                exc_info=error,
            )

    @staticmethod
    def _stat_signature(file_path: str) -> "tuple[int, int] | None":
        """Return ``(mtime_ns, size)`` for *file_path*, or ``None`` if it cannot be stat'd."""
        return file_signature(file_path)

    def mark_unprocessable(
        self,
        picture_id: "int | None",
        file_path: "str | None",
        *,
        reason: str = "",
        load: "SignedLoad | None" = None,
    ) -> bool:
        """Record *picture_id* as undecodable, pinned to *file_path*'s current signature.

        A picture already marked for the *same* file version is a no-op (and is not
        re-logged), so repeated sweeps of the same corrupt file do not spam the log.
        A picture whose file can no longer be stat'd is left unmarked - there is
        nothing to pin the suppression to, and the missing-file purge finder owns
        that case.

        **The caller's word is not enough.** The file is decoded here, after
        the stat, and the mark is recorded only when that decode fails
        permanently (:attr:`LoadFailure.permanent`: the bytes read and the
        decoder rejects them). A failure the operating system caused (EACCES,
        EIO, ENOENT, a stale handle), a ``MemoryError``, an exception nobody
        classified, or a file that decodes now, is refused with a warning:
        pinning any of those to a signature would hold a good picture out
        until the file happened to change. Every caller routes through here,
        so one that cannot tell a read failure from a decode failure (a loader
        that returns ``None`` for both) is still safe. The decode runs outside
        the lock, and the signature is taken before it, so a file rewritten
        meanwhile is pinned to its old signature and lifts on the next check.

        A caller that has already decoded the file through :func:`load_signed`
        passes that load as *load*, and the file is not decoded a second time:
        the mark uses the load's signature, taken before its decode as here, and
        its classified failure, held to the same rule.

        Args:
            picture_id: The picture that failed to decode.
            file_path: Absolute path to the file that could not be opened.
            reason: Short description of the failure, for the one-time log line.
            load: The caller's :func:`load_signed` of *file_path*, or ``None``
                to have the file decoded here.

        Returns:
            ``True`` if a new mark (or a re-mark for a changed file) was recorded.

        Raises:
            ValueError: *load* is a load of another file.
        """
        if picture_id is None or not file_path:
            return False
        if load is not None and load.file_path != str(file_path):
            raise ValueError(
                f"mark_unprocessable for picture id={picture_id}: the load given "
                f"is of {load.file_path!r}, not {str(file_path)!r}"
            )
        pid = int(picture_id)
        signature = (
            load.signature if load is not None else self._stat_signature(file_path)
        )
        if signature is None:
            logger.debug(
                "UnprocessableImageRegistry: not marking picture id=%s - file %s "
                "cannot be stat'd (leaving it to the missing-file purge).",
                pid,
                file_path,
            )
            return False
        mtime_ns, size = signature
        self._ensure_loaded()
        with self._lock:
            existing = self._entries.get(pid)
            if existing is not None and existing[1] == mtime_ns and existing[2] == size:
                return False  # already suppressed for this exact file version
        decoded = (
            load.result
            if load is not None
            else load_pil_classified(str(file_path), multiframe_count=1)
        )
        if decoded.failure is None or not decoded.failure.permanent:
            logger.warning(
                "UnprocessableImageRegistry: NOT marking picture id=%s path=%s "
                "undecodable (%s) - %s %s, so the failure is not the "
                "file's bytes; it will be retried.",
                pid,
                file_path,
                reason or "no reason given",
                "the caller's classified load"
                if load is not None
                else "a fresh decode",
                "succeeded"
                if decoded.failure is None
                else f"failed with {decoded.failure.value} ({decoded.error})",
            )
            return False
        with self._lock:
            existing = self._entries.get(pid)
            if existing is not None and existing[1] == mtime_ns and existing[2] == size:
                return False  # another thread marked this version meanwhile
            if existing is None and len(self._entries) >= self._max_entries:
                logger.warning(
                    "UnprocessableImageRegistry: cap (%d) reached; NOT suppressing "
                    "picture id=%s path=%s - it will keep being retried until an "
                    "existing entry is released.",
                    self._max_entries,
                    pid,
                    file_path,
                )
                return False
            self._entries[pid] = (str(file_path), mtime_ns, size)
            self._active_cache = None
            self._store(pid, self._entries[pid])
        logger.warning(
            "Unprocessable image: picture id=%s path=%s could not be decoded (%s); "
            "skipping it until the file changes.",
            pid,
            file_path,
            reason or "image could not be decoded",
        )
        return True

    def mark_unreachable(
        self, picture_id: "int | None", file_path: "str | None", *, reason: str = ""
    ) -> bool:
        """Record *picture_id* as unreachable: its file's whole LOCATION is gone.

        The other half of :meth:`mark_unprocessable`, which refuses a file it
        cannot ``stat`` and defers to the missing-file purge - and the purge
        deliberately refuses it too (``MissingFilePurgeTask`` skips a picture
        whose parent directory is missing, because purging would write
        ``file_removed=True`` and block a later restore). Between the two, a
        picture on an unplugged drive had no owner at all: every batch finder
        re-selected it on every sweep, produced nothing, and re-selected it
        again.

        Suppression is the right answer and deletion is not, because this
        state ends by itself: the entry lifts the moment the parent directory
        is back, so remounting the drive returns the picture to every stage
        with its pending work intact.

        Args:
            picture_id: The picture whose file could not be reached.
            file_path: Absolute path to the file, used to watch its parent.
            reason: Short description, for the one-time log line.

        Returns:
            ``True`` if a new mark was recorded.
        """
        if picture_id is None or not file_path:
            return False
        if not _location_is_unreachable(str(file_path)):
            # The directory is there, so the file is genuinely absent or
            # unopenable for some other reason - not this registry's case.
            return False
        pid = int(picture_id)
        self._ensure_loaded()
        with self._lock:
            if pid in self._entries:
                return False
            if len(self._entries) >= self._max_entries:
                logger.warning(
                    "UnprocessableImageRegistry: cap (%d) reached; NOT holding "
                    "picture id=%s path=%s as unreachable - it will keep being "
                    "retried until an existing entry is released.",
                    self._max_entries,
                    pid,
                    file_path,
                )
                return False
            # No signature: an unreachable file cannot be stat'd, and the
            # sentinel is what `_entry_is_live` reads to pick its predicate.
            self._entries[pid] = (str(file_path), _UNREACHABLE, _UNREACHABLE)
            self._active_cache = None
            self._store(pid, self._entries[pid])
        logger.warning(
            "Unreachable image: picture id=%s path=%s - its location is not "
            "mounted (%s); holding it out of the work finders until the "
            "directory is back. Nothing is deleted and no work is lost.",
            pid,
            file_path,
            reason or "parent directory missing",
        )
        return True

    def hold_unreadable(
        self, picture_id: "int | None", file_path: "str | None", *, reason: str = ""
    ) -> bool:
        """Hold *picture_id* out while its file exists and refuses to be read.

        For a stage with no retry hold of its own (the thumbnail finder), whose
        picture would otherwise be re-selected on every sweep while a read
        keeps failing: no permission, a bad sector, a share that answers stat
        and not read. It is not a verdict on the file. The entry is pinned to
        nothing and lives only while the file still stats and its first byte
        still cannot be read, so it lifts by itself, within
        :attr:`ACTIVE_CACHE_TTL_S`, the moment the file reads again - a
        ``chmod``, a remount, a repaired disk - with no restart and no change
        to the file.

        Refused (``False``) when the file cannot be stat'd (the missing-file
        purge owns that), when its location is unreachable
        (:meth:`mark_unreachable` owns that), when the file reads (there is
        nothing to hold), and when the picture already has an entry.

        Args:
            picture_id: The picture whose file could not be read.
            file_path: Absolute path to the file.
            reason: Short description, for the one-time log line.

        Returns:
            ``True`` if a new hold was recorded.
        """
        if picture_id is None or not file_path:
            return False
        pid = int(picture_id)
        file_path = str(file_path)
        if self._stat_signature(file_path) is None or _location_is_unreachable(
            file_path
        ):
            return False
        read_error = file_read_error(file_path)
        if read_error is None:
            return False
        self._ensure_loaded()
        with self._lock:
            if pid in self._entries:
                return False
            if len(self._entries) >= self._max_entries:
                logger.warning(
                    "UnprocessableImageRegistry: cap (%d) reached; NOT holding "
                    "picture id=%s path=%s as unreadable - it will keep being "
                    "retried until an existing entry is released.",
                    self._max_entries,
                    pid,
                    file_path,
                )
                return False
            self._entries[pid] = (file_path, _UNREADABLE, _UNREADABLE)
            self._active_cache = None
            self._store(pid, self._entries[pid])
        logger.warning(
            "Unreadable image: picture id=%s path=%s cannot be read (%s; %s); "
            "holding it out of the work finders until it reads again. The file "
            "is not marked as corrupt.",
            pid,
            file_path,
            reason or "read failed",
            read_error,
        )
        return True

    def _entry_is_live(self, path: str, mtime_ns: int, size: int) -> bool:
        """Whether a recorded entry still holds its picture out of the finders.

        Three kinds share one map. An *unreachable* entry lives while its
        location is still missing, and is dropped the moment the volume is
        back. An *unreadable* entry lives while the file still stats and still
        cannot be read, and is dropped the moment it reads (or is gone). An
        *undecodable* entry lives while the file is byte-for-byte the one that
        failed, and is dropped when it is rewritten or repaired.
        """
        if mtime_ns is _UNREACHABLE:
            return _location_is_unreachable(path)
        if mtime_ns is _UNREADABLE:
            return (
                self._stat_signature(path) is not None
                and file_read_error(path) is not None
            )
        signature = self._stat_signature(path)
        return signature is not None and signature == (mtime_ns, size)

    def is_suppressed(self, picture_id: "int | None") -> bool:
        """Return whether *picture_id* is currently suppressed.

        ``True`` only if the picture is recorded and its stored file's current
        ``(mtime_ns, size)`` still match the marked signature. If the file changed
        or can no longer be stat'd the stale entry is pruned and ``False`` is
        returned, so the picture is retried.

        The signature is re-checked against the registry's OWN stored path, so
        callers never read ``Picture.file_path``: that ORM attribute is often
        deferred and the candidate pictures are detached by the time a finder
        claims, so touching it there raised ``DetachedInstanceError`` (issue #585).
        """
        if picture_id is None:
            return False
        pid = int(picture_id)
        self._ensure_loaded()
        with self._lock:
            existing = self._entries.get(pid)
            if existing is None:
                return False
            path, mtime_ns, size = existing
            if self._entry_is_live(path, mtime_ns, size):
                return True
            # Rewritten, repaired, or the volume is back - retry it.
            self._entries.pop(pid, None)
            self._active_cache = None
            self._unstore([pid])
            return False

    def active_suppressed_ids(self) -> set[int]:
        """Return the set of currently-suppressed picture ids, pruning stale entries.

        Re-validates every stored file signature; entries whose file changed or
        disappeared are dropped. The answer is then reused for
        :attr:`ACTIVE_CACHE_TTL_S`, because six callers ask for it within one
        planner sweep and an unreachable entry costs a mount timeout per stat.
        Any write to the map drops the cache, so a newly marked picture is
        never served from a set built before it.
        """
        self._ensure_loaded()
        with self._lock:
            cached = self._active_cache
            if (
                cached is not None
                and time.monotonic() - cached[0] < self.ACTIVE_CACHE_TTL_S
            ):
                return set(cached[1])
            active: set[int] = set()
            lifted: list[int] = []
            for pid, (path, mtime_ns, size) in list(self._entries.items()):
                if self._entry_is_live(path, mtime_ns, size):
                    active.add(pid)
                else:
                    self._entries.pop(pid, None)
                    lifted.append(pid)
            self._unstore(lifted)
            # Stamped after the walk, not before: `_entry_is_live` can take a
            # while on a dead mount, and the window is meant to start when the
            # answer is known. The age is compared against the TTL on every
            # read rather than baked into a deadline, so lowering the TTL takes
            # effect at once.
            self._active_cache = (time.monotonic(), frozenset(active))
        return active

    def discard(self, picture_id: "int | None") -> None:
        """Forget *picture_id* (e.g. when the picture is deleted), and its row."""
        if picture_id is None:
            return
        self._ensure_loaded()
        with self._lock:
            if self._entries.pop(int(picture_id), None) is not None:
                self._unstore([int(picture_id)])
            self._active_cache = None

    def snapshot(self) -> dict[int, tuple[str, int, int]]:
        """Return a copy of the current ``{picture_id: (file_path, mtime_ns, size)}`` map."""
        self._ensure_loaded()
        with self._lock:
            return dict(self._entries)

    def __len__(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True)
class SignedLoad:
    """A classified decode of one file, and the file's signature just before it.

    What :meth:`UnprocessableImageRegistry.mark_unprocessable` accepts in place
    of decoding the file itself. The signature is taken first, as the
    registry's own decode does, so a file rewritten during the decode is pinned
    to its old signature and lifts on the next check.

    Attributes:
        file_path: The path that was stat'd and decoded.
        signature: ``(mtime_ns, size)`` before the decode, or ``None`` when the
            file could not be stat'd.
        result: The :func:`load_pil_classified` answer.
    """

    file_path: str
    signature: "tuple[int, int] | None"
    result: LoadResult


def load_signed(file_path: str, *, multiframe_count: int = 1) -> SignedLoad:
    """Stat *file_path*, then decode it with :func:`load_pil_classified`.

    For a task that loads a picture it may have to mark: the frames are its
    input, and the same load is the evidence ``mark_unprocessable(load=...)``
    needs, so the gate does not decode the file a second time.
    """
    path = str(file_path)
    signature = file_signature(path)
    return SignedLoad(
        file_path=path,
        signature=signature,
        result=load_pil_classified(path, multiframe_count=multiframe_count),
    )
