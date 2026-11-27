import asyncio
import contextvars
import errno
import functools
import hashlib
import json
import os
import posixpath
import shutil
import stat
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import (
    Body,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy import (
    delete,
)
from pydantic import BaseModel, ConfigDict
from sqlmodel import select, Session
from typing import Literal, Optional

from pixlstash.database import DBPriority

from pixlstash.db_models import (
    Picture,
    PictureProjectMember,
    Tag,
)
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services import (
    import_dedup_service,
    operation_log_service,
    scrapheap_service,
)
from pixlstash.tasks import TaskType
from pixlstash.tasks.picture_import_task import PictureImportTask
from pixlstash.db_models.tag import (
    TAG_PENDING_SENTINEL,
    is_tag_sentinel,
    TAG_SENTINEL_LIKE_PATTERN,
    TAG_SENTINEL_ESCAPE_CHAR,
)

from pixlstash.db_models.character import Character
from pixlstash.db_models.picture_set import PictureSet
from pixlstash.db_models.project import Project
from pixlstash.services.set_lock_service import locked_set_ids
from pixlstash.utils.links import is_link, is_link_or_missing
from pixlstash.utils.media_files import SUPPORTED_MEDIA_EXTS
from pixlstash.utils.quality.tag_exclusion import (
    filter_tag_names,
    session_tag_exclusions,
)
from pixlstash.utils.sqlite_ints import SqliteInt

from ._helpers import (
    _create_picture_imports,
    _normalise_sidecar_stem,
    _parse_sidecar_tags,
)


logger = get_logger(__name__)

# Media extensions accepted by both the one-shot upload import and the async
# streaming-staging import (#459). Kept module-level so both paths agree.
STAGING_ALLOWED_MEDIA_EXTS = SUPPORTED_MEDIA_EXTS

# Caption sidecar extension (mirrors the one-shot import's allowed_caption_exts).
STAGING_ALLOWED_CAPTION_EXTS = {".txt"}

# Bounds for a single staging session, mirroring the one-shot import limits.
_STAGING_MAX_FILES = 50_000
_STAGING_MAX_FILE_BYTES = 20 * 1024**3  # 20 GB per streamed file / zip
_STAGING_MAX_ZIP_ENTRIES = 50_000  # max files inside a single zip
_STAGING_MAX_ZIP_DECOMPRESSED_BYTES = 50 * 1024**3  # 50 GB total decompressed

# A caption file larger than this is left out of an import, by both import
# paths. Captions are a short tag list and their text is held in memory until
# the import runs.
_MAX_SIDECAR_BYTES = 1024 * 1024

# How much of an upload or zip entry is copied to disk at a time.
_COPY_CHUNK_BYTES = 1024 * 1024

# Why the one-shot import left a file out, chosen by what the file raised: the
# reader raises ``ValueError`` for content it cannot decode as a picture or a
# video, and a failed write or read raises ``OSError``. The exception's own
# text can name a server path, so it goes to the server log instead.
FILE_UNREADABLE_REASON = "Not a picture or video this library can read."
FILE_NOT_WRITTEN_REASON = "The file could not be written to the library."
FILE_FAILED_REASON = "The file could not be imported; the server log has the details."

# Reaper bounds. An opened-but-never-committed session leaks its .staging/ dir
# and memory; a finished session is never popped otherwise. The reaper (run
# opportunistically from the staging routes) evicts both.
_STAGING_SESSION_TTL_S = 3600  # evict a session with no activity for this long
_STAGING_TERMINAL_GRACE_S = 300  # keep a finished session briefly for a last poll

# An ``Idempotency-Key`` longer than this is refused; a UUID needs 36.
_IDEMPOTENCY_KEY_MAX_LEN = 255

# A ``relpath`` form field longer than this is refused. It only names the
# folder a dropped file sat in, and a browser path is far shorter.
_RELPATH_MAX_LEN = 4096

# Write errors that mean the volume holding the staging dir is full or over
# quota. A batch that hits one is refused with 507 rather than reported as a
# skipped file, because every later file would fail the same way. EDQUOT is not
# defined on every platform's errno module.
_DISK_FULL_ERRNOS = frozenset(
    {errno.ENOSPC} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())
)

# Session stages in which the files belong to an import (or are about to):
# nothing else may delete them and the session can no longer be cancelled.
_STAGING_HANDED_OFF_STAGES = ("committing", "importing", "completed", "failed")


def _remove_session_dir(server, staging_id: str, session: dict) -> bool:
    """Remove a session's staging dir; forget the session only once it is gone.

    The caller has claimed the removal by setting ``session["removing"]``
    under the session lock. A record whose directory could not be removed is
    kept and the claim dropped, so the reaper still owns the files and retries
    on its next pass instead of leaving them unreferenced.

    Returns:
        True when the directory is gone and the record was dropped.
    """
    staging_dir = session.get("staging_dir")
    if staging_dir and os.path.isdir(staging_dir):
        try:
            shutil.rmtree(staging_dir)
        except OSError as exc:
            logger.warning(
                "Staging %s: failed to remove staging dir %s (kept for retry): %s",
                staging_id,
                staging_dir,
                exc,
            )
    if staging_dir and os.path.lexists(staging_dir):
        with session["lock"]:
            session["removing"] = False
        return False
    if server.staging_sessions.get(staging_id) is session:
        server.staging_sessions.pop(staging_id, None)
    return True


def _sweep_orphan_staging_dirs(server, now_ms: int) -> None:
    """Remove entries under ``.staging/`` that no live session owns.

    Sessions live in memory only, so a restart between upload and import
    leaves a directory nothing refers to. An entry is removed once it is older
    than the idle TTL; a fresh one may belong to a session being opened right
    now. A symlink is unlinked, never followed.
    """
    staging_root = os.path.join(server.vault.image_root, ".staging")
    if is_link_or_missing(staging_root) or not os.path.isdir(staging_root):
        return
    live_ids = set(server.staging_sessions)
    cutoff_s = now_ms / 1000.0 - _STAGING_SESSION_TTL_S
    try:
        names = os.listdir(staging_root)
    except OSError as exc:
        logger.warning("Staging reaper: cannot list %s: %s", staging_root, exc)
        return
    for name in names:
        if name in live_ids:
            continue
        path = os.path.join(staging_root, name)
        try:
            info = os.lstat(path)
            if info.st_mtime > cutoff_s:
                continue
            if stat.S_ISDIR(info.st_mode):
                # rmtree refuses a path that has become a symlink since lstat.
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except OSError as exc:
            logger.warning(
                "Staging reaper: failed to remove orphaned entry %s: %s", path, exc
            )
            continue
        logger.info("Staging reaper: removed orphaned staging entry %s", name)


def _reap_staging_sessions(server, now_ms: Optional[int] = None) -> None:
    """Evict stale/finished staging sessions and remove their staging dirs.

    * A session that still owns work is never touched: an upload in flight, a
      commit in progress, or a background import that is pending or running.
    * A finished session (its import completed, failed or was cancelled, or it
      was cancelled before commit) is popped once it has been idle past
      ``_STAGING_TERMINAL_GRACE_S``, counted from its last poll or from when
      the import finished, whichever is later. That leaves a window for a
      final status poll.
    * Any other session (e.g. opened but never committed) is popped once idle
      past ``_STAGING_SESSION_TTL_S``. It is closed under its lock first, so no
      upload or commit can be admitted while its files are removed.
    * A session whose directory cannot be removed is kept for the next pass.
    * Entries under ``.staging/`` that no session owns are swept once older
      than the idle TTL.

    Called opportunistically from the staging routes, so no background thread is
    needed. Safe to call frequently - it only touches disk when it evicts.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    claimed: list[tuple[str, dict, str, int]] = []
    for staging_id, session in list(server.staging_sessions.items()):
        with session["lock"]:
            if (
                session["removing"]
                or session["inflight"] > 0
                or session["stage"] == "committing"
            ):
                continue
            task = session.get("task")
            status_value = (
                getattr(getattr(task, "status", None), "value", None)
                if task is not None
                else None
            )
            if status_value in ("pending", "running"):
                # Active import - never reap it out from under the running task.
                continue
            last_activity_ms = int(session.get("last_update_epoch_ms") or 0)
            completed_at = getattr(task, "completed_at", None)
            if completed_at is not None:
                last_activity_ms = max(
                    last_activity_ms, int(completed_at.timestamp() * 1000)
                )
            idle_ms = now_ms - last_activity_ms
            terminal = status_value in ("completed", "failed", "cancelled") or (
                task is None and session["stage"] == "cancelled"
            )
            if terminal and idle_ms > _STAGING_TERMINAL_GRACE_S * 1000:
                reason = f"finished (stage={session['stage']})"
            elif idle_ms > _STAGING_SESSION_TTL_S * 1000:
                reason = "idle TTL exceeded"
            else:
                continue
            session["removing"] = True
            if session["stage"] == "staging":
                session["stage"] = "cancelled"
        claimed.append((staging_id, session, reason, idle_ms))

    for staging_id, session, reason, idle_ms in claimed:
        if _remove_session_dir(server, staging_id, session):
            logger.info(
                "Staging reaper: evicted session %s (%s, idle=%.0fs)",
                staging_id,
                reason,
                idle_ms / 1000.0,
            )

    _sweep_orphan_staging_dirs(server, now_ms)


def _parse_idempotency_key(raw: Optional[str]) -> Optional[str]:
    """Return the batch identity carried by an ``Idempotency-Key`` header.

    The value is a Structured Field string (RFC 9651), so it normally arrives
    quoted: ``"8e03978e-40d5-43e8-bc93-6894a57f9324"``. A bare value is
    accepted too. A value that cannot be a key is refused rather than ignored:
    a caller that asked for idempotency must never be treated as one that did
    not.

    Returns:
        The unquoted key, or ``None`` when the header is absent.
    """
    if raw is None:
        return None
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if (
        not value
        or len(value) > _IDEMPOTENCY_KEY_MAX_LEN
        or any(ch in '"\\' or not 0x20 <= ord(ch) <= 0x7E for ch in value)
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_idempotency_key",
                "message": (
                    "Idempotency-Key must be a quoted string of 1 to "
                    f"{_IDEMPOTENCY_KEY_MAX_LEN} printable ASCII characters."
                ),
            },
        )
    return value


def _relpath_folder(raw: str) -> str:
    """The folder a dropped file sat in, read from its ``relpath`` form field.

    ``relpath`` is the dropped entry's path as the browser reports it
    (``/holiday/beach/x.png``). The caller controls it, so it is only ever a
    label that groups a caption with the media beside it: it is never joined
    to a filesystem path, opened or stored. A value that climbs (a ``..``
    segment), carries a NUL or is overlong is refused rather than tidied,
    because a browser never sends one. A backslash is not a separator, so
    ``a\\b\\x.png`` is a file name at the top of the drop.

    Returns:
        The folder as a ``/``-joined label with no leading ``/``, or ``""``
        for a file at the top of the drop or one sent without a path.

    Raises:
        HTTPException: 400 ``invalid_relpath`` for a refused value.
    """
    if "\x00" in raw or len(raw) > _RELPATH_MAX_LEN or ".." in raw.split("/"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_relpath",
                "message": (
                    "Each relpath must be a path inside the drop of at most "
                    f"{_RELPATH_MAX_LEN} characters, with no '..' segment or "
                    "NUL character."
                ),
            },
        )
    return posixpath.dirname(posixpath.normpath(raw.lstrip("/") or "."))


def _zip_member_folder(upload_folder: str, member_name: str) -> str:
    """The folder label of a zip member: the zip's own folder in the drop,
    then the member's folder inside the archive.

    Like ``_relpath_folder`` this only groups captions with media and is
    never used as a path, so an odd member name is not refused here.
    """
    inner = posixpath.dirname(posixpath.normpath(member_name.lstrip("/") or "."))
    return posixpath.join(upload_folder, inner) if inner else upload_folder


def _batch_fingerprint(
    files: list[UploadFile], keys: list[str], folders: list[str]
) -> str:
    """Identify a batch's payload: each part's client key, filename, size and
    folder label.

    A retried batch carries the same parts in the same order, so it matches; a
    different batch sent under a reused key does not, including the same files
    regrouped into other folders. Hashing the content is not needed to tell
    those apart.
    """
    parts = []
    for index, upload in enumerate(files):
        handle = upload.file
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(0)
        parts.append(
            [
                keys[index] if index < len(keys) else None,
                upload.filename or "",
                size,
                folders[index],
            ]
        )
    encoded = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _staging_conflict(code: str, message: str, session: dict) -> HTTPException:
    """A 409 that says who owns the session now, so a client can follow the
    import instead of reporting a failure. Call with the session lock held."""
    return HTTPException(
        status_code=409,
        detail={
            "code": code,
            "message": message,
            "stage": session["stage"],
            "task_id": session.get("task_id"),
        },
    )


def _remove_batch_files(staging_id: str, paths: list[str]) -> None:
    """Delete files an upload wrote that will not be staged: a partial write,
    or every file of a refused batch, so it leaves nothing behind."""
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                "Staging %s: failed to remove unstaged file %s: %s",
                staging_id,
                path,
                exc,
            )


class _UploadCopyStopped(Exception):
    """Raised on the worker thread by an upload's copy once the request that
    started it is cancelled, so the copy ends within one chunk. Neither an
    ``OSError`` nor a ``BadZipFile``: a stopped upload is not reported as
    unreadable, skipped or out of disk."""


async def _copy_upload_off_the_loop(
    staging_id: str, filename: Optional[str], copy, *args
):
    """Run *copy* on a worker thread and return what it returns.

    *copy* is a blocking copy of one upload into the staging dir. It gets a
    ``threading.Event`` as its last argument and raises ``_UploadCopyStopped``
    before its next chunk once that is set.

    A cancelled request keeps its upload lease and its files until the copy
    has stopped writing: the cancel sets the event, the request waits for the
    worker thread to return, and only then does the cancellation reach the
    caller, whose ``finally`` removes the files. The copy runs as a plain
    executor future, not a task, so tearing down the event loop cancels the
    request but not the copy, and the request still waits for it.
    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    future = loop.run_in_executor(
        None, contextvars.copy_context().run, copy, *args, stop
    )
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        stop.set()
        logger.info(
            "Staging %s: upload %s was cancelled; waiting for its copy to stop "
            "before its files are removed",
            staging_id,
            filename,
        )
        cancelled_again = False
        while not future.done():
            try:
                await asyncio.wait({future})
            except asyncio.CancelledError:
                if not cancelled_again:
                    cancelled_again = True
                    logger.warning(
                        "Staging %s: upload %s was cancelled again while its "
                        "copy is stopping; still waiting for the copy, since "
                        "removing its files before it ends would leave the "
                        "rest of the copy on disk with no owner",
                        staging_id,
                        filename,
                    )
        error = None if future.cancelled() else future.exception()
        if error is not None and not isinstance(error, _UploadCopyStopped):
            logger.error(
                "Staging %s: the copy of cancelled upload %s failed: %s",
                staging_id,
                filename,
                error,
                exc_info=error,
            )
        raise


def _copy_upload_part(upload: UploadFile, dest_path: str, stop: threading.Event) -> int:
    """Copy an uploaded part into *dest_path* a chunk at a time; return the
    bytes written.

    Blocking file I/O: the part is a spooled temporary file. Call it through
    ``_copy_upload_off_the_loop`` so neither the reads nor the writes hold the
    event loop. Raises ``_UploadCopyStopped`` before the next chunk once
    *stop* is set.
    """
    source = upload.file
    source.seek(0)
    written = 0
    with open(dest_path, "wb") as out:
        while True:
            if stop.is_set():
                raise _UploadCopyStopped(f"copy of {upload.filename!r} stopped")
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > _STAGING_MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File '{upload.filename}' exceeds the "
                        f"{_STAGING_MAX_FILE_BYTES // 1024**3} GB limit"
                    ),
                )
            out.write(chunk)
    return written


def _file_failure_reason(exc: BaseException) -> str:
    """The sentence an import status gives for a file that raised *exc*."""
    if isinstance(exc, ValueError):
        return FILE_UNREADABLE_REASON
    if isinstance(exc, OSError):
        return FILE_NOT_WRITTEN_REASON
    return FILE_FAILED_REASON


def _staging_file_limit(staged: int) -> HTTPException:
    """The 413 for a batch that would take the session past its file cap.

    The whole batch is refused: none of its files is staged, and the ones
    already in the session stay as they are.
    """
    return HTTPException(
        status_code=413,
        detail={
            "code": "staging_file_limit",
            "message": (
                f"Staging session exceeds the maximum of {_STAGING_MAX_FILES:,} files"
            ),
            "limit": _STAGING_MAX_FILES,
            "staged": staged,
        },
    )


def _staging_disk_full(
    staging_id: str, staging_dir: str, exc: OSError
) -> HTTPException:
    """The 507 for a batch whose write ran out of space on the library volume.

    ``free_bytes`` is what the volume reports now, or ``None`` when it cannot
    be read.
    """
    logger.error(
        "Staging %s: out of space writing into %s: %s", staging_id, staging_dir, exc
    )
    try:
        free_bytes: Optional[int] = shutil.disk_usage(staging_dir).free
    except OSError as usage_exc:
        logger.warning(
            "Staging %s: cannot read free space of %s: %s",
            staging_id,
            staging_dir,
            usage_exc,
        )
        free_bytes = None
    return HTTPException(
        status_code=507,
        detail={
            "code": "insufficient_storage",
            "message": (
                "Not enough disk space to stage the upload. Free up space on "
                "the library volume and try again."
            ),
            "free_bytes": free_bytes,
        },
    )


class StagingOpenRequest(BaseModel):
    """Body for opening an async-import staging session."""

    model_config = ConfigDict(extra="allow")

    project_id: Optional[SqliteInt] = None
    """Optional project every imported picture is added to on commit."""

    set_id: Optional[SqliteInt] = None
    """Optional picture set every imported picture is added to on commit
    (a drop-to-set target). Independent of ``character_id``."""

    character_id: Optional[SqliteInt] = None
    """Optional character every imported picture is associated with on commit
    (a drop-to-character target; deferred via ``pending_character_id`` until face
    extraction runs). Independent of ``set_id``."""

    total_files: Optional[int] = None
    """Client-declared file count (a hint for the progress UI / safe threshold);
    the actual import uses whatever was streamed before commit."""


class StagingOpenResponse(BaseModel):
    """Response for a newly-opened staging session."""

    model_config = ConfigDict(extra="allow")

    staging_id: str
    """Opaque id for the staging session; used on every subsequent call."""

    safe_threshold: int
    """The declared file count at which the client may hand off to the safe
    (background) window; ``0`` when the client did not declare a total."""


class StagingFilesResponse(BaseModel):
    """Response after streaming one batch of files into a staging session."""

    model_config = ConfigDict(extra="allow")

    staging_id: str
    staged: int
    """Total files staged in this session so far."""

    received: list[str]
    """Original filenames accepted in this request (media files staged, zips
    extracted, and caption sidecars stored)."""

    skipped: list[str]
    """Filenames rejected in this request (unsupported extension / empty /
    unreadable)."""

    sidecars: int
    """Total ``.txt`` caption sidecars stored in this session so far."""

    replayed: bool = False
    """``True`` when this answers a retry of a batch the session already
    accepted under the same ``Idempotency-Key``: the body is the original
    answer and nothing was staged again."""


class StagingCommitResponse(BaseModel):
    """Response for the safe handoff to the background import task."""

    model_config = ConfigDict(extra="allow")

    staging_id: str
    task_id: str
    """Id of the background ``PictureImportTask`` finishing the import."""

    staged_count: int

    replayed: bool = False
    """``True`` when the session was already handed off: ``task_id`` names the
    import the first commit started, and no second import was submitted."""


class StagingOutcome(BaseModel):
    """What the import decided for one staged file (#2512)."""

    key: Optional[str] = None
    """The ``key`` form field the file was streamed with, or ``None``."""

    outcome: Literal["new", "duplicate", "scrapheaped", "failed"]

    picture_id: Optional[int] = None
    """The picture a duplicate or scrapheaped file matched, when known. A
    scrapheaped match is restored by the import."""


class StagingStatusResponse(BaseModel):
    """Progress/status for a staging session and its background import."""

    model_config = ConfigDict(extra="allow")

    staging_id: str
    stage: str
    """One of ``staging`` / ``committing`` / ``importing`` / ``completed`` /
    ``failed`` / ``cancelled``. ``committing`` is the short window in which a
    commit has claimed the session but has not yet started the import."""

    phase: Optional[str] = None
    """What the import is doing while ``stage`` is ``importing``: ``queued``
    (waiting for a task worker), ``processing`` (checking each file for
    duplicates and reading it in), ``persisting`` (writing the new pictures and
    their project, set and character membership to the library in one go) or
    ``finalizing`` (restoring Scrapheap matches and removing the staged files).
    ``None`` in every other stage. ``processed`` stops changing before
    ``persisting`` starts, and no new picture exists until it ends."""

    cleanup: Optional[str] = None
    """Set by cancel only: ``done`` when the staged files were removed, or
    ``deferred`` when they are removed later - by an upload still being
    received, as it ends, or by the next reaper pass when removal failed now
    or is already under way."""

    staged: int
    total: int
    processed: int
    task_id: Optional[str] = None
    imported_count: Optional[int] = None
    """Staged files that became new pictures."""

    duplicate_count: Optional[int] = None
    """Staged files whose content is already in the vault as a LIVE picture
    (including a second copy inside this same batch)."""

    scrapheaped_count: Optional[int] = None
    """Staged files whose content matched a picture in the Scrapheap.

    Not imported again: the import restores the picture instead, so a re-import
    brings it back rather than writing a second copy. Counted per FILE, so
    ``imported + duplicate + scrapheaped + failed + cancelled == total``.
    """

    scrapheaped_picture_ids: Optional[list[int]] = None
    """Distinct pictures behind ``scrapheaped_count`` that the import restored
    from the Scrapheap, per PICTURE, so several incoming copies of one content
    name its id once."""

    failed_count: Optional[int] = None
    """Staged files that could not be hashed or ingested."""

    cancelled_count: Optional[int] = None
    """Staged files never reached because the import was cancelled. Present so
    the buckets still sum to ``total`` on a cancelled run."""

    error: Optional[str] = None

    outcomes: list[StagingOutcome] = []
    """Per-file outcomes the import has decided so far, in the order it decided
    them, starting at the ``outcomes_from`` the caller asked for. Each is
    ``{"key", "outcome", "picture_id"}``: ``key`` is the ``key`` form field the
    file was streamed with (``None`` when the client sent none; every entry
    extracted from one zip shares the zip's key), ``outcome`` is one of
    ``new`` (imported when the run commits), ``duplicate``, ``scrapheaped`` or
    ``failed``, and ``picture_id`` names the picture a duplicate or scrapheaped
    file matched (``None`` for a repeat inside the same batch, whose twin is
    not inserted yet). Files a cancelled run never reached have no entry. This
    is what lets the grid mark each arriving tile as it is checked instead of
    reporting one total at the end (#2512)."""


class ImportStartResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str


class ImportResultEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    """``success`` (a new picture), ``duplicate`` (content already live in the
    vault), ``scrapheaped`` (content matched a picture in the Scrapheap: not
    imported again, the picture is restored instead), or ``failed`` (the file
    could not be imported and was left out; the rest of the upload imports)."""

    picture_id: Optional[int] = None
    file: Optional[str] = None
    """The picture's stored path, or for a ``failed`` entry the uploaded file
    name."""

    error: Optional[str] = None
    """Why a ``failed`` file could not be imported."""


class ImportStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    stage: str
    total: int
    processed: int
    progress: float
    results: Optional[list[ImportResultEntry]] = None
    imported_count: Optional[int] = None
    """Files that became new pictures. Set once the import completes."""

    duplicate_count: Optional[int] = None
    """Files whose content is already in the vault as a LIVE picture."""

    scrapheaped_count: Optional[int] = None
    """Files whose content matched a picture in the Scrapheap.

    Counted per FILE, so ``imported_count + duplicate_count +
    scrapheaped_count + failed_count == total``. These files were not imported
    again (that would double the bytes on disk and refill the duplicate queue);
    the import restored the matching picture instead.
    """

    failed_count: Optional[int] = None
    """Files left out because they could not be imported, each with a
    ``failed`` entry in ``results``. When every file fails, the import's
    ``status`` is ``failed`` instead."""

    scrapheaped_picture_ids: Optional[list[int]] = None
    """Distinct pictures behind ``scrapheaped_count`` that the import restored.

    Per PICTURE, not per file: several incoming copies of the same content name
    one id once.
    """

    error: Optional[str] = None


def register_routes(router, server):
    @router.post(
        "/pictures/import",
        summary="Import media files",
        description="Starts an asynchronous import of uploaded image/video files (or zip contents) and returns a task id.",
        response_model=ImportStartResponse,
    )
    def import_pictures(
        request: Request,
        file: list[UploadFile] = File(None),
        project_id: SqliteInt | None = Form(None),
    ):
        # Capture the originating tab's client id synchronously, at request
        # entry. The import runs on a detached executor thread where the origin
        # contextvar is dead, so we stash this on the task record and carry it
        # explicitly into the PICTURE_IMPORTED event below.
        origin_client_id = getattr(request.state, "origin_client_id", None)
        # Who is importing, for the operation-log entry of any Scrapheap match
        # this import restores. Read here for the same reason as above.
        restore_context = operation_log_service.request_context(request)
        _MAX_UPLOAD_BYTES = 20 * 1024**3  # 20 GB per uploaded file / zip
        _MAX_ZIP_ENTRIES = 50_000  # max files inside a zip
        _MAX_ZIP_DECOMPRESSED_BYTES = 50 * 1024**3  # 50 GB total decompressed

        face_worker_problem = server.vault.worker_unavailable_reason(
            TaskType.FACE_EXTRACTION
        )
        if face_worker_problem:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot import: {face_worker_problem}.",
            )

        # The same fail-fast the staging path runs. Without it a project id that
        # does not exist here reaches the membership INSERT *after* every file
        # has been imported and committed, where it surfaces as a raw SQLite
        # FOREIGN KEY error and the caller is told the whole import failed while
        # the pictures are in fact already in the vault. Ids go stale for real:
        # a ComfyUI graph stores the project as "<name> #<id>", so a replayed
        # workflow carries whatever id was current when it was authored.
        project_created_at = _validate_association_targets(None, None, project_id)

        dest_folder = server.vault.image_root
        logger.debug("Importing pictures to folder: " + str(dest_folder))
        os.makedirs(dest_folder, exist_ok=True)
        uploaded_files = []
        # Captions and media are paired by ``(folder, stem)``: a caption only
        # tags media with its stem in the same folder. A plain upload has no
        # folder; a zip member's folder is its directory inside the archive.
        uploaded_file_keys: list[tuple[str, str]] = []
        sidecar_text_by_key: dict[tuple[str, str], str] = {}
        allowed_caption_exts = {".txt"}

        def _read_zip(upload: UploadFile) -> None:
            """Read an uploaded zip's media and caption entries.

            Blocking file I/O on the spooled temporary file: the route is a
            plain ``def``, so it runs on a threadpool worker, never on the
            event loop. Each media entry is read whole, because the
            import takes the media as bytes.
            """
            upload_file = upload.file
            upload_file.seek(0, 2)
            upload_size = upload_file.tell()
            upload_file.seek(0)
            if upload_size == 0:
                return
            if upload_size > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Uploaded file '{upload.filename}' exceeds the "
                        f"{_MAX_UPLOAD_BYTES // 1024**3} GB limit."
                    ),
                )
            with zipfile.ZipFile(upload_file) as zip_file:
                entries = [i for i in zip_file.infolist() if not i.is_dir()]
                if len(entries) > _MAX_ZIP_ENTRIES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Zip '{upload.filename}' contains too many files (max {_MAX_ZIP_ENTRIES:,}).",
                    )
                total_decompressed = sum(i.file_size for i in entries)
                if total_decompressed > _MAX_ZIP_DECOMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Zip '{upload.filename}' decompressed size exceeds the "
                            f"{_MAX_ZIP_DECOMPRESSED_BYTES // 1024**3} GB limit."
                        ),
                    )
                added = 0
                for info in entries:
                    inner_ext = os.path.splitext(info.filename)[1].lower()
                    is_caption = inner_ext in allowed_caption_exts
                    if not is_caption and inner_ext not in STAGING_ALLOWED_MEDIA_EXTS:
                        continue
                    with zip_file.open(info) as handle:
                        data = (
                            handle.read(_MAX_SIDECAR_BYTES + 1)
                            if is_caption
                            else handle.read()
                        )
                    if not data:
                        continue
                    base_name = os.path.basename(info.filename)
                    sidecar_key = (
                        _zip_member_folder("", info.filename),
                        _normalise_sidecar_stem(base_name),
                    )
                    if is_caption:
                        if len(data) > _MAX_SIDECAR_BYTES:
                            logger.warning(
                                "Leaving out caption %s in zip %s: "
                                "larger than %d bytes",
                                info.filename,
                                upload.filename,
                                _MAX_SIDECAR_BYTES,
                            )
                            continue
                        sidecar_text_by_key.setdefault(
                            sidecar_key,
                            data.decode("utf-8", errors="ignore"),
                        )
                        continue
                    uploaded_files.append((data, inner_ext, base_name))
                    uploaded_file_keys.append(sidecar_key)
                    added += 1
                if added == 0:
                    logger.warning(
                        "No valid media files found in zip: %s",
                        upload.filename,
                    )

        if file is not None:
            for upload in file:
                if not upload.filename:
                    continue
                ext = os.path.splitext(upload.filename)[1].lower()
                if ext == ".zip":
                    # Read from the spooled temp file, never the whole archive
                    # as one bytes object. This route is a plain ``def``, so the
                    # read runs on a threadpool worker.
                    try:
                        _read_zip(upload)
                    except zipfile.BadZipFile as exc:
                        logger.error("Invalid zip file: %s", upload.filename)
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid zip file",
                        ) from exc
                else:
                    if ext in allowed_caption_exts:
                        contents = upload.file.read(_MAX_SIDECAR_BYTES + 1)
                        if len(contents) > _MAX_SIDECAR_BYTES:
                            logger.warning(
                                "Leaving out caption %s: larger than %d bytes",
                                upload.filename,
                                _MAX_SIDECAR_BYTES,
                            )
                            continue
                    else:
                        # Media is held whole until the import thread writes
                        # it, because the import takes it as bytes.
                        contents = upload.file.read()
                    if not contents:
                        continue
                    if len(contents) > _MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Uploaded file '{upload.filename}' exceeds the "
                                f"{_MAX_UPLOAD_BYTES // 1024**3} GB limit."
                            ),
                        )
                    if ext in allowed_caption_exts:
                        sidecar_text_by_key.setdefault(
                            ("", _normalise_sidecar_stem(upload.filename)),
                            contents.decode("utf-8", errors="ignore"),
                        )
                        continue
                    if ext not in STAGING_ALLOWED_MEDIA_EXTS:
                        logger.warning(
                            "Skipping file with unsupported extension: %s",
                            upload.filename,
                        )
                        continue
                    uploaded_files.append((contents, ext, upload.filename))
                    uploaded_file_keys.append(
                        ("", _normalise_sidecar_stem(upload.filename))
                    )
        else:
            logger.error("No files provided for import")
            raise HTTPException(status_code=400, detail="No image provided")

        if not uploaded_files:
            logger.error("No valid media files found for import")
            raise HTTPException(
                status_code=400,
                detail="No valid media files found for import",
            )

        total_import_bytes_log = sum(len(data) for data, *_ in uploaded_files)
        logger.info(
            "Import request received: files=%d, sidecar_txt=%d, project_id=%s, total_bytes=%d",
            len(uploaded_files),
            len(sidecar_text_by_key),
            project_id,
            total_import_bytes_log,
        )

        sidecar_tags_by_key: dict[tuple[str, str], list[str]] = {}
        if sidecar_text_by_key:
            media_key_set = set(uploaded_file_keys)
            for sidecar_key, raw_text in sidecar_text_by_key.items():
                # Only consume caption sidecars that have a media file with the
                # same stem in the same folder.
                if sidecar_key not in media_key_set:
                    continue
                parsed_tags = _parse_sidecar_tags(raw_text)
                if parsed_tags:
                    sidecar_tags_by_key[sidecar_key] = parsed_tags

        total_import_bytes = sum(len(data) for data, *_ in uploaded_files)
        free_bytes = shutil.disk_usage(dest_folder).free
        required_bytes = int(total_import_bytes * 1.1)  # 10% headroom
        if required_bytes > free_bytes:
            free_gb = free_bytes / 1024**3
            needed_gb = required_bytes / 1024**3
            raise HTTPException(
                status_code=507,
                detail=(
                    f"Not enough disk space. "
                    f"Import needs {needed_gb:.2f} GB (including 10% headroom) "
                    f"but only {free_gb:.2f} GB is available."
                ),
            )

        task_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        server.import_tasks[task_id] = {
            "status": "in_progress",
            "stage": "queued",
            "total": len(uploaded_files),
            "processed": 0,
            "results": None,
            "error": None,
            "created_epoch_ms": now_ms,
            "last_update_epoch_ms": now_ms,
            "last_poll_log_epoch_ms": 0,
        }

        logger.info(
            "Import task queued: task_id=%s total=%d project_id=%s",
            task_id,
            len(uploaded_files),
            project_id,
        )

        def run_import_task(server, vault, background_lease):
            try:
                task = server.import_tasks[task_id]

                def _mark_stage(stage: str, **extra):
                    task["stage"] = stage
                    task["last_update_epoch_ms"] = int(time.time() * 1000)
                    task.update(extra)
                    logger.info(
                        "Import task stage: task_id=%s stage=%s processed=%d/%d",
                        task_id,
                        stage,
                        int(task.get("processed") or 0),
                        int(task.get("total") or 0),
                    )

                _mark_stage("hash_and_write")

                def _on_picture_written():
                    task["processed"] = task.get("processed", 0) + 1
                    task["last_update_epoch_ms"] = int(time.time() * 1000)

                # Upload index -> why that file could not be imported. The
                # other files of the upload still import.
                failed_errors: dict[int, str] = {}

                def _on_file_failed(index: int, exc: Exception):
                    failed_errors[index] = _file_failure_reason(exc)

                fingerprints, existing_map, scrapheaped_map, new_picture_map = (
                    _create_picture_imports(
                        SimpleNamespace(vault=vault),
                        uploaded_files,
                        dest_folder,
                        progress_callback=_on_picture_written,
                        on_file_failed=_on_file_failed,
                    )
                )
                if len(failed_errors) == len(uploaded_files):
                    # Nothing came in, so the import failed, and a caller that
                    # sends one file per import is told why.
                    reasons = "; ".join(
                        f"{uploaded_files[index][2]}: {reason}"
                        for index, reason in list(failed_errors.items())[:5]
                    )
                    raise RuntimeError(
                        f"None of the {len(uploaded_files)} file(s) could be "
                        f"imported: {reasons}"
                    )

                # Duplicates are instantly "processed" - credit them now so that
                # the progress bar stays accurate even when most files are dupes.
                # Scrapheap matches are equally instant and are counted in their
                # OWN bucket: they are neither imported nor ordinary duplicates.
                # Both are counted directly (one pass over fingerprints), never derived
                # by subtracting from the total.
                fresh_seen = set()
                duplicate_count_initial = 0
                for fingerprint in fingerprints:
                    if fingerprint in existing_map or fingerprint in fresh_seen:
                        duplicate_count_initial += 1
                    elif fingerprint not in scrapheaped_map:
                        fresh_seen.add(fingerprint)
                scrapheaped_count_initial = sum(
                    1 for fingerprint in fingerprints if fingerprint in scrapheaped_map
                )
                new_pictures = list(new_picture_map.values())
                task["processed"] = (
                    len(new_pictures)
                    + duplicate_count_initial
                    + scrapheaped_count_initial
                    + len(failed_errors)
                )
                task["last_update_epoch_ms"] = int(time.time() * 1000)

                _mark_stage(
                    "deduplicated",
                    duplicate_count_initial=duplicate_count_initial,
                    scrapheaped_count_initial=scrapheaped_count_initial,
                    new_count=len(new_pictures),
                )

                logger.debug(
                    f"Importing {len(new_pictures)} new pictures out of {len(uploaded_files)} uploaded."
                )

                if new_pictures:
                    _mark_stage("persisting_new_pictures")

                    def import_task(session, published):
                        session.add_all(new_pictures)
                        session.flush()
                        for pic in new_pictures:
                            session.add(
                                Tag(tag=TAG_PENDING_SENTINEL, picture_id=pic.id)
                            )
                        session.commit()
                        published.mark_committed()
                        for pic in new_pictures:
                            session.refresh(pic)
                        return new_pictures

                    with import_dedup_service.PublishedPictures(
                        dest_folder, f"Import task {task_id}"
                    ) as published:
                        for pic in new_pictures:
                            published.add(pic)
                        new_pictures = vault.db.run_task(import_task, published)
                    logger.debug(
                        f"Queuing likeness calculation for {len(new_pictures)} new pictures."
                    )
                else:
                    logger.warning(
                        "No new pictures to import; every file already matches a "
                        "live or scrapheaped picture."
                    )
                    new_pictures = []

                _mark_stage("building_results")
                results = []
                imported_count = 0
                duplicate_count = 0
                scrapheaped_count = 0
                failed_count = 0
                picture_id_sidecar_tags: dict[int, set[str]] = defaultdict(set)
                duplicate_picture_id_set: set[int] = set()
                # Distinct scrapheaped pictures this import restores. Several
                # incoming files can match the SAME scrapheaped picture: the
                # count is per FILE (so the buckets sum to the file total) while
                # this list is per PICTURE (how many rows come back).
                scrapheaped_picture_ids: list[int] = []
                seen_scrapheaped_ids: set[int] = set()
                seen_new_fingerprints = set()
                for index, (sidecar_key, file_entry, fingerprint) in enumerate(
                    zip(uploaded_file_keys, uploaded_files, fingerprints)
                ):
                    # Four disjoint outcomes per uploaded file. `existing_map`
                    # and `scrapheaped_map` are disjoint by construction (a live
                    # row outranks a soft-deleted one for the same hash), and a
                    # failed file is in neither, so the branches below cannot
                    # double-count.
                    if index in failed_errors:
                        results.append(
                            {
                                "status": "failed",
                                "picture_id": None,
                                "file": file_entry[2],
                                "error": failed_errors[index],
                            }
                        )
                        failed_count += 1
                        continue
                    if fingerprint in existing_map:
                        pic = existing_map[fingerprint]
                        results.append(
                            {
                                "status": "duplicate",
                                "picture_id": pic.id,
                                "file": pic.file_path,
                            }
                        )
                        duplicate_count += 1
                        if pic.id is not None:
                            duplicate_picture_id_set.add(pic.id)
                    elif fingerprint in scrapheaped_map:
                        pic = scrapheaped_map[fingerprint]
                        results.append(
                            {
                                "status": "scrapheaped",
                                "picture_id": pic.id,
                                "file": pic.file_path,
                            }
                        )
                        scrapheaped_count += 1
                        if pic.id is not None and pic.id not in seen_scrapheaped_ids:
                            seen_scrapheaped_ids.add(pic.id)
                            scrapheaped_picture_ids.append(pic.id)
                    elif fingerprint not in seen_new_fingerprints:
                        pic = new_picture_map[fingerprint]
                        results.append(
                            {
                                "status": "success",
                                "picture_id": pic.id,
                                "file": pic.file_path,
                            }
                        )
                        imported_count += 1
                        seen_new_fingerprints.add(fingerprint)
                    else:
                        # A byte-identical repeat inside this upload maps to the
                        # one row created for its first occurrence.
                        pic = new_picture_map[fingerprint]
                        results.append(
                            {
                                "status": "duplicate",
                                "picture_id": pic.id,
                                "file": pic.file_path,
                            }
                        )
                        duplicate_count += 1
                        if pic.id is not None:
                            duplicate_picture_id_set.add(pic.id)

                    if (
                        pic.id is not None
                        and sidecar_tags_by_key.get(sidecar_key)
                        # A restored match comes back as it was: its tags are
                        # not replaced (it may sit in a locked set).
                        and fingerprint not in scrapheaped_map
                    ):
                        picture_id_sidecar_tags[pic.id].update(
                            sidecar_tags_by_key[sidecar_key]
                        )

                if duplicate_count:
                    logger.warning(
                        "Import completed with %d duplicate(s) out of %d file(s).",
                        duplicate_count,
                        len(uploaded_files),
                    )
                if scrapheaped_picture_ids:
                    matched_ids = scrapheaped_picture_ids
                    # Report what was restored, not what matched: a match the
                    # retention sweep purged meanwhile did not come back.
                    scrapheaped_picture_ids = scrapheap_service.restore_pictures(
                        vault, matched_ids, **restore_context
                    )
                    logger.info(
                        "Import task %s: %d file(s) out of %d matched "
                        "scrapheaped picture(s) %s; restored %s instead of "
                        "importing a second copy.",
                        task_id,
                        scrapheaped_count,
                        len(uploaded_files),
                        matched_ids,
                        scrapheaped_picture_ids,
                    )
                if failed_count:
                    logger.warning(
                        "Import task %s left out %d file(s) out of %d that could "
                        "not be imported.",
                        task_id,
                        failed_count,
                        len(uploaded_files),
                    )
                bucket_total = (
                    imported_count + duplicate_count + scrapheaped_count + failed_count
                )
                if bucket_total != len(uploaded_files):
                    # The four buckets are the whole story of this import; a
                    # mismatch means the summary the user reads is wrong, so say
                    # so loudly rather than let a silently-lossy count ship.
                    logger.error(
                        "Import task %s bucket arithmetic is inconsistent: "
                        "imported=%d + duplicate=%d + scrapheaped=%d + failed=%d "
                        "= %d, but %d file(s) were uploaded.",
                        task_id,
                        imported_count,
                        duplicate_count,
                        scrapheaped_count,
                        failed_count,
                        bucket_total,
                        len(uploaded_files),
                    )
                server.import_tasks[task_id]["results"] = results
                server.import_tasks[task_id]["processed"] = len(uploaded_files)
                server.import_tasks[task_id]["imported_count"] = imported_count
                server.import_tasks[task_id]["duplicate_count"] = duplicate_count
                server.import_tasks[task_id]["scrapheaped_count"] = scrapheaped_count
                server.import_tasks[task_id]["failed_count"] = failed_count
                server.import_tasks[task_id]["scrapheaped_picture_ids"] = (
                    scrapheaped_picture_ids
                )
                server.import_tasks[task_id]["last_update_epoch_ms"] = int(
                    time.time() * 1000
                )
                # Only apply import context to pictures that were actually
                # touched by this request (one results row per uploaded file).
                # Restored Scrapheap matches come back as they were, as on the
                # staging path: no new project, no "added" announcement.
                all_imported_ids = list(
                    dict.fromkeys(
                        entry.get("picture_id")
                        for entry in results
                        if entry.get("picture_id") is not None
                        and entry.get("status") != "scrapheaped"
                    )
                )

                if picture_id_sidecar_tags:
                    _mark_stage("applying_sidecar_tags")

                    def apply_sidecar_tags(
                        session,
                        mapping: dict[int, set[str]],
                        replace_ids: set[int],
                    ):
                        if not mapping:
                            return []
                        pics = session.exec(
                            select(Picture)
                            .where(Picture.id.in_(list(mapping.keys())))
                            .where(Picture.deleted.is_(False))
                        ).all()
                        changed_ids = []
                        replaced_ids = []
                        exclusions = session_tag_exclusions(session)
                        left_out = 0
                        for pic in pics:
                            tag_values = mapping.get(pic.id) or set()
                            if not tag_values:
                                continue
                            # Tags the owner removed from tagging never come
                            # back through a sidecar.
                            permitted = set(filter_tag_names(tag_values, exclusions))
                            left_out += len(tag_values) - len(permitted)
                            existing_values = {
                                (row[0] if isinstance(row, tuple) else row or "")
                                .strip()
                                .lower()
                                for row in session.exec(
                                    select(Tag.tag).where(Tag.picture_id == pic.id)
                                ).all()
                            }
                            changed = False
                            if pic.id in replace_ids:
                                # For duplicate imports with sidecar captions,
                                # replace old tags with sidecar-provided tags.
                                # A sidecar left with nothing permitted keeps
                                # the tags the picture has.
                                if not permitted:
                                    continue
                                session.exec(
                                    delete(Tag).where(Tag.picture_id == pic.id)
                                )
                                for tag_value in sorted(permitted):
                                    session.add(Tag(tag=tag_value, picture_id=pic.id))
                                replaced_ids.append(pic.id)
                                changed = True
                            else:
                                # The sidecar tagged the picture even when it
                                # named only removed tags, so the pending
                                # sentinel goes either way. Deleted through the
                                # session so the metadata hash is recomputed
                                # when no tag is added in its place.
                                if any(is_tag_sentinel(v) for v in existing_values):
                                    for sentinel_row in session.exec(
                                        select(Tag).where(
                                            Tag.picture_id == pic.id,
                                            Tag.tag.like(
                                                TAG_SENTINEL_LIKE_PATTERN,
                                                escape=TAG_SENTINEL_ESCAPE_CHAR,
                                            ),
                                        )
                                    ).all():
                                        session.delete(sentinel_row)
                                    changed = True
                                for tag_value in sorted(permitted):
                                    if tag_value in existing_values:
                                        continue
                                    session.add(Tag(tag=tag_value, picture_id=pic.id))
                                    changed = True
                            if changed:
                                changed_ids.append(pic.id)
                        if left_out:
                            logger.debug(
                                "Import: left out %d sidecar tag(s) removed from "
                                "tagging.",
                                left_out,
                            )
                        session.commit()
                        vault.db.tag_resets.mark_reset(replaced_ids)
                        return changed_ids

                    tagged_ids = vault.db.run_task(
                        apply_sidecar_tags,
                        picture_id_sidecar_tags,
                        duplicate_picture_id_set,
                    )
                    if tagged_ids:
                        vault.notify(EventType.CHANGED_TAGS, tagged_ids)

                if all_imported_ids:
                    _mark_stage("finalizing_import_context")
                    # The workers find the new pictures themselves; nothing
                    # here waits for them. A vault can be without an engine: a
                    # server not readied at start-up, or a library switch whose
                    # engine build failed and kept the library open. Its model
                    # workers stay idle until one is built, so only an import
                    # that brought new pictures in builds it.
                    if new_pictures:
                        vault.ensure_ready()

                    def apply_import_context(
                        session,
                        ids: list[int],
                        project_id_value: int | None,
                    ):
                        if not ids:
                            return []
                        if project_id_value is not None:
                            # Ids are reused, so the project can be deleted
                            # during the import and a new one created under
                            # the same id: only the creation time tells them
                            # apart.
                            project = session.get(Project, project_id_value)
                            if (
                                project is None
                                or project.created_at != project_created_at
                            ):
                                logger.error(
                                    "Import task %s: project %s vanished before "
                                    "association",
                                    task_id,
                                    project_id_value,
                                )
                                project_id_value = None
                        now = datetime.now(timezone.utc)
                        pics = session.exec(
                            select(Picture).where(Picture.id.in_(ids))
                        ).all()
                        updated = []
                        for pic in pics:
                            if pic.imported_at is None:
                                pic.imported_at = now
                            if project_id_value is not None:
                                member = session.exec(
                                    select(PictureProjectMember).where(
                                        PictureProjectMember.picture_id == pic.id,
                                        PictureProjectMember.project_id
                                        == project_id_value,
                                    )
                                ).first()
                                if member is None:
                                    session.add(
                                        PictureProjectMember(
                                            picture_id=pic.id,
                                            project_id=project_id_value,
                                        )
                                    )
                                pic.project_id = project_id_value
                            session.add(pic)
                            updated.append(pic.id)
                        session.commit()
                        return updated

                    imported_ids = vault.db.run_task(
                        apply_import_context,
                        all_imported_ids,
                        project_id,
                    )
                    server.import_tasks[task_id]["status"] = "completed"
                    server.import_tasks[task_id]["stage"] = "completed"
                    server.import_tasks[task_id]["last_update_epoch_ms"] = int(
                        time.time() * 1000
                    )
                    vault.notify(
                        EventType.CHANGED_PICTURES,
                        {
                            "picture_ids": imported_ids or [],
                            "source": "ui" if origin_client_id else "external",
                            "origin_client_id": origin_client_id,
                            "change_kind": "added",
                        },
                    )
                    if imported_ids:
                        # A genuine PixlStash tab attaches X-Client-Id (captured
                        # as origin_client_id); an external API client - e.g. a
                        # ComfyUI node POSTing generated output to
                        # /pictures/import - does not. Tag the former "ui" (slick
                        # in-place insert) and the latter "external" so an
                        # outside push raises the New-pictures pill instead of
                        # auto-inserting cards under the user.
                        import_source = "ui" if origin_client_id else "external"
                        vault.notify(
                            EventType.PICTURE_IMPORTED,
                            {
                                "ids": imported_ids,
                                "source": import_source,
                                "origin_client_id": origin_client_id,
                                "change_kind": "added",
                            },
                        )
                else:
                    server.import_tasks[task_id]["status"] = "completed"
                    server.import_tasks[task_id]["stage"] = "completed"
                    server.import_tasks[task_id]["last_update_epoch_ms"] = int(
                        time.time() * 1000
                    )
                    vault.notify(
                        EventType.CHANGED_PICTURES,
                        {
                            "source": "ui" if origin_client_id else "external",
                            "origin_client_id": origin_client_id,
                            "change_kind": "updated",
                        },
                    )
                logger.info("Import task completed: task_id=%s", task_id)
            except Exception as exc:
                server.import_tasks[task_id]["status"] = "failed"
                server.import_tasks[task_id]["stage"] = "failed"
                server.import_tasks[task_id]["error"] = str(exc)
                server.import_tasks[task_id]["last_update_epoch_ms"] = int(
                    time.time() * 1000
                )
                logger.error(f"Import task {task_id} failed: {exc}")
            finally:
                server.library_coordinator.release_read(background_lease)

        # Schedule independently from the ASGI/event-loop lifecycle. Using
        # BackgroundTasks would tie the work to the response and a default
        # executor Future can be cancelled while its thread keeps running at
        # event-loop shutdown. The worker itself owns and finally releases the
        # extra library lease, so a switch cannot retire its vault underneath
        # it.
        background_lease = server.library_coordinator.acquire_read()
        if background_lease is None:
            server.import_tasks[task_id]["status"] = "failed"
            server.import_tasks[task_id]["stage"] = "failed"
            server.import_tasks[task_id]["error"] = "Library is unavailable."
            raise HTTPException(
                status_code=503,
                detail="Library is switching or unavailable. Try again.",
            )
        server.import_tasks[task_id]["library_uuid"] = background_lease.library_uuid
        server.import_tasks[task_id]["generation"] = background_lease.generation
        try:
            worker = threading.Thread(
                target=run_import_task,
                args=(server, background_lease.vault, background_lease),
                daemon=True,
            )
            worker.start()
        except Exception:
            server.library_coordinator.release_read(background_lease)
            raise
        return {"task_id": task_id}

    @router.get(
        "/pictures/import/status",
        summary="Get import job status",
        description="Returns progress and result information for a previously started import task.",
        response_model=ImportStatusResponse,
    )
    def import_status(request: Request, task_id: str):
        task = server.import_tasks.get(task_id)
        lease = getattr(request.state, "library_lease", None)
        if (
            not task
            or lease is None
            or task.get("library_uuid") != lease.library_uuid
            or task.get("generation") != lease.generation
        ):
            raise HTTPException(status_code=404, detail="Task not found")

        now_ms = int(time.time() * 1000)
        last_poll_log_epoch_ms = int(task.get("last_poll_log_epoch_ms") or 0)
        if (
            task.get("status") == "in_progress"
            and now_ms - last_poll_log_epoch_ms >= 10_000
        ):
            task["last_poll_log_epoch_ms"] = now_ms
            created_ms = int(task.get("created_epoch_ms") or now_ms)
            elapsed_s = max(0.0, (now_ms - created_ms) / 1000.0)
            logger.info(
                "Import task heartbeat: task_id=%s stage=%s processed=%d/%d elapsed=%.1fs",
                task_id,
                task.get("stage", "unknown"),
                int(task.get("processed") or 0),
                int(task.get("total") or 0),
                elapsed_s,
            )

        total = task.get("total") or 0
        processed = task.get("processed") or 0
        progress = (processed / total * 100.0) if total else 0.0

        payload = {
            "status": task["status"],
            "stage": task.get("stage", "unknown"),
            "total": total,
            "processed": processed,
            "progress": progress,
        }
        if task["status"] == "completed":
            payload["results"] = task.get("results") or []
            # The four disjoint buckets, each counted directly while the results
            # were built: never derived from the total by subtraction.
            payload["imported_count"] = task.get("imported_count")
            payload["duplicate_count"] = task.get("duplicate_count")
            payload["scrapheaped_count"] = task.get("scrapheaped_count")
            payload["failed_count"] = task.get("failed_count")
            payload["scrapheaped_picture_ids"] = (
                task.get("scrapheaped_picture_ids") or []
            )
        if task["status"] == "failed":
            payload["error"] = task.get("error")
        return payload

    # ------------------------------------------------------------------ #
    # Async streaming-staging import (#459)                               #
    #                                                                     #
    # Phase A (unsafe, tab must stay open): open a staging session, then  #
    # stream files into it. Phase B (safe, tab may close): commit hands   #
    # off to a background PictureImportTask on the shared TaskRunner,      #
    # whose progress surfaces in the task manager. All four mutating      #
    # routes are OWNER_ONLY (streaming client-provided bytes into the      #
    # vault - NOT a host-filesystem read), mirroring POST /pictures/import.#
    # ------------------------------------------------------------------ #

    def _staging_base_dir() -> str:
        path = os.path.join(server.vault.image_root, ".staging")
        if os.path.lexists(path) and (is_link(path) or not os.path.isdir(path)):
            raise HTTPException(
                status_code=500,
                detail="The library staging path is not a trusted directory.",
            )
        os.makedirs(path, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
        return path

    def _get_session_or_404(request: Request, staging_id: str) -> dict:
        session = server.staging_sessions.get(staging_id)
        lease = getattr(request.state, "library_lease", None)
        if (
            session is None
            or lease is None
            or session.get("library_uuid") != lease.library_uuid
            or session.get("generation") != lease.generation
        ):
            raise HTTPException(status_code=404, detail="Staging session not found")
        return session

    def _validate_association_targets(
        set_id, character_id, project_id
    ) -> datetime | None:
        """Fail fast if a drop-to-set / -character / -project target is invalid.

        A nonexistent set/character/project must error rather than silently
        no-op (or fail downstream after pictures are already imported), and a
        locked set must refuse new members (the same lock rule the add-to-set
        route enforces). Runs at open (on the drop) and again at commit.

        Returns:
            The project's ``created_at``, or ``None`` without a project. The
            import files its pictures only on a project with this creation
            time, since a deleted project's id can be given to a new one.
        """

        def _check(session: Session):
            if set_id is not None:
                picture_set = session.get(PictureSet, set_id)
                if picture_set is None:
                    return ("set_missing", None)
                if set_id in locked_set_ids(session, [set_id]):
                    return ("set_locked", picture_set.name)
            if character_id is not None:
                if session.get(Character, character_id) is None:
                    return ("character_missing", None)
            if project_id is not None:
                project = session.get(Project, project_id)
                if project is None:
                    return ("project_missing", None)
                return ("ok", project.created_at)
            return ("ok", None)

        kind, name = server.vault.db.run_task(_check, priority=DBPriority.IMMEDIATE)
        if kind == "ok":
            return name
        if kind == "set_missing":
            raise HTTPException(
                status_code=404, detail=f"Picture set {set_id} not found"
            )
        if kind == "set_locked":
            raise HTTPException(
                status_code=409,
                detail=f"Picture set '{name}' is locked and cannot take new members",
            )
        if kind == "character_missing":
            raise HTTPException(
                status_code=404, detail=f"Character {character_id} not found"
            )
        if kind == "project_missing":
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

    @router.post(
        "/pictures/import/staging",
        summary="Open an async import staging session",
        description=(
            "Opens a staging session for the async streaming import (#459). "
            "Stream files into it with POST "
            "/pictures/import/staging/{staging_id}/files while the tab stays "
            "open, then hand off to the background import with "
            "POST /pictures/import/staging/{staging_id}/commit."
        ),
        response_model=StagingOpenResponse,
    )
    def open_staging_session(
        request: Request,
        payload: StagingOpenRequest = Body(default_factory=StagingOpenRequest),
    ):
        origin_client_id = getattr(request.state, "origin_client_id", None)
        lease = request.state.library_lease
        # Opportunistically evict stale/finished sessions before opening a new
        # one, so abandoned staging dirs never accumulate.
        _reap_staging_sessions(server)
        # Reject an invalid drop target up front (on the drop), before any files
        # are streamed - a nonexistent set/character/project must not silently
        # no-op or fail downstream after import.
        project_created_at = _validate_association_targets(
            payload.set_id, payload.character_id, payload.project_id
        )

        staging_id = str(uuid.uuid4())
        staging_dir = os.path.join(_staging_base_dir(), staging_id)
        try:
            os.makedirs(staging_dir, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create staging dir %s: %s", staging_dir, exc)
            raise HTTPException(
                status_code=500, detail="Could not create staging directory"
            ) from exc

        declared_total = int(payload.total_files or 0)
        now_ms = int(time.time() * 1000)
        server.staging_sessions[staging_id] = {
            "staging_id": staging_id,
            "stage": "staging",
            "staging_dir": staging_dir,
            "project_id": payload.project_id,
            "project_created_at": project_created_at,
            "set_id": payload.set_id,
            "character_id": payload.character_id,
            "declared_total": declared_total,
            "staged_files": [],
            # (folder, stem) -> raw sidecar text, from .txt uploads and zip
            # .txt entries. The folder is a label from the upload's relpath
            # and the zip member's directory, never a filesystem path.
            "sidecar_text_by_key": {},
            "task": None,
            "task_id": None,
            "origin_client_id": origin_client_id,
            "library_uuid": lease.library_uuid,
            "generation": lease.generation,
            "error": None,
            "created_epoch_ms": now_ms,
            "last_update_epoch_ms": now_ms,
            # Ownership of the staged files (see the stage rules on the routes
            # below). The lock guards every read-modify-write of this record
            # and is only ever held for bookkeeping, never across an await or
            # disk I/O.
            "lock": threading.Lock(),
            # Upload requests admitted and not yet settled. While any is in
            # flight nobody may commit, reap or delete the files.
            "inflight": 0,
            # A cancel arrived while uploads were in flight: the last of them
            # removes the files as it ends.
            "cleanup_pending": False,
            # A cancel, the reaper or the last upload after a cancel has
            # claimed the removal of this session's files.
            "removing": False,
            # Idempotency-Key -> {"fingerprint", "state", "response"} for the
            # upload batches of this session; they expire with it.
            "batches": {},
        }
        logger.info(
            "Staging session opened: staging_id=%s declared_total=%d project_id=%s "
            "set_id=%s character_id=%s",
            staging_id,
            declared_total,
            payload.project_id,
            payload.set_id,
            payload.character_id,
        )
        return {"staging_id": staging_id, "safe_threshold": declared_total}

    @router.post(
        "/pictures/import/staging/{staging_id}/files",
        summary="Stream files into a staging session",
        description=(
            "Streams one batch of media files into an open staging session. "
            "May be called repeatedly during the unsafe (tab-open) window. "
            "Unsupported or empty files are skipped and reported. A batch is "
            "staged whole or not at all: a refused batch (409, 413, 507) leaves "
            "none of its files behind.\n\n"
            "Folders: send one `relpath` field per `file`, in the same order, "
            "with the path the file had in the drop (`/holiday/x.png`). It "
            "only pairs a `.txt` caption with the media of the same name in "
            "the same folder; the staged file is still named by the server. "
            "A relpath with a `..` segment, a NUL or over 4096 characters gets "
            "400 `invalid_relpath` and stages nothing.\n\n"
            "Retrying safely: send an `Idempotency-Key` header, a quoted string "
            "such as a UUID, chosen once per batch and reused by every retry of "
            "it. A retry of a batch the session already accepted returns the "
            'original answer with `"replayed": true` and stages nothing again. '
            "A retry while the first attempt is still being staged gets 409 "
            "`batch_in_progress`. The same key sent with different files, or "
            "the same files in other folders, gets 422 "
            "`idempotency_key_mismatch`. Keys are scoped to the staging "
            "session and expire with it: when the session is reaped (idle for "
            "an hour, or a few minutes after its import finished) its keys are "
            "forgotten. Without the header every request stages its files."
        ),
        response_model=StagingFilesResponse,
    )
    async def stage_files(
        request: Request,
        staging_id: str,
        file: list[UploadFile] = File(None),
        key: list[str] = Form(None),
        relpath: list[str] = Form(None),
        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    ):
        session = _get_session_or_404(request, staging_id)
        batch_key = _parse_idempotency_key(idempotency_key)
        # The client's own handle for each file, sent as one ``key`` field per
        # ``file`` in the same order. Carried onto the staged entry so the
        # import can say which file each outcome belongs to (#2512). Optional:
        # a caller that sends none gets ``None`` keys.
        keys: list[str] = list(key or [])
        # The folder each file sat in, from one ``relpath`` field per ``file``
        # in the same order; a file without one is at the top of the drop.
        # Every value is checked before anything is admitted or written.
        folders = [_relpath_folder(value) for value in relpath or []]
        folders += [""] * (len(file or []) - len(folders))
        fingerprint = (
            _batch_fingerprint(file, keys, folders) if batch_key and file else None
        )

        # Admission. An admitted request holds an upload lease (``inflight``)
        # until its ``finally``: while any is held nobody commits, reaps or
        # deletes the session's files.
        record: Optional[dict] = None
        with session["lock"]:
            if batch_key is not None and batch_key in session["batches"]:
                known = session["batches"][batch_key]
                if known["fingerprint"] != fingerprint:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "idempotency_key_mismatch",
                            "message": (
                                "This Idempotency-Key was already used for a "
                                "different batch in this staging session."
                            ),
                        },
                    )
                if known["state"] == "in_progress":
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "batch_in_progress",
                            "message": (
                                "This batch is still being staged by an earlier "
                                "request. Retry once that request has finished."
                            ),
                        },
                    )
                logger.info(
                    "Staging %s: batch %s already staged; returning its answer",
                    staging_id,
                    batch_key,
                )
                return {**known["response"], "replayed": True}
            if session["stage"] != "staging":
                raise _staging_conflict(
                    "staging_closed",
                    "Staging session is no longer accepting files "
                    f"(stage={session['stage']})",
                    session,
                )
            if not file:
                raise HTTPException(status_code=400, detail="No files provided")
            session["inflight"] += 1
            if batch_key is not None:
                record = {
                    "fingerprint": fingerprint,
                    "state": "in_progress",
                    "response": None,
                }
                session["batches"][batch_key] = record

        staging_dir = session["staging_dir"]
        # This batch's entries and sidecars join the session only once every
        # file of the batch is on disk (the publish step below), so a refused
        # or failed batch never leaves part of itself staged.
        batch_entries: list[dict] = []
        batch_sidecars: dict[tuple[str, str], str] = {}
        # Every file this batch created, removed again if the batch is refused.
        written_paths: list[str] = []
        received: list[str] = []
        skipped: list[str] = []
        current_key: Optional[str] = None
        current_folder = ""
        published = False

        def _check_file_cap() -> None:
            # Stops an oversized batch before it writes the rest of its files.
            # The binding check is the one at publish, under the lock, which
            # also sees batches staged concurrently.
            staged = len(session["staged_files"])
            if staged + len(batch_entries) >= _STAGING_MAX_FILES:
                raise _staging_file_limit(staged)

        def _stage_zip_entry(
            zip_file: zipfile.ZipFile,
            info: zipfile.ZipInfo,
            ext: str,
            stop: threading.Event,
        ) -> bool:
            """Copy one media entry of a zip into the staging dir, a chunk at a
            time, raising ``_UploadCopyStopped`` before the next chunk once
            *stop* is set."""
            if info.file_size == 0:
                return False
            _check_file_cap()
            dest_path = os.path.join(staging_dir, f"{uuid.uuid4()}{ext}")
            written_paths.append(dest_path)
            try:
                with zip_file.open(info) as src, open(dest_path, "wb") as out:
                    while True:
                        if stop.is_set():
                            raise _UploadCopyStopped(
                                f"copy of {info.filename!r} stopped"
                            )
                        chunk = src.read(_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
            except OSError as exc:
                _remove_batch_files(staging_id, [dest_path])
                if exc.errno in _DISK_FULL_ERRNOS:
                    raise _staging_disk_full(staging_id, staging_dir, exc) from exc
                logger.error(
                    "Staging %s: failed to write extracted file %s: %s",
                    staging_id,
                    info.filename,
                    exc,
                )
                return False
            except Exception:
                # A corrupt entry fails part-way through the copy, and a
                # cancelled upload stops part-way: leave no partial file
                # behind, whatever happens to the rest of the zip.
                _remove_batch_files(staging_id, [dest_path])
                raise
            batch_entries.append(
                {
                    "file_path": dest_path,
                    "original_file_name": os.path.basename(info.filename),
                    "client_key": current_key,
                    "sidecar_folder": _zip_member_folder(current_folder, info.filename),
                }
            )
            return True

        def _stage_zip(upload: UploadFile, stop: threading.Event) -> int:
            """Unpack an uploaded zip into the staging area; return how many
            media entries were staged.

            Media entries are staged and .txt entries kept as sidecars, with
            the one-shot import's entry and size guards. Blocking file I/O
            (the archive is a spooled temporary file and every entry is copied
            to disk a chunk at a time): call it through
            ``_copy_upload_off_the_loop``. Raises ``_UploadCopyStopped`` before
            the next entry, or the next chunk of one, once *stop* is set.
            """
            upload_file = upload.file
            upload_file.seek(0, 2)
            upload_size = upload_file.tell()
            upload_file.seek(0)
            if upload_size == 0:
                return 0
            if upload_size > _STAGING_MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Zip '{upload.filename}' exceeds the "
                        f"{_STAGING_MAX_FILE_BYTES // 1024**3} GB limit"
                    ),
                )
            with zipfile.ZipFile(upload_file) as zip_file:
                entries = [i for i in zip_file.infolist() if not i.is_dir()]
                if len(entries) > _STAGING_MAX_ZIP_ENTRIES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Zip '{upload.filename}' contains too "
                            f"many files (max "
                            f"{_STAGING_MAX_ZIP_ENTRIES:,})."
                        ),
                    )
                if (
                    sum(i.file_size for i in entries)
                    > _STAGING_MAX_ZIP_DECOMPRESSED_BYTES
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Zip '{upload.filename}' decompressed "
                            f"size exceeds the "
                            f"{_STAGING_MAX_ZIP_DECOMPRESSED_BYTES // 1024**3}"
                            " GB limit."
                        ),
                    )
                added = 0
                for info in entries:
                    if stop.is_set():
                        raise _UploadCopyStopped(
                            f"unpacking of {upload.filename!r} stopped"
                        )
                    inner_ext = os.path.splitext(info.filename)[1].lower()
                    base_name = os.path.basename(info.filename)
                    if inner_ext in STAGING_ALLOWED_CAPTION_EXTS:
                        with zip_file.open(info) as handle:
                            data = handle.read(_MAX_SIDECAR_BYTES + 1)
                        if len(data) > _MAX_SIDECAR_BYTES:
                            logger.warning(
                                "Staging %s: leaving out caption %s in zip %s: "
                                "larger than %d bytes",
                                staging_id,
                                info.filename,
                                upload.filename,
                                _MAX_SIDECAR_BYTES,
                            )
                        elif data:
                            batch_sidecars.setdefault(
                                (
                                    _zip_member_folder(current_folder, info.filename),
                                    _normalise_sidecar_stem(base_name),
                                ),
                                data.decode("utf-8", errors="ignore"),
                            )
                        continue
                    if inner_ext not in STAGING_ALLOWED_MEDIA_EXTS:
                        continue
                    if _stage_zip_entry(zip_file, info, inner_ext, stop):
                        added += 1
                return added

        try:
            for upload_index, upload in enumerate(file):
                current_key = keys[upload_index] if upload_index < len(keys) else None
                current_folder = folders[upload_index]
                if not upload.filename:
                    continue
                ext = os.path.splitext(upload.filename)[1].lower()

                # --- Caption sidecar (.txt): stored by folder and stem,
                # matched to an image at commit; mirrors the one-shot import's
                # sidecar_text_by_key. ---
                if ext in STAGING_ALLOWED_CAPTION_EXTS:
                    try:
                        await upload.seek(0)
                        raw = await upload.read(_MAX_SIDECAR_BYTES + 1)
                        if len(raw) > _MAX_SIDECAR_BYTES:
                            logger.warning(
                                "Staging %s: skipping caption %s: larger than %d bytes",
                                staging_id,
                                upload.filename,
                                _MAX_SIDECAR_BYTES,
                            )
                            skipped.append(upload.filename)
                            continue
                        batch_sidecars.setdefault(
                            (current_folder, _normalise_sidecar_stem(upload.filename)),
                            raw.decode("utf-8", errors="ignore"),
                        )
                        received.append(upload.filename)
                    except Exception as exc:
                        logger.warning(
                            "Staging %s: failed to read sidecar %s: %s",
                            staging_id,
                            upload.filename,
                            exc,
                        )
                        skipped.append(upload.filename)
                    continue

                # --- Zip archive: extracted server-side into the staging
                # area, on a worker thread. A cancelled request waits for the
                # unpacking to stop before its ``finally`` releases anything. ---
                if ext == ".zip":
                    try:
                        if await _copy_upload_off_the_loop(
                            staging_id, upload.filename, _stage_zip, upload
                        ):
                            received.append(upload.filename)
                        else:
                            logger.warning(
                                "Staging %s: no valid media in zip %s",
                                staging_id,
                                upload.filename,
                            )
                            skipped.append(upload.filename)
                    except HTTPException:
                        raise
                    except zipfile.BadZipFile:
                        logger.error(
                            "Staging %s: invalid zip file %s",
                            staging_id,
                            upload.filename,
                        )
                        skipped.append(upload.filename)
                    continue

                # --- Plain media: copied to disk a chunk at a time, on a
                # worker thread. A cancelled request waits for the copy to
                # stop before its ``finally`` releases anything. ---
                if ext not in STAGING_ALLOWED_MEDIA_EXTS:
                    logger.warning(
                        "Staging %s: skipping unsupported file %s",
                        staging_id,
                        upload.filename,
                    )
                    skipped.append(upload.filename)
                    continue
                _check_file_cap()
                dest_name = f"{uuid.uuid4()}{ext}"
                dest_path = os.path.join(staging_dir, dest_name)
                written_paths.append(dest_path)
                try:
                    written = await _copy_upload_off_the_loop(
                        staging_id,
                        upload.filename,
                        _copy_upload_part,
                        upload,
                        dest_path,
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    _remove_batch_files(staging_id, [dest_path])
                    if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
                        raise _staging_disk_full(staging_id, staging_dir, exc) from exc
                    logger.error(
                        "Staging %s: failed to write %s to %s: %s",
                        staging_id,
                        upload.filename,
                        dest_path,
                        exc,
                    )
                    skipped.append(upload.filename)
                    continue
                if written == 0:
                    skipped.append(upload.filename)
                    _remove_batch_files(staging_id, [dest_path])
                    continue
                batch_entries.append(
                    {
                        "file_path": dest_path,
                        "original_file_name": os.path.basename(upload.filename),
                        "client_key": current_key,
                        "sidecar_folder": current_folder,
                    }
                )
                received.append(upload.filename)

            # Publish: the whole batch joins the session at once, and only if
            # the session is still open and has room for all of it.
            with session["lock"]:
                if session["stage"] != "staging":
                    raise _staging_conflict(
                        "staging_closed",
                        "Staging session stopped accepting files while this "
                        f"batch was uploading (stage={session['stage']})",
                        session,
                    )
                staged_files: list[dict] = session["staged_files"]
                if len(staged_files) + len(batch_entries) > _STAGING_MAX_FILES:
                    raise _staging_file_limit(len(staged_files))
                staged_files.extend(batch_entries)
                sidecar_text_by_key: dict[tuple[str, str], str] = session[
                    "sidecar_text_by_key"
                ]
                for sidecar_key, text in batch_sidecars.items():
                    sidecar_text_by_key.setdefault(sidecar_key, text)
                response = {
                    "staging_id": staging_id,
                    "staged": len(staged_files),
                    "received": received,
                    "skipped": skipped,
                    "sidecars": len(sidecar_text_by_key),
                    "replayed": False,
                }
                if record is not None:
                    record["state"] = "done"
                    record["response"] = dict(response)
                session["last_update_epoch_ms"] = int(time.time() * 1000)
                published = True
        finally:
            if not published:
                _remove_batch_files(staging_id, written_paths)
            with session["lock"]:
                if (
                    not published
                    and record is not None
                    and session["batches"].get(batch_key) is record
                ):
                    # A refused attempt is not remembered: it left nothing
                    # staged, so the client may simply send it again.
                    del session["batches"][batch_key]
                session["inflight"] -= 1
                finish_cancel = (
                    session["cleanup_pending"]
                    and session["inflight"] == 0
                    and not session["removing"]
                )
                if finish_cancel:
                    session["cleanup_pending"] = False
                    session["removing"] = True
            if finish_cancel and _remove_session_dir(server, staging_id, session):
                logger.info(
                    "Staging %s: removed the cancelled session's files after its "
                    "last upload ended",
                    staging_id,
                )

        logger.debug(
            "Staging %s: received %d, skipped %d, total staged %d, sidecars %d",
            staging_id,
            len(received),
            len(skipped),
            response["staged"],
            response["sidecars"],
        )
        return response

    @router.post(
        "/pictures/import/staging/{staging_id}/commit",
        summary="Hand off a staging session to the background import",
        description=(
            "Closes the unsafe window and hands the staged files off to a "
            "background PictureImportTask on the shared task runner. The import "
            "completes server-side (the safe window); the tab may now close. "
            "Progress is reported via GET "
            "/pictures/import/staging/{staging_id}/status and the task manager.\n\n"
            "Commit is idempotent per staging session: once a commit has started "
            'the import, a repeat returns the same `task_id` with `"replayed": '
            "true` and starts nothing. A commit while another is still handing "
            "the session over gets 409 `commit_in_progress`; while an upload is "
            "still being received, 409 `uploads_in_flight`; after a cancel, 409 "
            "`staging_closed`. Each 409 body carries the session's `stage` and "
            "`task_id`. While the import runs it holds the library, so a library "
            "switch is refused until it ends."
        ),
        response_model=StagingCommitResponse,
    )
    def commit_staging_session(request: Request, staging_id: str):
        session = _get_session_or_404(request, staging_id)
        # The transition to ``committing`` is taken under the lock, so exactly
        # one commit builds the import; a repeat after it succeeded gets its
        # task id back.
        with session["lock"]:
            if session.get("task_id") is not None:
                logger.info(
                    "Staging %s: commit repeated; import %s already started",
                    staging_id,
                    session["task_id"],
                )
                return {
                    "staging_id": staging_id,
                    "task_id": session["task_id"],
                    "staged_count": len(session["staged_files"]),
                    "replayed": True,
                }
            if session["stage"] == "committing":
                raise _staging_conflict(
                    "commit_in_progress",
                    "This staging session is already being handed to the import.",
                    session,
                )
            if session["stage"] != "staging":
                raise _staging_conflict(
                    "staging_closed",
                    "Staging session can no longer be committed "
                    f"(stage={session['stage']})",
                    session,
                )
            if session["inflight"] > 0:
                raise _staging_conflict(
                    "uploads_in_flight",
                    "Files are still being uploaded into this staging session. "
                    "Commit once every upload has finished.",
                    session,
                )
            if not session["staged_files"]:
                raise HTTPException(status_code=400, detail="No staged files to import")
            session["stage"] = "committing"
            staged_files: list[dict] = list(session["staged_files"])
            sidecar_text_by_key: dict[tuple[str, str], str] = dict(
                session["sidecar_text_by_key"]
            )

        task: Optional[PictureImportTask] = None
        library_lease = None
        try:
            face_worker_problem = server.vault.worker_unavailable_reason(
                TaskType.FACE_EXTRACTION
            )
            if face_worker_problem:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot import: {face_worker_problem}.",
                )

            # Re-validate the drop target at handoff (it may have been
            # deleted/locked since open); still fail loudly rather than
            # silently skip association.
            _validate_association_targets(
                session["set_id"], session["character_id"], session["project_id"]
            )

            # Resolve caption sidecars to their images by folder and basename
            # stem, parsing each via the one-shot import's tag rule
            # (comma-separated tag lists). A caption only matches an image in
            # its own folder; one at the top of the drop only matches an image
            # there. Orphan sidecars (no matching staged image) are logged and
            # skipped, not applied.
            sidecar_tags_by_key: dict[tuple[str, str], list[str]] = {}
            if sidecar_text_by_key:
                media_keys = {
                    (
                        entry["sidecar_folder"],
                        _normalise_sidecar_stem(entry.get("original_file_name") or ""),
                    )
                    for entry in staged_files
                }
                for sidecar_key, raw_text in sidecar_text_by_key.items():
                    if sidecar_key not in media_keys:
                        logger.info(
                            "Staging %s: caption sidecar %r in folder %r has no "
                            "matching image; skipping",
                            staging_id,
                            sidecar_key[1],
                            sidecar_key[0],
                        )
                        continue
                    parsed = _parse_sidecar_tags(raw_text)
                    if parsed:
                        sidecar_tags_by_key[sidecar_key] = parsed

            # Guard against committing more than the vault can hold: the
            # background import copies each staged file into the canonical vault
            # location before the staging dir is removed, so it briefly needs
            # room for both. A volume whose free space cannot be read is refused
            # too: starting an import that may not fit is not a safe default.
            try:
                staged_bytes = sum(
                    os.path.getsize(entry["file_path"])
                    for entry in staged_files
                    if entry.get("file_path") and os.path.isfile(entry["file_path"])
                )
                free_bytes = shutil.disk_usage(server.vault.image_root).free
            except OSError as exc:
                logger.error(
                    "Staging %s: cannot check free space before import: %s",
                    staging_id,
                    exc,
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Cannot import: could not check free space on the "
                        f"library volume ({exc.strerror or exc})."
                    ),
                ) from exc
            if int(staged_bytes * 1.1) > free_bytes:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        "Not enough disk space to finish the import "
                        f"({staged_bytes / 1024**3:.2f} GB staged, "
                        f"{free_bytes / 1024**3:.2f} GB free)."
                    ),
                )

            # The import holds the library until it settles, so a library
            # switch cannot close the vault under it. The task runner's
            # completion callback releases the lease (see Vault).
            library_lease = server.library_coordinator.acquire_read()
            if library_lease is None:
                raise HTTPException(
                    status_code=503,
                    detail="The library is switching or unavailable. Try again.",
                )
            task = PictureImportTask(
                server.vault,
                staged_files,
                project_id=session["project_id"],
                project_created_at=session["project_created_at"],
                set_id=session["set_id"],
                character_id=session["character_id"],
                sidecar_tags_by_key=sidecar_tags_by_key,
                staging_id=staging_id,
                staging_dir=session["staging_dir"],
                origin_client_id=session["origin_client_id"],
                restore_context=operation_log_service.request_context(request),
                library_lease_release=functools.partial(
                    server.library_coordinator.release_read, library_lease
                ),
            )
            task_id = server.vault.submit_task(task)
            if task_id is None:
                raise HTTPException(
                    status_code=503, detail="Task runner is not available"
                )
        except BaseException:
            # Refused before the import started: the session goes back to
            # accepting files and the library is let go.
            if task is not None:
                task.release_library_lease()
            elif library_lease is not None:
                server.library_coordinator.release_read(library_lease)
            with session["lock"]:
                session["stage"] = "staging"
            raise

        with session["lock"]:
            session["task"] = task
            session["task_id"] = task_id
            session["stage"] = "importing"
            session["last_update_epoch_ms"] = int(time.time() * 1000)
        logger.info(
            "Staging session committed: staging_id=%s task_id=%s staged_count=%d",
            staging_id,
            task_id,
            len(staged_files),
        )
        return {
            "staging_id": staging_id,
            "task_id": task_id,
            "staged_count": len(staged_files),
            "replayed": False,
        }

    @router.delete(
        "/pictures/import/staging/{staging_id}",
        summary="Cancel a staging session",
        description=(
            "Cancels an async import staging session that has not yet been "
            "committed, discarding its streamed files. After commit the import "
            "runs to completion in the background and cannot be cancelled here "
            "(409 `import_committed`, with the session's `stage` and `task_id`). "
            "If an upload is still being received, the files are removed when "
            'it ends and the answer says `"cleanup": "deferred"`; that upload '
            "is refused with 409 `staging_closed`."
        ),
        response_model=StagingStatusResponse,
    )
    def cancel_staging_session(request: Request, staging_id: str):
        session = _get_session_or_404(request, staging_id)
        with session["lock"]:
            if (
                session.get("task_id") is not None
                or session["stage"] in _STAGING_HANDED_OFF_STAGES
            ):
                raise _staging_conflict(
                    "import_committed",
                    "Cannot cancel a committed staging session "
                    f"(stage={session['stage']})",
                    session,
                )
            session["stage"] = "cancelled"
            session["last_update_epoch_ms"] = int(time.time() * 1000)
            staged_count = len(session["staged_files"])
            remove_now = False
            if session["inflight"] > 0:
                # The last upload still being received removes the files as it
                # ends; deleting them now would pull them from under it.
                session["cleanup_pending"] = True
            elif not session["removing"]:
                session["removing"] = True
                remove_now = True
        cleanup = "deferred"
        if remove_now and _remove_session_dir(server, staging_id, session):
            cleanup = "done"
        logger.info(
            "Staging session cancelled: staging_id=%s cleanup=%s", staging_id, cleanup
        )
        return {
            "staging_id": staging_id,
            "stage": "cancelled",
            "staged": staged_count,
            "total": staged_count,
            "processed": 0,
            "cleanup": cleanup,
        }

    @router.get(
        "/pictures/import/staging/{staging_id}/status",
        summary="Get async import staging status",
        description=(
            "Returns the stage and live progress of a staging session and its "
            "background import task. While the import runs, `phase` says what "
            "it is doing: `queued`, `processing`, `persisting` or `finalizing`."
        ),
        response_model=StagingStatusResponse,
    )
    def staging_status(request: Request, staging_id: str, outcomes_from: int = 0):
        session = _get_session_or_404(request, staging_id)
        with session["lock"]:
            staged_count = len(session["staged_files"])
            task = session.get("task")
            stage = session["stage"]
            error = session.get("error")
            task_id = session.get("task_id")
        processed = 0
        total = staged_count
        phase = None
        imported_count = duplicate_count = failed_count = None
        scrapheaped_count = cancelled_count = None
        scrapheaped_picture_ids: list[int] = []
        outcomes: list[dict] = []

        if task is not None:
            total = int(getattr(task, "_total_count", total) or total)
            processed = int(getattr(task, "_processed_count", 0) or 0)
            # Only the tail the caller has not seen: a 2,000-file import is
            # polled every second, and the whole list each time grows with it.
            outcomes = list(task.outcomes[max(0, outcomes_from) :])
            status = getattr(task, "status", None)
            status_value = getattr(status, "value", status)
            if status_value == "completed":
                stage = "completed"
                result = task.result if isinstance(task.result, dict) else {}
                imported_count = result.get("imported_count")
                duplicate_count = result.get("duplicate_count")
                scrapheaped_count = result.get("scrapheaped_count")
                scrapheaped_picture_ids = result.get("scrapheaped_picture_ids") or []
                failed_count = result.get("failed_count")
                cancelled_count = result.get("cancelled_count")
            elif status_value == "failed":
                stage = "failed"
                error = str(getattr(task, "error", None) or "Import failed")
            elif status_value == "cancelled":
                stage = "cancelled"
            else:
                stage = "importing"
                phase = "queued" if status_value == "pending" else task.phase
            if status_value in ("completed", "failed", "cancelled"):
                with session["lock"]:
                    session["stage"] = stage
                    if stage == "failed":
                        session["error"] = error
                    # Keep the reaper's terminal grace window counting from the
                    # last poll, so an actively-polled finished session survives
                    # and a forgotten one ages out.
                    session["last_update_epoch_ms"] = int(time.time() * 1000)

        return {
            "staging_id": staging_id,
            "stage": stage,
            "phase": phase,
            "staged": staged_count,
            "total": total,
            "processed": processed,
            "task_id": task_id,
            "imported_count": imported_count,
            "duplicate_count": duplicate_count,
            "scrapheaped_count": scrapheaped_count,
            "scrapheaped_picture_ids": scrapheaped_picture_ids,
            "failed_count": failed_count,
            "cancelled_count": cancelled_count,
            "error": error,
            "outcomes": outcomes,
        }
