"""Models PixlStash downloads for a workflow, from ComfyUI's own template list.

A workflow that names a model this machine does not have is refused with
``missing_models`` (``services/workflow_run_service.py``), and each missing
file carries ``downloadable`` when :func:`download_source` knows where to get
it: a file of the same name in the committed template list
(``data/workflow_templates.json``, #1999), which gives its link and its
ComfyUI folder. **The link is the server's, never the caller's**: a request
names a file, so no client can point the server at an address of its choosing.
Where the committed list has no such file, the templates the owner's ComfyUI
serves are asked (:func:`live_download_source`), for one ComfyUI shipped after
the list was last rebuilt; a link from there is taken only when
:func:`trusted_live_link` holds (https on ``huggingface.co``, a format loading
cannot run as code), and carries no digest to check the download against.
:func:`start_download` fetches it into the download folder (:func:`~pixlstash.services.managed_model_store.find_download_folder`:
the one marked in the Model folders section, else the default folder) and
registers it on the shelf, so the next run finds it there.

**One thread per download, followed in memory.** A download is minutes of
waiting on a socket, not work for the task runner's queues, and the runs in
flight are followed the same way (``comfyui_service.live_runs``). At most
:data:`MAX_CONCURRENT_DOWNLOADS` fetch at once; a later one reads ``queued``
until one of them ends, and can be cancelled while it waits.
:func:`live_downloads` is what ``GET /workers/progress`` reports as
``downloads``; a finished one stays in it for :data:`FINISHED_KEEP_SECONDS` so
every client polling sees how it ended. ponytail: in memory, so a restart
forgets a download in flight and leaves its partial file behind (inert: no row
names it and the scanner skips the suffix).

**The order cannot leave a half file under the real name.** Bytes go to
``<name>.pixlstash-partial`` beside the destination, digested on the way in;
only a complete body is published (no clobber) and registered with that
digest. A file the scanner cannot read as a model (a login page saved under a
model's name) is removed again and reported, rather than left on the shelf.
Cancel, failure and refusal all discard the partial file, and a file the
hub then refuses to register is removed again. A response with no
``Content-Length`` is refused, since there would be no space check.

**What a failed download says is the server's own sentence.** A refusal and
a failed entry's ``error`` (shown to every client polling
``GET /workers/progress``) never repeat the caller's ``file``, which may carry
any subfolder in front of the name the list matched, nor a library's error
text: a :class:`DownloadStopped` is composed from fixed text, sizes, digests,
the registered folder and the hosts the link led to, and anything else is
answered with :data:`DOWNLOAD_FAILED` and logged with its traceback (CWE-209).

**Where it lands inside the folder.** The name is the one the workflow asked
for, its own subfolders included (``SDXL/foo.safetensors``). When the folder
has a subfolder named for the ComfyUI model folder the loader reads
(``loras``, ``checkpoints``), the file goes in it, so a ComfyUI ``models``
folder registered on the shelf is filled the way ComfyUI itself would read it.

**A folder being moved takes no download, and one being downloaded into is not
moved.** A move, removal or rename of a model folder's directory (the
relocations, ``dirs/move``, ``dirs/remove``, the rename) reserves that
directory with :func:`reserve_folder` for as long as its job runs, and
releases it with :func:`release_folder` however the job ends. The reservation
is refused (:class:`FolderBusy`) while a download that has not finished writes
inside the directory, and :func:`start_download` refuses
(:data:`FOLDER_BEING_MOVED`) a file whose destination lies inside a reserved
one. Both checks and both registrations happen under ``_lock``, so neither
side can slip in between the other's check and its registration. A download
counts until its thread has ended (``_done``), not until it reads
``cancelled``: a cancelled one can still be finishing a connect, and its
thread then removes its partial. ``SHELF_IO_LOCK`` alone is no answer: an
import, a forget and the intake hold it too, for minutes, and none of them
touches a download's folder.

**Lock order: ``SHELF_IO_LOCK``, then ``_lock``; never the other way.**
:func:`_publish` takes ``_lock`` while it holds ``SHELF_IO_LOCK``, and so do
the routes that reserve a folder, which reserve once they hold it. Nothing
here waits for ``SHELF_IO_LOCK`` while it holds ``_lock``: every ``_lock``
section is a short read or write of the registry, and :func:`_publish` waits
for ``SHELF_IO_LOCK`` before it takes ``_lock``. ``_slots`` and the adapter's
own lock are taken outside ``_lock`` too.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
import uuid
from functools import lru_cache
from typing import Callable, Iterator, Optional
from urllib.parse import urljoin

import requests

from pixlstash.hub.db import HubDatabase
from pixlstash.pixl_logging import get_logger
from pixlstash.services import comfyui_templates
from pixlstash.services.managed_model_store import find_download_folder
from pixlstash.services.model_folder_scanner import (
    SHELF_MODEL_SUFFIXES,
    ModelFolderScanner,
)
from pixlstash.services.workflow_templates import directory_slot, load_templates
from pixlstash.services.model_mover import (
    PARTIAL_SUFFIX,
    SHELF_IO_LOCK,
    MoveRefused,
    discard_partial,
    publish_no_clobber,
    require_space,
)
from pixlstash.utils.adapter_header import folder_name_role
from pixlstash.utils.bounded_http import (
    HandRedirectSession,
    PinnedAdapter,
    TargetRefused,
    stalled,
    vet_redirect_target,
    vet_target,
    wire_split,
)
from pixlstash.utils.path_utils import real_path, resolve_path_within

logger = get_logger(__name__)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
# A download that has not ended: waiting for a slot, or fetching.
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_RUNNING)

# How many downloads fetch at once; a later one waits as queued. Each is a
# thread and a connection held for minutes, and a click on every missing model
# of a large workflow would otherwise open them all together.
MAX_CONCURRENT_DOWNLOADS = 3
# How often a queued download looks again for a slot, and for its cancel.
_QUEUE_POLL_SECONDS = 0.25

# Long enough for a client on the idle 5 s poll to see the ending twice.
FINISHED_KEEP_SECONDS = 30

# What a link from a live ComfyUI template may be (`trusted_live_link`).
LIVE_HOSTS = frozenset({"huggingface.co"})
LIVE_SUFFIXES = (".safetensors", ".gguf")
# Where a live link's redirects may go: Hugging Face and its CDN. Every hop is
# checked before it is asked, since a live link has no digest to catch a file
# that came from somewhere else.
LIVE_HOP_DOMAINS = ("huggingface.co", "hf.co")
_MAX_HOPS = 10

# Small enough that a cancel is seen between two chunks of a slow body; a
# cancel during a chunk is ended by `cancel_download` shutting the socket.
_CHUNK_BYTES = 64 * 1024
# Each connect attempt (one per checked address), then the longest any one
# receive may wait. A receive can return a byte at a time, so this alone does
# not end a trickle: the stall floor below does.
_TIMEOUT_SECONDS = (15, 120)
# A response, headers and body alike, must bring this many bytes in every
# window this long or it is given up (`bounded_http.PinnedAdapter`'s
# `stall_floor`). The window is the read timeout's own, so a link that pauses
# no longer than it was already allowed to still downloads.
_STALL_BYTES = 64 * 1024
_STALL_SECONDS = _TIMEOUT_SECONDS[1]

# Refusal and failure texts the server writes. None repeats the caller's file
# name: a request may put any subfolder in front of the name the list matched.
UNKNOWN_SOURCE = "PixlStash does not know where to download that file from."
OUTSIDE_FOLDER = "That file would be written outside the download folder."
ALREADY_DOWNLOADING = "That file is already being downloaded."
# A retry of a cancelled download whose thread has not ended yet: the cancel
# was taken, and the thread is finishing a connect (at most 15 s).
STILL_STOPPING = "That file's download is still stopping. Try again in a few seconds."
# A start whose destination lies inside a folder a move, removal or rename is
# working on (`reserve_folder`).
FOLDER_BEING_MOVED = (
    "The folder this file would download into is being moved or removed. "
    "Try again when that is done."
)
START_FAILED = "The download could not be started. The server log says why."
DOWNLOAD_FAILED = "The download failed. The server log says why."
DOWNLOAD_STALLED = "The download stopped receiving data. Try again."
DOWNLOAD_ENDED_EARLY = "The download ended early. Try again."

_downloads: dict[str, dict] = {}
# The directories moves, removals and renames are working on, by the token
# `reserve_folder` handed out for each.
_reserved: dict[str, str] = {}
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_DOWNLOADS)


class DownloadRefused(ValueError):
    """The download was refused before a byte was fetched."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class DownloadStopped(ValueError):
    """A download ended for a reason the server put in its own words.

    Its message becomes the failed entry's ``error``, so it is composed only
    from fixed text, sizes, digests, the registered download folder and the
    host a link led to: never the caller's file name or subfolders, and never
    a library's error text, which goes to the log instead.
    """


class _Cancelled(Exception):
    """The owner cancelled the download."""


class FolderBusy(Exception):
    """A download that has not finished writes inside the folder, so it cannot
    be reserved for a move, a removal or a rename (:func:`reserve_folder`)."""


@lru_cache(maxsize=1)
def _sources() -> dict[str, dict]:
    """Every file the template list links, by case-folded name.

    Where one name links several files (``flux2-vae.safetensors`` is more than
    one VAE), the one most templates link: list order alone would let a new
    template's variant displace the file every other workflow means.
    """
    by_name: dict[str, dict[Optional[str], list[dict]]] = {}
    for template in load_templates().get("templates") or []:
        for entry in template.get("files") or []:
            name, url = entry.get("name"), entry.get("url")
            if name and isinstance(url, str) and url.startswith("https://"):
                by_name.setdefault(name.casefold(), {}).setdefault(
                    entry.get("sha256"), []
                ).append(
                    {
                        "url": url,
                        "directory": entry.get("directory"),
                        "sha256": entry.get("sha256"),
                    }
                )
    # max() keeps the first on a tie, so an unambiguous name is unchanged.
    return {name: max(files.values(), key=len)[0] for name, files in by_name.items()}


def download_source(file: str) -> Optional[dict]:
    """Where *file* can be downloaded from, ``{url, directory, sha256}``, or None.

    ``sha256`` is the digest the list records for the link, or None.

    By the file's own name, case folded and without the subfolder a loader may
    put in front of it: the list names files the way ComfyUI's loaders do.
    """
    name = os.path.basename((file or "").replace("\\", "/"))
    return _sources().get(name.casefold()) if name else None


def trusted_live_link(name: str, url: Optional[str]) -> bool:
    """Whether a link a live ComfyUI template gives for *name* may be fetched.

    Only ``https://huggingface.co/...``, exactly that host (no userinfo, no
    port), and only for a file format that loading cannot run as code: not
    ``.pth``, which is a pickle. The committed list's links were reviewed; a
    live template's were not, so it gets the narrower rule. The host is read
    as ``requests`` would send the link (``bounded_http.wire_split``), so a
    link ``urlsplit`` would place on Hugging Face cannot reach another host.
    """
    if not isinstance(url, str) or not name.lower().endswith(LIVE_SUFFIXES):
        return False
    parsed = _wire_parts(url)
    return (
        parsed is not None and parsed.scheme == "https" and parsed.netloc in LIVE_HOSTS
    )


def live_sources(comfyui_url: Optional[str], object_info: Optional[dict]) -> dict:
    """Every file the templates *comfyui_url* serves link trustably, by name.

    The fallback to :func:`download_source`, for a template ComfyUI shipped
    after the committed list was last rebuilt: ``{case-folded name: {url,
    directory, sha256: None, live: True}}`` for each file whose link passes
    :func:`trusted_live_link` (a live template records no digest), the first
    template's link where two share a name, as the committed list does.

    The listing is the one ``GET /comfyui/templates`` keeps for ten minutes
    (:func:`~pixlstash.services.comfyui_templates.read_listing`). Empty while
    ComfyUI is not answering (*object_info* None): the read would only wait
    out its timeouts, and the listing it kept would say no pack loaded.
    """
    if not comfyui_url or object_info is None:
        return {}
    try:
        listing = comfyui_templates.read_listing(comfyui_url, object_info)
    except comfyui_templates.TemplatesUnavailable as exc:
        logger.info(
            "No templates from ComfyUI at %s to find download links in: %s",
            comfyui_url,
            exc,
        )
        return {}
    found: dict[str, dict] = {}
    for template in listing.templates:
        for item in template.files:
            name = os.path.basename(item.name.replace("\\", "/"))
            if trusted_live_link(name, item.url):
                found.setdefault(
                    name.casefold(),
                    {
                        "url": item.url,
                        "directory": item.directory,
                        "sha256": None,
                        "live": True,
                    },
                )
    return found


def live_download_source(
    file: str, comfyui_url: Optional[str], object_info: Optional[dict]
) -> Optional[dict]:
    """Where *file* can be downloaded from by the templates *comfyui_url*
    serves (:func:`live_sources`), or None."""
    name = os.path.basename((file or "").replace("\\", "/")).casefold()
    return live_sources(comfyui_url, object_info).get(name) if name else None


def digest_source(sha256: str) -> Optional[dict]:
    """The file the template list links under *sha256*, or None.

    ``{name, url, directory}``, for a loader that names its model by digest
    alone (ComfyUI-PixlStash's): the list records the digest Hugging Face
    states for each link (``scripts/build_template_suggestions.py``).
    """
    digest = (sha256 or "").strip().lower()
    for template in load_templates().get("templates") or []:
        for entry in template.get("files") or []:
            url = entry.get("url")
            if (
                digest
                and entry.get("sha256") == digest
                and entry.get("name")
                and isinstance(url, str)
                and url.startswith("https://")
            ):
                return {
                    "name": entry["name"],
                    "url": url,
                    "directory": entry.get("directory"),
                }
    return None


def _placement(
    folder_path: str, file: str, comfyui_folder: Optional[str]
) -> tuple[Optional[str], str]:
    """Where *file* goes inside the download folder.

    Returns:
        The subfolder the server puts it under (``None`` for the folder's
        top), and its path relative to the folder, the caller's own
        subfolders included.
    """
    name = file.replace("\\", "/").strip().strip("/")
    if not comfyui_folder:
        return None, name
    head, sep, rest = name.partition("/")
    if sep and head.casefold() == comfyui_folder.casefold():
        # Already under it, in whatever case: once, in the folder's own
        # subfolder for it, else spelled as the list does.
        subfolder = _subfolder_for(folder_path, comfyui_folder) or comfyui_folder
        return subfolder, f"{subfolder}/{rest}"
    subfolder = _subfolder_for(folder_path, comfyui_folder)
    return subfolder, (f"{subfolder}/{name}" if subfolder else name)


def _subfolder_for(folder_path: str, comfyui_folder: str) -> Optional[str]:
    """The download folder's own subfolder for ComfyUI's *comfyui_folder*.

    That name as spelled, else a subfolder naming the same role however a
    launcher spells it (Stability Matrix's `VAE`, `TextEncoders`,
    `DiffusionModels`). The folder is what tells the shelf a VAE or a text
    encoder of a layout its tensors do not show (``role_from_folder``); one
    left at the top level is filed by its parameter count. ``None`` when the
    folder has neither.
    """
    if os.path.isdir(os.path.join(folder_path, comfyui_folder)):
        return comfyui_folder
    role = folder_name_role(comfyui_folder)
    if role is None:
        return None
    try:
        names = sorted(
            entry.name for entry in os.scandir(folder_path) if entry.is_dir()
        )
    except OSError as exc:
        logger.warning(
            "Could not list %s for a %s subfolder; downloading to its top: %s",
            folder_path,
            comfyui_folder,
            exc,
        )
        return None
    return next((name for name in names if folder_name_role(name) == role), None)


def start_download(
    hub: HubDatabase,
    file: str,
    on_arrived: Optional[Callable[[int], None]] = None,
    comfyui_url: Optional[str] = None,
    object_info: Optional[dict] = None,
) -> dict:
    """Start fetching *file* onto the shelf, and return its entry.

    Args:
        hub: The hub database holding the shelf tables.
        file: The model name the workflow asks for, as its loader spells it.
        on_arrived: Called with the new model id once it is on the shelf.
        comfyui_url: The owner's ComfyUI, whose templates are asked for a link
            when the committed list has none (:func:`live_download_source`).
        object_info: That ComfyUI's map, for the listing should it be read.

    Raises:
        DownloadRefused: Nothing was fetched; ``status_code`` says why.
    """
    source = download_source(file) or live_download_source(
        file, comfyui_url, object_info
    )
    if source is None:
        raise DownloadRefused(UNKNOWN_SOURCE, 404)
    url = source["url"]
    if not file.lower().endswith(SHELF_MODEL_SUFFIXES):
        raise DownloadRefused(
            "That file is not a model file the shelf keeps "
            f"({', '.join(SHELF_MODEL_SUFFIXES)})."
        )
    folder = find_download_folder(hub)
    if folder is None:
        raise DownloadRefused(
            "There is no folder to download into. Add a model folder first.", 409
        )
    if not os.path.isdir(folder["path"]):
        raise DownloadRefused(
            f"{folder['path']} is not a readable directory right now.", 409
        )
    subfolder, relpath = _placement(folder["path"], file, source["directory"])
    try:
        target = resolve_path_within(folder["path"], relpath)
    except ValueError as exc:
        logger.error(
            "Refusing to download %r into %s: it resolves outside it (%s).",
            file,
            folder["path"],
            exc,
        )
        raise DownloadRefused(OUTSIDE_FOLDER) from exc
    if os.path.lexists(target):
        raise DownloadRefused(f"That file is already in {folder['path']}.", 409)

    entry = {
        "id": uuid.uuid4().hex,
        "file": os.path.basename(target),
        "host": _host_named(url),
        "folder_path": folder["path"],
        "status": STATUS_QUEUED,
        "received": 0,
        "total": 0,
        "error": None,
        # The model's page, where a gated one's licence is accepted.
        "page_url": None,
        "model_id": None,
        "ended_at": None,
        # What a workflow or a set can name it by while it is still on its way:
        # the digest the list records (None for a gated link), the ComfyUI
        # folder it is for, and the set slot that folder fills.
        # Lowercased as every digest compared against it is.
        "sha256": (source.get("sha256") or "").lower() or None,
        "directory": source.get("directory"),
        "slot": directory_slot(source.get("directory")),
        # Where a gated file is to be saved by hand, in the server's words:
        # the registered folder and the ComfyUI subfolder the download would
        # use, not the subfolders the caller put in front of the name.
        "_save_into": (
            os.path.join(folder["path"], subfolder) if subfolder else folder["path"]
        ),
    }
    with _lock:
        # Case folded: on macOS and Windows two spellings are one partial file.
        # Until its thread has ended, not merely until it reads cancelled: a
        # thread still finishing a connect would discard a retry's partial.
        # One the owner cancelled is only stopping, and says so.
        busy = next(
            (
                d
                for d in _downloads.values()
                if d["_target"].casefold() == target.casefold()
                and not d["_done"].is_set()
            ),
            None,
        )
        if busy is not None:
            stopping = busy["status"] == STATUS_CANCELLED
            raise DownloadRefused(
                STILL_STOPPING if stopping else ALREADY_DOWNLOADING, 409
            )
        # Checked under the lock `reserve_folder` takes, so a move reserving
        # this folder either sees the entry below or is seen here. `target`
        # and every reserved path are already resolved.
        moving = next(
            (path for path in _reserved.values() if _resolved_within(target, path)),
            None,
        )
        if moving is not None:
            logger.warning(
                "Refusing to download %s: %s is being moved, removed or renamed.",
                target,
                moving,
            )
            raise DownloadRefused(FOLDER_BEING_MOVED, 409)
        entry["_target"] = target
        entry["_cancel"] = threading.Event()
        # Set once the thread has ended and left nothing of its own behind.
        entry["_done"] = threading.Event()
        _downloads[entry["id"]] = entry
    thread = threading.Thread(
        target=_run,
        args=(
            hub,
            entry,
            url,
            int(folder["id"]),
            relpath,
            on_arrived,
            source.get("sha256"),
            bool(source.get("live")),
        ),
        name=f"model-download-{entry['id'][:8]}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError as exc:
        # Left registered as running, it would refuse every retry as "already
        # being downloaded" until a restart.
        with _lock:
            _downloads.pop(entry["id"], None)
        logger.error(
            "Could not start download %s of %s.", entry["id"], target, exc_info=True
        )
        raise DownloadRefused(START_FAILED, 503) from exc
    logger.info("Downloading %s from %s into %s.", relpath, entry["host"], target)
    return _public(entry)


def reserve_folder(path: str) -> str:
    """Reserve the directory *path* for a move, a removal or a rename.

    Until :func:`release_folder` is called with the token returned,
    :func:`start_download` refuses a file whose destination lies inside
    *path*, spelled directly or through a symlink. The caller releases it
    however its job ends: success, failure, refusal or exception. Callers that
    hold ``SHELF_IO_LOCK`` reserve after taking it (the module docstring's
    lock order).

    Raises:
        FolderBusy: A download that has not finished (its thread has not ended,
            whatever its status reads) writes inside *path*. Nothing was
            reserved.
    """
    # Resolved before the lock, and spelled as every download's `_target` was
    # (resolve_path_within), so the checks under it compare strings and never
    # wait on the filesystem.
    resolved = real_path(path)
    with _lock:
        busy = next(
            (
                d
                for d in _downloads.values()
                if not d["_done"].is_set() and _resolved_within(d["_target"], resolved)
            ),
            None,
        )
        if busy is not None:
            logger.info(
                "Not reserving %s: download %s (%s) writes %s inside it.",
                path,
                busy["id"],
                busy["status"],
                busy["_target"],
            )
            raise FolderBusy(path)
        token = uuid.uuid4().hex
        _reserved[token] = resolved
        return token


def release_folder(token: str) -> None:
    """End the reservation :func:`reserve_folder` returned *token* for."""
    with _lock:
        del _reserved[token]


def _resolved_within(path: str, base: str) -> bool:
    """Whether *path* lies inside the directory *base*, both already resolved.

    Both were passed through ``real_path`` before ``_lock`` was taken, so
    this only compares strings. A filesystem call under ``_lock`` would hold
    up every download request behind a folder on an unresponsive network share.
    """
    norm_path = os.path.normcase(path)
    norm_base = os.path.normcase(base)
    try:
        return os.path.commonpath([norm_path, norm_base]) == norm_base
    except ValueError:
        # Paths on different drives (Windows): one cannot hold the other.
        return False


def cancel_download(download_id: str) -> bool:
    """Stop a queued or running download.

    True says the cancel was taken: the entry reads ``cancelled`` at once,
    and nothing it fetched is kept. Its thread stops on its own and discards
    the partial file; only that thread can linger, and for at most one
    connect attempt (:data:`_TIMEOUT_SECONDS`, 15 s), since a connect in
    progress is not interrupted from here (no socket is closed across
    threads) and the socket it opens is closed before a handshake or a byte
    of the request. A request in flight has its socket shut down here, so a
    receive blocked on a link that sends nothing, or a byte at a time, ends
    at once rather than when the link next sends. Until the thread has ended
    the same file cannot be started again. False when no such download is
    queued or running, or it is already being published and will land.
    """
    with _lock:
        entry = _downloads.get(download_id)
        # Past `_committed` the file is being published and will land.
        if (
            entry is None
            or entry["status"] not in ACTIVE_STATUSES
            or entry.get("_committed")
        ):
            return False
        entry["_cancel"].set()
        entry["status"] = STATUS_CANCELLED
        entry["ended_at"] = time.monotonic()
        adapter = entry.get("_adapter")
    # Outside the lock: the shutdown takes the adapter's own.
    if adapter is not None:
        adapter.abort()
    return True


def live_downloads() -> list[dict]:
    """The downloads queued or running, and those that ended in the last few
    seconds. An entry is kept while its thread runs, so a cancelled one is
    still there to refuse a second start of its file."""
    now = time.monotonic()
    with _lock:
        for key in [
            key
            for key, d in _downloads.items()
            if d["ended_at"] is not None
            and now - d["ended_at"] > FINISHED_KEEP_SECONDS
            and d["_done"].is_set()
        ]:
            del _downloads[key]
        return [_public(d) for d in _downloads.values()]


def running_download(
    file: str, sha256: Optional[str] = None, folder: Optional[str] = None
) -> Optional[dict]:
    """The queued or running download of *file* or of *sha256*, or None.

    *file* by its own name, case folded and without the loader's subfolder, as
    :func:`download_source` reads it; *sha256* where the list recorded one. A
    run that names a model still on its way is refused rather than submitted
    (``workflow_run_service.MODELS_DOWNLOADING``).

    *folder*, the ComfyUI folder the caller's loader reads, keeps a name match
    to a download of the same kind: a LoRA that shares its file name with a
    checkpoint being fetched is not on its way. Compared by the set slot each
    folder fills (:func:`directory_slot`), so ``unet`` and
    ``diffusion_models``, or ``clip`` and ``text_encoders``, are one kind. A
    digest names one file wherever it goes, so it is matched alone, and two
    known digests that differ are two files whatever their names.
    """
    name = os.path.basename((file or "").replace("\\", "/")).casefold()
    digest = (sha256 or "").strip().lower()
    kind = directory_slot(folder) if folder else None
    with _lock:
        for entry in _downloads.values():
            if entry["status"] not in ACTIVE_STATUSES:
                continue
            if digest and entry.get("sha256") == digest:
                return _public(entry)
            same_kind = kind is None or entry.get("slot") in (None, kind)
            other_file = bool(digest and entry.get("sha256"))
            if (
                name
                and same_kind
                and not other_file
                and entry["file"].casefold() == name
            ):
                return _public(entry)
    return None


def _public(entry: dict) -> dict:
    shown = {k: v for k, v in entry.items() if not k.startswith("_")}
    shown.pop("ended_at", None)
    total = entry["total"]
    shown["percent"] = round(100 * entry["received"] / total, 1) if total else 0
    return shown


def _end(entry: dict, status: str, error: Optional[str] = None) -> None:
    with _lock:
        if entry["status"] == STATUS_CANCELLED:
            # Taken at once by cancel_download; whatever the thread then ran
            # into (the shut-down socket's error) is the cancel, and the file
            # cannot have landed (`_committed` refuses a cancel).
            if status != STATUS_CANCELLED:
                logger.info(
                    "Download %s was cancelled; its thread ended as %s (%s).",
                    entry["id"],
                    status,
                    error,
                )
            return
        entry["status"] = status
        entry["error"] = error
        entry["ended_at"] = time.monotonic()


@contextlib.contextmanager
def _slot(entry: dict, cancel: threading.Event) -> Iterator[None]:
    """Hold one of the :data:`MAX_CONCURRENT_DOWNLOADS` slots while the block
    runs, waiting as ``queued`` for one; the entry reads ``running`` once it
    has one.

    Raises:
        _Cancelled: The download was cancelled while it waited.
    """
    while not _slots.acquire(timeout=_QUEUE_POLL_SECONDS):
        if cancel.is_set():
            raise _Cancelled()
    try:
        with _lock:
            if cancel.is_set():
                raise _Cancelled()
            entry["status"] = STATUS_RUNNING
        yield
    finally:
        _slots.release()


def _run(
    hub: HubDatabase,
    entry: dict,
    url: str,
    folder_id: int,
    relpath: str,
    on_arrived: Optional[Callable[[int], None]],
    expected: Optional[str] = None,
    live: bool = False,
) -> None:
    try:
        _settle(hub, entry, url, folder_id, relpath, on_arrived, expected, live)
    finally:
        entry["_done"].set()


def _settle(
    hub: HubDatabase,
    entry: dict,
    url: str,
    folder_id: int,
    relpath: str,
    on_arrived: Optional[Callable[[int], None]],
    expected: Optional[str],
    live: bool,
) -> None:
    """Fetch, check and publish the download, and end its entry however that
    goes; :func:`_run`'s body."""
    target = entry["_target"]
    partial = target + PARTIAL_SUFFIX
    cancel = entry["_cancel"]
    try:
        # The network slot is let go before publishing: a move waits on the
        # shelf's lock, not on another download's socket.
        with _slot(entry, cancel):
            digest = _fetch(url, partial, entry, cancel, live=live)
        # The file the list names, or nothing: another file under the same
        # link would land on the shelf and leave a loader that asks for the
        # listed digest still missing, with nothing said.
        if expected and digest != expected:
            raise DownloadStopped(
                "It arrived, but it is not the file the template list names "
                f"(its SHA-256 is {digest}, the list says {expected}), so it "
                "was removed again."
            )
        model_id = _publish(hub, partial, entry, folder_id, relpath, digest)
    except _Cancelled:
        discard_partial(partial)
        logger.info("Download of %s cancelled.", target)
        _end(entry, STATUS_CANCELLED)
        return
    except (DownloadStopped, TargetRefused) as exc:
        # Sentences the server composed (a TargetRefused names only the host
        # it refused), so the entry may show them.
        discard_partial(partial)
        logger.warning(
            "Download %s of %s from %s stopped: %s",
            entry["id"],
            target,
            entry["host"],
            exc,
        )
        _end(entry, STATUS_FAILED, str(exc))
        return
    except (OSError, requests.RequestException, MoveRefused, ValueError) as exc:
        # A library's text (a URL, a path, a header value): logged, not shown.
        discard_partial(partial)
        logger.warning(
            "Download %s of %s from %s failed.",
            entry["id"],
            target,
            entry["host"],
            exc_info=True,
        )
        # A link that went quiet is worth a retry; say so rather than send the
        # owner to the log.
        _end(
            entry, STATUS_FAILED, DOWNLOAD_STALLED if stalled(exc) else DOWNLOAD_FAILED
        )
        return
    except Exception:
        # A thread's last stop: anything else (the hub refusing the row) would
        # otherwise leave the entry reading "running" forever.
        discard_partial(partial)
        logger.exception("Download %s of %s failed unexpectedly.", entry["id"], target)
        _end(entry, STATUS_FAILED, DOWNLOAD_FAILED)
        return
    entry["model_id"] = model_id
    _end(entry, STATUS_DONE)
    logger.info("Downloaded %s onto the shelf as model %s.", target, model_id)
    if on_arrived is not None:
        try:
            on_arrived(model_id)
        except Exception as exc:  # A missed nudge must not fail a download.
            logger.warning(
                "Model %s arrived but its announcement failed: %s", model_id, exc
            )


def model_page(url: str) -> Optional[str]:
    """The Hugging Face page of the repository *url* downloads from, or None.

    ``https://huggingface.co/<org>/<repo>/resolve/...`` gives
    ``https://huggingface.co/<org>/<repo>``: where a gated model's licence is
    accepted. Only for that host, so the link handed to the browser is never
    one a template could point anywhere. Read as ``requests`` sends *url*
    (``bounded_http.wire_split``).
    """
    parsed = _wire_parts(url)
    if parsed is None or parsed.scheme != "https" or parsed.netloc not in LIVE_HOSTS:
        return None
    repo, sep, _rest = parsed.path.partition("/resolve/")
    return f"https://{parsed.netloc}{repo}" if sep and repo.count("/") >= 2 else None


def _wire_parts(url: str):
    """*url* split as ``requests`` sends it (``bounded_http.wire_split``), or
    None when ``requests`` would not send it at all. Every host decision on a
    link to fetch reads this, never ``urlparse``: the two disagree on a
    backslash in the host, and ``requests`` is what connects."""
    try:
        return wire_split(url)
    except (requests.RequestException, ValueError) as exc:
        logger.info("A link requests would not send was not trusted: %s", exc)
        return None


def _host_named(url: str) -> str:
    """The host *url* reaches, as ``requests`` sends it, for a sentence."""
    parsed = _wire_parts(url)
    return (parsed.hostname if parsed is not None else None) or "an unnamed host"


def _live_hop(url: str) -> bool:
    """Whether a live link may be asked at *url*: https on Hugging Face, read
    as ``requests`` sends it, with no port and no user name."""
    parsed = _wire_parts(url)
    if parsed is None:
        return False
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and parsed.netloc == host
        and any(
            host == domain or host.endswith("." + domain) for domain in LIVE_HOP_DOMAINS
        )
    )


@contextlib.contextmanager
def _cancellable(
    entry: dict, cancel: threading.Event, adapter: PinnedAdapter
) -> Iterator[None]:
    """Let :func:`cancel_download` stop the requests *adapter* makes for
    *entry* while the block runs.

    The adapter is put on the entry under the lock the cancel takes, so a
    cancel either finds it and shuts its sockets down, or is seen here first.
    Whatever a shut-down socket raises inside the block (requests' connection
    error, urllib3's cut-short body) is the cancel, and is raised as one.

    Raises:
        _Cancelled: *cancel* was set before or during the block.
    """
    with _lock:
        if cancel.is_set():
            raise _Cancelled()
        entry["_adapter"] = adapter
    try:
        yield
    except _Cancelled:
        raise
    except Exception as exc:
        if not cancel.is_set():
            raise
        logger.info(
            "Download %s of %s cancelled mid-request (%s: %s).",
            entry["id"],
            entry["_target"],
            type(exc).__name__,
            exc,
        )
        raise _Cancelled() from exc
    finally:
        with _lock:
            entry.pop("_adapter", None)


@contextlib.contextmanager
def _open(url: str, live: bool, adapter: PinnedAdapter) -> Iterator[requests.Response]:
    """GET *url* as a stream, its redirects followed by hand, at most
    :data:`_MAX_HOPS` of them; the answer is closed when the block ends.

    *url* is the server's own (the committed list, or a live link
    :func:`trusted_live_link` took). It and every hop a redirect names are
    checked before they are asked: https, no user name, and every address
    the host resolves to public (``bounded_http.vet_target`` and
    ``vet_redirect_target``), so a link cannot be bounced onto plain http or
    into this network; a *live* link's hops must also stay on Hugging Face
    (:func:`_live_hop`).

    Each request connects to the addresses its check resolved, each in turn
    until one answers (*adapter*, a ``bounded_http.PinnedAdapter`` the caller
    keeps so it can stop the request), never to a second lookup of the name
    that a short-lived DNS record could answer with a private address (DNS
    rebinding). The name is checked and pinned as ``requests`` sends it, a
    non-ASCII host in its ``xn--`` form. SNI, the certificate check and the
    ``Host`` header keep the name. Not ``bounded_request``: a model is
    gigabytes streamed for as long as it takes, past any one deadline, and
    :func:`_fetch` reads its headers before deciding to take the body.

    Raises:
        DownloadStopped: A live hop off Hugging Face, or too many redirects.
        TargetRefused: *url* failed ``vet_target``.
        RedirectRefused: A hop that failed ``vet_redirect_target``, or one
            named in bytes that are not UTF-8.
    """
    kwargs = {
        "stream": True,
        "timeout": _TIMEOUT_SECONDS,
        "allow_redirects": False,
        # Content-Length counts the bytes sent and requests decodes what it
        # is sent, so a compressed body would outgrow the space checked for it.
        "headers": {"User-Agent": "PixlStash", "Accept-Encoding": "identity"},
    }
    # Backstop: start_download only takes live links trusted_live_link passed.
    if live and not _live_hop(url):
        raise DownloadStopped(
            f"The link led to {_host_named(url)}, "
            "which is not Hugging Face, so it was not followed."
        )
    adapter.pin(url, vet_target(url))
    # Not requests.Session: it reads a redirect's whole body, uncapped.
    with HandRedirectSession() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        for _hop in range(_MAX_HOPS + 1):
            response = session.get(url, **kwargs)
            if not 300 <= response.status_code < 400:
                with response:
                    yield response
                return
            with response:
                # Its body unread. None for a 3xx other than 301, 302, 303,
                # 307 and 308, which vet_redirect_target refuses as no address.
                location = session.redirect_target(response)
            if _hop == _MAX_HOPS:
                break
            named = urljoin(url, (location or "").strip())
            if live and not _live_hop(named):
                raise DownloadStopped(
                    f"The link led to {_host_named(named)}"
                    ", which is not Hugging Face, so it was not followed."
                )
            url, addresses = vet_redirect_target(url, location)
            adapter.pin(url, addresses)
    raise DownloadStopped(f"The link redirected more than {_MAX_HOPS} times.")


def _fetch(
    url: str, partial: str, entry: dict, cancel: threading.Event, live: bool = False
) -> str:
    """Stream *url* into *partial* and return the SHA-256 of what was written.

    A response that brings fewer than :data:`_STALL_BYTES` in any
    :data:`_STALL_SECONDS` is given up as failed, and :func:`cancel_download`
    shuts the request's socket down, so neither a trickle nor a cancel holds
    the entry (and every retry of the file) as running.

    Raises:
        _Cancelled: The owner cancelled it.
        DownloadStopped: Among others, :data:`DOWNLOAD_ENDED_EARLY` for a body
            that ended short of its ``Content-Length``.
    """
    os.makedirs(os.path.dirname(partial), exist_ok=True)
    sha = hashlib.sha256()
    adapter = PinnedAdapter(stall_floor=(_STALL_BYTES, _STALL_SECONDS))
    with (
        _cancellable(entry, cancel, adapter),
        _open(url, live, adapter) as response,
    ):
        page = model_page(url)
        if page and response.headers.get("X-Error-Code") == "GatedRepo":
            # No Hugging Face token is sent, so a gated file is never served
            # here; the owner accepts its licence and downloads it themselves.
            # Said only of a Hugging Face link: another host's header falls
            # through to its plain error below.
            entry["page_url"] = page
            # Instruction first: the notice shows three lines. Where the
            # download would have put it, its ComfyUI subfolder included, so
            # the folder still says what kind of model it is.
            raise DownloadStopped(
                "It is gated on Hugging Face. Open its page, get access there, "
                f"and save the file into {entry['_save_into']}; "
                "PixlStash picks it up from there."
            )
        # The status, not requests' sentence, which carries the whole URL.
        if 400 <= response.status_code < 600:
            raise DownloadStopped(
                f"The link answered HTTP {response.status_code}, so nothing was "
                "downloaded."
            )
        # Content codings are case-insensitive (RFC 9110 section 8.4.1).
        encoding = response.headers.get("Content-Encoding") or "identity"
        if encoding.strip().lower() != "identity":
            raise DownloadStopped(
                "The link sent the file compressed, so its size cannot be checked."
            )
        if "text/html" in response.headers.get("Content-Type", ""):
            raise DownloadStopped(
                "The link answered with a web page, not a file. It may need "
                "you to sign in, or it is the model's page rather than its "
                "download link."
            )
        total = int(response.headers.get("Content-Length") or 0)
        if total <= 0:
            # Without a size there is no space check, and a stream that never
            # ends would fill the disk.
            raise DownloadStopped("The server did not say how large the file is.")
        entry["total"] = total
        try:
            require_space(os.path.dirname(partial), total)
        except MoveRefused as exc:
            # Its sentence names the subfolder the caller's name put the file
            # in; the registered folder and the size say the same to the owner.
            logger.warning(
                "Download %s: no room confirmed for %s bytes in %s.",
                entry["id"],
                total,
                os.path.dirname(partial),
                exc_info=True,
            )
            raise DownloadStopped(
                f"PixlStash could not confirm room for its {total / 1024**3:.2f} "
                f"GB in {entry['folder_path']}, so nothing was downloaded. The "
                "server log says why."
            ) from exc
        with open(partial, "wb") as out:
            for chunk in response.iter_content(_CHUNK_BYTES):
                if cancel.is_set():
                    raise _Cancelled()
                out.write(chunk)
                sha.update(chunk)
                entry["received"] += len(chunk)
                # A chunked body is not held to its Content-Length by urllib3,
                # and the space was checked for that length only.
                if entry["received"] > total:
                    raise DownloadStopped(
                        "The link sent more than the size it gave, so the "
                        "file was not kept."
                    )
            # A cancel's shutdown can read as the end of the body; it is the
            # cancel, and is said as one, before the length is judged.
            if cancel.is_set():
                raise _Cancelled()
            # urllib3 2.x raises a body cut short of its Content-Length itself
            # (`enforce_content_length`); this is the check where the bytes are
            # kept, so a stream that ends quietly is never published. The body
            # is identity here: any other coding was refused above.
            if entry["received"] != total:
                logger.warning(
                    "Download %s ended at %s of the %s bytes declared.",
                    entry["id"],
                    entry["received"],
                    total,
                )
                raise DownloadStopped(DOWNLOAD_ENDED_EARLY)
    return sha.hexdigest()


def _publish(
    hub: HubDatabase,
    partial: str,
    entry: dict,
    folder_id: int,
    relpath: str,
    digest: str,
) -> int:
    target = entry["_target"]
    cancel = entry["_cancel"]
    # Waited for rather than refused: the bytes are already here, and a move
    # running now is no reason to fetch gigabytes again.
    while not SHELF_IO_LOCK.acquire(timeout=1):
        if cancel.is_set():
            raise _Cancelled()
    try:
        # The last moment a cancel can win; after it `cancel_download` says no,
        # so it never answers "cancelled" for a file that then lands.
        with _lock:
            if cancel.is_set():
                raise _Cancelled()
            entry["_committed"] = True
        publish_no_clobber(partial, target)
        try:
            model_id = ModelFolderScanner(hub).register_file(
                folder_id, target, relpath, sha256=digest
            )
        except Exception:
            # The hub refused the row (the folder was forgotten meanwhile): a
            # file no row names would also refuse every retry as "already
            # there", so it goes too.
            discard_partial(target)
            raise
        if model_id is None:
            discard_partial(target)
            raise DownloadStopped(
                "It downloaded but does not read as a model file, so it was "
                "removed again. The server log says why."
            )
        return model_id
    finally:
        SHELF_IO_LOCK.release()
