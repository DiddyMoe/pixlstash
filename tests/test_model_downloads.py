"""Model downloads: where they land, what is refused, and how they end."""

import contextlib
import gzip
import hashlib
import importlib.util
import ipaddress
import json
import logging
import os
import socket
import sqlite3
import ssl
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
import requests
import urllib3.util.connection

from pixlstash.hub.db import HubDatabase
from pixlstash.routes.workflows import RunGroup, _mark_live_downloads
from pixlstash.services import comfyui_templates, model_downloads
from pixlstash.services.model_downloads import (
    ALREADY_DOWNLOADING,
    DOWNLOAD_ENDED_EARLY,
    DOWNLOAD_FAILED,
    DOWNLOAD_STALLED,
    FOLDER_BEING_MOVED,
    MAX_CONCURRENT_DOWNLOADS,
    OUTSIDE_FOLDER,
    START_FAILED,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STILL_STOPPING,
    UNKNOWN_SOURCE,
    DownloadRefused,
    DownloadStopped,
    FolderBusy,
    cancel_download,
    digest_source,
    download_source,
    live_download_source,
    live_downloads,
    release_folder,
    reserve_folder,
    start_download,
    trusted_live_link,
)
from pixlstash.services.model_folder_scanner import (
    SHELF_MODEL_SUFFIXES,
    ModelFolderScanner,
)
from pixlstash.services.model_mover import PARTIAL_SUFFIX, SHELF_IO_LOCK, MoveRefused
from pixlstash.services.workflow_run_service import (
    MISSING_MODELS,
    MODELS_DOWNLOADING,
    blocks_batch,
    bypass_missing_loras,
    judge,
    missing_digest_models,
)
from pixlstash.utils import bounded_http
from pixlstash.utils.bounded_http import PinnedAdapter, RedirectRefused, TargetRefused
from tests.network_vectors import LAN_IPV4, PRIVATE_10_IPV4
from tests.test_bounded_http import (
    _Peer,
    _rebinding_every_name,
    _tls_files,
    allow_local_link,
)
from tests.test_downloads_intake import _register, write_adapter

# A subfolder a request may put in front of a name the list matches: no
# refusal or failure the server answers may repeat it.
LEAK = "example-leak-marker"


@contextlib.contextmanager
def serve_bytes(
    body,
    content_type="application/octet-stream",
    gate=None,
    length=None,
    encoding=None,
    asked=None,
    status=200,
    headers=None,
):
    """Serve *body* at a loopback URL; *gate*, when set, holds the second half.

    *length* is the Content-Length claimed, the body's own unless given;
    ``-1`` sends none. *encoding* is a Content-Encoding to claim, and *asked*,
    a list, collects each request's headers. *status* and *headers* are the
    answer's status and any further headers.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if asked is not None:
                asked.append(dict(self.headers))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            if encoding:
                self.send_header("Content-Encoding", encoding)
            if length != -1:
                self.send_header(
                    "Content-Length", str(len(body) if length is None else length)
                )
            self.end_headers()
            half = len(body) // 2
            self.wfile.write(body[:half])
            self.wfile.flush()
            if gate is not None:
                gate.wait(10)
            try:
                self.wfile.write(body[half:])
            except OSError as exc:
                # The client hung up on a cancel: expected here.
                print(f"serve_bytes: client went away ({exc})")

        def log_message(self, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/model.safetensors"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def no_downloads_left():
    """The registry is the process's: start each test with it empty."""
    model_downloads._downloads.clear()
    yield
    model_downloads._downloads.clear()


@pytest.fixture
def linked(monkeypatch):
    """Point every file name at *url*, in the ComfyUI folder *directory*.

    A plain-http *url* is one of this file's loopback servers, let through
    the first request's public-https check by the one test allowance,
    ``allow_local_link``; any other address is still checked. An https *url*
    is checked as it would be for real.
    """

    def link(url, directory="loras", sha256=None):
        monkeypatch.setattr(
            model_downloads,
            "download_source",
            lambda file: {"url": url, "directory": directory, "sha256": sha256},
        )
        if urlsplit(url).scheme == "http":
            allow_local_link(monkeypatch, model_downloads, url)

    return link


@pytest.fixture
def hub(tmp_path):
    database = HubDatabase(str(tmp_path / "hub.db"))
    yield database
    database.close()


@pytest.fixture
def target(hub, tmp_path):
    """A default folder and a marked download folder with a `loras` subfolder."""
    default = tmp_path / "default"
    chosen = tmp_path / "chosen"
    for path in (default, chosen, chosen / "loras"):
        path.mkdir()
    _register(hub, default, "user", is_default=1)
    chosen_id = _register(hub, chosen, "user")
    with hub.transaction() as conn:
        conn.execute(
            "UPDATE model_folder SET is_download_target = 1 WHERE id = ?",
            (chosen_id,),
        )
    return chosen, chosen_id


@pytest.fixture
def adapter_bytes(tmp_path):
    source = tmp_path / "src.safetensors"
    write_adapter(source)
    return source.read_bytes()


def _wait(download_id, timeout=10):
    """The entry once its download has ended and its thread is done: a
    cancel reads ``cancelled`` at once, before the thread has discarded the
    partial file."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Read first: once the thread is done the status no longer moves.
        done = model_downloads._downloads[download_id]["_done"].is_set()
        entry = next(d for d in live_downloads() if d["id"] == download_id)
        if done and entry["status"] not in model_downloads.ACTIVE_STATUSES:
            return entry
        time.sleep(0.05)
    raise AssertionError(f"download {download_id} never ended")


def test_the_template_list_says_which_files_can_be_downloaded():
    # A file the committed list links, spelled as a loader might spell it.
    found = download_source("SDXL\\HUNYUAN_3D_V2.1.safetensors")
    assert found["url"].startswith("https://")
    assert found["directory"] == "checkpoints"
    assert download_source("my_own_merge_v3.safetensors") is None
    assert download_source("") is None


def test_a_download_lands_in_the_marked_folder_and_is_registered(
    hub, target, adapter_bytes, linked
):
    chosen, chosen_id = target
    arrived = []
    with serve_bytes(adapter_bytes) as url:
        linked(url)
        entry = start_download(
            hub, "SDXL\\Style.safetensors", on_arrived=arrived.append
        )
        done = _wait(entry["id"])
    # What a workflow row or a set slot offers it under while it is on its way.
    assert (entry["directory"], entry["slot"], entry["sha256"]) == (
        "loras",
        "lora",
        None,
    )
    # By its own name, never the loader's subfolder it was asked under: what
    # `running_download` and the client's `runningDownload` compare a name to.
    assert entry["file"] == done["file"] == "Style.safetensors"
    assert done["status"] == STATUS_DONE, done
    assert (done["received"], done["total"], done["percent"]) == (
        len(adapter_bytes),
        len(adapter_bytes),
        100.0,
    )
    path = chosen / "loras" / "SDXL" / "Style.safetensors"
    assert path.read_bytes() == adapter_bytes
    assert not os.path.exists(str(path) + PARTIAL_SUFFIX)
    row = hub.fetchone(
        "SELECT model_id FROM model_file WHERE model_folder_id = ? AND relpath = ?",
        (chosen_id, "loras/SDXL/Style.safetensors"),
    )
    assert row is not None and arrived == [row["model_id"]] == [done["model_id"]]


def test_a_download_that_is_not_the_listed_file_is_refused(
    hub, target, adapter_bytes, linked
):
    """Wrong if a body whose digest is not the one the list records lands on
    the shelf: a loader asking for the listed digest would stay missing."""
    chosen, chosen_id = target
    with serve_bytes(adapter_bytes) as url:
        linked(url, sha256="0" * 64)
        refused = _wait(start_download(hub, "other.safetensors")["id"])
        linked(url, sha256=hashlib.sha256(adapter_bytes).hexdigest())
        kept = _wait(start_download(hub, "listed.safetensors")["id"])
    assert refused["status"] == STATUS_FAILED, refused
    assert "not the file the template list names" in refused["error"]
    assert not (chosen / "loras" / "other.safetensors").exists()
    assert not (chosen / "loras" / ("other.safetensors" + PARTIAL_SUFFIX)).exists()
    assert kept["status"] == STATUS_DONE, kept
    assert (chosen / "loras" / "listed.safetensors").read_bytes() == adapter_bytes


def test_without_the_comfyui_subfolder_the_file_goes_at_the_top(
    hub, target, adapter_bytes, linked
):
    chosen, _ = target
    with serve_bytes(adapter_bytes) as url:
        linked(url, "checkpoints")
        done = _wait(start_download(hub, "flat.safetensors")["id"])
    assert done["status"] == STATUS_DONE, done
    assert (chosen / "flat.safetensors").is_file()


@pytest.mark.parametrize(
    "directory, subfolder",
    [("vae", "VAE"), ("text_encoders", "TextEncoders"), ("unet", "DiffusionModels")],
)
def test_a_launcher_spelled_subfolder_for_the_role_takes_the_file(
    hub, target, adapter_bytes, linked, directory, subfolder
):
    """Stability Matrix spells ComfyUI's folders its own way. The folder is
    what tells the shelf a VAE of a layout its tensors do not show, so the
    file goes into it rather than to the top of the download folder."""
    chosen, _ = target
    (chosen / "ApproxVAE").mkdir()
    (chosen / subfolder).mkdir()
    with serve_bytes(adapter_bytes) as url:
        linked(url, directory)
        done = _wait(start_download(hub, "component.safetensors")["id"])
        # A loader naming the folder itself goes there too, not into a new
        # folder of the list's spelling beside it.
        named = _wait(start_download(hub, f"{directory}/named.safetensors")["id"])
    assert done["status"] == STATUS_DONE, done
    assert named["status"] == STATUS_DONE, named
    assert (chosen / subfolder / "component.safetensors").is_file()
    assert (chosen / subfolder / "named.safetensors").is_file()
    # By name: on a case-insensitive disk (Windows, macOS) ``chosen / "vae"``
    # is the ``VAE`` folder itself, so exists() is true whatever was made.
    assert directory not in os.listdir(chosen)


def test_with_no_marked_folder_the_default_folder_takes_it(
    hub, tmp_path, adapter_bytes, linked
):
    default = tmp_path / "default"
    default.mkdir()
    _register(hub, default, "user", is_default=1)
    with serve_bytes(adapter_bytes) as url:
        linked(url)
        done = _wait(start_download(hub, "a.safetensors")["id"])
    assert done["status"] == STATUS_DONE, done
    assert (default / "a.safetensors").is_file()


def test_a_file_the_list_does_not_link_is_refused(hub, target):
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, f"{LEAK}/my_own_merge_v3.safetensors")
    assert caught.value.status_code == 404
    assert str(caught.value) == UNKNOWN_SOURCE
    assert live_downloads() == []


@pytest.mark.parametrize(
    "file, status, says",
    [
        (
            f"{LEAK}/a.txt",
            400,
            "That file is not a model file the shelf keeps "
            f"({', '.join(SHELF_MODEL_SUFFIXES)}).",
        ),
        (f"../../{LEAK}.safetensors", 400, OUTSIDE_FOLDER),
        (f"{LEAK}-taken.safetensors", 409, "That file is already in {chosen}."),
    ],
)
def test_refused_before_a_byte_is_fetched(hub, target, linked, file, status, says):
    """Wrong if a refusal repeats the name the request sent: the server's own
    sentence, naming at most the registered folder."""
    chosen, _ = target
    (chosen / f"{LEAK}-taken.safetensors").write_bytes(b"x")
    linked("https://example.com/never-fetched", None)
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, file)
    assert caught.value.status_code == status
    assert str(caught.value) == says.format(chosen=chosen)
    assert LEAK not in str(caught.value)
    assert not any(d["file"] == os.path.basename(file) for d in live_downloads())


def test_the_same_file_twice_at_once_is_refused(hub, target, adapter_bytes, linked):
    gate = threading.Event()
    with serve_bytes(adapter_bytes, gate=gate) as url:
        linked(url)
        first = start_download(hub, f"{LEAK}/twice.safetensors")
        with pytest.raises(DownloadRefused) as caught:
            start_download(hub, f"{LEAK}/twice.safetensors")
        assert caught.value.status_code == 409
        assert str(caught.value) == ALREADY_DOWNLOADING
        gate.set()
        assert _wait(first["id"])["status"] == STATUS_DONE


class _OwnedLock:
    """A stand-in for the registry's lock that knows which thread holds it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = None

    def __enter__(self):
        self._lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *exc_info):
        self.owner = None
        self._lock.release()


def test_a_folder_reserved_by_a_symlink_is_checked_without_disk_calls_under_the_lock(
    hub, target, linked, tmp_path, monkeypatch
):
    """Wrong if a folder reserved by another spelling (a symlink to it) still
    takes a download, if a download writing inside it lets that reservation
    through, or if either check resolves a path while it holds the registry's
    lock: a folder on an unresponsive network share would then hold up every
    download request behind it."""
    chosen, _ = target
    alias = tmp_path / "alias"
    alias.symlink_to(chosen, target_is_directory=True)
    lock = _OwnedLock()
    monkeypatch.setattr(model_downloads, "_lock", lock)
    real_realpath = os.path.realpath
    resolved_under_lock = []

    def spying_realpath(path, *args, **kwargs):
        if lock.owner == threading.get_ident():
            resolved_under_lock.append(os.fspath(path))
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", spying_realpath)

    token = reserve_folder(str(alias))
    try:
        linked("https://example.com/never-fetched", None)
        with pytest.raises(DownloadRefused) as caught:
            start_download(hub, "held.safetensors")
        assert (caught.value.status_code, str(caught.value)) == (
            409,
            FOLDER_BEING_MOVED,
        )
    finally:
        release_folder(token)

    writing = {
        "id": "dl-writing",
        "file": "inside.safetensors",
        "status": STATUS_RUNNING,
        "_target": real_realpath(chosen / "inside.safetensors"),
        "_done": threading.Event(),
    }
    model_downloads._downloads[writing["id"]] = writing
    with pytest.raises(FolderBusy):
        reserve_folder(str(alias))
    assert model_downloads._reserved == {}
    assert resolved_under_lock == []


def test_a_cancel_stops_it_and_leaves_nothing(hub, target, adapter_bytes, linked):
    chosen, chosen_id = target
    gate = threading.Event()
    with serve_bytes(adapter_bytes * 16384, gate=gate) as url:
        linked(url, None)
        entry = start_download(hub, "cancelled.safetensors")
        deadline = time.monotonic() + 5
        while not any(
            d["id"] == entry["id"] and d["received"] for d in live_downloads()
        ):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert cancel_download(entry["id"]) is True
        gate.set()
        done = _wait(entry["id"])
    assert done["status"] == STATUS_CANCELLED, done
    assert os.listdir(chosen) == ["loras"]
    assert (
        hub.fetchone("SELECT 1 FROM model_file WHERE model_folder_id = ?", (chosen_id,))
        is None
    )
    assert cancel_download(entry["id"]) is False


# A link that has begun to answer and then sends a byte every 50 ms: in its
# headers, or after 96 KiB of a body it says is 1 MiB.
_TRICKLE_HEADS = {
    "headers": b"HTTP/1.1 200 OK\r\nX-Slow: ",
    "body": (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
        b"Content-Length: %d\r\n\r\n" % (1 << 20)
    )
    + b"\0" * (96 * 1024),
}


def _trickling(head, release):
    """A ``_Peer`` handler that sends *head*, then a byte every 50 ms until
    *release* is set: a link that never falls silent and never finishes."""

    def handler(conn, _request):
        conn.sendall(head)
        while not release.wait(0.05):
            conn.sendall(b"x")

    return handler


def _under_way(peer, where):
    """Wait until the download is reading the trickle: its request asked and
    the head part sent, or a 64 KiB chunk of the body taken."""
    deadline = time.monotonic() + 5
    while True:
        if where == "headers" and peer.requests:
            time.sleep(0.2)
            return
        if where == "body" and any(
            d["received"] >= 64 * 1024 for d in live_downloads()
        ):
            return
        assert time.monotonic() < deadline, f"the download never read the {where}"
        time.sleep(0.02)


@pytest.mark.parametrize("where", ["headers", "body"])
def test_a_cancel_ends_a_link_that_trickles_at_once(hub, target, linked, where):
    """A byte every 50 ms never lets a receive time out, and a cancel was
    only looked at between chunks, so the entry read running, and a retry was
    refused as already downloading, until the link stopped on its own. The
    cancel shuts the socket down: the entry reads cancelled within moments,
    the partial file is gone, and the file can be asked for again at once.
    Wrong too if the body's first 64 KiB is not counted while the rest is
    still coming."""
    chosen, _ = target
    release = threading.Event()
    with _Peer(_trickling(_TRICKLE_HEADS[where], release)) as peer:
        try:
            linked(peer.url, None)
            entry = start_download(hub, "trickle.safetensors")
            _under_way(peer, where)
            assert cancel_download(entry["id"]) is True
            done = _wait(entry["id"], timeout=3)
            assert done["status"] == STATUS_CANCELLED, done
            assert os.listdir(chosen) == ["loras"]
            again = start_download(hub, "trickle.safetensors")
            assert cancel_download(again["id"]) is True
            assert _wait(again["id"], timeout=3)["status"] == STATUS_CANCELLED
        finally:
            release.set()


@pytest.mark.parametrize("where", ["headers", "body"])
def test_a_link_that_trickles_is_given_up_as_failed(
    hub, target, adapter_bytes, linked, monkeypatch, caplog, where
):
    """Nobody cancels it: fewer than ``_STALL_BYTES`` in a window, here 1 s,
    fails the download, the cause logged and the server's own sentence for a
    stall shown, which says to try again rather than send the owner to the log.
    The positive control: a link that keeps up lands under the same floor."""
    chosen, _ = target
    monkeypatch.setattr(model_downloads, "_STALL_SECONDS", 1.0)
    caplog.set_level(logging.WARNING, logger=model_downloads.logger.name)
    release = threading.Event()
    with _Peer(_trickling(_TRICKLE_HEADS[where], release)) as peer:
        try:
            linked(peer.url, None)
            done = _wait(start_download(hub, "slow.safetensors")["id"], timeout=5)
        finally:
            release.set()
    assert done["status"] == STATUS_FAILED, done
    assert done["error"] == DOWNLOAD_STALLED
    assert "fewer than 65536 bytes came in 1 s" in caplog.text, caplog.text
    assert os.listdir(chosen) == ["loras"]

    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        landed = _wait(start_download(hub, "slow.safetensors")["id"])
    assert landed["status"] == STATUS_DONE, landed


def test_a_body_cut_short_is_not_saved(hub, target, adapter_bytes, linked):
    chosen, _ = target
    with serve_bytes(adapter_bytes, length=len(adapter_bytes) + 100) as url:
        linked(url, None)
        done = _wait(start_download(hub, "short.safetensors")["id"])
    assert done["status"] == STATUS_FAILED, done
    assert os.listdir(chosen) == ["loras"]


def _ending_short(monkeypatch, then=None):
    """The body read whole but handed on only in its first half, then the
    stream ends without raising: a cut-off urllib3 did not notice. *then*,
    when given, runs after that half is taken and before the stream ends."""
    real = requests.Response.iter_content

    def iter_content(response, chunk_size=1, decode_unicode=False):
        whole = b"".join(real(response, chunk_size, decode_unicode))
        yield whole[: len(whole) // 2]
        if then is not None:
            then()

    monkeypatch.setattr(requests.Response, "iter_content", iter_content)


def test_a_body_that_ends_quietly_short_of_its_length_is_not_saved(
    hub, target, adapter_bytes, linked, monkeypatch
):
    """Wrong if half a model lands on the shelf because the stream simply
    stopped: the length it gave is checked where the bytes are kept, and the
    owner is told to try again. The positive control lands the same file whole."""
    chosen, chosen_id = target
    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        with monkeypatch.context() as short:
            _ending_short(short)
            done = _wait(start_download(hub, "half.safetensors")["id"])
        assert done["status"] == STATUS_FAILED, done
        assert done["error"] == DOWNLOAD_ENDED_EARLY
        assert done["received"] == len(adapter_bytes) // 2
        assert os.listdir(chosen) == ["loras"]
        assert (
            hub.fetchone(
                "SELECT 1 FROM model_file WHERE model_folder_id = ?", (chosen_id,)
            )
            is None
        )
        landed = _wait(start_download(hub, "half.safetensors")["id"])
    assert landed["status"] == STATUS_DONE, landed


def test_a_cancel_that_ends_the_stream_short_is_said_as_a_cancel(
    hub, target, adapter_bytes, linked, monkeypatch, caplog
):
    """A cancel's shutdown can read as the body simply ending. Wrong if that
    is judged as a body that ended early: the log would record a failed
    download the owner had cancelled."""
    chosen, _ = target
    caplog.set_level(logging.INFO, logger=model_downloads.logger.name)
    ids = []
    _ending_short(monkeypatch, then=lambda: cancel_download(ids[0]))
    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        ids.append(start_download(hub, "stopped.safetensors")["id"])
        done = _wait(ids[0])
    assert done["status"] == STATUS_CANCELLED, done
    assert os.listdir(chosen) == ["loras"]
    assert "stopped.safetensors cancelled." in caplog.text, caplog.text
    assert "bytes declared" not in caplog.text, caplog.text


def _statuses(ids):
    by_id = {d["id"]: d["status"] for d in live_downloads()}
    return [by_id[i] for i in ids]


def _until(condition, what, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"never {what}"
        time.sleep(0.02)


def test_downloads_past_the_limit_wait_queued_and_can_be_cancelled_there(
    hub, target, adapter_bytes, linked
):
    """Three downloads at once; the next ones read queued and ask nothing
    until a slot frees. Wrong if a fourth connects at once, if a queued one
    cannot be cancelled (it would start later regardless), or if one never
    leaves the queue once a slot is free."""
    assert MAX_CONCURRENT_DOWNLOADS == 3
    chosen, _ = target
    gate = threading.Event()
    asked = []
    with serve_bytes(adapter_bytes, gate=gate, asked=asked) as url:
        linked(url, None)
        try:
            running = [
                start_download(hub, f"slot{n}.safetensors")["id"] for n in range(3)
            ]
            _until(lambda: len(asked) == 3, "asked by all three")
            queued = start_download(hub, "waits.safetensors")["id"]
            later = start_download(hub, "later.safetensors")["id"]
            # Past two polls of the queue, and still nothing asked.
            time.sleep(2 * model_downloads._QUEUE_POLL_SECONDS + 0.2)
            assert _statuses(running) == [STATUS_RUNNING] * 3
            assert _statuses([queued, later]) == [STATUS_QUEUED] * 2
            assert len(asked) == 3

            assert cancel_download(queued) is True
            assert _statuses([queued]) == [STATUS_CANCELLED]
            assert _wait(queued, timeout=3)["status"] == STATUS_CANCELLED
        finally:
            gate.set()
        ended = [_wait(i)["status"] for i in [*running, later]]
    assert ended == [STATUS_DONE] * 4
    # The cancelled one never asked; the later one did once a slot was free.
    assert len(asked) == 4
    assert sorted(os.listdir(chosen)) == [
        "later.safetensors",
        "loras",
        "slot0.safetensors",
        "slot1.safetensors",
        "slot2.safetensors",
    ]


def test_a_cancel_during_the_connect_reads_cancelled_at_once_and_sends_nothing(
    hub, target, linked, monkeypatch
):
    """A connect in progress is not interrupted from another thread, so the
    entry reads cancelled at once and only the thread lingers, for at most
    the connect. Wrong if the entry reads running until the connect ends, if
    the file can be started again while that thread still runs, or if the
    socket it then opens carries a byte: the TLS handshake, which would start
    before any later check, or the request."""
    chosen, _ = target
    dialling = threading.Event()
    release = threading.Event()
    peers = []

    def connect(address, *args, **kwargs):
        dialling.set()
        assert release.wait(10), "the connect was never let go"
        ours, theirs = socket.socketpair()
        # Bounds a handshake that should never start.
        ours.settimeout(5)
        peers.append(theirs)
        return ours

    # An https link checked for real: its name stands for a public host.
    _rebinding_every_name(monkeypatch, 443, 443)
    monkeypatch.setattr(urllib3.util.connection, "create_connection", connect)
    linked("https://models.example/dialling.safetensors", None)
    try:
        entry = start_download(hub, "dialling.safetensors")
        assert dialling.wait(5), "the download never connected"
        assert cancel_download(entry["id"]) is True
        assert _statuses([entry["id"]]) == [STATUS_CANCELLED]
        with pytest.raises(DownloadRefused) as caught:
            start_download(hub, "dialling.safetensors")
        # Not "already being downloaded": the owner stopped it, and the
        # retry is told it is only stopping.
        assert str(caught.value) == STILL_STOPPING
        assert caught.value.status_code == 409
    finally:
        release.set()
    try:
        _until(lambda: peers, "returned from the connect")
        [peer] = peers
        peer.settimeout(5)
        # Closed with nothing written: the end of the stream, not a
        # ClientHello.
        assert peer.recv(4096) == b""
    finally:
        for peer in peers:
            peer.close()
    done = _wait(entry["id"])
    assert done["status"] == STATUS_CANCELLED, done
    assert os.listdir(chosen) == ["loras"]


def test_a_web_page_is_not_saved_as_a_model(hub, target, linked):
    chosen, _ = target
    with serve_bytes(b"<html>sign in</html>", content_type="text/html") as url:
        linked(url, None)
        done = _wait(start_download(hub, "page.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    assert "web page" in done["error"]
    assert os.listdir(chosen) == ["loras"]


def test_a_gated_header_from_another_host_is_its_plain_error(hub, target, linked):
    """Only a Hugging Face link is said to be gated on Hugging Face. Wrong if
    this loopback host's ``X-Error-Code: GatedRepo`` reads as one, or offers
    a page."""
    chosen, _ = target
    with serve_bytes(
        b"Access to model a/b is restricted.",
        content_type="text/plain",
        status=401,
        headers={"X-Error-Code": "GatedRepo"},
    ) as url:
        linked(url, "loras")
        done = _wait(start_download(hub, "gated.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    # The status in the server's words, not requests' sentence, which carries
    # the whole URL.
    assert done["error"] == "The link answered HTTP 401, so nothing was downloaded."
    assert url not in done["error"]
    assert done["page_url"] is None
    assert os.listdir(chosen) == ["loras"]


class _Answer:
    """A response with *status* and *headers*, for `_fetch` to read."""

    def __init__(self, status, headers):
        self.status_code = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "headers, page",
    [
        ({"X-Error-Code": "GatedRepo"}, "https://huggingface.co/org/repo"),
        # Any other 401 (a repository that is not there) is not a licence.
        ({}, None),
    ],
)
def test_a_gated_hugging_face_file_links_its_model_page(
    tmp_path, monkeypatch, headers, page
):
    """Wrong if a gated file's error is the bare 401, or does not name where
    the download would have put it (the entry's ``_save_into``)."""
    url = "https://huggingface.co/org/repo/resolve/main/x.safetensors"
    monkeypatch.setattr(
        model_downloads, "_open", lambda u, live, adapter: _Answer(401, headers)
    )
    entry = {"_save_into": str(tmp_path / "loras"), "page_url": None}
    with pytest.raises(DownloadStopped) as caught:
        model_downloads._fetch(url, str(tmp_path / "x.part"), entry, threading.Event())
    assert entry["page_url"] == page
    if page:
        assert str(caught.value).startswith("It is gated on Hugging Face.")
        assert f"into {tmp_path / 'loras'};" in str(caught.value)
    else:
        assert str(caught.value) == (
            "The link answered HTTP 401, so nothing was downloaded."
        )
    assert not (tmp_path / "x.part").exists()


@pytest.mark.parametrize(
    "file, into",
    [
        # The caller's subfolders are theirs: the folder named is the server's.
        (f"{LEAK}/x.safetensors", "loras"),
        # A name already under the ComfyUI folder, which the download folder
        # does not have yet: the download would make it, so it is named.
        (f"loras/{LEAK}/x.safetensors", "loras"),
    ],
)
def test_a_gated_file_names_the_servers_folder_not_the_callers_path(
    hub, tmp_path, linked, monkeypatch, file, into
):
    """Wrong if the gated sentence repeats a subfolder the request put in
    front of the name, or drops the ComfyUI subfolder the download uses."""
    folder = tmp_path / "plain"
    folder.mkdir()
    _register(hub, folder, "user", is_default=1)
    if not file.startswith("loras/"):
        (folder / "loras").mkdir()
    linked("https://huggingface.co/org/repo/resolve/main/x.safetensors", "loras")
    gated = _Answer(401, {"X-Error-Code": "GatedRepo"})
    monkeypatch.setattr(model_downloads, "_open", lambda u, live, adapter: gated)
    done = _wait(start_download(hub, file)["id"])
    assert done["status"] == STATUS_FAILED
    assert f"save the file into {folder / into};" in done["error"], done["error"]
    assert LEAK not in done["error"]
    assert done["page_url"] == "https://huggingface.co/org/repo"


@pytest.mark.parametrize(
    "url, page",
    [
        (
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/f.safetensors",
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8",
        ),
        (
            "https://huggingface.co/org/repo/resolve/main/sub/f.safetensors",
            "https://huggingface.co/org/repo",
        ),
        ("https://huggingface.co/resolve/main/f.safetensors", None),
        ("https://huggingface.co/org/repo/blob/main/f.safetensors", None),
        ("http://huggingface.co/org/repo/resolve/main/f.safetensors", None),
        ("https://cdn.hf.co/org/repo/resolve/main/f.safetensors", None),
        ("https://example.com/org/repo/resolve/main/f.safetensors", None),
    ],
)
def test_the_model_page_is_only_ever_a_hugging_face_repository(url, page):
    assert model_downloads.model_page(url) == page


def test_a_compressed_body_is_asked_against_and_refused(
    hub, target, adapter_bytes, linked
):
    """Content-Length would be the compressed size while the decoded bytes are
    what lands, so the space check would not hold. Wrong if the request lets
    the server compress, or if a server that compresses anyway gets saved."""
    chosen, _ = target
    asked = []
    body = gzip.compress(adapter_bytes)
    with serve_bytes(body, encoding="gzip", asked=asked) as url:
        linked(url, None)
        done = _wait(start_download(hub, "packed.safetensors")["id"])
    assert asked[0]["Accept-Encoding"] == "identity"
    assert done["status"] == STATUS_FAILED
    assert "compressed" in done["error"]
    assert os.listdir(chosen) == ["loras"]


@pytest.mark.parametrize("encoding", ["IDENTITY", "identity "])
def test_identity_spelled_any_way_is_not_a_compressed_body(
    hub, target, adapter_bytes, linked, encoding
):
    """Content codings are case-insensitive (RFC 9110 section 8.4.1) and a
    field value's trailing space is not part of it (section 5.5), which
    ``http.client`` leaves on. Wrong if such a file is refused as compressed."""
    chosen, _ = target
    with serve_bytes(adapter_bytes, encoding=encoding) as url:
        linked(url, None)
        done = _wait(start_download(hub, "plain.safetensors")["id"])
    assert done["status"] == STATUS_DONE, done
    assert (chosen / "plain.safetensors").is_file()


def test_bytes_that_are_not_a_model_are_removed_again(hub, target, linked):
    chosen, chosen_id = target
    with serve_bytes(b"not a safetensors header") as url:
        linked(url, None)
        done = _wait(start_download(hub, "junk.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    assert "does not read as a model" in done["error"]
    assert os.listdir(chosen) == ["loras"]


def test_an_ended_download_is_forgotten_after_a_while(
    hub, target, adapter_bytes, monkeypatch, linked
):
    with serve_bytes(adapter_bytes) as url:
        linked(url)
        entry = start_download(hub, "old.safetensors")
        _wait(entry["id"])
    monkeypatch.setattr(model_downloads, "FINISHED_KEEP_SECONDS", -1)
    assert all(d["id"] != entry["id"] for d in live_downloads())


def test_a_missing_model_the_list_links_is_marked_downloadable():
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "hunyuan_3d_v2.1.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "my_own_lora.safetensors"},
        },
        "3": {"class_type": "SaveImage", "inputs": {}},
    }
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["other.safetensors"], {}]}}
        },
        "LoraLoader": {
            "input": {"required": {"lora_name": [["other.safetensors"], {}]}}
        },
        "SaveImage": {"input": {"required": {}}},
    }
    reasons, _ = judge(graph, object_info, None)
    missing = next(r for r in reasons if r.code == MISSING_MODELS).detail["models"]
    assert missing == [
        {
            "file": "hunyuan_3d_v2.1.safetensors",
            "folder": "checkpoints",
            "downloadable": True,
        },
        {"file": "my_own_lora.safetensors", "folder": "loras"},
    ]


# The Qwen VAE's digest, as the template list records it for its link.
_QWEN_VAE = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
_ON_SHELF = "b" * 64
_UNLISTED = "c" * 64


def test_a_model_a_digest_loader_names_and_the_shelf_lacks_is_missing():
    """By the listed name and offered, or by its digest when the list has none.

    Wrong if a digest the shelf holds is reported, if an emptied adapter row
    counts, if a missing one is not reported at all (it would only fail inside
    ComfyUI), or if a caller that gives no shelf check sees a change.
    """
    graph = {
        "1": {"class_type": "PixlStashVAELoader", "inputs": {"vae_sha256": _QWEN_VAE}},
        "2": {
            "class_type": "PixlStashCLIPLoader",
            "inputs": {"clip_sha256": _ON_SHELF, "type": "qwen_image"},
        },
        "3": {
            "class_type": "PixlStashMultiAdapterLoader",
            "inputs": {"adapter_sha256": _UNLISTED.upper(), "adapter_sha256_2": ""},
        },
        "4": {"class_type": "SaveImage", "inputs": {}},
    }
    object_info = {
        cls: {"input": {"required": {}}}
        for cls in (
            "PixlStashVAELoader",
            "PixlStashCLIPLoader",
            "PixlStashMultiAdapterLoader",
            "SaveImage",
        )
    }
    reasons, _ = judge(
        graph, object_info, None, digest_present=lambda d: d == _ON_SHELF
    )
    missing = next(r for r in reasons if r.code == MISSING_MODELS).detail["models"]
    assert missing == [
        {
            "file": "qwen_image_vae.safetensors",
            "folder": "vae",
            "sha256": _QWEN_VAE,
            "downloadable": True,
        },
        {"file": _UNLISTED, "folder": "loras", "sha256": _UNLISTED},
    ]
    reasons, _ = judge(graph, object_info, None)
    assert all(r.code != MISSING_MODELS for r in reasons)


def test_every_digest_a_built_in_names_can_be_downloaded():
    """Each model the shipped workflows load by digest is a file the template
    list links under that digest, so a machine without it is offered it."""
    folder = os.path.join(
        os.path.dirname(model_downloads.__file__),
        "..",
        "data",
        "comfyui-workflows",
        "built-in",
    )
    seen = 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue  # a built-in's thumbnail
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            graph = json.load(handle)
        for model in missing_digest_models(graph, lambda _digest: False):
            seen += 1
            source = digest_source(model["sha256"])
            assert source is not None, (name, model)
            assert download_source(source["name"])["url"] == source["url"], name
    assert seen >= 5


def test_the_list_build_asks_every_link_and_keeps_a_digest_only_when_unreachable(
    monkeypatch,
):
    """Wrong if a link the old list holds is not asked again: a file
    re-uploaded under it would keep a digest every download is then refused
    against. Wrong too if one unreachable link loses its known digest."""
    spec = importlib.util.spec_from_file_location(
        "build_template_suggestions",
        os.path.join(
            os.path.dirname(__file__), "..", "scripts", "build_template_suggestions.py"
        ),
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    hf = "https://huggingface.co/"
    asked = []

    def sha256_of(url):
        asked.append(url)
        if url.endswith("down"):
            raise urllib.error.URLError("offline")
        return {hf + "moved": "1" * 64}.get(url)

    monkeypatch.setattr(build, "_sha256_of", sha256_of)

    def files(*pairs):
        return {"templates": [{"files": [dict(url=u, **d) for u, d in pairs]}]}

    document = files(
        (hf + "moved", {}), (hf + "down", {}), (hf + "moved", {}), (hf + "gated", {})
    )
    old = files(
        (hf + "moved", {"sha256": "0" * 64}),
        (hf + "down", {"sha256": "2" * 64}),
        (hf + "gated", {"sha256": "3" * 64}),
    )
    build.add_digests(document, old)
    assert [f.get("sha256") for f in document["templates"][0]["files"]] == [
        "1" * 64,
        "2" * 64,
        "1" * 64,
        None,
    ]
    assert sorted(asked) == [hf + "down", hf + "gated", hf + "moved"]


def test_a_bypassed_lora_the_list_links_is_marked_downloadable():
    """A missing LoRA is left out of the run, not refused, but still offered."""
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors",
                "model": ["1", 0],
            },
        },
        "3": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "my_own_lora.safetensors", "model": ["2", 0]},
        },
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }
    lora_spec = {
        "input": {
            "required": {
                "model": ["MODEL"],
                "lora_name": [["other.safetensors"], {}],
            }
        },
        "output": ["MODEL"],
    }
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {}}, "output": ["MODEL"]},
        "LoraLoader": lora_spec,
        "KSampler": {"input": {"required": {"model": ["MODEL"]}}, "output": []},
    }
    bypassed = bypass_missing_loras(graph, object_info)
    assert {entry["file"]: entry.get("downloadable") for entry in bypassed} == {
        "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors": True,
        "my_own_lora.safetensors": None,
    }
    assert graph["4"]["inputs"]["model"] == ["1", 0]


def _running(file, sha256=None, slot=None):
    """A download in flight, 43% in, as the registry holds one."""
    entry = {
        "id": f"dl-{file}",
        "file": file,
        "status": "running",
        "received": 43,
        "total": 100,
        "sha256": sha256,
        "slot": slot,
        "ended_at": None,
    }
    model_downloads._downloads[entry["id"]] = entry
    return entry


def test_a_model_still_downloading_refuses_the_run_until_it_lands():
    """Wrong if a model on its way is reported missing (and offered again for
    download), if a run naming it would go ahead, or if it stays refused once
    the download has ended. By name in any case, and by digest."""
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "SDXL/Hunyuan_3D_v2.1.safetensors"},
        },
        "2": {"class_type": "PixlStashVAELoader", "inputs": {"vae_sha256": _QWEN_VAE}},
        "3": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "my_own_lora.safetensors"},
        },
        "4": {"class_type": "SaveImage", "inputs": {}},
    }
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["other.safetensors"], {}]}}
        },
        "LoraLoader": {
            "input": {"required": {"lora_name": [["other.safetensors"], {}]}}
        },
        "PixlStashVAELoader": {"input": {"required": {}}},
        "SaveImage": {"input": {"required": {}}},
    }
    _running("hunyuan_3D_V2.1.safetensors", slot="checkpoint")
    # By digest, though the list files it under another kind's slot.
    _running("renamed_vae.safetensors", sha256=_QWEN_VAE, slot="other")
    reasons, _ = judge(graph, object_info, None, digest_present=lambda d: False)
    by_code = {r.code: r.detail["models"] for r in reasons}
    assert by_code[MODELS_DOWNLOADING] == [
        {
            "file": "SDXL/Hunyuan_3D_v2.1.safetensors",
            "folder": "checkpoints",
            "percent": 43.0,
            "status": "running",
        },
        {
            "file": "qwen_image_vae.safetensors",
            "folder": "vae",
            "sha256": _QWEN_VAE,
            "percent": 43.0,
            "status": "running",
        },
    ]
    assert by_code[MISSING_MODELS] == [
        {"file": "my_own_lora.safetensors", "folder": "loras"}
    ]
    assert blocks_batch([r for r in reasons if r.code == MODELS_DOWNLOADING])

    for entry in model_downloads._downloads.values():
        entry["status"] = STATUS_DONE
    reasons, _ = judge(graph, object_info, None, digest_present=lambda d: False)
    assert MODELS_DOWNLOADING not in {r.code for r in reasons}
    assert (
        len(next(r for r in reasons if r.code == MISSING_MODELS).detail["models"]) == 3
    )


def test_a_queued_download_reaches_the_run_refusal_as_queued():
    """Wrong if a download waiting for a slot reads as one at 0% under
    `models_downloading`: the entry carries its download's `status`, so the
    notice can say it is queued rather than stuck."""
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "waiting.safetensors"},
        },
        "2": {"class_type": "SaveImage", "inputs": {}},
    }
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["other.safetensors"], {}]}}
        },
        "SaveImage": {"input": {"required": {}}},
    }
    entry = _running("waiting.safetensors", slot="checkpoint")
    entry.update(status=STATUS_QUEUED, received=0, total=0)
    reasons, _ = judge(graph, object_info, None)
    assert [r.detail["models"] for r in reasons if r.code == MODELS_DOWNLOADING] == [
        [
            {
                "file": "waiting.safetensors",
                "folder": "checkpoints",
                "percent": 0,
                "status": STATUS_QUEUED,
            }
        ]
    ]


def test_a_lora_still_downloading_is_not_bypassed():
    """Wrong if the run quietly goes without a LoRA the owner put in and is
    fetching: it keeps its loader, and the pre-flight refuses on it."""
    lora_spec = {
        "input": {
            "required": {"model": ["MODEL"], "lora_name": [["other.safetensors"], {}]}
        },
        "output": ["MODEL"],
    }
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {}}, "output": ["MODEL"]},
        "LoraLoader": lora_spec,
        "KSampler": {"input": {"required": {"model": ["MODEL"]}}, "output": []},
    }
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "Coming.safetensors", "model": ["1", 0]},
        },
        "3": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "gone.safetensors", "model": ["2", 0]},
        },
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }
    _running("coming.safetensors", slot="lora")
    bypassed = bypass_missing_loras(graph, object_info)
    assert [entry["file"] for entry in bypassed] == ["gone.safetensors"]
    assert graph["4"]["inputs"]["model"] == ["2", 0]
    reasons, _ = judge(graph, object_info, None)
    assert [r.detail["models"] for r in reasons if r.code == MODELS_DOWNLOADING] == [
        [
            {
                "file": "Coming.safetensors",
                "folder": "loras",
                "percent": 43.0,
                "status": "running",
            }
        ]
    ]


def test_a_lora_named_like_a_checkpoint_on_its_way_is_not_on_its_way():
    """Wrong if a checkpoint download keeps a LoRA of the same file name from
    being bypassed, or refuses its run as downloading: nothing fetches it."""
    lora_spec = {
        "input": {
            "required": {"model": ["MODEL"], "lora_name": [["other.safetensors"], {}]}
        },
        "output": ["MODEL"],
    }
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {}}, "output": ["MODEL"]},
        "LoraLoader": lora_spec,
        "KSampler": {"input": {"required": {"model": ["MODEL"]}}, "output": []},
    }
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "Shared.safetensors", "model": ["1", 0]},
        },
        "3": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
    }
    _running("shared.safetensors", slot="checkpoint")
    assert [e["file"] for e in bypass_missing_loras(graph, object_info)] == [
        "Shared.safetensors"
    ]
    assert (
        model_downloads.running_download("Shared.safetensors", folder="loras") is None
    )
    # The kind it is: a diffusion model's folder is a checkpoint's slot too.
    assert model_downloads.running_download(
        "Shared.safetensors", folder="diffusion_models"
    )


def test_a_digest_another_download_shares_a_name_with_is_still_missing():
    """Wrong if a VAE asked for by one digest reads as on its way while a file
    of the same name but another digest downloads: that download will not
    satisfy the loader, so it must stay missing and offered."""
    _running("qwen_image_vae.safetensors", sha256="e" * 64, slot="vae")
    graph = {
        "1": {"class_type": "PixlStashVAELoader", "inputs": {"vae_sha256": _QWEN_VAE}},
        "2": {"class_type": "SaveImage", "inputs": {}},
    }
    object_info = {
        cls: {"input": {"required": {}}} for cls in ("PixlStashVAELoader", "SaveImage")
    }
    reasons, _ = judge(graph, object_info, None, digest_present=lambda d: False)
    assert MODELS_DOWNLOADING not in {r.code for r in reasons}
    assert [
        m["sha256"]
        for m in next(r for r in reasons if r.code == MISSING_MODELS).detail["models"]
    ] == [_QWEN_VAE]
    # A download with no listed digest still matches by its name.
    model_downloads._downloads.clear()
    _running("qwen_image_vae.safetensors", slot="vae")
    reasons, _ = judge(graph, object_info, None, digest_present=lambda d: False)
    assert MODELS_DOWNLOADING in {r.code for r in reasons}


def test_a_listed_digest_is_kept_lowercase(hub, target, adapter_bytes, linked):
    """Wrong if a digest the list spells in capitals is stored as spelled:
    every digest it is compared with (a loader's, a set member's) is lowercase."""
    with serve_bytes(adapter_bytes) as url:
        linked(url, sha256="AB" * 32)
        entry = start_download(hub, "upper.safetensors")
        assert entry["sha256"] == "ab" * 32
        _wait(entry["id"])


def test_a_body_of_unknown_length_is_refused(hub, target, adapter_bytes, linked):
    chosen, _ = target
    with serve_bytes(adapter_bytes, length=-1) as url:
        linked(url, None)
        done = _wait(start_download(hub, "unsized.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    assert "how large" in done["error"]
    assert os.listdir(chosen) == ["loras"]


def test_a_row_the_hub_refuses_takes_the_file_with_it(
    hub, target, adapter_bytes, linked, monkeypatch, caplog
):
    chosen, _ = target
    cause = sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    def refuse(*args, **kwargs):
        raise cause

    monkeypatch.setattr(ModelFolderScanner, "register_file", refuse)
    caplog.set_level(logging.WARNING, logger=model_downloads.logger.name)
    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        done = _wait(start_download(hub, "refused.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    # The database's sentence goes to the log with its traceback, not to
    # every client polling the progress.
    assert done["error"] == DOWNLOAD_FAILED
    assert [
        r.exc_info[1]
        for r in caplog.records
        if r.name == model_downloads.logger.name and r.exc_info
    ] == [cause]
    # Nothing stranded at the real name, so a retry is not "already there".
    assert os.listdir(chosen) == ["loras"]


def test_a_library_failure_is_the_servers_sentence_and_the_logs_cause(
    hub, target, linked, monkeypatch, caplog
):
    """Wrong if requests' own sentence, which carries the URL and the name
    asked for, reaches the entry every client polling the progress reads."""
    linked("https://example.com/never-fetched", None)
    cause = requests.ConnectionError(
        "HTTPSConnectionPool(host='example.com', port=443): Max retries "
        f"exceeded with url: /{LEAK}/x.safetensors"
    )

    def unreachable(url, live, adapter):
        raise cause

    monkeypatch.setattr(model_downloads, "_open", unreachable)
    caplog.set_level(logging.WARNING, logger=model_downloads.logger.name)
    done = _wait(start_download(hub, f"{LEAK}/x.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    assert done["error"] == DOWNLOAD_FAILED
    assert [
        r.exc_info[1]
        for r in caplog.records
        if r.name == model_downloads.logger.name and r.exc_info
    ] == [cause]


def test_no_room_names_the_registered_folder_not_the_callers_path(
    hub, target, adapter_bytes, linked, monkeypatch, caplog
):
    """The space check's sentence names the directory the caller's subfolders
    made; the entry says the registered folder and the size instead."""
    chosen, _ = target
    cause = MoveRefused(f"There is not enough free space in {chosen / 'loras' / LEAK}.")

    def full(directory, size):
        raise cause

    monkeypatch.setattr(model_downloads, "require_space", full)
    caplog.set_level(logging.WARNING, logger=model_downloads.logger.name)
    with serve_bytes(adapter_bytes) as url:
        linked(url, "loras")
        done = _wait(start_download(hub, f"{LEAK}/x.safetensors")["id"])
    assert done["status"] == STATUS_FAILED
    assert done["error"] == (
        f"PixlStash could not confirm room for its "
        f"{len(adapter_bytes) / 1024**3:.2f} GB in {chosen}, so nothing was "
        "downloaded. The server log says why."
    )
    assert LEAK not in done["error"]
    assert [
        r.exc_info[1]
        for r in caplog.records
        if r.name == model_downloads.logger.name and r.exc_info
    ] == [cause]


def test_a_cancel_while_waiting_for_the_shelf_wins(hub, target, adapter_bytes, linked):
    chosen, _ = target
    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        with SHELF_IO_LOCK:
            entry = start_download(hub, "waiting.safetensors")
            deadline = time.monotonic() + 5
            while next(d for d in live_downloads() if d["id"] == entry["id"])[
                "received"
            ] < len(adapter_bytes):
                assert time.monotonic() < deadline
                time.sleep(0.02)
            assert cancel_download(entry["id"]) is True
            done = _wait(entry["id"])
    assert done["status"] == STATUS_CANCELLED
    assert os.listdir(chosen) == ["loras"]


def test_a_cancel_after_the_commit_says_no_and_the_file_lands(
    hub, target, adapter_bytes, linked, monkeypatch
):
    chosen, _ = target
    committed, release = threading.Event(), threading.Event()
    real = model_downloads.publish_no_clobber

    def held(partial, final):
        committed.set()
        release.wait(5)
        real(partial, final)

    monkeypatch.setattr(model_downloads, "publish_no_clobber", held)
    with serve_bytes(adapter_bytes) as url:
        linked(url, None)
        entry = start_download(hub, "landing.safetensors")
        assert committed.wait(5)
        assert cancel_download(entry["id"]) is False
        release.set()
        done = _wait(entry["id"])
    assert done["status"] == STATUS_DONE
    assert (chosen / "landing.safetensors").is_file()


def test_two_spellings_of_one_file_are_one_download(hub, target, adapter_bytes, linked):
    gate = threading.Event()
    with serve_bytes(adapter_bytes, gate=gate) as url:
        linked(url, None)
        first = start_download(hub, "Spelled.safetensors")
        with pytest.raises(DownloadRefused) as caught:
            start_download(hub, "spelled.SAFETENSORS")
        assert caught.value.status_code == 409
        gate.set()
        assert _wait(first["id"])["status"] == STATUS_DONE


@pytest.mark.parametrize("file", ["loras/style.safetensors", "Loras/style.safetensors"])
def test_a_name_already_under_its_folder_is_not_put_under_it_twice(
    hub, target, adapter_bytes, linked, file
):
    """Wrong if a workflow's own spelling of the folder nests it a second time."""
    chosen, _ = target
    with serve_bytes(adapter_bytes) as url:
        linked(url, "loras")
        done = _wait(start_download(hub, file)["id"])
    assert done["status"] == STATUS_DONE, done
    assert (chosen / "loras" / "style.safetensors").is_file()


def _resolving_to(monkeypatch, *addresses):
    """Every name resolves to *addresses*, an IP literal to itself: no lookup
    leaves the machine."""
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        with contextlib.suppress(ValueError):
            ipaddress.ip_address(host)
            return real(host, port, *args, **kwargs)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
            for address in addresses
        ]

    monkeypatch.setattr(bounded_http.socket, "getaddrinfo", getaddrinfo)


def _redirecting(monkeypatch, hops):
    """The download's session answering each URL in *hops* with a redirect
    to its value (``None`` ends the chain); returns the URLs asked, in order."""
    asked = []

    def get(session, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        asked.append(url)
        return _Hop(hops.get(url))

    monkeypatch.setattr(model_downloads.requests.Session, "get", get)
    return asked


@pytest.mark.parametrize(
    "live, refusal, says",
    [(False, RedirectRefused, "not https"), (True, ValueError, "not Hugging Face")],
)
def test_a_redirect_onto_plain_http_is_never_asked(monkeypatch, live, refusal, says):
    """Both kinds of link. Wrong if the plain http hop is asked at all."""
    _resolving_to(monkeypatch, "8.8.8.8")
    first = "https://huggingface.co/a/resolve/main/x.safetensors"
    asked = _redirecting(monkeypatch, {first: "http://cdn-lfs.huggingface.co/x"})
    with pytest.raises(refusal, match=says):
        with model_downloads._open(first, live=live, adapter=PinnedAdapter()):
            raise AssertionError("a refused hop was answered")
    assert asked == [first]


@pytest.mark.parametrize(
    "location",
    [
        "https://127.0.0.1/x",
        f"https://{LAN_IPV4}/x",
        f"https://{PRIVATE_10_IPV4}/x",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/x",
        "https://[::ffff:127.0.0.1]/x",
    ],
)
def test_a_list_link_redirected_into_this_network_is_never_asked(monkeypatch, location):
    _resolving_to(monkeypatch, "8.8.8.8")
    first = "https://example.com/model.safetensors"
    asked = _redirecting(monkeypatch, {first: location})
    with pytest.raises(RedirectRefused, match="private or local"):
        with model_downloads._open(first, live=False, adapter=PinnedAdapter()):
            raise AssertionError("a refused hop was answered")
    assert asked == [first]


def test_a_list_link_whose_redirect_resolves_into_this_network_is_never_asked(
    monkeypatch,
):
    """A public-looking name is resolved before it is asked: the link's own
    host is public, the host it redirects to also resolves into the LAN."""
    _resolving_to(monkeypatch, "8.8.8.8")
    public = bounded_http.socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host == "cdn.example.com":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in ("8.8.8.8", LAN_IPV4)
            ]
        return public(host, port, *args, **kwargs)

    monkeypatch.setattr(bounded_http.socket, "getaddrinfo", getaddrinfo)
    first = "https://example.com/model.safetensors"
    asked = _redirecting(monkeypatch, {first: "https://cdn.example.com/x"})
    with pytest.raises(RedirectRefused, match="cdn.example.com"):
        with model_downloads._open(first, live=False, adapter=PinnedAdapter()):
            raise AssertionError("a refused hop was answered")
    assert asked == [first]


def test_a_list_link_redirected_to_a_public_https_host_is_followed(monkeypatch):
    _resolving_to(monkeypatch, "8.8.8.8")
    first = "https://example.com/model.safetensors"
    asked = _redirecting(monkeypatch, {first: "https://cdn.example.com/x"})
    with model_downloads._open(first, live=False, adapter=PinnedAdapter()) as final:
        assert final.status_code == 200
    assert asked == [first, "https://cdn.example.com/x"]


def test_a_link_that_never_stops_redirecting_is_given_up(monkeypatch):
    """Given up at the eleventh redirect before its target is checked, so the
    name it points at is never looked up. Wrong if ``hop-eleven.example`` is
    resolved: a chain would spend a lookup past its limit."""
    _resolving_to(monkeypatch, "8.8.8.8")
    resolve = bounded_http.socket.getaddrinfo
    looked_up = []

    def recording(host, port, *args, **kwargs):
        looked_up.append(host)
        return resolve(host, port, *args, **kwargs)

    monkeypatch.setattr(bounded_http.socket, "getaddrinfo", recording)
    last = model_downloads._MAX_HOPS
    hops = {
        f"https://example.com/{n}": f"https://example.com/{n + 1}" for n in range(last)
    }
    hops[f"https://example.com/{last}"] = "https://hop-eleven.example/x"
    asked = _redirecting(monkeypatch, hops)
    with pytest.raises(ValueError, match="more than 10 times"):
        with model_downloads._open(
            "https://example.com/0", live=False, adapter=PinnedAdapter()
        ):
            raise AssertionError("a chain past the limit was answered")
    assert len(asked) == last + 1
    # The hops within the limit were each checked, which is a lookup.
    assert looked_up.count("example.com") == last + 1
    assert "hop-eleven.example" not in looked_up


def test_a_plain_http_link_past_the_test_allowance_is_refused_unasked(
    linked, adapter_bytes
):
    """Only the test's own server is let through: a plain-http link anywhere
    else, here a second loopback server, is refused before it is asked."""
    asked = []
    with (
        serve_bytes(adapter_bytes) as own,
        serve_bytes(adapter_bytes, asked=asked) as other,
    ):
        linked(own)
        with pytest.raises(TargetRefused, match="is not https"):
            with model_downloads._open(other, live=False, adapter=PinnedAdapter()):
                raise AssertionError("a plain-http link was answered")
    assert asked == []


def test_a_chunked_body_past_its_content_length_is_not_saved(hub, target, linked):
    """urllib3 holds a plain body to its Content-Length, not a chunked one;
    the space was checked for that length."""
    chosen, _ = target

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", "10")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(64):
                    self.wfile.write(b"400\r\n" + b"x" * 1024 + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            except OSError as exc:
                # The client hung up past the size it was given: expected.
                print(f"chunked server: client went away ({exc})")

        def log_message(self, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        linked(f"http://127.0.0.1:{httpd.server_port}/big.safetensors", None)
        done = _wait(start_download(hub, "big.safetensors")["id"])
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert done["status"] == STATUS_FAILED, done
    assert "more than the size it gave" in done["error"]
    assert os.listdir(chosen) == ["loras"]


def test_a_thread_that_will_not_start_leaves_no_entry_behind(
    hub, target, linked, monkeypatch, caplog
):
    cause = RuntimeError("can't start new thread")

    def refuse(self):
        raise cause

    linked("https://example.com/never-fetched", None)
    monkeypatch.setattr(threading.Thread, "start", refuse)
    caplog.set_level(logging.ERROR, logger=model_downloads.logger.name)
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, "unstarted.safetensors")
    assert caught.value.status_code == 503
    # A fixed sentence; the cause and the name are the log's.
    assert str(caught.value) == START_FAILED
    assert [
        r.exc_info[1]
        for r in caplog.records
        if r.name == model_downloads.logger.name and r.exc_info
    ] == [cause]
    assert live_downloads() == []
    monkeypatch.undo()
    linked("https://example.com/never-fetched", None)
    # Not "already being downloaded": nothing was left registered.
    (target[0] / "unstarted.safetensors").write_bytes(b"x")
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, "unstarted.safetensors")
    assert "already in" in str(caught.value)


# A ComfyUI address for the live-template tests; nothing is ever asked there.
COMFYUI = "http://127.0.0.1:1"
HF_LINK = "https://huggingface.co/someone/repo/resolve/main/loras/{}"


@pytest.fixture
def live_listing(monkeypatch):
    """The owner's ComfyUI serves one template linking *files* ``{name: url}``."""

    def serve(files):
        listing = comfyui_templates.Listing(
            [
                comfyui_templates.Template(
                    module="default",
                    name="example_edit",
                    source="builtin",
                    title="Example edit",
                    files=[
                        comfyui_templates.TemplateFile(name, "loras", url)
                        for name, url in files.items()
                    ],
                )
            ],
            read_at=0.0,
        )
        monkeypatch.setattr(
            comfyui_templates, "read_listing", lambda url, object_info=None: listing
        )

    return serve


@pytest.mark.parametrize(
    "name, url, trusted",
    [
        ("a.safetensors", HF_LINK.format("a.safetensors"), True),
        ("a.gguf", HF_LINK.format("a.gguf"), True),
        ("a.safetensors", "http://huggingface.co/x/a.safetensors", False),
        ("a.safetensors", "https://example.com/x/a.safetensors", False),
        ("a.safetensors", "https://huggingface.co.example.com/a.safetensors", False),
        ("a.safetensors", "https://user@huggingface.co/a.safetensors", False),
        ("a.safetensors", "https://huggingface.co:8443/a.safetensors", False),
        # Hugging Face to urlsplit; evil.example, or nothing, to requests.
        ("a.safetensors", "https://evil.example\\.huggingface.co/a.safetensors", False),
        ("a.safetensors", "https://hugging\tface.co/a.safetensors", False),
        ("a.safetensors", "https://hugging\nface.co/a.safetensors", False),
        # A pickle: loading it runs code, so a live link never fetches one.
        ("a.pth", HF_LINK.format("a.pth"), False),
        ("a.ckpt", HF_LINK.format("a.ckpt"), False),
        ("a.safetensors", None, False),
    ],
)
def test_a_live_template_link_is_taken_only_from_hugging_face(name, url, trusted):
    assert trusted_live_link(name, url) is trusted


def test_a_file_only_the_live_templates_link_is_found_there(live_listing):
    live_listing(
        {
            "Fresh_Edit_LoRA.safetensors": HF_LINK.format(
                "Fresh_Edit_LoRA.safetensors"
            ),
            "elsewhere.safetensors": "https://example.com/elsewhere.safetensors",
        }
    )
    assert download_source("fresh_edit_lora.safetensors") is None
    found = live_download_source("loras/fresh_edit_lora.safetensors", COMFYUI, {})
    assert found == {
        "url": HF_LINK.format("Fresh_Edit_LoRA.safetensors"),
        "directory": "loras",
        "sha256": None,
        "live": True,
    }
    assert live_download_source("elsewhere.safetensors", COMFYUI, {}) is None
    assert live_download_source("fresh_edit_lora.safetensors", None, {}) is None


def test_no_live_templates_is_no_live_link(monkeypatch):
    def unavailable(url, object_info=None):
        raise comfyui_templates.TemplatesUnavailable("nothing answers")

    monkeypatch.setattr(comfyui_templates, "read_listing", unavailable)
    assert live_download_source("fresh.safetensors", COMFYUI, {}) is None


def test_no_templates_are_asked_while_comfyui_is_down(monkeypatch):
    def never(url, object_info=None):
        raise AssertionError("asked a ComfyUI that is not answering")

    monkeypatch.setattr(comfyui_templates, "read_listing", never)
    assert live_download_source("fresh.safetensors", COMFYUI, None) is None


def test_a_download_falls_back_to_the_live_templates(hub, target, live_listing):
    chosen, _ = target
    live_listing({"fresh.safetensors": HF_LINK.format("fresh.safetensors")})
    # Already there: refused only after a link was found, so nothing is fetched.
    (chosen / "loras" / "fresh.safetensors").write_bytes(b"x")
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, "fresh.safetensors", comfyui_url=COMFYUI, object_info={})
    assert caught.value.status_code == 409
    with pytest.raises(DownloadRefused) as caught:
        start_download(hub, "fresh.safetensors")
    assert caught.value.status_code == 404
    assert live_downloads() == []


def test_the_preflight_marks_what_only_the_live_templates_link(live_listing):
    live_listing({"fresh.safetensors": HF_LINK.format("fresh.safetensors")})
    group = RunGroup(
        reasons=[
            {
                "code": MISSING_MODELS,
                "models": [
                    {"file": "fresh.safetensors", "folder": "loras"},
                    {"file": "own_merge.safetensors", "folder": "checkpoints"},
                    # Named by digest: a live template has none to match.
                    {"file": "fresh.safetensors", "sha256": "ab" * 32},
                ],
            }
        ],
        bypassed_loras=[
            {"file": "fresh.safetensors", "requested": False},
            {"file": "fresh.safetensors", "requested": True},
        ],
    )
    _mark_live_downloads([group], COMFYUI, object_info={})
    models = group.reasons[0]["models"]
    assert [model.get("downloadable") for model in models] == [True, None, None]
    assert [lora.get("downloadable") for lora in group.bypassed_loras] == [True, None]


def test_the_preflight_asks_no_templates_while_comfyui_is_down(monkeypatch):
    def never(url, object_info=None):
        raise AssertionError("asked a ComfyUI that is not answering")

    monkeypatch.setattr(comfyui_templates, "read_listing", never)
    group = RunGroup(bypassed_loras=[{"file": "fresh.safetensors"}])
    _mark_live_downloads([group], COMFYUI, object_info=None)
    assert "downloadable" not in group.bypassed_loras[0]


@pytest.mark.parametrize(
    "url, allowed",
    [
        ("https://huggingface.co/a/b/resolve/main/x.safetensors", True),
        ("https://us.aws.cdn.hf.co/xet-bridge-us/abc", True),
        ("https://cdn-lfs.huggingface.co/repos/abc", True),
        ("http://us.aws.cdn.hf.co/abc", False),
        ("https://example.com/abc", False),
        ("https://hf.co.example.com/abc", False),
        ("https://evilhf.co/abc", False),
        ("https://user@cdn.hf.co/abc", False),
        ("https://cdn.hf.co:8443/abc", False),
        ("https://hf.co/abc", True),
        ("https://huggingface.co/abc", True),
        # urlsplit reads each host below as Hugging Face's; requests sends
        # the first two to evil.example and will not send the next two.
        ("https://evil.example\\.huggingface.co/abc", False),
        ("https://evil.example\\x.hf.co/abc", False),
        ("https://hugging\tface.co/abc", False),
        ("https://cdn.hf\n.co/abc", False),
        ("https://huggingface.co@evil.example/abc", False),
        ("https://evil.example@huggingface.co/abc", False),
    ],
)
def test_a_live_link_may_only_hop_within_hugging_face(url, allowed):
    """Judged as requests sends the link, never as urlsplit reads it."""
    assert model_downloads._live_hop(url) is allowed


class _Hop(requests.Response):
    """A response that redirects to *location*, or ends with *location* None.

    A real ``requests.Response`` with no connection behind it, so the session
    reads its status and Location the way it reads one off the wire.
    """

    def __init__(self, location=None):
        super().__init__()
        self.status_code = 302 if location is not None else 200
        if location is not None:
            self.headers["Location"] = location

    def close(self):
        """There is no connection to release."""
        return None


def test_a_live_link_redirected_off_hugging_face_is_never_asked(monkeypatch):
    _resolving_to(monkeypatch, "8.8.8.8")
    hops = {
        "https://huggingface.co/a/resolve/main/x.safetensors": "/b/resolve/main/x.safetensors",
        "https://huggingface.co/b/resolve/main/x.safetensors": "https://example.com/x",
    }
    asked = _redirecting(monkeypatch, hops)
    with pytest.raises(ValueError, match="example.com"):
        with model_downloads._open(
            "https://huggingface.co/a/resolve/main/x.safetensors",
            live=True,
            adapter=PinnedAdapter(),
        ):
            raise AssertionError("a hop off Hugging Face was answered")
    assert asked == list(hops)


def test_a_live_link_redirected_to_the_hugging_face_cdn_is_followed(monkeypatch):
    _resolving_to(monkeypatch, "8.8.8.8")
    hops = {
        "https://huggingface.co/a/resolve/main/x.safetensors": "https://us.aws.cdn.hf.co/x",
    }
    asked = _redirecting(monkeypatch, hops)
    with model_downloads._open(
        "https://huggingface.co/a/resolve/main/x.safetensors",
        live=True,
        adapter=PinnedAdapter(),
    ) as final:
        assert final.status_code == 200
    assert asked == [*hops, "https://us.aws.cdn.hf.co/x"]


@pytest.mark.parametrize(
    "url, named",
    [
        ("https://example.com/a/resolve/main/x.safetensors", "example.com"),
        ("https://evil.example\\.huggingface.co/x.safetensors", "evil.example"),
    ],
)
def test_a_live_first_link_off_hugging_face_is_refused_with_nothing_asked(
    monkeypatch, url, named
):
    """The backstop: start_download takes a live link only once
    trusted_live_link has passed it, and _open refuses one that did not,
    before it is checked, looked up or asked."""
    asked = _redirecting(monkeypatch, {})
    checked = []
    monkeypatch.setattr(
        model_downloads, "vet_target", lambda link: checked.append(link) or ()
    )
    with pytest.raises(DownloadStopped) as caught:
        with model_downloads._open(url, live=True, adapter=PinnedAdapter()):
            raise AssertionError("a live link off Hugging Face was answered")
    assert str(caught.value) == (
        f"The link led to {named}, which is not Hugging Face, so it was not followed."
    )
    assert (checked, asked) == ([], [])


def test_a_live_source_reaches_the_fetch_as_live(
    hub, target, live_listing, monkeypatch
):
    live_listing({"fresh.safetensors": HF_LINK.format("fresh.safetensors")})
    seen = threading.Event()
    calls = []

    def fetch(url, partial, entry, cancel, live=False):
        calls.append((url, live))
        seen.set()
        raise ValueError("not fetched in a test")

    monkeypatch.setattr(model_downloads, "_fetch", fetch)
    entry = start_download(
        hub, "fresh.safetensors", comfyui_url=COMFYUI, object_info={}
    )
    assert seen.wait(5)
    _wait(entry["id"])
    assert calls == [(HF_LINK.format("fresh.safetensors"), True)]


# ---------------------------------------------------------------------------
# DNS rebinding: every request a download makes, its link and each redirect
# hop, connects to the address its check resolved (bounded_http.PinnedAdapter)
# ---------------------------------------------------------------------------

ORIGIN_NAME = "origin.test"
CDN_NAME = "cdn.test"
STANDIN_PUBLIC = "127.0.0.1"
STANDIN_PRIVATE = "127.0.0.2"


@pytest.fixture
def tls_names(tmp_path, monkeypatch):
    """``files(name)`` gives ``(chain, key)`` for a certificate naming *name*,
    from its own CA; every CA made is trusted through ``REQUESTS_CA_BUNDLE``.
    No environment proxy is taken, so a request goes where it is pinned."""
    bundle = tmp_path / "ca-bundle.pem"
    bundle.write_bytes(b"")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")

    def files(name):
        folder = tmp_path / name
        folder.mkdir()
        ca, chain, key = _tls_files(folder, name)
        bundle.write_bytes(bundle.read_bytes() + ca.read_bytes())
        return chain, key

    return files


def _tls_server(chain, key, answer):
    """A local https peer; *answer(tls, head)* writes the response.
    ``.names`` records each handshake's SNI name, ``.heads`` each request
    head read over TLS, ``.refused`` each handshake the client refused."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(chain, key)
    names = []
    context.sni_callback = lambda sock, name, ctx: names.append(name)

    def handler(conn, _):
        try:
            with context.wrap_socket(conn, server_side=True) as tls:
                head = b""
                while b"\r\n\r\n" not in head:
                    piece = tls.recv(65536)
                    if not piece:
                        return
                    head += piece
                peer.heads.append(head)
                answer(tls, head)
        except ssl.SSLError as exc:
            # The client refused the certificate, which a test may intend.
            peer.refused.append(exc)

    peer = _Peer(handler, read_request=False)
    peer.names = names
    peer.heads = []
    peer.refused = []
    return peer


def _redirect_to(location):
    def answer(tls, _head):
        tls.sendall(
            b"HTTP/1.1 302 Found\r\nLocation: %s\r\nContent-Length: 0\r\n\r\n"
            % location.encode()
        )

    return answer


def _body(data, length=None):
    def answer(tls, _head):
        tls.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
            b"Content-Length: %d\r\n\r\n%s"
            % (len(data) if length is None else length, data)
        )

    return answer


def _resolving(monkeypatch, ports):
    """Each name in *ports* resolves to this computer at its listed ports in
    turn (the last one repeated), as a short-TTL record changed between the
    check and the connect would; an IP literal resolves to itself. 127.0.0.1
    stands in for a public address and every other one stays what it is.
    Returns ``{name: [port handed out, ...]}``."""
    real = socket.getaddrinfo
    handed = {name: [] for name in ports}

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in ports:
            return real(host, port, *args, **kwargs)
        address, *rest = ports[host]
        listed = [address, *rest]
        given = handed[host]
        chosen = listed[min(len(given), len(listed) - 1)]
        given.append(chosen[1])
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", chosen)]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(
        bounded_http,
        "_is_public",
        lambda address: str(address) == STANDIN_PUBLIC,
    )
    return handed


def _read(url):
    with model_downloads._open(url, live=False, adapter=PinnedAdapter()) as response:
        response.raise_for_status()
        return response.content


def test_a_redirect_hop_that_rebinds_connects_to_the_address_checked(
    tls_names, monkeypatch
):
    """The hop's name is looked up once, by the check; a second answer naming
    another place is never asked. Host and SNI keep the name. Wrong if the
    hop's connection makes its own lookup."""
    chain, key = tls_names(CDN_NAME)
    origin_chain, origin_key = tls_names(ORIGIN_NAME)
    with (
        _tls_server(chain, key, _body(b"vetted")) as vetted,
        _tls_server(chain, key, _body(b"rebound")) as rebound,
    ):
        hop = f"https://{CDN_NAME}:{vetted.port}/x"
        with _tls_server(origin_chain, origin_key, _redirect_to(hop)) as origin:
            handed = _resolving(
                monkeypatch,
                {
                    ORIGIN_NAME: [(STANDIN_PUBLIC, origin.port)],
                    CDN_NAME: [
                        (STANDIN_PUBLIC, vetted.port),
                        (STANDIN_PUBLIC, rebound.port),
                    ],
                },
            )
            body = _read(f"https://{ORIGIN_NAME}:{origin.port}/model.safetensors")
    assert body == b"vetted"
    assert handed[CDN_NAME] == [vetted.port]
    assert rebound.asked == 0
    assert vetted.names == [CDN_NAME]
    assert vetted.refused == []
    assert vetted.heads[0].startswith(b"GET /x HTTP/1.1\r\n")
    assert f"host: {CDN_NAME}:{vetted.port}\r\n".encode() in vetted.heads[0].lower()


def test_the_link_itself_connects_to_the_address_checked(tls_names, monkeypatch):
    """The first request is checked and pinned like a hop. Wrong if the link
    is asked by a fresh lookup of its name."""
    chain, key = tls_names(ORIGIN_NAME)
    with (
        _tls_server(chain, key, _body(b"vetted")) as vetted,
        _tls_server(chain, key, _body(b"rebound")) as rebound,
    ):
        handed = _resolving(
            monkeypatch,
            {
                ORIGIN_NAME: [
                    (STANDIN_PUBLIC, vetted.port),
                    (STANDIN_PUBLIC, rebound.port),
                ]
            },
        )
        body = _read(f"https://{ORIGIN_NAME}:{vetted.port}/model.safetensors")
    assert body == b"vetted"
    assert handed[ORIGIN_NAME] == [vetted.port]
    assert rebound.asked == 0
    assert vetted.names == [ORIGIN_NAME]
    host = f"host: {ORIGIN_NAME}:{vetted.port}\r\n".encode()
    assert host in vetted.heads[0].lower()


@pytest.mark.parametrize(
    "typed, sent",
    [
        ("bücher.example", "xn--bcher-kva.example"),
        ("straße.example", "xn--strae-oqa.example"),
    ],
)
@pytest.mark.parametrize("via", ["link", "redirect"])
def test_a_non_ascii_host_is_pinned_as_the_name_sent(
    tls_names, monkeypatch, typed, sent, via
):
    """A download's own requests check, pin and connect for the ``xn--`` name
    ``requests`` sends, looked up once. Wrong if the connection is refused as
    unchecked (the pin filed under the typed name), or looks the name up
    afresh and reaches the rebound address."""
    chain, key = tls_names(sent)
    with (
        _tls_server(chain, key, _body(b"vetted")) as vetted,
        _tls_server(chain, key, _body(b"rebound")) as rebound,
    ):
        asked = _rebinding_every_name(monkeypatch, vetted.port, rebound.port)
        link = f"https://{typed}:{vetted.port}/model.safetensors"
        if via == "link":
            body = _read(link)
        else:

            def handler(conn, _):
                # UTF-8 bytes in the Location, decoded back by requests.
                conn.sendall(
                    b"HTTP/1.1 302 Found\r\nLocation: %s\r\n"
                    b"Content-Length: 0\r\n\r\n" % link.encode("utf-8")
                )

            with _Peer(handler) as origin:
                allow_local_link(monkeypatch, model_downloads, origin.url)
                body = _read(f"{origin.url}/model.safetensors")
    assert body == b"vetted"
    assert asked == [sent]
    assert rebound.asked == 0
    assert vetted.names == [sent]
    assert f"host: {sent}:{vetted.port}\r\n".encode() in vetted.heads[0].lower()


def test_a_redirects_body_is_never_read(tls_names, monkeypatch):
    """A redirect that declares a GiB body and sends a little of it is
    followed at once. Wrong if the download waits for the rest, here until
    its read timeout: a plain ``requests.Session`` reads a redirect's whole
    body, under no cap, before it hands the answer over."""
    monkeypatch.setattr(model_downloads, "_TIMEOUT_SECONDS", (5, 3))
    chain, key = tls_names(CDN_NAME)
    with _tls_server(chain, key, _body(b"vetted")) as vetted:
        _resolving(monkeypatch, {CDN_NAME: [(STANDIN_PUBLIC, vetted.port)]})
        hop = f"https://{CDN_NAME}:{vetted.port}/model.safetensors"

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s\r\n"
                b"Content-Length: %d\r\n\r\n" % (hop.encode(), 1 << 30)
            )
            conn.sendall(b"x" * (256 * 1024))
            # Held open until the client hangs up.
            conn.recv(1)

        with _Peer(handler) as origin:
            allow_local_link(monkeypatch, model_downloads, origin.url)
            body = _read(f"{origin.url}/model.safetensors")
    assert body == b"vetted"
    assert vetted.asked == 1


def test_a_redirect_whose_location_is_not_utf8_is_refused_naming_its_host(
    monkeypatch,
):
    """Location bytes that are not UTF-8 are a RedirectRefused naming the host
    that sent them, and the place is never checked. Wrong if they are read as
    ``http.client``'s Latin-1 text and checked, or escape as a bare
    ``UnicodeDecodeError``."""
    checked = []

    def check(current, location):
        checked.append(location)
        return location, ("127.0.0.1",)

    monkeypatch.setattr(model_downloads, "vet_redirect_target", check)

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 302 Found\r\nLocation: https://b\xfccher.example/x\r\n"
            b"Content-Length: 0\r\n\r\n"
        )

    with _Peer(handler) as origin:
        allow_local_link(monkeypatch, model_downloads, origin.url)
        with pytest.raises(RedirectRefused) as refused:
            _read(f"{origin.url}/model.safetensors")
    assert str(refused.value) == "127.0.0.1 sent a redirect whose address is not UTF-8."
    assert checked == []


@contextlib.contextmanager
def _listening_at(address):
    """A socket listening at *address* that counts the connections it takes."""
    server = socket.socket()
    server.bind((address, 0))
    server.listen(8)
    server.settimeout(0.1)
    taken = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            taken.append(conn)
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1], taken
    finally:
        stop.set()
        thread.join(5)
        server.close()


@pytest.mark.parametrize("hop", [True, False], ids=["hop", "link"])
def test_a_private_address_is_refused_with_nothing_sent(tls_names, monkeypatch, hop):
    """A name resolving to a private address is refused before anything
    connects to it, whether a hop names it or the link itself does. Wrong if
    the check is skipped: the listener there would be connected to."""
    chain, key = tls_names(ORIGIN_NAME)
    with _listening_at(STANDIN_PRIVATE) as (port, taken):
        private = f"https://{CDN_NAME}:{port}/x"
        with _tls_server(chain, key, _redirect_to(private)) as origin:
            _resolving(
                monkeypatch,
                {
                    ORIGIN_NAME: [(STANDIN_PUBLIC, origin.port)],
                    CDN_NAME: [(STANDIN_PRIVATE, port)],
                },
            )
            first = f"https://{ORIGIN_NAME}:{origin.port}/m" if hop else private
            refusal = RedirectRefused if hop else TargetRefused
            with pytest.raises(refusal, match="private or local"):
                _read(first)
        assert origin.asked == (1 if hop else 0)
        assert taken == []


@pytest.mark.parametrize("hop", [True, False], ids=["hop", "link"])
def test_a_pinned_request_still_refuses_a_certificate_for_another_name(
    tls_names, monkeypatch, hop
):
    """Connecting by address must not turn the certificate check off: a
    certificate that does not name the host asked is refused, and SNI still
    carries that name."""
    chain, key = tls_names("other.test")
    origin_chain, origin_key = tls_names(ORIGIN_NAME)
    with _tls_server(chain, key, _body(b"impostor")) as impostor:
        named = CDN_NAME if hop else ORIGIN_NAME
        target = f"https://{named}:{impostor.port}/x"
        with _tls_server(origin_chain, origin_key, _redirect_to(target)) as origin:
            _resolving(
                monkeypatch,
                {
                    ORIGIN_NAME: [
                        (STANDIN_PUBLIC, origin.port if hop else impostor.port)
                    ],
                    CDN_NAME: [(STANDIN_PUBLIC, impostor.port)],
                },
            )
            first = f"https://{ORIGIN_NAME}:{origin.port}/m" if hop else target
            with pytest.raises(requests.exceptions.SSLError):
                _read(first)
    assert impostor.names == [named]
    assert impostor.heads == []


def _pinned_chain(tls_names, monkeypatch, cdn_answer):
    """A list link on ``origin.test`` redirected to ``cdn.test``, which answers
    with *cdn_answer*; yields ``(link, origin, cdn)``."""
    chain, key = tls_names(CDN_NAME)
    origin_chain, origin_key = tls_names(ORIGIN_NAME)

    @contextlib.contextmanager
    def chained():
        with _tls_server(chain, key, cdn_answer) as cdn:
            hop = f"https://{CDN_NAME}:{cdn.port}/blob"
            with _tls_server(origin_chain, origin_key, _redirect_to(hop)) as origin:
                _resolving(
                    monkeypatch,
                    {
                        ORIGIN_NAME: [(STANDIN_PUBLIC, origin.port)],
                        CDN_NAME: [(STANDIN_PUBLIC, cdn.port)],
                    },
                )
                link = f"https://{ORIGIN_NAME}:{origin.port}/model.safetensors"
                yield link, origin, cdn

    return chained()


def test_a_public_redirect_chain_still_downloads_onto_the_shelf(
    hub, target, adapter_bytes, linked, tls_names, monkeypatch
):
    """The positive control: a link redirected to a CDN, both pinned, lands
    on the shelf with the bytes the CDN sent."""
    chosen, chosen_id = target
    with _pinned_chain(tls_names, monkeypatch, _body(adapter_bytes)) as chained:
        link, origin, cdn = chained
        linked(link)
        done = _wait(start_download(hub, "pinned.safetensors")["id"])
    assert done["status"] == STATUS_DONE, done
    assert (chosen / "loras" / "pinned.safetensors").read_bytes() == adapter_bytes
    assert (origin.asked, cdn.asked) == (1, 1)
    assert (origin.names, cdn.names) == ([ORIGIN_NAME], [CDN_NAME])
    assert hub.fetchone(
        "SELECT 1 FROM model_file WHERE model_folder_id = ?", (chosen_id,)
    )


def test_a_pinned_download_streams_its_body(
    hub, target, adapter_bytes, linked, tls_names, monkeypatch
):
    """Bytes are written as they arrive, not after the whole body is in:
    the download has received some while the CDN still holds the rest. The
    body is several read chunks long, as in the cancel test above."""
    chosen, _ = target
    body = adapter_bytes * 16384
    assert len(body) // 2 > model_downloads._CHUNK_BYTES
    gate = threading.Event()

    def held(tls, _head):
        half = len(body) // 2
        tls.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body[:half])
        )
        gate.wait(10)
        tls.sendall(body[half:])

    with _pinned_chain(tls_names, monkeypatch, held) as chained:
        link, _origin, _cdn = chained
        linked(link)
        entry = start_download(hub, "streamed.safetensors")
        deadline = time.monotonic() + 5
        try:
            while not any(
                d["id"] == entry["id"] and d["received"] for d in live_downloads()
            ):
                assert time.monotonic() < deadline, "nothing arrived before the end"
                time.sleep(0.02)
            assert not gate.is_set()
        finally:
            gate.set()
        done = _wait(entry["id"])
    assert done["status"] == STATUS_DONE, done
    assert (chosen / "loras" / "streamed.safetensors").read_bytes() == body


def test_a_pinned_download_past_the_size_it_gave_is_not_saved(
    hub, target, linked, tls_names, monkeypatch
):
    """The size cap holds through a pinned chain: a chunked body that runs
    past its Content-Length is refused and nothing is kept."""
    chosen, _ = target

    def oversized(tls, _head):
        tls.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
            b"Content-Length: 10\r\nTransfer-Encoding: chunked\r\n\r\n"
        )
        for _ in range(64):
            tls.sendall(b"400\r\n" + b"x" * 1024 + b"\r\n")
        tls.sendall(b"0\r\n\r\n")

    with _pinned_chain(tls_names, monkeypatch, oversized) as chained:
        link, _origin, _cdn = chained
        linked(link)
        done = _wait(start_download(hub, "big.safetensors")["id"])
    assert done["status"] == STATUS_FAILED, done
    assert "more than the size it gave" in done["error"]
    assert os.listdir(chosen) == ["loras"]
