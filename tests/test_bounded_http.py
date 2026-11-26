"""The bounded request: a total deadline and a size cap on the whole answer.

Every peer here is a raw local socket, so it can misbehave in ways a real HTTP
server will not: dribble its status line or headers, or send without end.
"""

import contextlib
import gzip
import socket
import threading
import time

import pytest
import requests

from pixlstash.services import (
    comfyui_recipe_service,
    comfyui_service,
    comfyui_templates,
)
from pixlstash.utils.bounded_http import (
    ERROR_BODY_BYTES,
    ReadDeadlineExceeded,
    ResponseTooLarge,
    bounded_get,
    bounded_request,
)

NO_PROXY = {"http": None, "https": None, "all": None}


class _Peer:
    """A local HTTP peer that answers each connection with *handler*.

    *handler(conn, request)* gets the socket and the request (its head and
    its body to Content-Length, unless *read_request* is False); ``asked``
    counts the connections.
    """

    def __init__(self, handler, read_request=True):
        self._handler = handler
        self._read_request = read_request
        self.asked = 0
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(8)
        # accept() is not woken by close() on Linux, so it polls for the stop.
        self._server.settimeout(0.1)
        self.port = self._server.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(10)
        self._server.close()
        assert not self._thread.is_alive()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            conn.settimeout(10)
            self.asked += 1
            with conn:
                head = b""
                if self._read_request:
                    while b"\r\n\r\n" not in head:
                        piece = conn.recv(65536)
                        if not piece:
                            break
                        head += piece
                    # The body too: bytes left unread when the answer closes
                    # the connection reset it, which can lose the answer.
                    top, blank, _ = head.partition(b"\r\n\r\n")
                    if blank:
                        head = top + blank + _rest_of_body(conn, head)
                try:
                    self._handler(conn, head)
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up on this peer, which is the point.
                    continue


def _rest_of_body(conn, head):
    """The request body after *head*, read to its Content-Length."""
    top, _, body = head.partition(b"\r\n\r\n")
    length = 0
    for line in top.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value)
    while len(body) < length:
        piece = conn.recv(65536)
        if not piece:
            break
        body += piece
    return body


def _get(url, **kwargs):
    kwargs.setdefault("deadline_s", 1.0)
    kwargs.setdefault("max_bytes", 1 << 20)
    kwargs.setdefault("timeout", 5.0)
    kwargs.setdefault("proxies", NO_PROXY)
    return bounded_get(url, **kwargs)


def _trickle(conn, data, gap=0.1, hold=6.0):
    """Send *data* a byte per *gap*, for at most *hold* seconds."""
    started = time.monotonic()
    for byte in data:
        if time.monotonic() - started > hold:
            return
        conn.sendall(bytes([byte]))
        time.sleep(gap)


def _endless(conn, sent, total=256 << 20):
    """A chunked 200 sent as fast as it goes, until *total* bytes."""
    conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
    chunk = b"a" * 65536
    while sum(sent) < total:
        conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
        sent.append(len(chunk))


@contextlib.contextmanager
def _within(seconds):
    started = time.monotonic()
    yield
    took = time.monotonic() - started
    assert took < seconds, took


@pytest.mark.parametrize(
    "answer",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 60 + b"\r\nContent-Length: 2\r\n\r\n{}",
    ],
    ids=["status-line", "headers"],
)
def test_a_head_sent_a_byte_at_a_time_is_cut_at_the_deadline(answer):
    with _Peer(lambda conn, _: _trickle(conn, answer)) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_a_body_sent_a_byte_at_a_time_is_cut_at_the_deadline():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 60\r\n\r\n")
        _trickle(conn, b"a" * 60)

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_an_endless_body_is_cut_at_the_cap():
    sent = []
    with _Peer(lambda conn, _: _endless(conn, sent)) as peer, _within(5.0):
        with pytest.raises(ResponseTooLarge):
            _get(peer.url, deadline_s=30.0)
    assert sum(sent) < 64 << 20


def test_a_declared_length_over_the_cap_is_refused_before_the_body_is_read():
    release = threading.Event()

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % (1 << 30))
        release.wait(10)

    with _Peer(handler) as peer:
        try:
            with _within(2.0), pytest.raises(ResponseTooLarge, match="declared"):
                _get(peer.url, deadline_s=30.0)
        finally:
            release.set()


def test_an_ordinary_answer_is_read_whole():
    body = b'{"answer": "' + b"x" * 200000 + b'"}'

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body))
        conn.sendall(body)

    with _Peer(handler) as peer:
        answer = _get(peer.url, max_bytes=len(body))
    assert answer.status_code == 200
    assert answer.content == body
    assert answer.json() == {"answer": "x" * 200000}


def test_a_body_that_ends_short_of_its_declared_length_is_refused():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}")

    with _Peer(handler) as peer:
        with pytest.raises(requests.RequestException):
            _get(peer.url)


def test_a_compressed_body_is_refused_rather_than_handed_on_as_its_bytes():
    body = gzip.compress(b"{}")

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        )

    with _Peer(handler) as peer:
        with pytest.raises(requests.exceptions.ContentDecodingError):
            _get(peer.url)


def test_an_error_answer_keeps_only_the_start_of_its_body():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 500 Oops\r\nContent-Length: %d\r\n\r\n" % (1 << 20))
        conn.sendall(b"e" * (1 << 20))

    with _Peer(handler) as peer:
        answer = _get(peer.url)
    assert answer.status_code == 500
    assert len(answer.content) == ERROR_BODY_BYTES
    assert answer.truncated is True


def test_a_redirect_is_not_followed_and_its_endless_body_is_not_read():
    """``Session.send`` reads a redirect's whole body to note where it leads,
    even when told not to follow it; here the body is held to what an error
    answer keeps."""
    sent = []
    with _Peer(lambda conn, _: conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")) as target:

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s/\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n" % target.url.encode()
            )
            chunk = b"r" * 65536
            while sum(sent) < 256 << 20:
                conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                sent.append(len(chunk))

        with _Peer(handler) as peer, _within(5.0):
            answer = _get(peer.url, deadline_s=30.0)
    assert answer.status_code == 302
    assert answer.truncated is True
    assert target.asked == 0
    assert sum(sent) < 64 << 20


def test_a_request_body_the_server_will_not_take_is_cut_at_the_deadline():
    """An upload to a peer that accepts and never reads stops at the
    deadline, not at the per-operation timeout."""
    release = threading.Event()

    with _Peer(lambda conn, _: release.wait(10), read_request=False) as peer:
        try:
            with _within(4.0), pytest.raises(ReadDeadlineExceeded):
                bounded_request(
                    "POST",
                    peer.url,
                    data=b"x" * (256 << 20),
                    deadline_s=1.0,
                    max_bytes=1024,
                    timeout=30.0,
                    proxies=NO_PROXY,
                )
        finally:
            release.set()


# Each caller's own read, with its own deadline and cap made small here.
_CALLERS = {
    "ComfyUI templates": (
        comfyui_templates._get_json,
        comfyui_templates,
        "DEADLINE_S",
    ),
    "ComfyUI object_info": (
        comfyui_recipe_service._get_object_info_text,
        comfyui_recipe_service,
        "OBJECT_INFO_DEADLINE_S",
    ),
}


@pytest.mark.parametrize("caller", list(_CALLERS))
def test_a_caller_cuts_a_dribbled_answer_at_its_own_deadline(caller, monkeypatch):
    read, module, deadline = _CALLERS[caller]
    monkeypatch.setattr(module, deadline, 0.5)

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 60\r\n\r\n")
        _trickle(conn, b"a" * 60)

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises((ReadDeadlineExceeded, RuntimeError)) as raised:
            read(peer.url)
    # object_info reports any failure to reach ComfyUI as a RuntimeError.
    cause = raised.value
    while not isinstance(cause, ReadDeadlineExceeded) and cause.__cause__:
        cause = cause.__cause__
    assert isinstance(cause, ReadDeadlineExceeded)


def test_object_info_cuts_an_endless_answer_at_its_cap(monkeypatch):
    monkeypatch.setattr(comfyui_recipe_service, "OBJECT_INFO_MAX_BYTES", 64 * 1024)
    sent = []
    with _Peer(lambda conn, _: _endless(conn, sent, 64 << 20)) as peer, _within(5.0):
        with pytest.raises(RuntimeError) as raised:
            comfyui_recipe_service._get_object_info_text(peer.url)
    assert isinstance(raised.value.__cause__, ResponseTooLarge)
    assert sum(sent) < 16 << 20


def test_the_comfyui_probe_reads_a_dribbled_answer_as_no_answer():
    body = b'{"system": {"os": "posix"}}'

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body))
        _trickle(conn, body)

    with _Peer(handler) as peer, _within(2.0):
        result = comfyui_service.probe_comfyui(peer.url, timeout=0.5)
        assert result["reachable"] is False
        assert "did not answer within 0.5 seconds" in result["detail"]
