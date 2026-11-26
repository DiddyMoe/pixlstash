"""The suite's connect guard, ``no_test_connects_to_comfyui_or_off_this_machine``.

Every test here goes through the guard tests/conftest.py installs for the
whole session, not a copy of it: a guard that stopped being installed, or was
installed only for some tests, fails here rather than passing quietly.

"Fast" is measured against ``CONNECT_TIMEOUT_S``. Under WSL's mirrored
networking an unguarded connect to the default ComfyUI port on 127.0.0.1 hangs
until that timeout, so a refusal well inside it is the guard answering. On a
machine where the OS refuses at once, the record is what tells the guard's
refusal from the OS's.
"""

import _socket
import errno
import http.server
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import pytest
import requests

from pixlstash.routes.comfyui import DEFAULT_COMFYUI_URL
from pixlstash.services import comfyui_templates

COMFYUI_PORT = urlsplit(DEFAULT_COMFYUI_URL).port
# What the OS answers a connect to a closed port with, so what CI answers for
# the default ComfyUI address: Winsock's own code on Windows.
REFUSED = errno.WSAECONNREFUSED if sys.platform == "win32" else errno.ECONNREFUSED
CONNECT_TIMEOUT_S = 3.0
FAST_S = 0.5
COMFYUI_REASON = "the default ComfyUI address"
OUTSIDE_REASON = "not a loopback address"
# RFC 5737 TEST-NET-1: never routed, so an unguarded connect cannot reach
# anything and simply waits out the timeout.
OUTSIDE_IP = "192.0.2.1"
# RFC 5737 TEST-NET-2, what the model-host cases resolve their names to, and
# how long a connect the guard lets through there is given to go nowhere.
STAND_IN_IP = "198.51.100.7"
LET_THROUGH_WAIT_S = 0.05


def _connect(family, address):
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(CONNECT_TIMEOUT_S)
        sock.connect(address)


def _resolving(monkeypatch, name, address):
    """*name* resolves to *address*, without DNS and beneath the guard, which
    sees the lookup as it would a real one."""
    real = _socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if host == name:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, port),
                )
            ]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(_socket, "getaddrinfo", resolve)


@pytest.mark.parametrize(
    ("family", "address", "shown"),
    [
        (socket.AF_INET, ("127.0.0.1", COMFYUI_PORT), f"127.0.0.1:{COMFYUI_PORT}"),
        (socket.AF_INET, ("localhost", COMFYUI_PORT), f"localhost:{COMFYUI_PORT}"),
        (socket.AF_INET, ("127.0.0.2", COMFYUI_PORT), f"127.0.0.2:{COMFYUI_PORT}"),
        (socket.AF_INET6, ("::1", COMFYUI_PORT, 0, 0), f"[::1]:{COMFYUI_PORT}"),
        (
            socket.AF_INET6,
            ("::ffff:127.0.0.1", COMFYUI_PORT, 0, 0),
            f"[::ffff:127.0.0.1]:{COMFYUI_PORT}",
        ),
    ],
    ids=["ipv4", "localhost", "other-loopback", "ipv6", "ipv4-mapped"],
)
def test_the_default_comfyui_address_is_refused_fast_and_recorded(
    blocked_connects, family, address, shown
):
    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError) as refused:
        _connect(family, address)
    assert time.monotonic() - started < FAST_S
    assert refused.value.errno == REFUSED

    [record] = blocked_connects()
    assert record.address == shown
    assert record.reason == COMFYUI_REASON
    assert record.thread == threading.current_thread().name
    assert record.nodeid.startswith("tests/test_connect_guard.py::")
    # No pixlstash/ frame is on this stack, so the caller is the test's own.
    assert record.caller.startswith("tests/test_connect_guard.py:")
    assert record.caller.endswith(" in _connect")


def test_connect_ex_answers_connection_refused_and_is_recorded(blocked_connects):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(CONNECT_TIMEOUT_S)
        started = time.monotonic()
        result = sock.connect_ex(("127.0.0.1", COMFYUI_PORT))
    assert time.monotonic() - started < FAST_S
    assert result == REFUSED
    [record] = blocked_connects()
    assert record.reason == COMFYUI_REASON


def test_the_template_listing_read_fails_fast_and_names_its_product_line(
    blocked_connects,
):
    """The read that cost three workflow tests 20-40 s on a WSL desk."""
    started = time.monotonic()
    with pytest.raises(comfyui_templates.TemplatesUnavailable):
        comfyui_templates.read_listing(f"http://127.0.0.1:{COMFYUI_PORT}", refresh=True)
    assert time.monotonic() - started < FAST_S

    records = blocked_connects()
    assert len(records) == 2  # templates/index.json and api/workflow_templates
    for record in records:
        assert record.reason == COMFYUI_REASON
        # Past the shared HTTP helper's own frames, to the seam a test stubs.
        assert record.caller.startswith("pixlstash/")
        assert " <- pixlstash/services/comfyui_templates.py:" in record.caller
        assert record.test_frame.startswith("tests/test_connect_guard.py:")


@pytest.mark.parametrize(
    "address",
    [(OUTSIDE_IP, 80), ("pixlstash.invalid", 80)],
    ids=["ip-literal", "hostname-not-resolved"],
)
def test_an_address_off_this_machine_is_refused_and_recorded(blocked_connects, address):
    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        _connect(socket.AF_INET, address)
    assert time.monotonic() - started < FAST_S
    [record] = blocked_connects()
    assert record.address == f"{address[0]}:{address[1]}"
    assert record.reason == OUTSIDE_REASON


@pytest.mark.parametrize(
    ("name", "let_through"),
    [
        ("huggingface.co", True),
        ("cas-bridge.xethub.hf.co", True),
        ("release-assets.githubusercontent.com", True),
        ("models.example", False),
        ("nothuggingface.co", False),
        ("huggingface.co.example", False),
        ("gist.github.com", False),
    ],
)
def test_only_an_address_a_model_host_resolved_to_is_let_through(
    monkeypatch, blocked_connects, name, let_through
):
    """CI's shards fetch a model their cache lacks from the hosts the cache is
    filled from, mid-test, so a connect to where one of those names resolved
    passes; any other name is off this machine like an IP literal.

    One stand-in address serves every case, so an address left noted after a
    let-through case would let the refused cases after it through: the
    ``blocked_connects`` fixture forgets it.
    """
    _resolving(monkeypatch, name, STAND_IN_IP)
    started = time.monotonic()
    try:
        socket.create_connection((name, 443), timeout=LET_THROUGH_WAIT_S).close()
    except ConnectionRefusedError as refused:
        outcome = refused
    except OSError as nowhere:  # TEST-NET-2 is never routed
        outcome = nowhere
    assert time.monotonic() - started < FAST_S
    if let_through:
        assert blocked_connects() == [], outcome
    else:
        assert isinstance(outcome, ConnectionRefusedError), outcome
        [record] = blocked_connects()
        assert (record.address, record.reason) == (f"{STAND_IN_IP}:443", OUTSIDE_REASON)


def test_a_request_off_this_machine_is_a_connection_error(blocked_connects):
    started = time.monotonic()
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://{OUTSIDE_IP}/", timeout=CONNECT_TIMEOUT_S)
    assert time.monotonic() - started < FAST_S
    assert [record.address for record in blocked_connects()] == [f"{OUTSIDE_IP}:80"]


def test_a_loopback_server_the_test_binds_is_reachable(blocked_connects):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(
            server.getsockname(), timeout=CONNECT_TIMEOUT_S
        ) as client:
            accepted, _ = server.accept()
            with accepted:
                client.sendall(b"ping")
                assert accepted.recv(4) == b"ping"
    assert blocked_connects() == []


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"hello"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Keep the request log out of the test output."""
        return None


def test_a_loopback_http_server_the_test_binds_answers_requests(blocked_connects):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hello)
    serving = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        name="connect-guard-http",
    )
    serving.start()
    try:
        host, port = server.server_address[:2]
        response = requests.get(f"http://{host}:{port}/", timeout=CONNECT_TIMEOUT_S)
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=CONNECT_TIMEOUT_S)
    assert response.status_code == 200
    assert response.text == "hello"
    assert blocked_connects() == []


def test_a_udp_connect_is_not_judged(blocked_connects):
    """``listeners.py`` finds the LAN address with a UDP connect to a public
    IP, which sends nothing; the guard leaves datagram sockets alone."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        result = sock.connect_ex((OUTSIDE_IP, 9))
    assert result != REFUSED
    assert blocked_connects() == []
