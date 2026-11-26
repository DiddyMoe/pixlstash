"""The bounded request: a total deadline and a size cap on the whole answer.

Every peer here is a raw local socket, so it can misbehave in ways a real HTTP
server will not: dribble its status line or headers, send compressed bytes
that never decode to anything, or send without end.
"""

import contextlib
import datetime
import gzip
import ipaddress
import socket
import ssl
import threading
import time
import tracemalloc
import zlib
from urllib.parse import urlsplit

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from urllib3 import exceptions as urllib3_errors

from tests.network_vectors import (
    GLOBAL_BUT_NOT_PUBLIC_IPV6,
    LAN_IPV4,
    PRIVATE_10_IPV4,
    PUBLIC_ADDRESS_FORMS,
    WRAPPED_PRIVATE_IPV6,
    ipv4_compatible,
    nat64,
)
from pixlstash.resources import comfyui_tenant
from pixlstash.services import comfyui_templates
from pixlstash.utils import bounded_http
from pixlstash.utils.bounded_http import (
    ERROR_BODY_BYTES,
    ReadDeadlineExceeded,
    RedirectRefused,
    ResponseTooLarge,
    TargetRefused,
    bounded_get,
    bounded_request,
    vet_redirect_target,
    vet_target,
)

NO_PROXY = {"http": None, "https": None, "all": None}
# A global address for the accepted direction: RFC 5737 documentation ranges
# are not global, so they are refused like any other non-public address.
PUBLIC_IPV4 = "8.8.8.8"


class _Peer:
    """A local HTTP peer that answers each connection with *handler*.

    *handler(conn, request)* gets the socket and the request (its head and
    its body to Content-Length, unless *read_request* is False); ``requests``
    records them, ``asked`` counts the connections.
    """

    def __init__(self, handler, read_request=True):
        self._handler = handler
        self._read_request = read_request
        self.requests = []
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
                self.requests.append(head)
                try:
                    self._handler(conn, head)
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up on this peer, which is the point.
                    continue


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


@contextlib.contextmanager
def _within(seconds):
    started = time.monotonic()
    yield
    took = time.monotonic() - started
    assert took < seconds, took


def allow_local_link(monkeypatch, module, url):
    """The one test allowance past the first link's check (``vet_target``).

    *url*'s own origin, a plain-http server of the test's on 127.0.0.1, is let
    through and connected to as named. Any other link is still checked for
    real, so the refusal production relies on is never loosened, and a
    redirect is checked as usual unless a test stands in for that too.

    Args:
        monkeypatch: The test's ``monkeypatch``; the allowance ends with it.
        module: Where the caller looks the check up: ``bounded_http`` for
            ``bounded_request``, ``model_downloads`` for a download (it
            imports the name).
        url: A link to the test's own server.

    Returns:
        The links let through, in order.
    """
    served = urlsplit(url)
    assert (served.scheme, served.hostname) == ("http", "127.0.0.1"), url
    allowed = []

    def vet(asked, proxies=None):
        if urlsplit(asked)[:2] != served[:2]:
            return vet_target(asked, proxies)
        allowed.append(asked)
        return ("127.0.0.1",)

    monkeypatch.setattr(module, "vet_target", vet)
    return allowed


def test_a_status_line_sent_a_byte_at_a_time_is_cut_at_the_deadline():
    def handler(conn, _):
        _trickle(conn, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_headers_sent_a_byte_at_a_time_are_cut_at_the_deadline():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
        _trickle(conn, b"a" * 60)
        conn.sendall(b"\r\nContent-Length: 2\r\n\r\n{}")

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_a_body_sent_a_byte_at_a_time_is_cut_at_the_deadline():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 60\r\n\r\n")
        _trickle(conn, b"a" * 60)

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_compressed_bytes_that_never_decode_to_anything_are_cut_at_the_deadline():
    """Empty deflate blocks: bytes keep arriving, nothing is ever decoded."""

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        member = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
        conn.sendall(b"%x\r\n%s\r\n" % (len(member), member))
        block = b"\x00\x00\x00\xff\xff"
        started = time.monotonic()
        while time.monotonic() - started < 6.0:
            conn.sendall(b"%x\r\n%s\r\n" % (len(block), block))
            time.sleep(0.1)

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            _get(peer.url)


def test_an_endless_body_is_cut_at_the_cap():
    sent = []

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        chunk = b"a" * 65536
        while sum(sent) < 256 << 20:
            conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            sent.append(len(chunk))

    with _Peer(handler) as peer, _within(5.0):
        with pytest.raises(ResponseTooLarge):
            _get(peer.url, deadline_s=30.0)
    assert sum(sent) < 64 << 20


def test_compressed_bytes_without_end_are_cut_at_the_cap_on_bytes_received():
    """Empty deflate blocks as fast as they go: nothing is ever decoded, so
    only the cap on bytes received stops them."""
    sent = []

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        member = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
        conn.sendall(b"%x\r\n%s\r\n" % (len(member), member))
        blocks = b"\x00\x00\x00\xff\xff" * 13000
        while sum(sent) < 256 << 20:
            conn.sendall(b"%x\r\n%s\r\n" % (len(blocks), blocks))
            sent.append(len(blocks))

    with _Peer(handler) as peer, _within(5.0):
        with pytest.raises(ResponseTooLarge):
            _get(peer.url, deadline_s=30.0)
    assert sum(sent) < 64 << 20


# What reading the bomb below may allocate at its peak. Measured on 3.12:
# about 2.3 MiB as written, about 142 MiB when each read is inflated whole
# (`decompress` without its max_length); 16 MiB leaves 4x either way.
_BOMB_PEAK_BYTES = 16 << 20


def test_a_small_compressed_body_that_inflates_past_the_cap_is_refused():
    """A few KiB that would decode to 64 MiB: refused at the cap, never
    inflated whole. Wrong if the peak allocation while reading it is
    anywhere near the 64 MiB: refused only after inflating the lot."""
    bomb = gzip.compress(b"\0" * (64 << 20), 9)
    assert len(bomb) < 1 << 20
    answer = (
        b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n"
        b"Content-Length: %d\r\n\r\n%s" % (len(bomb), bomb)
    )

    def handler(conn, _):
        conn.sendall(answer)

    with _Peer(handler) as peer:
        tracemalloc.start()
        try:
            with pytest.raises(ResponseTooLarge):
                _get(peer.url, deadline_s=10.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak < _BOMB_PEAK_BYTES, peak


@pytest.mark.parametrize(
    "encoding, encode",
    [
        ("gzip", lambda body: gzip.compress(body)),
        ("x-gzip", lambda body: gzip.compress(body)),
        ("deflate", lambda body: zlib.compress(body)),
        ("identity", lambda body: body),
    ],
)
def test_a_body_sent_compressed_anyway_is_read_decoded(encoding, encode):
    body = encode(b'{"answer": 42}')

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: %s\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (encoding.encode(), len(body), body)
        )

    with _Peer(handler) as peer:
        answer = _get(peer.url)
    assert answer.json() == {"answer": 42}
    # Identity is asked for, so a well-behaved server never compresses.
    assert b"accept-encoding: identity\r\n" in peer.requests[0].lower()


@pytest.mark.parametrize(
    "encoding, body",
    [
        ("gzip", gzip.compress(b"a" * 4096)[:-12]),
        ("gzip", gzip.compress(b"a") + b"trailing"),
        ("gzip, gzip", gzip.compress(gzip.compress(b"a"))),
        ("br", b"a"),
    ],
    ids=["cut-short", "trailing-bytes", "stacked", "unknown"],
)
def test_a_compressed_body_that_is_not_one_whole_body_is_refused(encoding, body):
    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: %s\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (encoding.encode(), len(body), body)
        )

    with _Peer(handler) as peer:
        with pytest.raises(requests.exceptions.ContentDecodingError):
            _get(peer.url)


def test_a_redirect_is_not_followed_by_default():
    with _Peer(lambda conn, _: conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")) as target:

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s/\r\n"
                b"Content-Length: 0\r\n\r\n" % target.url.encode()
            )

        with _Peer(handler) as peer:
            answer = _get(peer.url)
    assert answer.status_code == 302
    assert target.asked == 0


@pytest.mark.parametrize("redirects", [0, 1])
@pytest.mark.parametrize("encoding", ["latin-1", "utf-8"])
def test_a_redirect_location_is_read_as_utf8_and_refused_when_it_is_not(
    monkeypatch, redirects, encoding
):
    """A followed redirect's Location bytes are read as UTF-8, as ``requests``
    reads them, not as ``http.client``'s Latin-1 text: the hop is checked and
    asked under the UTF-8 name. Bytes that are not UTF-8 are a RedirectRefused
    naming the host, never a bare ``UnicodeDecodeError`` that a caller's
    ``except requests.RequestException`` misses, and the place is neither
    checked nor asked. Not followed, either one comes back as its status."""
    checked = []

    def check(current, location, proxies=None):
        checked.append(location)
        return location, ("127.0.0.1",)

    monkeypatch.setattr(bounded_http, "vet_redirect_target", check)

    def answer_ok(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    with _Peer(answer_ok) as target:
        location = f"{target.url}/bücher"

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s\r\n"
                b"Content-Length: 0\r\n\r\n" % location.encode(encoding)
            )

        with _Peer(handler) as origin:
            allow_local_link(monkeypatch, bounded_http, origin.url)
            if redirects and encoding == "latin-1":
                with pytest.raises(RedirectRefused) as refused:
                    _get(origin.url, redirects=redirects)
                assert str(refused.value) == (
                    "127.0.0.1 sent a redirect whose address is not UTF-8."
                )
                assert isinstance(refused.value.__cause__, UnicodeDecodeError)
                answer = None
            else:
                answer = _get(origin.url, redirects=redirects)
    assert origin.asked == 1
    if redirects and encoding == "utf-8":
        assert answer.content == b"ok"
        assert checked == [location]
        assert target.requests[0].startswith(b"GET /b%C3%BCcher HTTP/1.1\r\n")
    else:
        assert answer is None or answer.status_code == 302
        assert (checked, target.asked) == ([], 0)


@pytest.mark.parametrize("redirects", [0, 1])
def test_a_redirects_body_is_never_read_past_what_is_kept(monkeypatch, redirects):
    """A redirect that declares a GiB body and sends a little of it is
    followed at once, or, not followed, comes back with its first
    ``ERROR_BODY_BYTES``. Wrong if the request waits for the rest and runs
    into its deadline: a plain ``requests.Session`` reads a redirect's whole
    body, under no cap, before it hands the answer over."""
    monkeypatch.setattr(
        bounded_http,
        "vet_redirect_target",
        lambda current, location, proxies=None: (location, ("127.0.0.1",)),
    )

    def answer_ok(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    with _Peer(answer_ok) as target:

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s/x\r\n"
                b"Content-Length: %d\r\n\r\n" % (target.url.encode(), 1 << 30)
            )
            conn.sendall(b"x" * (4 * ERROR_BODY_BYTES))
            # Held open until the client hangs up.
            conn.recv(1)

        with _Peer(handler) as origin:
            allow_local_link(monkeypatch, bounded_http, origin.url)
            answer = _get(origin.url, redirects=redirects, deadline_s=3.0)
    if redirects:
        assert answer.content == b"ok"
        assert target.asked == 1
    else:
        assert answer.status_code == 302
        assert answer.content == b"x" * ERROR_BODY_BYTES
        assert target.asked == 0


def test_an_error_answer_keeps_only_the_start_of_its_body():
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 500 Oops\r\nContent-Length: %d\r\n\r\n" % (1 << 20))
        conn.sendall(b"e" * (1 << 20))

    with _Peer(handler) as peer:
        answer = _get(peer.url)
    assert answer.status_code == 500
    assert len(answer.content) == ERROR_BODY_BYTES
    assert answer.truncated is True
    assert answer.excerpt(10) == "'eeeeeeeeee' (cut short)"


def test_a_sink_takes_the_body_as_it_arrives():
    taken = []

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 200000\r\n\r\n")
        conn.sendall(b"b" * 200000)

    with _Peer(handler) as peer:
        answer = _get(peer.url, sink=taken.append)
    assert answer.content == b""
    assert b"".join(taken) == b"b" * 200000


def test_a_request_body_the_server_will_not_take_is_cut_at_the_deadline():
    """An upload to a peer that accepts and never reads stops at the
    deadline, not at the per-operation timeout."""
    release = threading.Event()

    def handler(conn, _):
        release.wait(10)

    with _Peer(handler, read_request=False) as peer:
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


def test_each_connect_attempt_and_its_handshake_get_only_what_is_left():
    """A connect attempt is given what is left of the deadline at most, and
    once it returns the socket keeps only what is left then: CPython holds
    the whole TLS handshake that follows to that one timeout. A connect that
    returns after the deadline is closed, not used."""
    dialled = []

    class _SlowDial:
        timeout = 30.0
        takes = 0.4

        def _new_conn(self):
            dialled.append(self.timeout)
            time.sleep(self.takes)
            # As urllib3's create_connection leaves it.
            sock = socket.socket()
            sock.settimeout(self.timeout)
            made.append(sock)
            return sock

    class _Pool:
        ConnectionCls = _SlowDial

    made = []
    end = time.monotonic() + 1.0
    conn = bounded_http._deadline_pool(_Pool, end, 30.0).ConnectionCls()
    with conn._new_conn() as sock:
        assert 0 < dialled[0] <= 1.0
        assert sock.gettimeout() < 0.8

    late = bounded_http._deadline_pool(_Pool, time.monotonic() + 0.1, 30.0)
    with pytest.raises(bounded_http._DeadlinePassed):
        late.ConnectionCls()._new_conn()
    assert made[-1].fileno() == -1


def _resolving_to(monkeypatch, *addresses):
    def getaddrinfo(host, port, *args, **kwargs):
        family = {4: socket.AF_INET, 6: socket.AF_INET6}
        return [
            (family[6 if ":" in a else 4], socket.SOCK_STREAM, 6, "", (a, port))
            for a in addresses
        ]

    monkeypatch.setattr(bounded_http.socket, "getaddrinfo", getaddrinfo)


def test_a_redirect_to_a_public_https_host_is_followed_relative_or_absolute(
    monkeypatch,
):
    _resolving_to(monkeypatch, PUBLIC_IPV4)
    base = "https://models.example/a/resolve/main/x.bin"
    assert vet_redirect_target(base, "/b/x.bin") == (
        "https://models.example/b/x.bin",
        (PUBLIC_IPV4,),
    )
    assert vet_redirect_target(base, "https://cdn.example/x?sig=1") == (
        "https://cdn.example/x?sig=1",
        (PUBLIC_IPV4,),
    )


@pytest.mark.parametrize(
    "location, addresses",
    [
        ("http://cdn.example/x", [PUBLIC_IPV4]),
        ("https://user@cdn.example/x", [PUBLIC_IPV4]),
        ("", [PUBLIC_IPV4]),
        ("https://cdn.example/x", ["127.0.0.1"]),
        ("https://cdn.example/x", ["::1"]),
        ("https://cdn.example/x", ["::ffff:127.0.0.1"]),
        ("https://cdn.example/x", [LAN_IPV4]),
        ("https://cdn.example/x", [PUBLIC_IPV4, PRIVATE_10_IPV4]),
        ("https://cdn.example/x", ["169.254.169.254"]),
        ("https://cdn.example/x", ["fe80::1%eth0"]),
        ("https://cdn.example/x", ["100.64.0.1"]),
        ("https://cdn.example/x", ["0.0.0.0"]),
        ("https://cdn.example/x", ["224.0.0.1"]),
        ("https://cdn.example/x", ["192.0.2.10"]),
        # A name whose AAAA record is NAT64 for the link-local metadata host.
        ("https://cdn.example/x", [nat64("169.254.169.254")]),
    ],
)
@pytest.mark.parametrize("checked", ["redirect", "first link"])
def test_a_redirect_or_first_link_anywhere_but_a_public_https_host_is_refused(
    monkeypatch, location, addresses, checked
):
    """A redirect's target and the first link of a request that follows
    redirects are held to the same rules."""
    _resolving_to(monkeypatch, *addresses)
    if checked == "redirect":
        with pytest.raises(RedirectRefused):
            vet_redirect_target("https://models.example/x", location)
    else:
        with pytest.raises(TargetRefused):
            vet_target(location)


@pytest.mark.parametrize("address", WRAPPED_PRIVATE_IPV6)
@pytest.mark.parametrize("checked", ["redirect", "first link"])
def test_an_ipv6_literal_carrying_a_private_ipv4_is_refused(address, checked):
    """Resolved for real: a literal needs no DNS, so nothing stands in."""
    url = f"https://[{address}]/x"
    if checked == "redirect":
        with pytest.raises(RedirectRefused, match="private or local"):
            vet_redirect_target("https://models.example/x", url)
    else:
        with pytest.raises(TargetRefused, match="private or local"):
            vet_target(url)


@pytest.mark.parametrize("address", GLOBAL_BUT_NOT_PUBLIC_IPV6)
@pytest.mark.parametrize("checked", ["redirect", "first link"])
def test_an_address_python_calls_global_that_no_public_host_owns_is_refused(
    address, checked
):
    """Site-local, reserved (SIIT, SRv6) and documentation IPv6: each
    ``ipaddress`` calls global. Resolved for real, as a literal."""
    url = f"https://[{address}]/x"
    if checked == "redirect":
        with pytest.raises(RedirectRefused, match="private or local"):
            vet_redirect_target("https://models.example/x", url)
    else:
        with pytest.raises(TargetRefused, match="private or local"):
            vet_target(url)


@pytest.mark.parametrize("address", PUBLIC_ADDRESS_FORMS)
def test_a_public_address_in_each_of_its_forms_is_still_followed(address):
    """The over-blocking direction of the test above, through the same real
    check: NAT64's 64:ff9b::/96 is itself reserved, so it is judged by the
    IPv4 it carries, never refused as a form."""
    host = f"[{address}]" if ":" in address else address
    assert vet_target(f"https://{host}/x") == (address,)


@pytest.mark.parametrize(
    "resolved, carried",
    [
        ("::ffff:203.0.113.10", "203.0.113.10"),
        ("::ffff:127.0.0.1", "127.0.0.1"),
        (nat64("203.0.113.10"), "203.0.113.10"),
        (nat64("127.0.0.1"), "127.0.0.1"),
    ],
)
def test_an_address_carrying_an_ipv4_is_judged_by_that_ipv4(
    monkeypatch, resolved, carried
):
    """Whatever this Python's ``ipaddress`` makes of the IPv6 form: 3.11 calls
    every IPv4-mapped address non-global where 3.12 judges its IPv4, and both
    call all of 64:ff9b::/96 global."""
    real = bounded_http._is_public
    judged = []

    def recording(address):
        judged.append(address)
        return real(address)

    monkeypatch.setattr(bounded_http, "_is_public", recording)
    _resolving_to(monkeypatch, resolved)
    with pytest.raises(TargetRefused, match="private or local"):
        vet_target("https://cdn.example/x")
    assert judged == [ipaddress.IPv4Address(carried)]


@pytest.mark.parametrize("resolved", ["::ffff:8.8.8.8", nat64(PUBLIC_IPV4)])
def test_a_public_ipv4_carried_in_ipv6_is_followed(monkeypatch, resolved):
    """The direction an over-broad refusal breaks: on an IPv6-only network
    DNS64 answers every public IPv4 host with a 64:ff9b:: address."""
    _resolving_to(monkeypatch, resolved)
    assert vet_target("https://cdn.example/x") == (resolved,)


@pytest.mark.parametrize(
    "address",
    [
        # RFC 8215 local-use NAT64: the network decides where the IPv4 sits.
        "64:ff9b:1::808:808",
        ipv4_compatible(PUBLIC_IPV4),
        # 6to4 (RFC 3056), the IPv4 in bits 16-48.
        "2002:808:808::1",
    ],
)
def test_an_ipv6_form_whose_ipv4_cannot_be_judged_is_refused_whatever_it_carries(
    monkeypatch, address
):
    """Refused as a form: each carries a public IPv4 and the check is told
    every address is public, yet each is refused, while a plain IPv6
    address under the same stand-in is followed."""
    monkeypatch.setattr(bounded_http, "_is_public", lambda judged: True)
    _resolving_to(monkeypatch, address)
    with pytest.raises(TargetRefused, match="private or local"):
        vet_target("https://cdn.example/x")
    _resolving_to(monkeypatch, "2001:4860:4860::8888")
    assert vet_target("https://cdn.example/x") == ("2001:4860:4860::8888",)


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


def test_a_followed_redirect_is_checked_and_leaves_credentials_behind(monkeypatch):
    """The hop is checked before it is asked. It carries no body, and only
    the allowlisted headers go to another host, so a credential under a name
    nobody listed (``X-Api-Key``) stays behind as ``Authorization`` does."""
    checked = []
    body = b'{"secret": "test-body"}'
    bodies = {"first": [], "target": []}

    def check(current, location, proxies=None):
        checked.append(location)
        return location, ("127.0.0.1",)

    def answer_ok(conn, head):
        bodies["target"].append(_rest_of_body(conn, head))
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    monkeypatch.setattr(bounded_http, "vet_redirect_target", check)
    with _Peer(answer_ok) as target:

        def handler(conn, head):
            # Read before answering: bytes left unread at close reset the
            # connection, which can lose the answer.
            bodies["first"].append(_rest_of_body(conn, head))
            conn.sendall(
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: %s/x\r\n"
                b"Content-Length: 0\r\n\r\n" % target.url.encode()
            )

        with _Peer(handler) as peer:
            first = allow_local_link(monkeypatch, bounded_http, peer.url)
            answer = _get(
                peer.url,
                redirects=3,
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Api-Key": "test-api-key",
                    "User-Agent": "t",
                },
                data=body,
            )
    assert answer.content == b"ok"
    assert first == [peer.url]
    assert checked == [f"{target.url}/x"]
    sent, followed = peer.requests[0].lower(), target.requests[0].lower()
    assert b"authorization: bearer test-token" in sent
    assert b"x-api-key: test-api-key" in sent
    assert bodies["first"] == [body]
    assert b"authorization" not in followed
    assert b"x-api-key" not in followed
    assert b"user-agent: t\r\n" in followed
    assert b"content-length" not in followed
    assert bodies["target"] == [b""]


def test_a_redirect_to_this_computer_is_refused_unasked(monkeypatch):
    with _Peer(lambda conn, _: conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")) as target:

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: https://127.0.0.1:%d/\r\n"
                b"Content-Length: 0\r\n\r\n" % target.port
            )

        with _Peer(handler) as peer:
            allow_local_link(monkeypatch, bounded_http, peer.url)
            with pytest.raises(RedirectRefused):
                _get(peer.url, redirects=3)
    assert target.asked == 0


def test_redirects_past_the_number_allowed_are_refused(monkeypatch):
    monkeypatch.setattr(
        bounded_http,
        "vet_redirect_target",
        lambda current, location, proxies=None: (location, ("127.0.0.1",)),
    )

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:%d/\r\n"
            b"Content-Length: 0\r\n\r\n" % peer.port
        )

    with _Peer(handler) as peer:
        allow_local_link(monkeypatch, bounded_http, peer.url)
        with pytest.raises(RedirectRefused):
            _get(peer.url, redirects=2)
    assert peer.asked == 3


def test_the_addresses_a_redirect_is_connected_to_are_every_one_resolved_in_order(
    monkeypatch,
):
    """All of them, so a host whose first address does not answer is still
    reached; each once, in the order the resolver gave."""
    _resolving_to(monkeypatch, PUBLIC_IPV4, "2001:4860:4860::8888", PUBLIC_IPV4)
    assert vet_redirect_target("https://models.example/x", "https://cdn.example/y") == (
        "https://cdn.example/y",
        (PUBLIC_IPV4, "2001:4860:4860::8888"),
    )


# ---------------------------------------------------------------------------
# DNS rebinding: a followed redirect connects to the address that was checked
# ---------------------------------------------------------------------------

PINNED_NAME = "pinned.test"


def _tls_files(tmp_path, name):
    """A throwaway CA and a certificate for *name* it signed, as PEM files:
    ``(ca, chain, key)``."""
    now = datetime.datetime.now(datetime.timezone.utc)
    start, end = now - datetime.timedelta(hours=1), now + datetime.timedelta(days=1)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    paths = (tmp_path / "ca.pem", tmp_path / "chain.pem", tmp_path / "key.pem")
    paths[0].write_bytes(ca.public_bytes(pem))
    paths[1].write_bytes(leaf.public_bytes(pem))
    paths[2].write_bytes(
        key.private_bytes(
            pem,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return paths


def _tls_peer(chain, key, body=b"ok", answer=None):
    """A local https peer answering 200 *body*, or *answer* whole when given;
    ``.names`` records the SNI name each handshake carried, ``.heads`` each
    request head read over TLS."""
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
                tls.sendall(
                    answer
                    or b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s"
                    % (len(body), body)
                )
        except ssl.SSLError as exc:
            # The client refused the certificate, which a test may intend.
            peer.refused.append(exc)

    peer = _Peer(handler, read_request=False)
    peer.names = names
    peer.heads = []
    peer.refused = []
    return peer


def _rebinding(monkeypatch, name, first_port, then_port):
    """*name* resolves to this computer at *first_port* the first time and at
    *then_port* after that, as a short-TTL record changed between the check
    and the connect would; every other name resolves as it does. Returns the
    list of ports handed out for *name*."""
    real = socket.getaddrinfo
    handed = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host != name:
            return real(host, port, *args, **kwargs)
        handed.append(first_port if not handed else then_port)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", handed[-1]))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # This computer stands in for the public address the check accepts.
    monkeypatch.setattr(bounded_http, "_is_public", lambda address: True)
    return handed


def _redirecting_to(location):
    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 302 Found\r\nLocation: %s\r\nContent-Length: 0\r\n\r\n"
            % location.encode()
        )

    return _Peer(handler)


def test_a_followed_redirect_is_connected_to_the_address_checked_not_a_new_lookup(
    tmp_path, monkeypatch
):
    """The name is resolved once, by the check; a second answer pointing
    somewhere else is never asked for. Host and SNI still carry the name, and
    the certificate is verified against it."""
    ca, chain, key = _tls_files(tmp_path, PINNED_NAME)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    with (
        _tls_peer(chain, key, b"vetted") as vetted,
        _tls_peer(chain, key, b"rebound") as rebound,
    ):
        handed = _rebinding(monkeypatch, PINNED_NAME, vetted.port, rebound.port)
        target = f"https://{PINNED_NAME}:{vetted.port}/x"
        with _redirecting_to(target) as origin:
            allow_local_link(monkeypatch, bounded_http, origin.url)
            answer = _get(origin.url, redirects=2, deadline_s=5.0)
    assert answer.content == b"vetted"
    assert handed == [vetted.port]
    assert rebound.asked == 0
    assert vetted.names == [PINNED_NAME]
    assert vetted.refused == []
    assert f"host: {PINNED_NAME}:{vetted.port}\r\n".encode() in vetted.heads[0].lower()
    assert vetted.heads[0].startswith(b"GET /x HTTP/1.1\r\n")


def test_a_pinned_redirect_still_refuses_a_certificate_for_another_name(
    tmp_path, monkeypatch
):
    """Connecting by address must not turn the certificate check off: a
    certificate that does not name the redirect's host is refused."""
    ca, chain, key = _tls_files(tmp_path, "other.test")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    with _tls_peer(chain, key) as impostor:
        _rebinding(monkeypatch, PINNED_NAME, impostor.port, impostor.port)
        with _redirecting_to(f"https://{PINNED_NAME}:{impostor.port}/x") as origin:
            allow_local_link(monkeypatch, bounded_http, origin.url)
            with pytest.raises(requests.exceptions.SSLError):
                _get(origin.url, redirects=2, deadline_s=5.0)
    assert impostor.names == [PINNED_NAME]
    assert impostor.heads == []


# The first link of a request that follows redirects is checked and pinned
# like a hop: it reaches a host nobody typed just as a redirect does.


def test_the_first_link_of_a_redirecting_request_connects_to_the_address_checked(
    tmp_path, monkeypatch
):
    """The link itself is resolved once, by the check; a second answer
    pointing somewhere else is never asked for, and nothing reaches it."""
    ca, chain, key = _tls_files(tmp_path, PINNED_NAME)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    with (
        _tls_peer(chain, key, b"vetted") as vetted,
        _tls_peer(chain, key, b"rebound") as rebound,
    ):
        handed = _rebinding(monkeypatch, PINNED_NAME, vetted.port, rebound.port)
        answer = _get(
            f"https://{PINNED_NAME}:{vetted.port}/x", redirects=2, deadline_s=5.0
        )
    assert answer.content == b"vetted"
    assert handed == [vetted.port]
    assert rebound.asked == 0
    assert vetted.names == [PINNED_NAME]
    assert f"host: {PINNED_NAME}:{vetted.port}\r\n".encode() in vetted.heads[0].lower()


def test_a_first_link_to_this_computer_is_refused_with_nothing_sent(monkeypatch):
    _resolving_to(monkeypatch, "127.0.0.1")
    with _Peer(lambda conn, _: conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")) as peer:
        with pytest.raises(TargetRefused, match="private or local"):
            _get(f"https://{PINNED_NAME}:{peer.port}/x", redirects=2)
    assert peer.asked == 0


def test_a_public_first_link_and_its_redirect_are_both_followed(tmp_path, monkeypatch):
    """A link and its redirect that pass their checks are read as before,
    each name looked up once, by its check, and each certificate verified."""
    ca, chain, key = _tls_files(tmp_path, PINNED_NAME)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    real = socket.getaddrinfo
    looked_up = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host != PINNED_NAME:
            return real(host, port, *args, **kwargs)
        looked_up.append(port)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # This computer stands in for the public address the checks accept.
    monkeypatch.setattr(bounded_http, "_is_public", lambda address: True)
    with _tls_peer(chain, key, b"the file") as cdn:
        moved = (
            b"HTTP/1.1 302 Found\r\nLocation: https://%s:%d/file\r\n"
            b"Content-Length: 0\r\n\r\n" % (PINNED_NAME.encode(), cdn.port)
        )
        with _tls_peer(chain, key, answer=moved) as origin:
            answer = _get(
                f"https://{PINNED_NAME}:{origin.port}/a", redirects=2, deadline_s=5.0
            )
    assert answer.content == b"the file"
    assert answer.url == f"https://{PINNED_NAME}:{cdn.port}/file"
    assert looked_up == [origin.port, cdn.port]
    assert origin.names == [PINNED_NAME]
    assert cdn.names == [PINNED_NAME]
    assert origin.refused == cdn.refused == []


def _rebinding_every_name(monkeypatch, first_port, then_port):
    """Every name resolves to this computer, at *first_port* for the first
    name asked and at *then_port* after that, as two names one attacker
    controls would; an IP literal resolves as it does. Returns the names
    asked, in order."""
    real = socket.getaddrinfo
    asked = []

    def getaddrinfo(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            asked.append(host)
            chosen = first_port if len(asked) == 1 else then_port
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", chosen))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # This computer stands in for the public address the check accepts.
    monkeypatch.setattr(bounded_http, "_is_public", lambda address: True)
    return asked


@pytest.mark.parametrize(
    "typed, sent",
    [
        ("bücher.example", "xn--bcher-kva.example"),
        ("straße.example", "xn--strae-oqa.example"),
        ("BÜCHER.example", "xn--bcher-kva.example"),
    ],
)
@pytest.mark.parametrize("via", ["first link", "redirect"])
def test_a_non_ascii_host_connects_to_the_address_checked_for_the_name_sent(
    tmp_path, monkeypatch, typed, sent, via
):
    """The name checked is the one ``requests`` sends, its ``xn--`` form, and
    it is looked up once. Python's own codec would check ``strasse`` while
    the connection looked ``xn--strae-oqa`` up afresh: two names, which one
    attacker can point at two places. A redirect delivers such a host as
    UTF-8 bytes in its Location, which ``requests`` decodes back to the
    Unicode name."""
    ca, chain, key = _tls_files(tmp_path, sent)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    with (
        _tls_peer(chain, key, b"vetted") as vetted,
        _tls_peer(chain, key, b"rebound") as rebound,
    ):
        asked = _rebinding_every_name(monkeypatch, vetted.port, rebound.port)
        link = f"https://{typed}:{vetted.port}/x"
        if via == "first link":
            answer = _get(link, redirects=1, deadline_s=5.0)
        else:

            def handler(conn, _):
                conn.sendall(
                    b"HTTP/1.1 302 Found\r\nLocation: %s\r\n"
                    b"Content-Length: 0\r\n\r\n" % link.encode("utf-8")
                )

            with _Peer(handler) as origin:
                allow_local_link(monkeypatch, bounded_http, origin.url)
                answer = _get(origin.url, redirects=1, deadline_s=5.0)
    assert answer.content == b"vetted"
    assert asked == [sent]
    assert rebound.asked == 0
    assert vetted.names == [sent]
    assert f"host: {sent}:{vetted.port}\r\n".encode() in vetted.heads[0].lower()


def test_a_pinned_adapter_refuses_an_origin_it_holds_no_pin_for():
    """Refused with nothing sent, never resolved afresh: with no pins at all,
    and with a pin for another origin only. The pinned origin is reached."""

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    with _Peer(handler) as peer, requests.Session() as session:
        adapter = bounded_http.PinnedAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        with pytest.raises(TargetRefused, match="was not checked"):
            session.get(peer.url, proxies=NO_PROXY, timeout=5)
        adapter.pin("http://other.example/", ("127.0.0.1",))
        with pytest.raises(TargetRefused, match="was not checked"):
            session.get(peer.url, proxies=NO_PROXY, timeout=5)
        assert peer.asked == 0
        adapter.pin(peer.url, ("127.0.0.1",))
        assert session.get(peer.url, proxies=NO_PROXY, timeout=5).content == b"ok"
    assert peer.asked == 1


# Linux routes all of 127.0.0.0/8 to loopback and the peers listen on
# 127.0.0.1 alone, so a connection here is refused at once.
_NOTHING_LISTENS = "127.0.0.2"


def test_a_pinned_request_dials_each_checked_address_until_one_answers(
    tmp_path, monkeypatch
):
    """A first address that refuses the connection (a dead IPv6 route, say)
    does not fail the request: the next checked address is dialled, in the
    order they resolved, with the name still on SNI and Host."""
    ca, chain, key = _tls_files(tmp_path, PINNED_NAME)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    real = socket.getaddrinfo
    asked = []

    def getaddrinfo(host, port, *args, **kwargs):
        asked.append(host)
        if host != PINNED_NAME:
            return real(host, port, *args, **kwargs)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
            for address in (_NOTHING_LISTENS, "127.0.0.1")
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(bounded_http, "_is_public", lambda address: True)
    with _tls_peer(chain, key, b"second") as peer:
        answer = _get(
            f"https://{PINNED_NAME}:{peer.port}/x", redirects=1, deadline_s=5.0
        )
    assert answer.content == b"second"
    # The name once, by the check; then each address, as urllib3 dials one.
    assert asked == [PINNED_NAME, _NOTHING_LISTENS, "127.0.0.1"]
    assert peer.names == [PINNED_NAME]
    assert f"host: {PINNED_NAME}:{peer.port}\r\n".encode() in peer.heads[0].lower()


def test_a_pinned_connection_with_no_route_dials_nothing():
    """The second lock behind the pool key: a connection made on the
    adapter's pools some other way than through its pin lookup is refused
    before any dial, and dials once its origin is pinned."""
    adapter = bounded_http.PinnedAdapter()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(0.5)
        port = server.getsockname()[1]
        pool = adapter.poolmanager.connection_from_host(
            "127.0.0.1", port, scheme="http"
        )
        conn = pool._new_conn()
        with pytest.raises(
            urllib3_errors.NewConnectionError,
            match="was not dialled",
        ):
            conn.connect()
        with pytest.raises(TimeoutError):
            server.accept()
        adapter.pin(f"http://127.0.0.1:{port}/", ("127.0.0.1",))
        conn.connect()
        try:
            accepted, _ = server.accept()
            accepted.close()
        finally:
            conn.close()


def test_a_stopped_pinned_adapter_dials_nothing_more():
    """A cancel that lands between two connections of a request (a redirect
    hop, a retry) stops the next one before it is dialled. The same request
    is answered before the stop."""

    def handler(conn, _):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        )

    with _Peer(handler) as peer, requests.Session() as session:
        adapter = bounded_http.PinnedAdapter()
        session.mount("http://", adapter)
        adapter.pin(peer.url, ("127.0.0.1",))
        assert session.get(peer.url, proxies=NO_PROXY, timeout=5).content == b"ok"
        adapter.abort()
        with pytest.raises(requests.ConnectionError, match="The request was stopped"):
            session.get(peer.url, proxies=NO_PROXY, timeout=5)
    assert peer.asked == 1


def _read_through_floor(piece, gap, times, floor, window):
    """What a :class:`bounded_http._FloorReader` reads from a peer sending
    *piece* every *gap* seconds, *times* times, then closing."""
    ours, theirs = socket.socketpair()
    stop = threading.Event()

    def feed():
        with theirs:
            for _ in range(times):
                if stop.wait(gap):
                    return
                theirs.sendall(piece)

    feeder = threading.Thread(target=feed, daemon=True)
    reader = bounded_http._FloorReader(ours, floor, window)
    got = bytearray()
    buffer = bytearray(64)
    feeder.start()
    try:
        while count := reader.readinto(buffer):
            got += buffer[:count]
    finally:
        stop.set()
        feeder.join(5)
        reader.close()
        ours.close()
    return bytes(got)


def test_a_stall_floor_opens_a_new_window_each_time_its_bytes_have_come():
    """Four bytes every 0.2 s against a floor of four a second is read to the
    end, well past the first second: each four bytes open a new window."""
    started = time.monotonic()
    assert _read_through_floor(b"abcd", 0.2, 8, 4, 1.0) == b"abcd" * 8
    assert time.monotonic() - started > 1.0


def test_a_stall_floor_cuts_off_a_trickle_within_one_window():
    """A byte every 0.4 s never brings four in a second: cut off when the
    first window closes, however long the peer would keep trickling."""
    with (
        _within(2.5),
        pytest.raises(bounded_http._Stalled, match="fewer than 4 bytes came in 1 s"),
    ):
        _read_through_floor(b"a", 0.4, 20, 4, 1.0)


def test_the_local_link_allowance_lets_only_its_own_server_through(monkeypatch):
    """A test reaches a plain-http server of its own only through
    ``allow_local_link``, and only that server: a link anywhere else, even
    one port away on this computer, is refused as it is without the
    allowance, with nothing sent."""

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    with _Peer(handler) as own, _Peer(handler) as other:
        allowed = allow_local_link(monkeypatch, bounded_http, own.url)
        assert _get(f"{own.url}/x", redirects=2).content == b"ok"
        with pytest.raises(TargetRefused, match="is not https"):
            _get(f"{other.url}/x", redirects=2)
    assert allowed == [f"{own.url}/x"]
    assert other.asked == 0


# The callers whose reads are tested here rather than through a stand-in for
# bounded_get: each keeps its own deadline, cap and no-redirect rule, and the
# local peers below show those reach the wire.
_CALLERS = {
    "ComfyUI probe": (
        lambda url: comfyui_tenant._read_json(url),
        comfyui_tenant,
        "READ_DEADLINE_S",
        "READ_MAX_BYTES",
    ),
    "ComfyUI templates": (
        lambda url: comfyui_templates._get_json(url),
        comfyui_templates,
        "DEADLINE_S",
        "MAX_GRAPH_BYTES",
    ),
}


@pytest.mark.parametrize("caller", list(_CALLERS))
def test_a_caller_cuts_a_dribbled_answer_at_its_own_deadline(caller, monkeypatch):
    read, module, deadline, _cap = _CALLERS[caller]
    monkeypatch.setattr(module, deadline, 0.5)

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 60\r\n\r\n")
        _trickle(conn, b"a" * 60)

    with _Peer(handler) as peer, _within(3.0):
        with pytest.raises(ReadDeadlineExceeded):
            read(peer.url)


@pytest.mark.parametrize("caller", list(_CALLERS))
def test_a_caller_cuts_an_endless_answer_at_its_own_cap(caller, monkeypatch):
    read, module, _deadline, cap = _CALLERS[caller]
    monkeypatch.setattr(module, cap, 64 * 1024)
    sent = []

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        chunk = b"a" * 65536
        while sum(sent) < 64 << 20:
            conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            sent.append(len(chunk))

    with _Peer(handler) as peer, _within(5.0):
        with pytest.raises(ResponseTooLarge):
            read(peer.url)
    assert sum(sent) < 16 << 20


@pytest.mark.parametrize("caller", list(_CALLERS))
def test_a_caller_never_follows_a_redirect(caller):
    read = _CALLERS[caller][0]
    with _Peer(lambda conn, _: conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")) as target:

        def handler(conn, _):
            conn.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: %s/\r\n"
                b"Content-Length: 0\r\n\r\n" % target.url.encode()
            )

        with _Peer(handler) as peer:
            with pytest.raises(requests.HTTPError, match="302"):
                read(peer.url)
    assert target.asked == 0


def test_a_host_is_checked_as_requests_sends_it_not_as_urlsplit_reads_it(
    monkeypatch,
):
    """``urlsplit`` keeps a backslash inside the host, where ``requests``
    ends the host at it and connects to what comes before. The name looked
    up, judged and named is the one ``requests`` would connect to."""
    looked_up = []

    def getaddrinfo(host, port, *args, **kwargs):
        looked_up.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (LAN_IPV4, port))]

    monkeypatch.setattr(bounded_http.socket, "getaddrinfo", getaddrinfo)
    link = "https://evil.example\\.models.example/x"
    assert urlsplit(link).hostname == "evil.example\\.models.example"
    with pytest.raises(TargetRefused, match="^The link to evil.example leads"):
        vet_target(link)
    assert looked_up == ["evil.example"]


@pytest.mark.parametrize(
    "link",
    [
        "https://models.example\t.other.example/x",
        "https://models.example\n.other.example/x",
        "https://user@models.example/x",
        "https://user:test-password@models.example/x",
    ],
    ids=["tab", "newline", "user", "user-and-password"],
)
def test_a_link_requests_would_not_send_as_named_or_that_names_a_user_is_refused(
    monkeypatch, link
):
    looked_up = []
    monkeypatch.setattr(
        bounded_http.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: looked_up.append(host) or [],
    )
    with pytest.raises(TargetRefused, match="named no plain host"):
        vet_target(link)
    assert looked_up == []


def _cut_off_after(monkeypatch, first):
    """Make every body read return *first*, then end, however much the peer
    sends: a stream that ends early without raising, as urllib3 1.26 left
    one, or a 2.x that stopped holding a body to its length."""
    reads = iter([first])
    monkeypatch.setattr(
        bounded_http, "_receive", lambda response, url, end: next(reads, b"")
    )


def test_an_identity_body_that_ends_short_of_its_length_is_refused(monkeypatch):
    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n0123456789")

    _cut_off_after(monkeypatch, b"012")
    with _Peer(handler) as peer:
        with pytest.raises(
            requests.exceptions.ChunkedEncodingError, match="sent 3 of the 10 bytes"
        ):
            _get(peer.url)


@pytest.mark.parametrize("method, status", [("GET", 200), ("HEAD", 200), ("GET", 204)])
def test_an_answer_whole_by_its_length_or_with_no_body_is_read(
    monkeypatch, method, status
):
    """The other direction: three bytes declared and three read is whole, and
    a HEAD's or a 204's Content-Length promises no body (RFC 9112 section
    6.3), so none is waited for or missed."""

    def handler(conn, _):
        declared = 3 if method == "GET" and status == 200 else 10
        conn.sendall(
            b"HTTP/1.1 %d X\r\nContent-Length: %d\r\n\r\n" % (status, declared)
        )
        if method == "GET" and status == 200:
            conn.sendall(b"012")

    with _Peer(handler) as peer:
        answer = bounded_request(
            method,
            peer.url,
            deadline_s=2.0,
            max_bytes=1024,
            timeout=2.0,
            proxies=NO_PROXY,
        )
    assert answer.status_code == status
    assert answer.content == (b"012" if method == "GET" and status == 200 else b"")


def test_a_request_body_sent_slowly_is_cut_at_the_deadline_mid_send():
    """A 64 MiB upload to a peer that takes 64 KiB every 0.1 s: each send
    goes, so no one send ever waits a whole per-operation timeout, yet the
    request ends at its 2 s deadline, not after the 100 s the upload would
    take. Wrong if a send is given the full timeout rather than what is left
    of the deadline."""
    stop = threading.Event()

    def medium_reader(conn, _):
        started = time.monotonic()
        while not stop.is_set() and time.monotonic() - started < 15.0:
            if not conn.recv(65536):
                return
            time.sleep(0.1)

    chunks = (b"\0" * 65536 for _ in range(1024))
    with _Peer(medium_reader, read_request=False) as peer:
        try:
            with _within(6.0), pytest.raises(ReadDeadlineExceeded):
                bounded_request(
                    "POST",
                    peer.url,
                    data=chunks,
                    deadline_s=2.0,
                    max_bytes=1 << 20,
                    timeout=10.0,
                    proxies=NO_PROXY,
                )
        finally:
            stop.set()


@contextlib.contextmanager
def _proxy_answering_403(monkeypatch, proxy_env=True, no_proxy=None):
    """A peer standing in for an HTTPS proxy: it records each ``CONNECT``
    head and refuses it. Named by ``https_proxy`` when *proxy_env*; every
    other proxy setting in the environment is cleared, so only this test's
    count."""
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    def refuse(conn, _):
        conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")

    with _Peer(refuse) as proxy:
        if proxy_env:
            monkeypatch.setenv("https_proxy", proxy.url)
        if no_proxy:
            monkeypatch.setenv("no_proxy", no_proxy)
        yield proxy


def _models_example_resolves(monkeypatch, *addresses):
    """``models.example`` resolves to *addresses*, or fails to resolve with
    none; every other name (the proxy's 127.0.0.1) resolves for real."""
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host != "models.example":
            return real(host, port, *args, **kwargs)
        if not addresses:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize("resolves", ["public", "not at all"])
@pytest.mark.parametrize("named", ["environment", "argument"])
def test_a_pinned_request_through_a_proxy_asks_it_to_connect_to_the_name(
    monkeypatch, resolves, named
):
    """Behind a proxy the proxy dials: it is asked to CONNECT to the host by
    name, whether local DNS answered with a public address or not at all
    (a network where only the proxy resolves names). The same proxy named
    in *proxies* rather than the environment counts the same."""
    with _proxy_answering_403(monkeypatch, proxy_env=named == "environment") as proxy:
        _models_example_resolves(
            monkeypatch, *([PUBLIC_IPV4] if resolves == "public" else [])
        )
        proxies = {"https": proxy.url} if named == "argument" else None
        with pytest.raises(requests.exceptions.ProxyError):
            _get("https://models.example/x", redirects=1, proxies=proxies)
    assert proxy.asked == 1
    assert proxy.requests[0].startswith(b"CONNECT models.example:443 HTTP/1.1\r\n")


def test_a_host_no_proxy_exempts_must_resolve_here(monkeypatch):
    """``no_proxy`` sends the request direct, so a name local DNS cannot
    resolve is refused, with nothing sent to the proxy or anywhere. Refused
    by the check, which names the host, not by the pin's backstop."""
    with _proxy_answering_403(monkeypatch, no_proxy="models.example") as proxy:
        _models_example_resolves(monkeypatch)
        with pytest.raises(TargetRefused, match="to models.example could not be"):
            _get("https://models.example/x", redirects=1, proxies=None)
    assert proxy.asked == 0


def test_a_private_address_is_refused_even_through_a_proxy(monkeypatch):
    """The proxy owns egress, but a name local DNS places inside this
    network is still refused before the proxy is asked."""
    with _proxy_answering_403(monkeypatch) as proxy:
        _models_example_resolves(monkeypatch, LAN_IPV4)
        with pytest.raises(TargetRefused, match="private or local"):
            _get("https://models.example/x", redirects=1, proxies=None)
    assert proxy.asked == 0


def test_a_pin_without_addresses_is_refused_when_sent_without_a_proxy():
    """A pin is empty only because a proxy was to resolve the name; sent
    direct after all, it is refused with nothing dialled."""

    def handler(conn, _):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    with _Peer(handler) as peer, requests.Session() as session:
        adapter = bounded_http.PinnedAdapter()
        session.mount("http://", adapter)
        adapter.pin(peer.url, ())
        with pytest.raises(TargetRefused, match="could not be resolved"):
            session.get(peer.url, proxies=NO_PROXY, timeout=5)
    assert peer.asked == 0


@pytest.mark.parametrize(
    "address",
    [
        "192.88.99.0",
        "192.88.99.1",
        # The 6a44 relay (RFC 6751), which IANA marks not globally reachable.
        "192.88.99.2",
        "192.88.99.255",
        "::ffff:192.88.99.1",
        nat64("192.88.99.1"),
    ],
)
@pytest.mark.parametrize("checked", ["redirect", "first link"])
def test_the_deprecated_6to4_relay_block_is_refused(address, checked):
    """192.88.99.0/24, deprecated by RFC 7526, which ``ipaddress`` calls
    global: a relay there is whichever the network routes to. Wrong if any of
    the block passes, as itself or carried in IPv6."""
    host = f"[{address}]" if ":" in address else address
    url = f"https://{host}/x"
    if checked == "redirect":
        with pytest.raises(RedirectRefused, match="private or local"):
            vet_redirect_target("https://models.example/x", url)
    else:
        with pytest.raises(TargetRefused, match="private or local"):
            vet_target(url)


@pytest.mark.parametrize("address", ["192.88.98.255", "192.88.100.0"])
def test_the_addresses_either_side_of_the_6to4_relay_block_are_followed(address):
    """The over-blocking direction: only the /24 goes."""
    assert vet_target(f"https://{address}/x") == (address,)
