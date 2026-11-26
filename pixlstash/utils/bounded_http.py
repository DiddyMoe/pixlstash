"""HTTP requests whose whole answer is read under a total deadline and a size cap.

``requests``' ``timeout`` bounds each socket operation, not the request
(https://requests.readthedocs.io/en/latest/user/advanced/#timeouts): a server
that sends one byte every few seconds keeps a request going for as long as it
likes, and one that sends a body without end fills memory. A ComfyUI is an
address the owner typed, and what answers there is not vetted, so every read
of one goes through here, as do the other outbound reads that have a size.

**The deadline covers the whole answer.** Each request gets its own session
whose connections read the socket through :class:`_DeadlineReader`, which sets
the socket timeout to what is left of the deadline before every receive. The
status line, the headers, each chunk-size line and the body all come through
that one stream (``http.client`` reads them from ``sock.makefile``), so a
server dribbling any of them is cut off when the time is up. Sending the
request is held to the same deadline (``sendall``'s timeout is the total for
the call since Python 3.5). Each connect attempt's timeout is never longer
than what is left, and neither is the TLS handshake after it: the socket's
timeout is cut to what remains once the connect returns, and CPython holds
the whole handshake to that one timeout. This does not depend on the urllib3
version: urllib3 1.26 and 2.x both build the response through
``http.client``'s ``response_class`` and dial in ``_new_conn``.

**The size cap covers raw and decoded bytes.** The request asks for
``Accept-Encoding: identity`` and the body is read undecoded; a body sent
compressed anyway (gzip or deflate) is inflated here, ``max_length`` at a
time, so a small compressed body cannot grow past the cap in memory before
it is counted. The bytes received and the bytes decoded are each held to the
cap. A compressed body cut short, or with bytes after its end, is refused.
A redirect's body is read only when the redirect comes back as the answer,
and then only its first ``ERROR_BODY_BYTES``: a plain ``requests.Session``
reads all of it before handing the answer over (:class:`HandRedirectSession`).

What the deadline does not bound: **the DNS lookup.** ``getaddrinfo`` takes
no timeout, so a lookup lasts as long as the system resolver's own timeouts
and retries allow. That is :func:`vet_target`'s lookup for a request that
follows redirects (and each hop's), and urllib3's lookup of the host for one
that does not. And a request that does not follow redirects, to a host with
several addresses, lets urllib3 try each with the timeout cut once before
the first: it can overrun by one connect timeout per address after the first.
A pinned request (below) cuts the timeout again before each address.

**Redirects are not followed** unless the caller asks for a number of them,
and then only to an https address whose every resolved address is public
(:func:`vet_redirect_target`), judging an address under the NAT64 prefix by
the IPv4 it carries; the first URL of such a request is held to the same
check (:func:`vet_target`). A ComfyUI or PixlStash's own MCP never follows
one: a redirect names a host nobody checked.

**A request to one of PixlStash's fixed https hosts checks no address.**
Civitai's hash lookup (``civitai_lookup.CIVITAI_BY_HASH``), the telemetry
ping (``telemetry.sender.TELEMETRY_ENDPOINT``), the training-preset updates
(``training.preset_updates.SITE_URL``) and the plugin repository's zip from
codeload.github.com (``plugin_install.CODELOAD_ZIP_URL``) are https
addresses written in the code, never one a user gave, and follow no redirect.
Such a request rests on TLS: urllib3 verifies the certificate against the
fixed name, so a DNS answer that points the name at a private address costs a
TCP connect and a TLS ClientHello there, and the handshake then fails before
a byte of the request is sent. The certificate check is what keeps the name
in charge (OWASP's SSRF Prevention Cheat Sheet,
https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html).
That holds only while certificate checking stays on and the four stay https:
``tests/test_architecture_guardrails.py`` fails on any ``verify=`` argument
under ``pixlstash/`` and on a non-https address in those four.

**A followed redirect, and the first URL of a request that follows them, is
connected only to the addresses that were checked**, not to a second lookup
of its name, which a name with a short TTL could answer with a private
address (DNS rebinding). :class:`PinnedAdapter` keys the connection pool on
the first vetted IP (``build_connection_pool_key_attributes``, requests' hook
for this since 2.32), dials each vetted address in the order they resolved
until one answers, and keeps the name where it belongs: the ``Host`` header,
and ``server_hostname`` for SNI and certificate checking, which urllib3
verifies against that name
(https://urllib3.readthedocs.io/en/stable/advanced-usage.html#custom-sni-hostname).
The pin is keyed on the host exactly as ``requests`` sends it, a non-ASCII
name in its IDNA ``xn--`` form, and that same form is what was resolved; a
request whose origin has no pin is refused, never looked up afresh.

Behind a proxy (one ``requests`` would use for the URL, ``NO_PROXY``
included: :func:`proxy_for`), a pinned request asks the proxy to ``CONNECT``
to the host by name, and the proxy resolves and dials it: the proxy owns
egress. The check above still runs whenever local DNS answers, and any
address that is not public refuses the request. Only when a proxy applies
may local DNS give no answer, as on a network where only the proxy resolves
names; with no proxy that is refused.

**Every host and scheme decision reads the URL as ``requests`` sends it**
(:func:`wire_split`), never ``urllib.parse`` on the text: ``urlsplit`` keeps a
backslash inside the host (``https://evil.example\\.huggingface.co/`` names
``evil.example\\.huggingface.co`` to it), where ``requests`` ends the host at
the backslash, as the WHATWG URL Standard does for a special scheme, and
connects to ``evil.example``.
"""

from __future__ import annotations

import functools
import http.client
import io
import ipaddress
import json as jsonlib
import socket
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy
from urllib3 import exceptions as urllib3_errors

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)

#: The most one receive asks for.
READ_CHUNK_BYTES = 64 * 1024
#: The most of a non-2xx answer's body that is read, for the caller's log.
ERROR_BODY_BYTES = 64 * 1024
#: The most of an error body a log line quotes.
LOG_EXCERPT_CHARS = 300
#: The most redirect hops a caller may ask to follow.
MAX_REDIRECTS = 10
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
#: The only headers a redirect carries to another host: an allowlist, so a
#: credential under a name nobody thought of (``X-Api-Key``) stays behind.
_PORTABLE_HEADERS = frozenset(
    {"user-agent", "accept", "accept-language", "accept-encoding"}
)
#: RFC 6052's well-known NAT64 prefix: a translator connects an address here
#: to the IPv4 in its last 32 bits, so that IPv4 is what is judged.
_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
#: IPv6 forms that carry an IPv4 which cannot be judged here: refused.
_REFUSED_WRAPPERS = (
    # RFC 8215 local-use NAT64: where the IPv4 sits is the network's choice.
    ipaddress.IPv6Network("64:ff9b:1::/48"),
    # IPv4-compatible, deprecated by RFC 4291 section 2.5.5.1.
    ipaddress.IPv6Network("::/96"),
    # 6to4 (RFC 3056): the IPv4 in bits 16-48 is where the packet goes.
    ipaddress.IPv6Network("2002::/16"),
)
#: RFC 9637's IPv6 documentation prefix, which ``ipaddress`` calls global.
_DOCUMENTATION_V6 = ipaddress.IPv6Network("3fff::/20")
#: The 6to4 relay anycast block, deprecated by RFC 7526 and left with no
#: "globally reachable" value in IANA's IPv4 special-purpose registry; it
#: holds the 6a44 relay (RFC 6751), which that registry marks not reachable.
#: ``ipaddress`` calls the whole block global.
_SIX_TO_FOUR_RELAY_V4 = ipaddress.IPv4Network("192.88.99.0/24")


class ResponseTooLarge(requests.RequestException):
    """The body ran past the cap the caller set."""


class ReadDeadlineExceeded(requests.Timeout):
    """The answer was still arriving when the caller's deadline passed."""


class TargetRefused(requests.RequestException):
    """An address that must be public https is not (:func:`vet_target`)."""


class RedirectRefused(TargetRefused):
    """A redirect led somewhere it may not, or there were too many."""


class _DeadlinePassed(TimeoutError):
    """Raised inside a socket read; urllib3 and requests turn it into their
    own read timeout, which :func:`bounded_request` reports as
    :class:`ReadDeadlineExceeded`."""


class _Stalled(TimeoutError):
    """Raised inside a socket read when too few bytes came in one window
    (:class:`_FloorReader`); urllib3 and requests turn it into their own read
    timeout."""


def stalled(exc: BaseException) -> bool:
    """Whether *exc* is a stall floor's cut-off (:class:`PinnedAdapter`'s
    *stall_floor*), or was raised from one: urllib3 and requests raise their
    own read errors from it, so it is found on the cause or context chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, _Stalled):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class BoundedResponse:
    """What a bounded request read: its status, headers and body.

    For a 2xx answer *content* is the whole body (empty when it went to a
    sink). For any other, it is at most ``ERROR_BODY_BYTES`` of the body,
    *truncated* saying whether there was more: enough to log, never more.
    """

    url: str
    status_code: int
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)
    truncated: bool = False

    def json(self) -> Any:
        """The body as JSON; ``ValueError`` when it is not."""
        return jsonlib.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def excerpt(self, limit: int = LOG_EXCERPT_CHARS) -> str:
        """The start of the body for a log line, marked when cut short."""
        text = self.text
        cut = self.truncated or len(text) > limit
        return repr(text[:limit]) + (" (cut short)" if cut else "")


def _left(end: float) -> float:
    """Seconds left before *end*; past it, :class:`_DeadlinePassed`."""
    left = end - time.monotonic()
    if left <= 0:
        raise _DeadlinePassed("the answer was not in by its deadline")
    return left


class _DeadlineReader(io.RawIOBase):
    """A response's socket, each receive held to what is left of *end*.

    A socket timeout bounds one receive; setting it to the time remaining
    before every receive makes it bound the whole response instead. Each
    receive is also held to *receive_s*, the caller's per-receive timeout.
    """

    def __init__(self, sock: socket.socket, end: float, receive_s: float) -> None:
        super().__init__()
        self._sock = sock
        # makefile, not a bare SocketIO: it counts a reference on the socket,
        # which keeps it open after the connection closes its own handle.
        self._raw = sock.makefile("rb", buffering=0)
        self._end = end
        self._receive_s = receive_s

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        self._sock.settimeout(min(_left(self._end), self._receive_s))
        return self._raw.readinto(buffer)

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


class _FloorReader(io.RawIOBase):
    """A response's socket that must deliver *floor* bytes every *window* s.

    A socket timeout bounds one receive, so a peer sending a byte a minute
    never trips it. This sets the timeout to what is left of the current
    window and opens a new window each time *floor* bytes have come, so a
    trickle is cut off within one window, whether it is the status line, the
    headers or the body. Each receive is still held to the timeout the socket
    had when the response began: the caller's read timeout.
    """

    def __init__(self, sock: socket.socket, floor: int, window: float) -> None:
        super().__init__()
        self._sock = sock
        # makefile, not a bare SocketIO: see _DeadlineReader.
        self._raw = sock.makefile("rb", buffering=0)
        self._floor = floor
        self._window = window
        self._receive_s = sock.gettimeout()
        self._due = time.monotonic() + window
        self._got = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        left = self._due - time.monotonic()
        if left <= 0:
            raise _Stalled(self._stalled())
        # Whichever is sooner bounds this receive: the window's end, or the
        # caller's read timeout, which keeps its own meaning.
        window_bounds = self._receive_s is None or left <= self._receive_s
        self._sock.settimeout(left if window_bounds else self._receive_s)
        try:
            got = self._raw.readinto(buffer) or 0
        except TimeoutError as exc:
            if window_bounds:
                raise _Stalled(self._stalled()) from exc
            raise
        self._got += got
        if self._got >= self._floor:
            self._got = 0
            self._due = time.monotonic() + self._window
        return got

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()

    def _stalled(self) -> str:
        return f"fewer than {self._floor} bytes came in {self._window:g} s"


class _ReadThrough:
    """The socket as ``http.client.HTTPResponse`` sees it: one stream, read
    through *reader* (made per response, as ``HTTPResponse`` asks once)."""

    def __init__(self, reader: io.RawIOBase) -> None:
        self._reader = reader

    def makefile(self, mode: str) -> io.BufferedReader:
        return io.BufferedReader(self._reader)


def _deadline_response(sock, *args, end: float, receive_s: float, **kwargs):
    """An ``http.client.HTTPResponse`` reading *sock* through a deadline."""
    return http.client.HTTPResponse(
        _ReadThrough(_DeadlineReader(sock, end, receive_s)), *args, **kwargs
    )


def _floor_response(sock, *args, floor: int, window: float, **kwargs):
    """An ``http.client.HTTPResponse`` reading *sock* through a stall floor."""
    return http.client.HTTPResponse(
        _ReadThrough(_FloorReader(sock, floor, window)), *args, **kwargs
    )


def _deadline_pool(pool_cls: type, end: float, receive_s: float) -> type:
    """*pool_cls* with connections whose response must be in by *end*.

    Made per request: urllib3's pool key is a fixed namedtuple, so the
    deadline cannot travel as a pool keyword.
    """

    class _Connection(pool_cls.ConnectionCls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.response_class = functools.partial(
                _deadline_response, end=end, receive_s=receive_s
            )

        def _new_conn(self):
            # The connect attempt gets what is left at most; the TLS
            # handshake that follows runs under the socket's timeout, as one
            # total, so that is cut to what is left once the connect is in.
            timeout = self.timeout if isinstance(self.timeout, (int, float)) else None
            left = _left(end)
            self.timeout = left if timeout is None else min(timeout, left)
            sock = super()._new_conn()
            try:
                left = _left(end)
            except _DeadlinePassed:
                sock.close()
                raise
            current = sock.gettimeout()
            sock.settimeout(left if current is None else min(current, left))
            return sock

        def send(self, data) -> None:
            if self.sock is not None:
                self.sock.settimeout(min(_left(end), receive_s))
            super().send(data)

    class _Pool(pool_cls):
        ConnectionCls = _Connection

    return _Pool


def _watched_pool(pool_cls: type, adapter: PinnedAdapter, routed: bool) -> type:
    """*pool_cls* with connections *adapter* can stop (:meth:`PinnedAdapter.
    abort`), read through its stall floor if it has one; and, when *routed*,
    dialling each vetted address of their pin in turn.

    A proxy's pools are not routed: their connection dials the proxy, which
    does its own connecting.
    """

    class _Connection(pool_cls.ConnectionCls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if adapter.stall_floor is not None:
                floor, window = adapter.stall_floor
                self.response_class = functools.partial(
                    _floor_response, floor=floor, window=window
                )

        def _new_conn(self):
            if not routed:
                adapter._check_running(self)
                return adapter._dialled(self, super()._new_conn())
            # The pool's host is the pin's first address: its key.
            named = self._dns_host
            server_hostname = getattr(self, "server_hostname", None)
            addresses = adapter._route(named, self.port, server_hostname)
            if addresses is None:
                raise urllib3_errors.NewConnectionError(
                    self, "The link's host was not checked, so it was not dialled."
                )
            try:
                for index, address in enumerate(addresses):
                    adapter._check_running(self)
                    self._dns_host = address
                    try:
                        return adapter._dialled(self, super()._new_conn())
                    except urllib3_errors.ConnectTimeoutError as exc:
                        # Refused or timed out (NewConnectionError is one):
                        # the next vetted address may answer, as urllib3
                        # tries each address of a name it resolves itself.
                        if index == len(addresses) - 1:
                            raise
                        logger.info(
                            "%s:%s for %s did not answer (%s); trying the "
                            "next checked address",
                            address,
                            self.port,
                            server_hostname or named,
                            exc,
                        )
            finally:
                # Back to the key, so a reconnect finds its route again.
                self._dns_host = named
            raise AssertionError("unreachable: a pin has at least one address")

        def connect(self) -> None:
            super().connect()
            # Here, not in _new_conn: TLS wraps the socket after it, and the
            # wrapped one is what a receive blocks on.
            adapter._watch(self, self.sock)

    class _Pool(pool_cls):
        ConnectionCls = _Connection

    return _Pool


def _wrap_pools(manager, wrap: Callable[[type], type]) -> None:
    """Make every pool *manager* makes from now on of ``wrap(its class)``."""
    manager.pool_classes_by_scheme = {
        scheme: wrap(pool_cls)
        for scheme, pool_cls in manager.pool_classes_by_scheme.items()
    }


def _wire_url(url: str) -> str:
    """*url* as ``requests`` sends it: a non-ASCII host in its IDNA (UTS 46)
    ``xn--`` form, the escapes requoted.

    Raises:
        requests.RequestException: ``requests`` would not send *url*
            (``InvalidURL``, ``MissingSchema``).
    """
    prepared = requests.PreparedRequest()
    prepared.prepare_url(url, None)
    return prepared.url or ""


def wire_split(url: str):
    """*url* split as ``requests`` sends it (:func:`_wire_url`): the scheme
    and host lowered, a non-ASCII host in its ``xn--`` form, and the host
    ended where ``requests`` ends it. The parse for any decision on which
    host or scheme an outbound request reaches.

    Raises:
        requests.RequestException: ``requests`` would not send *url*
            (``InvalidURL`` for a tab or newline in the host, say).
    """
    return urlsplit(_wire_url(url))


def proxy_for(url: str, proxies: dict | None = None) -> str | None:
    """The proxy ``requests`` sends *url* through, or None.

    Resolved as ``Session.request`` resolves it for a session with
    ``requests``' defaults, which is what every caller here sends with:
    *proxies* as given (a ``None`` value says no proxy for that scheme), the
    environment's (``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``) filled in
    under them, then ``select_proxy``.
    """
    with requests.Session() as session:
        settings = session.merge_environment_settings(
            url, dict(proxies or {}), None, None, None
        )
    return select_proxy(url, settings["proxies"])


def _origin(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` of *url* as ``requests`` sends it
    (:func:`wire_split`), the port filled in, the host lowered: the key a pin
    is stored under and the key a request looks its pin up by."""
    parts = wire_split(url)
    scheme = parts.scheme.lower()
    return (
        scheme,
        (parts.hostname or ""),
        parts.port or (443 if scheme == "https" else 80),
    )


class _Connecting(threading.local):
    """Per thread: whether the request :class:`PinnedAdapter` is connecting
    now goes through a proxy."""

    proxied = False


class PinnedAdapter(HTTPAdapter):
    """An adapter that connects only to the addresses pinned for a request's
    origin (:meth:`pin`), and refuses a request whose origin has none.

    The one owner of DNS-rebinding pinning: :func:`bounded_request` uses it
    through :class:`_PinnedDeadlineAdapter`, and a caller that streams a
    download too long for one deadline (``services.model_downloads``) mounts
    it on its own session. ``max_retries`` stays at requests' default of no
    retries.

    Args:
        stall_floor: ``(bytes, seconds)``: a response that delivers fewer
            than *bytes* in any *seconds* is cut off (:class:`_FloorReader`).
            None, the default, sets no floor.
    """

    def __init__(
        self, *args, stall_floor: tuple[int, float] | None = None, **kwargs
    ) -> None:
        # Set first: HTTPAdapter.__init__ calls init_poolmanager.
        self._pins: dict[tuple[str, str, int], tuple[str, ...]] = {}
        self._routes: dict[tuple[str, int, str | None], tuple[str, ...]] = {}
        self._guard = threading.Lock()
        self._aborted = False
        self._sockets: list[socket.socket] = []
        self._connecting = _Connecting()
        self.stall_floor = stall_floor
        super().__init__(*args, **kwargs)

    def pin(self, url: str, addresses: tuple[str, ...]) -> None:
        """Connect for *url*'s origin to *addresses*, in order, until one
        answers; never to a fresh lookup of its name.

        Args:
            url: Any URL of the origin.
            addresses: The vetted addresses (:func:`vet_target`). Empty only
                when local DNS gave none and a proxy applies, which then
                connects by name; a request sent without a proxy is refused
                when its pin has none.
        """
        if isinstance(addresses, str):
            raise TypeError("pin takes the tuple of vetted addresses")
        scheme, host, port = origin = _origin(url)
        self._pins[origin] = tuple(addresses)
        if not addresses:
            return
        # What the connection knows of itself: the pool's host (the first
        # address, as urllib3 lowers it), the port, and the name it checks
        # the certificate against.
        sni = host if scheme == "https" else None
        self._routes[(addresses[0].lower(), port, sni)] = tuple(addresses)

    def abort(self) -> None:
        """Stop every request on this adapter, from another thread.

        Each socket it opened is shut down, which ends a receive blocked on
        it at once (closing a socket another thread is blocked on does not
        wake that thread on Linux), and no new connection is made.
        """
        with self._guard:
            self._aborted = True
            sockets = list(self._sockets)
        for sock in sockets:
            if sock.fileno() == -1:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError as exc:
                # Closed by its connection in the meantime, or reset by the
                # peer: either way nothing is left reading it.
                logger.warning(
                    "A stopped request's socket could not be shut down, as it "
                    "was already closing: %s",
                    exc,
                )

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        # requests picks the proxy here and does not pass it on to
        # build_connection_pool_key_attributes, which must know.
        self._connecting.proxied = select_proxy(request.url, proxies) is not None
        try:
            return super().get_connection_with_tls_context(
                request, verify, proxies, cert
            )
        finally:
            self._connecting.proxied = False

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        addresses = self._pins.get(_origin(request.url))
        if addresses is None:
            # Refused, never resolved afresh: a name nothing checked could
            # lead anywhere.
            raise TargetRefused(
                "The link's host was not checked, so it was not followed."
            )
        if self._connecting.proxied:
            # The proxy is asked to CONNECT to the name and dials it itself:
            # it owns egress. The name was still checked wherever local DNS
            # answered (_vet).
            return host_params, pool_kwargs
        if not addresses:
            # Pinned with none only because a proxy was to resolve the name;
            # without one, nothing was checked to connect to.
            raise TargetRefused(
                "The link's host could not be resolved, so it was not followed."
            )
        # SNI and the certificate check stay on the name; only the socket
        # goes to a vetted address.
        pool_kwargs["server_hostname"] = host_params["host"]
        host_params["host"] = addresses[0]
        return host_params, pool_kwargs

    def add_headers(self, request, **kwargs) -> None:
        super().add_headers(request, **kwargs)
        # urllib3 would otherwise name the pool's host, the IP.
        request.headers["Host"] = urlsplit(request.url).netloc

    def init_poolmanager(self, *args, **kwargs) -> None:
        super().init_poolmanager(*args, **kwargs)
        _wrap_pools(
            self.poolmanager,
            functools.partial(_watched_pool, adapter=self, routed=True),
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        fresh = proxy not in self.proxy_manager
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        if fresh:
            _wrap_pools(
                manager, functools.partial(_watched_pool, adapter=self, routed=False)
            )
        return manager

    def _route(
        self, dns_host: str, port: int, server_hostname: str | None
    ) -> tuple[str, ...] | None:
        """The vetted addresses for a connection to *dns_host*, or None."""
        sni = server_hostname.lower() if server_hostname else None
        return self._routes.get((dns_host.lower(), port, sni))

    def _check_running(self, conn) -> None:
        """Raise, as urllib3 would for a dial that failed, once aborted."""
        if self._aborted:
            raise urllib3_errors.NewConnectionError(conn, "The request was stopped.")

    def _dialled(self, conn, sock: socket.socket) -> socket.socket:
        """*sock*, just connected, unless :meth:`abort` ran while it was
        connecting: then it is closed before a TLS handshake or a byte of the
        request, and the dial raises as a failed one would. A connect itself
        is not interrupted (its own timeout ends it)."""
        if self._aborted:
            sock.close()
        self._check_running(conn)
        return sock

    def _watch(self, conn, sock: socket.socket) -> None:
        """Keep *sock* for :meth:`abort`; raise if that has already run."""
        with self._guard:
            self._sockets.append(sock)
        self._check_running(conn)


class _DeadlineAdapter(HTTPAdapter):
    """An adapter, for one request that follows no redirect, whose
    connections read by *end*."""

    def __init__(self, end: float, receive_s: float) -> None:
        # Set first: HTTPAdapter.__init__ calls init_poolmanager.
        self._end = end
        self._receive_s = receive_s
        super().__init__()

    def init_poolmanager(self, *args, **kwargs) -> None:
        super().init_poolmanager(*args, **kwargs)
        _wrap_pools(
            self.poolmanager,
            functools.partial(_deadline_pool, end=self._end, receive_s=self._receive_s),
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        fresh = proxy not in self.proxy_manager
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        if fresh:
            _wrap_pools(
                manager,
                functools.partial(
                    _deadline_pool, end=self._end, receive_s=self._receive_s
                ),
            )
        return manager


class _PinnedDeadlineAdapter(PinnedAdapter, _DeadlineAdapter):
    """A :class:`PinnedAdapter` whose connections read by *end*: for one
    request that follows redirects. Its pools are the deadline's, watched and
    routed around them."""


class HandRedirectSession(requests.Session):
    """A ``requests.Session`` for a caller that follows redirects by hand.

    Asked with ``allow_redirects=False``, ``Session.send`` still walks
    ``resolve_redirects`` once to note the next request on ``Response.next``,
    and that walk reads the redirect's whole body into memory, under no cap
    and before the caller sees a byte, then decodes its Location, raising a
    bare ``UnicodeDecodeError`` for bytes that are not UTF-8. Here the walk
    finds nothing: the body is left for the caller to read or close, and
    :meth:`redirect_target` reads the Location. ``Response.next`` is always
    None, and ``allow_redirects=True`` follows nothing.
    """

    def resolve_redirects(self, resp, req, *args, **kwargs):
        """Follow nothing and read nothing (see the class docstring)."""
        return iter(())

    def redirect_target(self, response: requests.Response) -> str | None:
        """The address a 301, 302, 303, 307 or 308 *response* names, read as
        ``requests`` reads it (its Location bytes as UTF-8, not
        ``http.client``'s Latin-1 text of them); None for any other answer
        or no Location.

        Raises:
            RedirectRefused: The Location is not UTF-8. The message names
                only the host that sent it.
        """
        try:
            return self.get_redirect_target(response)
        except UnicodeDecodeError as exc:
            raise RedirectRefused(
                f"{urlsplit(response.url).hostname or 'The host'} sent a "
                "redirect whose address is not UTF-8."
            ) from exc


def _judged(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address *address* leads to, for :func:`_is_public` to judge; None
    for a form refused whatever it carries.

    An IPv4-mapped address is its IPv4: some Python releases call every
    mapped address non-global and others judge the IPv4, and this makes all
    of them judge the IPv4. An address under the NAT64 well-known prefix is
    the IPv4 a translator connects it to (RFC 6052): ``ipaddress`` calls the
    whole prefix global, which would pass ``64:ff9b::7f00:1``, loopback.
    The local-use NAT64 prefix, IPv4-compatible and 6to4 addresses are
    refused (``_REFUSED_WRAPPERS``).
    """
    if isinstance(address, ipaddress.IPv4Address):
        return address
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    if any(address in wrapper for wrapper in _REFUSED_WRAPPERS):
        return None
    return address


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Global (``is_global``: not loopback, private, link-local, shared,
    documentation or unspecified) and not multicast, nor the space
    ``is_global`` still passes.

    For IPv4 that is the deprecated 6to4 relay anycast block
    ``192.88.99.0/24`` (RFC 7526, ``_SIX_TO_FOUR_RELAY_V4``): a relay there
    is whichever one the network routes to, not a host on the internet.

    For IPv6 it is site-local ``fec0::/10`` (``is_site_local``: deprecated by
    RFC 3879, still routed inside a network that kept it), reserved space
    (``is_reserved``: IANA's unassigned blocks, among them the SIIT prefix
    ``::ffff:0:0/96`` that a translator turns into an IPv4 (RFC 6052 section
    2.1) and SRv6's ``5f00::/16``, RFC 9602), and RFC 9637's documentation
    prefix ``3fff::/20`` (``_DOCUMENTATION_V6``), which ``ipaddress`` does not
    know.

    *address* is the one :func:`_judged` unwrapped, so a NAT64 address is
    judged by the IPv4 it carries: ``64:ff9b::/96``, where DNS64 puts every
    public IPv4 on an IPv6-only network, is itself reserved space.
    """
    if not address.is_global or address.is_multicast:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address not in _SIX_TO_FOUR_RELAY_V4
    return not (
        address.is_site_local or address.is_reserved or address in _DOCUMENTATION_V6
    )


def vet_redirect_target(
    current_url: str, location: str | None, proxies: dict | None = None
) -> tuple[str, tuple[str, ...]]:
    """``(url, addresses)``: the absolute URL a redirect from *current_url*
    leads to, once checked, and the resolved addresses to connect to.

    For hosts that legitimately redirect (model hosts send downloads on to a
    CDN). The target must be https, carry no user name, and every address
    its host resolves to must be public: not loopback, private, link-local,
    shared, multicast or unspecified, judging an IPv4-mapped or NAT64
    address by the IPv4 it carries, and refusing the other IPv6 forms that
    carry one (:func:`_judged`). OWASP SSRF prevention:
    https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html

    The host is resolved once, here, under the name ``requests`` will send
    (a non-ASCII host in its ``xn--`` form): :func:`bounded_request`
    connects to the addresses returned, each in turn until one answers, so a
    name that resolves differently a moment later (DNS rebinding) is never
    looked up again.

    *proxies*, as the request will be sent with them (:func:`proxy_for`):
    when a proxy applies, a host local DNS cannot resolve is let through with
    no addresses, for the proxy to resolve; any address local DNS does give
    is still judged.

    Raises:
        RedirectRefused: no Location, or a target that fails a check above.
            The message names only the parsed host.
    """
    if not location:
        raise RedirectRefused("The redirect named no address.")
    target = urljoin(current_url, location.strip())
    return target, _vet(target, "redirect", RedirectRefused, proxies)


def vet_target(url: str, proxies: dict | None = None) -> tuple[str, ...]:
    """The addresses to connect to for *url*, in the order they resolved,
    once checked by the rules :func:`vet_redirect_target` holds a redirect
    to: https, a plain host, and every address it resolves to public.

    For the first request of one that follows redirects and pins every hop
    (:class:`PinnedAdapter`): :func:`bounded_request` with *redirects*, or a
    caller that follows them itself. The first connection, too, then goes to
    an address that was checked and is never looked up again. *proxies* as
    :func:`vet_redirect_target` takes them.

    Raises:
        TargetRefused: *url* fails a check. The message names only the
            parsed host.
    """
    return _vet(url, "link", TargetRefused, proxies)


def _vet(
    url: str, noun: str, refused: type[TargetRefused], proxies: dict | None
) -> tuple[str, ...]:
    """Every address *url*'s host resolves to, in order, each checked;
    *refused*, its message naming the *noun*, when a check fails.

    The scheme, user name and host are read as ``requests`` will send them
    (:func:`wire_split`), so the URL checked and the URL connected for are
    one: ``urlsplit`` reads ``https://evil.example\\.ok.example/`` as a host
    under ``ok.example`` where ``requests`` asks ``evil.example``, and
    Python's own IDNA codec would resolve ``straße`` as ``strasse``, where
    ``requests`` sends ``xn--strae-oqa``. A user name either parse finds is
    refused.

    Empty when local DNS gives no answer and a proxy applies to *url*
    (:func:`proxy_for`): the proxy resolves the name. Any address local DNS
    does give is judged, proxy or not.
    """
    plain = f"The {noun} named no plain host, so it was not followed."
    try:
        wire_url = _wire_url(url)
        wire = urlsplit(wire_url)
        raw = urlsplit(url)
    except (requests.RequestException, ValueError) as exc:
        raise refused(plain) from exc
    host = wire.hostname or ""
    if wire.scheme != "https":
        raise refused(
            f"The {noun} to {host or 'an unnamed host'} is not https, "
            "so it was not followed."
        )
    named_user = any(
        part is not None
        for part in (wire.username, wire.password, raw.username, raw.password)
    )
    if not host or named_user:
        raise refused(plain)
    unresolved = f"The {noun} to {host} could not be resolved, so it was not followed."
    failure: Exception | None = None
    try:
        port = wire.port or 443
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as exc:
        infos, failure = [], exc
    if not infos:
        if proxy_for(wire_url, proxies) is None:
            raise refused(unresolved) from failure
        logger.info(
            "%s did not resolve here (%s); the %s goes through a proxy, which "
            "resolves it.",
            host,
            failure or "no addresses",
            noun,
        )
        return ()
    for info in infos:
        judged = _judged(ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]))
        if judged is None or not _is_public(judged):
            raise refused(
                f"The {noun} to {host} leads to a private or local address, "
                "so it was not followed."
            )
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


def _decoder(url: str, encoding: str | None):
    """A decompressor for *encoding*, None for identity; anything else is
    refused (RFC 9110 section 8.4: deflate is the zlib format)."""
    name = (encoding or "identity").strip().lower()
    if name == "identity":
        return None
    if name in ("gzip", "x-gzip"):
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if name == "deflate":
        return zlib.decompressobj()
    raise requests.exceptions.ContentDecodingError(
        f"{url} sent its body in an encoding that is not read here"
    )


def _receive(response: requests.Response, url: str, end: float) -> bytes:
    """One read of *response*'s undecoded body, raised as ``requests`` would.

    Reading the raw stream directly skips ``requests``' translation of
    urllib3's errors; this does the same, so a caller's
    ``except requests.RequestException`` still covers every failure.
    """
    try:
        return response.raw.read(READ_CHUNK_BYTES, decode_content=False)
    except urllib3_errors.ReadTimeoutError as exc:
        if time.monotonic() >= end:
            raise ReadDeadlineExceeded(f"{url} was still sending its body") from exc
        raise requests.exceptions.ReadTimeout(f"{url}: {exc}") from exc
    except urllib3_errors.SSLError as exc:
        raise requests.exceptions.SSLError(f"{url}: {exc}") from exc
    except urllib3_errors.ProtocolError as exc:
        raise requests.exceptions.ChunkedEncodingError(f"{url}: {exc}") from exc


def _read_body(
    response: requests.Response,
    url: str,
    end: float,
    *,
    limit: int,
    truncate: bool,
    sink: Callable[[bytes], None] | None,
) -> tuple[bytes, bool]:
    """``(body, truncated)``: the decoded body, held to *limit* bytes received
    and *limit* bytes decoded.

    Past *limit*, a *truncate* read stops and says so; any other raises
    :class:`ResponseTooLarge`. With a *sink*, decoded bytes go to it as they
    come and the body returned is empty.
    """
    decoder = _decoder(url, response.headers.get("Content-Encoding"))
    kept = bytearray()
    decoded = 0
    received = 0

    def emit(piece: bytes) -> bool:
        """Count *piece* in; False once a truncating read is full."""
        nonlocal decoded
        if decoded + len(piece) > limit:
            if truncate:
                kept.extend(piece[: limit - decoded])
                return False
            raise ResponseTooLarge(f"{url} sent more than the {limit} bytes allowed")
        decoded += len(piece)
        if sink is not None:
            sink(piece)
        else:
            kept.extend(piece)
        return True

    while True:
        if time.monotonic() >= end:
            raise ReadDeadlineExceeded(f"{url} was still sending its body")
        chunk = _receive(response, url, end)
        if not chunk:
            break
        received += len(chunk)
        if received > limit:
            if truncate:
                return bytes(kept), True
            raise ResponseTooLarge(f"{url} sent more than the {limit} bytes allowed")
        if decoder is None:
            if not emit(chunk):
                return bytes(kept), True
            continue
        if decoder.eof:
            raise requests.exceptions.ContentDecodingError(
                f"{url} sent bytes after the end of its compressed body"
            )
        data = chunk
        while data:
            # max_length is at least 1 (0 would mean unlimited), and one byte
            # over what is left, so a body that will not fit is seen at once.
            if not emit(decoder.decompress(data, limit - decoded + 1)):
                return bytes(kept), True
            data = decoder.unconsumed_tail
            if time.monotonic() >= end:
                raise ReadDeadlineExceeded(f"{url} was still sending its body")
        if decoder.unused_data:
            raise requests.exceptions.ContentDecodingError(
                f"{url} sent bytes after the end of its compressed body"
            )
    if decoder is not None and not decoder.eof:
        raise requests.exceptions.ContentDecodingError(
            f"{url} sent a compressed body that ended early"
        )
    if decoder is None:
        _require_declared_length(response, url, received)
    return bytes(kept), False


def _require_declared_length(
    response: requests.Response, url: str, received: int
) -> None:
    """Raise when an identity body ended short of its ``Content-Length``.

    urllib3 2.x raises this itself (``enforce_content_length``); this is the
    check where the bytes are kept, so a stream that ends quietly is not
    taken as whole. Not for an answer with no body (a HEAD, 1xx, 204 or
    304, RFC 9112 section 6.3), nor one with a ``Transfer-Encoding``, which
    overrides any ``Content-Length``.
    """
    if (
        response.request.method == "HEAD"
        or response.status_code < 200
        or response.status_code in (204, 304)
        or "Transfer-Encoding" in response.headers
    ):
        return
    declared = response.headers.get("Content-Length", "").strip()
    if declared.isdigit() and received < int(declared):
        raise requests.exceptions.ChunkedEncodingError(
            f"{url} sent {received} of the {declared} bytes it declared"
        )


def bounded_request(
    method: str,
    url: str,
    *,
    deadline_s: float,
    max_bytes: int,
    timeout: float,
    proxies: dict | None = None,
    params: dict | None = None,
    json: Any = None,
    data: Any = None,
    files: Any = None,
    headers: dict[str, str] | None = None,
    sink: Callable[[bytes], None] | None = None,
    redirects: int = 0,
) -> BoundedResponse:
    """Send one request and read its whole answer within *deadline_s*.

    Args:
        method: The HTTP method.
        url: The address to ask.
        deadline_s: How long the whole exchange may take, from this call:
            connect, TLS handshake, sending, status line, headers and body,
            every redirect hop included. Not the DNS lookups, which take no
            timeout (see the module docstring).
        max_bytes: The most body kept, received and decoded alike; one byte
            more is :class:`ResponseTooLarge`. A 2xx answer declaring a
            larger Content-Length is refused before its body is read.
        timeout: The longest any one connect or receive may take.
        proxies: Passed to ``requests``; callers that must not take the
            environment's proxy pass ``None`` for every scheme.
        params, json, data, files: As ``requests`` takes them. A ``data``
            file object is sent under one deadline check, so give bytes.
        headers: Extra request headers. ``Accept-Encoding`` is always
            ``identity``.
        sink: Takes a 2xx body's decoded bytes as they arrive, instead of
            their being kept in memory.
        redirects: How many redirects to follow, at most ``MAX_REDIRECTS``,
            each checked by :func:`vet_redirect_target`, connected to the
            addresses that check resolved (each in turn until one answers),
            and asked with GET and no body: *params*, *json*, *data* and
            *files* go to *url* only. A hop to another origin carries only
            the ``_PORTABLE_HEADERS`` of *headers*, never a credential. With
            any, *url* itself is checked and pinned the same way
            (:func:`vet_target`) before anything is sent. 0 (the default)
            follows none and checks nothing: a redirect comes back as its
            status. That suits an address the owner typed (a ComfyUI) and a
            fixed https host in the code, which rests on the certificate
            check (see the module docstring).

    Raises:
        ReadDeadlineExceeded: The answer was not in by the deadline.
        ResponseTooLarge: The body, or its declared length, is over the cap.
        TargetRefused: With *redirects*, *url* failed its check, or a
            request found no pin for its origin; nothing was sent.
        RedirectRefused: A redirect failed its check (or named its target
            in bytes that are not UTF-8), or there were more than
            *redirects*.
        requests.RequestException: Anything else ``requests`` raises.
    """
    if not 0 <= redirects <= MAX_REDIRECTS:
        raise ValueError(f"redirects must be 0 to {MAX_REDIRECTS}")
    if redirects and method.upper() != "GET":
        raise ValueError("only a GET follows redirects")
    end = time.monotonic() + deadline_s
    sent_headers = {**(headers or {}), "Accept-Encoding": "identity"}
    current = url
    try:
        # Not requests.Session: it reads a redirect's whole body, uncapped.
        with HandRedirectSession() as session:
            pinned = _PinnedDeadlineAdapter(end, timeout) if redirects else None
            adapter = _DeadlineAdapter(end, timeout) if pinned is None else pinned
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            if pinned is not None:
                # A request that may be sent on to other hosts starts at a
                # checked one: the first link gets the same check and pin as
                # each hop.
                pinned.pin(url, vet_target(url, proxies))
            for hop in range(redirects + 1):
                connect_s = min(timeout, max(end - time.monotonic(), 0.001))
                first = hop == 0
                response = session.request(
                    method,
                    current,
                    params=params if first else None,
                    json=json if first else None,
                    data=data if first else None,
                    files=files if first else None,
                    headers=sent_headers,
                    timeout=(connect_s, timeout),
                    allow_redirects=False,
                    proxies=proxies,
                    stream=True,
                )
                with response:
                    redirected = response.status_code in _REDIRECT_STATUSES
                    if redirected and pinned is not None:
                        if hop == redirects:
                            raise RedirectRefused(
                                f"{url} redirected more than {redirects} times"
                            )
                        target, addresses = vet_redirect_target(
                            current, session.redirect_target(response), proxies
                        )
                        pinned.pin(target, addresses)
                        if _origin(target) != _origin(current):
                            sent_headers = {
                                name: value
                                for name, value in sent_headers.items()
                                if name.lower() in _PORTABLE_HEADERS
                            }
                        current = target
                        continue
                    return _answer(response, current, end, max_bytes, sink)
    except (ResponseTooLarge, ReadDeadlineExceeded, TargetRefused):
        raise
    except requests.RequestException as exc:
        if time.monotonic() >= end:
            raise ReadDeadlineExceeded(
                f"{current} was still answering after {deadline_s:g} s"
            ) from exc
        raise
    raise AssertionError("unreachable: every hop returns or raises")


def _answer(
    response: requests.Response,
    url: str,
    end: float,
    max_bytes: int,
    sink: Callable[[bytes], None] | None,
) -> BoundedResponse:
    """Read *response* into a :class:`BoundedResponse`."""
    headers = dict(response.headers)
    if not 200 <= response.status_code < 300:
        # Not an answer to use: read only enough of it to log.
        body, truncated = _read_body(
            response, url, end, limit=ERROR_BODY_BYTES, truncate=True, sink=None
        )
        return BoundedResponse(url, response.status_code, body, headers, truncated)
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ResponseTooLarge(
            f"{url} declared {declared} bytes, more than the {max_bytes} allowed"
        )
    body, _ = _read_body(response, url, end, limit=max_bytes, truncate=False, sink=sink)
    return BoundedResponse(url, response.status_code, body, headers)


def bounded_get(url: str, **kwargs) -> BoundedResponse:
    """:func:`bounded_request` with GET."""
    return bounded_request("GET", url, **kwargs)
