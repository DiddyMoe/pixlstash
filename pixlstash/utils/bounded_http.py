"""One outbound HTTP request, its whole answer read within a deadline and a cap.

``requests``' ``timeout`` bounds each connect and each receive, not the
exchange: a server that sends a byte just inside every timeout holds the
request open without limit, and one that sends without end fills memory.
:func:`bounded_request` gives the whole exchange one deadline (connect, TLS
handshake, sending, status line, headers and body) by setting the socket
timeout to what is left before every receive, and holds the body to a byte cap
as it arrives. A redirect is never followed: it comes back as its status.

The DNS lookup takes no timeout (``getaddrinfo`` has none), so it is outside
the deadline. The body is asked for, and read, undecoded
(``Accept-Encoding: identity``); an answer in any other encoding is refused.
"""

import functools
import http.client
import io
import ipaddress
import json as jsonlib
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3 import exceptions as urllib3_errors

#: The most one receive asks for.
READ_CHUNK_BYTES = 64 * 1024
#: The most of a non-2xx answer's body that is read, for the caller's log.
ERROR_BODY_BYTES = 64 * 1024


class ResponseTooLarge(requests.RequestException):
    """The body ran past the cap the caller set."""


class ReadDeadlineExceeded(requests.Timeout):
    """The answer was still arriving when the caller's deadline passed."""


class RedirectRefused(requests.RequestException):
    """A redirect led somewhere :func:`vet_redirect_target` does not allow."""


class _DeadlinePassed(TimeoutError):
    """Raised inside a socket read; urllib3 and requests turn it into their
    own read timeout, which :func:`bounded_request` reports as
    :class:`ReadDeadlineExceeded`."""


@dataclass(frozen=True)
class BoundedResponse:
    """What a bounded request read: its status, headers and body.

    For a 2xx answer *content* is the whole body. For any other, it is at most
    ``ERROR_BODY_BYTES`` of the body, *truncated* saying whether there was
    more: enough to log, never more.
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


def _wrap_pools(manager, wrap: Callable[[type], type]) -> None:
    """Make every pool *manager* makes from now on of ``wrap(its class)``."""
    manager.pool_classes_by_scheme = {
        scheme: wrap(pool_cls)
        for scheme, pool_cls in manager.pool_classes_by_scheme.items()
    }


class _DeadlineAdapter(HTTPAdapter):
    """An adapter, for one request, whose connections read by *end*, direct
    or through a proxy."""

    def __init__(self, end: float, receive_s: float) -> None:
        # Set first: HTTPAdapter.__init__ calls init_poolmanager.
        self._wrap = functools.partial(_deadline_pool, end=end, receive_s=receive_s)
        super().__init__()

    def init_poolmanager(self, *args, **kwargs) -> None:
        super().init_poolmanager(*args, **kwargs)
        _wrap_pools(self.poolmanager, self._wrap)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        fresh = proxy not in self.proxy_manager
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        if fresh:
            _wrap_pools(manager, self._wrap)
        return manager


class HandRedirectSession(requests.Session):
    """A ``requests.Session`` that follows no redirect and reads none.

    Asked with ``allow_redirects=False``, ``Session.send`` still walks
    ``resolve_redirects`` once to note the next request on ``Response.next``,
    and that walk reads the redirect's whole body into memory, under no cap.
    Here the walk finds nothing, so the body is left to :func:`_answer`.
    """

    def resolve_redirects(self, resp, req, *args, **kwargs):
        return iter(())


def _wire_url(url: str) -> str:
    """*url* as ``requests`` sends it: a non-ASCII host in its ``xn--`` form,
    the escapes requoted.

    Raises:
        requests.RequestException: ``requests`` would not send *url*.
    """
    prepared = requests.PreparedRequest()
    prepared.prepare_url(url, None)
    return prepared.url or ""


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Global (not loopback, private, link-local, shared, documentation or
    unspecified) and not multicast: ``is_global`` alone admits multicast."""
    return address.is_global and not address.is_multicast


def vet_redirect_target(current_url: str, location: str) -> str:
    """The absolute URL a redirect from *current_url* to *location* leads to,
    once checked: https, a host and no user name, and every address the host
    resolves to public (:func:`_is_public`).

    The scheme, host and user name are read as ``requests`` will send them
    (:func:`_wire_url`), so the host checked is the host asked: ``urlsplit``
    reads ``https://10.0.0.1\\@example.com/`` as ``example.com`` where
    ``requests`` asks ``10.0.0.1``. A user name either parse finds is refused.
    The request resolves the host again when it connects, so a name that
    answers with another address the second time is not caught here;
    :func:`get_vetted` checks the address it connects to.

    Raises:
        RedirectRefused: a check failed. The message names only the host.
    """
    target = urljoin(current_url, location.strip())
    plain = "The redirect named no plain host, so it was not followed."
    try:
        wire = urlsplit(_wire_url(target))
        raw = urlsplit(target)
    except (requests.RequestException, ValueError) as exc:
        raise RedirectRefused(plain) from exc
    host = wire.hostname or ""
    if wire.scheme != "https":
        raise RedirectRefused(
            f"The redirect to {host or 'an unnamed host'} is not https, "
            "so it was not followed."
        )
    named_user = any(
        part is not None
        for part in (wire.username, wire.password, raw.username, raw.password)
    )
    if not host or named_user:
        raise RedirectRefused(plain)
    try:
        infos = socket.getaddrinfo(host, wire.port or 443, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as exc:
        raise RedirectRefused(
            f"The redirect to {host} could not be resolved, so it was not followed."
        ) from exc
    for info in infos:
        if not _is_public(ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])):
            raise RedirectRefused(
                f"The redirect to {host} leads to a private or local address, "
                "so it was not followed."
            )
    return target


class _PrivatePeer(Exception):
    """A connect reached an address :func:`_is_public` refuses. Not an
    ``OSError``, as :class:`RedirectRefused` is: urllib3 and requests would
    turn that into a ``ConnectionError``, and they pass this up unchanged."""


def _public_peer_pool(pool_cls: type) -> type:
    """*pool_cls* with connections that close and raise :class:`_PrivatePeer`
    when the address connected to is not public."""

    class _Connection(pool_cls.ConnectionCls):
        def _new_conn(self):
            sock = super()._new_conn()
            peer = ipaddress.ip_address(str(sock.getpeername()[0]).split("%", 1)[0])
            if not _is_public(peer):
                sock.close()
                raise _PrivatePeer(self.host)
            return sock

    class _Pool(pool_cls):
        ConnectionCls = _Connection

    return _Pool


def get_vetted(url: str, **kwargs) -> requests.Response:
    """``requests.get(url, **kwargs)`` for a URL :func:`vet_redirect_target`
    passed, refused when the address it connects to is not public: the vet
    and the connect each resolve the host, and the answers can differ.

    A connection through a proxy reaches the proxy, so it is not checked.

    Raises:
        RedirectRefused: the address connected to is private or local.
    """
    adapter = HTTPAdapter()
    # The direct pools only: a proxy manager's peer is the proxy.
    _wrap_pools(adapter.poolmanager, _public_peer_pool)
    try:
        with requests.Session() as session:
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            return session.get(url, **kwargs)
    except _PrivatePeer as exc:
        raise RedirectRefused(
            f"The redirect to {exc.args[0]} leads to a private or local address, "
            "so it was not followed."
        ) from exc


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
) -> tuple[bytes, bool]:
    """``(body, truncated)``: the body, held to *limit* bytes.

    Past *limit*, a *truncate* read stops and says so; any other raises
    :class:`ResponseTooLarge`.
    """
    kept = bytearray()
    received = 0
    while True:
        if time.monotonic() >= end:
            raise ReadDeadlineExceeded(f"{url} was still sending its body")
        chunk = _receive(response, url, end)
        if not chunk:
            break
        if received + len(chunk) > limit:
            if truncate:
                kept.extend(chunk[: limit - received])
                return bytes(kept), True
            raise ResponseTooLarge(f"{url} sent more than the {limit} bytes allowed")
        received += len(chunk)
        kept.extend(chunk)
    _require_declared_length(response, url, received)
    return bytes(kept), False


def _require_declared_length(
    response: requests.Response, url: str, received: int
) -> None:
    """Raise when a body ended short of its ``Content-Length``, so a stream
    that ends quietly is not taken as whole (urllib3 1.26 does not check).

    Not for an answer with no body (a HEAD, 1xx, 204 or 304), nor one with a
    ``Transfer-Encoding``, which overrides any ``Content-Length``.
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
    json: Any = None,
    data: Any = None,
    headers: dict[str, str] | None = None,
) -> BoundedResponse:
    """Send one request and read its whole answer within *deadline_s*.

    Args:
        method: The HTTP method.
        url: The address to ask.
        deadline_s: How long the whole exchange may take, from this call:
            connect, TLS handshake, sending, status line, headers and body.
            Not the DNS lookup (see the module docstring).
        max_bytes: The most body kept; one byte more is
            :class:`ResponseTooLarge`. A 2xx answer declaring a larger
            Content-Length is refused before its body is read.
        timeout: The longest any one connect or receive may take.
        proxies: Passed to ``requests``; callers that must not take the
            environment's proxy pass ``None`` for every scheme.
        json, data: As ``requests`` takes them. A ``data`` file object is
            sent under one deadline check, so give bytes.
        headers: Extra request headers. ``Accept-Encoding`` is always
            ``identity``.

    Raises:
        ReadDeadlineExceeded: The answer was not in by the deadline.
        ResponseTooLarge: The body, or its declared length, is over the cap.
        requests.RequestException: Anything else ``requests`` raises.
    """
    end = time.monotonic() + deadline_s
    sent_headers = {**(headers or {}), "Accept-Encoding": "identity"}
    try:
        with HandRedirectSession() as session:
            adapter = _DeadlineAdapter(end, timeout)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            connect_s = min(timeout, max(end - time.monotonic(), 0.001))
            response = session.request(
                method,
                url,
                json=json,
                data=data,
                headers=sent_headers,
                timeout=(connect_s, timeout),
                allow_redirects=False,
                proxies=proxies,
                stream=True,
            )
            with response:
                return _answer(response, url, end, max_bytes)
    except (ResponseTooLarge, ReadDeadlineExceeded):
        raise
    except requests.RequestException as exc:
        if time.monotonic() >= end:
            raise ReadDeadlineExceeded(
                f"{url} was still answering after {deadline_s:g} s"
            ) from exc
        raise


def _answer(
    response: requests.Response,
    url: str,
    end: float,
    max_bytes: int,
) -> BoundedResponse:
    """Read *response* into a :class:`BoundedResponse`."""
    headers = dict(response.headers)
    if not 200 <= response.status_code < 300:
        # Not an answer to use: read only enough of it to log.
        body, truncated = _read_body(
            response, url, end, limit=ERROR_BODY_BYTES, truncate=True
        )
        return BoundedResponse(url, response.status_code, body, headers, truncated)
    encoding = (response.headers.get("Content-Encoding") or "identity").strip()
    if encoding.lower() != "identity":
        raise requests.exceptions.ContentDecodingError(
            f"{url} sent its body in an encoding that is not read here"
        )
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ResponseTooLarge(
            f"{url} declared {declared} bytes, more than the {max_bytes} allowed"
        )
    body, _ = _read_body(response, url, end, limit=max_bytes, truncate=False)
    return BoundedResponse(url, response.status_code, body, headers)


def bounded_get(url: str, **kwargs) -> BoundedResponse:
    """:func:`bounded_request` with GET."""
    return bounded_request("GET", url, **kwargs)
