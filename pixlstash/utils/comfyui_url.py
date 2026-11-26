"""The one spelling of a ComfyUI address, and the refusal of an ambiguous one.

A low module so both the ComfyUI service and the arbiter's ComfyUI probe
(``pixlstash.resources.comfyui_tenant``, which the service imports) can
normalise an address before using it.

The normaliser is a whitelist, not a list of known-bad marks. URL parsers
disagree on everything outside the plain form: ``urlsplit`` and urllib3 read
``http://evil.example\\@127.0.0.1/`` as different hosts, the WHATWG host parser
and glibc read ``2130706433``, ``0x7f.1``, ``127.1`` and ``0177.0.0.1`` as
127.0.0.1 while ``ipaddress`` refuses them, and urllib3 sends ``:0`` to port 80.
So an address is accepted only in a form every one of them reads the same way:

- scheme ``http`` or ``https``;
- host: a bracketed IPv6 literal without a zone id (RFC 9844: zone ids are
  local UI only, never on the wire), a canonical dotted-quad IPv4, or a DNS
  name of letters, digits and hyphens (RFC 1123 section 2.1) with no trailing
  dot. A name whose last label is a number must be a canonical IPv4 (the
  WHATWG "ends in a number" rule), so no other spelling of an address passes;
- port: 1 to 65535 when a colon is present;
- path: plain segments of unreserved characters (RFC 3986 section 2.3).

Case (RFC 4343), leading zeros in the port, the IPv6 spelling and the scheme's
default port are canonicalised, since each has exactly one meaning. Everything
else is refused, since a guess at what was meant is how the parsers disagree.
"""

import ipaddress
import re
import threading

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}
# RFC 1123 section 2.1: letters, digits and hyphens, no hyphen at either end.
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
# WHATWG URL "ends in a number": decimal, or 0x followed by hex digits.
_NUMBER_LABEL = re.compile(r"[0-9]+|0x[0-9a-f]*")
_PORT = re.compile(r"[0-9]{1,5}")
# RFC 3986 section 2.3 unreserved characters.
_SEGMENT = re.compile(r"[A-Za-z0-9._~-]+")
_MAX_NAME = 253

NOT_AN_ADDRESS = (
    "A ComfyUI address is http:// or https://, a host, an optional port and "
    "an optional path, such as http://127.0.0.1:8188."
)
NOT_PLAIN = (
    "A ComfyUI address cannot contain spaces, user names, backslashes, "
    "queries, fragments or percent signs."
)
BAD_HOST = (
    "The host in that ComfyUI address is not a host name, an IPv4 address in "
    "dotted form or a bracketed IPv6 address."
)
BAD_PORT = "The port in that ComfyUI address must be a number from 1 to 65535."
BAD_PATH = (
    "The path in that ComfyUI address may use only letters, digits and . _ ~ -, "
    "with no . or .. parts."
)
SAVED_ADDRESS_INVALID = (
    "The saved ComfyUI address is not valid, so nothing was sent to it. "
    "Set the address again in Settings."
)


class InvalidComfyUIAddress(ValueError):
    """A typed ComfyUI address the normaliser refuses; the message is fixed
    text that never repeats the address."""


class SavedComfyUIAddressInvalid(Exception):
    """A saved ComfyUI address no longer passes the normaliser.

    Not a ``ValueError``: it is not the caller's input, so it must not be
    reported as a 400 by a handler that maps ``ValueError`` to one.
    """

    def __init__(self, connection_id: str | None):
        super().__init__(SAVED_ADDRESS_INVALID)
        self.connection_id = connection_id


def _host(text: str) -> str:
    """The canonical host for the authority's host part, else refused."""
    if text.startswith("["):
        if not text.endswith("]"):
            raise InvalidComfyUIAddress(BAD_HOST)
        try:
            address = ipaddress.IPv6Address(text[1:-1])
        except ValueError:
            raise InvalidComfyUIAddress(BAD_HOST) from None
        if address.ipv4_mapped:
            # RFC 5952 section 5, and one spelling on every Python: 3.13
            # changed ``str()`` of a mapped address to this dotted form.
            return f"[::ffff:{address.ipv4_mapped}]"
        return f"[{address.compressed}]"
    host = text.lower()
    labels = host.split(".")
    if len(host) > _MAX_NAME or not all(_LABEL.fullmatch(lab) for lab in labels):
        raise InvalidComfyUIAddress(BAD_HOST)
    if _NUMBER_LABEL.fullmatch(labels[-1]):
        try:
            address = ipaddress.IPv4Address(host)
        except ValueError:
            raise InvalidComfyUIAddress(BAD_HOST) from None
        if str(address) != host:
            raise InvalidComfyUIAddress(BAD_HOST)
    return host


def _split_authority(authority: str) -> tuple[str, str | None]:
    """``(host, port)``; *port* is None when there is no colon."""
    if authority.startswith("["):
        end = authority.find("]")
        if end < 0:
            raise InvalidComfyUIAddress(BAD_HOST)
        host, rest = authority[: end + 1], authority[end + 1 :]
        if rest and not rest.startswith(":"):
            raise InvalidComfyUIAddress(BAD_HOST)
        return host, rest[1:] if rest else None
    host, colon, port = authority.partition(":")
    if ":" in port:
        # An unbracketed IPv6 address, or two ports.
        raise InvalidComfyUIAddress(BAD_HOST)
    return host, port if colon else None


def normalize_comfyui_url(url: str) -> str:
    """``scheme://host[:port][/path]/`` for a plain http(s) address.

    The trailing slash is the form ``comfyui_url`` has always been saved in.

    Raises:
        InvalidComfyUIAddress: *url* is not in the plain form the module
            docstring describes. The message never repeats *url*.
    """
    text = str(url or "").strip()
    # Printable ASCII only: no space, control character or non-ASCII host.
    if not text or not all("!" <= ch <= "~" for ch in text):
        raise InvalidComfyUIAddress(NOT_AN_ADDRESS if not text else NOT_PLAIN)
    if any(mark in text for mark in "@\\?#%"):
        raise InvalidComfyUIAddress(NOT_PLAIN)
    scheme, sep, rest = text.partition("://")
    scheme = scheme.lower()
    if not sep or scheme not in _DEFAULT_PORTS:
        raise InvalidComfyUIAddress(NOT_AN_ADDRESS)
    authority, _, path = rest.partition("/")
    if not authority:
        raise InvalidComfyUIAddress(NOT_AN_ADDRESS)
    host_text, port_text = _split_authority(authority)
    host = _host(host_text)
    netloc = host
    if port_text is not None:
        if not _PORT.fullmatch(port_text) or not 1 <= int(port_text) <= 65535:
            raise InvalidComfyUIAddress(BAD_PORT)
        port = int(port_text)
        if port != _DEFAULT_PORTS[scheme]:
            netloc = f"{host}:{port}"
    segments = path.rstrip("/").split("/") if path.rstrip("/") else []
    for segment in segments:
        if not _SEGMENT.fullmatch(segment) or segment in (".", ".."):
            raise InvalidComfyUIAddress(BAD_PATH)
    tail = "".join(f"{segment}/" for segment in segments)
    return f"{scheme}://{netloc}/{tail}"


# connection id -> the refused saved address last logged for it.
_refused_logged: dict[str | None, str] = {}
_refusals_lock = threading.Lock()


def request_base(stored: str, *, connection_id: str | None) -> str:
    """The address requests to a saved ComfyUI are built on, without the
    trailing slash (paths are joined on as ``/prompt``).

    The one read path for a saved address: every saved address goes through
    the normaliser again, so a value saved before it was strict, or written by
    hand into the settings, is refused rather than sent somewhere the parsers
    disagree on.

    Raises:
        SavedComfyUIAddressInvalid: *stored* no longer passes the normaliser.
            Logged here with *connection_id*, once until that connection's
            saved address changes, since pollers (the training watcher, the
            UI) read it every few seconds; the message is fixed text.
    """
    try:
        base = normalize_comfyui_url(stored).rstrip("/")
    except InvalidComfyUIAddress as exc:
        with _refusals_lock:
            first = _refused_logged.get(connection_id) != stored
            _refused_logged[connection_id] = stored
        if first:
            logger.error(
                "Saved ComfyUI address of connection %s is not valid (%s); "
                "nothing is sent to it until it is set again",
                connection_id,
                exc,
            )
        raise SavedComfyUIAddressInvalid(connection_id) from exc
    with _refusals_lock:
        _refused_logged.pop(connection_id, None)
    return base


def forget_logged_refusals() -> None:
    """Forget which refusals were logged, so the next one is logged again.

    For tests: the guard is a process global keyed by connection id, every test
    server saves its address under the same id, and a shard runs many servers
    in one process, so a refusal logged by one test would silence the line the
    next test asserts. A server has one process to itself, so production never
    needs this.
    """
    with _refusals_lock:
        _refused_logged.clear()


def is_valid_saved(stored: str) -> bool:
    """Whether a saved address passes the normaliser, for showing the owner
    which saved connection needs its address set again."""
    try:
        normalize_comfyui_url(stored)
    except InvalidComfyUIAddress:
        return False
    return True


def is_loopback_host(host: str) -> bool:
    """Whether a normalised host names this computer without a DNS lookup.

    ``localhost`` (RFC 6761 section 6.3), or an IP literal in 127.0.0.0/8,
    ``::1`` or an IPv4-mapped loopback. For every literal the normaliser
    accepts this is the answer the link resolver gives
    (``comfyui_link_service.locate``); a DNS name other than ``localhost`` is
    not assumed to be this computer.
    """
    name = host.strip("[]").lower()
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback
