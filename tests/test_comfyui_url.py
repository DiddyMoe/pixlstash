"""The ComfyUI address normaliser, in both directions.

Every accepted address is read the same way by ``urlsplit``, by urllib3 (which
sends the request) and by ``requests`` once prepared; every refused one is a
spelling at least two of them disagree on, and the refusal never repeats it.
"""

import json
import logging
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
from urllib3.util import parse_url

from tests.network_vectors import LAN_IPV4
from pixlstash.services import comfyui_connections, comfyui_link_service
from pixlstash.utils.comfyui_url import (
    InvalidComfyUIAddress,
    SavedComfyUIAddressInvalid,
    is_loopback_host,
    is_valid_saved,
    normalize_comfyui_url,
    request_base,
)

ACCEPTED = [
    ("http://127.0.0.1:8188", "http://127.0.0.1:8188/"),
    ("http://127.0.0.1:8188/", "http://127.0.0.1:8188/"),
    ("  http://127.0.0.1:8188/  ", "http://127.0.0.1:8188/"),
    ("http://localhost:8188", "http://localhost:8188/"),
    ("http://LocalHost:8188", "http://localhost:8188/"),
    ("HTTP://Comfy.Example.COM/", "http://comfy.example.com/"),
    ("https://comfy.example.com:443", "https://comfy.example.com/"),
    ("https://comfy.example.com:80", "https://comfy.example.com:80/"),
    ("http://127.0.0.1:80/comfy/", "http://127.0.0.1/comfy/"),
    ("http://127.0.0.1:08188", "http://127.0.0.1:8188/"),
    (
        "http://comfy-box.example:8188/a/b.c_d~e-f",
        "http://comfy-box.example:8188/a/b.c_d~e-f/",
    ),
    ("http://comfy.example.com/sub//", "http://comfy.example.com/sub/"),
    ("http://192.0.2.10:65535", "http://192.0.2.10:65535/"),
    ("http://xn--bcher-kva.example/", "http://xn--bcher-kva.example/"),
    ("http://0x.example/", "http://0x.example/"),
    ("http://[::1]:8188", "http://[::1]:8188/"),
    ("http://[::1]:80", "http://[::1]/"),
    ("http://[2001:DB8:0:0::1]:8188", "http://[2001:db8::1]:8188/"),
    ("http://[::FFFF:127.0.0.1]:8188", "http://[::ffff:127.0.0.1]:8188/"),
]

REFUSED = [
    "",
    "127.0.0.1:8188",
    "ftp://comfy.example.com/",
    "http://",
    "http:///path",
    # Port: zero, empty, out of range, signed, too long, two of them.
    "http://127.0.0.1:0/",
    "http://127.0.0.1:/",
    "http://127.0.0.1:65536/",
    "http://127.0.0.1:+80/",
    "http://127.0.0.1:008188/",
    "http://127.0.0.1:81:82/",
    # Percent anywhere: zone ids raw or encoded, encoded hosts and paths.
    "http://[fe80::1%eth0]:8188/",
    "http://[fe80::1%25eth0]:8188/",
    "http://evil.example%40127.0.0.1/",
    "http://%31%32%37.0.0.1/",
    "http://comfy.example.com/%2e%2e/",
    # Where urlsplit and urllib3 read different hosts.
    "http://evil.example\\@127.0.0.1:8188",
    "http://user:pass@127.0.0.1:8188/",
    "http://127.0.0.1:8188/?x=1",
    "http://127.0.0.1:8188/#@evil.example",
    # Not LDH names.
    "http://127.0.0.1;evil.example/",
    "http://127.0.0.1,evil.example/",
    "http://exam_ple.example/",
    "http://-comfy.example/",
    "http://comfy-.example/",
    "http://" + "a" * 64 + ".example/",
    "http://" + ".".join(["a" * 63] * 4) + "/",
    # Every other spelling of 127.0.0.1, and a trailing dot.
    "http://2130706433/",
    "http://0x7f.1/",
    "http://0x7f000001/",
    "http://127.1/",
    "http://0177.0.0.1/",
    "http://127.000.0.1/",
    "http://1.2.3.04/",
    "http://comfy.0x/",
    "http://localhost./",
    "http://comfy.example.com./",
    # IPv6 spelled wrong, or IPv4 in brackets.
    "http://::1:8188/",
    "http://[::1]8188/",
    "http://[::1/",
    "http://[127.0.0.1]/",
    "http://[::g]/",
    # Space, control and non-ASCII characters.
    "http://comfy.example.com/a b/",
    "http://comfy\t.example.com/",
    "http://comfy.example.com\x00/",
    "http://ｃomfy.example.com/",
    "http://bücher.example/",
    "http://127.0.0.1:８１８８/",
    # Dot segments, and characters outside the unreserved set.
    "http://comfy.example.com/./",
    "http://comfy.example.com/../",
    "http://comfy.example.com/a/../b",
    "http://comfy.example.com/a;b",
    "http://comfy.example.com/a//b",
]


@pytest.mark.parametrize("typed, saved", ACCEPTED)
def test_a_plain_address_is_saved_in_its_one_spelling(typed, saved):
    assert normalize_comfyui_url(typed) == saved
    # Normalising is idempotent: the saved form is its own spelling.
    assert normalize_comfyui_url(saved) == saved


@pytest.mark.parametrize("typed, saved", ACCEPTED)
def test_every_parser_reads_an_accepted_address_the_same(typed, saved):
    """The host and port the address names are the ones the request goes to."""
    split = urlsplit(saved)
    sent = parse_url(requests.Request("GET", f"{saved}prompt").prepare().url)
    default = {"http": 80, "https": 443}[split.scheme]
    assert sent.host.strip("[]") == split.hostname
    assert (sent.port or default) == (split.port or default)
    assert sent.path == f"{split.path}prompt"


@pytest.mark.parametrize("typed", REFUSED)
def test_an_ambiguous_address_is_refused_without_repeating_it(typed):
    with pytest.raises(InvalidComfyUIAddress) as caught:
        normalize_comfyui_url(typed)
    message = str(caught.value)
    assert message.endswith(".")
    for part in ("evil", "eth0", "2130706433", "0x7f", "exam_ple", "bücher"):
        if part in typed:
            assert part not in message, message
    # A ValueError, so callers that already map ValueError to a 400 keep doing so.
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.0.0.2",
        "[::1]",
        "[::ffff:127.0.0.1]",
        "192.0.2.10",
        "[2001:db8::1]",
        "0.0.0.1",
    ],
)
def test_the_loopback_check_agrees_with_the_link_resolver(host):
    """``default_name`` and the arbiter probe call a host this computer exactly
    when the link's resolver does, for every literal the normaliser accepts."""
    base = normalize_comfyui_url(f"http://{host}:8188")
    canonical = urlsplit(base).hostname
    resolved = comfyui_link_service._addresses(canonical)
    assert resolved, canonical
    assert is_loopback_host(canonical) == all(a.is_loopback for a in resolved)


def test_a_name_is_not_assumed_to_be_this_computer():
    assert is_loopback_host("comfy.example") is False
    assert is_loopback_host("localhost.example") is False


def test_a_saved_address_is_read_through_the_normaliser():
    assert (
        request_base("http://LocalHost:8188/", connection_id="example-a")
        == "http://localhost:8188"
    )
    assert is_valid_saved("http://127.0.0.1:8188/") is True


def test_a_saved_address_that_fails_is_refused_logged_and_not_sent(caplog):
    stored = "http://evil.example\\@127.0.0.1:8188"
    with caplog.at_level(logging.ERROR, logger="pixlstash.utils.comfyui_url"):
        with pytest.raises(SavedComfyUIAddressInvalid) as caught:
            request_base(stored, connection_id="example-legacy")
    assert caught.value.connection_id == "example-legacy"
    assert "evil" not in str(caught.value)
    # Not a ValueError: the saved value is not the caller's input, so it must
    # not surface as a 400 from a handler that maps ValueError to one.
    assert not isinstance(caught.value, ValueError)
    assert "example-legacy" in caplog.text
    assert is_valid_saved(stored) is False


def test_a_refused_saved_address_is_logged_once_until_it_changes(caplog):
    """The training watcher and the UI read it every few seconds; one line per
    change, not one per read. The guard is per connection: a second connection
    saved with the same bad address gets its own line."""
    first = "http://evil.example\\@127.0.0.1:8188"
    second = "http://other.example\\@127.0.0.1:8188"

    def refusals(connection_id="example-polled"):
        with pytest.raises(SavedComfyUIAddressInvalid):
            request_base(stored, connection_id=connection_id)

    with caplog.at_level(logging.ERROR, logger="pixlstash.utils.comfyui_url"):
        for stored in (first, first, first):
            refusals()
        assert len(caplog.records) == 1
        stored = second
        refusals()
        refusals()
        assert len(caplog.records) == 2
        request_base("http://127.0.0.1:8188", connection_id="example-polled")
        stored = second
        refusals()
        assert len(caplog.records) == 3
        refusals("example-second")
        refusals("example-second")
        assert len(caplog.records) == 4
    named = [r.args[0] for r in caplog.records]
    assert named == ["example-polled"] * 3 + ["example-second"]


def _owner(url, *entries):
    """A user with *url* saved and *entries* as its stored connection list."""
    listed = json.dumps([{"id": i, "name": n, "url": u} for i, n, u in entries])
    return SimpleNamespace(comfyui_url=url, comfyui_connections=listed)


def test_two_spellings_of_one_address_are_one_connection():
    user = _owner(None, ("example-a", "Box", "http://localhost:8188/"))
    for spelling in (
        "http://LocalHost:8188",
        "  http://localhost:8188/ ",
        "HTTP://localhost:8188",
    ):
        assert comfyui_connections.is_listed(user, spelling), spelling
    assert not comfyui_connections.is_listed(user, "http://localhost:8189/")


def test_the_current_connection_is_found_and_read_in_its_one_spelling():
    user = _owner(
        "http://LOCALHOST:8188",
        ("example-a", "Box", "http://localhost:8188/"),
        ("example-b", "Other", "http://127.0.0.1:8188/"),
    )
    assert comfyui_connections.current_id(user) == "example-a"
    assert comfyui_connections.current_base(user, "http://127.0.0.1:8188") == (
        "http://localhost:8188"
    )


def test_with_no_address_saved_the_default_is_read():
    user = _owner(None)
    assert comfyui_connections.current_base(user, "http://127.0.0.1:8188/") == (
        "http://127.0.0.1:8188"
    )


def test_a_legacy_saved_address_is_found_but_refused_with_its_connection_id(caplog):
    """The loose key still finds the entry, so it can be removed or set again;
    nothing is built on it."""
    stored = "http://evil.example\\@127.0.0.1:8188/"
    user = _owner(stored, ("example-legacy", "Old", stored))
    assert comfyui_connections.current_id(user) == "example-legacy"
    with caplog.at_level(logging.ERROR, logger="pixlstash.utils.comfyui_url"):
        with pytest.raises(SavedComfyUIAddressInvalid) as caught:
            comfyui_connections.current_base(user, "http://127.0.0.1:8188")
    assert caught.value.connection_id == "example-legacy"
    assert "example-legacy" in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8188/",
        "http://127.0.0.1:8188/",
        "http://127.0.0.2:8188/",
        "http://[::1]:8188/",
        "http://[::ffff:127.0.0.1]:8188/",
    ],
)
def test_every_spelling_of_this_computer_is_named_this_pc(url):
    assert comfyui_connections.default_name(url) == "This PC"


@pytest.mark.parametrize(
    "url, name",
    [
        (f"http://{LAN_IPV4}:8188/", LAN_IPV4),
        ("http://comfy.example:8188/", "comfy.example"),
        ("http://localhost.example:8188/", "localhost.example"),
    ],
)
def test_another_host_is_named_for_itself(url, name):
    assert comfyui_connections.default_name(url) == name
