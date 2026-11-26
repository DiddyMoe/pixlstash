"""The session connect guard tests/conftest.py installs for every test."""

import _socket
import socket
from urllib.parse import urlsplit

import pytest

from pixlstash.routes.comfyui import DEFAULT_COMFYUI_URL

COMFYUI_PORT = urlsplit(DEFAULT_COMFYUI_URL).port
CONNECT_TIMEOUT_S = 3.0
# Documentation ranges: never routed, so an unguarded connect only waits.
OUTSIDE_IP = "192.0.2.1"
STAND_IN_IP = "198.51.100.7"


@pytest.mark.parametrize(
    "address",
    [("127.0.0.1", COMFYUI_PORT), (OUTSIDE_IP, 80)],
    ids=["default-comfyui", "off-this-machine"],
)
def test_the_guard_refuses_the_default_comfyui_and_off_machine_addresses(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(CONNECT_TIMEOUT_S)
        with pytest.warns(UserWarning, match="test connect guard"):
            with pytest.raises(ConnectionRefusedError):
                sock.connect(address)


def test_a_loopback_server_the_test_binds_is_reachable():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(
            server.getsockname(), CONNECT_TIMEOUT_S
        ) as client:
            accepted, _ = server.accept()
            with accepted:
                client.sendall(b"ping")
                assert accepted.recv(4) == b"ping"


def test_an_address_a_model_host_resolved_to_is_let_through(monkeypatch):
    real = _socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if host == "huggingface.co":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (STAND_IN_IP, port))]
        return real(host, port, *args, **kwargs)

    # Beneath the guard, which sees the lookup as it would a real one.
    monkeypatch.setattr(_socket, "getaddrinfo", resolve)
    try:
        socket.create_connection(("huggingface.co", 443), timeout=0.05).close()
    except OSError as outcome:  # the stand-in address is never routed
        assert "test connect guard" not in str(outcome)
