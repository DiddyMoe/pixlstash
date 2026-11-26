"""The 1012 close reasons the server sends are the ones the SPA matches.

``frontend/src/composables/useUpdatesSocket.js`` tells a library switch (reload
the tab) and a full restore (wait for the barrier) apart from a plain server
shutdown by the close reason alone. Both server sites take their reason from
``pixlstash/ws/close_reasons.py``; this file pins that module to the client's
literals and both sites to the module.
"""

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock

from pixlstash.auth import AuthService
from pixlstash.ws.broadcaster import WsBroadcasterMixin
from pixlstash.ws.close_reasons import (
    DATABASE_RESTORE_IN_PROGRESS,
    LIBRARY_SWITCHED,
    WS_CLOSE_SERVICE_RESTART,
)

_CLIENT = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "composables"
    / "useUpdatesSocket.js"
)


def _client_literal(name: str) -> str:
    match = re.search(rf'^const {name} = "([^"]*)";$', _CLIENT.read_text(), re.M)
    assert match, f"{name} is no longer declared in {_CLIENT.name}"
    return match.group(1)


class _Socket:
    def __init__(self):
        self.closed = None

    async def close(self, code, reason):
        self.closed = (code, reason)


def test_the_server_reasons_are_the_client_reasons():
    assert LIBRARY_SWITCHED == _client_literal("LIBRARY_SWITCH_CLOSE_REASON")
    assert DATABASE_RESTORE_IN_PROGRESS == _client_literal("FULL_RESTORE_CLOSE_REASON")
    assert WS_CLOSE_SERVICE_RESTART == 1012


def test_a_library_switch_closes_with_the_switch_reason():
    socket = _Socket()
    asyncio.run(WsBroadcasterMixin._close_websocket_clients([{"ws": socket}]))
    assert socket.closed == (1012, LIBRARY_SWITCHED)


def test_the_restore_barrier_closes_with_the_restore_reason():
    socket = _Socket()
    auth = AuthService(
        db=None, server_config={}, server_config_path="", logger=MagicMock()
    )
    asyncio.run(auth._terminate_authenticated_websocket(1, socket, None))
    assert socket.closed == (1012, DATABASE_RESTORE_IN_PROGRESS)
