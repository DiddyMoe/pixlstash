"""The reasons the server gives when it closes an updates socket with 1012.

Code 1012 (Service Restart) is overloaded: uvicorn sends it with no reason on
every graceful shutdown. Only the reason tells the SPA what happened, and
``frontend/src/composables/useUpdatesSocket.js`` matches these strings exactly:

- ``LIBRARY_SWITCHED``: another library is now served, so every id the tab
  holds belongs to the old one and it reloads (``ws/broadcaster.py``).
- ``DATABASE_RESTORE_IN_PROGRESS``: the full-restore barrier is draining the
  established sockets (``auth.py``, ``_terminate_authenticated_websocket``).

Change one here and the client stops recognising it, so
``tests/test_ws_close_reasons.py`` pins both to the client's literals.
"""

WS_CLOSE_SERVICE_RESTART = 1012
LIBRARY_SWITCHED = "Library switched"
DATABASE_RESTORE_IN_PROGRESS = "Database restore in progress"
