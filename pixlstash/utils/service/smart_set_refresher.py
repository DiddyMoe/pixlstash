"""Keeps smart picture sets (#2148) up to date as the library changes.

A smart set's rule is a ``GET /pictures`` query string, so it is evaluated by
the listing itself (`select_pictures_for_listing`) on a request built from that
string: the set holds what the grid showed for the same filter, by the same
code. The refresher runs it when library events say pictures may have changed,
at most once per :data:`REFRESH_WINDOW_S`, and writes the members through
`services/smart_sets.py`.
"""

import sys
import threading
from typing import Iterable

from starlette.requests import Request

from pixlstash.database import DBPriority
from pixlstash.event_types import EventType
from pixlstash.pixl_logging import get_logger
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    SwitchState,
)
from pixlstash.services.smart_sets import (
    apply_members_in_session,
    smart_sets_to_refresh_in_session,
)
from pixlstash.utils.service.library_lease import (
    UNAVAILABLE_RETRIES,
    PinnedVaultServer,
)

logger = get_logger(__name__)

# The events after which a picture may match a rule it did not before.
REFRESH_EVENTS = frozenset(
    {
        EventType.CHANGED_PICTURES,
        EventType.PICTURE_IMPORTED,
        EventType.CHANGED_TAGS,
        EventType.CLEARED_TAGS,
        EventType.CHANGED_CHARACTERS,
        EventType.CHANGED_FACES,
        EventType.CHANGED_DESCRIPTIONS,
        EventType.QUALITY_UPDATED,
        EventType.RESTORE_COMPLETED,
        EventType.LIBRARY_SWITCHED,
    }
)

# The `source` of the change event a refresh sends. A refresh does not wake
# another one for its own change.
SMART_SET_SOURCE = "smart_set"

# How long a burst of events (an import, a tagging pass) is gathered before one
# refresh runs. Fixed, not trailing: steady background work still refreshes.
REFRESH_WINDOW_S = 2.0

# How many times one refresh re-runs while a pass still changes something: a
# smart set whose rule names another smart set sees that set's new members.
MAX_PASSES = 3


def rule_picture_ids(server, query: str, project_ids: Iterable[int]) -> set[int]:
    """The pictures a rule matches, as the owner's grid would list them.

    A set in projects matches only pictures in one of them: its projects scope
    its rule, so a smart set never pulls a picture into a project by itself.
    """
    # Imported here, not at the top: the listing imports the routes package,
    # which imports the server's routers, and this module is imported by the
    # server.
    from pixlstash.routes.pictures import select_pictures_for_listing

    scopes = [str(pid) for pid in project_ids] or [None]
    found: set[int] = set()
    for project_id in scopes:
        request = Request(
            {"type": "http", "query_string": query.encode(), "headers": []}
        )
        found.update(
            select_pictures_for_listing(
                server=server,
                request=request,
                sort=None,
                descending=True,
                offset=0,
                limit=sys.maxsize,
                metadata_fields=[],
                return_ids_only=True,
                project_id=project_id,
            )
        )
    return found


def refresh_smart_sets(server, set_ids: Iterable[int] | None = None) -> list[int]:
    """Bring smart sets' members up to date now; returns the ids that changed.

    Every unlocked smart set, or only ``set_ids`` (and then every set, if they
    changed, for the rules that name them). One set's failure is logged and
    leaves the others to run.
    """
    vault = server.vault
    changed: list[int] = []
    # A rule may name another smart set (`set_id`), and the refresher ignores
    # its own change events, so a pass that changed something runs again here.
    # ponytail: a fixed pass count, so sets naming each other cannot loop; a
    # chain deeper than this catches up on the next library event.
    targets = list(set_ids) if set_ids is not None else None
    for _ in range(MAX_PASSES):
        rows = vault.db.run_immediate_read_task(
            smart_sets_to_refresh_in_session, targets
        )
        changed_this_pass: list[int] = []
        for set_id, query, project_ids in rows:
            try:
                ids = rule_picture_ids(server, query, project_ids)
                changed_this_pass += vault.db.run_task(
                    apply_members_in_session,
                    set_id,
                    ids,
                    priority=DBPriority.IMMEDIATE,
                )
            except Exception as exc:
                logger.warning(
                    "Smart set %s was not refreshed (filter %r): %s",
                    set_id,
                    query,
                    exc,
                    exc_info=True,
                )
        changed += changed_this_pass
        if not changed_this_pass:
            break
        # The sets that name a changed set may be any of them.
        targets = None
    if changed:
        vault.notify(
            EventType.CHANGED_PICTURES,
            {
                "picture_ids": sorted(set(changed)),
                "source": SMART_SET_SOURCE,
                "change_kind": "updated",
            },
        )
    return changed


class SmartSetRefresher:
    """Runs :func:`refresh_smart_sets` after library events, gathered.

    Each refresh holds a library read lease, as a request does, so a library
    switch or a full restore waits for it and it never writes members into a
    library file that replaced the one it read. While admission is SWITCHING
    the refresh waits one more window, for as long as that lasts; while no
    coherent library is open (UNAVAILABLE) it waits one more window at most
    ``UNAVAILABLE_RETRIES`` times, then gives up until the next library
    event.

    Attributes:
        server: The server whose current vault is refreshed.
    """

    def __init__(self, server):
        self.server = server
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._closed = False
        # Windows waited in a row while no coherent library was open.
        self._unavailable_tries = 0

    def on_event(self, event_type: EventType, data=None) -> None:
        """A vault event: schedule a refresh if pictures may have changed."""
        if event_type not in REFRESH_EVENTS:
            return
        if isinstance(data, dict) and data.get("source") == SMART_SET_SOURCE:
            return
        self._arm()

    def _arm(self) -> None:
        """Schedule one refresh a window from now, unless one is pending."""
        with self._lock:
            if self._closed or self._timer is not None:
                return
            self._timer = threading.Timer(REFRESH_WINDOW_S, self._run)
            self._timer.daemon = True
            self._timer.start()

    def close(self) -> None:
        """Stop: a pending refresh is dropped and none is scheduled again."""
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _run(self) -> None:
        with self._lock:
            self._timer = None
            if self._closed:
                return
        # Built after this refresher during start-up (server.py), so an event
        # in between finds none yet and waits one more window for it.
        coordinator = getattr(self.server, "library_coordinator", None)
        if coordinator is None:
            logger.info(
                "Smart set refresh waits %.0f s: the library is being opened.",
                REFRESH_WINDOW_S,
            )
            self._arm()
            return
        try:
            lease = coordinator.acquire_read_or_raise()
        except LibraryNotReadyError as exc:
            self._wait_for_the_library(exc.state)
            return
        with self._lock:
            self._unavailable_tries = 0
        try:
            refresh_smart_sets(PinnedVaultServer(self.server, lease.vault))
        except Exception as exc:
            logger.warning("Smart set refresh failed: %s", exc, exc_info=True)
        finally:
            coordinator.release_read(lease)

    def _wait_for_the_library(self, state: SwitchState) -> None:
        """Admission refused the lease in *state*: wait a window, or give up."""
        if state is SwitchState.SWITCHING:
            logger.info(
                "Smart set refresh waits %.0f s: the library is being switched "
                "or restored.",
                REFRESH_WINDOW_S,
            )
            self._arm()
            return
        with self._lock:
            self._unavailable_tries += 1
            tries = self._unavailable_tries
            if tries > UNAVAILABLE_RETRIES:
                self._unavailable_tries = 0
        if tries > UNAVAILABLE_RETRIES:
            logger.warning(
                "Smart sets were not refreshed: no library was open after %d "
                "tries. The next library event refreshes them.",
                tries,
            )
            return
        logger.info(
            "Smart set refresh waits %.0f s: no library is open (wait %d of %d).",
            REFRESH_WINDOW_S,
            tries,
            UNAVAILABLE_RETRIES,
        )
        self._arm()
