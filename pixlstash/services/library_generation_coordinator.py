"""Admission and writer fencing for the process-wide active vault generation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum


class SwitchState(str, Enum):
    """What the server is currently able to serve."""

    READY = "ready"
    """A vault is open and requests are served normally."""

    SWITCHING = "switching"
    """Mid-swap. Data requests are refused with 503 until this clears."""

    UNAVAILABLE = "unavailable"
    """No verified coherent open vault exists; restart is required."""


@dataclass(frozen=True)
class LibraryReadLease:
    generation: int
    library_uuid: str
    vault: object
    db: object


class LibraryNotReadyError(RuntimeError):
    """Library admission is not READY, so the work cannot start now.

    Raised by ``begin_switch`` when another switch (a library switch, a
    restore) already holds admission, or the runtime is UNAVAILABLE. A caller
    that runs outside a request, such as the training import, can tell from
    ``state`` whether to wait and retry (SWITCHING) or give up and log
    (UNAVAILABLE).

    Attributes:
        state: The ``SwitchState`` admission was in.
    """

    def __init__(self, state: SwitchState):
        self.state = state
        super().__init__(f"Library admission is not ready ({state.value})")


class LibrarySwitchTimeoutError(RuntimeError):
    """``begin_switch`` gave up: a read lease was still held at its timeout.

    Admission is READY again (or UNAVAILABLE when the runtime is no longer
    coherent) and no switch began. A ``RuntimeError``, so a caller that
    catches that still catches it; a caller that must tell this timeout from
    any other failure catches this class.
    """


class LibraryGenerationCoordinator:
    def __init__(self, server):
        self.server = server
        self._condition = threading.Condition()
        self.state = SwitchState.READY
        self.generation = 0
        self._readers: dict[int, int] = {}
        # Whether the switch in progress, or the last one, is a full restore
        # (``begin_switch(restoring=True)``). Read with ``state``: it only
        # names what a SWITCHING state is for, so a refused request can say so.
        self.restoring = False

    def _coherent(self) -> bool:
        library = self.server.library_registry.active_library()
        vault = getattr(self.server, "vault", None)
        auth = getattr(self.server, "auth", None)
        db = getattr(vault, "db", None)
        return bool(
            library is not None
            and vault is not None
            and getattr(vault, "is_open", False)
            and db is not None
            and getattr(db, "is_open", False)
            and auth is not None
            and auth.vault_db is db
            and vault.image_root == library.path
        )

    def _admit_locked(self) -> tuple[LibraryReadLease | None, SwitchState]:
        """Admit one reader, or say why not. The caller holds ``_condition``.

        Returns:
            ``(lease, READY)`` when admitted; ``(None, state)`` when not, where a
            READY admission with no coherent library open is UNAVAILABLE.
        """
        if self.state is not SwitchState.READY:
            return None, self.state
        if not self._coherent():
            return None, SwitchState.UNAVAILABLE
        library = self.server.library_registry.active_library()
        vault = self.server.vault
        auth = getattr(self.server, "auth", None)
        db = getattr(vault, "db", None)
        if library is None or vault is None or db is None or auth is None:
            return None, SwitchState.UNAVAILABLE
        if auth.vault_db is not db:
            return None, SwitchState.UNAVAILABLE
        lease = LibraryReadLease(self.generation, library.uuid, vault, db)
        self._readers[self.generation] = self._readers.get(self.generation, 0) + 1
        return lease, SwitchState.READY

    def acquire_read(self) -> LibraryReadLease | None:
        """A read lease on the active library, or None when admission refuses."""
        with self._condition:
            lease, _ = self._admit_locked()
            return lease

    def acquire_read_or_raise(self) -> LibraryReadLease:
        """A read lease on the active library, or the reason there is none.

        The reason is read under the same lock as the refusal. Read after
        :meth:`acquire_read` has returned None, ``state`` may already have moved
        on: a switch that ended in between reads as READY, which looks like
        another library being active rather than a switch to wait out.

        Returns:
            The lease, which the caller gives back with :meth:`release_read`.

        Raises:
            LibraryNotReadyError: No lease. ``state`` is SWITCHING while a switch
                or a restore holds admission, and UNAVAILABLE when admission
                is READY but no coherent library is open.
        """
        with self._condition:
            lease, state = self._admit_locked()
            if lease is None:
                raise LibraryNotReadyError(state)
            return lease

    def release_read(self, lease: LibraryReadLease) -> None:
        with self._condition:
            remaining = self._readers.get(lease.generation, 0) - 1
            if remaining > 0:
                self._readers[lease.generation] = remaining
            else:
                self._readers.pop(lease.generation, None)
            self._condition.notify_all()

    def begin_switch(
        self,
        timeout: float = 30.0,
        exclude: LibraryReadLease | None = None,
        restoring: bool = False,
    ) -> None:
        """Refuse new leases, then wait for the current generation's readers.

        Args:
            timeout: Seconds to wait for the readers before giving up. On
                timeout admission returns to READY (or UNAVAILABLE when the
                runtime is no longer coherent) and
                ``LibrarySwitchTimeoutError`` is raised.
            exclude: A lease the caller itself holds, not waited for. A request
                that switches the library (a full restore) entered through the
                admission middleware and holds a lease of its own; counting it
                would make the switch wait on itself until it timed out. Its
                holder still releases it as usual.
            restoring: The switch is a full restore, not a library switch;
                ``restoring`` says so while admission is SWITCHING, for the
                text a refused request is given.

        Raises:
            LibraryNotReadyError: Admission was not READY, so no switch began
                and the state is unchanged.
            LibrarySwitchTimeoutError: The readers were still there at the
                timeout.
        """
        with self._condition:
            if self.state is not SwitchState.READY:
                raise LibraryNotReadyError(self.state)
            self.state = SwitchState.SWITCHING
            self.restoring = restoring
            deadline = time.monotonic() + timeout
            own = (
                1
                if exclude is not None and exclude.generation == self.generation
                else 0
            )
            while self._readers.get(self.generation, 0) > own:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._coherent():
                        self.state = SwitchState.READY
                    else:
                        self.state = SwitchState.UNAVAILABLE
                    self._condition.notify_all()
                    raise LibrarySwitchTimeoutError(
                        "Timed out waiting for active-library readers"
                    )
                self._condition.wait(timeout=remaining)

    def publish_ready(self) -> None:
        with self._condition:
            if not self._coherent():
                self.state = SwitchState.UNAVAILABLE
                raise RuntimeError("Refusing READY for an incoherent library runtime")
            self.generation += 1
            self.state = SwitchState.READY
            self._condition.notify_all()

    def restore_ready(self) -> None:
        with self._condition:
            if not self._coherent():
                self.state = SwitchState.UNAVAILABLE
                raise RuntimeError("Previous library runtime is not coherent")
            self.state = SwitchState.READY
            self._condition.notify_all()

    def mark_unavailable(self) -> None:
        with self._condition:
            self.state = SwitchState.UNAVAILABLE
            self._condition.notify_all()
