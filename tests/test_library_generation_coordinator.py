"""Generation admission keeps every active-vault request on one runtime tuple."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from pixlstash.authz.gate import AuthzGate
from pixlstash.authz.policy import AccessPolicy, LibraryAccessMode, RoutePolicy
from pixlstash.authz.registry import ROUTE_POLICIES
from pixlstash.middleware.library_admission import (
    BUSY_DETAIL,
    BUSY_RETRY_AFTER,
    LibraryAdmissionMiddleware,
    person_request_active,
)
from pixlstash.services.library_generation_coordinator import (
    LibraryGenerationCoordinator,
    LibraryNotReadyError,
    LibrarySwitchTimeoutError,
)
from pixlstash.services.library_switch_service import (
    LibrarySwitchService,
    SwitchState,
)


class _Registry:
    def __init__(self, library):
        self.library = library

    def active_library(self):
        return self.library


def _coherent_server():
    db = SimpleNamespace(is_open=True)
    vault = SimpleNamespace(is_open=True, db=db, image_root="/library-a")
    library = SimpleNamespace(uuid="library-a", path="/library-a")
    return SimpleNamespace(
        vault=vault,
        auth=SimpleNamespace(vault_db=db),
        library_registry=_Registry(library),
    )


def test_lease_captures_one_coherent_generation_and_runtime_tuple():
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)

    lease = coordinator.acquire_read()

    assert lease is not None
    assert (lease.generation, lease.library_uuid) == (0, "library-a")
    assert lease.vault is server.vault
    assert lease.db is server.vault.db
    coordinator.release_read(lease)


def test_mismatched_registry_vault_auth_tuple_is_never_admitted():
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)

    server.auth.vault_db = SimpleNamespace(is_open=True)
    assert coordinator.acquire_read() is None


def test_writer_wait_is_bounded_and_restores_ready_for_coherent_runtime():
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)
    lease = coordinator.acquire_read()
    assert lease is not None

    with pytest.raises(LibrarySwitchTimeoutError, match="Timed out"):
        coordinator.begin_switch(timeout=0.01)

    assert coordinator.state is SwitchState.READY
    coordinator.release_read(lease)


def test_outer_http_lease_lives_through_the_final_body_frame():
    events = []

    class Coordinator:
        state = SwitchState.READY

        def acquire_read_or_raise(self):
            events.append("acquire")
            return SimpleNamespace(library_uuid="library-a")

        def release_read(self, _lease):
            events.append("release")

    async def app(scope, receive, send):
        assert scope["state"]["library_lease"].library_uuid == "library-a"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        assert events == ["acquire"]
        await send({"type": "http.response.body", "body": b"last"})
        assert events == ["acquire"]

    server = SimpleNamespace(library_coordinator=Coordinator())
    middleware = LibraryAdmissionMiddleware(app, server=server)
    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/pictures",
        "state": {},
    }

    asyncio.run(middleware(scope, None, send))

    assert events == ["acquire", "release"]
    assert sent[-1]["body"] == b"last"


class _OpenCoordinator:
    """Admits every request; records nothing."""

    def acquire_read_or_raise(self):
        return SimpleNamespace(library_uuid="library-a", generation=1)

    def release_read(self, _lease):
        return None


def _call(middleware, method, path="/api/v1/pictures"):
    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "state": {}}
    asyncio.run(middleware(scope, None, send))
    return sent


@pytest.mark.parametrize(
    "method, expected",
    [
        ("POST", True),
        ("PUT", True),
        ("PATCH", True),
        ("DELETE", True),
        ("GET", False),
        ("HEAD", False),
        ("OPTIONS", False),
    ],
)
def test_only_a_request_that_changes_something_is_a_person_s(method, expected):
    """What a POST submits is the person's; what a progress poll submits is not."""
    seen = []

    async def app(scope, receive, send):
        seen.append(person_request_active())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = LibraryAdmissionMiddleware(
        app, server=SimpleNamespace(library_coordinator=_OpenCoordinator())
    )

    _call(middleware, method)

    assert seen == [expected]
    assert person_request_active() is False


_RESTORING = "PixlStash is restoring the library. Try again in a moment."
_SWITCHING = "PixlStash is switching library. Try again in a moment."
_UNAVAILABLE = "PixlStash has no verified open library. Restart the server."
_BUSY = "PixlStash is busy. Try again in a moment."


async def _route_must_not_run(scope, receive, send):
    raise AssertionError("a refused request must not reach the route")


@pytest.mark.parametrize(
    "close, detail",
    [
        (lambda c: c.begin_switch(timeout=0.01, restoring=True), _BUSY),
        (lambda c: c.begin_switch(timeout=0.01), _BUSY),
        (lambda c: setattr(c.server.vault, "is_open", False), _UNAVAILABLE),
    ],
    ids=["restore", "library switch", "no coherent library"],
)
def test_before_authentication_a_restore_and_a_switch_read_the_same(close, detail):
    """The middleware answers before anyone is authenticated, so a caller with
    no credentials learns only that the library is busy, not whether a restore
    or a library switch holds it."""
    coordinator = LibraryGenerationCoordinator(_coherent_server())
    close(coordinator)
    middleware = LibraryAdmissionMiddleware(
        _route_must_not_run, server=SimpleNamespace(library_coordinator=coordinator)
    )

    sent = _call(middleware, "GET")

    assert sent[0]["status"] == 503
    assert (b"retry-after", b"2") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == {"detail": detail}


@pytest.mark.parametrize(
    "close, detail",
    [
        (lambda c: c.begin_switch(timeout=0.01, restoring=True), _RESTORING),
        (lambda c: c.begin_switch(timeout=0.01), _SWITCHING),
        (lambda c: c.mark_unavailable(), _UNAVAILABLE),
    ],
    ids=["restore", "library switch", "unavailable"],
)
def test_the_gate_names_the_work_for_an_authenticated_caller(close, detail):
    """A request the middleware let through just before the switch began
    meets the gate's own 503. An authenticated caller is told whether a
    restore or a library switch holds the library.
    """
    server = _coherent_server()
    server.library_coordinator = LibraryGenerationCoordinator(server)
    server.library_switch = LibrarySwitchService(server)
    close(server.library_coordinator)

    with pytest.raises(HTTPException) as refused:
        AuthzGate._refuse_while_switching(
            SimpleNamespace(_server=server),
            SimpleNamespace(state=SimpleNamespace(auth_user_id=1)),
            RoutePolicy(AccessPolicy.OWNER_ONLY),
        )

    assert refused.value.status_code == 503
    assert refused.value.detail == detail
    assert refused.value.headers == {"Retry-After": BUSY_RETRY_AFTER}


_SHARE_LINK_POLICY = ROUTE_POLICIES[("GET", "/share/{token_slug}")]


@pytest.mark.parametrize(
    "route_policy",
    [_SHARE_LINK_POLICY, RoutePolicy(AccessPolicy.PUBLIC)],
    ids=["share link route", "any PUBLIC route"],
)
@pytest.mark.parametrize(
    "close, detail",
    [
        (lambda c: c.begin_switch(timeout=0.01, restoring=True), _BUSY),
        (lambda c: c.begin_switch(timeout=0.01), _BUSY),
        (lambda c: c.mark_unavailable(), _UNAVAILABLE),
    ],
    ids=["restore", "library switch", "unavailable"],
)
def test_the_gate_tells_a_caller_with_no_credentials_only_that_it_is_busy(
    route_policy, close, detail
):
    """A PUBLIC route reaches the gate with no principal, so the gate answers
    as admission does: a share-link holder cannot tell a restore from a
    library switch (CWE-204)."""
    assert route_policy.policy is AccessPolicy.PUBLIC
    assert route_policy.library_access is LibraryAccessMode.ACTIVE_VAULT
    server = _coherent_server()
    server.library_coordinator = LibraryGenerationCoordinator(server)
    server.library_switch = LibrarySwitchService(server)
    close(server.library_coordinator)

    with pytest.raises(HTTPException) as refused:
        AuthzGate._refuse_while_switching(
            SimpleNamespace(_server=server),
            SimpleNamespace(state=SimpleNamespace()),
            route_policy,
        )

    assert refused.value.status_code == 503
    assert refused.value.detail == detail
    assert refused.value.headers == {"Retry-After": BUSY_RETRY_AFTER}


def test_the_busy_answer_is_defined_once():
    """The gate, auth's restore fence and admission share one definition."""
    assert _BUSY == BUSY_DETAIL
    assert BUSY_RETRY_AFTER == "2"


class _MovesOnAfterRefusing:
    """Refuses in one state, then moves on before anyone can look again: the
    switch it refused for ends and admission is somewhere else by then."""

    def __init__(self, refused_in, restoring, then, then_restoring):
        self._refused = (refused_in, restoring)
        self._then = (then, then_restoring)
        self.state = refused_in
        self.restoring = restoring

    def _refuse(self):
        self.state, self.restoring = self._then
        return self._refused

    def acquire_read(self):
        self._refuse()
        return None

    def acquire_read_or_raise(self):
        state, _ = self._refuse()
        raise LibraryNotReadyError(state)


@pytest.mark.parametrize(
    "refused_in, restoring, then, then_restoring, detail",
    [
        # A restore held admission; it ended and a library switch began.
        (SwitchState.SWITCHING, True, SwitchState.SWITCHING, False, _BUSY),
        # A library switch held admission; it failed into UNAVAILABLE.
        (SwitchState.SWITCHING, False, SwitchState.UNAVAILABLE, False, _BUSY),
    ],
    ids=["restore then switch", "switch then unavailable"],
)
def test_a_refusal_names_the_state_it_was_refused_in(
    refused_in, restoring, then, then_restoring, detail
):
    coordinator = _MovesOnAfterRefusing(refused_in, restoring, then, then_restoring)
    middleware = LibraryAdmissionMiddleware(
        _route_must_not_run, server=SimpleNamespace(library_coordinator=coordinator)
    )

    sent = _call(middleware, "GET")

    assert sent[0]["status"] == 503
    assert json.loads(sent[1]["body"]) == {"detail": detail}


def test_a_full_restore_s_switch_says_it_is_a_restore():
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)

    coordinator.begin_switch(timeout=0.01, restoring=True)
    assert coordinator.restoring is True
    coordinator.restore_ready()

    coordinator.begin_switch(timeout=0.01)
    assert coordinator.restoring is False
    coordinator.restore_ready()


def test_route_mode_resolution_prefers_api_route_over_spa_catch_all():
    server = SimpleNamespace(library_coordinator=SimpleNamespace())
    middleware = LibraryAdmissionMiddleware(lambda *_args: None, server=server)

    assert (
        middleware._mode({"method": "GET", "path": "/api/v1/pictures"})
        is LibraryAccessMode.ACTIVE_VAULT
    )
    assert (
        middleware._mode({"method": "GET", "path": "/api/v1/pictures/42/metadata"})
        is LibraryAccessMode.ACTIVE_VAULT
    )
    assert (
        middleware._mode({"method": "GET", "path": "/api/v1/future-undeclared-route"})
        is LibraryAccessMode.ACTIVE_VAULT
    )
    assert (
        middleware._mode({"method": "GET", "path": "/some/frontend/deep-link"})
        is LibraryAccessMode.HUB_ONLY
    )
    assert (
        middleware._mode({"method": "POST", "path": "/api/v1/libraries/active"})
        is LibraryAccessMode.SWITCH_WRITER
    )


def test_switch_excludes_the_callers_own_lease():
    """A full restore holds its request's lease: the switch must not wait on it."""
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)
    own = coordinator.acquire_read()
    assert own is not None

    coordinator.begin_switch(timeout=0.01, exclude=own)

    assert coordinator.state is SwitchState.SWITCHING
    assert coordinator.acquire_read() is None
    coordinator.release_read(own)
    coordinator.restore_ready()
    assert coordinator.state is SwitchState.READY


def test_switch_excluding_own_lease_still_waits_for_another_reader():
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)
    own = coordinator.acquire_read()
    other = coordinator.acquire_read()
    assert own is not None and other is not None

    with pytest.raises(LibrarySwitchTimeoutError, match="Timed out"):
        coordinator.begin_switch(timeout=0.01, exclude=own)

    assert coordinator.state is SwitchState.READY
    coordinator.release_read(other)
    coordinator.begin_switch(timeout=0.01, exclude=own)
    assert coordinator.state is SwitchState.SWITCHING
    coordinator.release_read(own)


def test_switch_does_not_discount_a_lease_from_an_older_generation():
    """An excluded lease from before the last publish is not one of the
    current generation's readers, so it must not hide one of them."""
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)
    stale = coordinator.acquire_read()
    coordinator.begin_switch(timeout=0.01, exclude=stale)
    coordinator.publish_ready()
    current = coordinator.acquire_read()
    assert stale is not None and current is not None
    assert current.generation == stale.generation + 1

    with pytest.raises(LibrarySwitchTimeoutError, match="Timed out"):
        coordinator.begin_switch(timeout=0.01, exclude=stale)

    coordinator.release_read(stale)
    coordinator.release_read(current)


def test_a_second_writer_is_told_admission_is_not_ready_rather_than_timing_out():
    """A restore that meets a switch refuses at once instead of waiting on it."""
    server = _coherent_server()
    coordinator = LibraryGenerationCoordinator(server)
    coordinator.begin_switch(timeout=0.01)

    with pytest.raises(LibraryNotReadyError) as raised:
        coordinator.begin_switch(timeout=0.01)

    assert raised.value.state is SwitchState.SWITCHING
    assert coordinator.state is SwitchState.SWITCHING
    coordinator.restore_ready()
    assert coordinator.state is SwitchState.READY
