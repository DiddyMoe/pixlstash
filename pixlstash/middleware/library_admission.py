"""Outermost ASGI admission fence for active-library reads.

It also marks the work a person asks for. A request with an unsafe method
(RFC 9110 §9.2.1: POST, PUT, PATCH, DELETE) runs with
``person_request_active()`` True, and ``TaskRunner.submit`` stamps every task
submitted inside it as started by a person. The mark is a ``ContextVar`` set
around the rest of the app, so it reaches the endpoint whether it runs on the
event loop, on Starlette's threadpool (``run_in_threadpool`` runs the call in
a copy of the context) or through ``asyncio.to_thread``. It does not reach a
``threading.Thread`` or a ``loop.run_in_executor`` call, which start from an
empty context on Python 3.12, nor the WorkPlanner's thread: work submitted
there is the machine's.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from starlette.responses import JSONResponse
from starlette.routing import compile_path

from pixlstash.authz.policy import LibraryAccessMode
from pixlstash.authz.registry import ROUTE_POLICIES
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    SwitchState,
)

# RFC 9110 §9.2.1: the methods that ask the server to change something. A safe
# method (GET, HEAD, OPTIONS) only reads, so what it submits - the idle model
# unload a progress poll triggers - is not something a person asked for.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_PERSON_REQUEST: ContextVar[bool] = ContextVar(
    "pixlstash_person_request", default=False
)

# The one answer a caller with no credentials gets while a restore or a
# library switch holds the library. Auth's restore fence and the authz gate
# answer anonymous callers with this same response, so a restore and a switch
# read alike before authentication (CWE-204). RFC 9110 §10.2.3: Retry-After in
# seconds.
BUSY_DETAIL = "PixlStash is busy. Try again in a moment."
BUSY_RETRY_AFTER = "2"
_UNAVAILABLE_DETAIL = "PixlStash has no verified open library. Restart the server."


def busy_response() -> JSONResponse:
    """The 503 a caller with no credentials gets while the library is held."""
    return JSONResponse(
        status_code=503,
        content={"detail": BUSY_DETAIL},
        headers={"Retry-After": BUSY_RETRY_AFTER},
    )


def person_request_active() -> bool:
    """True while the code runs for a person's request (an unsafe method)."""
    return _PERSON_REQUEST.get()


@contextmanager
def person_request() -> Iterator[None]:
    """Run the block as a person's request does: what it submits is theirs."""
    token = _PERSON_REQUEST.set(True)
    try:
        yield
    finally:
        _PERSON_REQUEST.reset(token)


class LibraryAdmissionMiddleware:
    def __init__(self, app, *, server):
        self.app = app
        self.server = server
        self._exact_modes = {
            (method, path): policy.library_access
            for (method, path), policy in ROUTE_POLICIES.items()
            if "{" not in path
        }
        self._modes = [
            (
                method,
                compile_path(path)[0],
                policy.library_access,
                len(re.sub(r"\{[^}]+\}", "", path)),
            )
            for (method, path), policy in ROUTE_POLICIES.items()
            if "{" in path
        ]
        # The SPA catch-all is declared near the public routes but must never
        # shadow a more-specific API template. Starlette resolves routes by
        # specificity/order; mirror that here by trying the longest literal
        # template first and the root catch-all last.
        self._modes.sort(key=lambda item: item[3], reverse=True)

    def _mode(self, scope) -> LibraryAccessMode:
        method = scope.get("method")
        path = scope.get("path", "")
        exact = self._exact_modes.get((method, path))
        if exact is not None:
            return exact
        for declared_method, regex, mode, literal_length in self._modes:
            # The frontend's root catch-all also regex-matches every unknown
            # API URL. Undeclared API work must retain the safe ACTIVE default;
            # it must never gain HUB_ONLY admission by falling through to SPA.
            if path.startswith("/api/") and literal_length <= 1:
                continue
            if declared_method == method and regex.match(path):
                return mode
        return LibraryAccessMode.ACTIVE_VAULT

    @staticmethod
    async def _reject(scope, receive, send, unavailable: bool) -> None:
        # Answered before authentication, so one sentence serves a restore and
        # a library switch alike: a caller with no credentials learns that the
        # library is unavailable for a moment, not which kind of work holds it.
        # The authz gate and the detach route answer authenticated callers, and
        # name the restore there.
        if unavailable:
            response = JSONResponse(
                status_code=503,
                content={"detail": _UNAVAILABLE_DETAIL},
                headers={"Retry-After": BUSY_RETRY_AFTER},
            )
        else:
            response = busy_response()
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") in _UNSAFE_METHODS:
            with person_request():
                return await self._admit(scope, receive, send)
        return await self._admit(scope, receive, send)

    async def _admit(self, scope, receive, send):
        mode = self._mode(scope)
        if mode is not LibraryAccessMode.ACTIVE_VAULT:
            return await self.app(scope, receive, send)
        coordinator = self.server.library_coordinator
        try:
            # The state comes with the refusal, read under the same lock: read
            # after it, a switch that ended in between would read as READY.
            lease = coordinator.acquire_read_or_raise()
        except LibraryNotReadyError as exc:
            return await self._reject(
                scope,
                receive,
                send,
                unavailable=exc.state is SwitchState.UNAVAILABLE,
            )
        scope.setdefault("state", {})["library_lease"] = lease
        try:
            return await self.app(scope, receive, send)
        finally:
            coordinator.release_read(lease)
