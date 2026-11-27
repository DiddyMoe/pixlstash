"""Outermost ASGI admission fence for active-library reads."""

import re

from starlette.responses import JSONResponse
from starlette.routing import compile_path

from pixlstash.authz.policy import LibraryAccessMode
from pixlstash.authz.registry import ROUTE_POLICIES

# The one answer a caller with no credentials gets while a restore or a
# library switch holds the library, so the two cannot be told apart before
# authentication. Auth's restore fence and the authz gate use it too.
BUSY_DETAIL = "PixlStash is busy. Try again in a moment."
BUSY_RETRY_AFTER = "2"


def busy_response() -> JSONResponse:
    """The 503 a caller with no credentials gets while the library is held."""
    return JSONResponse(
        status_code=503,
        content={"detail": BUSY_DETAIL},
        headers={"Retry-After": BUSY_RETRY_AFTER},
    )


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
        if unavailable:
            response = JSONResponse(
                status_code=503,
                content={
                    "detail": "PixlStash has no verified open library. "
                    "Restart the server."
                },
                headers={"Retry-After": BUSY_RETRY_AFTER},
            )
        else:
            response = busy_response()
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        mode = self._mode(scope)
        if mode is not LibraryAccessMode.ACTIVE_VAULT:
            if self.server.library_coordinator.state.value == "switching":
                # Authentication then gives a caller with no credentials
                # busy_response(), as it does during a restore.
                scope.setdefault("state", {})["library_switching"] = True
            return await self.app(scope, receive, send)
        lease = self.server.library_coordinator.acquire_read()
        if lease is None:
            return await self._reject(
                scope,
                receive,
                send,
                self.server.library_coordinator.state.value == "unavailable",
            )
        scope.setdefault("state", {})["library_lease"] = lease
        try:
            return await self.app(scope, receive, send)
        finally:
            self.server.library_coordinator.release_read(lease)
