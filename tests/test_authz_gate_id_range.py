"""A scoped token's object id outside SQLite's range fails closed with a 403.

The gate parses a route's raw id itself (a path segment or a body value) before
the handler's validation runs, so ``int()`` of a long digit string succeeded and
the value reached the membership query, where the driver raised
``OverflowError`` binding it: a 500 where the gate means "not yours". The id
resolver for tag suggestions had the same parse. Both now use
``parse_sqlite_int`` and refuse such an id like any other malformed one.

Every check runs both directions: the out-of-range id is refused with the gate's
403, and the in-scope id on the same route still reaches the handler.
"""

import gc
import json
import os
import sqlite3
import tempfile
import types

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from starlette.testclient import TestClient

from pixlstash.authz.gate import AuthzGate
from pixlstash.authz.policy import AccessPolicy, RoutePolicy
from pixlstash.server import Server
from tests.authz_guard import no_spa_fallback  # noqa: F401
from tests.utils import upload_pictures_and_wait

API = "/api/v1"
PAST_SQLITE = str(2**63)
GATE_REFUSAL = "Token is not authorised to access this resource"

pytestmark = pytest.mark.usefixtures("no_spa_fallback")


def _good_picture_file():
    pictures_dir = os.path.join(os.path.dirname(__file__), "..", "pictures", "good")
    for name in sorted(os.listdir(pictures_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            content_type = "image/png" if ext == ".png" else "image/jpeg"
            with open(os.path.join(pictures_dir, name), "rb") as fh:
                return name, fh.read(), content_type
    raise AssertionError("no test picture under pictures/good")


@pytest.fixture(scope="module")
def env():
    """A live server with one picture in one set and a READ token for that set."""
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000}))
    server = Server(config_path)
    server.__enter__()
    try:
        owner = TestClient(server.api, raise_server_exceptions=False)
        anon = TestClient(server.api, raise_server_exceptions=False)
        r = owner.post(
            f"{API}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert r.status_code == 200, r.text
        status = upload_pictures_and_wait(
            owner, [("file", _good_picture_file())], timeout_s=30
        )
        assert status["status"] == "completed", status
        picture_id = owner.get(f"{API}/pictures").json()[0]["id"]
        set_id = owner.post(f"{API}/picture_sets", json={"name": "S"}).json()[
            "picture_set"
        ]["id"]
        r = owner.post(f"{API}/picture_sets/{set_id}/members/{picture_id}")
        assert r.status_code in (200, 201), r.text
        r = owner.post(
            f"{API}/users/me/token",
            json={
                "description": "set token",
                "scope": "READ",
                "resource_type": "picture_set",
                "resource_id": set_id,
            },
        )
        assert r.status_code == 200, r.text
        yield {
            "server": server,
            "anon": anon,
            "picture_id": picture_id,
            "headers": {"Authorization": f"Bearer {r.json()['token']}"},
        }
    finally:
        server.__exit__(None, None, None)
        temp_dir.cleanup()
        gc.collect()


def test_a_set_token_naming_a_picture_past_sqlite_is_refused_not_a_500(env):
    """GET /pictures/{id}/metadata is PICTURE_SCOPED; a set token's membership
    check queries the set's members with the picture id bound."""
    assert env["server"].authz._enforcing is True
    anon, headers = env["anon"], env["headers"]
    r = anon.get(f"{API}/pictures/{env['picture_id']}/metadata", headers=headers)
    assert r.status_code == 200, f"in-scope picture: {r.status_code} {r.text}"
    r = anon.get(f"{API}/pictures/{PAST_SQLITE}/metadata", headers=headers)
    assert r.status_code == 403, f"past SQLite's range: {r.status_code} {r.text}"
    assert r.json()["detail"] == GATE_REFUSAL


# --- the tag-suggestion id resolver --------------------------------------


class _Scope:
    """The ``token_scope`` fields the gate reads: a picture-scoped READ token."""

    scope = "READ"
    resource_type = "picture"

    def __init__(self, resource_id):
        self.resource_id = resource_id


def _picture_scope(request: Request):
    """Stand in for the auth middleware: no header means the owner."""
    picture_id = request.headers.get("x-test-picture-scope")
    if picture_id:
        request.state.token_scope = _Scope(int(picture_id))


class _SqliteBindingSession:
    """``session.get`` that binds the id the way the real lookup does.

    The id goes through an in-memory SQLite statement, so a value the driver
    cannot bind raises the driver's own ``OverflowError``. The gate calls it
    from a threadpool worker, hence ``check_same_thread=False``.
    """

    def __init__(self, suggestion_to_picture):
        self._map = suggestion_to_picture
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)

    def get(self, _model, suggestion_id):
        (bound,) = self._conn.execute("SELECT ?", (suggestion_id,)).fetchone()
        picture_id = self._map.get(bound)
        if picture_id is None:
            return None
        return types.SimpleNamespace(picture_id=picture_id)


class _Server:
    def __init__(self, session):
        class _DB:
            def run_immediate_read_task(self, func, *args, **kwargs):
                return func(session, *args, **kwargs)

        self.vault = types.SimpleNamespace(db=_DB())


def test_a_suggestion_id_past_sqlite_fails_closed_through_the_resolver():
    registry = {
        ("POST", "/api/v1/id-range/sugg/{suggestion_id}/accept"): RoutePolicy(
            AccessPolicy.PICTURE_SCOPED,
            id_param="suggestion_id",
            id_resolver="tag_suggestion",
        ),
    }
    gate = AuthzGate(
        registry=registry,
        enforcing=True,
        auth=None,
        server=_Server(_SqliteBindingSession({100: 7})),
    )
    router = APIRouter()

    @router.post("/sugg/{suggestion_id}/accept")
    async def _accept():
        return {"ok": True}

    app = FastAPI()
    app.include_router(
        router,
        prefix="/api/v1/id-range",
        dependencies=[Depends(_picture_scope), Depends(gate)],
    )
    gate.resolve_routes(app)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Test-Picture-Scope": "7"}

    r = client.post("/api/v1/id-range/sugg/100/accept", headers=headers)
    assert r.status_code == 200, r.text
    r = client.post(f"/api/v1/id-range/sugg/{PAST_SQLITE}/accept", headers=headers)
    assert r.status_code == 403, f"{r.status_code} {r.text}"
    assert r.json()["detail"] == GATE_REFUSAL
