"""No request integer reaches SQLite outside the range SQLite stores.

SQLite's INTEGER is a signed 64-bit value, and a larger Python int raises
``OverflowError`` while the driver binds it: a 500, not a 422. Every integer a
request can carry is declared :data:`~pixlstash.utils.sqlite_ints.SqliteInt`
(or a narrower bound inside its range), so validation refuses it first.

The guardrail walks the real app: every mounted route, HTTP and WebSocket,
through FastAPI's dependant tree (``Depends`` included), and every parameter's
pydantic schema with nested models, lists, unions and dict values followed.
There is no allowlist here. An integer that is legitimately wider than SQLite
(a ComfyUI widget value, a sampler seed kept as TEXT) says so where it is
declared, with :class:`~pixlstash.utils.sqlite_ints.NotASqliteInteger`.

What the walk cannot see is an integer a handler parses itself, out of a
``str`` path segment or a ``dict`` body. Those sites use
:func:`~pixlstash.utils.sqlite_ints.parse_sqlite_int`; the HTTP tests below
pin a sample of them at the boundary.
"""

import json
import tempfile
from collections import Counter
from types import SimpleNamespace
from typing import Annotated

import pytest
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from pixlstash.route_inventory import flatten_dependant_fields
from pixlstash.server import Server
from pixlstash.utils.sqlite_ints import (
    SQLITE_INT_MAX,
    SQLITE_INT_MIN,
    is_sqlite_int,
    parse_sqlite_int,
)

API = "/api/v1"
_PARAM_KINDS = (
    "path_params",
    "query_params",
    "header_params",
    "cookie_params",
    "body_params",
)
_MARKER = "x-not-sqlite-integer"


@pytest.fixture(scope="module")
def int_env():
    """One Server, one owner login, one character and one picture set."""
    tmp = tempfile.TemporaryDirectory()
    config_path = f"{tmp.name}/server-config.json"
    with open(config_path, "w") as handle:
        json.dump({"port": 8000}, handle)
    server = Server(config_path)
    server.__enter__()
    try:
        owner = TestClient(server.api, raise_server_exceptions=False)
        r = owner.post(
            f"{API}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert r.status_code == 200, r.text
        r = owner.post(f"{API}/characters", json={"name": "Int Bounds Character"})
        assert r.status_code in {200, 201}, r.text
        character_id = r.json().get("id") or r.json()["character"]["id"]
        r = owner.post(f"{API}/picture_sets", json={"name": "Int Bounds Set"})
        assert r.status_code in {200, 201}, r.text
        set_id = r.json()["picture_set"]["id"]
        yield SimpleNamespace(
            server=server, owner=owner, character_id=character_id, set_id=set_id
        )
    finally:
        server.__exit__(None, None, None)
        tmp.cleanup()


def _integer_nodes(schema, defs, where, seen=()):
    """Every integer node reachable from a pydantic JSON schema, as
    ``(where, node)``. ``$ref`` is followed into ``$defs`` once per model (a
    recursive model terminates), and so are unions, tuple and list items,
    object properties and dict values."""
    if not isinstance(schema, dict):
        return
    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        if name not in seen:
            yield from _integer_nodes(
                defs[name], defs, f"{where}<{name}>", seen + (name,)
            )
        return
    types = schema.get("type")
    if "integer" in (types if isinstance(types, list) else [types]):
        yield where, schema
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for branch in schema.get(key) or []:
            yield from _integer_nodes(branch, defs, where, seen)
    yield from _integer_nodes(schema.get("items"), defs, f"{where}[]", seen)
    for prop, sub in (schema.get("properties") or {}).items():
        yield from _integer_nodes(sub, defs, f"{where}.{prop}", seen)
    yield from _integer_nodes(
        schema.get("additionalProperties"), defs, f"{where}{{}}", seen
    )


def _fits_sqlite(node) -> bool:
    """An integer node whose values SQLite can all store, or one declared as
    never bound as an INTEGER. Bounds compare as exact ints: pydantic's own
    schema keeps them so, unlike FastAPI's OpenAPI model."""
    if _MARKER in node:
        return True
    allowed = node.get("enum") or ([node["const"]] if "const" in node else None)
    if allowed is not None:
        return all(is_sqlite_int(value) for value in allowed)
    low = node.get("minimum", node.get("exclusiveMinimum"))
    high = node.get("maximum", node.get("exclusiveMaximum"))
    return (
        low is not None
        and high is not None
        and low >= SQLITE_INT_MIN
        and high <= SQLITE_INT_MAX
    )


def _request_integers(app):
    """Every integer input of every mounted route, as
    ``(module, route, where, node)``."""
    for ctx in iter_route_contexts(app.routes):
        route = ctx.original_route
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        module = getattr(getattr(route, "endpoint", None), "__module__", "?")
        methods = ",".join(sorted((ctx.methods or {"WS"}) - {"HEAD", "OPTIONS"}))
        label = f"{methods} {ctx.path or route.path}"
        for kind in _PARAM_KINDS:
            for field in flatten_dependant_fields(dependant, kind):
                info = field.field_info
                schema = TypeAdapter(Annotated[info.annotation, info]).json_schema()
                source = f"{kind.split('_')[0]}:{field.alias or field.name}"
                for where, node in _integer_nodes(
                    schema, schema.get("$defs", {}), source
                ):
                    yield module, label, where, node


def test_every_request_integer_fits_sqlite(int_env):
    """No allowlist: an integer input is bounded inside SQLite's range or it
    is declared ``NotASqliteInteger`` where it is defined."""
    checked = Counter()
    unbounded = []
    for module, label, where, node in _request_integers(int_env.server.api):
        checked[where.split(":")[0]] += 1
        if not _fits_sqlite(node):
            unbounded.append(f"{module}: {label} {where}")
    total = sum(checked.values())
    print(
        f"Integer bounds: {total} integer inputs checked: {dict(sorted(checked.items()))}"
    )
    # The walk must be reading the routes, and the body models in them.
    assert checked["path"] and checked["query"] and checked["body"], checked
    assert not unbounded, (
        f"{len(unbounded)} of {total} request integers accept a value SQLite "
        "cannot store (a 500, not a 422):\n" + "\n".join(sorted(unbounded))
    )


def test_the_walk_sees_a_nested_model_field_and_the_marker(int_env):
    """The walk is not vacuous: it reaches an integer inside a nested body
    model, and the declared exception carries its reason into the schema."""
    seen = {
        (label, where): node
        for _module, label, where, node in _request_integers(int_env.server.api)
    }
    nested = [where for _label, where in seen if where.count(".") >= 2]
    assert nested, "the walk never reached a field of a nested body model"
    seeds = [
        node
        for (label, where), node in seen.items()
        if label.startswith("POST")
        and "/workflows/" in label
        and where.endswith(".seed")
    ]
    assert seeds, sorted(where for _label, where in seen if "seed" in where)
    for seed in seeds:
        assert seed.get(_MARKER), seed
        assert seed["maximum"] == 2**64 - 1, seed


def test_parse_sqlite_int_holds_the_exact_edges():
    assert parse_sqlite_int(str(SQLITE_INT_MAX)) == SQLITE_INT_MAX
    assert parse_sqlite_int(SQLITE_INT_MIN) == SQLITE_INT_MIN
    for outside in (SQLITE_INT_MAX + 1, str(SQLITE_INT_MIN - 1)):
        with pytest.raises(ValueError):
            parse_sqlite_int(outside)
    with pytest.raises(ValueError):
        parse_sqlite_int("5e0")
    assert not is_sqlite_int(True)
    assert not is_sqlite_int(SQLITE_INT_MAX + 1)


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("GET", f"/reviews/{2**63}", {}),
        ("GET", "/tag_suggestions", {"params": {"offset": 2**63}}),
        ("POST", "/tag_suggestions/bulk-reopen", {"json": {"ids": [2**63]}}),
        ("GET", "/reviews/preview", {"params": {"set_id": -(2**63) - 1}}),
    ],
)
def test_a_declared_integer_past_sqlite_is_a_422(int_env, method, path, kwargs):
    """A path, query and body list integer: refused by validation, naming
    the bound, before any query runs."""
    r = int_env.owner.request(method, f"{API}{path}", **kwargs)
    assert r.status_code == 422, (r.status_code, r.text[:500])
    assert any(
        error.get("type")
        in {"less_than_equal", "less_than", "greater_than_equal", "greater_than"}
        for error in r.json()["detail"]
    ), r.text[:500]


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", f"/pictures/{2**63}/faces", None),
        ("GET", f"/pictures/{2**63}/text", None),
        ("GET", f"/pictures/{2**63}.png", None),
        ("POST", "/picture_sets/{set_id}/members", {"picture_ids": [2**63]}),
        ("PUT", "/picture_sets/{set_id}/members", {"picture_ids": [2**63]}),
    ],
)
def test_a_parsed_id_past_sqlite_is_a_400(int_env, method, path, body):
    """Ids a handler parses itself: a ``str`` path segment and a ``dict``
    body list. Each was a 500 (``OverflowError`` in the driver)."""
    url = f"{API}{path.format(set_id=int_env.set_id)}"
    r = int_env.owner.request(method, url, json=body)
    assert r.status_code == 400, (method, url, r.status_code, r.text[:300])
    assert str(2**63) not in r.text


def test_picture_set_parent_id_past_sqlite_is_a_400(int_env):
    r = int_env.owner.post(
        f"{API}/picture_sets", json={"name": "Child", "parent_set_id": 2**63}
    )
    assert r.status_code == 400, (r.status_code, r.text[:300])
    assert str(2**63) not in r.text


@pytest.mark.parametrize("bad", ["5e0", "5.0", 2**63, True, [5], None])
def test_face_routes_refuse_an_id_the_scope_check_cannot_read(int_env, bad):
    """The face assign/unassign routes parse ``face_ids`` and ``picture_ids``
    once. SQLite matches ``"5e0"`` against INTEGER id 5 while ``int()``
    refuses it, so an entry the scope check skipped would reach the write."""
    url = f"{API}/characters/{int_env.character_id}/faces"
    for method in ("POST", "DELETE"):
        for key in ("face_ids", "picture_ids"):
            r = int_env.owner.request(method, url, json={key: [bad]})
            assert r.status_code == 400, (method, key, bad, r.status_code, r.text)
            assert r.json()["detail"] == f"{key} must be a list of ids"
