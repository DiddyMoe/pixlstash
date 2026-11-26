"""Characters and picture-listing routes: no request integer reaches SQLite
outside the range SQLite stores.

SQLite's INTEGER is a signed 64-bit value and the driver raises
``OverflowError`` binding a larger one, which reached the client as a 500. A
declared integer is :data:`~pixlstash.utils.sqlite_ints.SqliteInt`, so
validation answers 422; an id a handler reads itself (a ``str`` path segment, a
``str`` query value, a ``dict`` body) goes through
:func:`~pixlstash.utils.sqlite_ints.parse_sqlite_int`, so the handler's
existing refusal answers 400. ``tests/test_route_int_bounds.py`` walks the
declared ones; this file pins the hand-parsed ones, which that walk cannot see,
at the HTTP boundary of ``routes/characters.py`` and ``routes/pictures/``
(``_listing``, ``_search``, ``_misc``).
"""

import json
import tempfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from pixlstash.db_models import Face, Picture, SortMechanism
from pixlstash.image_plugins.service import PictureNotFoundError
from pixlstash.routes.pictures import _misc as pictures_misc
from pixlstash.server import Server
from pixlstash.services import plugin_service
from pixlstash.utils.service.smart_set_refresher import rule_picture_ids
from tests.utils import stop_planner_and_settle

API = "/api/v1"
BIG = 2**63


@pytest.fixture(scope="module")
def env():
    """One Server with the planner stopped, an owner login, one character,
    one picture set and one picture row with a face on the character (so a
    listing, and a person's listing, reaches its SQL)."""
    tmp = tempfile.TemporaryDirectory()
    config_path = f"{tmp.name}/server-config.json"
    with open(config_path, "w") as handle:
        json.dump({"port": 8000}, handle)
    server = Server(config_path)
    server.__enter__()
    try:
        stop_planner_and_settle(server)
        owner = TestClient(server.api, raise_server_exceptions=False)
        r = owner.post(
            f"{API}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert r.status_code == 200, r.text
        r = owner.post(f"{API}/characters", json={"name": "Int Bounds Person"})
        assert r.status_code == 200, r.text
        character_id = r.json()["character"]["id"]
        r = owner.post(f"{API}/picture_sets", json={"name": "Int Bounds Set"})
        assert r.status_code in {200, 201}, r.text
        set_id = r.json()["picture_set"]["id"]

        def add_picture(session):
            pic = Picture(file_path=f"{tmp.name}/bounds.png", width=8, height=8)
            session.add(pic)
            session.commit()
            # A person with no face returns an empty page before reading its
            # project filter, so the person's listings need one.
            session.add(
                Face(
                    picture_id=pic.id,
                    frame_index=0,
                    face_index=0,
                    character_id=character_id,
                    bbox=[0, 0, 4, 4],
                )
            )
            session.commit()
            return pic.id

        picture_id = server.vault.db.run_task(add_picture)
        yield SimpleNamespace(
            server=server,
            owner=owner,
            character_id=character_id,
            set_id=set_id,
            picture_id=picture_id,
        )
    finally:
        server.__exit__(None, None, None)
        tmp.cleanup()


def _request(env, method, path, params=None, body=None):
    url = f"{API}{path.format(cid=env.character_id, sid=env.set_id)}"
    return env.owner.request(method, url, params=params, json=body)


# Declared integers: validation refuses them before the handler runs.
DECLARED = [
    ("GET", f"/characters/{BIG}", None, None),
    ("PATCH", f"/characters/{BIG}", None, {"name": "x"}),
    ("DELETE", f"/characters/{BIG}", None, None),
    ("GET", f"/characters/{BIG}/faces", None, None),
    ("GET", f"/characters/{BIG}/reference_pictures", None, None),
    ("GET", f"/characters/{BIG}/thumbnail", None, None),
    ("GET", f"/characters/{BIG}/name", None, None),
    ("POST", "/characters/membership", None, {"picture_ids": [BIG]}),
    ("GET", "/pictures", {"offset": BIG}, None),
    ("GET", "/pictures", {"limit": BIG}, None),
    ("GET", "/pictures", {"min_score": BIG}, None),
    ("GET", "/pictures", {"max_score": BIG}, None),
    ("GET", "/pictures", {"character_id": "{cid}", "face_cluster": BIG}, None),
    ("GET", "/pictures/count", {"min_score": BIG}, None),
    ("GET", "/pictures/count", {"max_score": BIG}, None),
    ("GET", "/pictures/count", {"character_id": "{cid}", "face_cluster": BIG}, None),
    ("GET", "/pictures/stream", {"offset": BIG}, None),
    ("GET", "/pictures/stream", {"min_score": BIG}, None),
    ("GET", "/pictures/stream", {"max_score": BIG}, None),
    ("GET", "/pictures/search", {"query": "x", "offset": BIG}, None),
    ("GET", "/pictures/search", {"query": "x", "limit": BIG}, None),
    # The search declares the listing's filters (``PictureListFilters``), so it
    # refuses a score as the listing does.
    ("GET", "/pictures/search", {"query": "x", "min_score": BIG}, None),
    ("GET", "/pictures/search", {"query": "x", "max_score": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"set_id": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"set_ids": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"min_score": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"max_score": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"min_group_size": BIG}, None),
    ("GET", "/open-with", {"picture_id": BIG}, None),
    ("POST", "/open-with", None, {"picture_id": BIG, "other": True}),
]


def _ids(cases):
    return [f"{m} {p} {q or ''} {b or ''}" for m, p, q, b in cases]


@pytest.mark.parametrize("method, path, params, body", DECLARED, ids=_ids(DECLARED))
def test_a_declared_integer_past_sqlite_is_a_422(env, method, path, params, body):
    if params:
        params = {
            k: v.format(cid=env.character_id) if isinstance(v, str) else v
            for k, v in params.items()
        }
    r = _request(env, method, path, params, body)
    assert r.status_code == 422, (r.status_code, r.text[:400])
    assert any(
        error.get("type") in {"less_than_equal", "greater_than_equal"}
        for error in r.json()["detail"]
    ), r.text[:400]


# Ids a handler parses itself: each one reached the driver as an int.
PARSED = [
    ("GET", f"/characters/{BIG}/summary", None, None),
    ("GET", "/characters/UNASSIGNED/summary", {"project_id": BIG}, None),
    ("GET", "/characters/{cid}/summary", {"project_id": BIG}, None),
    ("GET", "/characters", {"project_id": BIG}, None),
    ("PATCH", "/characters/{cid}", None, {"thumbnail_picture_id": BIG}),
    ("PATCH", "/characters/{cid}", None, {"project_ids": [BIG]}),
    ("PATCH", "/characters/{cid}", None, {"project_id": BIG}),
    ("POST", "/characters", None, {"name": "Big project", "project_ids": [BIG]}),
    ("POST", "/characters", None, {"name": "Big project 2", "project_id": BIG}),
    ("POST", f"/pictures/{BIG}/open-location", None, None),
    ("POST", "/pictures/plugins/example-plugin", None, {"picture_ids": [BIG]}),
    ("GET", "/pictures", {"character_id": BIG}, None),
    ("GET", "/pictures/count", {"character_id": BIG}, None),
    (
        "GET",
        "/pictures",
        {"reference_character_id": BIG, "sort": "CHARACTER_LIKENESS"},
        None,
    ),
    ("GET", "/pictures/search", {"query": "x", "character_id": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"character_id": BIG}, None),
    ("GET", "/pictures/likeness-groups", {"project_id": BIG}, None),
    ("GET", "/pictures/stats", {"character_id": BIG}, None),
    ("GET", "/pictures/stats", {"project_id": BIG}, None),
]


@pytest.mark.parametrize("method, path, params, body", PARSED, ids=_ids(PARSED))
def test_a_parsed_id_past_sqlite_is_a_400(env, method, path, params, body):
    r = _request(env, method, path, params, body)
    assert r.status_code == 400, (r.status_code, r.text[:400])
    assert str(BIG) not in r.text


# Read by the shared ``PredicateFilter.from_query_params`` parser, which every
# listing route calls before its own parsing: a 422 naming the key only.
PREDICATE_PARSED = [
    ("/pictures/stats", {"min_score": BIG}, "min_score"),
    ("/pictures/stats", {"max_score": BIG}, "max_score"),
]


@pytest.mark.parametrize(
    "route, params, name", PREDICATE_PARSED, ids=[str(c) for c in PREDICATE_PARSED]
)
def test_a_score_past_sqlite_is_refused_by_the_shared_parser(env, route, params, name):
    r = env.owner.get(f"{API}{route}", params=params)
    assert r.status_code == 422, (r.status_code, r.text[:400])
    assert r.json()["detail"] == f"Invalid {name}: must be an integer"


# Listing filters a handler reads out of the raw query string. Some are
# refused (400), some fall back to "no such filter value" as a malformed
# value of the same key always has; none may reach the driver.
LISTING_PARSED = [
    {"character_id": BIG},
    {"character_ids": BIG},
    {"set_id": BIG},
    {"set_ids": BIG},
    {"set_id": "{sid}", "set_mode": "difference", "base_set_id": BIG},
    {"reference_character_id": BIG, "sort": "CHARACTER_LIKENESS"},
    {"reference_folder_id": BIG},
    {"id": BIG},
    {"project_id": BIG},
    {"character_id": "UNASSIGNED", "project_id": BIG},
    {"character_id": "UNASSIGNED", "id": BIG},
    {"min_score": BIG},
    {"max_score": BIG},
    {"offset": BIG},
    {"limit": BIG},
]
LISTING_ROUTES = ["/pictures", "/pictures/count", "/pictures/stream", "/pictures/stats"]


@pytest.mark.parametrize("route", LISTING_ROUTES)
@pytest.mark.parametrize("params", LISTING_PARSED, ids=[str(p) for p in LISTING_PARSED])
def test_a_listing_filter_past_sqlite_never_reaches_the_driver(env, route, params):
    params = {
        k: v.format(sid=env.set_id) if isinstance(v, str) else v
        for k, v in params.items()
    }
    r = env.owner.get(f"{API}{route}", params=params)
    assert r.status_code < 500, (r.status_code, r.text[:400])


@pytest.mark.parametrize("params", LISTING_PARSED, ids=[str(p) for p in LISTING_PARSED])
def test_a_search_filter_past_sqlite_never_reaches_the_driver(env, params):
    params = {
        k: v.format(sid=env.set_id) if isinstance(v, str) else v
        for k, v in params.items()
    }
    r = env.owner.get(f"{API}/pictures/search", params={"query": "x", **params})
    assert r.status_code < 500, (r.status_code, r.text[:400])


# The project filter as the branches that read it see it: a person's own
# listing (one person, several), and the likeness sort, alone and in the
# unassigned scope.
PERSON_PROJECT_PARSED = [
    {"character_id": "{cid}", "project_id": BIG},
    {"character_ids": "{cid}", "project_id": BIG},
    {
        "sort": "CHARACTER_LIKENESS",
        "reference_character_id": "{cid}",
        "project_id": BIG,
    },
    {
        "sort": "CHARACTER_LIKENESS",
        "reference_character_id": "{cid}",
        "character_id": "UNASSIGNED",
        "project_id": BIG,
    },
]


@pytest.mark.parametrize("route", ["/pictures", "/pictures/count", "/pictures/stream"])
@pytest.mark.parametrize(
    "params", PERSON_PROJECT_PARSED, ids=[str(p) for p in PERSON_PROJECT_PARSED]
)
def test_a_person_listing_project_past_sqlite_never_reaches_the_driver(
    env, route, params
):
    params = {
        k: v.format(cid=env.character_id) if isinstance(v, str) else v
        for k, v in params.items()
    }
    r = env.owner.get(f"{API}{route}", params=params)
    assert r.status_code < 500, (r.status_code, r.text[:400])


# A smart set's rule is a raw query string handed to the listing selector,
# with no route declaring its integers: the selector's own parse refuses them.
@pytest.mark.parametrize("key", ["offset", "limit"])
def test_a_smart_rule_page_bound_past_sqlite_is_refused(env, key):
    with pytest.raises(HTTPException) as refused:
        rule_picture_ids(env.server, f"{key}={BIG}", [])
    assert refused.value.status_code == 400
    assert refused.value.detail == "Invalid offset or limit"


LIKENESS_PARSED = [
    {"character_id": "UNASSIGNED", "project_id": BIG},
    {"character_id": BIG},
    {"project_id": BIG},
]


@pytest.mark.parametrize(
    "params", LIKENESS_PARSED, ids=[str(p) for p in LIKENESS_PARSED]
)
def test_a_likeness_group_filter_past_sqlite_never_reaches_the_driver(env, params):
    r = env.owner.get(f"{API}/pictures/likeness-groups", params=params)
    assert r.status_code < 500, (r.status_code, r.text[:400])


# A refusal names a value only once it is known good: an unknown sort or
# combine mode, or an unknown plugin, is refused in a fixed sentence.
UNKNOWN_NAME = "example-unknown-name"


@pytest.mark.parametrize(
    "path, params",
    [
        ("/pictures", {"sort": UNKNOWN_NAME}),
        ("/pictures/search", {"query": "x", "sort": UNKNOWN_NAME}),
        ("/picture_sets/{set_id}", {"sort": UNKNOWN_NAME}),
    ],
)
def test_an_unknown_sort_is_refused_without_echoing_it(env, path, params):
    r = env.owner.get(f"{API}{path.format(set_id=env.set_id)}", params=params)
    assert r.status_code == 400, (r.status_code, r.text[:400])
    assert r.json()["detail"] == "Invalid sort mechanism"
    assert UNKNOWN_NAME not in r.text


def test_an_unknown_sort_key_stays_out_of_the_parse_error():
    # The handlers above answer their own sentence; the parse error beneath
    # them carries none of the caller's text either, should one pass it on.
    with pytest.raises(ValueError) as refused:
        SortMechanism.from_string(UNKNOWN_NAME)
    assert UNKNOWN_NAME not in str(refused.value)


def test_an_unknown_combine_mode_is_refused_without_echoing_it(env):
    r = env.owner.post(
        f"{API}/characters/likeness-search",
        params={"combine": UNKNOWN_NAME},
        files=[("files", ("face.png", b"not-an-image", "image/png"))],
    )
    assert r.status_code == 400, (r.status_code, r.text[:400])
    assert r.json()["detail"].startswith("Invalid combine mode. Must be one of: ")
    assert UNKNOWN_NAME not in r.text


@pytest.mark.parametrize(
    "error, status, detail",
    [
        (
            plugin_service.PluginNotFoundError(f"Plugin not found: {UNKNOWN_NAME!r}"),
            404,
            "Plugin not found",
        ),
        (
            RuntimeError(f"{UNKNOWN_NAME} raised inside the plugin"),
            500,
            "The plugin failed; the log says why.",
        ),
    ],
    ids=["not-found", "failed"],
)
def test_a_plugin_refusal_is_a_fixed_sentence(
    env, monkeypatch, caplog, error, status, detail
):
    async def run_plugin_on_pictures(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        pictures_misc.plugin_service, "run_plugin_on_pictures", run_plugin_on_pictures
    )
    r = env.owner.post(
        f"{API}/pictures/plugins/{UNKNOWN_NAME}",
        json={"picture_ids": [env.picture_id]},
    )
    assert r.status_code == status, (r.status_code, r.text[:400])
    assert r.json()["detail"] == detail
    assert str(error) not in r.text
    # The cause goes to the log, with the plugin's name.
    assert any(UNKNOWN_NAME in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


def test_a_plugin_run_on_a_missing_picture_names_that_picture(env, monkeypatch):
    """An id with no row is the caller's mistake: a 404 naming that id, which
    is one of the parsed ``picture_ids``. Wrong if it reads ``Plugin not
    found`` or comes back as the failed run's 500."""
    missing = env.picture_id + 1000

    async def run_plugin_on_pictures(*args, **kwargs):
        raise PictureNotFoundError(missing)

    monkeypatch.setattr(
        pictures_misc.plugin_service, "run_plugin_on_pictures", run_plugin_on_pictures
    )
    r = env.owner.post(
        f"{API}/pictures/plugins/{UNKNOWN_NAME}",
        json={"picture_ids": [missing]},
    )
    assert r.status_code == 404, (r.status_code, r.text[:400])
    assert r.json()["detail"] == f"Picture not found: {missing}"
