"""What the config routes (pixlstash/routes/config.py) answer for a bad input.

- An integer SQLite cannot store is a 422 from validation, naming the bound,
  never a 500 from the driver (``OverflowError`` while binding it).
- A watermark upload Pillow cannot read is a 400 with a fixed sentence: the
  parser's own text (offsets, internal names) stays in the server log.
- The shared-picture batch check answers for as many ids as the grid sends,
  however many that is: the lookup is chunked under SQLite's variable limit.
- A config patch with a value a setting cannot hold is a 400 whose detail
  names only the setting, never the value sent (nor a 500 for a dict, nor an
  overflow at commit); a patch naming no setting does not echo the name.
"""

import json
import logging
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event as sa_event

from pixlstash.server import Server
from pixlstash.utils.sql_chunking import SQLITE_ID_CHUNK

API = "/api/v1"
_TOO_BIG = 2**63


@pytest.fixture(scope="module")
def env():
    """One Server and one owner login."""
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
        yield SimpleNamespace(server=server, owner=owner)
    finally:
        server.__exit__(None, None, None)
        tmp.cleanup()


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("DELETE", f"/users/me/token/{_TOO_BIG}", {}),
        ("PATCH", f"/users/me/token/{_TOO_BIG}", {"json": {"watermark": False}}),
        (
            "PUT",
            f"/users/me/token/{_TOO_BIG}/libraries",
            {"json": {"all_libraries": True}},
        ),
        (
            "POST",
            "/users/me/token",
            {
                "json": {
                    "scope": "READ",
                    "resource_type": "picture",
                    "resource_id": _TOO_BIG,
                }
            },
        ),
        (
            "POST",
            "/users/me/shared-picture-ids/batch",
            {"json": {"picture_ids": [1, _TOO_BIG]}},
        ),
        (
            "DELETE",
            "/users/me/tokens/by-resource",
            {"params": {"resource_type": "picture", "resource_id": _TOO_BIG}},
        ),
        (
            "GET",
            "/server-config/scrapheap-retention/impact",
            {"params": {"days": _TOO_BIG}},
        ),
        (
            "PATCH",
            "/server-config/scrapheap-retention",
            {"json": {"scrapheap_retention_days": _TOO_BIG}},
        ),
        (
            "DELETE",
            "/server-config/ghost-retention/model-ghosts",
            {"params": {"expected": _TOO_BIG}},
        ),
    ],
)
def test_an_integer_past_sqlite_is_a_422(env, method, path, kwargs):
    r = env.owner.request(method, f"{API}{path}", **kwargs)
    assert r.status_code == 422, (method, path, r.status_code, r.text[:400])
    assert any(
        error.get("type") in {"less_than_equal", "less_than"}
        for error in r.json()["detail"]
    ), r.text[:400]


def test_an_unreadable_watermark_is_refused_with_a_fixed_sentence(env, caplog):
    # A PNG signature followed by bytes that are no PNG: Pillow's message
    # for it would name the chunk it choked on.
    junk = b"\x89PNG\r\n\x1a\n" + b"example-not-an-image" * 4
    with caplog.at_level(logging.WARNING, logger="pixlstash.routes.config"):
        r = env.owner.post(
            f"{API}/users/me/watermark",
            files={"file": ("mark.png", junk, "image/png")},
        )
    assert r.status_code == 400, r.text
    assert r.json() == {"detail": "The file is not a readable PNG, JPEG or WebP image."}
    refused = [rec for rec in caplog.records if "watermark upload" in rec.message]
    assert refused and refused[0].exc_info, "the parse failure is not logged"


def test_the_shared_picture_check_answers_past_the_sqlite_variable_limit(env):
    """Run against SQLite's historical 999-variable floor (some builds still
    ship it), so a single unchunked ``IN`` fails here as it would there."""
    r = env.owner.post(
        f"{API}/users/me/token",
        json={"scope": "READ", "resource_type": "picture", "resource_id": 2000},
    )
    assert r.status_code == 200, r.text
    ids = list(range(1, 2 * SQLITE_ID_CHUNK + 300))
    engine = env.server.hub_engine._engine

    def _set_limit(dbapi_conn, _record):
        dbapi_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)

    sa_event.listen(engine, "connect", _set_limit)
    engine.dispose()
    try:
        r = env.owner.post(
            f"{API}/users/me/shared-picture-ids/batch", json={"picture_ids": ids}
        )
    finally:
        sa_event.remove(engine, "connect", _set_limit)
        engine.dispose()
    assert r.status_code == 200, (r.status_code, r.text[:400])
    assert r.json() == {"shared_ids": [2000]}


_NOT_AN_INT = ["example-abc", {}, [1], 2**70, "9" * 30, "nan", float("inf"), True]
_NOT_A_NUMBER = ["example-abc", {}, [1], "nan", "inf", float("inf"), "1e400", True]
# The sidebar sizes ignore text that is not a number (as before); only a value
# that parses but cannot be stored is refused.
_NOT_A_SIZE = [2**70, "9" * 30, float("inf")]


def _patch(env, body: dict):
    """PATCH the owner's config; NaN and Infinity go out as JSON allows them."""
    return env.owner.patch(
        f"{API}/users/me/config",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )


@pytest.mark.parametrize(
    "key, value, detail",
    [
        *[
            (key, value, f"{key} must be an integer")
            for key in ("similarity_character", "columns", "thumbnail_size_level")
            for value in _NOT_AN_INT
        ],
        *[
            (key, value, f"{key} must be a number")
            for key in ("max_vram_gb", "stack_strictness")
            for value in _NOT_A_NUMBER
        ],
        *[
            (key, value, f"{key} must be an integer")
            for key in ("sidebar_thumbnail_size", "sidebar_width")
            for value in _NOT_A_SIZE
        ],
    ],
)
def test_a_setting_refuses_a_value_it_cannot_hold_naming_only_the_key(
    env, key, value, detail
):
    r = _patch(env, {key: value})
    assert r.status_code == 400, (key, value, r.status_code, r.text[:400])
    assert r.json() == {"detail": detail}


def test_a_patch_naming_no_setting_does_not_echo_the_name(env):
    r = _patch(env, {"example-no-such-setting": 1})
    assert r.status_code == 400, r.text
    assert r.json() == {"detail": "The patch names a setting that does not exist."}


@pytest.mark.parametrize(
    "key, value, stored",
    [
        ("similarity_character", "3", 3),
        ("columns", 5, 5),
        ("thumbnail_size_level", "4", 4),
        ("max_vram_gb", "1.5", 1.5),
        ("stack_strictness", 0.5, 0.5),
        ("sidebar_thumbnail_size", "40", 40),
        ("sidebar_width", 250, 250),
    ],
)
def test_a_setting_still_saves_a_value_it_can_hold(env, key, value, stored):
    r = _patch(env, {key: value})
    assert r.status_code == 200, (key, value, r.status_code, r.text[:400])
    r = env.owner.get(f"{API}/users/me/config")
    assert r.status_code == 200, r.text
    assert r.json()[key] == stored
