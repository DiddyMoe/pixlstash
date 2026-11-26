"""A listing asked for a list of fields answers with those fields.

``GET /pictures`` and ``GET /pictures/stream`` take ``fields`` as ``grid``, a
comma list, or nothing. A comma list used to come back with every key the grid
model declares, the ones not asked for as null: the grid's reconnect read asks
for ``id,stack_id`` and paid for some thirty keys a row. ``grid`` and no
``fields`` keep the padded shape the grid and the API clients read, byte for
byte.
"""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.db_models.picture_stack import PictureStack
from pixlstash.routes.pictures._listing import LIBRARY_HEADER, GridPicture
from pixlstash.server import Server

CREATED = datetime(2026, 1, 14, 9, 30, tzinfo=timezone.utc)
IMPORTED = datetime(2026, 1, 15, 18, 2, 11, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A lone picture and a stack of two, every grid value fixed."""
    root = tmp_path_factory.mktemp("listing-field-projection")
    config = root / "server-config.json"
    # No background work: a sweep would write values the bodies pin.
    config.write_text(json.dumps({"port": 8000, "disable_background_workers": True}))
    server = Server(str(config))
    try:
        client = TestClient(server.api)
        resp = client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        )
        assert resp.status_code == 200, resp.text

        def seed(session):
            stack = PictureStack(name="example-stack")
            session.add(stack)
            session.flush()
            ids = {}
            for key, stack_id, position in (
                ("lone", None, None),
                ("cover", stack.id, 0),
                ("member", stack.id, 1),
            ):
                pic = Picture(
                    file_path=f"example-{key}.png",
                    width=64,
                    height=48,
                    format="PNG",
                    score=3,
                    smart_score=2.5,
                    created_at=CREATED,
                    imported_at=IMPORTED,
                    stack_id=stack_id,
                    stack_position=position,
                )
                session.add(pic)
                session.flush()
                ids[key] = pic.id
            session.commit()
            ids["stack"] = stack.id
            return ids

        yield client, server.vault.db.run_task(seed, priority=DBPriority.IMMEDIATE)
    finally:
        server.close()


def _body(resp):
    assert resp.status_code == 200, resp.text
    return resp.content


def _json(value) -> bytes:
    """``value`` as the routes render JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _grid_row(ids, key, **values):
    """A row as the grid model answers it: every key it declares."""
    row = dict.fromkeys(GridPicture.model_fields)
    row.update(
        id=ids[key],
        width=64,
        height=48,
        format="PNG",
        score=3,
        favourite=False,
        smart_score=2.5,
        smartScore=2.5,
        created_at="2026-01-14T09:30:00Z",
        imported_at="2026-01-15T18:02:11Z",
        file_path=f"example-{key}.png",
    )
    row.update(values)
    return row


def _views(ids):
    """Each padded answer, and the body it must stay."""
    lone = _grid_row(ids, "lone")
    cover = _grid_row(ids, "cover", stack_id=ids["stack"], stack_position=0)
    member = _grid_row(ids, "member", stack_id=ids["stack"], stack_position=1)
    # The grid folds the stack into its cover and counts it.
    grid = [{**lone, "stack_count": 0}, {**cover, "stack_count": 2}]
    full = [lone, cover, member]
    lite = [{**row, "file_path": None} for row in grid]
    return {
        "list-grid": ("/pictures", {"fields": "grid"}, grid),
        "list-full": ("/pictures", {}, full),
        "stream-grid": ("/pictures/stream", {"fields": "grid"}, grid),
        "stream-grid-lite": (
            "/pictures/stream",
            {"fields": "grid", "grid_lite": "true"},
            lite,
        ),
        "stream-full": ("/pictures/stream", {}, full),
    }


@pytest.mark.parametrize(
    "view", ["list-grid", "list-full", "stream-grid", "stream-grid-lite", "stream-full"]
)
def test_the_grid_and_the_full_answer_keep_every_declared_key(env, view):
    """``grid`` (lite or not) and no ``fields`` answer byte for byte as before."""
    client, ids = env
    route, params, rows = _views(ids)[view]
    expected = (
        rows
        if route == "/pictures"
        else {"pictures": rows, "done": True, "next_offset": len(rows)}
    )

    assert _body(client.get(route, params=params)) == _json(expected)


def test_a_field_list_lists_only_those_fields(env):
    client, ids = env
    listed = client.get("/pictures", params={"fields": "id,stack_id"})

    assert _body(listed) == _json(
        [
            {"id": ids["lone"], "stack_id": None},
            {"id": ids["cover"], "stack_id": ids["stack"]},
            {"id": ids["member"], "stack_id": ids["stack"]},
        ]
    )


def test_the_reconnect_read_streams_only_its_fields(env):
    """The grid's reconnect read: one ``id`` and ``stack_id`` per card.

    The collapse still counts the stack; the count is not asked for, so it is
    not sent.
    """
    client, ids = env
    streamed = client.get(
        "/pictures/stream",
        params={"fields": "id,stack_id", "stack_leaders_only": "true"},
    )

    assert _body(streamed) == _json(
        {
            "pictures": [
                {"id": ids["lone"], "stack_id": None},
                {"id": ids["cover"], "stack_id": ids["stack"]},
            ],
            "done": True,
            "next_offset": 2,
        }
    )


def test_a_field_list_still_answers_only_what_the_grid_model_declares(env):
    """A name the model does not declare is not served by asking for it.

    ``project_id`` is the case that matters: the stored column can name a
    project the reader may not see, and the model is what keeps it out
    (backend architecture §16.6).
    """
    client, ids = env
    listed = client.get("/pictures", params={"fields": "id,project_id,file_path"})

    assert _body(listed) == _json(
        [
            {"id": ids[key], "file_path": f"example-{key}.png"}
            for key in ("lone", "cover", "member")
        ]
    )


def test_a_field_list_keeps_the_library_header(env):
    client, _ids = env
    grid = client.get("/pictures", params={"fields": "grid"})
    listed = client.get("/pictures", params={"fields": "id"})

    assert grid.headers.get(LIBRARY_HEADER)
    assert listed.headers.get(LIBRARY_HEADER) == grid.headers[LIBRARY_HEADER]
