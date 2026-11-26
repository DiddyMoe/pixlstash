"""A stored picture value its column does not declare is null, not a 500.

SQLite does not enforce a column's type: a row written by a tool outside the
app, or before a check existed, can hold a REAL ``inf`` or a BLOB in an INTEGER
column, or a BLOB in a TEXT one. The grid's response model refused such a
value and JSON cannot carry ``inf``, so one bad row failed the whole page (and
the raw field route failed outright). Every route here serves the value as
null, logs it with the picture and the column, and serves the rest of the row
and every other row as stored.
"""

import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import pixlstash.routes.pictures._listing as listing_routes
import pixlstash.routes.pictures._search as search_routes
from pixlstash.database import DBPriority
from pixlstash.db_models.face import Face
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_likeness import PictureLikeness
from pixlstash.server import Server

_LISTING_LOGGER = "pixlstash.routes.pictures._listing"

# A column, the value stored in it, and the column type it breaks.
BAD_VALUES = {
    "inf-score": ("score", float("inf")),
    "fractional-score": ("score", 3.5),
    "blob-score": ("score", b"\x00\xff"),
    "blob-width": ("width", b"\x00\xff"),
    "text-height": ("height", "wide"),
    "inf-smart-score": ("smart_score", float("inf")),
    "blob-format": ("format", b"\x00\xff"),
}
# What each picture holds before one value is broken.
STORED = {"width": 64, "height": 48, "score": 3, "smart_score": 2.5, "format": "png"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("stored-picture-values")
    config = root / "server-config.json"
    # No background work: every row here is written by hand.
    config.write_text(json.dumps({"port": 8000, "disable_background_workers": True}))
    server = Server(str(config))
    try:
        client = TestClient(server.api)
        resp = client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        )
        assert resp.status_code == 200, resp.text
        yield client, server
    finally:
        server.close()


def _seed(server, case):
    """A bad picture, its look-alike and an unrelated good one, all as STORED.

    The bad one and its twin are alike, so the likeness groups list them.
    """

    def insert(session):
        ids = {}
        for role in ("bad", "twin", "good"):
            pic = Picture(file_path=f"/nonexistent/{case}-{role}.png", **STORED)
            session.add(pic)
            session.flush()
            ids[role] = int(pic.id)
        low, high = sorted((ids["bad"], ids["twin"]))
        session.add(
            PictureLikeness(
                picture_id_a=low, picture_id_b=high, likeness=0.9, metric="test"
            )
        )
        session.commit()
        return ids

    return server.vault.db.run_task(insert, priority=DBPriority.IMMEDIATE)


def _store_raw(server, column, picture_id, value):
    def write(session):
        session.execute(
            text(f"UPDATE picture SET {column} = :value WHERE id = :id"),
            {"value": value, "id": picture_id},
        )
        session.commit()

    server.vault.db.run_task(write, priority=DBPriority.IMMEDIATE)


@pytest.fixture(params=list(BAD_VALUES))
def broken(env, request):
    """A seeded trio whose bad picture holds the case's value."""
    client, server = env
    column, value = BAD_VALUES[request.param]
    ids = _seed(server, request.param)
    _store_raw(server, column, ids["bad"], value)
    try:
        yield client, server, ids, column
    finally:
        _heap(server, ids)


def _heap(server, ids):
    """Out of every later listing: each case seeds its own trio."""

    def heap(session):
        session.execute(
            text("UPDATE picture SET deleted = 1 WHERE id IN (:a, :b, :c)"),
            {"a": ids["bad"], "b": ids["twin"], "c": ids["good"]},
        )
        session.commit()

    server.vault.db.run_task(heap, priority=DBPriority.IMMEDIATE)


def _logged(caplog, picture_id, column):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LISTING_LOGGER
        and f"Picture {picture_id} has a stored {column} " in r.getMessage()
    ]


def _assert_row(row, column, *, broken):
    expected = {**STORED, column: None} if broken else STORED
    assert {key: row[key] for key in STORED} == expected, row


def _by_id(rows):
    return {row["id"]: row for row in rows}


def test_the_grid_serves_the_bad_value_as_null_and_every_row(broken, caplog):
    client, _server, ids, column = broken
    with caplog.at_level(logging.WARNING, logger=_LISTING_LOGGER):
        listed = client.get("/pictures", params={"fields": "grid"})
    assert listed.status_code == 200, listed.text
    rows = _by_id(listed.json())
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["good"]], column, broken=False)
    assert _logged(caplog, ids["bad"], column)


def test_the_stream_serves_the_bad_value_as_null_and_every_row(broken):
    client, _server, ids, column = broken
    streamed = client.get("/pictures/stream", params={"fields": "grid"})
    assert streamed.status_code == 200, streamed.text
    rows = _by_id(streamed.json()["pictures"])
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["good"]], column, broken=False)


def test_the_metadata_serves_the_bad_value_as_null(broken):
    client, _server, ids, column = broken
    bad = client.get(f"/pictures/{ids['bad']}/metadata")
    assert bad.status_code == 200, bad.text
    _assert_row(bad.json(), column, broken=True)
    good = client.get(f"/pictures/{ids['good']}/metadata")
    _assert_row(good.json(), column, broken=False)


def test_the_raw_field_serves_the_bad_value_as_null(broken, caplog):
    client, _server, ids, column = broken
    with caplog.at_level(logging.WARNING, logger=_LISTING_LOGGER):
        bad = client.get(f"/pictures/{ids['bad']}/{column}")
    assert bad.status_code == 200, bad.text
    assert bad.json() == {column: None}
    assert _logged(caplog, ids["bad"], column)
    good = client.get(f"/pictures/{ids['good']}/{column}")
    assert good.json() == {column: STORED[column]}


def test_the_likeness_groups_serve_the_bad_value_as_null(broken):
    client, _server, ids, column = broken
    grouped = client.get("/pictures/likeness-groups", params={"threshold": 0.5})
    assert grouped.status_code == 200, grouped.text
    rows = _by_id(grouped.json())
    # The positive control: the trio's pair is grouped at all.
    assert {ids["bad"], ids["twin"]} <= set(rows), rows
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["twin"]], column, broken=False)


def test_the_search_serves_the_bad_value_as_null(broken, monkeypatch):
    """The search's rows, past the embedding match it is handed."""
    client, server, ids, column = broken

    def matches(_server, _request, _query, **_kwargs):
        def load(session):
            pics = {pic.id: pic for pic in Picture.find(session, id=list(ids.values()))}
            return [(pics[ids["bad"]], 0.9), (pics[ids["good"]], 0.8)]

        return server.vault.db.run_task(load, priority=DBPriority.IMMEDIATE)

    monkeypatch.setattr(search_routes, "search_results", matches)
    found = client.get("/pictures/search", params={"query": "example query"})
    assert found.status_code == 200, found.text
    rows = _by_id(found.json())
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["good"]], column, broken=False)


def _character_with_faces(client, server, ids):
    """A person whose reference face is on the good picture, and faces like it
    (no person) on the bad picture and its twin, for the likeness sorts.

    A face with a person moves the smart-score anchors, which clears every
    stored smart score, so the trio's are written back as they were.
    """
    resp = client.post("/characters", json={"name": f"test-likeness-{ids['bad']}"})
    assert resp.status_code == 200, resp.text
    character_id = resp.json()["character"]["id"]
    along = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    near = [0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def smart_scores(session):
        return {
            role: session.execute(
                text("SELECT smart_score FROM picture WHERE id = :id"),
                {"id": ids[role]},
            ).scalar_one()
            for role in ("bad", "twin", "good")
        }

    stored = server.vault.db.run_task(smart_scores, priority=DBPriority.IMMEDIATE)

    def add(session):
        for role, person, vector in (
            ("good", character_id, along),
            ("bad", None, near),
            ("twin", None, near),
        ):
            session.add(
                Face(
                    picture_id=ids[role],
                    frame_index=0,
                    face_index=0,
                    character_id=person,
                    bbox=[0, 0, 10, 10],
                    features=np.asarray(vector, dtype=np.float32).tobytes(),
                )
            )
        session.commit()

    server.vault.db.run_task(add, priority=DBPriority.IMMEDIATE)
    for role, value in stored.items():
        _store_raw(server, "smart_score", ids[role], value)
    return character_id


def test_the_stack_serves_the_bad_value_as_null(broken):
    client, _server, ids, column = broken
    stacked = client.post("/stacks", json={"picture_ids": [ids["bad"], ids["twin"]]})
    assert stacked.status_code == 200, stacked.text
    stack_id = stacked.json()["id"]
    for fields in ("grid", "full"):
        resp = client.get(f"/stacks/{stack_id}/pictures", params={"fields": fields})
        assert resp.status_code == 200, (fields, resp.text)
        rows = _by_id(resp.json())
        assert set(rows) == {ids["bad"], ids["twin"]}, fields
        _assert_row(rows[ids["bad"]], column, broken=True)
        _assert_row(rows[ids["twin"]], column, broken=False)


def test_the_likeness_sort_serves_the_bad_value_as_null(broken):
    """The grid's likeness sort, which reads its rows itself."""
    client, server, ids, column = broken
    character_id = _character_with_faces(client, server, ids)
    for fields in ("grid", None):
        params = {
            "sort": "CHARACTER_LIKENESS",
            "reference_character_id": character_id,
            "id": [ids["bad"], ids["twin"], ids["good"]],
        }
        if fields:
            params["fields"] = fields
        resp = client.get("/pictures", params=params)
        assert resp.status_code == 200, (fields, resp.text)
        rows = _by_id(resp.json())
        assert set(rows) == set(ids.values()), fields
        _assert_row(rows[ids["bad"]], column, broken=True)
        _assert_row(rows[ids["twin"]], column, broken=False)


def test_the_set_likeness_sort_serves_the_bad_value_as_null(broken):
    """A set's likeness sort, which reads its rows a second way."""
    client, server, ids, column = broken
    character_id = _character_with_faces(client, server, ids)
    created = client.post("/picture_sets", json={"name": f"test-set-{ids['bad']}"})
    assert created.status_code == 200, created.text
    set_id = created.json()["picture_set"]["id"]
    added = client.post(
        f"/picture_sets/{set_id}/members",
        json={"picture_ids": [ids["bad"], ids["twin"]]},
    )
    assert added.status_code == 200, added.text
    resp = client.get(
        f"/picture_sets/{set_id}",
        params={
            "sort": "CHARACTER_LIKENESS",
            "reference_character_id": character_id,
        },
    )
    assert resp.status_code == 200, resp.text
    rows = _by_id(resp.json()["pictures"])
    assert set(rows) == {ids["bad"], ids["twin"]}
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["twin"]], column, broken=False)


def test_the_patch_echo_is_the_committed_row(broken):
    """The echo carries the edit, read through the column types, no blobs."""
    client, _server, ids, column = broken
    edited = client.patch(
        f"/pictures/{ids['bad']}", json={"description": "test-edited description"}
    )
    assert edited.status_code == 200, edited.text
    echo = edited.json()["picture"]
    assert echo["id"] == ids["bad"]
    assert echo["description"] == "test-edited description"
    _assert_row(echo, column, broken=True)
    assert set(echo) == set(Picture.metadata_fields()), echo
    # A PATCH that changes nothing echoes the row the same way.
    unchanged = client.patch(
        f"/pictures/{ids['twin']}", json={"score": STORED["score"]}
    )
    assert unchanged.status_code == 200, unchanged.text
    twin = unchanged.json()["picture"]
    _assert_row(twin, column, broken=False)
    assert set(twin) == set(Picture.metadata_fields()), twin


def _set_of(client, ids):
    """A set holding the bad picture and its twin."""
    created = client.post("/picture_sets", json={"name": f"test-set-{ids['bad']}"})
    assert created.status_code == 200, created.text
    set_id = created.json()["picture_set"]["id"]
    added = client.post(
        f"/picture_sets/{set_id}/members",
        json={"picture_ids": [ids["bad"], ids["twin"]]},
    )
    assert added.status_code == 200, added.text
    return set_id


def test_the_set_serves_the_bad_value_as_null(broken):
    """A set in its default order, which reads its rows a third way."""
    client, _server, ids, column = broken
    set_id = _set_of(client, ids)
    for fields in ("grid", "full"):
        resp = client.get(f"/picture_sets/{set_id}", params={"fields": fields})
        assert resp.status_code == 200, (fields, resp.text)
        rows = _by_id(resp.json()["pictures"])
        assert set(rows) == {ids["bad"], ids["twin"]}, fields
        _assert_row(rows[ids["bad"]], column, broken=True)
        _assert_row(rows[ids["twin"]], column, broken=False)
        # The set's rows still leave out the path and the blobs.
        assert not {"file_path", "thumbnail", "image_embedding"} & set(
            rows[ids["bad"]]
        ), fields


def test_the_set_smart_sort_serves_the_bad_value_as_null(broken):
    """A set sorted by smart score, which reads its rows a fourth way."""
    client, _server, ids, column = broken
    set_id = _set_of(client, ids)
    resp = client.get(f"/picture_sets/{set_id}", params={"sort": "SMART_SCORE"})
    assert resp.status_code == 200, resp.text
    rows = _by_id(resp.json()["pictures"])
    assert set(rows) == {ids["bad"], ids["twin"]}
    _assert_row(rows[ids["bad"]], column, broken=True)
    _assert_row(rows[ids["twin"]], column, broken=False)


def test_a_fractional_thumbnail_geometry_is_null(env, caplog):
    """The thumbnail batch reads its numbers through the same rule."""
    client, server = env
    ids = _seed(server, "fractional-geometry")
    try:
        geometry = {
            "thumbnail_width": 32,
            "thumbnail_height": 24,
            "square_crop_x": 4,
            "square_crop_y": 0,
            "square_crop_side": 24,
        }
        for role in ("bad", "good"):
            for name, value in geometry.items():
                _store_raw(server, name, ids[role], value)
        _store_raw(server, "square_crop_x", ids["bad"], 4.5)
        with caplog.at_level(
            logging.WARNING, logger="pixlstash.routes.pictures._thumbnails"
        ):
            resp = client.post(
                "/pictures/thumbnails", json={"ids": [ids["bad"], ids["good"]]}
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert {k: body[str(ids["good"])][k] for k in geometry} == geometry
        assert {k: body[str(ids["bad"])][k] for k in geometry} == {
            **geometry,
            "square_crop_x": None,
        }
        assert [
            r.getMessage()
            for r in caplog.records
            if f"Picture {ids['bad']} has a stored square_crop_x " in r.getMessage()
        ]
    finally:
        _heap(server, ids)


def _metadata_logs(client, caplog, picture_id, column):
    """Read the picture's metadata; the lines logged for *column* by that read."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_LISTING_LOGGER):
        resp = client.get(f"/pictures/{picture_id}/metadata")
    assert resp.status_code == 200, resp.text
    return _logged(caplog, picture_id, column)


def test_a_refused_value_is_logged_once_per_row_and_column(env, caplog):
    client, server = env
    ids = _seed(server, "logged-once")
    try:
        _store_raw(server, "score", ids["bad"], float("inf"))
        assert len(_metadata_logs(client, caplog, ids["bad"], "score")) == 1
        assert _metadata_logs(client, caplog, ids["bad"], "score") == []
        # Another column of the same row is logged.
        _store_raw(server, "width", ids["bad"], b"\x00\xff")
        assert len(_metadata_logs(client, caplog, ids["bad"], "width")) == 1
        assert _metadata_logs(client, caplog, ids["bad"], "score") == []
        # The same column of another row is logged.
        _store_raw(server, "score", ids["twin"], float("inf"))
        assert len(_metadata_logs(client, caplog, ids["twin"], "score")) == 1
    finally:
        _heap(server, ids)


def test_the_record_starts_over_at_its_cap(env, caplog, monkeypatch):
    client, server = env
    ids = _seed(server, "record-cap")
    monkeypatch.setattr(listing_routes, "_REFUSED_STORED_VALUES_LIMIT", 1)
    try:
        _store_raw(server, "score", ids["bad"], float("inf"))
        assert len(_metadata_logs(client, caplog, ids["bad"], "score")) == 1
        assert _metadata_logs(client, caplog, ids["bad"], "score") == []
        # A second refusal fills the record past its cap of one: it starts
        # over, so the first value is logged once more.
        _store_raw(server, "score", ids["twin"], float("inf"))
        assert len(_metadata_logs(client, caplog, ids["twin"], "score")) == 1
        assert len(_metadata_logs(client, caplog, ids["bad"], "score")) == 1
    finally:
        _heap(server, ids)


def test_a_library_switch_starts_the_record_over(env, caplog):
    """The next library reuses row ids, so its damaged rows are logged."""
    client, server = env
    ids = _seed(server, "record-switch")
    try:
        _store_raw(server, "score", ids["bad"], float("inf"))
        assert len(_metadata_logs(client, caplog, ids["bad"], "score")) == 1
        assert _metadata_logs(client, caplog, ids["bad"], "score") == []
        server._clear_thumbnail_runtime_cache()
        assert len(_metadata_logs(client, caplog, ids["bad"], "score")) == 1
    finally:
        _heap(server, ids)
