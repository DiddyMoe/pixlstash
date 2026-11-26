"""The Scrapheap view's ``only_deleted=true`` reaches every query it sends.

The frontend scopes a Scrapheap view with ``only_deleted=true``; the legacy
``character_id=SCRAPHEAP`` is still read. Search, likeness-groups and export
read only the legacy id, so in the Scrapheap a search returned live pictures,
the likeness view grouped live pictures, and an export packaged nothing (the
ids were found, then fetched with the heap excluded).

Also here: the listing routes refuse more than ``LIST_MAX_IDS`` repeated ``id``
params; the frontend chunks a longer list. And they ignore a raw query
name that is a parameter of ``Picture.find`` or ``run_immediate_read_task``
(X-2), which used to reach both through the leftover ``**query_params``.
"""

import gc
import inspect
import json
import os
import tempfile
import time
import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from pixlstash.database import VaultDatabase
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_likeness import PictureLikeness
from pixlstash.routes.pictures._listing import LIST_MAX_IDS
from pixlstash.server import Server
from tests.utils import API_PREFIX, upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")

# The two spellings of the Scrapheap scope: what the frontend sends, and the
# deprecated pseudo character id older clients send.
SCRAPHEAP_SCOPES = [
    pytest.param({"only_deleted": "true"}, id="only_deleted"),
    pytest.param({"character_id": "SCRAPHEAP"}, id="legacy_character_id"),
]


def _server(temp_dir: str, **config) -> tuple[Server, TestClient]:
    server_config_path = os.path.join(temp_dir, "server-config.json")
    with open(server_config_path, "w") as f:
        f.write(json.dumps({"port": 8000, **config}))
    server = Server(server_config_path)
    client = TestClient(server.api)
    resp = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert resp.status_code == 200
    return server, client


def _upload_picture(client, filename):
    with open(os.path.join(PICTURES_DIR, filename), "rb") as f:
        result = upload_pictures_and_wait(
            client, [("file", (filename, f, "image/png"))]
        )
    assert result["status"] == "completed"
    return result["results"][0]["picture_id"]


def _wait_for_export(client, task_id, timeout_s=30):
    start = time.time()
    while time.time() - start < timeout_s:
        resp = client.get(
            f"{API_PREFIX}/pictures/export/status", params={"task_id": task_id}
        )
        assert resp.status_code == 200, resp.text
        status = resp.json()
        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.2)
    raise AssertionError(f"Export task did not finish within {timeout_s}s")


def _exported_image_count(client, task_id) -> int:
    resp = client.get(f"{API_PREFIX}/pictures/export/download/{task_id}")
    assert resp.status_code == 200
    with zipfile.ZipFile(BytesIO(resp.content)) as archive:
        return sum(1 for name in archive.namelist() if name.lower().endswith(".png"))


@pytest.fixture(scope="module")
def heap_library():
    """One live and one scrapheaped picture, both tagged ``zebra``."""
    temp_dir = tempfile.TemporaryDirectory()
    server, client = _server(temp_dir.name)
    try:
        live = _upload_picture(client, "Bad1.png")
        gone = _upload_picture(client, "Bad2.png")
        for pid in (live, gone):
            resp = client.post(f"/pictures/{pid}/tags", json={"tag": "zebra"})
            assert resp.status_code == 200
        assert client.delete(f"/pictures/{gone}").status_code == 200
        yield client, live, gone
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _search_ids(client, **extra) -> set[int]:
    resp = client.get(
        "/pictures/search", params={"query": "zebra", "threshold": 0.01, **extra}
    )
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()}


@pytest.mark.parametrize("scope", SCRAPHEAP_SCOPES)
def test_search_in_the_scrapheap_returns_only_the_heap(heap_library, scope):
    client, live, gone = heap_library
    assert _search_ids(client, **scope) == {gone}
    # The live view still finds the live picture, and only it.
    assert _search_ids(client) == {live}


@pytest.mark.parametrize("scope", SCRAPHEAP_SCOPES)
def test_export_of_selected_heap_pictures_packages_them(heap_library, scope):
    client, _live, gone = heap_library
    resp = client.get("/pictures/export", params={"id": [gone], **scope})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    status = _wait_for_export(client, task_id)
    assert status["status"] == "completed", status
    assert status["total"] == 1
    assert _exported_image_count(client, task_id) == 1


@pytest.mark.parametrize("scope", SCRAPHEAP_SCOPES)
def test_export_of_the_whole_scrapheap_packages_the_heap(heap_library, scope):
    client, _live, _gone = heap_library
    resp = client.get("/pictures/export", params=scope)
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    status = _wait_for_export(client, task_id)
    assert status["status"] == "completed", status
    assert status["total"] == 1
    assert _exported_image_count(client, task_id) == 1


@pytest.fixture(scope="module")
def seeded_library():
    """Hand-placed pictures and likeness pairs, with background work off.

    The likeness pipeline would rewrite hand-placed pairs, so nothing runs.
    """
    temp_dir = tempfile.TemporaryDirectory()
    server, client = _server(temp_dir.name, disable_background_workers=True)
    try:

        def seed(session):
            ids = {}
            for key, deleted in (
                ("live_a", False),
                ("live_b", False),
                ("gone_a", True),
                ("gone_b", True),
            ):
                pic = Picture(
                    file_path=f"{key}.png", width=64, height=64, deleted=deleted
                )
                session.add(pic)
                session.flush()
                ids[key] = pic.id
            for x, y in (("live_a", "live_b"), ("gone_a", "gone_b")):
                a, b = sorted((ids[x], ids[y]))
                session.add(
                    PictureLikeness(
                        picture_id_a=a, picture_id_b=b, likeness=0.95, metric="test"
                    )
                )
            session.commit()
            return ids

        ids = server.vault.db.run_task(seed)
        yield client, ids
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _likeness_ids(client, **extra) -> set[int]:
    resp = client.get(
        f"{API_PREFIX}/pictures/likeness-groups", params={"threshold": 0.9, **extra}
    )
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()}


@pytest.mark.parametrize("scope", SCRAPHEAP_SCOPES)
def test_likeness_groups_in_the_scrapheap_group_only_the_heap(seeded_library, scope):
    client, ids = seeded_library
    assert _likeness_ids(client, **scope) == {ids["gone_a"], ids["gone_b"]}
    assert _likeness_ids(client) == {ids["live_a"], ids["live_b"]}


@pytest.mark.parametrize("path", ["/pictures", "/pictures/stream", "/pictures/count"])
def test_listing_refuses_more_than_the_id_cap(seeded_library, path):
    client, ids = seeded_library
    resp = client.get(path, params={"id": list(range(1, LIST_MAX_IDS + 2))})
    assert resp.status_code == 400, resp.text
    assert f"maximum is {LIST_MAX_IDS}" in resp.json()["detail"]

    # At the cap the request is served, and still filters by its ids.
    at_cap = [ids["live_a"]] + list(range(10_000, 10_000 + LIST_MAX_IDS - 1))
    resp = client.get(path, params={"id": at_cap})
    assert resp.status_code == 200, resp.text


def test_listing_at_the_id_cap_returns_the_named_pictures(seeded_library):
    client, ids = seeded_library
    at_cap = [ids["live_a"], ids["live_b"]] + list(
        range(10_000, 10_000 + LIST_MAX_IDS - 2)
    )
    resp = client.get("/pictures", params={"id": at_cap})
    assert resp.status_code == 200, resp.text
    assert {row["id"] for row in resp.json()} == {ids["live_a"], ids["live_b"]}


def _splat_target_names() -> list[str]:
    """Every name the listing route's leftover-query splat can collide with.

    The route ends in ``run_immediate_read_task(Picture.find, <explicit
    kwargs>, **query_params)``, so a raw query name equal to a parameter of
    either function would either repeat a keyword the route passes itself (a
    TypeError, so a 500) or set one the route never meant a client to set.
    Read from the two signatures, not copied, so a keyword added later is
    covered without editing this list.
    """
    names = set()
    for fn in (Picture.find.__func__, VaultDatabase.run_immediate_read_task):
        for name, param in inspect.signature(fn).parameters.items():
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                names.add(name)
    return sorted(names)


# The ones a client may send by that name: the route reads each from the raw
# query and pops it before the splat. A value for these is a real filter, so it
# is not expected to leave the result unchanged; it only must not 500.
# (``impossible_sources`` is not one: the client's name is
# ``impossible_tag_source``.)
_ROUTE_READ_FILTERS = {
    "face_filter",
    "file_path_prefix",
    "format",
    "limit",
    "max_score",
    "min_score",
    "offset",
    "only_deleted",
    "resolution_bucket",
    "smart_score_bucket",
    "stack_state",
    "tag_state",
    "unscored",
}


@pytest.mark.parametrize("name", _splat_target_names())
@pytest.mark.parametrize("deleted", [False, True], ids=["live", "scrapheap"])
def test_listing_ignores_a_query_name_it_passes_to_find_itself(
    seeded_library, name, deleted
):
    """X-2: ``?select_fields=id`` and the like 500'd the whole listing.

    Every name the route does not read itself must leave the answer exactly
    as it was: a client cannot turn on ``include_deleted``, ``count_only`` or
    a bare-string list filter through the leftover splat.
    """
    client, ids = seeded_library
    scope = {"only_deleted": "true"} if deleted else {}
    expected = (
        {ids["gone_a"], ids["gone_b"]} if deleted else {ids["live_a"], ids["live_b"]}
    )
    params = {**scope, name: "1"}

    listed = client.get("/pictures", params=params)
    counted = client.get("/pictures/count", params=params)
    streamed = client.get("/pictures/stream", params=params)
    assert listed.status_code != 500, listed.text
    assert counted.status_code != 500, counted.text
    assert streamed.status_code != 500, streamed.text
    if name in _ROUTE_READ_FILTERS:
        return
    assert listed.status_code == 200, listed.text
    assert {row["id"] for row in listed.json()} == expected
    assert counted.json()["count"] == len(expected)
    assert {row["id"] for row in streamed.json()["pictures"]} == expected
