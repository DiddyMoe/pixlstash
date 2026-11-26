"""A text search narrows by every filter the grid it searches narrows by.

The grid's listing (``GET /pictures``) and its search (``GET /pictures/search``)
read the same filter params through one parse
(``_listing.parse_picture_filter``). The search used to drop most of them, so a
search in a filtered grid answered from the whole library: a ``max_score``,
``resolution_bucket``, ``file_path_prefix``, ``workflow``, ``id`` or
``shared_only`` the grid honoured was ignored. Here every such param must give
a search exactly the listing's pictures among the ones the search matches, and
a value the listing refuses is refused by the search with the same status.
"""

import gc
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from pixlstash.db_models.picture import Picture
from pixlstash.server import Server
from tests.utils import upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")
SEARCH = {"query": "zebra", "threshold": 0.01}
# A value no filter can read, so a 4xx detail that names it would show.
BAD_VALUE = "test-unreadable-value"


def _server(temp_dir: str) -> tuple[Server, TestClient]:
    server_config_path = os.path.join(temp_dir, "server-config.json")
    with open(server_config_path, "w") as f:
        f.write(json.dumps({"port": 8000}))
    server = Server(server_config_path)
    client = TestClient(server.api)
    resp = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert resp.status_code == 200
    return server, client


def _upload_picture(client, filename) -> int:
    with open(os.path.join(PICTURES_DIR, filename), "rb") as f:
        result = upload_pictures_and_wait(
            client, [("file", (filename, f, "image/png"))]
        )
    assert result["status"] == "completed"
    return result["results"][0]["picture_id"]


@pytest.fixture(scope="module")
def library():
    """Three pictures tagged ``zebra`` that differ in what the filters read.

    ``large`` is 1280x1920, ``huge`` 1792x2304 (the only one of 4-8 MP),
    ``small`` 896x1152.
    """
    temp_dir = tempfile.TemporaryDirectory()
    server, client = _server(temp_dir.name)
    try:
        ids = {
            "large": _upload_picture(client, "Bad1.png"),
            "huge": _upload_picture(client, "Bad2.png"),
            "small": _upload_picture(client, "Changed1.png"),
        }
        for pid in ids.values():
            resp = client.post(f"/pictures/{pid}/tags", json={"tag": "zebra"})
            assert resp.status_code == 200, resp.text

        def seed(session):
            for key, score, smart_score, source in (
                ("large", 5, 4.5, "/srv/example-import"),
                ("huge", 2, 2.5, None),
                ("small", None, 1.5, None),
            ):
                pic = session.get(Picture, ids[key])
                pic.score = score
                pic.smart_score = smart_score
                pic.import_source_folder = source
                session.add(pic)
            session.commit()

        server.vault.db.run_task(seed)
        yield client, ids
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _listed(client, params) -> set[int]:
    resp = client.get("/pictures", params=params)
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()}


def _searched(client, params) -> set[int]:
    resp = client.get("/pictures/search", params={**SEARCH, **params})
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()}


# Each narrows the three pictures, so a search that dropped it would answer
# with more than the listing does.
NARROWING = [
    pytest.param({"max_score": 3}, id="max_score"),
    pytest.param({"min_score": 4}, id="min_score"),
    pytest.param({"smart_score_bucket": "2-3"}, id="smart_score_bucket"),
    pytest.param({"resolution_bucket": "4-8mp"}, id="resolution_bucket"),
    pytest.param(
        {"import_source_folder": "/srv/example-import"}, id="import_source_folder"
    ),
    pytest.param(
        {"file_path_prefix": "/srv/example-no-such-folder"}, id="file_path_prefix"
    ),
    pytest.param({"reference_folder_id": 2147483}, id="reference_folder_id"),
    pytest.param({"workflow": "test-no-such-workflow"}, id="workflow"),
    pytest.param({"shared_only": "true"}, id="shared_only"),
]


@pytest.mark.parametrize("params", NARROWING)
def test_a_search_narrows_by_each_filter_the_listing_narrows_by(library, params):
    client, ids = library
    matched = _searched(client, {})
    assert matched == set(ids.values())
    listed = _listed(client, params)
    # The case is only worth its name if the filter removes a match.
    assert matched - listed, f"{params} narrows nothing here"
    assert _searched(client, params) == matched & listed


def test_a_search_narrows_by_the_listing_ids(library):
    client, ids = library
    picked = [ids["large"], ids["small"]]
    assert _listed(client, {"id": picked}) == set(picked)
    assert _searched(client, {"id": picked}) == set(picked)


# Filters whose outcome depends on what the taggers and the face detector made
# of the pictures: the search still answers exactly the listing's pictures.
DERIVED = [
    pytest.param({"face_filter": "without_face"}, id="face_filter"),
    pytest.param({"tag_state": "none"}, id="tag_state"),
    pytest.param({"tag_confidence_above": "zebra:0.5"}, id="tag_confidence_above"),
    pytest.param({"tag_confidence_below": "zebra:0.5"}, id="tag_confidence_below"),
    pytest.param({"impossible_tag_source": "no_face"}, id="impossible_tag_source"),
]


@pytest.mark.parametrize("params", DERIVED)
def test_a_search_agrees_with_the_listing_on_derived_filters(library, params):
    client, _ids = library
    matched = _searched(client, {})
    assert _searched(client, params) == matched & _listed(client, params)


REFUSED = [
    pytest.param({"min_score": BAD_VALUE}, id="min_score"),
    pytest.param({"max_score": BAD_VALUE}, id="max_score"),
    pytest.param({"max_score": str(2**70)}, id="max_score_beyond_sqlite"),
    pytest.param({"tag_state": BAD_VALUE}, id="tag_state"),
    pytest.param({"tag_confidence_above": BAD_VALUE}, id="tag_confidence_above"),
    pytest.param({"tag_confidence_below": "zebra:" + BAD_VALUE}, id="conf_below"),
    pytest.param({"reference_folder_id": BAD_VALUE}, id="reference_folder_id"),
    pytest.param({"shared_only": BAD_VALUE}, id="shared_only"),
    pytest.param({"face_cluster": 0}, id="face_cluster_zero"),
    # A cluster belongs to one person: without one, there is no cluster.
    pytest.param({"face_cluster": 1}, id="face_cluster_without_person"),
]


@pytest.mark.parametrize("params", REFUSED)
def test_a_search_refuses_what_the_listing_refuses(library, params):
    client, _ids = library
    listed = client.get("/pictures", params=params)
    searched = client.get("/pictures/search", params={**SEARCH, **params})
    assert listed.status_code == 422, listed.text
    assert searched.status_code == 422, searched.text
    for resp in (listed, searched):
        # The detail names the param, never the value it was sent. (The short
        # integers are left out: a digit is in any message.)
        sent = [str(v) for v in params.values() if len(str(v)) > 2]
        detail = json.dumps(resp.json()["detail"])
        assert not [value for value in sent if value in detail], resp.text
