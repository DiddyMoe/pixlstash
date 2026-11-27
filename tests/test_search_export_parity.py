"""The export of a search view packages the pictures the search shows.

The grid's search view is ``GET /pictures/search`` with the view's filters, and
its "Export" sends the same query string to ``GET /pictures/export``. The
export must answer with the search's pictures, in the search's order: the same
filters, the same Scrapheap meaning, the same set, the same ranking.
"""

import gc
import io
import json
import os
import tempfile
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_set import PictureSet, PictureSetMember
from pixlstash.server import Server
from tests.utils import upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")
EXPORT = "/pictures/export"
SEARCH = {"query": "zebra", "threshold": 0.01}

# File name, score, and whether the picture is in the Scrapheap.
LIBRARY = [
    ("Bad1.png", 5, False),
    ("Bad2.png", 4, False),
    ("Changed1.png", 1, False),
    ("Changed2.png", 5, True),
    ("Reference1.png", 4, True),
    ("Changed3.png", 1, True),
]
# A set holding a picture the min_score filter keeps and one it leaves out.
SET_MEMBERS = ("Bad1.png", "Changed1.png")


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
    """Six pictures tagged ``zebra``: three live, three in the Scrapheap.

    In each half the scores are 5, 4 and 1, so ``min_score=3`` keeps two of
    the three: a search that ignored the filter would show a third.
    """
    temp_dir = tempfile.TemporaryDirectory()
    server, client = _server(temp_dir.name)
    try:
        ids = {name: _upload_picture(client, name) for name, _, _ in LIBRARY}
        for pid in ids.values():
            resp = client.post(f"/pictures/{pid}/tags", json={"tag": "zebra"})
            assert resp.status_code == 200, resp.text

        def seed(session):
            for name, score, deleted in LIBRARY:
                pic = session.get(Picture, ids[name])
                pic.score = score
                pic.deleted = deleted
                session.add(pic)
            picture_set = PictureSet(name="example-set")
            session.add(picture_set)
            session.flush()
            for name in SET_MEMBERS:
                session.add(
                    PictureSetMember(set_id=picture_set.id, picture_id=ids[name])
                )
            session.commit()
            return picture_set.id

        set_id = server.vault.db.run_task(seed)
        yield server, client, set_id
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _searched(client, params) -> list[int]:
    resp = client.get("/pictures/search", params={**SEARCH, **params})
    assert resp.status_code == 200, resp.text
    return [row["id"] for row in resp.json()]


def _finish(client, task_id, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"{EXPORT}/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"export {task_id} did not finish within {timeout_s}s")


def _exported_names(client, params) -> list[str]:
    """Export with the search's query string; the ZIP's pictures in order."""
    r = client.get(
        EXPORT,
        params={
            **SEARCH,
            **params,
            "caption_mode": "none",
            "use_original_file_names": "true",
        },
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    status = _finish(client, task_id)
    assert status["status"] == "completed", status
    r = client.get(f"{EXPORT}/download/{task_id}")
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        return [name for name in archive.namelist() if name.endswith(".png")]


def _file_names(server, picture_ids) -> list[str]:
    def read(session):
        rows = session.exec(
            select(Picture.id, Picture.original_file_name).where(
                Picture.id.in_(picture_ids)
            )
        ).all()
        return {pid: name for pid, name in rows}

    by_id = server.vault.db.run_task(read)
    return [by_id[pid] for pid in picture_ids]


@pytest.mark.parametrize(
    "params, expected",
    [
        pytest.param({"min_score": 3}, {"Bad1.png", "Bad2.png"}, id="min_score"),
        pytest.param(
            {"only_deleted": "true", "min_score": 3},
            {"Changed2.png", "Reference1.png"},
            id="scrapheap-min_score",
        ),
        pytest.param({"set": True, "min_score": 3}, {"Bad1.png"}, id="set-min_score"),
    ],
)
def test_a_search_export_packages_the_search_views_pictures_in_its_order(
    library, params, expected
):
    server, client, set_id = library
    params = dict(params)
    if params.pop("set", False):
        params["set_id"] = set_id
    shown = _searched(client, params)
    # The view itself is the one the filter and the Scrapheap flag make, so
    # the comparison below is not between two empty answers.
    assert set(_file_names(server, shown)) == expected

    assert _exported_names(client, params) == _file_names(server, shown)
