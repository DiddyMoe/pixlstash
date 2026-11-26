"""The Scrapheap counts as one bin on the grid, the header and the panel.

"Empty" acts on the whole heap, so the Scrapheap view lists every scrapheaped
picture whatever set, project or person the view still carries: the grid
(``GET /pictures``) and the header (``GET /pictures/count``) always did. The
Details panel (``GET /pictures/stats``) narrowed the heap to that set, project
or person instead, and showed a smaller total than the grid beside it.
"""

import json

import pytest
from fastapi.testclient import TestClient

from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureSet,
    PictureSetMember,
    Project,
)
from pixlstash.db_models.picture_project import PictureProjectMember
from pixlstash.server import Server
from pixlstash.utils.service.picture_stats import clear_stats_cache


@pytest.fixture(scope="module")
def heap(tmp_path_factory):
    """A heap of four, one in a set, one in a project and one of a person's,
    beside a live picture in each of them."""
    tmp_path = tmp_path_factory.mktemp("scrapheap-scope")
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps({"port": 8000, "disable_background_workers": True})
    )
    server = Server(str(config_path))
    client = TestClient(server.api)
    assert (
        client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        ).status_code
        == 200
    )
    try:

        def seed(session):
            picture_set = PictureSet(name="test-set")
            project = Project(name="test-project")
            person = Character(name="test-person")
            session.add_all([picture_set, project, person])
            session.flush()
            ids = {}
            for key, deleted in (
                ("gone_in_set", True),
                ("gone_in_project", True),
                ("gone_of_person", True),
                ("gone_alone", True),
                ("live_in_set", False),
                ("live_in_project", False),
                ("live_of_person", False),
            ):
                path = tmp_path / f"{key}.png"
                path.write_bytes(b"test-not-an-image")
                pic = Picture(file_path=str(path), width=8, height=8, deleted=deleted)
                session.add(pic)
                session.flush()
                ids[key] = pic.id
            for key in ("gone_in_set", "live_in_set"):
                session.add(
                    PictureSetMember(set_id=picture_set.id, picture_id=ids[key])
                )
            for key in ("gone_in_project", "live_in_project"):
                session.add(
                    PictureProjectMember(picture_id=ids[key], project_id=project.id)
                )
            for key in ("gone_of_person", "live_of_person"):
                session.add(
                    Face(
                        picture_id=ids[key],
                        face_index=0,
                        bbox=[1, 1, 7, 7],
                        character_id=person.id,
                    )
                )
            session.commit()
            ids["set"], ids["project"], ids["person"] = (
                picture_set.id,
                project.id,
                person.id,
            )
            return ids

        ids = server.vault.db.run_task(seed)
        yield client, ids
    finally:
        clear_stats_cache()
        server.close()


def _views(ids):
    return {
        "set_id": {"set_id": str(ids["set"])},
        "set_ids": {"set_ids": [str(ids["set"])], "set_mode": "union"},
        "project_id": {"project_id": str(ids["project"])},
        "character_id": {"character_id": str(ids["person"])},
        "character_ids": {"character_ids": [str(ids["person"])]},
    }


VIEWS = ("set_id", "set_ids", "project_id", "character_id", "character_ids")


def _counted(client, params):
    listed = client.get("/pictures", params={**params, "fields": "grid"})
    assert listed.status_code == 200, (params, listed.text)
    counted = client.get("/pictures/count", params=params)
    assert counted.status_code == 200, (params, counted.text)
    clear_stats_cache()
    stats = client.get("/pictures/stats", params=params)
    assert stats.status_code == 200, (params, stats.text)
    return (
        {item["id"] for item in listed.json()},
        counted.json()["count"],
        stats.json()["total"],
    )


@pytest.mark.parametrize("view", VIEWS)
def test_the_scrapheap_counts_the_whole_heap_in_any_view(heap, view):
    """The grid, the header and the panel all count the four heaped pictures."""
    client, ids = heap
    whole_heap = {
        ids[key]
        for key in ("gone_in_set", "gone_in_project", "gone_of_person", "gone_alone")
    }

    listed, counted, total = _counted(
        client, {**_views(ids)[view], "only_deleted": "true"}
    )

    assert listed == whole_heap
    assert counted == len(whole_heap)
    assert total == len(whole_heap)


@pytest.mark.parametrize("view", VIEWS)
def test_outside_the_scrapheap_the_view_still_narrows_the_panel(heap, view):
    """Live, the same set, project or person narrows all three to its own
    live picture."""
    client, ids = heap
    own = {
        "set_id": "live_in_set",
        "set_ids": "live_in_set",
        "project_id": "live_in_project",
        "character_id": "live_of_person",
        "character_ids": "live_of_person",
    }[view]

    listed, counted, total = _counted(client, _views(ids)[view])

    assert listed == {ids[own]}
    assert counted == 1
    assert total == 1


def test_a_person_past_sqlite_is_refused_in_the_scrapheap_too(heap):
    """The panel refuses the id the grid refuses, before setting it aside."""
    client, _ids = heap
    params = {"only_deleted": "true", "character_id": str(2**63)}

    assert client.get("/pictures", params=params).status_code == 400
    clear_stats_cache()
    refused = client.get("/pictures/stats", params=params)
    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == "Invalid character_id"
