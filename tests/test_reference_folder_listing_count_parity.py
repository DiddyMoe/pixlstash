"""Invariant guard: /pictures and /pictures/count agree for reference-folder browsing.

Motivated by a smoke-test report that browsing a reference folder
(``reference_folder_id`` + ``file_path_prefix``) as the owner returned an empty
picture list while the count for the "same" folder was non-zero.

Both endpoints funnel through ``select_pictures_for_listing`` and read every filter
from ``request.query_params`` -- there is no separate filter-extraction path -- so
for identical query parameters they MUST return the same population.  These tests
pin that invariant (list length == count when count > 0) across both percent- and
plus-encoding of ``file_path_prefix``, so any future refactor that lets the two
diverge fails here.

Note: the reported empty-list/non-zero-count divergence is NOT reproducible with
identical parameters (see the investigation report); the real reproducible failure
mode is a stored-path vs browse-prefix mismatch, which is out of scope for this
guard.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture, PictureStack
from pixlstash.db_models.reference_folder import ReferenceFolder
from pixlstash.server import Server
from pixlstash.utils.service.picture_stats import clear_stats_cache


PREFIX = "/home/user/Bilder fra 2020"


@pytest.fixture(scope="module")
def served():
    """One server for the module, and a client logged in to it.

    The server is closed before the module ends, so its worker threads do not
    outlive the session (tests/conftest.py fails a run that leaves any behind).
    """
    with tempfile.TemporaryDirectory() as tmp:
        image_root = os.path.join(tmp, "images")
        os.makedirs(image_root, exist_ok=True)
        config_path = os.path.join(tmp, "server-config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"port": 0}))
        with Server(config_path) as server:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200, resp.text
            yield server, client


@pytest.fixture(scope="module")
def env(served):
    """The module's client and its reference folder, seeded once: the tests only read."""
    server, client = served
    return client, _seed(server)


def _seed(server):
    def _insert(session):
        folder = ReferenceFolder(id=3, folder=PREFIX, label="Bilder fra 2020")
        session.add(folder)
        session.flush()
        ids = []
        for i in range(5):
            pic = Picture(
                file_path=f"{PREFIX}/img_{i:03d}.jpg",
                reference_folder_id=3,
                imported_at=datetime.now(timezone.utc),
            )
            session.add(pic)
            session.flush()
            ids.append(pic.id)
        # A file in a sub-directory: children-only browsing must exclude it,
        # so list and count both ignore it consistently.
        deep = Picture(
            file_path=f"{PREFIX}/sub/deep.jpg",
            reference_folder_id=3,
            imported_at=datetime.now(timezone.utc),
        )
        session.add(deep)
        session.commit()
        return ids

    return server.vault.db.run_task(_insert, priority=DBPriority.IMMEDIATE)


def _run(env, prefix_value: str):
    client, seeded = env
    list_resp = client.get(
        f"/pictures?reference_folder_id=3&file_path_prefix={prefix_value}"
    )
    count_resp = client.get(
        "/pictures/count?stack_leaders_only=true"
        f"&reference_folder_id=3&file_path_prefix={prefix_value}"
    )
    assert list_resp.status_code == 200, list_resp.text
    assert count_resp.status_code == 200, count_resp.text
    return seeded, list_resp.json(), count_resp.json()["count"]


def test_reference_folder_list_matches_count_percent_encoded(env):
    seeded, pics, count = _run(env, quote(PREFIX))
    assert count == len(seeded), f"count={count} expected {len(seeded)}"
    assert len(pics) == count, (
        f"list returned {len(pics)} pictures but count reported {count}"
    )
    assert {p["id"] for p in pics} == set(seeded)


def test_reference_folder_list_matches_count_plus_encoded(env):
    # Browsers encode spaces in query strings as '+'; both endpoints must decode
    # it the same way.
    seeded, pics, count = _run(env, PREFIX.replace(" ", "+"))
    assert count == len(seeded)
    assert len(pics) == count


# A reference folder holding a stack whose cover sits outside it. Collapsed to
# one row per stack, the grid has to represent that stack by its best member
# inside the folder; when the collapse did not know about the folder it
# dropped the stack, and the header (the same collapse) agreed with the grid
# while both disagreed with the folder's own pictures, which the stats panel
# counts.
STRADDLED_PREFIX = "/example-library/straddled"


@pytest.fixture(scope="module")
def straddled(served):
    """Folder 4: a stack straddling its edge, a stack inside it, a loose picture."""
    server, _ = served

    def _insert(session):
        folder = ReferenceFolder(id=4, folder=STRADDLED_PREFIX, label="straddled")
        session.add(folder)
        session.flush()

        def picture(path, folder_id, **kwargs):
            pic = Picture(
                file_path=path,
                reference_folder_id=folder_id,
                imported_at=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(pic)
            session.flush()
            return pic.id

        straddling = PictureStack(name="straddling")
        inside = PictureStack(name="inside")
        session.add_all([straddling, inside])
        session.flush()
        picture(
            "/example-library/elsewhere/cover.jpg",
            None,
            stack_id=straddling.id,
            stack_position=0,
        )
        first = picture(
            f"{STRADDLED_PREFIX}/first.jpg",
            4,
            stack_id=straddling.id,
            stack_position=1,
        )
        second = picture(
            f"{STRADDLED_PREFIX}/second.jpg",
            4,
            stack_id=straddling.id,
            stack_position=2,
        )
        inside_cover = picture(
            f"{STRADDLED_PREFIX}/inside_cover.jpg",
            4,
            stack_id=inside.id,
            stack_position=0,
        )
        inside_member = picture(
            f"{STRADDLED_PREFIX}/inside_member.jpg",
            4,
            stack_id=inside.id,
            stack_position=1,
        )
        loose = picture(f"{STRADDLED_PREFIX}/loose.jpg", 4)
        session.commit()
        return {
            "rows": {first, inside_cover, loose},
            "pictures": {first, second, inside_cover, inside_member, loose},
        }

    return server.vault.db.run_task(_insert, priority=DBPriority.IMMEDIATE)


@pytest.mark.parametrize("with_prefix", [False, True], ids=["api", "sidebar"])
@pytest.mark.parametrize(
    "view",
    [{}, {"character_id": "UNASSIGNED"}, {"inbox": "all"}, {"inbox": "7d"}],
    ids=["plain", "unassigned", "inbox all", "inbox 7d"],
)
def test_folder_view_keeps_a_stack_whose_cover_is_elsewhere(
    served, straddled, view, with_prefix
):
    """Header, grid and stats agree on a folder holding a straddling stack.

    ``sidebar`` is the request the folder tree sends (the folder and its path);
    ``api`` names the folder alone. The grid shows one row per stack, each
    represented inside the folder; the header counts those rows; the stats
    panel counts the folder's pictures, every one of which a row represents.
    """
    _, client = served
    params = {"reference_folder_id": "4", **view}
    if with_prefix:
        params["file_path_prefix"] = STRADDLED_PREFIX

    listed = client.get("/pictures", params={**params, "fields": "grid"})
    assert listed.status_code == 200, listed.text
    assert {item["id"] for item in listed.json()} == straddled["rows"]

    counted = client.get(
        "/pictures/count", params={**params, "stack_leaders_only": "true"}
    )
    assert counted.status_code == 200, counted.text
    assert counted.json()["count"] == len(straddled["rows"])

    clear_stats_cache()
    stats = client.get("/pictures/stats", params=params)
    assert stats.status_code == 200, stats.text
    assert stats.json()["total"] == len(straddled["pictures"])
