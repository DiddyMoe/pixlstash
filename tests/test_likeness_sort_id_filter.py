"""The ``id`` filter admits the same pictures under the likeness sort.

A filter admits the same rows under every sort; the sort only orders them.
The likeness sort builds its own candidate list, which read the person, the
project, the sets and the other filters but not ``id=``, so a grid narrowed to
named pictures listed every picture with a face once sorted by likeness. The
``id`` list also carries the narrowing of shared-only, so that went too.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixlstash.database import DBPriority
from pixlstash.db_models.face import Face
from pixlstash.db_models.picture import Picture
from pixlstash.server import Server

# Each picture's one face; the reference person's face points along x, so the
# likeness falls from "near" to "far".
_FACES = {
    "reference": [1.0, 0.0],
    "near": [0.95, 0.31],
    "middle": [0.7, 0.7],
    "far": [0.0, 1.0],
}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("likeness-sort-id-filter")
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
        resp = client.post("/characters", json={"name": "test-reference"})
        assert resp.status_code == 200, resp.text
        reference_id = resp.json()["character"]["id"]

        def seed(session):
            ids = {}
            for role, vector in _FACES.items():
                pic = Picture(
                    file_path=f"/nonexistent/{role}.png",
                    width=64,
                    height=48,
                    format="png",
                )
                session.add(pic)
                session.flush()
                ids[role] = int(pic.id)
                session.add(
                    Face(
                        picture_id=pic.id,
                        frame_index=0,
                        face_index=0,
                        character_id=reference_id if role == "reference" else None,
                        bbox=[0, 0, 10, 10],
                        features=np.asarray(
                            vector + [0.0] * 6, dtype=np.float32
                        ).tobytes(),
                    )
                )
            session.commit()
            return ids

        ids = server.vault.db.run_task(seed, priority=DBPriority.IMMEDIATE)
        yield client, reference_id, ids
    finally:
        server.close()


def _likeness(client, reference_id, route="/pictures", **params):
    resp = client.get(
        route,
        params={
            "sort": "CHARACTER_LIKENESS",
            "reference_character_id": reference_id,
            **params,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _listed(client, reference_id, **params):
    return [row["id"] for row in _likeness(client, reference_id, **params)]


def test_the_listed_ids_come_back_in_likeness_order(env):
    client, reference_id, ids = env
    # The positive control: unfiltered, the sort orders all four.
    assert _listed(client, reference_id, descending="true") == [
        ids["reference"],
        ids["near"],
        ids["middle"],
        ids["far"],
    ]
    named = [ids["far"], ids["near"]]
    assert _listed(client, reference_id, id=named, descending="true") == [
        ids["near"],
        ids["far"],
    ]
    assert _listed(client, reference_id, id=named, descending="false") == [
        ids["far"],
        ids["near"],
    ]


def test_an_unlisted_id_is_absent_on_every_likeness_read(env):
    client, reference_id, ids = env
    named = [ids["near"], ids["far"]]
    for fields in ("grid", None):
        params = {"id": named} if fields is None else {"id": named, "fields": fields}
        listed = _listed(client, reference_id, **params)
        assert sorted(listed) == sorted(named), (fields, listed)
        assert ids["middle"] not in listed
    streamed = _likeness(client, reference_id, route="/pictures/stream", id=named)
    assert sorted(row["id"] for row in streamed["pictures"]) == sorted(named)
    counted = _likeness(client, reference_id, route="/pictures/count", id=named)
    assert counted["count"] == len(named), counted


def test_the_id_filter_admits_the_same_rows_under_another_sort(env):
    client, reference_id, ids = env
    named = [ids["near"], ids["far"]]
    by_date = client.get("/pictures", params={"sort": "DATE", "id": named})
    assert by_date.status_code == 200, by_date.text
    assert sorted(row["id"] for row in by_date.json()) == sorted(
        _listed(client, reference_id, id=named)
    )


def test_an_id_that_names_no_picture_lists_none(env):
    client, reference_id, _ids = env
    assert _listed(client, reference_id, id=["not-an-id"]) == []
    by_date = client.get("/pictures", params={"sort": "DATE", "id": ["not-an-id"]})
    assert by_date.status_code == 200, by_date.text
    assert by_date.json() == []


def test_shared_only_narrows_the_likeness_sort(env):
    """Nothing is shared here, so shared-only lists nothing, sorted or not."""
    client, reference_id, ids = env
    assert ids["near"] in _listed(client, reference_id)
    assert _listed(client, reference_id, shared_only="true") == []


def test_shared_only_narrows_the_unassigned_view(env):
    """An ``id`` list narrowed to nothing is not an absent one."""
    client, _reference_id, ids = env
    unassigned = client.get("/pictures", params={"character_id": "UNASSIGNED"})
    assert unassigned.status_code == 200, unassigned.text
    assert ids["near"] in {row["id"] for row in unassigned.json()}
    shared = client.get(
        "/pictures", params={"character_id": "UNASSIGNED", "shared_only": "true"}
    )
    assert shared.status_code == 200, shared.text
    assert shared.json() == []
    # The header's count reads the same narrowing.
    for shared_only, expected in (("false", len(unassigned.json())), ("true", 0)):
        counted = client.get(
            "/pictures/count",
            params={"character_id": "UNASSIGNED", "shared_only": shared_only},
        )
        assert counted.status_code == 200, counted.text
        assert counted.json()["count"] == expected, (shared_only, counted.json())
