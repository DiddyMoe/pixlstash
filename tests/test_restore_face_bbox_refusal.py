"""A picture's restore copies its face boxes as stored, or refuses the restore.

The per-resource restore rebuilt each face from the snapshot through
``Face(bbox=face.bbox)``. That re-encoded the box: a stored ``"null"`` came
back as NULL, which the ``bbox IS NOT NULL`` readers count differently, and
the text was rewritten. A box the model cannot read raised a bare
``JSONDecodeError`` out of the restore (a 500); a ``NaN`` was re-encoded into
a row the model refuses. Now the stored text is copied verbatim, and a box the
model refuses (not JSON, or not ``null`` or four finite numbers) refuses the
whole restore with a 409 that names the face and its picture. Nothing in the
library changes: the rows the restore had already replaced are rolled back.
"""

import json
import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import select

from pixlstash.db_models import Face, Picture, PictureSetMember, Tag
from pixlstash.db_models.picture_project import PictureProjectMember
from pixlstash.db_models.snapshot import Snapshot
from pixlstash.server import Server
from pixlstash.services.restore import RestoreRefusedError
from tests.utils import wipe_tables

API = "/api/v1"

# Each is a box text the model refuses, stored the only way it can be: by raw
# SQL, as a row written before the check or by a tool outside the app.
BAD_SNAPSHOT_BOXES = {
    "not-json": "0,0,10,10",
    "nan": "[0, NaN, 10, 10]",
    "huge-int": f"[0, 0, {'9' * 400}, 10]",
}


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        # No background workers: nothing writes the picture or its faces
        # between a test's own writes and the restore.
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def clean_db(server):
    server.vault.db.run_task(
        wipe_tables,
        [Snapshot, PictureProjectMember, PictureSetMember, Face, Tag, Picture],
    )
    snapshot_dir = os.path.join(server.vault.image_root, "snapshots")
    if os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    yield


@pytest.fixture
def owner(server):
    client = TestClient(server.api, raise_server_exceptions=False)
    response = client.post(
        f"{API}/login",
        json={"username": "owner", "password": "example-owner-password"},
    )
    assert response.status_code == 200, response.text
    return client


def _add_picture(server, filename: str, description: str, faces: list[dict]):
    """A picture with a file on disk and *faces* (``Face`` keyword sets)."""
    open(os.path.join(server.vault.image_root, filename), "wb").close()

    def _do(session):
        pic = Picture(file_path=filename, filename=filename, description=description)
        session.add(pic)
        session.flush()
        rows = [Face(picture_id=pic.id, **face) for face in faces]
        session.add_all(rows)
        session.commit()
        return pic.id, [row.id for row in rows]

    return server.vault.db.run_task(_do)


def _set_bbox_text(server, face_id: int, bbox_text: str | None):
    def _do(session):
        session.execute(
            text("UPDATE face SET bbox = :bbox WHERE id = :id"),
            {"bbox": bbox_text, "id": face_id},
        )
        session.commit()

    server.vault.db.run_task(_do)


def _set_description(server, picture_id: int, description: str):
    def _do(session):
        session.execute(
            text("UPDATE picture SET description = :d WHERE id = :id"),
            {"d": description, "id": picture_id},
        )
        session.commit()

    server.vault.db.run_task(_do)


def _live_state(server, picture_id: int):
    """The picture's description and its face rows as stored, by face index."""

    def _do(session):
        description = session.get(Picture, picture_id).description
        faces = session.exec(
            select(Face.id, Face.face_index, Face.bbox_)
            .where(Face.picture_id == picture_id)
            .order_by(Face.face_index)
        ).all()
        return description, [tuple(row) for row in faces]

    return server.vault.db.run_immediate_read_task(_do)


def _snapshot_with_a_bad_box(server, name: str, bad_text: str):
    """A snapshot whose one face holds *bad_text*; live has moved on since."""
    pic_id, (face_id,) = _add_picture(
        server,
        f"refuse-{name}.jpg",
        "in the snapshot",
        [{"face_index": 0, "bbox": [1, 2, 3, 4]}],
    )
    _set_bbox_text(server, face_id, bad_text)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "live")
    _set_bbox_text(server, face_id, "[5, 6, 7, 8]")
    assert _live_state(server, pic_id) == ("live", [(face_id, 0, "[5, 6, 7, 8]")])
    return cp, pic_id, face_id


@pytest.mark.parametrize(
    ("case", "bad_text"),
    list(BAD_SNAPSHOT_BOXES.items()),
    ids=list(BAD_SNAPSHOT_BOXES),
)
def test_a_snapshot_box_the_model_refuses_refuses_the_restore(server, case, bad_text):
    cp, pic_id, face_id = _snapshot_with_a_bad_box(server, case, bad_text)

    with pytest.raises(
        RestoreRefusedError,
        match=rf"^Face {face_id} on picture {pic_id} in the snapshot has a box",
    ):
        server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    # Nothing changed: not the description, and not the face the restore had
    # already deleted to replace (same id, same text).
    assert _live_state(server, pic_id) == ("live", [(face_id, 0, "[5, 6, 7, 8]")])


def test_the_route_answers_a_refused_restore_with_a_409_naming_the_face(server, owner):
    cp, pic_id, face_id = _snapshot_with_a_bad_box(server, "route", "0,0,10,10")

    response = owner.post(f"{API}/snapshots/{cp.id}/restore/picture/{pic_id}", json={})

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        f"Face {face_id} on picture {pic_id} in the snapshot has a box that "
        "cannot be read, so the restore was refused. Nothing was changed."
    )
    assert _live_state(server, pic_id) == ("live", [(face_id, 0, "[5, 6, 7, 8]")])


def test_snapshot_boxes_are_restored_exactly_as_stored(server):
    # The positive control: a readable box comes back as the snapshot stored
    # it. Re-encoding would turn the unspaced text into "[1, 2, 3, 4]" and the
    # stored "null" into NULL.
    pic_id, face_ids = _add_picture(
        server,
        "verbatim.jpg",
        "in the snapshot",
        [
            {"face_index": -1},
            {"face_index": 0, "bbox": [0, 0, 10, 10]},
            {"face_index": 1, "bbox": None},
        ],
    )
    _set_bbox_text(server, face_ids[1], "[1,2,3,4]")
    _set_bbox_text(server, face_ids[2], "null")
    stored = [(-1, None), (0, "[1,2,3,4]"), (1, "null")]
    assert [row[1:] for row in _live_state(server, pic_id)[1]] == stored
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    _set_description(server, pic_id, "live")
    for face_id in face_ids:
        _set_bbox_text(server, face_id, "[5, 6, 7, 8]")

    report = server.vault.restore_service.restore_resource(cp.id, "picture", pic_id)

    assert not report.errors, report.errors
    description, faces = _live_state(server, pic_id)
    assert description == "in the snapshot"
    assert [row[1:] for row in faces] == stored
