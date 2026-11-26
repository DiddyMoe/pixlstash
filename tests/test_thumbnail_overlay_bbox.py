"""A face or detection box the model cannot read costs only its own overlay entry.

``POST /pictures/thumbnails`` builds the grid's face and detection overlays.
It used to read each box under ``except Exception: bbox = None``, so a bad row
vanished with no trace. ``Face`` and ``Detection`` now refuse such a
box on write, but the database load does not run that check, so a row written
before it, or by a tool outside the app, can still hold one. Such a row is
logged with its id and the picture's id and left out; the rest of the
picture's entry, and every other picture in the batch, is served.

A box is four finite numbers. ``json.loads`` also reads ``NaN``,
``Infinity`` and integers too large for a float: on a picture with thumbnail
dimensions those were clamped into a box drawn at the edge, or raised
``OverflowError`` and blanked the picture's entry; on one without, the raw
value was served, and a ``NaN`` made the response fail for the whole batch.
``POST /pictures/{id}/face`` answers such a box with a 400.
"""

import io
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text

from pixlstash.db_models import Detection, Face
from pixlstash.server import Server
from pixlstash.tasks import TaskType
from tests.utils import upload_pictures_and_wait, wait_for_faces

API = "/api/v1"
# The route reads each box through ``read_stored_bbox``, which logs a bad one
# once per row and text.
_FACE_LOGGER = "pixlstash.db_models.face"

_HUGE_INT = "9" * 400
# Past Python's int-string limit (4300 digits): ``json.loads`` fails on it
# with a plain ValueError, not a JSONDecodeError.
_OVERLONG_INT = "9" * 5000

# Each but the last is JSON that ``json.loads`` reads and that is not four
# finite numbers; the last is text it cannot read at all.
NOT_FINITE = {
    "nan": "[NaN, 4, 20, 20]",
    "infinity": "[4, Infinity, 20, 20]",
    "minus-infinity": "[4, 4, -Infinity, 20]",
    "huge-int": f"[4, 4, 20, {_HUGE_INT}]",
    "bool": "[true, 4, 20, 20]",
    "string": '["4", 4, 20, 20]',
    "overlong-int": f"[4, 4, 20, {_OVERLONG_INT}]",
}
# What the warning for each case says about the box.
_WARNING_PHRASE = {"overlong-int": "not JSON"}

# Each is a request body ``json.loads`` reads; the route parses the box with
# ``int(round(float(v)))``, which raises OverflowError on the first four.
NOT_FINITE_REQUEST = {
    "infinity": "[Infinity, 0, 10, 10]",
    "minus-infinity": "[0, -Infinity, 10, 10]",
    "overflowing-float": "[0, 0, 1e999, 10]",
    "huge-int": f"[0, 0, {_HUGE_INT}, 10]",
    "nan": "[0, NaN, 10, 10]",
}

# Every picture the module uses, by name. Each colour is distinct, so no file
# is a duplicate of another.
_COLOURS = {
    "bad-box": (200, 10, 40),
    **{f"sized-{case}": (30 + 30 * i, 120, 10) for i, case in enumerate(NOT_FINITE)},
    **{f"unsized-{case}": (30 + 30 * i, 120, 210) for i, case in enumerate(NOT_FINITE)},
    **{
        f"request-{case}": (250, 250, 5 + 40 * i)
        for i, case in enumerate(NOT_FINITE_REQUEST)
    },
    "stored-box": (5, 250, 250),
}
# How long the one import, and then face extraction of all of it, may take.
# Generous for the gate's CPU-only runners; a fast machine takes seconds.
_SETTLE_S = 180


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("overlay")
    (root / "images").mkdir()
    config = root / "server-config.json"
    config.write_text(json.dumps({"port": 8000}))
    server = Server(str(config))
    try:
        client = TestClient(server.api)
        resp = client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        )
        assert resp.status_code == 200
        # The tagger's preload reads every face box of a new picture; left
        # running it would read the row this module corrupts on purpose. The
        # thumbnail pass is left out because the tests here set a picture's
        # thumbnail dimensions by hand. Face extraction stays: the import
        # refuses without it, so ``pictures`` waits it out instead.
        detached = (TaskType.TAGGER, TaskType.THUMBNAIL_GENERATION)
        for task_type in detached:
            server.vault._planner_work_finders.pop(task_type, None)
        server.vault._work_planner.detach_finders(detached)
        yield client, server
    finally:
        server.close()


def _png(colour):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color=colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def pictures(env):
    """Every picture in ``_COLOURS``, imported and face-extracted once.

    One import and one wait for the module instead of one of each per test:
    21 imports, each waiting out face extraction alone, took 164 s on the
    gate's CPU-only runners and timed out there.

    Returns:
        The picture id of each name in ``_COLOURS``.
    """
    client, _server = env
    names = list(_COLOURS)
    files = [
        ("file", (f"overlay-{name}.png", _png(_COLOURS[name]), "image/png"))
        for name in names
    ]
    result = upload_pictures_and_wait(client, files, timeout_s=_SETTLE_S)
    # The import answers one entry per file, in upload order.
    entries = result["results"]
    assert [e["status"] for e in entries] == ["success"] * len(names), entries
    ids = {name: entry["picture_id"] for name, entry in zip(names, entries)}
    # Extraction writes the no-face sentinel and the thumbnail crop in one
    # writer task, so once the sentinel shows, every later ``run_task`` (the
    # serial writer) lands after the crop and is not overwritten by it.
    deadline = time.monotonic() + _SETTLE_S
    not_extracted = [
        name
        for name, picture_id in ids.items()
        if not wait_for_faces(
            client, picture_id, timeout_s=max(0.0, deadline - time.monotonic())
        )
    ]
    assert not not_extracted, f"face extraction did not run for {not_extracted}"
    return ids


def _add_rows(server, picture_id, face_text="0,0,10,10", detection_text="[1, 2, 3]"):
    """Two faces and two detections, one of each then corrupted in place."""

    def _add(session):
        faces = [
            Face(picture_id=picture_id, face_index=20 + i, bbox=[4, 4, 20, 20])
            for i in range(2)
        ]
        detections = [
            Detection(
                picture_id=picture_id,
                detection_index=20 + i,
                label="hand",
                score=0.9,
                bbox=[8, 8, 30, 30],
            )
            for i in range(2)
        ]
        session.add_all(faces + detections)
        session.flush()
        # Raw SQL is the only way to store what the model now refuses: it is
        # how a pre-validator or out-of-app row looks once loaded.
        session.execute(
            text("UPDATE face SET bbox = :bbox WHERE id = :id"),
            {"bbox": face_text, "id": faces[1].id},
        )
        session.execute(
            text("UPDATE detection SET bbox = :bbox WHERE id = :id"),
            {"bbox": detection_text, "id": detections[1].id},
        )
        session.commit()
        return [f.id for f in faces], [d.id for d in detections]

    return server.vault.db.run_task(_add)


def test_a_bad_box_is_logged_and_left_out_and_the_rest_is_served(env, pictures, caplog):
    client, server = env
    picture_id = pictures["bad-box"]
    (good_face, bad_face), (good_det, bad_det) = _add_rows(server, picture_id)

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        resp = client.post(f"{API}/pictures/thumbnails", json={"ids": [picture_id]})
    assert resp.status_code == 200, resp.text
    entry = resp.json()[str(picture_id)]

    assert entry["thumbnail"], "one bad box must not blank the picture's entry"
    face_ids = {f["id"] for f in entry["faces"]}
    assert good_face in face_ids
    assert bad_face not in face_ids
    det_ids = {d["id"] for d in entry["detections"]}
    assert good_det in det_ids
    assert bad_det not in det_ids

    warnings = [r.getMessage() for r in caplog.records if r.name == _FACE_LOGGER]
    assert any(
        f"Face {bad_face} on picture {picture_id}" in m and "not JSON" in m
        for m in warnings
    ), warnings
    assert any(
        f"Detection {bad_det} on picture {picture_id}" in m and "[1, 2, 3]" in m
        for m in warnings
    ), warnings


def _set_thumbnail_size(server, picture_id, width, height):
    def _set(session):
        session.execute(
            text(
                "UPDATE picture SET thumbnail_width = :w, thumbnail_height = :h "
                "WHERE id = :id"
            ),
            {"w": width, "h": height, "id": picture_id},
        )
        session.commit()

    server.vault.db.run_task(_set)


@pytest.mark.parametrize(
    ("case", "bad_text"), list(NOT_FINITE.items()), ids=list(NOT_FINITE)
)
def test_a_box_that_is_not_four_finite_numbers_is_left_out_and_the_batch_served(
    env, pictures, caplog, case, bad_text
):
    client, server = env
    sized = pictures[f"sized-{case}"]
    unsized = pictures[f"unsized-{case}"]
    # The 64x48 picture's bitmap at half size, so a box is mapped; and no
    # bitmap yet, so the stored box would go out as it is.
    _set_thumbnail_size(server, sized, 32, 24)
    _set_thumbnail_size(server, unsized, None, None)
    rows = {
        picture_id: _add_rows(server, picture_id, bad_text, bad_text)
        for picture_id in (sized, unsized)
    }

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        resp = client.post(f"{API}/pictures/thumbnails", json={"ids": [sized, unsized]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Which path each picture took: mapped, or served unscaled.
    assert body[str(sized)]["thumbnail_width"] == 32
    assert body[str(unsized)]["thumbnail_width"] is None
    good_face_box = {sized: [2, 2, 10, 10], unsized: [4, 4, 20, 20]}
    good_detection_box = {sized: [4, 4, 15, 15], unsized: [8, 8, 30, 30]}
    warnings = [r.getMessage() for r in caplog.records if r.name == _FACE_LOGGER]
    for picture_id, ((good_face, bad_face), (good_det, bad_det)) in rows.items():
        entry = body[str(picture_id)]
        assert entry["thumbnail"], "one bad box must not blank the picture's entry"
        faces = {f["id"]: f["bbox"] for f in entry["faces"]}
        assert faces.get(good_face) == good_face_box[picture_id], faces
        assert bad_face not in faces, faces
        detections = {d["id"]: d["bbox"] for d in entry["detections"]}
        assert detections.get(good_det) == good_detection_box[picture_id], detections
        assert bad_det not in detections, detections
        phrase = _WARNING_PHRASE.get(case, "finite")
        for kind, row_id in (("Face", bad_face), ("Detection", bad_det)):
            assert any(
                f"{kind} {row_id} on picture {picture_id}" in m and phrase in m
                for m in warnings
            ), (kind, row_id, warnings)


@pytest.mark.parametrize(
    ("case", "bad_text"),
    list(NOT_FINITE_REQUEST.items()),
    ids=list(NOT_FINITE_REQUEST),
)
def test_a_new_face_box_that_is_not_finite_is_a_400(env, pictures, case, bad_text):
    client, _server = env
    picture_id = pictures[f"request-{case}"]

    resp = client.post(
        f"{API}/pictures/{picture_id}/face",
        content=f'{{"bbox": {bad_text}}}',
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "bbox values must be finite numbers"
    faces = client.get(f"{API}/pictures/{picture_id}/faces").json()["faces"]
    assert [f for f in faces if f["face_index"] >= 0] == []


def test_a_new_face_box_of_four_numbers_is_stored(env, pictures):
    # The positive control for the 400 above.
    client, _server = env
    picture_id = pictures["stored-box"]

    resp = client.post(
        f"{API}/pictures/{picture_id}/face", json={"bbox": [1.4, 2, 30, 40.6]}
    )

    assert resp.status_code == 200, resp.text
    faces = client.get(f"{API}/pictures/{picture_id}/faces").json()["faces"]
    assert [f["bbox"] for f in faces if f["face_index"] >= 0] == [[1, 2, 30, 41]]
