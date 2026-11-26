"""A stored face or detection box that cannot be read costs only that box.

``Face.bbox_`` and ``Detection.bbox_`` are JSON text checked when a value is
set, not when a row is loaded, so a row written before that check, or by a
tool outside the app, can hold text that is not JSON, ``NaN``, an integer too
large for a float, or the wrong shape. Every reader of a stored box goes
through ``read_stored_bbox``: such a box is logged once, with the picture id,
the row id and a shortened copy of the text, and left out. No made-up box
takes its place.

Covered here, one reader each: the face list and the detections route (200,
``bbox`` null), the smart score's face size, the thumbnail pass and the
face-model refresh (each finishes, so its finder stops selecting the
picture), the COCO and Ideogram export sidecars (no ``NaN`` written), the
character avatar (the next face is used), the character-sheet readers, a
face's area (``Face.width`` / ``height``: no area, so a folder move gives the
person the good face) and a rotate (the good boxes turn, the bad one is left
as stored).

A bad box is stored with raw SQL through the test's own engine, because the
model refuses it on write.
"""

import io
import json
import logging
import math
import shutil
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, text
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, select

from pixlstash.database import DBPriority
from pixlstash.db_models import Character, Detection, Face, Picture
from pixlstash.db_models import face as face_module
from pixlstash.db_models.face import read_stored_bbox
from pixlstash.scoring import fetch_main_face_mpx
from pixlstash.server import Server
from pixlstash.services import move_reconciliation_service, operation_log_service
from pixlstash.services.character_sheet import (
    sheet_face_boxes_in_session,
    sheet_faces_in_session,
)
from pixlstash.tasks.face_model_refresh_task import FaceModelRefreshTask
from pixlstash.tasks.missing_face_model_refresh_finder import (
    MissingFaceModelRefreshFinder,
)
from pixlstash.tasks.missing_thumbnail_finder import MissingThumbnailFinder
from pixlstash.tasks.thumbnail_generation_task import ThumbnailGenerationTask
from pixlstash.utils.service.export_utils import ExportUtils

API = "/api/v1"
_FACE_LOGGER = "pixlstash.db_models.face"
_GOOD = [4, 4, 20, 20]

# Each kind of text a stored box can hold that is not a box.
BAD_BOXES = {
    "not-json": "not json",
    "nan": "[NaN, 0, 1, 1]",
    "huge-int": "[" + "9" * 400 + ", 0, 1, 1]",
    "three-numbers": "[0, 0, 1]",
    # Nested far past the recursion limit: json.loads raises RecursionError.
    "too-deep": "[" * 100_000 + "]" * 100_000,
}
BAD_IDS = list(BAD_BOXES)


@pytest.fixture(autouse=True)
def _forget_logged_boxes():
    """Each test sees its own first warning: the once-per-row set is global."""
    face_module._REFUSED_BBOXES.clear()
    yield
    face_module._REFUSED_BBOXES.clear()


def _refusals(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _FACE_LOGGER]


def _store_bad_box(session, table: str, row_id: int, raw: str) -> None:
    """Write *raw* into a row's box column, past the model's write check."""
    session.execute(
        text(f"UPDATE {table} SET bbox = :bbox WHERE id = :id"),
        {"bbox": raw, "id": row_id},
    )


@pytest.fixture(scope="module")
def schema_file(tmp_path_factory):
    """An empty database with the whole schema, built once for the module.

    ``create_all`` costs more than any test that reads a box, so each test
    copies this file (``engine``) instead of building the schema again.
    """
    path = tmp_path_factory.mktemp("schema") / "schema.db"
    template = create_engine(f"sqlite:///{path}", poolclass=NullPool)
    SQLModel.metadata.create_all(template)
    template.dispose()
    return path


@pytest.fixture
def engine(schema_file, tmp_path):
    """The test's own database, a copy of ``schema_file``, foreign keys on."""
    path = tmp_path / "t.db"
    shutil.copyfile(schema_file, path)
    engine = create_engine(f"sqlite:///{path}", poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):  # noqa: ANN001
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    yield engine
    engine.dispose()


def _png_bytes(width=64, height=48, colour=(200, 10, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=colour).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# The shared reader
# --------------------------------------------------------------------------- #


def test_a_good_box_is_returned_as_stored(caplog):
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        box = read_stored_bbox("[1, 2, 3.5, 4]", owner="Face", row_id=7, picture_id=3)
    assert box == [1, 2, 3.5, 4]
    assert _refusals(caplog) == []


@pytest.mark.parametrize("raw", [None, "null"])
def test_no_box_is_none_and_not_logged(raw, caplog):
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        assert read_stored_bbox(raw, owner="Face", row_id=7, picture_id=3) is None
    assert _refusals(caplog) == []


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_a_bad_box_is_logged_once_with_its_ids_and_left_out(raw, caplog):
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        for _ in range(3):
            got = read_stored_bbox(raw, owner="Detection", row_id=41, picture_id=9)
            assert got is None
    messages = _refusals(caplog)
    assert len(messages) == 1, messages
    assert "Detection 41 on picture 9" in messages[0]
    assert raw[:20] in messages[0]


def test_a_long_bad_box_is_quoted_shortened():
    raw = BAD_BOXES["huge-int"]
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger(_FACE_LOGGER)
    logger.addHandler(handler)
    try:
        read_stored_bbox(raw, owner="Face", row_id=5, picture_id=2)
    finally:
        logger.removeHandler(handler)
    (record,) = records
    message = record.getMessage()
    assert f"({len(raw)} chars)" in message
    assert len(message) < len(raw), "the whole 400-digit text was logged"


def test_a_box_nested_past_the_recursion_limit_is_logged_as_not_json(caplog):
    # A RecursionError, not a ValueError, out of json.loads: still malformed
    # JSON, so the box is left out and the line names the error.
    raw = BAD_BOXES["too-deep"]
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        assert read_stored_bbox(raw, owner="Face", row_id=6, picture_id=4) is None
    (message,) = _refusals(caplog)
    assert "Face 6 on picture 4 has a stored bbox that is not JSON" in message
    assert "(RecursionError)" in message


# --------------------------------------------------------------------------- #
# Readers over a test engine
# --------------------------------------------------------------------------- #


def _picture_with_faces(session, boxes, *, width=1000, height=800, name="p.png"):
    """A picture and one face per box; a text box is stored raw after insert."""
    picture = Picture(file_path=name, format="png", width=width, height=height)
    session.add(picture)
    session.flush()
    faces = []
    for index, box in enumerate(boxes):
        face = Face(
            picture_id=picture.id,
            frame_index=0,
            face_index=index,
            bbox=box if isinstance(box, list) else _GOOD,
        )
        session.add(face)
        session.flush()
        if isinstance(box, str):
            _store_bad_box(session, "face", face.id, box)
        faces.append(int(face.id))
    session.commit()
    return int(picture.id), faces


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_smart_score_face_size_skips_a_bad_box(engine, raw):
    with Session(engine) as session:
        only_bad, _ = _picture_with_faces(session, [raw], name="a.png")
        mixed, _ = _picture_with_faces(
            session, [raw, [100, 100, 300, 300]], name="b.png"
        )

    with Session(engine) as session:
        sizes = fetch_main_face_mpx(session, [only_bad, mixed])

    assert only_bad not in sizes, "a picture whose only box is bad has no face size"
    assert math.isfinite(sizes[mixed])
    # 200 x 200 shrunk back by the stored-box scale: the good face alone.
    assert 0 < sizes[mixed] <= 0.04


class _EngineDB:
    """The two database calls the thumbnail task makes, over a test engine."""

    def __init__(self, engine, image_root):
        self._engine = engine
        self.image_root = image_root

    def run_task(self, fn, *args, priority=None, **kwargs):
        with Session(self._engine) as session:
            return fn(session, *args, **kwargs)


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_thumbnail_pass_finishes_a_picture_with_a_bad_box(tmp_path, engine, raw):
    (tmp_path / "t.png").write_bytes(_png_bytes())
    with Session(engine) as session:
        picture_id, _ = _picture_with_faces(
            session, [raw, [8, 8, 30, 30]], width=64, height=48, name="t.png"
        )

    with Session(engine) as session:
        selected = MissingThumbnailFinder._fetch_missing(session, 10)
    assert [p.id for p in selected] == [picture_id]

    result = ThumbnailGenerationTask(
        _EngineDB(engine, str(tmp_path)), selected
    )._run_task()

    assert result == {"changed_count": 1}
    with Session(engine) as session:
        assert session.get(Picture, picture_id).thumbnail_width
        again = MissingThumbnailFinder._fetch_missing(session, 10)
    assert again == [], "the picture is selected again every sweep"


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_face_model_refresh_finishes_a_picture_with_a_bad_box(engine, raw):
    with Session(engine) as session:
        picture = Picture(file_path="r.png")
        session.add(picture)
        session.flush()
        old = Face(
            picture_id=picture.id,
            frame_index=0,
            face_index=0,
            bbox=_GOOD,
            features=np.zeros(512, dtype="float32").tobytes(),
        )
        old.model_pack = "buffalo_l"
        session.add(old)
        session.flush()
        _store_bad_box(session, "face", old.id, raw)
        session.commit()
        picture_id = int(picture.id)

    finder = MissingFaceModelRefreshFinder(
        database=None,
        engine_getter=lambda: SimpleNamespace(insightface_model_pack="auraface"),
    )
    with Session(engine) as session:
        stale = finder._fetch_stale_pack_pictures(session, "auraface")
    assert [p.id for p in stale] == [picture_id]

    # Index (0, 1) pairs with nothing, so the old row's box is read for overlap.
    new_face = Face(
        picture_id=picture_id,
        frame_index=0,
        face_index=1,
        bbox=[10, 10, 30, 30],
        features=np.ones(512, dtype="float32").tobytes(),
        model_pack="auraface",
    )
    with Session(engine) as session:
        FaceModelRefreshTask._refresh_picture_faces(
            session, picture_id, [new_face], "auraface"
        )

    with Session(engine) as session:
        faces = session.exec(select(Face).where(Face.picture_id == picture_id)).all()
        stale = finder._fetch_stale_pack_pictures(session, "auraface")
    # The bad row overlaps nothing, so it is replaced by the new detection.
    assert [(f.model_pack, f.bbox) for f in faces] == [("auraface", [10, 10, 30, 30])]
    assert stale == [], "the picture is selected again every sweep"


def _detections(engine, raw):
    """A good and a bad detection on one picture, loaded back from the engine."""
    with Session(engine) as session:
        picture = Picture(file_path="d.png", format="png", width=2000, height=1000)
        session.add(picture)
        session.flush()
        rows = [
            Detection(
                picture_id=picture.id,
                detection_index=i,
                label=label,
                score=0.9,
                bbox=[100, 200, 1100, 800],
            )
            for i, label in enumerate(("dog", "cat"))
        ]
        session.add_all(rows)
        session.flush()
        _store_bad_box(session, "detection", rows[1].id, raw)
        session.commit()
    with Session(engine) as session:
        picture = session.exec(select(Picture)).one()
        dets = Detection.find(session, picture_id=picture.id)
        session.expunge_all()
    return picture, dets


def _zip_json(write) -> dict:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        write(zf)
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        raw = zf.read("IMG.json").decode("utf-8")
    body = json.loads(raw)
    # Strict JSON: NaN or Infinity in the file fails here.
    json.dumps(body, allow_nan=False)
    assert "NaN" not in raw and "Infinity" not in raw
    return body


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_coco_sidecar_leaves_out_a_bad_box(engine, raw):
    picture, dets = _detections(engine, raw)

    body = _zip_json(
        lambda zf: ExportUtils._write_detection_sidecar(
            zf, "IMG", "IMG.png", picture, dets, 1.0
        )
    )

    assert [o["label"] for o in body["objects"]] == ["dog"]
    assert body["objects"][0]["bbox"] == [100, 200, 1100, 800]


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_ideogram_sidecar_leaves_out_a_bad_box(engine, raw):
    picture, dets = _detections(engine, raw)

    body = _zip_json(
        lambda zf: ExportUtils._write_ideogram_sidecar(zf, "IMG", picture, dets, "x")
    )

    elements = body["compositional_deconstruction"]["elements"]
    assert [e["desc"] for e in elements] == ["dog"]


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_the_character_sheet_readers_leave_out_a_bad_box(engine, raw):
    with Session(engine) as session:
        character = Character(name="Sheet")
        session.add(character)
        session.flush()
        character_id = int(character.id)
        picture_id, _ = _picture_with_faces(
            session, [raw, [100, 100, 300, 300]], name="s.png"
        )

    with Session(engine) as session:
        faces, _references, _formats = sheet_faces_in_session(
            session, character_id, [picture_id]
        )
        boxes = sheet_face_boxes_in_session(session, [picture_id])

    assert sorted(faces[picture_id], key=lambda row: row[1] is None) == [
        (None, "[100, 100, 300, 300]"),
        (None, None),
    ], "the bad face still counts, without its box"
    assert boxes == {picture_id: [[100, 100, 300, 300]]}


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_a_bad_face_box_has_no_area_so_the_good_face_is_chosen(engine, raw, caplog):
    with Session(engine) as session:
        character = Character(name="Mover")
        session.add(character)
        session.flush()
        character_id = int(character.id)
        picture_id, (bad_id, good_id) = _picture_with_faces(
            session, [raw, [100, 100, 300, 300]], name="m.png"
        )

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        with Session(engine) as session:
            bad = session.get(Face, bad_id)
            assert (bad.width, bad.height) == (0.0, 0.0)
            # A folder move's person assignment ranks the faces by area.
            picture = session.get(Picture, picture_id)
            assert move_reconciliation_service.add_person(
                session, picture, character_id
            )
            session.commit()

    with Session(engine) as session:
        faces = session.exec(select(Face).where(Face.picture_id == picture_id)).all()
    assert {f.id: f.character_id for f in faces} == {
        bad_id: None,
        good_id: character_id,
    }
    messages = _refusals(caplog)
    assert len(messages) == 1, messages
    assert f"Face {bad_id} on picture {picture_id}" in messages[0]


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=BAD_IDS)
def test_a_rotate_turns_the_good_boxes_and_leaves_a_bad_one(
    tmp_path, engine, raw, caplog
):
    library = tmp_path / "library"
    library.mkdir()
    (library / "o.png").write_bytes(_png_bytes())
    with Session(engine) as session:
        picture_id, (bad_face, good_face) = _picture_with_faces(
            session, [raw, [100, 100, 300, 300]], name="o.png"
        )
        detections = [
            Detection(
                picture_id=picture_id,
                detection_index=i,
                label=label,
                score=0.9,
                bbox=[100, 200, 500, 600],
            )
            for i, label in enumerate(("dog", "cat"))
        ]
        session.add_all(detections)
        session.flush()
        good_det, bad_det = (int(d.id) for d in detections)
        _store_bad_box(session, "detection", bad_det, raw)
        session.commit()

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        with Session(engine) as session:
            turned = operation_log_service.apply_orientation(
                session, picture_id, 6, image_root=str(library)
            )
            session.commit()

    assert turned is True
    with Session(engine) as session:
        stored = {
            "good face": session.get(Face, good_face).bbox_,
            "bad face": session.get(Face, bad_face).bbox_,
            "good detection": session.get(Detection, good_det).bbox_,
            "bad detection": session.get(Detection, bad_det).bbox_,
        }
    # A quarter turn clockwise of the 1000 x 800 display: (x, y) -> (800 - y, x).
    assert stored == {
        "good face": "[500, 100, 700, 300]",
        "bad face": raw,
        "good detection": "[200, 100, 600, 500]",
        "bad detection": raw,
    }
    messages = _refusals(caplog)
    assert len(messages) == 2, messages
    assert f"Face {bad_face} on picture {picture_id}" in messages[0]
    assert f"Detection {bad_det} on picture {picture_id}" in messages[1]


# --------------------------------------------------------------------------- #
# Routes, over one server for the module
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("stored-bbox")
    config = root / "server-config.json"
    # No background work: every row here is written by hand, and the
    # thumbnail, tagger and face passes would read or rewrite them.
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


def _seed(server, name, face_boxes, *, character=False, detection_box=None):
    """A picture file and row, its faces (a text box stored raw) and detections.

    With *character* the faces are a new character's, in id order. With
    *detection_box* a good detection and one holding that text are added.
    """
    with open(f"{server.vault.image_root}/{name}", "wb") as fh:
        fh.write(_png_bytes())

    def insert(session):
        character_id = None
        if character:
            person = Character(name=f"Person {name}")
            session.add(person)
            session.flush()
            character_id = int(person.id)
        picture = Picture(file_path=name, format="png", width=64, height=48)
        session.add(picture)
        session.flush()
        face_ids = []
        for index, box in enumerate(face_boxes):
            face = Face(
                picture_id=picture.id,
                frame_index=0,
                face_index=index,
                character_id=character_id,
                bbox=box if isinstance(box, list) else _GOOD,
            )
            session.add(face)
            session.flush()
            if isinstance(box, str):
                _store_bad_box(session, "face", face.id, box)
            face_ids.append(int(face.id))
        detection_ids = []
        if detection_box is not None:
            for index in range(2):
                det = Detection(
                    picture_id=picture.id,
                    detection_index=index,
                    label="hand",
                    score=0.9,
                    bbox=[8, 8, 30, 30],
                )
                session.add(det)
                session.flush()
                detection_ids.append(int(det.id))
            _store_bad_box(session, "detection", detection_ids[1], detection_box)
        session.commit()
        return int(picture.id), character_id, face_ids, detection_ids

    return server.vault.db.run_task(insert, priority=DBPriority.IMMEDIATE)


@pytest.mark.parametrize("kind", BAD_IDS)
def test_the_face_list_serves_a_bad_box_as_null(env, kind, caplog):
    client, server = env
    name = f"faces-{kind}.png"
    raw = BAD_BOXES[kind]
    picture_id, _, (bad_id, good_id), _ = _seed(server, name, [raw, _GOOD])

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        resp = client.get(f"{API}/pictures/{picture_id}/faces")

    assert resp.status_code == 200, resp.text
    boxes = {f["id"]: f["bbox"] for f in resp.json()["faces"]}
    assert boxes == {bad_id: None, good_id: _GOOD}
    assert any(f"Face {bad_id} on picture {picture_id}" in m for m in _refusals(caplog))


@pytest.mark.parametrize("kind", BAD_IDS)
def test_the_detections_route_serves_a_bad_box_as_null(env, kind, caplog):
    client, server = env
    name = f"dets-{kind}.png"
    raw = BAD_BOXES[kind]
    picture_id, _, _, (good_id, bad_id) = _seed(server, name, [], detection_box=raw)

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        resp = client.get(f"{API}/pictures/{picture_id}/detections")

    assert resp.status_code == 200, resp.text
    boxes = {d["id"]: d["bbox"] for d in resp.json()}
    assert boxes == {good_id: [8, 8, 30, 30], bad_id: None}, "the row is kept"
    assert any(
        f"Detection {bad_id} on picture {picture_id}" in m for m in _refusals(caplog)
    )


@pytest.mark.parametrize("kind", BAD_IDS)
def test_the_character_avatar_passes_over_a_bad_box(env, kind):
    client, server = env
    name = f"avatar-{kind}.png"
    raw = BAD_BOXES[kind]
    _, character_id, _, _ = _seed(server, name, [raw, _GOOD], character=True)

    resp = client.get(f"/characters/{character_id}/thumbnail")

    assert resp.status_code == 200, resp.text
    assert Image.open(io.BytesIO(resp.content)).size == (256, 256)


@pytest.mark.parametrize("kind", BAD_IDS)
def test_a_character_with_only_bad_boxes_has_no_avatar(env, kind):
    client, server = env
    name = f"no-avatar-{kind}.png"
    raw = BAD_BOXES[kind]
    _, character_id, _, _ = _seed(server, name, [raw], character=True)

    resp = client.get(f"/characters/{character_id}/thumbnail")

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "No face thumbnail found for character"


_THUMBNAILS_LOGGER = "pixlstash.routes.pictures._thumbnails"


def _thumbnail_batch(client, *picture_ids) -> dict:
    resp = client.post(f"{API}/pictures/thumbnails", json={"ids": list(picture_ids)})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("kind", ["not-json", "nan"])
def test_the_thumbnail_batch_logs_a_bad_box_once_not_per_request(env, kind, caplog):
    # The grid asks for the same page on every scroll; a line per request
    # buried the log in one row.
    client, server = env
    raw = BAD_BOXES[kind]
    picture_id, _, (bad_face, good_face), (good_det, bad_det) = _seed(
        server, f"batch-once-{kind}.png", [raw, _GOOD], detection_box=raw
    )

    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        for _ in range(2):
            entry = _thumbnail_batch(client, picture_id)[str(picture_id)]
            assert {f["id"] for f in entry["faces"]} == {good_face}
            assert {d["id"] for d in entry["detections"]} == {good_det}

    lines = _refusals(caplog)
    for owner, row_id in (("Face", bad_face), ("Detection", bad_det)):
        mine = [m for m in lines if f"{owner} {row_id} on picture {picture_id}" in m]
        assert len(mine) == 1, (owner, lines)


# A stored value of each kind JSON cannot carry, or that is not a number:
# SQLite keeps a REAL infinity even in an INTEGER column, and text or a blob
# from a tool outside the app. (It stores NaN as NULL.)
_BAD_NUMBERS = {
    "inf-width": ("picture", "thumbnail_width", float("inf")),
    "minus-inf-side": ("picture", "square_crop_side", float("-inf")),
    "text-height": ("picture", "thumbnail_height", "wide"),
    "blob-crop-x": ("picture", "square_crop_x", b"\x00\x01"),
    "inf-detection-score": ("detection", "score", float("inf")),
    "inf-face-frame": ("face", "frame_index", float("inf")),
}
_GEOMETRY = {
    "thumbnail_width": 32,
    "thumbnail_height": 24,
    "square_crop_x": 4,
    "square_crop_y": 0,
    "square_crop_side": 24,
}


def _store_raw(server, table, column, row_id, value):
    def write(session):
        session.execute(
            text(f"UPDATE {table} SET {column} = :value WHERE id = :id"),
            {"value": value, "id": row_id},
        )
        session.commit()

    server.vault.db.run_task(write, priority=DBPriority.IMMEDIATE)


@pytest.mark.parametrize("case", list(_BAD_NUMBERS))
def test_a_stored_number_json_cannot_carry_is_null_and_the_batch_served(
    env, case, caplog
):
    client, server = env
    table, column, value = _BAD_NUMBERS[case]
    seeded = {
        role: _seed(
            server,
            f"numbers-{case}-{role}.png",
            [_GOOD],
            detection_box="[8, 8, 30, 30]",
        )
        for role in ("bad", "good")
    }
    for picture_id, _, _, _ in seeded.values():
        for name, stored in _GEOMETRY.items():
            _store_raw(server, "picture", name, picture_id, stored)
    bad_picture, _, (bad_face,), (bad_det, _) = seeded["bad"]
    good_picture, _, (good_face,), (good_det, _) = seeded["good"]
    row_id = {"picture": bad_picture, "face": bad_face, "detection": bad_det}[table]
    _store_raw(server, table, column, row_id, value)

    with caplog.at_level(logging.WARNING, logger=_THUMBNAILS_LOGGER):
        for _ in range(2):
            body = _thumbnail_batch(client, bad_picture, good_picture)

    good = body[str(good_picture)]
    assert {k: good[k] for k in _GEOMETRY} == _GEOMETRY
    assert [f["bbox"] for f in good["faces"]] == [[2, 2, 10, 10]]
    assert [d["score"] for d in good["detections"]] == [0.9, 0.9]

    bad = body[str(bad_picture)]
    assert bad["thumbnail"], "one bad value must not blank the picture's entry"
    expected = dict(_GEOMETRY)
    if table == "picture":
        expected[column] = None
    assert {k: bad[k] for k in _GEOMETRY} == expected
    faces = {f["id"]: f for f in bad["faces"]}
    assert set(faces) == {bad_face}
    detections = {d["id"]: d for d in bad["detections"]}
    assert set(detections) == {bad_det, seeded["bad"][3][1]}
    if table == "face":
        assert faces[bad_face][column] is None
    if table == "detection":
        assert detections[bad_det][column] is None

    what = column if table == "picture" else f"{table} {row_id} {column}"
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == _THUMBNAILS_LOGGER
        and f"Picture {bad_picture} has a stored {what} " in r.getMessage()
    ]
    assert len(lines) == 1, "logged once, not once per request"
