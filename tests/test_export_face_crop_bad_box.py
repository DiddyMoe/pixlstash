"""A face export skips a face whose stored box cannot be read, and only that face.

The face-crop export reads each face's box, clamps it to the picture and cuts
the crop. A stored box that is not four finite numbers (text that is not JSON,
``NaN``, an integer too large for a float, the wrong shape) used to raise in
the progress count, which failed the whole export, or in the clamp, which
dropped every face of that picture. Now both read through
``read_stored_bbox``: the bad face is logged once with its picture and face
ids and left out, and the picture's other faces are still written and counted.

The progress count and the writer count the same items, so a finished export's
``processed`` reaches its ``total`` even when an item is left out: the faces of
a picture that will not open, or a picture whose file is gone.

A bad box is stored with raw SQL, because the model refuses it on write. The
export helpers run against a real engine through a minimal stand-in for the
server's ``vault.db.run_task``.
"""

import io
import logging
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import event, text
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine

from pixlstash.db_models import Face, Picture
from pixlstash.db_models import face as face_module
from pixlstash.utils.service.export_utils import ExportUtils

_FACE_LOGGER = "pixlstash.db_models.face"
_EXPORT_LOGGER = "pixlstash.utils.service.export_utils"
_WIDTH, _HEIGHT = 64, 48
# Face 0 sits inside the picture. Face 2 runs past its right and bottom edges,
# so its crop shows that the writer still clamps.
_INSIDE = [4, 4, 20, 20]
_PAST_THE_EDGE = [30, 10, 90, 70]

BAD_BOXES = {
    "not-json": "not json",
    "nan": "[NaN, 0, 1, 1]",
    "huge-int": "[" + "9" * 400 + ", 0, 1, 1]",
    "three-numbers": "[0, 0, 1]",
}


@pytest.fixture(autouse=True)
def _forget_logged_boxes():
    """Each test sees its own first warning: the once-per-row set is global."""
    face_module._REFUSED_BBOXES.clear()
    yield
    face_module._REFUSED_BBOXES.clear()


def _export_env(tmp_path, raw):
    """A picture file with three faces, the middle one's box stored as *raw*."""
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}", poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):  # noqa: ANN001
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    SQLModel.metadata.create_all(engine)
    Image.new("RGB", (_WIDTH, _HEIGHT), color=(200, 10, 40)).save(tmp_path / "p.png")
    with Session(engine) as session:
        picture = Picture(file_path="p.png", format="png", width=_WIDTH, height=_HEIGHT)
        session.add(picture)
        session.flush()
        faces = [
            Face(picture_id=picture.id, frame_index=0, face_index=i, bbox=box)
            for i, box in enumerate((_INSIDE, _INSIDE, _PAST_THE_EDGE))
        ]
        session.add_all(faces)
        session.flush()
        session.execute(
            text("UPDATE face SET bbox = :bbox WHERE id = :id"),
            {"bbox": raw, "id": faces[1].id},
        )
        session.commit()
        picture_id, bad_face_id = int(picture.id), int(faces[1].id)

    def run_task(fn, *args):
        with Session(engine) as session:
            result = fn(session, *args)
            session.expunge_all()
            return result

    server = SimpleNamespace(
        vault=SimpleNamespace(
            db=SimpleNamespace(run_task=run_task, tag_exclusions=set()),
            image_root=str(tmp_path),
        )
    )
    pics = run_task(lambda s: [s.get(Picture, picture_id)])
    return server, pics, picture_id, bad_face_id


def _params():
    return {
        "export_type_d": Picture.ExportType.FACE,
        "bbox_mode_d": "none",
        "caption_mode_d": "none",
        "include_character_name_enabled": False,
        "scale_factor": 1.0,
    }


@pytest.mark.parametrize("raw", BAD_BOXES.values(), ids=list(BAD_BOXES))
def test_the_face_export_skips_only_the_bad_box(tmp_path, raw, caplog):
    server, pics, picture_id, bad_face_id = _export_env(tmp_path, raw)
    params = _params()
    caplog.set_level(logging.WARNING, logger=_FACE_LOGGER)

    faces_by_pic, tags_by_face, detections_by_pic, total = (
        ExportUtils._prepare_feature_maps_and_total(server, pics, params)
    )
    assert total == 2

    export_tasks = {"t": {"processed": 0}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        ExportUtils._write_export_pictures(
            zf,
            server,
            "t",
            export_tasks,
            pics,
            params,
            faces_by_pic,
            tags_by_face,
            detections_by_pic,
        )

    assert export_tasks["t"]["processed"] == 2
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        assert sorted(zf.namelist()) == [
            "image_00001_face_001.png",
            "image_00001_face_003.png",
        ]
        sizes = {
            name: Image.open(io.BytesIO(zf.read(name))).size for name in zf.namelist()
        }
    assert sizes["image_00001_face_001.png"] == (16, 16)
    # Clamped to the picture: x 30..64, y 10..48.
    assert sizes["image_00001_face_003.png"] == (34, 38)

    refusals = [r.getMessage() for r in caplog.records if r.name == _FACE_LOGGER]
    assert len(refusals) == 1, refusals
    assert f"Face {bad_face_id} on picture {picture_id} " in refusals[0]


def _export(server, pics, params):
    """Count the export's items, then write it: (total, processed, members)."""
    faces_by_pic, tags_by_face, detections_by_pic, total = (
        ExportUtils._prepare_feature_maps_and_total(server, pics, params)
    )
    export_tasks = {"t": {"processed": 0}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        ExportUtils._write_export_pictures(
            zf,
            server,
            "t",
            export_tasks,
            pics,
            params,
            faces_by_pic,
            tags_by_face,
            detections_by_pic,
        )
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        members = sorted(zf.namelist())
    return total, export_tasks["t"]["processed"], members


def test_the_faces_of_a_picture_that_will_not_open_are_counted_as_done(
    tmp_path, caplog
):
    server, pics, _picture_id, _face_id = _export_env(tmp_path, "[4, 4, 20, 20]")
    (tmp_path / "p.png").write_bytes(b"test-not-an-image")
    caplog.set_level(logging.WARNING, logger=_EXPORT_LOGGER)

    total, processed, members = _export(server, pics, _params())

    assert total == 3
    assert processed == total, "the finished count stopped below the total"
    assert members == []
    failures = [r.getMessage() for r in caplog.records if r.name == _EXPORT_LOGGER]
    assert len(failures) == 1, failures
    # The warning's own slot must carry the full path; the decoder's message,
    # which the warning embeds, names it too and must not be what passes.
    assert f"could not export the faces of {tmp_path / 'p.png'} (" in failures[0]
    assert "its 3 face(s) are counted as done" in failures[0]


@pytest.mark.parametrize(
    "export_type, expected_total",
    [(Picture.ExportType.FULL, 1), (Picture.ExportType.FACE, 0)],
    ids=["full", "face"],
)
def test_a_picture_whose_file_is_gone_is_counted_as_its_total_counts_it(
    tmp_path, caplog, export_type, expected_total
):
    server, pics, picture_id, _face_id = _export_env(tmp_path, "[4, 4, 20, 20]")
    (tmp_path / "p.png").unlink()
    caplog.set_level(logging.WARNING, logger=_EXPORT_LOGGER)

    total, processed, members = _export(
        server, pics, {**_params(), "export_type_d": export_type}
    )

    # A whole-picture export counts every picture; a face export counts only
    # the faces of a file that is there.
    assert total == expected_total
    assert processed == total, "the finished count does not match the total"
    assert members == []
    gone = [r.getMessage() for r in caplog.records if r.name == _EXPORT_LOGGER]
    assert len(gone) == 1, gone
    assert f"picture {picture_id} has no file" in gone[0]


def test_a_picture_the_trainer_conversion_cannot_read_is_counted_as_done(tmp_path):
    server, pics, _picture_id, _face_id = _export_env(tmp_path, "[4, 4, 20, 20]")
    (tmp_path / "p.png").write_bytes(b"test-not-an-image")
    params = {
        **_params(),
        "export_type_d": Picture.ExportType.FULL,
        "trainer_images": True,
    }

    total, processed, members = _export(server, pics, params)

    assert total == 1
    assert processed == total, "the finished count stopped below the total"
    assert members == []
