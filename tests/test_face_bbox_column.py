"""``Face.bbox_`` only ever holds text that ``Face.bbox`` can read back.

The column is JSON text. Writing it directly (``Face(bbox_=...)`` or
``face.bbox_ = ...``) used to skip ``json.dumps``, so a fixture that stored
``"0,0,10,10"`` made a later ``face.bbox`` read raise ``JSONDecodeError`` in
an unrelated code path. The model now refuses such a write.
"""

import json
import math

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from pixlstash.db_models import Face, Picture


@pytest.mark.parametrize(
    "text",
    [
        "0,0,10,10",  # the fixture value that broke a later read: not JSON
        "[0, 0, 10]",  # three numbers
        "[0, 0, 10, 10, 1]",  # five numbers
        '{"x": 1}',  # an object
        '"[0, 0, 10, 10]"',  # a JSON string holding a list
        "[0, 0, true, 10]",  # bool is not a coordinate
        '[0, 0, "10", 10]',  # a string coordinate
        "[0, 0, NaN, 10]",  # json.loads accepts NaN; a box cannot
        "[0, 0, Infinity, 10]",
        "",  # empty text is not JSON either
    ],
)
def test_a_bbox_text_the_reader_cannot_use_is_refused(text):
    with pytest.raises(ValueError, match="Face.bbox_"):
        Face(picture_id=1, bbox_=text)

    face = Face(picture_id=1, bbox=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="Face.bbox_"):
        face.bbox_ = text
    assert face.bbox == [1, 2, 3, 4]


def test_the_setter_refuses_a_box_that_is_not_four_numbers():
    face = Face(picture_id=1)
    with pytest.raises(ValueError, match="Face.bbox_"):
        face.bbox = [0, 0, 10]
    with pytest.raises(ValueError, match="Face.bbox_"):
        face.bbox = [0, 0, math.inf, 10]
    assert face.bbox_ is None


@pytest.mark.parametrize(
    "box",
    [[0, 0, 10, 10], [0.5, 1.25, 30.0, 40.75], [-3, -4, 5, 6]],
)
def test_a_four_number_box_round_trips_through_the_database(box):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[Picture.__table__, Face.__table__])
    with Session(engine) as session:
        picture = Picture(file_path="a.png", format="PNG", width=64, height=64)
        session.add(picture)
        session.flush()
        session.add(Face(picture_id=picture.id, bbox=box))
        session.add(Face(picture_id=picture.id, face_index=1, bbox_=json.dumps(box)))
        session.commit()
    with Session(engine) as session:
        faces = session.exec(select(Face).order_by(Face.face_index)).all()
        assert [f.bbox for f in faces] == [box, box]


def test_no_face_and_a_null_box_both_read_as_none():
    # A frame with no face is stored with bbox=None (no column value); a box
    # cleared through the setter (``clamp_bbox`` can return None) is "null".
    assert Face(picture_id=1).bbox is None
    face = Face(picture_id=1, bbox=[0, 0, 1, 1])
    face.bbox = None
    assert face.bbox_ == "null"
    assert face.bbox is None
    assert Face(picture_id=1, bbox_="null").bbox is None
