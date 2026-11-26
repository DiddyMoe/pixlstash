"""A ``Detection`` box, like a ``Face`` box, is four finite numbers or nothing.

``Face`` refuses on write a ``bbox`` column value its reader cannot use.
``Detection`` stores the same JSON text and had no such check, so a
``NaN`` or a non-JSON value written through ``bbox_`` reached the grid overlay
and could fail the whole thumbnail batch. Both models now share one check. It
also refuses an integer too large for a float, which ``json.loads`` reads:
``math.isfinite`` raised ``OverflowError`` on one, not the ``ValueError`` the
models promise, so a caller catching ``ValueError`` let it through.
"""

import json
import logging
import math
import sys

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from pixlstash.db_models import Detection, Face, Picture

_HUGE_INT = "9" * 400
# Longer than Python's int-string limit (4300 digits), so ``json.loads`` itself
# fails, with a plain ValueError rather than a JSONDecodeError.
_OVERLONG_INT = "9" * 5000
# Arrays nested far past the recursion limit: ``json.loads`` fails with a
# RecursionError, a RuntimeError, not a ValueError.
_TOO_DEEP = "[" * 100_000 + "]" * 100_000

BAD_TEXTS = [
    "0,0,10,10",  # not JSON
    "[0, 0, 10]",  # three numbers
    "[0, 0, 10, 10, 1]",  # five numbers
    '{"x": 1}',  # an object
    '"[0, 0, 10, 10]"',  # a JSON string holding a list
    "[0, 0, true, 10]",  # bool is not a coordinate
    '[0, 0, "10", 10]',  # a string coordinate
    "[0, 0, NaN, 10]",  # json.loads accepts NaN; a box cannot
    "[0, 0, Infinity, 10]",
    "[0, 0, 1e999, 10]",  # json.loads reads it as Infinity
    f"[0, 0, {_HUGE_INT}, 10]",  # an int no float can hold
    f"[-{_HUGE_INT}, 0, 10, 10]",
    f"[0, 0, {_OVERLONG_INT}, 10]",  # past the int-string limit: json.loads fails
    _TOO_DEEP,  # nested past the recursion limit: json.loads fails
    "",  # empty text is not JSON either
]


@pytest.mark.parametrize("text", BAD_TEXTS)
def test_a_detection_box_the_reader_cannot_use_is_refused(text):
    with pytest.raises(ValueError, match="Detection.bbox_"):
        Detection(picture_id=1, label="hand", bbox_=text)

    detection = Detection(picture_id=1, label="hand", bbox=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="Detection.bbox_"):
        detection.bbox_ = text
    assert detection.bbox == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "box",
    [[0, 0, 10], [0, 0, math.nan, 10], [0, 0, 10**400, 10], [True, 0, 1, 1]],
    ids=["three", "nan", "huge-int", "bool"],
)
def test_the_detection_setter_refuses_a_box_that_is_not_four_finite_numbers(box):
    detection = Detection(picture_id=1)
    with pytest.raises(ValueError, match="Detection.bbox_"):
        detection.bbox = box
    assert detection.bbox_ is None


@pytest.mark.parametrize(
    "text",
    [
        f"[0, 0, {_HUGE_INT}, 10]",
        f"[-{_HUGE_INT}, 0, 10, 10]",
        f"[0, 0, {_OVERLONG_INT}, 10]",
    ],
    ids=["huge-int", "minus-huge-int", "overlong-int"],
)
def test_a_face_box_too_large_for_a_float_is_refused_as_a_bad_value(text):
    with pytest.raises(ValueError, match="Face.bbox_"):
        Face(picture_id=1, bbox_=text)

    face = Face(picture_id=1)
    with pytest.raises(ValueError, match="Face.bbox_"):
        face.bbox = [0, 0, 10**400, 10]
    assert face.bbox_ is None


@pytest.mark.parametrize("model", [Face, Detection], ids=["face", "detection"])
def test_the_largest_finite_coordinates_are_kept(model):
    # The positive control: the check refuses what no float can hold, not
    # what is merely large.
    largest = sys.float_info.max
    box = [-largest, 0, largest, 10**308]
    row = model(picture_id=1, bbox=box)
    assert row.bbox == box
    assert model(picture_id=1, bbox_=json.dumps(box)).bbox == box


@pytest.mark.parametrize(
    "box",
    [[0, 0, 10, 10], [0.5, 1.25, 30.0, 40.75], [-3, -4, 5, 6]],
)
def test_a_four_number_detection_box_round_trips_through_the_database(box):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[Picture.__table__, Detection.__table__]
    )
    with Session(engine) as session:
        picture = Picture(file_path="a.png", format="PNG", width=64, height=64)
        session.add(picture)
        session.flush()
        session.add(Detection(picture_id=picture.id, label="hand", bbox=box))
        session.add(
            Detection(
                picture_id=picture.id,
                detection_index=1,
                label="hand",
                bbox_=json.dumps(box),
            )
        )
        session.commit()
    with Session(engine) as session:
        rows = session.exec(select(Detection).order_by(Detection.detection_index))
        assert [d.bbox for d in rows.all()] == [box, box]


def test_no_detection_box_and_a_null_box_both_read_as_none():
    assert Detection(picture_id=1).bbox is None
    detection = Detection(picture_id=1, bbox=[0, 0, 1, 1])
    detection.bbox = None
    assert detection.bbox_ == "null"
    assert detection.bbox is None
    assert Detection(picture_id=1, bbox_="null").bbox is None


_FACE_LOGGER = "pixlstash.db_models.face"


def _refusal_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _FACE_LOGGER]


@pytest.mark.parametrize("model", [Face, Detection], ids=["face", "detection"])
@pytest.mark.parametrize("text", BAD_TEXTS)
def test_a_refused_box_text_is_logged_not_echoed_in_the_error(model, text, caplog):
    # The error's text can travel (a restore logs it, a route may answer
    # with it), so it is fixed; the refused value goes to the log, cut short.
    owner = f"{model.__name__}.bbox_"
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        with pytest.raises(ValueError) as excinfo:
            model(picture_id=1, bbox_=text)
    assert str(excinfo.value) in {
        f"{owner} must be JSON text",
        f"{owner} must be null or [x1, y1, x2, y2] of finite numbers",
    }
    (line,) = _refusal_lines(caplog)
    assert line.startswith(f"{owner} refused a bbox text"), line
    assert repr(text)[:40] in line
    assert len(line) < 300, "the whole refused text was logged"


def test_a_box_value_that_is_not_text_is_refused_by_its_type(caplog):
    # Assigned past the setter: json.loads raises TypeError on an int, and the
    # repr of one past the int-string limit raises too, so neither the error
    # nor the log line may be built from it.
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        with pytest.raises(ValueError, match=r"^Face\.bbox_ must be JSON text$"):
            Face(picture_id=1, bbox_=10**5000)
    (line,) = _refusal_lines(caplog)
    assert "a int, not text" in line


@pytest.mark.parametrize("model", [Face, Detection], ids=["face", "detection"])
def test_a_box_text_nested_past_the_recursion_limit_is_refused_as_not_json(
    model, caplog
):
    # The parse's RecursionError is malformed JSON like any other: the fixed
    # ValueError, and a log line naming the error, not a RuntimeError out of
    # the attribute set.
    owner = f"{model.__name__}.bbox_"
    with caplog.at_level(logging.WARNING, logger=_FACE_LOGGER):
        with pytest.raises(ValueError, match=rf"^{owner} must be JSON text$") as info:
            model(picture_id=1, bbox_=_TOO_DEEP)
    assert isinstance(info.value.__cause__, RecursionError)
    (line,) = _refusal_lines(caplog)
    assert "is not JSON (RecursionError)" in line, line
    assert f"({len(_TOO_DEEP)} chars)" in line
