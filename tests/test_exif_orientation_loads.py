"""Every image loader turns a picture upright by its EXIF orientation.

The fixture carries EXIF that Pillow reads but cannot write back: the
orientation beside one LONG8 tag. Writing it back raises ``struct.error``, so a
turn must not re-serialise the EXIF block.
"""

from __future__ import annotations

import io
import struct

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps, PngImagePlugin

from pixlstash.utils.image_processing import image_utils as image_utils_module
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.orientation import ORIENTATION_TAG
from pixlstash.utils.service.share_copies import make_share_copies

# Stored 80 wide and 60 high, one colour per quadrant, so a turn the wrong way
# puts the colours in the wrong place.
_SIZE = (80, 60)
_QUADRANT_COLOURS = ((220, 30, 30), (30, 200, 30), (30, 30, 220), (240, 240, 240))


def _quadrants() -> Image.Image:
    img = Image.new("RGB", _SIZE)
    half_w, half_h = _SIZE[0] // 2, _SIZE[1] // 2
    boxes = (
        (0, 0, half_w, half_h),
        (half_w, 0, _SIZE[0], half_h),
        (0, half_h, half_w, _SIZE[1]),
        (half_w, half_h, _SIZE[0], _SIZE[1]),
    )
    for box, colour in zip(boxes, _QUADRANT_COLOURS):
        img.paste(colour, box)
    return img


def _unwritable_exif(orientation: int) -> bytes:
    """EXIF Pillow reads and cannot write back: the orientation and one LONG8 tag."""

    def entry(tag, kind, value):
        return struct.pack("<HHI", tag, kind, 1) + value

    return (
        b"Exif\x00\x00II"
        + struct.pack("<HI", 42, 8)
        + struct.pack("<H", 2)
        + entry(ORIENTATION_TAG, 3, struct.pack("<HH", orientation, 0))
        + entry(0x9999, 16, struct.pack("<I", 0))
        + struct.pack("<I", 0)
    )


def _upright_by_6() -> np.ndarray:
    """The quadrant picture as Pillow turns it from EXIF it can write back."""
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = 6
    buffer = io.BytesIO()
    _quadrants().save(buffer, "PNG", exif=exif.tobytes())
    buffer.seek(0)
    with Image.open(buffer) as img:
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def _quadrant_samples(arr: np.ndarray) -> list:
    h, w = arr.shape[:2]
    return [
        arr[y, x].astype(int).tolist()
        for y in (h // 4, 3 * h // 4)
        for x in (w // 4, 3 * w // 4)
    ]


def _via_load_image_or_video(path, monkeypatch):
    return ImageUtils.load_image_or_video(path)


def _via_load_image_reduced(path, monkeypatch):
    return ImageUtils.load_image_reduced(path, 1024)


def _via_load_image_bgr_reduced(path, monkeypatch):
    bgr, _inv_scale = ImageUtils.load_image_bgr_reduced(path, 1024)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _via_load_image_or_video_bgr(path, monkeypatch):
    # OpenCV reads first; refuse it so the Pillow branch runs.
    monkeypatch.setattr(image_utils_module.cv2, "imread", lambda *_a, **_k: None)
    bgr = ImageUtils.load_image_or_video_bgr(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _via_render_thumbnail(path, monkeypatch):
    with Image.open(path) as img:
        rendered = ImageUtils.render_thumbnail(img)
    if rendered is None:
        return None
    with Image.open(io.BytesIO(rendered[0])) as thumbnail:
        return np.array(thumbnail.convert("RGB"))


@pytest.mark.parametrize(
    "load",
    [
        _via_load_image_or_video,
        _via_load_image_reduced,
        _via_load_image_bgr_reduced,
        _via_load_image_or_video_bgr,
        _via_render_thumbnail,
    ],
)
def test_a_photo_whose_exif_cannot_be_written_back_loads_upright(
    tmp_path, monkeypatch, load
):
    path = str(tmp_path / "photo.jpg")
    _quadrants().save(path, "JPEG", quality=95, exif=_unwritable_exif(6))

    arr = load(path, monkeypatch)

    assert arr is not None, "the photo did not load"
    assert arr.shape[:2] == (80, 60), "the photo was not turned"
    # JPEG and the WebP thumbnail are lossy, so colours compare with a margin.
    for got, want in zip(_quadrant_samples(arr), _quadrant_samples(_upright_by_6())):
        assert max(abs(g - w) for g, w in zip(got, want)) < 40, (got, want)


def test_a_resized_share_copy_of_such_a_photo_is_made_upright(tmp_path):
    image_root = tmp_path / "library"
    image_root.mkdir()
    _quadrants().save(
        image_root / "photo.jpg", "JPEG", quality=95, exif=_unwritable_exif(6)
    )
    picture = {"id": 1, "file_path": "photo.jpg", "original_file_name": "photo.jpg"}

    result = make_share_copies(
        [picture],
        str(image_root),
        str(tmp_path / "share-copies"),
        keep_generation_data=True,
        max_long_edge=40,
    )

    assert result["skipped"] == []
    (entry,) = result["files"]
    with Image.open(entry["path"]) as img:
        assert img.size == (30, 40)
        assert img.getexif().get(ORIENTATION_TAG) is None


def _png_with_exif_that_does_not_parse(damage: str) -> bytes:
    """A PNG whose EXIF block Pillow cannot read, one way per error it raises."""
    kwargs = {}
    if damage == "header":
        kwargs["exif"] = b"XX*\x00\x08\x00\x00\x00"  # SyntaxError
    elif damage == "cut-short":
        kwargs["exif"] = b"II+\x00\x08\x00\x00\x00"  # struct.error
    else:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Raw profile type exif", "\nexif\n   8\nnot hex\n")
        kwargs["pnginfo"] = pnginfo  # ValueError
    buffer = io.BytesIO()
    _quadrants().save(buffer, "PNG", **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize("damage", ["header", "cut-short", "raw-profile-not-hex"])
def test_a_picture_whose_exif_does_not_parse_imports_as_stored(tmp_path, damage):
    data = _png_with_exif_that_does_not_parse(damage)
    path = tmp_path / "picture.png"
    path.write_bytes(data)

    arr = ImageUtils.load_image_or_video(str(path))
    picture = ImageUtils.create_picture_from_bytes(
        image_root_path=str(tmp_path / "library"),
        image_bytes=data,
        picture_uuid="picture.png",
    )

    assert arr is not None and arr.shape[:2] == (60, 80)
    assert (picture.width, picture.height, picture.orientation) == (80, 60, 1)
    assert picture.thumbnail_width is not None


def test_a_resized_share_copy_of_a_picture_whose_exif_does_not_parse_is_made(
    tmp_path,
):
    image_root = tmp_path / "library"
    image_root.mkdir()
    (image_root / "picture.png").write_bytes(
        _png_with_exif_that_does_not_parse("header")
    )
    picture = {"id": 1, "file_path": "picture.png", "original_file_name": "picture.png"}

    result = make_share_copies(
        [picture],
        str(image_root),
        str(tmp_path / "share-copies"),
        keep_generation_data=True,
        max_long_edge=40,
    )

    assert result["skipped"] == []
    (entry,) = result["files"]
    with Image.open(entry["path"]) as img:
        assert img.size == (40, 30)
