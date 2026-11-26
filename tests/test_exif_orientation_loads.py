"""Every image loader turns a picture upright by its EXIF orientation.

All of them go through ``ImageUtils.apply_exif_orientation``, which reads tag
274 and makes the turn without writing the EXIF back. ``ImageOps.exif_transpose``
writes it back, and that write raises for EXIF Pillow reads but cannot write
(here: one LONG8 tag beside the orientation), so such a picture came out of
the loaders as ``None``: no thumbnail, and sideways or untagged downstream.
EXIF too damaged to read (a header that does not parse, one cut short, a PNG
raw profile that is not hex) holds no orientation, and is shown as stored
rather than costing the picture its thumbnail.

The first tests measure that premise in the installed Pillow; if they fail, the
fixtures no longer reproduce the damage and the rest prove nothing.
"""

from __future__ import annotations

import io
import re
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps, PngImagePlugin

from pixlstash.services.folder_structure_commit_service import _build_managed_picture
from pixlstash.tasks.quality_task import QualityTask
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask
from pixlstash.tasks.thumbnail_generation_task import ThumbnailGenerationTask
from pixlstash.utils.image_processing import image_utils as image_utils_module
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.orientation import ORIENTATION_TAG
from pixlstash.utils.unprocessable_image_registry import load_signed

# Stored 80 wide and 60 high, one colour per quadrant: every one of the eight
# orientations puts the colours somewhere different.
_SIZE = (80, 60)
_QUADRANT_COLOURS = ((220, 30, 30), (30, 200, 30), (30, 30, 220), (240, 240, 240))
_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
# An EXIF block whose TIFF header does not parse at all.
_UNPARSEABLE_EXIF = b"XX" + b"\x00" * 30
# A big-endian TIFF header cut short before its first IFD offset.
_TRUNCATED_EXIF = b"MM\x00*\x00\x00"
# Each EXIF block too damaged to read, and what Pillow's getexif() raises for it.
_UNREADABLE_EXIF = {
    "unparseable": (_UNPARSEABLE_EXIF, SyntaxError),
    "truncated": (_TRUNCATED_EXIF, struct.error),
}
# Formats whose EXIF Pillow parses only when asked, with the save options that
# keep it so: a JPEG with no JFIF resolution has its EXIF read (and its error
# swallowed) on open, one with a resolution does not.
_EXIF_PARSED_ON_ASKING = {"PNG": {}, "WEBP": {}, "JPEG": {"dpi": (72, 72)}}


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


def _healthy_exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = orientation
    return exif.tobytes()


def _unwritable_exif(orientation: int) -> bytes:
    """EXIF Pillow reads and cannot write back: the orientation and one LONG8 tag."""

    def entry(tag, kind, value):
        return struct.pack("<HHI", tag, kind, 1) + value

    return (
        b"II"
        + struct.pack("<HI", 42, 8)
        + struct.pack("<H", 2)
        + entry(ORIENTATION_TAG, 3, struct.pack("<HH", orientation, 0))
        + entry(0x9999, 16, struct.pack("<I", 0))
        + struct.pack("<I", 0)
    )


def _save(directory, name: str, exif: bytes, fmt: str, **save_options) -> str:
    """Save the quadrant picture as *fmt* carrying *exif*; return its path."""
    path = str(directory / f"{name}.{_FORMATS[fmt]}")
    options = {} if fmt == "PNG" else {"quality": 95}
    options.update(save_options)
    # The APP1 prefix a JPEG needs; PNG and WebP readers accept it too.
    _quadrants().save(path, fmt, exif=b"Exif\x00\x00" + exif, **options)
    return path


def _save_unreadable(directory, damage: str, fmt: str) -> str:
    """Save the quadrant picture as *fmt* with the *damage* EXIF block."""
    exif, _error = _UNREADABLE_EXIF[damage]
    return _save(directory, damage, exif, fmt, **_EXIF_PARSED_ON_ASKING[fmt])


def _upright(orientation: int) -> np.ndarray:
    """The quadrant picture as Pillow itself turns it, from undamaged EXIF."""
    buffer = io.BytesIO()
    _quadrants().save(buffer, "PNG", exif=_healthy_exif(orientation))
    buffer.seek(0)
    with Image.open(buffer) as img:
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def _quadrant_samples(arr: np.ndarray) -> list:
    """The colour at the centre of each quadrant, row by row."""
    h, w = arr.shape[:2]
    return [
        arr[y, x].astype(int).tolist()
        for y in (h // 4, 3 * h // 4)
        for x in (w // 4, 3 * w // 4)
    ]


def _assert_turned_by_6(arr: np.ndarray, what: str) -> None:
    """Stored 80x60 with orientation 6: seen 60 wide and 80 high, colours turned.

    Compared by quadrant colour with a margin, as JPEG and the WebP thumbnail
    are lossy.
    """
    assert arr.shape[:2] == (80, 60), f"{what}: {arr.shape[:2]}, sideways"
    for got, want in zip(_quadrant_samples(arr), _quadrant_samples(_upright(6))):
        assert max(abs(g - w) for g, w in zip(got, want)) < 40, (
            f"{what}: turned the wrong way, {got} where {want} belongs"
        )


def _via_load_image_or_video(path, monkeypatch):
    return ImageUtils.load_image_or_video(path)


def _via_load_image_reduced(path, monkeypatch):
    return ImageUtils.load_image_reduced(path, 1024)


def _via_load_image_bgr_reduced(path, monkeypatch):
    bgr, inv_scale = ImageUtils.load_image_bgr_reduced(path, 1024)
    assert bgr is not None
    assert inv_scale == 1.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _via_load_image_or_video_bgr_pil_branch(path, monkeypatch):
    # OpenCV reads first; refuse it so the Pillow branch runs.
    monkeypatch.setattr(image_utils_module.cv2, "imread", lambda *_a, **_k: None)
    bgr = ImageUtils.load_image_or_video_bgr(path)
    assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _via_render_thumbnail(path, monkeypatch):
    # As import renders it: from the open file, EXIF and all.
    with Image.open(path) as img:
        rendered = ImageUtils.render_thumbnail(img)
    assert rendered is not None
    data, width, height, _crop = rendered
    with Image.open(io.BytesIO(data)) as thumbnail:
        arr = np.array(thumbnail.convert("RGB"))
    assert arr.shape[:2] == (height, width)
    return arr


_LOADERS = {
    "load_image_or_video": _via_load_image_or_video,
    "load_image_reduced": _via_load_image_reduced,
    "load_image_bgr_reduced": _via_load_image_bgr_reduced,
    "load_image_or_video_bgr": _via_load_image_or_video_bgr_pil_branch,
    "render_thumbnail": _via_render_thumbnail,
}


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_the_installed_pillow_cannot_write_this_exif_back(tmp_path, fmt):
    path = _save(tmp_path, "unwritable", _unwritable_exif(6), fmt)
    with Image.open(path) as img:
        assert img.getexif().get(ORIENTATION_TAG) == 6, "the orientation reads"
    with Image.open(path) as img:
        with pytest.raises(struct.error):
            ImageOps.exif_transpose(img)


@pytest.mark.parametrize("fmt", sorted(_EXIF_PARSED_ON_ASKING))
@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_the_installed_pillow_cannot_parse_this_exif_header(tmp_path, damage, fmt):
    path = _save_unreadable(tmp_path, damage, fmt)
    _exif, error = _UNREADABLE_EXIF[damage]
    with Image.open(path) as img:
        with pytest.raises(error):
            img.getexif()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_every_orientation_turns_the_picture_as_pillow_does(tmp_path, orientation):
    """Lossless, so pixel for pixel, against Pillow's own turn of sound EXIF."""
    expected = _upright(orientation)
    healthy = _save(tmp_path, "healthy", _healthy_exif(orientation), "PNG")
    unwritable = _save(tmp_path, "unwritable", _unwritable_exif(orientation), "PNG")

    assert np.array_equal(ImageUtils.load_image_or_video(healthy), expected)
    assert np.array_equal(ImageUtils.load_image_or_video(unwritable), expected)


@pytest.mark.parametrize("loader", sorted(_LOADERS))
@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_every_loader_turns_a_picture_whose_exif_cannot_be_written_back(
    tmp_path, monkeypatch, fmt, loader
):
    """Orientation 6: stored 80x60, seen 60 wide and 80 high."""
    path = _save(tmp_path, "unwritable", _unwritable_exif(6), fmt)

    arr = _LOADERS[loader](path, monkeypatch)

    assert arr is not None, f"{loader} answered None for a {fmt} that decodes"
    _assert_turned_by_6(arr, f"{loader} on a {fmt}")


def test_a_picture_whose_exif_cannot_be_written_back_imports_with_a_thumbnail(
    tmp_path,
):
    """Import renders the thumbnail from the open file; that used to fail it."""
    source = _save(tmp_path, "unwritable", _unwritable_exif(6), "PNG")
    with open(source, "rb") as handle:
        data = handle.read()
    library = tmp_path / "library"
    library.mkdir()

    picture = ImageUtils.create_picture_from_bytes(str(library), data)

    assert (picture.format, picture.orientation) == ("PNG", 6)
    assert (picture.width, picture.height) == (80, 60), "stored size"
    assert (picture.thumbnail_width, picture.thumbnail_height) == (60, 80)


def test_the_reduced_bgr_load_scales_back_to_the_full_size(tmp_path):
    """Face boxes are scaled by ``inv_scale``: full size 60x80 over loaded 30x40."""
    path = _save(tmp_path, "unwritable", _unwritable_exif(6), "JPEG")

    bgr, inv_scale = ImageUtils.load_image_bgr_reduced(path, 40)

    assert bgr.shape[:2] == (40, 30)
    assert inv_scale == 2.0


@pytest.mark.parametrize("loader", sorted(_LOADERS))
def test_a_tiff_that_pillow_turns_as_it_loads_is_not_turned_twice(
    tmp_path, monkeypatch, loader
):
    """Pillow turns a TIFF by the orientation in its IFD as it loads it."""
    path = str(tmp_path / "turned.tif")
    _quadrants().save(path, "TIFF", tiffinfo={ORIENTATION_TAG: 6})
    with Image.open(path) as img:
        assert ImageUtils.exif_orientation(img) == 6, "stored turned"

    arr = _LOADERS[loader](path, monkeypatch)

    assert arr is not None
    _assert_turned_by_6(arr, f"{loader} on a TIFF")


def test_a_turned_picture_carries_no_orientation_to_turn_it_twice(tmp_path):
    path = _save(tmp_path, "unwritable", _unwritable_exif(6), "PNG")
    with Image.open(path) as img:
        upright = ImageUtils.apply_exif_orientation(img)

    assert upright.size == (60, 80)
    assert ImageUtils.exif_orientation(upright) == 1
    assert ImageUtils.apply_exif_orientation(upright) is upright


@pytest.mark.parametrize(
    "key, packet",
    [
        ("XML:com.adobe.xmp", '<rdf:Description tiff:Orientation="6"/>'),
        ("xmp", b"<tiff:Orientation>6</tiff:Orientation>"),
    ],
)
def test_an_xmp_orientation_is_applied_once(key, packet):
    """Pillow reads an XMP orientation when the EXIF has none; so does the turn."""
    img = _quadrants()
    img.info[key] = packet

    upright = ImageUtils.apply_exif_orientation(img)

    assert upright.size == (60, 80)
    assert ImageUtils.exif_orientation(upright) == 1
    assert np.array_equal(np.array(upright), _upright(6))


@pytest.mark.parametrize("value", [0, 9])
def test_a_reserved_orientation_is_shown_as_stored(tmp_path, value):
    path = _save(tmp_path, "reserved", _healthy_exif(value), "PNG")
    with Image.open(path) as img:
        assert ImageUtils.exif_orientation(img) == 1

    assert np.array_equal(ImageUtils.load_image_or_video(path), np.array(_quadrants()))


@pytest.mark.parametrize("fmt", sorted(_EXIF_PARSED_ON_ASKING))
@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_an_exif_header_that_does_not_parse_is_shown_as_stored(tmp_path, damage, fmt):
    """No orientation can be read from it, so the picture is shown as stored.

    That is how a browser shows it. The turn used to raise, and every loader
    answered ``None`` for a picture that decodes. Each read here is the first
    on a freshly opened file: Pillow answers an empty EXIF, without raising,
    the second time it is asked.
    """
    path = _save_unreadable(tmp_path, damage, fmt)

    with Image.open(path) as img:
        assert ImageUtils.exif_orientation(img) == 1
    with Image.open(path) as img:
        upright = ImageUtils.apply_exif_orientation(img)
        assert upright is img
        assert upright.size == _SIZE


@pytest.mark.parametrize("loader", sorted(_LOADERS))
@pytest.mark.parametrize("fmt", sorted(_EXIF_PARSED_ON_ASKING))
@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_every_loader_shows_a_picture_whose_exif_header_does_not_parse(
    tmp_path, monkeypatch, damage, fmt, loader
):
    path = _save_unreadable(tmp_path, damage, fmt)
    # OpenCV's video reader opens a WebP too; refuse it so this measures the
    # Pillow branch, which every caller relies on for the orientation.
    monkeypatch.setattr(
        image_utils_module.VideoUtils,
        "read_first_video_frame_bgr",
        lambda *_a, **_k: None,
    )

    arr = _LOADERS[loader](path, monkeypatch)

    assert arr is not None, f"{loader} answered None for a {fmt} that decodes"
    assert arr.shape[:2] == (_SIZE[1], _SIZE[0]), f"{loader} on a {fmt}: turned"


@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_a_picture_whose_exif_header_does_not_parse_imports_as_a_picture(
    tmp_path, damage
):
    """It was read as a video: the import's orientation read raised first."""
    source = _save_unreadable(tmp_path, damage, "PNG")
    with open(source, "rb") as handle:
        data = handle.read()
    library = tmp_path / "library"
    library.mkdir()

    picture = ImageUtils.create_picture_from_bytes(str(library), data)

    assert (picture.format, picture.orientation) == ("PNG", 1)
    assert (picture.width, picture.height) == _SIZE
    assert (picture.thumbnail_width, picture.thumbnail_height) == _SIZE


@pytest.mark.parametrize("fmt", ["JPEG", "TIFF"])
def test_a_turned_picture_records_its_stored_size_whatever_its_format(tmp_path, fmt):
    """Width and height are as stored; the frontend swaps them for 5-8.

    Pillow swaps a TIFF's size on open when it will turn it as it loads, so a
    TIFF recorded its upright size, and the frontend swapped it back sideways.
    """
    buffer = io.BytesIO()
    if fmt == "TIFF":
        _quadrants().save(buffer, "TIFF", tiffinfo={ORIENTATION_TAG: 6})
    else:
        _quadrants().save(buffer, "JPEG", exif=_healthy_exif(6))
    library = tmp_path / "library"
    library.mkdir()

    picture = ImageUtils.create_picture_from_bytes(str(library), buffer.getvalue())

    assert (picture.format, picture.orientation) == (fmt, 6)
    assert (picture.width, picture.height) == _SIZE, "stored size"
    assert (picture.thumbnail_width, picture.thumbnail_height) == (60, 80)


def _scanned_size(path, library):
    task = ReferenceFolderScanTask(
        SimpleNamespace(image_root=str(library)), 1, str(path.parent), str(path.parent)
    )
    picture = task._build_picture(str(path), "test-sha", 1)
    return picture.width, picture.height


def _committed_size(path, library):
    picture = _build_managed_picture(str(path), path.name, str(library))
    return picture.width, picture.height


def _backfilled_size(path, library):
    task = QualityTask(database=None, pictures=[])
    return task._stored_size(SimpleNamespace(id=1, file_path=path.name), str(path))


# Every other place that records a picture's size from a file on disk.
_ON_DISK_SIZE_READERS = {
    "reference scan": _scanned_size,
    "local import": _committed_size,
    "quality backfill": _backfilled_size,
}


@pytest.mark.parametrize("reader", sorted(_ON_DISK_SIZE_READERS))
def test_every_reader_of_a_file_on_disk_records_a_turned_tiff_at_its_stored_size(
    tmp_path, reader
):
    """The reference scan, the local import and the quality backfill record a
    TIFF with orientation 6 at its stored size, as import does."""
    library = tmp_path / "library"
    library.mkdir()
    path = library / "test-turned.tif"
    _quadrants().save(path, "TIFF", tiffinfo={ORIENTATION_TAG: 6})
    with Image.open(path) as img:
        assert img.size == (60, 80), "Pillow no longer turns this TIFF's size"

    assert _ON_DISK_SIZE_READERS[reader](path, library) == _SIZE


def test_the_quality_backfill_records_a_video_at_its_frame_size(tmp_path):
    """Pillow does not open a video, so its decoded frame gives the size."""
    path = tmp_path / "test-clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, _SIZE)
    for shade in (40, 120, 200):
        writer.write(np.full((_SIZE[1], _SIZE[0], 3), shade, np.uint8))
    writer.release()

    assert _backfilled_size(path, tmp_path) == _SIZE


@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_the_thumbnail_hold_probes_with_the_loaders_own_turn(
    tmp_path, monkeypatch, damage
):
    """Held as an orientation failure only when the loaders' turn fails too.

    The loaders turn EXIF they cannot write back and show EXIF that does not
    parse as stored, so neither is held; a turn that still raises is.
    """
    unwritable = _save(tmp_path, "unwritable", _unwritable_exif(6), "PNG")
    unparseable = _save_unreadable(tmp_path, damage, "PNG")
    task = ThumbnailGenerationTask(database=None, pictures=[])

    task._note_orientation_failure(1, load_signed(unwritable, multiframe_count=1))
    task._note_orientation_failure(2, load_signed(unparseable, multiframe_count=1))
    assert task._orientation_failures == [], "the loaders read both: no hold"

    def failing_turn(img):
        raise ValueError("test-turn-fails")

    monkeypatch.setattr(ImageUtils, "apply_exif_orientation", failing_turn)
    task._note_orientation_failure(3, load_signed(unparseable, multiframe_count=1))
    assert [entry[:2] for entry in task._orientation_failures] == [(3, unparseable)]


@pytest.mark.parametrize("damage", sorted(_UNREADABLE_EXIF))
def test_a_picture_whose_exif_does_not_parse_gets_its_thumbnail(tmp_path, damage):
    """The thumbnail task renders it as stored: no ORIENTATION_FAILED hold."""
    path = _save_unreadable(tmp_path, damage, "PNG")
    database = SimpleNamespace(image_root=str(tmp_path), unprocessable_images=None)
    picture = SimpleNamespace(id=1, file_path=path.rsplit("/", 1)[-1], faces=[])
    task = ThumbnailGenerationTask(database=database, pictures=[picture])

    columns = task._resolve_columns(picture)

    assert task._orientation_failures == [], "held as ORIENTATION_FAILED"
    assert columns is not None, "no thumbnail for a picture that decodes"
    assert (columns["thumbnail_width"], columns["thumbnail_height"]) == _SIZE


def _save_with_raw_profile(directory, text: str) -> str:
    """Save the quadrant picture as a PNG whose EXIF rides in a raw profile."""
    path = str(directory / "raw-profile.png")
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Raw profile type exif", text)
    _quadrants().save(path, "PNG", pnginfo=pnginfo)
    return path


# A "Raw profile type exif" text chunk whose payload is not hex.
_NOT_HEX_RAW_PROFILE = "\nexif\n      10\nzz-not-hex\n"


def test_the_installed_pillow_cannot_read_a_raw_profile_that_is_not_hex(tmp_path):
    path = _save_with_raw_profile(tmp_path, _NOT_HEX_RAW_PROFILE)
    with Image.open(path) as img:
        with pytest.raises(ValueError):
            img.getexif()


def test_a_raw_profile_that_is_not_hex_is_shown_as_stored(tmp_path):
    path = _save_with_raw_profile(tmp_path, _NOT_HEX_RAW_PROFILE)

    with Image.open(path) as img:
        assert ImageUtils.exif_orientation(img) == 1
    assert np.array_equal(ImageUtils.load_image_or_video(path), np.array(_quadrants()))


# Carriers of an orientation besides the EXIF block, for the writers that keep
# a picture's text and XMP (share copies, the served PNG).
_XMP_ORIENTATION = re.compile(rb'tiff:Orientation(?:="|>)\s*(\d+)')


def _xmp_packet(orientation: int) -> bytes:
    """An XMP packet with a description and a tiff:Orientation."""
    return (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf='
        b'"http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
        b'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
        b'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        b'tiff:Orientation="%d"><dc:description>test-kept</dc:description>'
        b"</rdf:Description></rdf:RDF></x:xmpmeta>" % orientation
    )


def _raw_profile_text(orientation: int) -> str:
    """A PNG "Raw profile type exif" chunk carrying *orientation*."""
    raw = _healthy_exif(orientation)
    return "\nexif\n%8d\n%s\n" % (len(raw), raw.hex())


def _orientations_carried(data: bytes) -> dict:
    """Every orientation the encoded picture carries, by carrier.

    The EXIF block, each XMP packet (JPEG and WebP: ``xmp``; PNG: the iTXt
    ``XML:com.adobe.xmp``), and PNG's ``Raw profile type exif`` text chunk.
    A carrier the file does not have is left out.
    """
    carried = {}
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        exif = img.info.get("exif")
        if exif:
            block = Image.Exif()
            block.load(exif)
            carried["exif"] = block.get(ORIENTATION_TAG)
        for key in ("xmp", "XML:com.adobe.xmp"):
            packet = img.info.get(key)
            if packet:
                packet = packet.encode() if isinstance(packet, str) else packet
                found = _XMP_ORIENTATION.search(packet)
                carried[key] = int(found.group(1)) if found else None
        raw_profile = img.info.get("Raw profile type exif")
        if raw_profile is not None:
            block = Image.Exif()
            block.load(bytes.fromhex("".join(raw_profile.split("\n")[3:])))
            carried["Raw profile type exif"] = block.get(ORIENTATION_TAG)
    return carried


def _assert_carried_upright(data: bytes, what: str) -> None:
    """No carrier in *data* would turn its (already upright) pixels again."""
    carried = _orientations_carried(data)
    turning = {k: v for k, v in carried.items() if v not in (None, 1)}
    assert not turning, f"{what}: upright pixels still carry {turning}"


def test_the_orientation_carriers_are_all_read():
    """The helper the writers' tests rely on finds each carrier's 6."""
    jpeg, webp, png = io.BytesIO(), io.BytesIO(), io.BytesIO()
    _quadrants().save(jpeg, "JPEG", exif=_healthy_exif(6), xmp=_xmp_packet(6))
    _quadrants().save(webp, "WEBP", exif=_healthy_exif(6), xmp=_xmp_packet(6))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Raw profile type exif", _raw_profile_text(6))
    pnginfo.add_itxt("XML:com.adobe.xmp", _xmp_packet(6).decode())
    _quadrants().save(png, "PNG", pnginfo=pnginfo)

    assert _orientations_carried(jpeg.getvalue()) == {"exif": 6, "xmp": 6}
    assert _orientations_carried(webp.getvalue()) == {"exif": 6, "xmp": 6}
    assert _orientations_carried(png.getvalue()) == {
        "xmp": 6,
        "XML:com.adobe.xmp": 6,
        "Raw profile type exif": 6,
    }
