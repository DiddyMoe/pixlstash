"""Tests for Picture.is_video persistence and fetch_best_picture_id behavior.

Every test here drives the real production code path, because the flag is only
ever wrong in production code:

1. The import ctor (``ImageUtils.create_picture_from_bytes``) sets the flag from
   PIL decode success/failure -- asserted in *both* directions, since the Python
   field default is ``False`` and a still-image-only test passes with the
   classification deleted.
2. The reference-folder scan (``ReferenceFolderScanTask._build_picture``) sets it
   from the file extension -- the second writer of the column.
3. Migration 0096 backfills it from ``file_path``, run as a real Alembic upgrade
   over a seeded pre-0096 database. Pasting the migration's ``UPDATE`` into the
   test body would only assert that SQLite's ``LIKE`` works.
4. ``fetch_best_picture_id`` prefers still images over higher-scored videos,
   exercised through ``GET /characters/{id}/thumbnail``. That function is a
   closure inside the route handler and cannot be imported, so the route is the
   only way to run it; it decides the value written into the thumbnail cache's
   metadata sidecar, which is what the test reads back.
5. Every path that ingests a file already on disk dates it the same way (EXIF,
   then a video's movie header, then the file's mtime) and reads it from its
   path rather than whole into memory - asserted over all four such paths, so
   a sibling left on the old bytes path is red.
"""

import contextlib
import gc
import glob
import json
import os
import sqlite3
import struct
import tempfile
import tracemalloc
from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
from sqlalchemy import text
from sqlalchemy.orm import configure_mappers
from sqlmodel import Session, select

from pixlstash.db_models import Face, Picture
from pixlstash.server import Server
from pixlstash.services.folder_structure_commit_service import _build_managed_picture
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.video_utils import VideoUtils
from pixlstash.utils.likeness.likeness_parameter_utils import LikenessParameterUtils
from pixlstash.vault import Vault
from tests.test_migrations import (
    _MIGRATIONS_DIR,
    _insert_minimal_row,
    _run_alembic,
)
from tests.utils import upload_pictures_and_wait


class _FakeVideoCapture:
    """Stand-in for ``cv2.VideoCapture`` that yields a single black frame.

    Mirrors ``tests/test_picture_plugins.py``. The branch under test is selected
    by PIL *failing* to decode the bytes, so undecodable bytes are the genuine
    trigger and are passed unmodified; the fake only supplies the frame the
    thumbnail step then needs, which keeps a binary video fixture out of the
    repository.
    """

    def __init__(self, _path):
        self._read = False

    def read(self):
        if self._read:
            return False, None
        self._read = True
        return True, np.zeros((32, 48, 3), dtype=np.uint8)

    def release(self):
        return None


class TestPictureIsVideoImport:
    """Test the import path sets is_video flag correctly."""

    def test_still_image_sets_is_video_false(self, tmp_path):
        """An imported still image (jpg) gets is_video=False."""
        # Create a minimal JPEG in memory
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Import the picture
        pic = ImageUtils.create_picture_from_bytes(
            image_root_path=str(tmp_path),
            image_bytes=img_bytes.getvalue(),
            picture_uuid="test.jpg",
        )

        assert pic.is_video is False, "Still image should have is_video=False"

    def test_video_bytes_set_is_video_true(self, tmp_path, monkeypatch):
        """An imported video gets is_video=True.

        This is the direction the Python field default cannot fake: without the
        ``is_video=`` argument on the ``Picture(...)`` constructor the record
        comes back ``False`` here, which is exactly the classification bug the
        column exists to prevent.
        """
        monkeypatch.setattr(
            "pixlstash.utils.image_processing.image_utils.cv2.VideoCapture",
            _FakeVideoCapture,
        )

        pic = ImageUtils.create_picture_from_bytes(
            image_root_path=str(tmp_path),
            image_bytes=b"not-a-decodable-image",
            picture_uuid="clip.mp4",
        )

        assert pic.is_video is True, "Video should have is_video=True"
        assert pic.format == "MP4"


class TestIsVideoColumnShape:
    """Pin the column as NOT NULL DEFAULT 0.

    This is a correctness constraint, not style. SQLite sorts NULL FIRST, so a
    nullable ``is_video`` lets a row of unknown type outrank a genuine still
    image in ``fetch_best_picture_id`` -- something the CASE this column
    replaced could never do, since it only ever yielded 0 or 1. A NULL would
    also slip past the ``is_video = 0`` guard in migration 0096 and never be
    classified. Both regressions are silent, so assert the schema directly.
    """

    def test_column_is_not_nullable_with_server_default(self):
        column = Picture.__table__.c["is_video"]
        assert column.nullable is False
        assert column.server_default is not None
        assert column.server_default.arg == "0"

    def test_a_row_inserted_without_the_flag_is_false_not_null(self, tmp_path):
        """The DB default, not just the Python default, classifies a bare row."""
        with Vault(image_root=str(tmp_path)) as vault:

            def insert_bare_row(session: Session):
                # Deliberately bypasses the ORM default so the server default is
                # what is under test.
                session.execute(
                    text(
                        "INSERT INTO picture (file_path, deleted) "
                        "VALUES ('/x/plain.jpg', 0)"
                    )
                )
                session.commit()
                return session.exec(
                    select(Picture.is_video).where(Picture.file_path == "/x/plain.jpg")
                ).one()

            assert vault.db.run_task(insert_bare_row) is False


class TestReferenceFolderScanIsVideo:
    """The reference-folder scan is the second writer of the column.

    Files indexed in place never go through ``create_picture_from_bytes``, so
    ``_build_picture`` classifies them itself from the extension. Without these
    tests that call site has no coverage at all and its ``is_video=`` argument
    can be deleted silently.
    """

    @staticmethod
    def _build(tmp_path, file_name, contents):
        folder_dir = tmp_path / "refs"
        folder_dir.mkdir()
        target = folder_dir / file_name
        target.write_bytes(contents)

        with Vault(image_root=str(tmp_path / "images")) as vault:
            task = ReferenceFolderScanTask(
                vault.db, 1, str(folder_dir), str(folder_dir)
            )
            return task._build_picture(str(target), "sha-for-test", 1)

    def test_scan_marks_a_video_file(self, tmp_path):
        """A .mp4 found on disk is indexed as a video."""
        pic = self._build(tmp_path, "clip.mp4", b"not-a-decodable-image")
        assert pic.is_video is True

    def test_scan_marks_a_still_image(self, tmp_path):
        """A .png found on disk is not."""
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (64, 64), color="blue").save(buf, format="PNG")
        pic = self._build(tmp_path, "still.png", buf.getvalue())
        assert pic.is_video is False


def _install_recording_capture(
    monkeypatch, *, frame_count: int, opened: bool = True, read_raises: bool = False
) -> dict:
    """Replace ``cv2.VideoCapture`` with a double that records what is done to it.

    Each frame's pixels hold its index, so a test can tell which frames were
    read. A real backend reports a frame count of 0 when it does not support
    the property, which is the case this double makes reachable without a
    binary fixture.
    """
    log = {"opens": 0, "releases": 0, "reads": 0, "seeks": []}

    class _RecordingCapture:
        def __init__(self, _path):
            log["opens"] += 1
            self._pos = 0

        def isOpened(self):
            return opened

        def get(self, prop):
            return float(frame_count) if prop == cv2.CAP_PROP_FRAME_COUNT else 0.0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                log["seeks"].append(int(value))
                self._pos = int(value)
            return True

        def read(self):
            log["reads"] += 1
            if read_raises:
                raise cv2.error("fake-decoder-failure")
            if frame_count > 0 and self._pos >= frame_count:
                return False, None
            frame = np.full((4, 4, 3), self._pos, dtype=np.uint8)
            self._pos += 1
            return True, frame

        def release(self):
            log["releases"] += 1

    monkeypatch.setattr(
        "pixlstash.utils.image_processing.video_utils.cv2.VideoCapture",
        _RecordingCapture,
    )
    return log


_FRAME_READERS = {
    "representative": lambda path: VideoUtils.extract_representative_video_frames(
        path, count=3
    ),
    "indices": lambda path: VideoUtils.extract_video_frames(path, [0, 1]),
    "first_bgr": VideoUtils.read_first_video_frame_bgr,
}


class TestVideoFrameSampler:
    """The video frame helpers every stage samples a clip through."""

    def test_an_unknown_frame_count_still_reads_frame_zero(self, monkeypatch):
        """A backend reporting 0 frames gets one sequential read, not none."""
        log = _install_recording_capture(monkeypatch, frame_count=0)

        frames = VideoUtils.extract_representative_video_frames(
            "/home/me/clips/test-clip.webm", count=3
        )

        assert [f.getpixel((0, 0)) for f in frames] == [(0, 0, 0)]
        assert log == {"opens": 1, "releases": 1, "reads": 1, "seeks": []}

    def test_a_known_frame_count_reads_evenly_spaced_frames(self, monkeypatch):
        log = _install_recording_capture(monkeypatch, frame_count=10)

        frames = VideoUtils.extract_representative_video_frames(
            "/home/me/clips/test-clip.mp4", count=3
        )

        assert [f.getpixel((0, 0))[0] for f in frames] == [0, 4, 9]
        assert log["seeks"] == [0, 4, 9]
        assert (log["opens"], log["releases"]) == (1, 1)

    @pytest.mark.parametrize("reader", sorted(_FRAME_READERS))
    def test_a_failing_read_still_releases_the_capture(self, monkeypatch, reader):
        log = _install_recording_capture(monkeypatch, frame_count=10, read_raises=True)

        with pytest.raises(cv2.error):
            _FRAME_READERS[reader]("/home/me/clips/test-clip.mp4")

        assert (log["opens"], log["releases"]) == (1, 1)

    @pytest.mark.parametrize("reader", sorted(_FRAME_READERS))
    def test_a_capture_that_did_not_open_is_released_unread(self, monkeypatch, reader):
        log = _install_recording_capture(monkeypatch, frame_count=10, opened=False)

        result = _FRAME_READERS[reader]("/home/me/clips/test-clip.mp4")

        if reader == "first_bgr":
            assert result is None
        else:
            assert result == []
        assert log == {"opens": 1, "releases": 1, "reads": 0, "seeks": []}


# ---------------------------------------------------------------------------
# Ingesting a file already on disk: its date, and how much of it is read
# ---------------------------------------------------------------------------

_MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
_SOURCE_MTIME = datetime(2020, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
_RECORDED_AT = datetime(2023, 8, 9, 10, 11, 12, tzinfo=timezone.utc)
_DECOY_AT = datetime(2001, 1, 1, tzinfo=timezone.utc)
_MIB = 1024 * 1024


def _seconds_since_1904(moment: datetime) -> int:
    return int((moment - _MP4_EPOCH).total_seconds())


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _mvhd(version: int, created: datetime) -> bytes:
    """A movie header box: creation and modification time, then the rest zeroed."""
    seconds = _seconds_since_1904(created)
    times = struct.pack(">II" if version == 0 else ">QQ", seconds, seconds)
    return _box(b"mvhd", bytes([version, 0, 0, 0]) + times + bytes(80))


def _write_mp4(path, *, version: int, mdat_payload: int, large_mdat: bool) -> None:
    """``ftyp | mdat | moov{mvhd}``, the movie header after the media, as a camera writes it.

    The media data opens with a box-shaped decoy - a ``moov`` holding an
    ``mvhd`` dated ``_DECOY_AT`` - so a reader that scans for the header text,
    or that descends into the media data instead of stepping over it by its
    size, answers the decoy. The rest of the media data is a sparse hole, so a
    large file costs no time to write.
    """
    decoy = _box(b"moov", _mvhd(0, _DECOY_AT))
    assert mdat_payload > len(decoy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41"))
        if large_mdat:
            handle.write(struct.pack(">I4sQ", 1, b"mdat", 16 + mdat_payload))
        else:
            handle.write(struct.pack(">I4s", 8 + mdat_payload, b"mdat"))
        handle.write(decoy)
        handle.seek(mdat_payload - len(decoy), os.SEEK_CUR)
        handle.write(_box(b"moov", _mvhd(version, _RECORDED_AT)))
    os.utime(path, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))


def _write_png_with_mvhd_text(path) -> None:
    """A PNG without EXIF whose bytes happen to contain a movie-header shape.

    A private ancillary chunk carries ``mvhd``, version 0, flags, then a
    nonzero 32-bit time: exactly what an unstructured scan takes for a
    recording date.
    """
    info = PngImagePlugin.PngInfo()
    info.add(
        b"prVt",
        b"\x00\x00\x00\x6cmvhd"
        + bytes(4)
        + struct.pack(">I", _seconds_since_1904(_DECOY_AT)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color="green").save(path, "PNG", pnginfo=info)
    os.utime(path, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))


def _write_noise_png(path, side: int) -> None:
    """A PNG of random pixels, which does not compress: file size ~ 3 * side^2."""
    pixels = np.random.default_rng(7).integers(0, 256, (side, side, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, "PNG", compress_level=0)
    os.utime(path, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))


_INGEST_PATHS = ("staged import", "reference scan", "local import", "likeness backfill")


@pytest.fixture(params=_INGEST_PATHS)
def created_at_via(request, tmp_path, monkeypatch):
    """The ``created_at`` one on-disk ingest path gives a file.

    Yields ``read(path) -> datetime``. Anything a path needs set up (a vault
    for the reference scan) is built here, outside the call, so a memory
    measurement around ``read`` sees only the ingest itself.
    """
    # A staged video is opened by OpenCV for its thumbnail frame; the fake
    # keeps a binary video fixture out of the repository.
    monkeypatch.setattr(
        "pixlstash.utils.image_processing.image_utils.cv2.VideoCapture",
        _FakeVideoCapture,
    )
    if request.param == "staged import":
        library = tmp_path / "library"

        def _staged(path):
            return ImageUtils.create_picture_from_file(
                image_root_path=str(library), source_file_path=str(path)
            ).created_at

        yield _staged
    elif request.param == "reference scan":
        with Vault(image_root=str(tmp_path / "images")) as vault:

            def _scanned(path):
                task = ReferenceFolderScanTask(
                    vault.db, 1, str(path.parent), str(path.parent)
                )
                return task._build_picture(str(path), "sha-for-test", 1).created_at

            yield _scanned
    elif request.param == "local import":

        def _local(path):
            return _build_managed_picture(
                str(path), path.name, str(path.parent)
            ).created_at

        yield _local
    else:

        def _backfilled(path):
            return LikenessParameterUtils.compute_created_at_from_file_static(
                str(path), path.name
            )

        yield _backfilled


def _traced_peak(call):
    """Run *call* under ``tracemalloc``; return its result and the heap peak.

    The ORM's mappers are configured first: the first ``Picture`` a process
    builds does that, a one-time ~3 MiB that is not the file being measured,
    and which test of a shard builds it first is down to the shard.
    """
    configure_mappers()
    tracemalloc.start()
    try:
        result = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


_FTYP = _box(b"ftyp", b"isom\x00\x00\x02\x00isom")
_DECOY_SECONDS = struct.pack(">I", _seconds_since_1904(_DECOY_AT))


class TestIngestFromAPath:
    """A file already on disk is dated and described from its path."""

    def test_mvhd_text_in_a_png_is_not_a_timestamp(self, tmp_path, created_at_via):
        """A still image is never searched for a movie header: no EXIF date
        means the file's own mtime, not a date found in its pixels or chunks."""
        source = tmp_path / "src" / "test-still.png"
        _write_png_with_mvhd_text(source)

        assert created_at_via(source) == _SOURCE_MTIME

    @pytest.mark.parametrize(
        "version,large_mdat", [(0, False), (1, True)], ids=["v0", "v1-largesize"]
    )
    def test_mp4_creation_time_is_read_with_moov_after_mdat(
        self, tmp_path, created_at_via, version, large_mdat
    ):
        """The recording time comes from the real movie header, found by
        stepping over the media data by its declared size."""
        source = tmp_path / "src" / "test-clip.mp4"
        _write_mp4(source, version=version, mdat_payload=_MIB, large_mdat=large_mdat)

        assert created_at_via(source) == _RECORDED_AT

    def test_a_cut_short_mp4_is_dated_by_its_mtime_and_says_why(
        self, tmp_path, created_at_via, caplog
    ):
        """A box header the file ends inside is a broken container, not one
        that records no time: the mtime is used and the reason is logged."""
        source = tmp_path / "src" / "test-cut-short.mp4"
        source.parent.mkdir(parents=True)
        ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isom")
        source.write_bytes(ftyp + struct.pack(">I4s", 1, b"mdat") + bytes(3))
        os.utime(source, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))

        with caplog.at_level("WARNING"):
            assert created_at_via(source) == _SOURCE_MTIME
        assert f"the 64-bit box size at byte {len(ftyp)} is cut short" in caplog.text

    @pytest.mark.parametrize("version", [0, 1], ids=["v0", "v1"])
    def test_a_movie_header_with_no_creation_time_is_dated_by_its_mtime(
        self, tmp_path, created_at_via, version
    ):
        """A creation time of 0 is how an encoder such as OpenCV's says it
        recorded none: the file's own mtime is used, not 1904-01-01."""
        source = tmp_path / "src" / "test-undated.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(_FTYP + _box(b"moov", _mvhd(version, _MP4_EPOCH)))
        os.utime(source, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))

        assert created_at_via(source) == _SOURCE_MTIME

    # Each file holds a 2001 date in the four bytes after the movie header's
    # version and flags, but outside the header. Its fields start at byte 36:
    # after the ftyp and the moov's and the mvhd's own 8-byte box headers.
    @pytest.mark.parametrize(
        "movie,reason",
        [
            (
                _box(b"moov", _box(b"mvhd", b"")) + bytes(4) + _DECOY_SECONDS,
                "the movie header at byte 36 is shorter than its fields",
            ),
            (
                _box(
                    b"moov",
                    _box(b"mvhd", b"\x01\x00\x00\x00" + bytes(4)) + _DECOY_SECONDS,
                ),
                "the movie header at byte 36 is shorter than its fields",
            ),
            (
                struct.pack(">I4s", 16, b"moov")
                + struct.pack(">I4s", 108, b"mvhd")
                + bytes(4)
                + _DECOY_SECONDS
                + bytes(96),
                "the movie header at byte 36 runs to byte 136, past the end of "
                "its moov box or of the file",
            ),
        ],
        ids=["empty", "v1-shorter-than-its-time", "past-its-moov"],
    )
    def test_a_movie_header_is_dated_only_from_inside_itself(
        self, tmp_path, created_at_via, caplog, movie, reason
    ):
        """A movie header too short for its creation time, or one running past
        its moov box, is a broken container: the bytes beyond it are not its
        date. The mtime is used and the reason is logged."""
        source = tmp_path / "src" / "test-malformed.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(_FTYP + movie)
        os.utime(source, (_SOURCE_MTIME.timestamp(), _SOURCE_MTIME.timestamp()))

        with caplog.at_level("WARNING"):
            assert created_at_via(source) == _SOURCE_MTIME
        assert reason in caplog.text

    def test_a_video_is_not_read_whole(self, tmp_path, created_at_via):
        """A 24 MiB clip is ingested with a heap peak of a few MiB at most."""
        source = tmp_path / "src" / "test-large-clip.mp4"
        _write_mp4(source, version=0, mdat_payload=24 * _MIB, large_mdat=False)

        created_at, peak = _traced_peak(lambda: created_at_via(source))

        assert peak < 4 * _MIB, f"heap peak {peak / _MIB:.1f} MiB"
        assert created_at == _RECORDED_AT

    def test_an_image_is_not_read_whole(self, tmp_path, created_at_via):
        """A ~14 MiB PNG is ingested with a heap peak under a quarter of its size.

        Decoded pixels live in Pillow's own allocations, outside the traced
        heap, so what this measures is the encoded file held in memory.
        """
        source = tmp_path / "src" / "test-large-still.png"
        _write_noise_png(source, side=2200)
        size = os.path.getsize(source)

        created_at, peak = _traced_peak(lambda: created_at_via(source))

        assert peak < size / 4, f"heap peak {peak / _MIB:.1f} MiB of {size / _MIB:.1f}"
        assert created_at == _SOURCE_MTIME


# ---------------------------------------------------------------------------
# 0096 - picture.is_video backfill
# ---------------------------------------------------------------------------

_REVISION_BEFORE_IS_VIDEO = "0095_add_finder_partial_indexes"

# Rows seeded at 0095, all left at the column default, plus the value the
# backfill must produce for each.
_BACKFILL_CASES = {
    "2024/01/15/clip.mp4": 1,
    "2024/01/15/CLIP.MOV": 1,
    "2024/01/15/still.jpg": 0,
}

# Classified True at import despite an extension the backfill does not know.
# The ``is_video = 0`` guard has to leave it alone, or a replay would undo a
# correct runtime classification.
_RUNTIME_CLASSIFIED_PATH = "2024/01/15/odd_extension.bin"


@pytest.fixture(scope="module")
def backfilled_is_video():
    """Seed a pre-0096 database, run the real migration, return the results.

    Alembic is driven exactly as ``tests/test_migrations.py`` drives it for
    0051/0075/0087/0090: step to the revision before the one under test, seed,
    then upgrade to head. Module-scoped because the upgrade is a subprocess and
    every case can be read off one run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_vault.db")
        db_url = f"sqlite:///{db_path}"

        stepped = _run_alembic(
            ["upgrade", _REVISION_BEFORE_IS_VIDEO], db_url, _MIGRATIONS_DIR
        )
        assert stepped.returncode == 0, (
            f"upgrade to {_REVISION_BEFORE_IS_VIDEO} failed:\n"
            f"stdout: {stepped.stdout}\nstderr: {stepped.stderr}"
        )

        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            for row_id, file_path in enumerate(_BACKFILL_CASES, start=1):
                _insert_minimal_row(conn, "picture", id=row_id, file_path=file_path)
            _insert_minimal_row(
                conn,
                "picture",
                id=99,
                file_path=_RUNTIME_CLASSIFIED_PATH,
                is_video=1,
            )
            conn.commit()
            seeded = dict(conn.execute("SELECT file_path, is_video FROM picture"))

        # Nothing may be classified before the migration runs, or the assertions
        # below would pass without it.
        assert seeded == {
            **dict.fromkeys(_BACKFILL_CASES, 0),
            _RUNTIME_CLASSIFIED_PATH: 1,
        }, f"seed state is not pre-migration: {seeded}"

        result = _run_alembic(["upgrade", "head"], db_url, _MIGRATIONS_DIR)
        assert result.returncode == 0, (
            f"alembic upgrade head failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            yield dict(conn.execute("SELECT file_path, is_video FROM picture"))


class TestMigrationBackfill:
    """Test migration 0096 backfills is_video from file_path extension."""

    def test_backfill_mp4_extension(self, backfilled_is_video):
        """Pictures with .mp4 extension get is_video=1 from the backfill."""
        assert backfilled_is_video["2024/01/15/clip.mp4"] == 1

    def test_backfill_uppercase_extension(self, backfilled_is_video):
        """UPPER-CASE extensions are also matched (the UPDATE lower()s the path)."""
        assert backfilled_is_video["2024/01/15/CLIP.MOV"] == 1

    def test_backfill_still_image_unchanged(self, backfilled_is_video):
        """Still-image extensions are not changed by the backfill."""
        assert backfilled_is_video["2024/01/15/still.jpg"] == 0

    def test_backfill_does_not_clobber_a_runtime_classification(
        self, backfilled_is_video
    ):
        """A row already True keeps its value -- the ``is_video = 0`` guard."""
        assert backfilled_is_video[_RUNTIME_CLASSIFIED_PATH] == 1


# ---------------------------------------------------------------------------
# fetch_best_picture_id, via GET /characters/{id}/thumbnail
# ---------------------------------------------------------------------------


def _setup():
    """Start a real Server with a logged-in client (mirrors the thumbnail tests)."""
    temp_dir = tempfile.TemporaryDirectory()
    image_root = os.path.join(temp_dir.name, "images")
    os.makedirs(image_root, exist_ok=True)
    server_config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(server_config_path, "w") as handle:
        handle.write(json.dumps({"port": 8000}))
    server = Server(server_config_path)
    client = TestClient(server.api)
    resp = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert resp.status_code == 200, resp.text
    return temp_dir, client, server


def _import_one_picture(client):
    """Import a real still image so the route has a file it can actually crop."""
    candidates = sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..", "pictures", "*.png"))
    )
    assert candidates, "No test images found in pictures/ directory"
    path = candidates[0]
    with open(path, "rb") as image_file:
        import_resp = upload_pictures_and_wait(
            client, [("file", (os.path.basename(path), image_file, "image/png"))]
        )
    return import_resp["results"][0]["picture_id"]


def _add_rival_picture(server, *, file_path, is_video, score, pixel_sha):
    """Insert a competing Picture row and return its id."""

    def _add(session: Session):
        pic = Picture(
            file_path=file_path,
            format="MP4" if is_video else "PNG",
            width=64,
            height=64,
            size_bytes=1024,
            is_video=is_video,
            score=score,
            pixel_sha=pixel_sha,
        )
        session.add(pic)
        session.commit()
        return pic.id

    return server.vault.db.run_task(_add)


def _score_and_link(server, char_id, entries):
    """Set each picture's score and attach it to the character with a face.

    ``entries`` is an ordered list of ``(picture_id, score)``. The order is the
    face insertion order, which decides which picture the *generation* half of
    the route crops -- deliberately independent of the ordering under test, so
    that a regression in ``fetch_best_picture_id`` shows up as a wrong cache key
    rather than as an unrelated thumbnail failure.
    """

    def _apply(session: Session):
        for face_index, (picture_id, score) in enumerate(entries):
            pic = session.get(Picture, picture_id)
            pic.score = score
            session.add(pic)
            session.add(
                Face(
                    picture_id=picture_id,
                    frame_index=0,
                    # Away from 0 so an asynchronously detected face cannot hit
                    # the (picture_id, frame_index, face_index) unique index.
                    face_index=5 + face_index,
                    character_id=char_id,
                    bbox=[0, 0, 64, 64],
                )
            )
        session.commit()

    server.vault.db.run_task(_apply)


def _cached_best_picture_id(server, char_id):
    """Read back the picture id ``fetch_best_picture_id`` chose.

    The route writes its result into the thumbnail cache's metadata sidecar, so
    this is the handler's own answer, not a re-derivation of it.
    """
    meta_path = os.path.join(
        server.vault.image_root,
        "tmp",
        "face_thumbnails",
        f"character_{char_id}.json",
    )
    assert os.path.isfile(meta_path), f"No thumbnail metadata written at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["picture_id"]


class TestFetchBestPictureId:
    """Test fetch_best_picture_id prefers still images over videos.

    ``fetch_best_picture_id`` is a closure inside ``get_character_field_by_id``
    (``pixlstash/routes/characters.py``) and cannot be imported, so these tests
    call it the only way it can be called: through the route.
    """

    def test_prefers_still_image_over_higher_scored_video(self):
        """A still image wins even when a video scores higher."""
        temp_dir, client, server = _setup()
        try:
            still_id = _import_one_picture(client)
            video_id = _add_rival_picture(
                server,
                file_path="2024/01/15/video.mp4",
                is_video=True,
                score=100,
                pixel_sha="video-sha",
            )
            char_id = client.post("/characters", json={"name": "Mixed"}).json()[
                "character"
            ]["id"]
            _score_and_link(server, char_id, [(still_id, 50), (video_id, 100)])

            resp = client.get(f"/characters/{char_id}/thumbnail")
            assert resp.status_code == 200, resp.text

            assert _cached_best_picture_id(server, char_id) == still_id, (
                "the ordering must put is_video ascending first, so the "
                "lower-scored still image outranks the higher-scored video"
            )
        finally:
            server.close()
            temp_dir.cleanup()
            gc.collect()

    def test_prefers_higher_scored_still_image(self):
        """Between two still images, the higher score wins."""
        temp_dir, client, server = _setup()
        try:
            low_id = _import_one_picture(client)

            def _path_of(session: Session):
                return session.get(Picture, low_id).file_path

            # Same file on disk, so whichever row the generation half picks can
            # always be cropped; only the ordering result is under test.
            high_id = _add_rival_picture(
                server,
                file_path=server.vault.db.run_task(_path_of),
                is_video=False,
                score=100,
                pixel_sha="still-sha",
            )
            char_id = client.post("/characters", json={"name": "Stills"}).json()[
                "character"
            ]["id"]
            _score_and_link(server, char_id, [(low_id, 50), (high_id, 100)])

            resp = client.get(f"/characters/{char_id}/thumbnail")
            assert resp.status_code == 200, resp.text

            assert _cached_best_picture_id(server, char_id) == high_id, (
                "with is_video equal, score must decide"
            )
        finally:
            server.close()
            temp_dir.cleanup()
            gc.collect()
