"""Video frame extraction and metadata utilities."""

import cv2
import json
import os
import struct
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from PIL import Image

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)

# MP4/MOV timestamps count seconds since midnight, 1 January 1904 (UTC).
_MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)

# Supported video file extensions (lowercase).
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

# The most of a file `extract_embedded_metadata` will read: the `moov` box or
# the `Tags` element. Either is kilobytes, a few megabytes with a large
# workflow in it. It is read once and walked through views, never copied.
_MAX_TAG_BYTES = 64 * 1024 * 1024

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
# Matroska element ids, as written (the length marker is part of an id).
_MKV_SEGMENT = 0x18538067
_MKV_TAGS = 0x1254C367
_MKV_CLUSTER = 0x1F43B675
_MKV_TAG = 0x7373
_MKV_SIMPLE_TAG = 0x67C8
_MKV_TAG_NAME = 0x45A3
_MKV_TAG_STRING = 0x4487


def _text(data) -> str:
    return str(data, "utf-8", "replace")


def _boxes(data):
    """Yield ``(type, payload)`` for each ISO-BMFF box laid end to end in *data*."""
    pos = 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > len(data):
                return
            size, header = struct.unpack_from(">Q", data, pos + 8)[0], 16
        elif size == 0:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            return
        yield kind, data[pos + header : pos + size]
        pos += size


def _box_spans(handle, start: int, end: int):
    """Yield ``(type, payload start, box end)`` for each box between *start* and *end*.

    Reads only the box headers, seeking from one to the next by its declared
    size (a 64-bit ``largesize`` when the size field is 1, the rest of the
    range when it is 0), so a payload is never read to be stepped over.

    Raises:
        ValueError: A box header is cut short by the end of the file.
    """
    pos = start
    while pos + 8 <= end:
        handle.seek(pos)
        header = handle.read(16)
        if len(header) < 8:
            raise ValueError(f"the box header at byte {pos} is cut short")
        size, kind = struct.unpack_from(">I4s", header)
        skip = 8
        if size == 1:
            if len(header) < 16:
                raise ValueError(f"the 64-bit box size at byte {pos} is cut short")
            size, skip = struct.unpack_from(">Q", header, 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < skip:
            return
        yield kind, pos + skip, pos + size
        pos += size


def _mp4_tags(handle, file_size: int) -> dict:
    """The text tags of an MP4 or MOV file, read off its ``moov`` box.

    Top-level boxes are stepped over by their size, so the media data is never
    read whichever side of it ``moov`` is on.
    """
    moov = b""
    for kind, body, box_end in _box_spans(handle, 0, file_size):
        if kind != b"moov":
            continue
        if box_end - body > _MAX_TAG_BYTES:
            raise ValueError(f"its moov box is {box_end - body} bytes")
        handle.seek(body)
        moov = memoryview(handle.read(box_end - body))
        break

    tags: dict = {}
    top = dict(_boxes(moov))
    user = dict(_boxes(top.get(b"udta", b"")))
    # QuickTime's own comment atom: [length:2][language:2][text]. The length
    # is 16 bits and wraps on a longer text, which a workflow usually is, so
    # the box's own size is what bounds it.
    comment = user.get(b"\xa9cmt", b"")
    if len(comment) > 4:
        tags["comment"] = _text(comment[4:])
    for meta in (user.get(b"meta"), top.get(b"meta")):
        if not meta:
            continue
        # A full box in MP4 (version and flags first), a plain one in QuickTime.
        inner = dict(_boxes(meta if meta[4:8] == b"hdlr" else meta[4:]))
        # `keys` names the items `ilst` then numbers from 1; without it the
        # items are iTunes atoms, of which the comment is the one read.
        names = [name for _, name in _boxes(inner.get(b"keys", b"")[8:])]
        for kind, item in _boxes(inner.get(b"ilst", b"")):
            data = dict(_boxes(item)).get(b"data")
            if data is None or data[:4] != b"\x00\x00\x00\x01":
                continue  # Not UTF-8 text: cover art, a number.
            index = int.from_bytes(kind, "big")
            if kind == b"\xa9cmt":
                name = b"comment"
            elif 1 <= index <= len(names):
                name = names[index - 1]
            else:
                continue
            tags.setdefault(_text(name).lower(), _text(data[8:]))
    return tags


def _mvhd_creation_seconds(handle, file_size: int) -> Optional[int]:
    """The ``mvhd`` creation time of an MP4/MOV, in seconds since 1904.

    Walks the top-level boxes to ``moov`` and its children to ``mvhd``, then
    reads only that header's version, flags and creation time. ``None`` when
    the file has no movie header or it records no time (0 means unset).

    Raises:
        ValueError: A box header is cut short, or the movie header runs past
            its ``moov`` box or the file, or is shorter than its fields.
    """
    for kind, body, box_end in _box_spans(handle, 0, file_size):
        if kind != b"moov":
            continue
        for child, child_body, child_end in _box_spans(handle, body, box_end):
            if child != b"mvhd":
                continue
            if child_end > min(box_end, file_size):
                raise ValueError(
                    f"the movie header at byte {child_body} runs to byte "
                    f"{child_end}, past the end of its moov box or of the file"
                )
            handle.seek(child_body)
            # [version:1][flags:3][creation_time: 4 bytes in v0, 8 in v1]
            header = handle.read(min(12, child_end - child_body))
            if header and header[0] not in (0, 1):
                return None
            if len(header) < (12 if header[:1] == b"\x01" else 8):
                raise ValueError(
                    f"the movie header at byte {child_body} is shorter than its fields"
                )
            if header[0] == 0:
                seconds = struct.unpack_from(">I", header, 4)[0]
            else:
                seconds = struct.unpack_from(">Q", header, 4)[0]
            return seconds or None
        return None
    return None


def _ebml_header(data, pos: int) -> tuple:
    """Read an EBML element's id and size at *pos*.

    Returns ``(id, size, payload position)``; the size is ``None`` when the
    element was written without one (a live stream's).
    """
    values = []
    for _ in range(2):
        if pos >= len(data) or not data[pos]:
            raise ValueError("a truncated or malformed EBML element")
        length = 9 - data[pos].bit_length()
        if pos + length > len(data):
            raise ValueError("a truncated EBML element")
        values.append((int.from_bytes(data[pos : pos + length], "big"), length))
        pos += length
    (ident, _), (size, length) = values
    mask = (1 << (7 * length)) - 1
    size &= mask
    return ident, None if size == mask else size, pos


def _ebml(data):
    """Yield ``(id, payload)`` for each EBML element laid end to end in *data*."""
    pos = 0
    while pos < len(data):
        ident, size, pos = _ebml_header(data, pos)
        end = len(data) if size is None else pos + size
        yield ident, data[pos:end]
        pos = end


def _matroska_tags(handle, file_size: int) -> dict:
    """The text tags of a WebM or Matroska file, read off its ``Tags`` elements.

    The segment's children are stepped over by their size, so no cluster is
    read. ffmpeg upper-cases the names; they come back lower-cased.
    """
    tags: dict = {}
    pos = 0
    while pos < file_size:
        handle.seek(pos)
        ident, size, used = _ebml_header(handle.read(12), 0)
        pos += used
        if ident == _MKV_SEGMENT:
            continue  # Its children are the elements this loop is after.
        if size is None:
            break  # Unsized: nothing past it can be stepped to.
        if ident == _MKV_CLUSTER and tags:
            break  # ffmpeg writes its tags ahead of the media; they are read.
        if ident == _MKV_TAGS:
            if size > _MAX_TAG_BYTES:
                raise ValueError(f"its Tags element is {size} bytes")
            handle.seek(pos)
            for tag_id, tag in _ebml(memoryview(handle.read(size))):
                if tag_id != _MKV_TAG:
                    continue
                for simple_id, simple in _ebml(tag):
                    if simple_id != _MKV_SIMPLE_TAG:
                        continue
                    fields = dict(_ebml(simple))
                    if _MKV_TAG_NAME in fields and _MKV_TAG_STRING in fields:
                        # An EBML string may be padded with NULs.
                        name, value = (
                            _text(fields[key]).rstrip("\x00")
                            for key in (_MKV_TAG_NAME, _MKV_TAG_STRING)
                        )
                        tags.setdefault(name.lower(), value)
        pos += size
    return tags


class VideoUtils:
    """Utility methods for video file handling."""

    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Return True if the file is a supported video format."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in VIDEO_EXTENSIONS

    @staticmethod
    def is_animated_gif(file_path: str) -> bool:
        """Return True for a .gif with more than one frame.

        Only the frame-sampling sites consult this; a GIF stays a picture
        everywhere else (thumbnails, export, ``is_video``).
        """
        if os.path.splitext(file_path)[1].lower() != ".gif":
            return False
        try:
            with Image.open(file_path) as img:
                # is_animated only looks for a second frame; n_frames walks
                # the whole file.
                return bool(getattr(img, "is_animated", False))
        except Exception as exc:
            logger.debug(
                "Could not open %s to check for animation, treating it as a still: %s",
                file_path,
                exc,
            )
            return False

    @staticmethod
    def is_multiframe_file(file_path: str) -> bool:
        """Return True for a video or an animated GIF."""
        return VideoUtils.is_video_file(file_path) or VideoUtils.is_animated_gif(
            file_path
        )

    @staticmethod
    def extract_embedded_metadata(file_path: str) -> dict:
        """The text tags a video's container carries, keyed by lower-cased name.

        This is where ComfyUI's video savers write the graph they ran, as its
        image savers write PNG text chunks: a ``prompt`` and a ``workflow`` tag
        each (SaveVideo, SaveWEBM), or both inside one JSON ``comment``
        (VideoHelperSuite), which is unpacked into its keys so every writer
        reads the same. MP4/MOV and WebM/Matroska are read; any other
        container, and a file that cannot be parsed, answers ``{}``.
        """
        try:
            with open(file_path, "rb") as handle:
                file_size = os.fstat(handle.fileno()).st_size
                if handle.read(4) == _EBML_MAGIC:
                    tags = _matroska_tags(handle, file_size)
                else:
                    tags = _mp4_tags(handle, file_size)
        except (OSError, struct.error, ValueError) as exc:
            logger.warning(
                "Could not read the container tags of %s, so it is read as "
                "carrying none: %s",
                file_path,
                exc,
            )
            return {}
        comment = tags.get("comment", "")
        if comment.lstrip().startswith("{"):
            try:
                envelope = json.loads(comment)
            except (ValueError, RecursionError):
                # Somebody's own comment that happens to open with a brace.
                envelope = None
            if isinstance(envelope, dict):
                del tags["comment"]
                for key, value in envelope.items():
                    tags.setdefault(str(key).lower(), value)
        return tags

    @staticmethod
    def extract_created_at_from_path(file_path: str) -> Optional[datetime]:
        """The recording time an MP4/MOV's movie header (``mvhd``) carries.

        The box structure is walked by seeking, so the media data is stepped
        over whichever side of the ``moov`` box it is on and at most a few
        header bytes are read. WebM/Matroska and AVI carry no ``mvhd`` and
        answer ``None`` without being walked.

        Args:
            file_path: Path of the video file.

        Returns:
            The time as a timezone-aware UTC datetime, or ``None`` when the
            container records none or cannot be read.
        """
        try:
            with open(file_path, "rb") as handle:
                file_size = os.fstat(handle.fileno()).st_size
                magic = handle.read(4)
                if magic in (_EBML_MAGIC, b"RIFF"):
                    return None
                seconds = _mvhd_creation_seconds(handle, file_size)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not read the movie header of %s, so its date falls back "
                "to the file's own: %s",
                file_path,
                exc,
            )
            return None
        if seconds is None:
            return None
        try:
            return _MP4_EPOCH + timedelta(seconds=seconds)
        except OverflowError:
            logger.warning(
                "The movie header of %s records an impossible time (%d s), so "
                "its date falls back to the file's own.",
                file_path,
                seconds,
            )
            return None

    @staticmethod
    def read_first_video_frame_bgr(file_path: str) -> Optional[np.ndarray]:
        """Read the first frame of a video file and return it as a BGR numpy array."""
        cap = cv2.VideoCapture(file_path)
        try:
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
        finally:
            cap.release()
        if ret and frame is not None:
            return frame
        return None

    @staticmethod
    def _read_indices(cap, indices) -> List[Image.Image]:
        """Read *indices* from an open capture whose frame count is known.

        Indices outside ``[0, frame count)`` are skipped. A frame the backend
        cannot seek to or decode is left out rather than failing the rest.
        """
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in sorted(set(indices)):
            if not 0 <= idx < frame_count:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames

    @staticmethod
    def _read_first_frame(cap) -> List[Image.Image]:
        """One sequential read from an open capture, with no seek.

        For a container whose backend reports a frame count of 0 (it does not
        support the property, as some backends do for WebM and MKV): seeking is
        meaningless there, but the first frame can still be read.
        """
        ret, frame = cap.read()
        if not ret or frame is None:
            return []
        return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]

    @staticmethod
    def extract_video_frames(file_path: str, frame_indices) -> List[Image.Image]:
        """
        Extract frames from a video file and return them as PIL Images.

        When the backend reports no frame count, only the first frame is read,
        whatever *frame_indices* asks for.

        Args:
            file_path: Path to video file.
            frame_indices: Frame indices to extract (0-based).

        Returns:
            List of PIL.Image objects; empty when the file cannot be opened.
        """
        cap = cv2.VideoCapture(file_path)
        try:
            if not cap.isOpened():
                return []
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
                return VideoUtils._read_first_frame(cap)
            return VideoUtils._read_indices(cap, frame_indices)
        finally:
            cap.release()

    @staticmethod
    def extract_representative_video_frames(
        file_path: str, count: int = 3
    ) -> List[Image.Image]:
        """
        Extract ``count`` evenly spaced frames from a video (e.g. start, middle, end).

        The file is opened once. When the backend reports no frame count, the
        first frame is read sequentially instead.

        Args:
            file_path: Path to video file.
            count: Number of frames to extract (evenly spaced).

        Returns:
            List of PIL.Image objects; empty when the file cannot be opened.
        """
        cap = cv2.VideoCapture(file_path)
        try:
            if not cap.isOpened():
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return VideoUtils._read_first_frame(cap)
            if count == 1:
                indices = [0]
            else:
                step = (total_frames - 1) / (count - 1)
                indices = sorted(set(int(i * step) for i in range(count)))
            return VideoUtils._read_indices(cap, indices)
        finally:
            cap.release()
