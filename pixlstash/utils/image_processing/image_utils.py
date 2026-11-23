"""Image loading, thumbnail generation, metadata extraction, and picture creation utilities."""

import cv2
import errno
import functools
import hashlib
import os
import piexif
import piexif.helper
import shutil
import tempfile
import uuid

from datetime import datetime, timezone
from fractions import Fraction
import json

from io import BytesIO
from typing import Callable, Optional

import numpy as np
from PIL import ExifTags, Image, ImageOps

try:
    from PIL.TiffImagePlugin import IFDRational
except Exception:  # pragma: no cover - optional import
    IFDRational = None

from pixlstash.pixl_logging import get_logger
from pixlstash.db_models.picture import Picture
from pixlstash.utils.comfyui_utilities import extract_comfy_workflow_info
from pixlstash.utils.image_processing.orientation import ORIENTATION_TAG
from pixlstash.utils.image_processing.video_utils import VideoUtils

logger = get_logger(__name__)

THUMBNAIL_FORMAT = "WEBP"
THUMBNAIL_EXTENSION = ".webp"
#: Where every thumbnail lives, under the library root. One hidden folder for
#: managed and reference pictures alike, so the owner's own folders hold only
#: the owner's files and a walk of the library never meets a thumbnail (#1164).
THUMBNAIL_DIR_NAME = ".pixlstash-thumbnails"
#: The folder reference-folder thumbnails lived in before THUMBNAIL_DIR_NAME.
_LEGACY_REF_THUMB_DIR = ".ref_thumbs"
THUMBNAIL_QUALITY = 80
THUMBNAIL_WEBP_METHOD = 2
# A thumbnail is ONE aspect-ratio-preserving bitmap of the whole frame. It is
# sized so the SHORT edge is ``THUMBNAIL_SHORT_EDGE`` px, with the long edge
# capped at ``THUMBNAIL_LONG_EDGE_CAP`` px. For aspect ratios beyond
# (cap / short) the short edge shrinks below the target so the long edge hits the
# cap (extreme panoramas). The bitmap is never upscaled beyond the source.
THUMBNAIL_SHORT_EDGE = 384
THUMBNAIL_LONG_EDGE_CAP = 1024


class ImageUtils:
    """Utility methods for image loading, thumbnails, metadata, and picture creation."""

    @staticmethod
    def _coerce_metadata_value(value):
        """Coerce a raw metadata value to a JSON-serialisable Python type."""
        # Best-effort coercion: when a numeric/bytes value cannot be converted the
        # str()/repr() fallback below IS the JSON-serialisable result, not an error
        # path (these swallows are allowlisted in the except-hygiene guardrail).
        if IFDRational is not None and isinstance(value, IFDRational):
            try:
                return float(value)
            except Exception:
                return str(value)
        if isinstance(value, Fraction):
            try:
                return float(value)
            except Exception:
                return str(value)
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            try:
                return float(value)
            except Exception:
                return str(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return repr(value)
        if isinstance(value, (list, tuple)):
            return [ImageUtils._coerce_metadata_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): ImageUtils._coerce_metadata_value(v) for k, v in value.items()
            }
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def extract_embedded_metadata(file_path: str) -> dict:
        """Extract embedded EXIF and PNG metadata from an image file.

        A video answers with the graphs its container's tags carry instead, as
        ``{"comfyui": {"prompt": ..., "workflow": ...}}``.

        Results are cached in memory keyed by (file_path, mtime) so repeated
        calls for the same unchanged file (e.g. on every overlay open) are free.
        """
        if not file_path or not os.path.exists(file_path):
            return {}
        try:
            mtime = os.stat(file_path).st_mtime
        except OSError:
            return {}
        return ImageUtils._extract_embedded_metadata_cached(file_path, mtime)

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _extract_embedded_metadata_cached(file_path: str, mtime: float) -> dict:
        # Pillow cannot open a video; its tags are read off the container. Only
        # the two ComfyUI writes are kept, in a block of their own: a tag is
        # named by whoever wrote the file, and one called `png` or `exif` would
        # otherwise stand where this dict's own sections do.
        if VideoUtils.is_video_file(file_path):
            tags = VideoUtils.extract_embedded_metadata(file_path)
            graphs = {key: tags[key] for key in ("prompt", "workflow") if key in tags}
            return {"comfyui": graphs} if graphs else {}
        metadata = {}
        try:
            with Image.open(file_path) as img:
                info = img.info or {}
                png_text = {}
                for key, value in info.items():
                    if key == "exif":
                        continue
                    png_text[str(key)] = ImageUtils._coerce_metadata_value(value)
                if png_text:
                    metadata["png"] = png_text

                try:
                    exif_data = img.getexif()
                    if exif_data:
                        exif_map = {}
                        for tag_id, value in exif_data.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            exif_map[str(tag_name)] = ImageUtils._coerce_metadata_value(
                                value
                            )
                        comment = ImageUtils._user_comment(exif_data)
                        if comment is not None:
                            exif_map["UserComment"] = comment
                        if exif_map:
                            metadata["exif"] = exif_map
                except Exception as exc:
                    logger.debug("Failed to extract EXIF tags from image: %s", exc)
        except Exception as exc:
            logger.warning("Failed to extract embedded metadata: %s", exc)
        return metadata

    @staticmethod
    def _user_comment(exif_data) -> Optional[str]:
        """The EXIF ``UserComment``, decoded, or ``None`` if there is none.

        It lives in the Exif sub-IFD, which ``getexif()`` does not walk, and it
        is where A1111 and its forks put a JPEG's or WebP's generation data
        (``services/a1111_recipe.py``). The value carries its own 8-byte
        encoding tag, so ``piexif`` decodes it rather than the bytes fallback in
        :meth:`_coerce_metadata_value`, which would garble the UTF-16 one.
        """
        try:
            raw = exif_data.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.UserComment)
        except Exception as exc:
            logger.debug("Failed to read the EXIF sub-IFD: %s", exc)
            return None
        if isinstance(raw, str) or raw is None:
            return raw
        try:
            return piexif.helper.UserComment.load(raw)
        except Exception as exc:
            # A comment with no encoding tag, or one piexif does not know.
            logger.debug("Failed to decode the EXIF UserComment: %s", exc)
            return None

    @staticmethod
    def frame_count(file_path: str) -> Optional[int]:
        """Number of frames in a video or image file, or ``None`` if unreadable.

        A still image is 1; an animated GIF, WebP or APNG is its frame count.
        A video's is the container's own figure, which some codecs estimate.
        Cached by (file_path, mtime) like :meth:`extract_embedded_metadata`.
        """
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            mtime = os.stat(file_path).st_mtime
        except OSError as exc:
            logger.warning("Failed to stat %s for its frame count: %s", file_path, exc)
            return None
        return ImageUtils._frame_count_cached(file_path, mtime)

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _frame_count_cached(file_path: str, mtime: float) -> Optional[int]:
        try:
            if VideoUtils.is_video_file(file_path):
                cap = cv2.VideoCapture(file_path)
                try:
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
            else:
                with Image.open(file_path) as img:
                    count = int(getattr(img, "n_frames", 1))
        except Exception as exc:
            logger.warning("Failed to count frames in %s: %s", file_path, exc)
            return None
        return count if count > 0 else None

    @staticmethod
    def resolve_picture_path(
        image_root: Optional[str], file_path: Optional[str]
    ) -> Optional[str]:
        """
        Resolve a stored picture path to an absolute file path.

        If file_path is already absolute it is returned unchanged.
        If file_path is relative it is joined with image_root.
        """
        if not file_path:
            return None
        if os.path.isabs(file_path):
            return file_path
        if not image_root:
            return file_path
        return os.path.join(image_root, file_path)

    @staticmethod
    def _hashed_thumbnail_name(file_path: str) -> str:
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(file_path))[0]
        return f"{stem}_{path_hash}_thumb{THUMBNAIL_EXTENSION}"

    @staticmethod
    def get_thumbnail_path(
        image_root: Optional[str], file_path: Optional[str]
    ) -> Optional[str]:
        """Return where the thumbnail for *file_path* belongs.

        ``image_root/.pixlstash-thumbnails/<stem>_<sha256(file_path)[:16]>_thumb.webp``
        for every picture. *file_path* is ``Picture.file_path`` in its STORED
        form - relative for a library picture, absolute for a reference-folder
        one - because the hash is of that string; the two forms of one file
        would hash to two names. The key is the path, so a followed move has to
        carry the bitmap (``_carry_thumbnail`` in the scan task and the move
        engine), which it did before this folder existed too.

        A bitmap written before 1.11.1 may still sit at one of
        :meth:`legacy_thumbnail_paths`; :meth:`find_thumbnail` looks there and
        brings it home. Callers that only READ should use that.
        """
        if not file_path:
            return None
        if not image_root:
            # No root to put the folder under (tests, some tools): the old
            # sibling rule is the only answer that names a real place.
            base, _ = os.path.splitext(file_path)
            return f"{base}_thumb{THUMBNAIL_EXTENSION}"
        return os.path.join(
            image_root, THUMBNAIL_DIR_NAME, ImageUtils._hashed_thumbnail_name(file_path)
        )

    @staticmethod
    def legacy_thumbnail_paths(
        image_root: Optional[str], file_path: Optional[str]
    ) -> list[str]:
        """Where the thumbnail for *file_path* was written before #1164.

        A managed picture's sat beside it as ``<stem>_thumb.webp``; a
        reference-folder picture's sat under ``image_root/.ref_thumbs/`` with the
        same hashed name the new folder uses. Order matters to nobody: at most
        one of them ever existed for a given picture.
        """
        if not file_path or not image_root:
            return []
        if os.path.isabs(file_path):
            return [
                os.path.join(
                    image_root,
                    _LEGACY_REF_THUMB_DIR,
                    ImageUtils._hashed_thumbnail_name(file_path),
                )
            ]
        resolved = ImageUtils.resolve_picture_path(image_root, file_path)
        base, _ = os.path.splitext(resolved)
        return [f"{base}_thumb{THUMBNAIL_EXTENSION}"]

    @staticmethod
    def find_thumbnail(
        image_root: Optional[str], file_path: Optional[str]
    ) -> Optional[str]:
        """Return the path of the existing thumbnail for *file_path*, or ``None``.

        Looks at :meth:`get_thumbnail_path` first. A bitmap still at a legacy
        location is MOVED home and the new path returned, so the library
        migrates itself one picture at a time - the startup pass in
        ``maintenance.py`` visits every row, the thumbnail route catches the
        rest - and no thumbnail is ever re-rendered for having moved house. A
        move that fails is logged and the legacy path is served from where it
        is; the next read tries again.
        """
        thumb_path = ImageUtils.get_thumbnail_path(image_root, file_path)
        if not thumb_path:
            return None
        if os.path.isfile(thumb_path):
            return thumb_path
        for legacy in ImageUtils.legacy_thumbnail_paths(image_root, file_path):
            if legacy == thumb_path or not os.path.isfile(legacy):
                continue
            try:
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                os.replace(legacy, thumb_path)
                return thumb_path
            except OSError as exc:
                logger.warning(
                    "Could not move thumbnail %s into %s (%s); serving it from "
                    "where it is.",
                    legacy,
                    THUMBNAIL_DIR_NAME,
                    exc,
                )
                return legacy
        return None

    @staticmethod
    def remove_thumbnail(image_root: Optional[str], file_path: Optional[str]) -> int:
        """Delete the thumbnail for *file_path* wherever it is; count removed.

        Both the current home and every legacy location, so deleting a picture
        indexed before #1164 does not leave its bitmap behind in the owner's
        folder. Failures are logged, not raised: the picture is already gone
        and a stray bitmap is the lesser problem.
        """
        removed = 0
        candidates = [ImageUtils.get_thumbnail_path(image_root, file_path)]
        candidates += ImageUtils.legacy_thumbnail_paths(image_root, file_path)
        for path in dict.fromkeys(p for p in candidates if p):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                logger.warning("Failed to delete thumbnail %s: %s", path, exc)
        return removed

    @staticmethod
    def write_thumbnail_bytes(
        image_root: Optional[str], file_path: Optional[str], thumbnail: bytes
    ) -> Optional[str]:
        """Write thumbnail bytes to disk and return the path, or None on failure.

        The write is atomic: bytes are written to a temp file in the same
        directory and then ``os.replace``d onto the final path. This matters
        because background tasks (e.g. face extraction) overwrite an existing
        thumbnail while the web server may be serving it via ``FileResponse``.
        An in-place ``open(..., "wb")`` truncates first, so a concurrent reader
        can observe a 0-byte or partial file and cache a broken image. A rename
        is atomic on the same filesystem, so readers always see either the old
        or the new complete thumbnail, never a partial one.
        """
        if not thumbnail:
            return None
        thumb_path = ImageUtils.get_thumbnail_path(image_root, file_path)
        if not thumb_path:
            return None
        thumb_dir = os.path.dirname(thumb_path)
        tmp_path = None
        try:
            os.makedirs(thumb_dir, exist_ok=True)
            # delete=False so we can os.replace it; same dir guarantees the
            # rename stays on one filesystem (a cross-device rename is not
            # atomic and would raise).
            with tempfile.NamedTemporaryFile(
                dir=thumb_dir,
                prefix=".thumb-",
                suffix=THUMBNAIL_EXTENSION,
                delete=False,
            ) as handle:
                tmp_path = handle.name
                handle.write(thumbnail)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, thumb_path)
            return thumb_path
        except Exception as exc:
            logger.warning("Failed to write thumbnail %s: %s", thumb_path, exc)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as rm_err:
                    logger.warning(
                        "Failed to remove temp thumbnail %s: %s", tmp_path, rm_err
                    )
            return None

    @staticmethod
    def load_image_or_video_bgr(file_path: str) -> Optional[np.ndarray]:
        """Load an image or the first video frame as a BGR numpy array."""
        if not file_path or not os.path.exists(file_path):
            return None
        if VideoUtils.is_video_file(file_path):
            return VideoUtils.read_first_video_frame_bgr(file_path)

        img = cv2.imread(file_path)
        if img is not None:
            return img
        try:
            with Image.open(file_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB")
                rgb = np.array(pil_img)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.debug(
                "Could not load image %s via PIL fallback (%s); returning None.",
                file_path,
                exc,
            )
            return None

    @staticmethod
    def _encode_thumbnail(pil_img: Image.Image) -> Optional[bytes]:
        """Encode a PIL image as a WebP thumbnail and return the bytes."""
        buf = BytesIO()
        try:
            pil_img.save(
                buf,
                format=THUMBNAIL_FORMAT,
                quality=THUMBNAIL_QUALITY,
                method=THUMBNAIL_WEBP_METHOD,
            )
        except Exception as exc:
            logger.error("Error encoding thumbnail bytes: %s", exc)
            return None
        return buf.getvalue()

    @staticmethod
    def clamp_bbox(bbox, width, height):
        """
        Clamp a bounding box [x_min, y_min, x_max, y_max] to image bounds.

        Returns a new list [x_min, y_min, x_max, y_max] or None if invalid.
        """
        if not bbox or len(bbox) != 4:
            return None
        x_min, y_min, x_max, y_max = [int(round(v)) for v in bbox]
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        if x_max <= x_min or y_max <= y_min:
            return None
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def _exif_created_at(img: Image.Image) -> Optional[datetime]:
        """The capture time an open image's EXIF records, in UTC, or ``None``."""
        try:
            exif_data = img.info.get("exif")
            if exif_data and piexif:
                exif_dict = piexif.load(exif_data)
                date_str = None
                for tag in ("DateTimeOriginal", "DateTime", "DateTimeDigitised"):
                    val = exif_dict["0th"].get(
                        piexif.ImageIFD.__dict__.get(tag)
                    ) or exif_dict["Exif"].get(piexif.ExifIFD.__dict__.get(tag))
                    if val:
                        date_str = val.decode() if isinstance(val, bytes) else val
                        break
                if date_str:

                    def _read_exif_offset() -> Optional[str]:
                        try:
                            exif_ifd = exif_dict.get("Exif") or {}
                            offset_tag_ids = [
                                getattr(piexif.ExifIFD, "OffsetTimeOriginal", None),
                                getattr(piexif.ExifIFD, "OffsetTime", None),
                                getattr(piexif.ExifIFD, "OffsetTimeDigitized", None),
                                36880,
                                36881,
                                36882,
                            ]
                            for tag_id in offset_tag_ids:
                                if tag_id is None:
                                    continue
                                raw_val = exif_ifd.get(tag_id)
                                if not raw_val:
                                    continue
                                text = (
                                    raw_val.decode(errors="replace")
                                    if isinstance(raw_val, bytes)
                                    else str(raw_val)
                                ).strip()
                                if text:
                                    return text
                        except Exception as exc:
                            logger.debug(
                                "Could not read EXIF timezone offset (%s).", exc
                            )
                            return None
                        return None

                    try:
                        offset_text = _read_exif_offset()
                        if offset_text:
                            normalized_offset = offset_text.replace(" ", "")
                            dt = datetime.strptime(
                                f"{date_str} {normalized_offset}",
                                "%Y:%m:%d %H:%M:%S %z",
                            )
                            return dt.astimezone(timezone.utc)

                        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        local_tz = (
                            datetime.now(timezone.utc).astimezone().tzinfo
                            or timezone.utc
                        )
                        return dt.replace(tzinfo=local_tz).astimezone(timezone.utc)
                    except Exception as exc:
                        logger.debug("Failed to parse EXIF datetime: %s", exc)
        except Exception as exc:
            logger.debug("Failed to extract EXIF metadata from image: %s", exc)
        return None

    @staticmethod
    def extract_created_at_from_path(
        file_path: str,
        *,
        mtime_source: Optional[str] = None,
        is_video: Optional[bool] = None,
    ) -> Optional[datetime]:
        """When a media file on disk was made, read from the file itself.

        In order: an image's EXIF capture time; for a video file only, the
        recording time its container's movie header carries; then the
        filesystem mtime. The mtime is the import time for an uploaded file,
        which is why both are tried first. Nothing is read whole: the image
        is opened lazily for its EXIF and the video's boxes are walked by
        seeking.

        Args:
            file_path: The file to read the metadata from.
            mtime_source: The file whose mtime is the fallback, when that is
                not *file_path* - the original a working copy was made from,
                whose mtime is the one that still means something.
            is_video: Whether the file is a video, when the caller has already
                found that out from its content (an import's working copy may
                have no extension). ``None`` decides by the extension.

        Returns:
            A timezone-aware datetime in UTC, or ``None`` if none is found.
        """
        if is_video is None:
            is_video = VideoUtils.is_video_file(file_path)
        created_at = None
        if is_video:
            created_at = VideoUtils.extract_created_at_from_path(file_path)
        else:
            try:
                with Image.open(file_path) as img:
                    created_at = ImageUtils._exif_created_at(img)
            except Exception as exc:
                logger.debug("Could not open %s to read its EXIF: %s", file_path, exc)
        if created_at is not None:
            return created_at

        mtime_path = mtime_source or file_path
        try:
            return datetime.fromtimestamp(os.path.getmtime(mtime_path), tz=timezone.utc)
        except OSError as exc:
            logger.debug("Failed to read file mtime for %s: %s", mtime_path, exc)
            return None

    @staticmethod
    def load_metadata(file_path):
        """
        Efficiently return ``(height, width, channels)`` for an image or video without
        loading full pixel data.
        """
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                mode = img.mode
                if mode == "RGB":
                    c = 3
                elif mode == "L":
                    c = 1
                else:
                    c = len(img.getbands())
                return (h, w, c)
        except Exception as exc:
            logger.debug("PIL failed to read metadata for %s: %s", file_path, exc)
        try:
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                c = frame.shape[2] if len(frame.shape) > 2 else 1
                return (h, w, c)
        except Exception as exc:
            logger.debug("cv2 failed to read metadata for %s: %s", file_path, exc)
        logger.error(f"Failed to read metadata for {file_path}")
        return None

    @staticmethod
    def load_image_or_video(file_path):
        """Load an image or video (first frame) and return an RGB numpy array."""
        try:
            try:
                with Image.open(file_path) as img:
                    img = ImageOps.exif_transpose(img)
                    return np.array(img.convert("RGB"))
            except Exception as exc:
                logger.debug(
                    "PIL failed to load image %s; trying video: %s", file_path, exc
                )
            frame = VideoUtils.read_first_video_frame_bgr(file_path)
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raise ValueError("Could not read image or first frame from video.")
        except Exception as e:
            logger.error(f"Failed to load image at {file_path} for quality worker: {e}")
            return None

    @staticmethod
    def load_image_reduced(file_path: str, max_side: int) -> Optional[np.ndarray]:
        """Load an image at reduced resolution, returning an RGB numpy array.

        For JPEG files, PIL's ``draft()`` instructs the JPEG decoder to use DCT
        subsampling (1/2, 1/4, or 1/8 of the original size), which is much
        faster than decoding at full resolution and then resizing - for a 4K
        JPEG targeting 256 px, this decodes roughly 64× fewer pixels.  For PNG,
        WebP, and other formats ``draft()`` is a no-op and a normal decode +
        cv2 resize is performed instead.

        The returned array has its longest side <= ``max_side``.
        """
        try:
            arr = None
            try:
                with Image.open(file_path) as img:
                    # Must be called before load()/convert(); no-op for non-JPEG.
                    img.draft("RGB", (max_side, max_side))
                    img = ImageOps.exif_transpose(img)
                    arr = np.array(img.convert("RGB"))
            except Exception as exc:
                logger.debug(
                    "PIL failed to load image %s for reduced load; trying video: %s",
                    file_path,
                    exc,
                )
            if arr is None:
                frame = VideoUtils.read_first_video_frame_bgr(file_path)
                if frame is None:
                    return None
                arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = arr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return arr
        except Exception as exc:
            logger.error("Failed to load image at %s (reduced): %s", file_path, exc)
            return None

    @staticmethod
    def thumbnail_bitmap_size(
        src_w: int,
        src_h: int,
        short_edge: int = THUMBNAIL_SHORT_EDGE,
        long_edge_cap: int = THUMBNAIL_LONG_EDGE_CAP,
    ) -> Optional[tuple]:
        """Return the aspect-ratio-preserving bitmap ``(w, h)`` for a source size.

        The short edge targets ``short_edge`` px; the long edge is capped at
        ``long_edge_cap`` px, so aspect ratios beyond ``long_edge_cap /
        short_edge`` get a short edge below the target (extreme panoramas). The
        bitmap is never upscaled beyond the source (so a source smaller than the
        target keeps its native size). Returns ``None`` for a degenerate size.
        """
        if src_w <= 0 or src_h <= 0:
            return None
        short = min(src_w, src_h)
        long_edge = max(src_w, src_h)
        scale = short_edge / float(short)
        if long_edge * scale > long_edge_cap:
            scale = long_edge_cap / float(long_edge)
        scale = min(scale, 1.0)  # never upscale beyond the source
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        return out_w, out_h

    @staticmethod
    def render_thumbnail(
        img,
        face_bboxes: Optional[list] = None,
        short_edge: int = THUMBNAIL_SHORT_EDGE,
        long_edge_cap: int = THUMBNAIL_LONG_EDGE_CAP,
    ) -> Optional[tuple]:
        """Render the whole-frame AR bitmap thumbnail and its square-crop rect.

        Generation is MODE-AGNOSTIC: it always produces ONE
        aspect-ratio-preserving bitmap of the entire frame (no crop baked in),
        sized by :meth:`thumbnail_bitmap_size`, plus the face-weighted square-crop
        rectangle *within* that bitmap for clients that render a square cell.

        Accepts either a PIL Image or a numpy array (OpenCV BGR image).
        ``face_bboxes`` are ``[x1, y1, x2, y2]`` boxes in the SOURCE image's pixel
        space (optional); when omitted the square crop is landscape-centred and
        portrait-top-anchored.

        Returns ``(thumbnail_bytes, bitmap_w, bitmap_h, square_crop)`` where
        ``square_crop`` is ``{"x", "y", "side"}`` in BITMAP pixel space
        (origin top-left, ``side == min(bitmap_w, bitmap_h)``). Returns ``None``
        on failure.
        """
        from pixlstash.utils.image_processing.face_utils import FaceUtils

        try:
            if isinstance(img, Image.Image):
                pil_img = img.copy()
                pil_img = ImageOps.exif_transpose(pil_img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            src_w, src_h = pil_img.width, pil_img.height
            dims = ImageUtils.thumbnail_bitmap_size(
                src_w, src_h, short_edge, long_edge_cap
            )
            if dims is None:
                return None
            bmp_w, bmp_h = dims
            if (bmp_w, bmp_h) != (src_w, src_h):
                pil_img = pil_img.resize((bmp_w, bmp_h), resample=Image.LANCZOS)

            thumbnail_bytes = ImageUtils._encode_thumbnail(pil_img)
            if thumbnail_bytes is None:
                return None

            # Map source-space face boxes into bitmap space (uniform scale).
            faces_bitmap = None
            if face_bboxes:
                scale = bmp_w / float(src_w) if src_w else 1.0
                faces_bitmap = [
                    [v * scale for v in bbox]
                    for bbox in face_bboxes
                    if bbox and len(bbox) == 4
                ]
            crop_x, crop_y, crop_side = FaceUtils.square_crop_rect(
                bmp_w, bmp_h, faces_bitmap
            )
            crop = {"x": crop_x, "y": crop_y, "side": crop_side}
            return thumbnail_bytes, bmp_w, bmp_h, crop
        except Exception as e:
            logger.error(f"Error rendering thumbnail: {e}")
            return None

    @staticmethod
    def generate_thumbnail_bytes(
        img,
        face_bboxes: Optional[list] = None,
    ) -> Optional[bytes]:
        """Return only the encoded whole-frame AR bitmap bytes.

        Thin wrapper over :meth:`render_thumbnail` that discards the dimensions
        and square-crop metadata. Accepts a PIL Image or a numpy (OpenCV BGR)
        array.
        """
        rendered = ImageUtils.render_thumbnail(img, face_bboxes=face_bboxes)
        if rendered is None:
            return None
        return rendered[0]

    @staticmethod
    def _calculate_sha256_digest(
        file_size: int,
        read_chunk,
        source_label: Optional[str] = None,
    ) -> str:
        """Compute a SHA-256 digest by either reading the whole file or sampling it."""
        chunk_size = 8192
        sample_count = 8
        whole_file_threshold = 128 * 1024

        sha256 = hashlib.sha256()
        if file_size <= whole_file_threshold:
            for offset in range(0, file_size, chunk_size):
                chunk = read_chunk(offset, chunk_size)
                if chunk:
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            if source_label:
                logger.debug(f"WHOLE: {source_label} size={file_size} hash={digest}")
            else:
                logger.debug(f"WHOLE: size={file_size} hash={digest}")
            return digest

        offsets = [
            int(i * (file_size - chunk_size) / (sample_count - 1))
            for i in range(sample_count)
        ]
        for offset in offsets:
            chunk = read_chunk(offset, chunk_size)
            if chunk:
                sha256.update(chunk)
        digest = sha256.hexdigest()
        if source_label:
            logger.debug(f"SAMPLED: {source_label} size={file_size} hash={digest}")
        else:
            logger.debug(f"SAMPLED: hash={digest}")
        return digest

    @staticmethod
    def calculate_hash_from_file_path(file_path: str) -> str:
        """Compute a content hash for a file on disk."""
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:

            def _read_chunk(offset, size):
                f.seek(offset)
                return f.read(size)

            return ImageUtils._calculate_sha256_digest(
                file_size=file_size,
                read_chunk=_read_chunk,
                source_label=file_path,
            )

    @staticmethod
    def calculate_full_hash_from_file_path(file_path: str) -> str:
        """Compute SHA-256 over every byte of a file.

        ``calculate_hash_from_file_path`` deliberately samples files larger than
        128 KiB and is suitable as a cheap candidate key, not as final identity.
        Import de-duplication calls this method only after ``(sampled hash,
        size)`` has selected a small candidate set.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def thumbnail_cache_version(
        thumbnail_width: Optional[int],
        thumbnail_height: Optional[int],
        orientation: Optional[int] = None,
        *,
        file_path: Optional[str],
    ) -> str:
        """The ``?v=`` cache-buster for a picture's thumbnail URL.

        Keyed on the stored bitmap's own dimensions, so any regeneration that
        repopulates them changes the URL and the browser refetches instead of
        painting a stale bitmap. ``"0"`` until the picture has been processed.

        **The orientation is part of the key, and it is not optional polish.**
        Thumbnails are served ``Cache-Control: private, max-age=3600,
        must-revalidate``, and a 180° in-place rotate - or a 90° one of a square
        picture - regenerates a bitmap with exactly the dimensions it had before.
        On dimensions alone the URL would be identical and the browser would go
        on painting the pre-rotate bitmap for up to an hour.

        **So is the file.** SQLite hands a deleted picture's id to the next
        import, and a new picture the size of the one deleted gets exactly the
        URL it had: the browser then paints the deleted picture's bitmap in its
        place for up to an hour. Every picture has a file of its own, so a
        digest of its path tells two holders of one id apart. Required, so no
        caller can leave it out.

        Single source of truth on purpose: the batch-thumbnail endpoint and the
        duplicate queue both hand this version to the same frontend cache, and two
        independent copies of the formula would eventually disagree and
        reintroduce the stale-thumbnail bug this version exists to fix.

        Args:
            thumbnail_width: Stored bitmap width, or ``None`` when unprocessed.
            thumbnail_height: Stored bitmap height, or ``None`` when unprocessed.
            orientation: The picture's stored EXIF orientation, 1-8, or ``None``
                when it has not been read yet. ``None`` and ``1`` produce the
                same version: an unrotated picture keeps the URL it has always had,
                so backfilling the mirror does not invalidate every thumbnail in
                the library at once.
            file_path: The picture's stored ``file_path``; ``None`` only for a
                row with no file, which has no thumbnail to serve either.

        The orientation is kept while the dimensions are missing too. A rotate
        NULLs them until the thumbnail sweep regenerates the bitmap, so a second
        turn inside that window, or a turn of a picture not yet processed,
        would otherwise leave the URL at ``"0"`` both sides of the turn.

        Returns:
            ``"<width>x<height>"`` for an unrotated picture,
            ``"<width>x<height>o<orientation>"`` for a rotated one, ``"0"`` (or
            ``"0o<orientation>"``) when either dimension is missing, then
            ``-<file digest>``.
        """
        if not (thumbnail_width and thumbnail_height):
            version = "0"
        else:
            version = f"{thumbnail_width}x{thumbnail_height}"
        if orientation and int(orientation) != 1:
            version = f"{version}o{int(orientation)}"
        if not file_path:
            return version
        digest = hashlib.blake2s(file_path.encode("utf-8"), digest_size=4).hexdigest()
        return f"{version}-{digest}"

    @staticmethod
    def calculate_hash_from_bytes(image_bytes: bytes) -> str:
        """Compute a content hash for raw image bytes."""
        file_size = len(image_bytes)

        def _read_chunk(offset, size):
            return image_bytes[offset : offset + size]

        return ImageUtils._calculate_sha256_digest(
            file_size=file_size,
            read_chunk=_read_chunk,
        )

    @staticmethod
    def calculate_full_hash_from_bytes(image_bytes: bytes) -> str:
        """Compute SHA-256 over every byte in an in-memory file."""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def create_picture_from_file(
        image_root_path: str,
        source_file_path: str,
        picture_uuid: Optional[str] = None,
        pixel_sha: Optional[str] = None,
        subfolder: Optional[str] = None,
        *,
        output_dir: Optional[str] = None,
        reference_folder_id: Optional[int] = None,
        record_source_name: bool = True,
    ) -> Picture:
        """Create a Picture from a file on disk, copying it into the library.

        The source is copied in chunks, never read whole into memory, and
        never moved: it belongs to whoever handed it over (a staging folder, a
        watch folder, a ComfyUI run's fetch folder), who removes it when they
        are done. Everything recorded is derived from the copy, so it
        describes exactly the bytes that are published - except the mtime
        fallback for ``created_at``, which is read from the source, because
        the copy's own mtime is now.

        Args:
            subfolder: Relative folder under ``image_root_path`` to write into,
                for a library whose root has a v1.11 layout. See
                :meth:`create_picture_from_bytes`.
            output_dir: As for :meth:`create_picture_from_bytes`.
            reference_folder_id: As for :meth:`create_picture_from_bytes`.
            record_source_name: Whether the source's file name is recorded as
                the picture's ``original_file_name``. False for a file named
                only for the fetch that brought it, such as a ComfyUI output:
                that name is not the picture's own.
        """
        if not os.path.exists(source_file_path):
            raise ValueError(f"Source file path does not exist: {source_file_path}")
        return ImageUtils._publish_picture(
            image_root_path,
            lambda part_path: shutil.copyfile(source_file_path, part_path),
            source_label=source_file_path,
            part_suffix=os.path.splitext(source_file_path)[1].lower(),
            mtime_source=source_file_path,
            picture_uuid=picture_uuid,
            pixel_sha=pixel_sha,
            created_at=None,
            original_file_name=(
                os.path.basename(source_file_path) if record_source_name else None
            ),
            output_dir=output_dir,
            reference_folder_id=reference_folder_id,
            subfolder=subfolder,
        )

    @staticmethod
    def create_picture_from_bytes(
        image_root_path: str,
        image_bytes: bytes,
        picture_uuid: Optional[str] = None,
        pixel_sha: Optional[str] = None,
        created_at: Optional[str] = None,
        original_file_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        reference_folder_id: Optional[int] = None,
        subfolder: Optional[str] = None,
    ) -> Picture:
        """Create a Picture from raw bytes, deriving metadata and saving the file.

        The bytes are written to an in-progress file and described from there,
        through the same steps :meth:`create_picture_from_file` takes.

        Args:
            image_root_path: The vault's image root directory. Always used for
                thumbnail storage even when ``output_dir`` is set.
            image_bytes: Raw image data to save.
            picture_uuid: Optional filename (with extension) for the saved file.
            pixel_sha: Pre-computed pixel hash; computed from bytes if omitted.
            created_at: Optional creation timestamp.
            original_file_name: Original filename to store on the Picture record.
            output_dir: When set, save the image file into this directory instead
                of ``image_root_path``.  The Picture's ``file_path`` is stored as
                the full absolute path, which is what marks a reference-folder
                picture everywhere else.
            reference_folder_id: When set, assigned to the Picture so that the
                reference-folder scan recognises the record as already indexed.
            subfolder: A ``/``-separated relative folder under
                *image_root_path* to write into - where the v1.11 layout says a
                new picture belongs (``services/layout_move_service.render``).
                The stored ``file_path`` stays RELATIVE, so the picture is still
                a library picture and its thumbnail is still a sibling file;
                only ``output_dir`` makes a path absolute, and that means a
                reference folder. Ignored when *output_dir* is given.
        """

        def _write_part(part_path: str) -> None:
            with open(part_path, "xb") as handle:
                handle.write(image_bytes)

        inferred_ext = ImageUtils._extension_of(picture_uuid)
        return ImageUtils._publish_picture(
            image_root_path,
            _write_part,
            source_label=original_file_name or picture_uuid or "uploaded bytes",
            part_suffix=f".{inferred_ext}" if inferred_ext else "",
            mtime_source=None,
            picture_uuid=picture_uuid,
            pixel_sha=pixel_sha,
            created_at=created_at,
            original_file_name=original_file_name,
            output_dir=output_dir,
            reference_folder_id=reference_folder_id,
            subfolder=subfolder,
        )

    @staticmethod
    def _extension_of(picture_uuid: Optional[str]) -> str:
        """The lower-case extension, without its dot, of a picture's file name."""
        if not picture_uuid:
            return ""
        return os.path.splitext(str(picture_uuid))[1].lower().lstrip(".")

    @staticmethod
    def _publish_picture(
        image_root_path: str,
        write_part: Callable[[str], None],
        *,
        source_label: str,
        part_suffix: str,
        mtime_source: Optional[str],
        picture_uuid: Optional[str],
        pixel_sha: Optional[str],
        created_at: Optional[datetime],
        original_file_name: Optional[str],
        output_dir: Optional[str],
        reference_folder_id: Optional[int],
        subfolder: Optional[str],
    ) -> Picture:
        """Write a new picture's file, describe it, and publish it at its name.

        *write_part* writes the content to the path it is handed: a hidden
        in-progress file in the destination folder, hidden so no folder walk
        indexes it. Hash, format, dimensions, orientation, thumbnail and date
        are all derived from that file; only then is it renamed onto its final
        name with ``os.replace``, which is atomic on one filesystem. A reader
        never sees a partial file at that name, and a full disk or a process
        that dies mid-write leaves none there; on any failure the process sees,
        the in-progress file is removed. Nothing is fsynced, so a power loss
        straight after the rename can still leave the name holding fewer bytes.

        The parameters are those of :meth:`create_picture_from_bytes`, plus:

        Args:
            source_label: Where the content came from, for log lines.
            part_suffix: Extension (with its dot) for the in-progress file, so
                OpenCV picks the demuxer for the video's own container. Empty
                for content with no name, which OpenCV is handed as ``.mp4``.
            mtime_source: The file whose mtime is the ``created_at`` fallback,
                when that is not the in-progress file.
        """
        inferred_ext = ImageUtils._extension_of(picture_uuid)
        if output_dir:
            dest_dir = output_dir
        elif subfolder:
            dest_dir = os.path.join(image_root_path, *subfolder.split("/"))
        else:
            dest_dir = image_root_path
        os.makedirs(dest_dir, exist_ok=True)
        part_path = os.path.join(
            dest_dir, f".{uuid.uuid4().hex}.part{part_suffix or '.mp4'}"
        )

        full_path = None
        published = False
        try:
            write_part(part_path)
            if not pixel_sha:
                pixel_sha = ImageUtils.calculate_hash_from_file_path(part_path)

            img_format = None
            width = height = None
            orientation = None
            thumbnail_bytes = None
            # Thumbnail column values (AR-bitmap dims + faceless square crop).
            # Faces are not known at import; ``FaceExtractionTask`` refines the
            # square crop once faces are detected.
            thumb_cols: dict = {}
            is_video = False
            try:
                with Image.open(part_path) as img:
                    img_format = img.format or "PNG"
                    width, height = img.size
                    # Read here, from the image that is already open, rather
                    # than left to `MissingOrientationFinder`. That finder
                    # exists to fill rows that predate the column; a NEW
                    # picture whose orientation is NULL for an indeterminate
                    # window is a value anything reading it right after import
                    # races - and the operation log records it as a facet, so a
                    # rotate landing in that window would snapshot `None` as
                    # the prior state and its undo would have nothing to write
                    # back.
                    try:
                        raw_orientation = int(
                            (img.getexif() or {}).get(ORIENTATION_TAG, 1)
                        )
                    except (TypeError, ValueError):
                        raw_orientation = 1
                    orientation = raw_orientation if 1 <= raw_orientation <= 8 else 1
                    rendered = ImageUtils.render_thumbnail(img)
                    if rendered is not None:
                        thumbnail_bytes, bmp_w, bmp_h, crop = rendered
                        thumb_cols = {
                            "thumbnail_width": bmp_w,
                            "thumbnail_height": bmp_h,
                            "square_crop_x": crop["x"],
                            "square_crop_y": crop["y"],
                            "square_crop_side": crop["side"],
                        }
            except Exception as exc:
                logger.debug(
                    "%s is not an image Pillow reads, so it is read as a video: %s",
                    source_label,
                    exc,
                )
                is_video = True

            if not is_video and (
                thumbnail_bytes is None or width is None or height is None
            ):
                raise ValueError(f"Failed to generate thumbnail for {source_label}")

            if is_video:
                cap = cv2.VideoCapture(part_path)
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(
                            "Could not read first frame from video %s for thumbnail.",
                            source_label,
                        )
                        raise ValueError("Failed to read first frame from video")
                    height, width = frame.shape[:2]
                    rendered = ImageUtils.render_thumbnail(frame)
                    if rendered is None:
                        raise ValueError("Failed to generate thumbnail for video")
                    thumbnail_bytes, bmp_w, bmp_h, crop = rendered
                    thumb_cols = {
                        "thumbnail_width": bmp_w,
                        "thumbnail_height": bmp_h,
                        "square_crop_x": crop["x"],
                        "square_crop_y": crop["y"],
                        "square_crop_side": crop["side"],
                    }
                finally:
                    cap.release()
                img_format = inferred_ext.upper() if inferred_ext else "MP4"

            if not created_at:
                created_at = ImageUtils.extract_created_at_from_path(
                    part_path, mtime_source=mtime_source, is_video=is_video
                )

            if not picture_uuid:
                picture_uuid = str(uuid.uuid4()) + f".{img_format.lower()}"
            file_name = os.path.basename(picture_uuid)
            full_path = os.path.join(dest_dir, file_name)
            if output_dir:
                # Saved into the caller-specified directory (e.g. a reference
                # folder). Store the absolute path: that is what marks a
                # reference-folder picture for every reader.
                picture_file_path: str = full_path
            elif subfolder:
                # Placement on write, v1.11 Phase 4b. Relative, so the picture
                # stays a library picture; the layout decided the folder and the
                # caller has already checked that the root has one.
                picture_file_path = "/".join([*subfolder.split("/"), file_name])
            else:
                picture_file_path = file_name
            size_bytes = os.path.getsize(part_path)

            ImageUtils._move_into_place(part_path, full_path)
            published = True
        finally:
            if not published:
                ImageUtils._discard_part(part_path, source_label, full_path or dest_dir)

        if thumbnail_bytes:
            saved_thumb = ImageUtils.write_thumbnail_bytes(
                image_root_path, picture_file_path, thumbnail_bytes
            )
            if not saved_thumb:
                logger.warning("Failed to persist thumbnail for %s", file_name)

        if not created_at:
            created_at = datetime.now(timezone.utc)

        # Extract ComfyUI generation metadata once at import time so that
        # text-embedding generation can read it from the DB without re-opening
        # the file on every embedding run.
        comfyui_positive_prompt = None
        # Always set to at least "[]" for non-video pictures so that
        # comfyui_models IS NULL can serve as the "not yet checked" sentinel.
        comfyui_models_json = None
        comfyui_loras_json = None
        if not is_video:
            models = []
            loras = []
            try:
                embedded_metadata = ImageUtils.extract_embedded_metadata(full_path)
                workflow_info = extract_comfy_workflow_info(embedded_metadata)
                if workflow_info:
                    comfyui_positive_prompt = (
                        workflow_info.get("positive_prompt") or None
                    )
                    models = workflow_info.get("models") or []
                    loras = workflow_info.get("loras") or []
            except Exception as exc:
                logger.debug("ComfyUI extraction failed for %s: %s", full_path, exc)
            comfyui_models_json = json.dumps(models)
            comfyui_loras_json = json.dumps(loras)

        return Picture(
            file_path=picture_file_path,
            format=img_format,
            width=width,
            height=height,
            orientation=orientation,
            size_bytes=size_bytes,
            created_at=created_at,
            pixel_sha=pixel_sha,
            original_file_name=original_file_name,
            reference_folder_id=reference_folder_id,
            comfyui_positive_prompt=comfyui_positive_prompt,
            comfyui_models=comfyui_models_json,
            comfyui_loras=comfyui_loras_json,
            is_video=is_video,
            **thumb_cols,
        )

    @staticmethod
    def _move_into_place(part_path: str, full_path: str) -> None:
        """Rename a finished in-progress file onto its final name.

        A file already at that name when it is checked is never overwritten,
        and never taken as the new picture unless it is a regular file holding
        exactly the same bytes (same size and SHA-256); then it is kept and the
        in-progress copy dropped. A link there is never followed. Two imports
        racing to the same name are not guarded against: one that takes the
        name between the check and the rename is replaced by the other.

        Raises:
            FileExistsError: Something else already has that name.
        """
        if os.path.lexists(full_path):
            same = (
                not os.path.islink(full_path)
                and os.path.isfile(full_path)
                and os.path.getsize(full_path) == os.path.getsize(part_path)
            )
            if same:
                same = ImageUtils.calculate_full_hash_from_file_path(
                    full_path
                ) == ImageUtils.calculate_full_hash_from_file_path(part_path)
            if not same:
                raise FileExistsError(
                    errno.EEXIST, "A different file already has this name", full_path
                )
            logger.debug("%s already holds these bytes; keeping it.", full_path)
            os.remove(part_path)
            return
        os.replace(part_path, full_path)

    @staticmethod
    def _discard_part(part_path: str, source_label: str, destination: str) -> None:
        """Remove the in-progress file of an import that did not complete."""
        logger.warning(
            "Import of %s into %s did not complete; removing its in-progress copy.",
            source_label,
            destination,
        )
        try:
            os.remove(part_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.error(
                "Could not remove the in-progress copy %s of %s: %s",
                part_path,
                source_label,
                exc,
            )

    @staticmethod
    def cosine_similarity(a: bytes, b: bytes) -> float:
        """Compute cosine similarity between two embedding byte-strings."""
        try:
            if a is None or b is None:
                return 0.0
            arr_a = (
                np.frombuffer(a, dtype=np.float32)
                if isinstance(a, bytes)
                else np.array(a, dtype=np.float32)
            )
            arr_b = (
                np.frombuffer(b, dtype=np.float32)
                if isinstance(b, bytes)
                else np.array(b, dtype=np.float32)
            )
            if arr_a.shape != arr_b.shape or arr_a.size == 0:
                return 0.0
            dot = np.dot(arr_a, arr_b)
            norm_a = np.linalg.norm(arr_a)
            norm_b = np.linalg.norm(arr_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f"cosine_similarity error: {e}")
            return 0.0

    @staticmethod
    def load_image_bgr_reduced(file_path: str, max_side: int) -> tuple:
        """Load a still image at reduced resolution, returning ``(bgr_array, inv_scale)``.

        ``inv_scale`` converts coordinates in the returned image back to the
        original (exif-corrected) image space: multiply any bbox or pixel
        coordinate by ``inv_scale`` to get the original-space value.

        Uses PIL ``draft()`` for JPEG files so the JPEG decoder subsamples at
        the DCT level - much faster than full decode + resize for large JPEGs.
        For HEIF, PNG, WebP the image is decoded at full resolution and then
        resized with ``cv2.INTER_AREA``.

        Returns ``(None, 1.0)`` on failure.
        """
        try:
            with Image.open(file_path) as img:
                # Read stored dimensions from the file header (lazy, no pixel decode).
                stored_w, stored_h = img.size
                # Determine the logical (exif-corrected) original dimensions cheaply.
                # Orientations 5-8 rotate 90° or 270°, which swaps width and height.
                try:
                    orientation = (img.getexif() or {}).get(274, 1)
                except Exception:
                    orientation = 1
                if orientation in (5, 6, 7, 8):
                    orig_w, _ = stored_h, stored_w
                else:
                    orig_w, _ = stored_w, stored_h

                # draft() tells the JPEG decoder to use DCT subsampling - a no-op
                # for other formats.  Must be called before load()/convert().
                img.draft("RGB", (max_side, max_side))
                img = ImageOps.exif_transpose(img)
                arr = np.array(img.convert("RGB"))

            h, w = arr.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                new_w = max(1, int(round(w * s)))
                new_h = max(1, int(round(h * s)))
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = arr.shape[:2]

            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            # inv_scale: multiply loaded-space coords by this to recover original-space
            # coords.  Both axes have the same factor because the resize is proportional.
            inv_scale = orig_w / w if w > 0 else 1.0
            return bgr, inv_scale
        except Exception as exc:
            logger.debug("load_image_bgr_reduced failed for %s: %s", file_path, exc)
            return None, 1.0
