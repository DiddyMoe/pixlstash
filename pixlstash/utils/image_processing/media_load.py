"""Load a picture's frames, and say why when they cannot be loaded.

A background stage that cannot load a picture has to decide whether to ask for
it again, and both wrong answers are expensive. Marking a good file
undecodable hides it from every work finder for the rest of the session
(:class:`~pixlstash.utils.unprocessable_image_registry.UnprocessableImageRegistry`),
and retrying a broken file on every sweep keeps the pipeline busy with work
that can never finish. The right answer depends on *why* the load failed, so
the loader here returns a classified :class:`LoadResult` instead of ``None``:

- ``UNDECODABLE`` / ``NO_FRAMES``: the file's bytes can be read and the
  decoder rejects them. The file stays broken until it is rewritten, so these
  are the only permanent failures.
- ``IO_ERROR``: the operating system refused (no file descriptor left, an I/O
  error, a stale network handle, a missing file). A later attempt may work.
  Whether the file's whole location is gone (an unmounted drive) is the
  caller's question, answered by ``scrapheap_service.file_location_is_unreachable``.
- ``RESOURCE``: the process ran out of memory.
- ``UNCLASSIFIED``: an exception type this module does not recognise. It is
  not guessed to be permanent; the caller retries it and logs it as an error.

The discriminator for ``OSError`` is ``errno``, and it is exact: a filesystem or
resource failure carries one (``EMFILE`` 24, ``EIO`` 5, ``ESTALE`` 116), while
Pillow's decode failures (``UnidentifiedImageError``, ``"image file is
truncated"``, a broken data stream) leave it ``None``.
"""

import os
import struct
from dataclasses import dataclass, field
from enum import Enum

import cv2
from PIL import Image

from pixlstash.utils.image_processing.video_utils import VideoUtils


class LoadFailure(str, Enum):
    """Why a picture's frames could not be loaded."""

    UNDECODABLE = "undecodable"
    NO_FRAMES = "no_frames"
    IO_ERROR = "io_error"
    RESOURCE = "resource"
    UNCLASSIFIED = "unclassified"

    @property
    def permanent(self) -> bool:
        """True when retrying the same bytes cannot succeed."""
        return self in (LoadFailure.UNDECODABLE, LoadFailure.NO_FRAMES)


@dataclass
class LoadResult:
    """The frames of one picture, or why there are none.

    Attributes:
        frames: RGB PIL images; one for a still, up to the requested count for
            a video or an animated GIF. Empty when ``failure`` is set.
        failure: ``None`` on success.
        error: What went wrong, for the caller's log line. The file is named by
            its base name only.
    """

    frames: list = field(default_factory=list)
    failure: LoadFailure | None = None
    error: str = ""


# What a decoder raises for bytes it cannot make sense of. An ``OSError`` only
# counts here once `is_transient_load_error` has ruled out an errno.
_DECODER_ERRORS = (
    OSError,
    ValueError,
    SyntaxError,
    EOFError,
    struct.error,
    Image.DecompressionBombError,
    cv2.error,
)


def is_transient_load_error(exc: BaseException) -> bool:
    """True when *exc* means the machine failed, not that the file is corrupt."""
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, OSError) and exc.errno is not None


def file_read_error(file_path: str) -> OSError | None:
    """The error from reading the first byte of *file_path*, or ``None`` if it reads.

    The last check before calling a file undecodable, for decoders that do not
    say why they failed: ``extract_representative_video_frames`` returns ``[]``
    whenever ``cv2.VideoCapture`` cannot open the file, which includes running
    out of file descriptors.
    """
    try:
        with open(file_path, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        return exc
    return None


def _describe(file_path: str, exc: BaseException) -> str:
    """``exc`` as text, with the file named by its base name only."""
    message = str(exc).replace(file_path, os.path.basename(file_path))
    return f"{type(exc).__name__}: {message}"


def _classify(file_path: str, exc: BaseException) -> LoadResult:
    if isinstance(exc, MemoryError):
        return LoadResult(failure=LoadFailure.RESOURCE, error=_describe(file_path, exc))
    if is_transient_load_error(exc):
        return LoadResult(failure=LoadFailure.IO_ERROR, error=_describe(file_path, exc))
    if isinstance(exc, _DECODER_ERRORS):
        read_error = file_read_error(file_path)
        if read_error is None:
            return LoadResult(
                failure=LoadFailure.UNDECODABLE, error=_describe(file_path, exc)
            )
        return LoadResult(
            failure=LoadFailure.IO_ERROR, error=_describe(file_path, read_error)
        )
    return LoadResult(failure=LoadFailure.UNCLASSIFIED, error=_describe(file_path, exc))


def load_pil_classified(file_path: str, *, multiframe_count: int) -> LoadResult:
    """Decode *file_path* to RGB frames, classifying any failure.

    A video or an animated GIF yields up to *multiframe_count* evenly spaced
    frames; anything else yields one. A still is decoded as stored, with no
    EXIF orientation applied: what the image-embedding stage has always
    embedded, so stored embeddings stay comparable.
    """
    try:
        if VideoUtils.is_multiframe_file(file_path):
            frames = [
                frame.convert("RGB")
                for frame in VideoUtils.extract_representative_video_frames(
                    file_path, count=multiframe_count
                )
            ]
        else:
            with Image.open(file_path) as img:
                frames = [img.convert("RGB")]
    except Exception as exc:
        return _classify(file_path, exc)
    if frames:
        return LoadResult(frames=frames)
    read_error = file_read_error(file_path)
    if read_error is not None:
        return _classify(file_path, read_error)
    return LoadResult(failure=LoadFailure.NO_FRAMES, error="no frame could be decoded")
