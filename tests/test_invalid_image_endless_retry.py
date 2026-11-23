"""Regression test for GitHub issue #585 - "Invalid Images break tasks".

A corrupt/undecodable image used to be retried forever: a ``Missing*Finder``
re-selects any picture whose target column is unset, the task cannot decode the
file, and nothing durably marks it done (``ImageEmbeddingTask`` writes an *empty*
embedding blob that ``fetch_work`` still treats as missing), so the same picture
is picked up on every sweep.

The fix is an in-memory, process-lifetime ``UnprocessableImageRegistry`` (owned by
the database, reachable from both task and finder threads via ``self._db``):

* a decode failure in an image-reading task calls ``mark_unprocessable`` (the
  shared choke point), pinning the picture to the file's current ``(mtime, size)``;
* ``BaseTaskFinder._filter_and_claim`` skips suppressed pictures for every finder,
  and ``count_remaining`` / ``fetch_work`` exclude them so progress can reach 0;
* suppression is keyed on the file signature, so rewriting the file (the "still
  being written" / repaired case) lifts it and the picture is retried.

These tests assert the FIXED behavior. They exercise the real decode-failure path
(``ImageEmbeddingTask``'s preload and ``_process_preloaded`` without CLIP) and
the real suppression path (``_filter_and_claim`` / ``count_remaining``).
ML-free: no CLIP / insightface / tagger inference is run.

Only a file whose bytes can be read and do not decode is suppressed. A load that
fails because the machine failed (no file descriptor left, an I/O error) is
retried, and a file whose whole location is gone is held until it is back.
"""

import errno
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlmodel import Session

from pixlstash.db_models import Picture
from pixlstash.task_runner import TaskCancelledError
from pixlstash.tasks.base_task import TaskStatus
from pixlstash.tasks import image_embedding_task as embedding_module
from pixlstash.tasks.base_task_finder import BaseTaskFinder
from pixlstash.tasks.image_embedding_task import ImageEmbeddingTask
from pixlstash.tasks.missing_image_embedding_finder import MissingImageEmbeddingFinder
from pixlstash.tasks.missing_thumbnail_finder import MissingThumbnailFinder
from pixlstash.tasks.quality_task import QualityTask
from pixlstash.tasks.task_outcome import Outcome, Reason, RetryHold, outcomes_payload
from pixlstash.utils.image_processing import media_load
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.unprocessable_image_registry import UnprocessableImageRegistry
from pixlstash.vault import Vault

# Bytes that exist on disk as a ``.jpg`` but are not a decodable image. cv2 and
# the PIL fallback in ``load_image_or_video_bgr`` both refuse them.
_CORRUPT_JPEG_BYTES = b"this is not a real JPEG, just some garbage bytes \xff\xd8\x00"


def _write_corrupt_jpeg(path, payload: bytes = _CORRUPT_JPEG_BYTES) -> str:
    with open(path, "wb") as handle:
        handle.write(payload)
    return str(path)


def _seed_picture(vault, file_path: str) -> int:
    """Insert one Picture row (embedding unset) and return its id."""
    now = datetime.now(timezone.utc)

    def seed(session: Session):
        picture = Picture(
            file_path=file_path,
            format="jpg",
            width=64,
            height=64,
            deleted=False,
            imported_at=now,
            image_embedding=None,
            aesthetic_score=3.0,
            created_at=now,
        )
        session.add(picture)
        session.commit()
        return picture.id

    return vault.db.run_task(seed)


def _fetch_work_ids(vault, suppressed_ids=None) -> set:
    work = vault.db.run_task(
        lambda session: ImageEmbeddingTask.fetch_work(
            session,
            aesthetic_disabled=True,
            suppressed_ids=suppressed_ids,
        )
    )
    return {row.id for row in work}


def _count_remaining(vault, suppressed_ids=None) -> int:
    return int(
        vault.db.run_task(
            lambda session: ImageEmbeddingTask.count_remaining(
                session,
                aesthetic_disabled=True,
                suppressed_ids=suppressed_ids,
            )
        )
        or 0
    )


def _run_failed_decode(vault, pic_id: int, file_path: str) -> list:
    """Drive ImageEmbeddingTask's real preload and settle step for one picture.

    The preload loads and classifies the file; the settle step hands a failure
    to the registry - without ever touching CLIP. Returns what was preloaded.
    """
    task = ImageEmbeddingTask(
        database=vault.db,
        clip_workflow=None,
        batch=[(pic_id, file_path)],
    )
    task._preload_images_task()
    preloaded = task._wait_for_preload()
    task._process_preloaded(preloaded)
    return preloaded


class _MetadatalessPic:
    """Stands in for a Picture whose format/width/height are unset."""

    def __init__(self, pid: int, file_path: str):
        self.id = pid
        self.file_path = file_path
        self.format = None
        self.width = None
        self.height = None


def _refuse_reads(monkeypatch, errno_value: int = errno.EIO) -> None:
    """Make the media-load module's read probe fail as an I/O error would."""

    def refuse(file_path, *args, **kwargs):
        raise OSError(errno_value, os.strerror(errno_value), file_path)

    # Shadows the builtin for media_load only; nothing else's open() changes.
    monkeypatch.setattr(media_load, "open", refuse, raising=False)


def test_585_corrupt_image_is_suppressed_after_failed_decode(tmp_path):
    """#585 (fixed): a corrupt image is NOT re-selected after a failed decode."""
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, corrupt_path)

        # Root cause is real: the file genuinely fails to open.
        assert os.path.exists(corrupt_path)
        assert ImageUtils.load_image_or_video_bgr(corrupt_path) is None

        # Before the fix runs: the finder's raw query selects it (still missing).
        assert pic_id in _fetch_work_ids(vault)
        assert _count_remaining(vault) == 1

        # Drive one real, failed processing attempt (marks it unprocessable).
        _run_failed_decode(vault, pic_id, corrupt_path)

        registry = vault.db.unprocessable_images
        assert registry.is_suppressed(pic_id)
        suppressed = registry.active_suppressed_ids()
        assert suppressed == {pic_id}

        # The finder now excludes it at the shared claim choke point...
        finder = MissingImageEmbeddingFinder(
            database=vault.db, engine_getter=lambda: None
        )
        rows = vault.db.run_task(
            lambda session: ImageEmbeddingTask.fetch_work(
                session, aesthetic_disabled=True
            )
        )
        selected = finder._filter_and_claim(rows, batch_limit=10)
        assert pic_id not in {getattr(row, "id", None) for row in selected}, (
            "#585: suppressed corrupt image must not be claimed for a new task"
        )

        # ...and both the candidate query and progress count drop it to zero.
        assert _fetch_work_ids(vault, suppressed_ids=suppressed) == set()
        assert _count_remaining(vault, suppressed_ids=suppressed) == 0


def test_585_suppression_lifts_when_file_is_modified(tmp_path):
    """#585 "until modified": rewriting the file makes the picture selectable again."""
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, corrupt_path)
        _run_failed_decode(vault, pic_id, corrupt_path)

        registry = vault.db.unprocessable_images
        assert registry.active_suppressed_ids() == {pic_id}
        assert _count_remaining(vault, suppressed_ids={pic_id}) == 0

        # The file is modified (e.g. an interrupted write finally completes). A
        # different size guarantees the signature moves even on coarse-mtime
        # filesystems, so suppression must lift regardless of timer resolution.
        _write_corrupt_jpeg(
            corrupt_path, payload=_CORRUPT_JPEG_BYTES + b" now a different length"
        )

        assert not registry.is_suppressed(pic_id)
        suppressed = registry.active_suppressed_ids()
        assert suppressed == set()

        # It is selectable again: the finder claims it and it counts as remaining.
        finder = MissingImageEmbeddingFinder(
            database=vault.db, engine_getter=lambda: None
        )
        rows = vault.db.run_task(
            lambda session: ImageEmbeddingTask.fetch_work(
                session, aesthetic_disabled=True
            )
        )
        selected = finder._filter_and_claim(rows, batch_limit=10)
        assert pic_id in {getattr(row, "id", None) for row in selected}, (
            "#585: a modified file must be retried, not permanently suppressed"
        )
        assert pic_id in _fetch_work_ids(vault, suppressed_ids=suppressed)
        assert _count_remaining(vault, suppressed_ids=suppressed) == 1


def test_585_filter_and_claim_never_reads_file_path(tmp_path):
    """#585 regression: ``_filter_and_claim`` must not touch ``picture.file_path``.

    The CI failure was a ``DetachedInstanceError``: some finders' candidate
    queries return detached ``Picture`` objects whose ``file_path`` is deferred,
    and the suppression check used to read it while claiming, crashing the
    WorkPlanner thread. Suppression now looks the picture up by id against the
    registry's own stored path, so ``file_path`` is never accessed here.
    """

    class _FilePathExplodes:
        """Stands in for a detached Picture whose file_path would lazy-load."""

        def __init__(self, pid: int):
            self.id = pid

        @property
        def file_path(self):
            raise AssertionError(
                "_filter_and_claim must not access picture.file_path (#585)"
            )

    with Vault(image_root=str(tmp_path)) as vault:
        finder = MissingImageEmbeddingFinder(
            database=vault.db, engine_getter=lambda: None
        )
        candidates = [_FilePathExplodes(1), _FilePathExplodes(2)]
        selected = finder._filter_and_claim(candidates, batch_limit=10)
        assert [c.id for c in selected] == [1, 2]


def test_585_thumbnail_task_suppresses_corrupt_image(tmp_path):
    """#585 follow-up: ThumbnailGenerationTask must mark undecodable sources.

    Reported against v1.8.0: the original fix marked decode failures only in
    ImageEmbeddingTask, so a corrupt picture whose thumbnail columns were NULL
    (e.g. after the v1.8.0 thumbnail-regeneration reset) was re-selected by
    MissingThumbnailFinder on every sweep forever.
    """
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, corrupt_path)

        # The finder selects it (thumbnail_width is NULL) and builds a task,
        # and the "Upgrading thumbnails" progress count sees it as remaining.
        finder = MissingThumbnailFinder(vault.db)
        task = finder.find_task()
        assert task is not None
        assert task.params["picture_ids"] == [pic_id]
        assert vault.db.run_task(vault._count_missing_thumbnails) == 1

        # The task fails to decode the source: no columns are written...
        result = task._run_task()
        assert result == {"changed_count": 0}

        # ...but the picture is now suppressed, so after the claim is released
        # the finder no longer re-selects it - the old endless-retry loop -
        # and the progress count reaches 0 so the bar can complete.
        assert vault.db.unprocessable_images.is_suppressed(pic_id)
        finder.on_task_complete(task, None)
        assert finder.find_task() is None
        assert vault.db.run_task(vault._count_missing_thumbnails) == 0


def test_585_quality_metadata_backfill_marks_instead_of_raising(tmp_path):
    """#585 follow-up: an undecodable file must not abort quality backfill.

    ``_backfill_missing_picture_metadata`` used to raise on the first corrupt
    picture, killing the whole batch (good pictures included) and leaving every
    column unset, so the finder re-selected the same batch on every sweep.
    """
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")
    valid_path = str(tmp_path / "valid.jpg")
    PILImage.new("RGB", (8, 8), "red").save(valid_path)

    corrupt_pic = _MetadatalessPic(1, corrupt_path)
    valid_pic = _MetadatalessPic(2, valid_path)

    with Vault(image_root=str(tmp_path)) as vault:
        task = QualityTask(database=vault.db, pictures=[corrupt_pic, valid_pic])

        # Corrupt picture first: the loop must survive it and reach the valid one.
        task._backfill_missing_picture_metadata([corrupt_pic, valid_pic])

        assert vault.db.unprocessable_images.is_suppressed(corrupt_pic.id)
        assert not vault.db.unprocessable_images.is_suppressed(valid_pic.id)
        assert (valid_pic.width, valid_pic.height) == (8, 8)
        assert valid_pic.format == "JPG"


def test_a_truncated_png_is_suppressed_until_it_is_rewritten(tmp_path):
    """A PNG cut off mid-copy opens, then fails while decoding: that is corrupt.

    Pillow reads the header lazily, so this failure comes from ``convert`` and
    not from ``open``, with an ``OSError`` that carries no errno.
    """
    pixels = np.random.default_rng(585).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    whole = tmp_path / "whole.png"
    PILImage.fromarray(pixels).save(whole)
    payload = whole.read_bytes()
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(payload[: len(payload) // 2])

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, str(truncated))
        preloaded = _run_failed_decode(vault, pic_id, str(truncated))

        assert [entry[5] for entry in preloaded] == [media_load.LoadFailure.UNDECODABLE]
        registry = vault.db.unprocessable_images
        assert registry.is_suppressed(pic_id)

        # The copy completes: the picture is work again, and now loads.
        truncated.write_bytes(payload)
        assert not registry.is_suppressed(pic_id)
        assert pic_id in _fetch_work_ids(
            vault, suppressed_ids=registry.active_suppressed_ids()
        )
        loaded = media_load.load_pil_classified(str(truncated), multiframe_count=3)
        assert loaded.failure is None and len(loaded.frames) == 1


def test_a_load_that_runs_out_of_file_descriptors_is_retried(tmp_path, monkeypatch):
    """``EMFILE`` says the process is out of descriptors, not that the file is bad.

    Marking it would hide a good picture for the rest of the session.
    """
    path = tmp_path / "fine.png"
    PILImage.new("RGB", (8, 8), "red").save(path)

    def no_descriptor_left(file_path, *args, **kwargs):
        raise OSError(errno.EMFILE, os.strerror(errno.EMFILE), file_path)

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, str(path))
        with monkeypatch.context() as patch:
            patch.setattr(media_load.Image, "open", no_descriptor_left)
            preloaded = _run_failed_decode(vault, pic_id, str(path))

        assert [entry[5] for entry in preloaded] == [media_load.LoadFailure.IO_ERROR]
        registry = vault.db.unprocessable_images
        assert not registry.is_suppressed(pic_id)
        assert pic_id in _fetch_work_ids(
            vault, suppressed_ids=registry.active_suppressed_ids()
        )


def _thumbnail_attempt(vault, pic_id: int, file_path: str) -> None:
    finder = MissingThumbnailFinder(vault.db)
    task = finder.find_task()
    assert task is not None and task.params["picture_ids"] == [pic_id]
    assert task._run_task() == {"changed_count": 0}
    finder.on_task_complete(task, None)


def _quality_attempt(vault, pic_id: int, file_path: str) -> None:
    pic = _MetadatalessPic(pic_id, file_path)
    QualityTask(database=vault.db, pictures=[pic])._backfill_missing_picture_metadata(
        [pic]
    )


@pytest.mark.parametrize("readable", [True, False], ids=["reads", "read-fails"])
def test_a_video_that_yields_no_frame_is_marked_only_when_it_reads(
    tmp_path, monkeypatch, readable
):
    """OpenCV says nothing about why a capture would not open.

    A file it cannot open while the bytes read is one this build cannot decode.
    The same silence follows a process out of file descriptors, so the read
    probe decides.
    """
    clip = _write_corrupt_jpeg(tmp_path / "clip.mp4")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, clip)
        if not readable:
            _refuse_reads(monkeypatch, errno.EMFILE)
        preloaded = _run_failed_decode(vault, pic_id, clip)

        expected = (
            media_load.LoadFailure.NO_FRAMES
            if readable
            else media_load.LoadFailure.IO_ERROR
        )
        assert [entry[5] for entry in preloaded] == [expected]
        assert vault.db.unprocessable_images.is_suppressed(pic_id) is readable


@pytest.mark.parametrize(
    "attempt",
    [_run_failed_decode, _quality_attempt],
    ids=["embedding", "quality"],
)
def test_a_source_that_cannot_be_read_is_not_marked(tmp_path, monkeypatch, attempt):
    """A decoder failure on a file that cannot be read says nothing about the file.

    The same garbage file is marked by every stage when it reads (the #585
    tests above). When reading it fails with an I/O error, it is not marked.
    The quality loader answers only ``None``, so the read decides.
    """
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, corrupt_path)
        _refuse_reads(monkeypatch)

        attempt(vault, pic_id, corrupt_path)

        assert not vault.db.unprocessable_images.is_suppressed(pic_id)


def test_a_thumbnail_source_that_cannot_be_read_is_not_selected_again(
    tmp_path, monkeypatch
):
    """MissingThumbnailFinder has no retry hold, so a source it cannot load is marked.

    A file that cannot be read (no permission, a bad sector) fails the same way
    on every sweep. Left unmarked, it would be selected again on every sweep
    (#585).
    """
    corrupt_path = _write_corrupt_jpeg(tmp_path / "corrupt.jpg")

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, corrupt_path)
        _refuse_reads(monkeypatch, errno.EACCES)

        _thumbnail_attempt(vault, pic_id, corrupt_path)

        assert MissingThumbnailFinder(vault.db).find_task() is None


def test_a_picture_on_an_unmounted_drive_is_held_until_the_drive_is_back(tmp_path):
    """No retry can succeed while the location is gone, and none is needed after.

    The hold lifts by itself when the directory returns, so the picture's
    pending work is kept: nothing about the file is wrong.
    """
    drive = tmp_path / "drive"
    drive.mkdir()
    path = drive / "pic.png"
    PILImage.new("RGB", (8, 8), "red").save(path)

    with Vault(image_root=str(tmp_path)) as vault:
        pic_id = _seed_picture(vault, str(path))
        registry = vault.db.unprocessable_images
        registry.ACTIVE_CACHE_TTL_S = 0.0

        shutil.move(drive, tmp_path / "unplugged")
        preloaded = _run_failed_decode(vault, pic_id, str(path))

        assert [entry[5] for entry in preloaded] == [media_load.LoadFailure.IO_ERROR]
        assert registry.is_suppressed(pic_id)
        assert pic_id not in _fetch_work_ids(
            vault, suppressed_ids=registry.active_suppressed_ids()
        )

        shutil.move(tmp_path / "unplugged", drive)
        assert not registry.is_suppressed(pic_id)
        assert pic_id in _fetch_work_ids(
            vault, suppressed_ids=registry.active_suppressed_ids()
        )


def test_the_suppressed_set_is_built_once_per_window_and_drops_on_a_mark(tmp_path):
    """#1206 item 9b: four finders rebuilt this set on every planner sweep.

    Each rebuild re-stats every entry, and an entry whose volume has gone away
    costs a mount timeout per stat - on the planner thread. The answer is
    therefore reused for ``ACTIVE_CACHE_TTL_S``, and any write to the map drops
    the cache so a newly marked picture is never served from a set built before
    it.
    """
    registry = UnprocessableImageRegistry()
    first = str(tmp_path / "one.jpg")
    second = str(tmp_path / "two.jpg")
    _write_corrupt_jpeg(tmp_path / "one.jpg")
    _write_corrupt_jpeg(tmp_path / "two.jpg")
    assert registry.mark_unprocessable(1, first, reason="test fixture")

    stats: list[str] = []
    original = registry._stat_signature

    def counting_stat(file_path: str):
        stats.append(file_path)
        return original(file_path)

    registry._stat_signature = staticmethod(counting_stat)

    assert registry.active_suppressed_ids() == {1}
    assert stats == [first]
    # Within the window, the second sweep re-stats nothing.
    assert registry.active_suppressed_ids() == {1}
    assert stats == [first]

    # A new mark must be visible at once, cache or no cache.
    assert registry.mark_unprocessable(2, second, reason="test fixture")
    assert registry.active_suppressed_ids() == {1, 2}

    # And closing the window re-validates.
    stats.clear()
    registry.ACTIVE_CACHE_TTL_S = 0.0
    assert registry.active_suppressed_ids() == {1, 2}
    assert sorted(stats) == sorted([first, second])


# --- Retryable failures: held back, never marked --------------------------
#
# A picture that failed for a reason that may pass is held back by the finder
# on a capped, jittered exponential backoff, instead of being selected again on
# the next sweep. These tests share one vault and drive the finder's clock;
# each asserts on the pictures it seeds.

_FINDER_LOGGER = MissingImageEmbeddingFinder.__module__


@pytest.fixture(scope="module")
def hold_vault(tmp_path_factory):
    root = tmp_path_factory.mktemp("retry-holds")
    with Vault(image_root=str(root)) as vault:
        yield vault


class _Clip:
    """Just enough of a CLIP workflow for the finder to size a batch."""

    def __init__(self, batch_size: int = 64):
        self._batch_size = batch_size

    def suggested_batch_size(self):
        return self._batch_size


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def make_finder(hold_vault, clock, monkeypatch):
    """A finder whose holds run on *clock* with no jitter (each wait is d/2)."""
    # Only the missing-embedding arm, so the scan order is the row order.
    monkeypatch.setattr(ImageEmbeddingTask, "_aesthetic_disabled", True)

    def make(batch_size: int = 64):
        engine = SimpleNamespace(clip_embedding_workflow=_Clip(batch_size))
        return MissingImageEmbeddingFinder(
            hold_vault.db,
            lambda: engine,
            retry_hold=RetryHold(clock=clock, rng=lambda: 0.0),
            backend_hold=RetryHold(cap_s=60.0, clock=clock, rng=lambda: 0.0),
        )

    return make


def _new_picture(vault, name: str) -> int:
    return _seed_picture(vault, os.path.join(vault.db.image_root, name))


def _found(finder) -> set:
    """The picture ids of the task the finder builds now.

    Its claims are released by the base class alone, so looking settles nothing.
    """
    task = finder.find_task()
    if task is None:
        return set()
    BaseTaskFinder.on_task_complete(finder, task, None)
    return set(task.params["picture_ids"])


def _complete(finder, vault, outcomes, *, error=None, result=None):
    """Settle a finished task over the pictures *outcomes* names."""
    task = ImageEmbeddingTask(
        database=vault.db,
        clip_workflow=None,
        batch=[(pid, f"test-{pid}.png") for pid, *_ in outcomes],
    )
    task.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
    task.result = outcomes_payload(outcomes) if result is None else result
    finder.on_task_complete(task, error)


def test_a_retryable_failure_is_held_back_longer_each_time_and_never_marked(
    hold_vault, make_finder, clock
):
    finder = make_finder()
    pid = _new_picture(hold_vault, "test-flaky.png")
    assert pid in _found(finder)

    failed = [(pid, Outcome.FAILED_RETRYABLE, Reason.IO_TRANSIENT)]
    _complete(finder, hold_vault, failed)
    assert pid not in _found(finder), "held for 2 s"
    clock.now += 2.01
    assert pid in _found(finder)

    _complete(finder, hold_vault, failed)
    clock.now += 3.99
    assert pid not in _found(finder), "held for 4 s the second time"
    clock.now += 0.02
    assert pid in _found(finder)

    # A success starts the backoff over.
    _complete(finder, hold_vault, [(pid, Outcome.SUCCEEDED, None)])
    _complete(finder, hold_vault, failed)
    clock.now += 2.01
    assert pid in _found(finder), "held for 2 s again after a success"
    assert not hold_vault.db.unprocessable_images.is_suppressed(pid)


def test_held_pictures_do_not_fill_the_candidate_window(hold_vault, make_finder):
    """Held ids are left out by the query itself, before its LIMIT.

    Filtered out after the query instead, a window of held pictures would come
    back empty on every sweep while the work behind it waited.
    """
    finder = make_finder(batch_size=1)  # a window of IMAGE_EMBEDDING_MAX_INFLIGHT
    for i in range(4):
        _new_picture(hold_vault, f"test-crowd-{i}.png")
    waiting = _fetch_work_ids(hold_vault)
    for held in waiting:
        finder._retry_hold.hold(held)
    fresh = _new_picture(hold_vault, "test-behind-the-crowd.png")

    assert _found(finder) == {fresh}


def test_an_unavailable_backend_holds_the_finder_not_the_pictures(
    hold_vault, make_finder, clock
):
    """No batch is built while CLIP is down, and its pictures are not held.

    Several in-flight tasks fail together when the backend goes; only the
    first lengthens the wait. The first picture that embeds lifts it.
    """
    finder = make_finder()
    first = _new_picture(hold_vault, "test-backend-a.png")
    second = _new_picture(hold_vault, "test-backend-b.png")
    other = _new_picture(hold_vault, "test-backend-ok.png")
    down = Reason.BACKEND_UNAVAILABLE
    _complete(finder, hold_vault, [(first, Outcome.FAILED_RETRYABLE, down)])
    _complete(finder, hold_vault, [(second, Outcome.FAILED_RETRYABLE, down)])

    assert finder.find_task() is None, "no batch while the backend is held"
    clock.now += 2.01
    assert {first, second} <= _found(finder), "one 2 s hold for both tasks"

    _complete(
        finder,
        hold_vault,
        [(pid, Outcome.FAILED_RETRYABLE, down) for pid in (first, second)],
    )
    clock.now += 3.99
    assert finder.find_task() is None, "the next probe waits 4 s"

    _complete(finder, hold_vault, [(other, Outcome.SUCCEEDED, None)])
    assert {first, second} <= _found(finder), "a success lifts the hold at once"


@pytest.mark.parametrize(
    "how",
    ["runner-cancelled", "cancelled-before-start", "deferred"],
)
def test_a_picture_that_was_not_attempted_is_not_held(
    hold_vault, make_finder, caplog, how
):
    finder = make_finder()
    pid = _new_picture(hold_vault, f"test-not-attempted-{how}.png")
    task = finder.find_task()
    assert pid in task.params["picture_ids"]
    error = None
    if how == "runner-cancelled":
        error = TaskCancelledError("test-stopped")
        task.status = TaskStatus.CANCELLED
    elif how == "cancelled-before-start":
        task.status = TaskStatus.CANCELLED
    else:
        task.status = TaskStatus.COMPLETED
        task.result = outcomes_payload(
            (p, Outcome.DEFERRED, Reason.CANCELLED) for p in task.params["picture_ids"]
        )

    with caplog.at_level(logging.WARNING, logger=_FINDER_LOGGER):
        finder.on_task_complete(task, error)

    assert pid in _found(finder)
    assert [r.getMessage() for r in caplog.records if r.name == _FINDER_LOGGER] == []


@pytest.mark.parametrize(
    "how, logged",
    [
        ("raised", []),
        ("no-outcomes", ["returned no outcomes"]),
        ("not-an-outcome", ["is not one", "returned no outcomes"]),
    ],
)
def test_a_task_that_failed_or_did_not_say_holds_its_pictures(
    hold_vault, make_finder, clock, caplog, how, logged
):
    """Not known to have succeeded: held, so a bug cannot become a loop."""
    finder = make_finder()
    pid = _new_picture(hold_vault, f"test-unsaid-{how}.png")
    outcomes = [(pid, Outcome.SUCCEEDED, None)]

    with caplog.at_level(logging.WARNING, logger=_FINDER_LOGGER):
        if how == "raised":
            _complete(finder, hold_vault, outcomes, error=RuntimeError("test-boom"))
        elif how == "no-outcomes":
            _complete(finder, hold_vault, outcomes, result={"changed": []})
        else:
            _complete(
                finder,
                hold_vault,
                outcomes,
                result={"outcomes": [(pid, "test-bogus", None)]},
            )

    assert pid not in _found(finder)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert [part for part in logged if any(part in m for m in errors)] == logged
    clock.now += 2.01
    assert pid in _found(finder), "held, not suppressed"


def test_a_picture_held_while_the_finder_was_claiming_is_released(
    hold_vault, make_finder, monkeypatch
):
    """A task that fails between the query and the claim cannot slip through."""
    finder = make_finder()
    pid = _new_picture(hold_vault, "test-raced.png")
    claim = finder._filter_and_claim

    def claim_then_fail(candidates, batch_limit, pixel_budget=None):
        selected = claim(candidates, batch_limit, pixel_budget=pixel_budget)
        finder._retry_hold.hold(pid)  # that task settles now
        return selected

    monkeypatch.setattr(finder, "_filter_and_claim", claim_then_fail)
    task = finder.find_task()

    assert task is None or pid not in task.params["picture_ids"]
    assert pid not in finder._claimed_picture_ids


def test_no_new_batch_while_a_decode_outlives_its_task(
    hold_vault, make_finder, monkeypatch, caplog
):
    """A task that stopped waiting for a stuck decode leaves the decode running.

    Until it returns the finder builds no batch, so decodes cannot pile up
    behind one that does not return. The task's picture that was never
    started is not started after the task completed.
    """
    finder = make_finder()
    root = hold_vault.db.image_root
    names = ("test-stuck-decode.png", "test-queued-behind-it.png")
    for name in names:
        PILImage.new("RGB", (32, 24), "plum").save(os.path.join(root, name))
    stuck, behind = (_new_picture(hold_vault, name) for name in names)
    started, release = threading.Event(), threading.Event()
    seen = []
    load = embedding_module.load_pil_classified

    def blocking_load(path, **kwargs):
        seen.append(os.path.basename(path))
        if path.endswith(names[0]):
            started.set()
            release.wait(10)
        return load(path, **kwargs)

    monkeypatch.setattr(embedding_module, "load_pil_classified", blocking_load)
    task = ImageEmbeddingTask(
        database=hold_vault.db,
        clip_workflow=None,
        batch=[
            (pid, os.path.join(root, name)) for pid, name in zip((stuck, behind), names)
        ],
    )
    task._PRELOAD_WORKERS = 1  # the second picture waits its turn
    try:
        task.on_queued()
        assert started.wait(5)
        task.status = TaskStatus.COMPLETED
        task.result = outcomes_payload(
            [
                (stuck, Outcome.FAILED_RETRYABLE, Reason.DECODE_TIMEOUT),
                (behind, Outcome.DEFERRED, Reason.DECODE_TIMEOUT),
            ]
        )
        with caplog.at_level(logging.INFO, logger=_FINDER_LOGGER):
            finder.on_task_complete(task, None)
            fresh = _new_picture(hold_vault, "test-fresh-while-decoding.png")
            assert finder.find_task() is None, "no batch while the decode runs"
            release.set()
            task._preload_thread.join(10)
            assert not task._preload_thread.is_alive()
            found = _found(finder)
    finally:
        release.set()

    assert {fresh, behind} <= found, "the batch is built once the decode returns"
    assert stuck not in found, "held: its decode timed out"
    assert names[1] not in seen
    logged = [r.getMessage() for r in caplog.records if r.name == _FINDER_LOGGER]
    assert [m for m in logged if "decodes still running" in m and f"[{stuck}]" in m]
    assert [m for m in logged if "have returned" in m]


def test_no_new_batch_when_a_task_leaves_a_decode_running_during_the_query(
    hold_vault, make_finder, monkeypatch
):
    """A task can complete while ``find_task`` is querying for candidates.

    ``find_task`` looks for running decodes before its query. A completion that
    lands during the query and leaves a decode running must still stop the
    batch being built, and leave none of its pictures claimed.
    """
    finder = make_finder()
    root = hold_vault.db.image_root
    name = "test-stuck-during-query.png"
    PILImage.new("RGB", (32, 24), "plum").save(os.path.join(root, name))
    stuck = _new_picture(hold_vault, name)
    _new_picture(hold_vault, "test-fresh-during-query.png")
    started, release = threading.Event(), threading.Event()
    load = embedding_module.load_pil_classified

    def blocking_load(path, **kwargs):
        started.set()
        release.wait(10)
        return load(path, **kwargs)

    monkeypatch.setattr(embedding_module, "load_pil_classified", blocking_load)
    task = ImageEmbeddingTask(
        database=hold_vault.db,
        clip_workflow=None,
        batch=[(stuck, os.path.join(root, name))],
    )
    query = hold_vault.db.run_immediate_read_task

    def complete_during_query(fn, *args, **kwargs):
        finder.on_task_complete(task, None)
        return query(fn, *args, **kwargs)

    try:
        task.on_queued()
        assert started.wait(5)
        task.status = TaskStatus.COMPLETED
        task.result = outcomes_payload(
            [(stuck, Outcome.FAILED_RETRYABLE, Reason.DECODE_TIMEOUT)]
        )
        monkeypatch.setattr(
            hold_vault.db, "run_immediate_read_task", complete_during_query
        )
        assert finder.find_task() is None, "no batch while the decode runs"
        assert not finder._claimed_picture_ids
    finally:
        release.set()
        task._preload_thread.join(10)


def test_the_retry_hold_doubles_up_to_its_cap_with_a_half_delay_floor():
    clock = _Clock()
    start = clock.now
    low = RetryHold(base_s=2.0, cap_s=300.0, clock=clock, rng=lambda: 0.0)
    high = RetryHold(base_s=2.0, cap_s=300.0, clock=clock, rng=lambda: 1.0)

    assert [low.hold("k") for _ in range(4)] == [2.0, 4.0, 8.0, 16.0]
    assert [high.hold("k") for _ in range(4)] == [4.0, 8.0, 16.0, 32.0]
    # Thousands of failures in, still a number: at the cap, never past it.
    for _ in range(2000):
        low.hold("k")
        high.hold("k")
    assert (low.hold("k"), high.hold("k")) == (150.0, 300.0)

    clock.now = start + 149.99
    assert low.active() == {"k"}
    clock.now = start + 150.01
    assert low.active() == set()

    low.clear("k")
    assert low.hold("k") == 2.0
