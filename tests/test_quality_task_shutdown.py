import time
from types import SimpleNamespace

from PIL import Image

from pixlstash.database import DBPriority
from pixlstash.tasks.quality_task import QualityTask
from pixlstash.utils.image_processing.image_utils import ImageUtils


class _FakeDb:
    def __init__(self):
        self.image_root = "/tmp"


def test_quality_task_on_cancel_stops_preload_thread(monkeypatch):
    SLOW_LOADER_DELAY_SECONDS = 0.02
    # Wait several loader cycles so preloading has time to start before cancellation.
    PRELOAD_START_WAIT_SECONDS = SLOW_LOADER_DELAY_SECONDS * 4

    pictures = [SimpleNamespace(id=i, file_path=f"img-{i}.jpg") for i in range(1, 100)]
    task = QualityTask(database=_FakeDb(), pictures=pictures)

    monkeypatch.setattr(
        ImageUtils,
        "resolve_picture_path",
        staticmethod(lambda image_root, file_path: file_path),
    )

    def _slow_loader(_file_path, _max_side):
        time.sleep(SLOW_LOADER_DELAY_SECONDS)
        return None

    # The preload's loader; the slow load keeps the thread busy at cancel time.
    monkeypatch.setattr(ImageUtils, "load_image_reduced", staticmethod(_slow_loader))

    task.on_queued()
    time.sleep(PRELOAD_START_WAIT_SECONDS)
    task.on_cancel()

    assert task._preload_cancel.is_set()
    assert task._preload_thread is not None
    assert not task._preload_thread.is_alive()


class _RecordingDb(_FakeDb):
    """Records the priority of every write the task queues."""

    def __init__(self):
        super().__init__()
        self.priorities = []

    def run_task(self, func, *args, priority=DBPriority.IMMEDIATE, **kwargs):
        self.priorities.append(priority)


def test_the_metadata_backfill_writes_at_the_background_priority(tmp_path):
    """A quality batch's writes queue behind a user's, not ahead of them.

    The backfill of a picture's format and size used ``run_task``'s default,
    IMMEDIATE, the priority of a request someone is waiting on. The picture is
    a real file, so the backfill reads its size through the task's own loader.
    """
    Image.new("RGB", (8, 6), "white").save(tmp_path / "img-1.png")
    db = _RecordingDb()
    db.image_root = str(tmp_path)
    picture = SimpleNamespace(
        id=1, file_path="img-1.png", format=None, width=None, height=None
    )

    QualityTask(database=db, pictures=[picture])._backfill_missing_picture_metadata(
        [picture]
    )

    assert (picture.format, picture.width, picture.height) == ("PNG", 8, 6)
    assert db.priorities == [DBPriority.LOW]
