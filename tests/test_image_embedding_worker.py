import os
import shutil
import threading
import time
from datetime import datetime, timezone

import logging

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from sqlmodel import Session, select, text

from pixlstash.db_models import Picture
from pixlstash.tasks import image_embedding_task as embedding_module
from pixlstash.tasks.image_embedding_task import ImageEmbeddingTask
from pixlstash.tasks.task_outcome import Outcome, Reason
from pixlstash.utils.image_processing.media_load import LoadFailure
from pixlstash.vault import Vault

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")


def test_empty_embedding_blob_is_work_again_after_the_0112_reset(tmp_path):
    """A failed picture is stored as NULL and re-selected by ``fetch_work``.

    The empty-blob marker older releases wrote is no longer matched (the
    ``length()`` arm defeated the probe's partial index); migration 0112 resets
    those rows to NULL once, after which they are ordinary missing work.
    """
    # Copy real images so MissingFilePurgeTask doesn't delete these records
    shutil.copy(os.path.join(PICTURES_DIR, "Bad1.png"), tmp_path / "missing.jpg")
    shutil.copy(os.path.join(PICTURES_DIR, "Bad1.png"), tmp_path / "empty.jpg")
    shutil.copy(os.path.join(PICTURES_DIR, "Bad1.png"), tmp_path / "done.jpg")
    with Vault(image_root=str(tmp_path)) as vault:
        now = datetime.now(timezone.utc)

        def seed(session: Session):
            missing = Picture(
                file_path=str(tmp_path / "missing.jpg"),
                format="jpg",
                width=64,
                height=64,
                deleted=False,
                imported_at=now,
                image_embedding=None,
                aesthetic_score=3.0,
                created_at=now,
            )
            empty = Picture(
                file_path=str(tmp_path / "empty.jpg"),
                format="jpg",
                width=64,
                height=64,
                deleted=False,
                imported_at=now,
                image_embedding=np.array([], dtype=np.float32).tobytes(),
                aesthetic_score=3.0,
                created_at=now,
            )
            done = Picture(
                file_path=str(tmp_path / "done.jpg"),
                format="jpg",
                width=64,
                height=64,
                deleted=False,
                imported_at=now,
                image_embedding=np.ones(512, dtype=np.float32).tobytes(),
                aesthetic_score=3.0,
                created_at=now,
            )
            session.add(missing)
            session.add(empty)
            session.add(done)
            session.commit()
            return missing.id, empty.id

        missing_id, empty_id = vault.db.run_task(seed)

        def fetch(session: Session):
            work = ImageEmbeddingTask.fetch_work(session, aesthetic_disabled=True)
            remaining = ImageEmbeddingTask.count_remaining(
                session, aesthetic_disabled=True
            )
            return {row.id for row in work}, int(remaining or 0)

        assert vault.db.run_task(fetch) == ({missing_id}, 1)

        # The task writes nothing at all for a failure, never an empty blob:
        # the row stays missing work.
        task = ImageEmbeddingTask(
            database=vault.db,
            clip_workflow=None,
            batch=[(missing_id, str(tmp_path / "missing.jpg"))],
        )
        result = task._process_preloaded(
            [(missing_id, "missing.jpg", None, None, None, LoadFailure.RESOURCE)]
        )
        assert result["changed"] == []
        assert result["outcomes"] == [
            (missing_id, Outcome.FAILED_RETRYABLE, Reason.RESOURCE)
        ]
        assert vault.db.run_task(fetch) == ({missing_id}, 1)

        def reset(session: Session):  # the 0112 one-time NULL-reset
            session.execute(
                text(
                    "UPDATE picture SET image_embedding = NULL "
                    "WHERE image_embedding IS NOT NULL AND length(image_embedding) = 0"
                )
            )
            session.commit()

        vault.db.run_task(reset)
        assert vault.db.run_task(fetch) == ({missing_id, empty_id}, 2)


def test_fetch_work_includes_missing_aesthetic_when_embedding_exists(tmp_path):
    """Pictures with valid embeddings but missing aesthetic score should be selected."""
    # Create actual files so MissingFilePurgeTask doesn't delete these records
    shutil.copy(
        os.path.join(PICTURES_DIR, "Bad1.png"), tmp_path / "needs_aesthetic.jpg"
    )
    shutil.copy(os.path.join(PICTURES_DIR, "Bad1.png"), tmp_path / "complete.jpg")
    with Vault(image_root=str(tmp_path)) as vault:
        now = datetime.now(timezone.utc)

        def seed(session: Session):
            needs_aesthetic = Picture(
                file_path=str(tmp_path / "needs_aesthetic.jpg"),
                format="jpg",
                width=64,
                height=64,
                deleted=False,
                imported_at=now,
                image_embedding=np.ones(512, dtype=np.float32).tobytes(),
                aesthetic_score=None,
                created_at=now,
            )
            complete = Picture(
                file_path=str(tmp_path / "complete.jpg"),
                format="jpg",
                width=64,
                height=64,
                deleted=False,
                imported_at=now,
                image_embedding=np.ones(512, dtype=np.float32).tobytes(),
                aesthetic_score=2.5,
                created_at=now,
            )
            session.add(needs_aesthetic)
            session.add(complete)
            session.commit()

        vault.db.run_task(seed)

        work = vault.db.run_task(
            lambda session: ImageEmbeddingTask.fetch_work(
                session,
                aesthetic_disabled=False,
            )
        )
        remaining = int(
            vault.db.run_task(
                lambda session: ImageEmbeddingTask.count_remaining(
                    session,
                    aesthetic_disabled=False,
                )
            )
            or 0
        )

        assert len(work) == 1
        assert remaining == 1


class _RecordingWorkflow:
    """Records what the task hands the workflow; embeds nothing real."""

    def __init__(self):
        self.calls = []
        self.preprocessed = []

    def is_ready(self):
        return True

    def ensure_ready(self):
        return None

    def reduce_for_preprocess(self, images):
        return images

    def preprocess_images(self, images):
        tensors = [f"tensor-for-{id(img)}" for img in images]
        self.preprocessed.extend(tensors)
        return tensors

    def encode_images(self, images, tensors=None):
        self.calls.append({"n": len(images), "tensors": tensors})
        return np.ones((len(images), 4), dtype=np.float32)


def test_the_preload_pool_does_the_cpu_work_and_keeps_batch_order(tmp_path):
    """Decode, dhash and CLIP preprocessing happen off the GPU worker.

    On the worker they were 4 s of a 4.2 s batch of 128; the forward pass is
    a tenth of a second. The pool must also keep the batch's order, or a
    hash lands on the wrong picture.
    """
    from PIL import Image as PILImage

    from pixlstash.tasks.image_embedding_task import ImageEmbeddingTask
    from pixlstash.vault import Vault

    names = [f"p{i}.png" for i in range(9)]
    for i, name in enumerate(names):
        PILImage.new("RGB", (32, 24), (i * 20, 40, 60)).save(tmp_path / name)
    with Vault(image_root=str(tmp_path)) as vault:
        workflow = _RecordingWorkflow()
        task = ImageEmbeddingTask(
            database=vault.db,
            clip_workflow=workflow,
            batch=[(i + 1, name) for i, name in enumerate(names)],
        )
        task._preload_images_task()
        preloaded = task._wait_for_preload()

        assert [entry[0] for entry in preloaded] == list(range(1, 10)), "order kept"
        assert all(len(entry) == 5 for entry in preloaded)
        assert all(entry[3] is not None for entry in preloaded), "dhash preloaded"
        # The pool interleaves calls, so the workflow's own record is in call
        # order; what matters is that every picture's tensor is ITS tensor.
        assert [entry[4] for entry in preloaded] == [
            f"tensor-for-{id(entry[2])}" for entry in preloaded
        ], "each picture carries the tensor preprocessed from its own image"
        assert sorted(workflow.preprocessed) == sorted(e[4] for e in preloaded)


@pytest.mark.parametrize("clip_loaded", [False, True])
def test_the_embedding_preload_keeps_only_what_clip_reads(tmp_path, clip_loaded):
    """No decoded original survives the preload (#2748), and the tensor CLIP
    is given is bit-identical to the one the original produces, whether or
    not CLIP was loaded when the preload ran (the first batches after start-up).

    Holding the originals beside their tensors kept 6.6 GB for a batch of
    64 large photos, three batches preloading at once. An image CLIP would
    enlarge is held as it is, never upscaled.
    """
    from types import SimpleNamespace

    import open_clip
    import torch
    from PIL import Image as PILImage

    from pixlstash.inference.workflows.clip_embedding import ClipEmbeddingWorkflow
    from pixlstash.tagger_plugins.clip_service import (
        CLIP_MODEL_NAME,
        CLIP_MODEL_WEIGHTS,
        ClipService,
        _unloaded_first_resize,
    )

    # The preprocessing a real load gets, without its weights.
    _model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_MODEL_WEIGHTS, load_weights=False
    )
    assert repr(_unloaded_first_resize()) == repr(preprocess.transforms[0])

    service = ClipService(device="cpu")
    if clip_loaded:
        service._model = service._tokenizer = object()
        service._preprocess = preprocess
    workflow = ClipEmbeddingWorkflow(SimpleNamespace(clip_service=service))

    rng = np.random.default_rng(2748)
    # Landscape, portrait, one smaller than CLIP's input, two thin strips:
    # the last two would be enlarged by the resize, and are held as they are.
    sizes = [(2400, 1600), (333, 1001), (100, 50), (1201, 77), (2000, 20)]
    names = [f"p{i}.png" for i in range(len(sizes))]
    for (w, h), name in zip(sizes, names):
        array = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        PILImage.fromarray(array).save(tmp_path / name)
    with Vault(image_root=str(tmp_path)) as vault:
        task = ImageEmbeddingTask(
            database=vault.db,
            clip_workflow=workflow,
            batch=[(i + 1, name) for i, name in enumerate(names)],
        )
        task._preload_images_task()
        preloaded = task._wait_for_preload()

    assert [entry[0] for entry in preloaded] == [1, 2, 3, 4, 5]
    for (_pid, name, held, dhash, tensor), size in zip(preloaded, sizes):
        original = PILImage.open(tmp_path / name).convert("RGB")
        expected = size if min(size) < 224 else None
        if expected is None:
            assert min(held.size) == 224, f"{name} {size} held at {held.size}"
        else:
            assert held.size == expected, f"{name} {size} upscaled to {held.size}"
        assert torch.equal(preprocess(held), preprocess(original)), name
        assert dhash == ImageEmbeddingTask._compute_dhash(original), name
        if clip_loaded:
            assert torch.equal(tensor, preprocess(original)), name
        else:
            assert tensor is None


def test_an_animated_gif_preloads_three_frames_and_a_still_gif_one(tmp_path):
    """An animated GIF is averaged over the frames a video is (#1487)."""
    from PIL import Image as PILImage

    frames = [PILImage.new("RGB", (32, 24), c) for c in ("red", "green", "blue")]
    frames[0].save(
        tmp_path / "anim.gif", save_all=True, append_images=frames[1:], duration=100
    )
    frames[0].save(tmp_path / "still.gif")
    with Vault(image_root=str(tmp_path)) as vault:
        task = ImageEmbeddingTask(
            database=vault.db,
            clip_workflow=_RecordingWorkflow(),
            batch=[(1, "anim.gif"), (2, "still.gif")],
        )
        task._preload_images_task()

        assert [entry[0] for entry in task._wait_for_preload()] == [1, 1, 1, 2]


def test_the_worker_uses_the_preloaded_tensors_and_hashes(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    from pixlstash.tasks.image_embedding_task import ImageEmbeddingTask
    from pixlstash.vault import Vault

    img = PILImage.new("RGB", (32, 24), "blue")
    with Vault(image_root=str(tmp_path)) as vault:
        workflow = _RecordingWorkflow()
        task = ImageEmbeddingTask(database=vault.db, clip_workflow=workflow, batch=[])
        monkeypatch.setattr(
            ImageEmbeddingTask,
            "_compute_dhash",
            staticmethod(
                lambda *_: pytest.fail("the worker re-hashed a preloaded image")
            ),
        )
        monkeypatch.setattr(task, "_save_results", staticmethod(lambda *_: []))
        monkeypatch.setattr(
            ImageEmbeddingTask,
            "_save_results",
            staticmethod(lambda session, updates: []),
        )

        task._process_preloaded(
            [(1, "a.png", img, "aa" * 8, "t1"), (2, "b.png", img, "bb" * 8, "t2")]
        )

        assert workflow.calls == [{"n": 2, "tensors": ["t1", "t2"]}]

        # A batch with any tensor missing is handed over whole, untensored:
        # one forward pass, preprocessed by the service in one go.
        workflow.calls.clear()
        task._process_preloaded(
            [(1, "a.png", img, "aa" * 8, "t1"), (2, "b.png", img, "bb" * 8, None)]
        )
        assert workflow.calls == [{"n": 2, "tensors": None}]


# --- Per-picture outcomes -------------------------------------------------
#
# The tests below share one vault (never started: no planner races them) and
# assert on the pictures each one seeds, never on counts of the whole table.


@pytest.fixture(scope="module")
def shared_vault(tmp_path_factory):
    root = tmp_path_factory.mktemp("embedding-outcomes")
    with Vault(image_root=str(root)) as vault:
        yield vault


@pytest.fixture
def no_scorer(monkeypatch):
    """No aesthetic scorer, and none is looked for or downloaded."""
    monkeypatch.setattr(ImageEmbeddingTask, "_aesthetic_model", None)
    monkeypatch.setattr(ImageEmbeddingTask, "_aesthetic_disabled", True)


def _add_picture(vault, name, **columns) -> int:
    """One picture row; ``file_path`` is *name* under the vault's image root."""
    if "file_path" not in columns:
        columns["file_path"] = os.path.join(vault.db.image_root, name)

    def add(session: Session):
        picture = Picture(format="png", width=32, height=24, **columns)
        session.add(picture)
        session.commit()
        return picture.id

    return vault.db.run_task(add)


def _stored(vault, pid) -> tuple:
    """``(image_embedding, aesthetic_score)`` as the row holds them now."""
    return vault.db.run_task(
        lambda session: tuple(
            session.exec(
                select(Picture.image_embedding, Picture.aesthetic_score).where(
                    Picture.id == pid
                )
            ).one()
        )
    )


class _FailingWorkflow(_RecordingWorkflow):
    """A workflow whose ``encode_images`` answers with *encode(images)*."""

    def __init__(self, encode):
        super().__init__()
        self._encode = encode

    def encode_images(self, images, tensors=None):
        return self._encode(images)


def _raise(exc):
    def encode(images):
        raise exc

    return encode


_EMBEDDED = np.ones(512, dtype=np.float32).tobytes()


@pytest.mark.parametrize(
    "failure",
    ["load", "clip"],
    ids=["load-runs-out-of-resource", "clip-raises"],
)
def test_a_failure_never_erases_an_embedding_already_stored(
    shared_vault, no_scorer, failure
):
    """A picture that only lacks its score keeps its embedding when it fails.

    Such a picture is work because of its score alone; a failed attempt used
    to write the embedding column too, and lost one that was good.
    """
    pid = _add_picture(shared_vault, "test-scored-later.png", image_embedding=_EMBEDDED)
    img = PILImage.new("RGB", (32, 24), "teal")
    if failure == "load":
        workflow = _RecordingWorkflow()
        preloaded = [(pid, "p.png", None, None, None, LoadFailure.RESOURCE)]
        expected = (pid, Outcome.FAILED_RETRYABLE, Reason.RESOURCE)
    else:
        workflow = _FailingWorkflow(_raise(RuntimeError("test-boom")))
        preloaded = [(pid, "p.png", img, "aa" * 8, None)]
        expected = (pid, Outcome.FAILED_RETRYABLE, Reason.INFERENCE_ERROR)
    task = ImageEmbeddingTask(
        database=shared_vault.db, clip_workflow=workflow, batch=[(pid, "p.png")]
    )

    result = task._process_preloaded(preloaded)

    assert result["outcomes"] == [expected]
    assert result["changed"] == []
    assert _stored(shared_vault, pid) == (_EMBEDDED, None)


def test_one_batch_reports_each_picture_in_batch_order(shared_vault, no_scorer):
    """A good still, an undecodable file and an animated GIF, in one batch.

    Each gets its own outcome, in the order the batch named them; only the
    undecodable one is kept out of every finder, and only the two that loaded
    are written.
    """
    root = shared_vault.db.image_root
    PILImage.new("RGB", (32, 24), "red").save(os.path.join(root, "test-good.png"))
    with open(os.path.join(root, "test-broken.jpg"), "wb") as handle:
        handle.write(b"test- these bytes are not a JPEG \xff\xd8\x00")
    frames = [PILImage.new("RGB", (32, 24), c) for c in ("red", "green", "blue")]
    frames[0].save(
        os.path.join(root, "test-anim.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100,
    )
    good = _add_picture(shared_vault, "test-good.png")
    broken = _add_picture(shared_vault, "test-broken.jpg")
    anim = _add_picture(shared_vault, "test-anim.gif")
    workflow = _RecordingWorkflow()
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=workflow,
        batch=[
            (anim, os.path.join(root, "test-anim.gif")),
            (broken, os.path.join(root, "test-broken.jpg")),
            (good, os.path.join(root, "test-good.png")),
        ],
    )

    task._preload_images_task()
    result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [
        (anim, Outcome.SUCCEEDED, None),
        (broken, Outcome.FAILED_PERMANENT, Reason.UNDECODABLE),
        (good, Outcome.SUCCEEDED, None),
    ]
    assert sorted(pid for _, pid, _, _ in result["changed"]) == sorted([anim, good])
    assert [call["n"] for call in workflow.calls] == [4], "3 frames + 1 still"
    assert shared_vault.db.unprocessable_images.is_suppressed(broken)
    assert not shared_vault.db.unprocessable_images.is_suppressed(good)
    assert _stored(shared_vault, broken) == (None, None)
    assert _stored(shared_vault, good)[0] is not None


@pytest.mark.parametrize(
    "exc, reason",
    [
        (RuntimeError("can't start new thread"), Reason.RESOURCE),
        (ValueError("test-bug"), Reason.LOAD_ERROR),
    ],
    ids=["out-of-threads", "a-bug"],
)
def test_a_preload_pool_that_fails_leaves_every_picture_to_retry(
    shared_vault, no_scorer, monkeypatch, caplog, exc, reason
):
    """The pool failing says nothing about the files: none is marked."""
    pids = [_add_picture(shared_vault, f"test-pool-{i}.png") for i in range(2)]
    start = threading.Thread.start

    def no_decode_threads(thread):
        if thread.name.startswith("EmbedDecode-"):
            raise exc
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", no_decode_threads)
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(pid, f"test-pool-{pid}.png") for pid in pids],
    )

    with caplog.at_level(logging.ERROR, logger=embedding_module.__name__):
        task._preload_images_task()
        result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [
        (pid, Outcome.FAILED_RETRYABLE, reason) for pid in pids
    ]
    assert result["changed"] == []
    assert [r.getMessage() for r in caplog.records if "preload pool" in r.getMessage()]
    for pid in pids:
        assert not shared_vault.db.unprocessable_images.is_suppressed(pid)
        assert _stored(shared_vault, pid) == (None, None)


@pytest.mark.parametrize(
    "encode, reason",
    [
        (_raise(RuntimeError("test-boom")), Reason.INFERENCE_ERROR),
        (_raise(MemoryError()), Reason.RESOURCE),
        (
            _raise(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
            Reason.RESOURCE,
        ),
        (
            lambda images: np.ones((len(images) + 1, 4), np.float32),
            Reason.INFERENCE_ERROR,
        ),
        (lambda images: None, Reason.INFERENCE_ERROR),
    ],
    ids=["raises", "out-of-memory", "out-of-vram", "a-row-too-many", "no-rows"],
)
def test_clip_failing_on_a_batch_leaves_every_picture_to_retry(
    shared_vault, no_scorer, encode, reason
):
    """Rows that cannot be matched to pictures are not written to any of them."""
    pids = [_add_picture(shared_vault, f"test-encode-{i}.png") for i in range(2)]
    img = PILImage.new("RGB", (32, 24), "navy")
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_FailingWorkflow(encode),
        batch=[(pid, "p.png") for pid in pids],
    )

    result = task._process_preloaded([(pid, "p.png", img, "aa" * 8) for pid in pids])

    assert result["outcomes"] == [
        (pid, Outcome.FAILED_RETRYABLE, reason) for pid in pids
    ]
    assert result["changed"] == []
    for pid in pids:
        assert _stored(shared_vault, pid) == (None, None)


class _OneScore(torch.nn.Module):
    """A scorer that answers one score whatever it is given."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(1, 1)


@pytest.mark.parametrize(
    "scorer",
    [lambda: torch.nn.Linear(5, 1), _OneScore],
    ids=["raises", "one-score-for-two"],
)
def test_a_scorer_failure_keeps_the_embedding_and_retries_the_score(
    shared_vault, monkeypatch, scorer
):
    """The embedding is written; the score is left as it was and asked for again."""
    monkeypatch.setattr(ImageEmbeddingTask, "_aesthetic_model", scorer())
    pids = [_add_picture(shared_vault, f"test-score-{i}.png") for i in range(2)]
    img = PILImage.new("RGB", (32, 24), "olive")
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(pid, "p.png") for pid in pids],
    )

    result = task._process_preloaded([(pid, "p.png", img, "aa" * 8) for pid in pids])

    assert result["outcomes"] == [
        (pid, Outcome.FAILED_RETRYABLE, Reason.SCORER_ERROR) for pid in pids
    ]
    assert sorted(pid for _, pid, _, _ in result["changed"]) == sorted(pids)
    for pid in pids:
        embedding, score = _stored(shared_vault, pid)
        assert embedding is not None and score is None


def test_a_picture_whose_row_is_gone_is_not_reported_done(shared_vault, no_scorer):
    """Deleted while it was being embedded: nothing to write it to."""
    kept = _add_picture(shared_vault, "test-kept.png")
    gone = 10**9
    img = PILImage.new("RGB", (32, 24), "maroon")
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(kept, "k.png"), (gone, "g.png")],
    )

    result = task._process_preloaded(
        [(kept, "k.png", img, "aa" * 8), (gone, "g.png", img, "bb" * 8)]
    )

    assert result["outcomes"] == [
        (kept, Outcome.SUCCEEDED, None),
        (gone, Outcome.FAILED_RETRYABLE, Reason.PERSIST_ERROR),
    ]
    assert [pid for _, pid, _, _ in result["changed"]] == [kept]


@pytest.mark.parametrize(
    "workflow",
    [None, _FailingWorkflow(_raise(RuntimeError("test-unused")))],
    ids=["no-workflow", "not-ready"],
)
def test_an_unavailable_backend_fails_the_batch_retryably_and_stops_the_preload(
    shared_vault, no_scorer, monkeypatch, workflow
):
    if workflow is not None:
        monkeypatch.setattr(workflow, "is_ready", lambda: False)
        monkeypatch.setattr(
            workflow,
            "ensure_ready",
            _raise(RuntimeError("test-no-backend")),
        )
    pids = [_add_picture(shared_vault, f"test-backend-{i}.png") for i in range(2)]
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=workflow,
        batch=[(pid, "p.png") for pid in pids],
    )

    result = task._run_task()

    assert result["outcomes"] == [
        (pid, Outcome.FAILED_RETRYABLE, Reason.BACKEND_UNAVAILABLE) for pid in pids
    ]
    assert result["changed"] == []
    assert task._preload_cancel.is_set()


def test_a_cancelled_preload_defers_every_picture_it_did_not_load(
    shared_vault, no_scorer
):
    """Not attempted is not failed: nothing is marked and nothing is held."""
    pids = [_add_picture(shared_vault, f"test-cancel-{i}.png") for i in range(2)]
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(pid, "p.png") for pid in pids],
    )
    task._preload_cancel.set()

    task._preload_images_task()
    result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [
        (pid, Outcome.DEFERRED, Reason.CANCELLED) for pid in pids
    ]
    for pid in pids:
        assert not shared_vault.db.unprocessable_images.is_suppressed(pid)


@pytest.fixture
def stuck_decode(monkeypatch):
    """Decodes of a file named ``test-stuck*`` block until released.

    Yields ``(started, release, seen)``: *started* is set when such a decode
    begins, *release* lets it return, and *seen* names every file the preload
    decoded. A blocked decode gives up by itself after 10 s and is released on
    teardown, so a failing test leaves no thread behind for long.
    """
    started = threading.Event()
    release = threading.Event()
    seen = []
    load = embedding_module.load_pil_classified

    def blocking_load(path, **kwargs):
        name = os.path.basename(path)
        seen.append(name)
        if name.startswith("test-stuck"):
            started.set()
            release.wait(10)
        return load(path, **kwargs)

    monkeypatch.setattr(embedding_module, "load_pil_classified", blocking_load)
    yield started, release, seen
    release.set()


def _png(vault, name: str) -> tuple:
    """A small PNG called *name* under the vault's root: ``(pid, path)``."""
    path = os.path.join(vault.db.image_root, name)
    PILImage.new("RGB", (32, 24), "purple").save(path)
    return _add_picture(vault, name), path


def test_a_stuck_decode_holds_neither_the_worker_nor_the_batch(
    shared_vault, no_scorer, stuck_decode, caplog
):
    """The wait for the preload is bounded; a decode cannot be interrupted.

    Past the deadline the pictures that loaded are embedded, the one still
    decoding is retried later, and the one never started stays eligible and is
    not started after all. The decode threads are daemons, so one that never
    returns cannot hold the interpreter's exit either.
    """
    started, release, seen = stuck_decode
    ok, ok_path = _png(shared_vault, "test-before-the-stuck-one.png")
    stuck, stuck_path = _png(shared_vault, "test-stuck-in-a-batch.png")
    later, later_path = _png(shared_vault, "test-after-the-stuck-one.png")
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(ok, ok_path), (stuck, stuck_path), (later, later_path)],
    )
    task._PRELOAD_WORKERS = 1  # one decode at a time: the third waits its turn

    task.on_queued()
    assert started.wait(5)
    with caplog.at_level(logging.WARNING, logger=embedding_module.__name__):
        began = time.perf_counter()
        preloaded = task._wait_for_preload(timeout_s=0.05)
        waited_s = time.perf_counter() - began
    decoders = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(f"EmbedDecode-{task.id[:8]}")
    ]
    result = task._process_preloaded(preloaded)
    release.set()
    task._preload_thread.join(10)

    assert waited_s < 1.0
    assert result["outcomes"] == [
        (ok, Outcome.SUCCEEDED, None),
        (stuck, Outcome.FAILED_RETRYABLE, Reason.DECODE_TIMEOUT),
        (later, Outcome.DEFERRED, Reason.DECODE_TIMEOUT),
    ]
    assert [pid for _, pid, _, _ in result["changed"]] == [ok]
    assert _stored(shared_vault, stuck) == (None, None)
    assert not shared_vault.db.unprocessable_images.is_suppressed(stuck)
    warned = [r.getMessage() for r in caplog.records if "stopped waiting" in r.msg]
    assert len(warned) == 1 and f"[{stuck}]" in warned[0]
    assert decoders and all(thread.daemon for thread in decoders)
    assert not task._preload_thread.is_alive()
    assert "test-after-the-stuck-one.png" not in seen


def test_a_cancel_does_not_wait_on_a_stuck_decode(
    shared_vault, no_scorer, stuck_decode, monkeypatch, caplog
):
    """A pre-empted task gives the worker back and says a decode is still running."""
    started, release, _ = stuck_decode
    pid, path = _png(shared_vault, "test-stuck-when-cancelled.png")
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(pid, path)],
    )
    monkeypatch.setattr(ImageEmbeddingTask, "_CANCEL_JOIN_S", 0.05)

    task.on_queued()
    assert started.wait(5)
    with caplog.at_level(logging.WARNING, logger=embedding_module.__name__):
        began = time.perf_counter()
        task.on_cancel()
        waited_s = time.perf_counter() - began
    release.set()
    task._preload_thread.join(10)

    assert waited_s < 1.0
    assert [r.levelno for r in caplog.records if "did not stop in time" in r.msg] == [
        logging.WARNING
    ]


def test_a_cancel_before_inference_embeds_nothing_and_a_requeue_runs_afresh(
    shared_vault, no_scorer
):
    """Pre-empted after the decode, the batch is left for later, untouched.

    A requeued task is a fresh run: the earlier cancel no longer applies.
    """
    pid, path = _png(shared_vault, "test-preempted.png")
    workflow = _RecordingWorkflow()
    task = ImageEmbeddingTask(
        database=shared_vault.db, clip_workflow=workflow, batch=[(pid, path)]
    )

    task._preload_images_task()
    task.on_cancel()
    result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [(pid, Outcome.DEFERRED, Reason.CANCELLED)]
    assert result["changed"] == []
    assert workflow.calls == []
    assert _stored(shared_vault, pid) == (None, None)
    assert not shared_vault.db.unprocessable_images.is_suppressed(pid)

    task.on_queued()
    result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [(pid, Outcome.SUCCEEDED, None)]
    assert [call["n"] for call in workflow.calls] == [1]
    assert _stored(shared_vault, pid)[0] is not None


def test_entries_keep_the_batch_order_when_decodes_finish_out_of_order(
    shared_vault, no_scorer, stuck_decode
):
    """Each decode publishes when it returns; the worker still reads batch order."""
    started, release, _ = stuck_decode
    first, first_path = _png(shared_vault, "test-stuck-but-first.png")
    rest = [_png(shared_vault, f"test-fast-{i}.png") for i in range(2)]
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(first, first_path), *rest],
    )
    task._PRELOAD_WORKERS = 2

    task.on_queued()
    assert started.wait(5)
    deadline = time.monotonic() + 5
    while len(task._preload_done) < len(rest) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(task._preload_done) == len(rest), "the later pictures finished first"
    release.set()
    preloaded = task._wait_for_preload()

    assert [entry[0] for entry in preloaded] == [first] + [pid for pid, _ in rest]


@pytest.mark.parametrize(
    "exc, reason",
    [(MemoryError(), Reason.RESOURCE), (ValueError("test-bug"), Reason.LOAD_ERROR)],
    ids=["out-of-memory", "a-bug"],
)
def test_a_decode_that_raises_fails_only_its_own_picture(
    shared_vault, no_scorer, monkeypatch, caplog, exc, reason
):
    """An exception the loader did not classify is retried, never marked."""
    bad, bad_path = _png(shared_vault, f"test-decode-raises-{reason.value}.png")
    good, good_path = _png(shared_vault, f"test-decodes-fine-{reason.value}.png")
    load = embedding_module.load_pil_classified

    def raising_load(path, **kwargs):
        if os.path.basename(path).startswith("test-decode-raises"):
            raise exc
        return load(path, **kwargs)

    monkeypatch.setattr(embedding_module, "load_pil_classified", raising_load)
    task = ImageEmbeddingTask(
        database=shared_vault.db,
        clip_workflow=_RecordingWorkflow(),
        batch=[(bad, bad_path), (good, good_path)],
    )

    with caplog.at_level(logging.ERROR, logger=embedding_module.__name__):
        task._preload_images_task()
        result = task._process_preloaded(task._wait_for_preload())

    assert result["outcomes"] == [
        (bad, Outcome.FAILED_RETRYABLE, reason),
        (good, Outcome.SUCCEEDED, None),
    ]
    assert not shared_vault.db.unprocessable_images.is_suppressed(bad)
    assert _stored(shared_vault, bad) == (None, None)
    assert [
        r.levelno for r in caplog.records if f"picture {bad} failed" in r.getMessage()
    ] == [logging.ERROR]


def test_the_scrapheap_and_rows_without_a_file_are_not_embedding_work(shared_vault):
    def counted(session: Session) -> int:
        return int(
            ImageEmbeddingTask.count_remaining(session, aesthetic_disabled=False)
        )

    before = shared_vault.db.run_task(counted)
    live = _add_picture(shared_vault, "test-live.png")
    deleted = _add_picture(shared_vault, "test-deleted.png", deleted=True)
    no_file = _add_picture(shared_vault, "test-no-file.png", file_path=None)

    work = shared_vault.db.run_task(
        lambda session: ImageEmbeddingTask.fetch_work(
            session, aesthetic_disabled=False, limit=10**6
        )
    )
    ids = {row.id for row in work}

    assert live in ids
    assert deleted not in ids
    assert no_file not in ids
    assert shared_vault.db.run_task(counted) == before + 1
