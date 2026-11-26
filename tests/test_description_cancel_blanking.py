"""A description batch writes only what it learnt, and only over what it read.

Cancelling mid-batch makes the workflow return early with a partial result. If
``DescriptionTask`` then writes ``""`` to every picture the workflow did not
caption, those pictures are permanently uncaptionable: ``MissingDescriptionFinder``
selects only ``NULL`` or a ``__description::`` sentinel, and an empty string is
neither. So a cancel must persist nothing for the pictures it skipped.

The same holds for a batch that captioned nothing at all (the workflow raised,
returned nothing, or a captioning model that failed to load returned no caption
for anyone): that says the backend is down, not that these pictures cannot be
captioned, so nothing is written, every picture is reported as retryable and
the finder holds the batch back on a backoff instead of selecting it again on
the next sweep. A picture whose own file cannot be loaded is not cleared either:
it is retried later, or held out by the unprocessable-image registry when the
file does not decode. Only a picture whose file decodes and that the model left
uncaptioned while it captioned others is cleared - that is what stops an
uncaptionable picture being retried for ever.

A caption is written only over the description the inference started from: a
person who edits a caption while the batch is running keeps their edit. The
write itself makes that comparison, on the one writer every edit goes through,
so an edit queued just ahead of it counts too. Nor is a caption written to a
picture in the Scrapheap: the finder and the "awaiting captions" count leave
out Scrapheap pictures and pictures with no file, and the write skips a
picture moved to the Scrapheap while its batch ran.

Three tests cover the routing rather than the writes: the cancel event is the
task's own (the workflow object running a batch is not always the one the task
was constructed with), it is checked per picture in the Florence video loop, and
``BaseTask.run`` refuses a task cancelled before it started while still reporting
COMPLETED for one that returned normally.

Deliberately Server-free. An in-memory SQLite engine and a small stub are
everything ``DescriptionTask`` and the finder touch, so this file builds no
environment (CLAUDE.md, "Tests: reuse the environment, don't rebuild it") and
costs milliseconds rather than the ~1.35 s a ``Server`` does.
"""

import os
import threading
import types

import pytest
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine

from pixlstash.db_models import Picture, make_description_sentinel
from pixlstash.inference.workflows.description import DescriptionWorkflow
from pixlstash.task_runner import TaskCancelledError
from pixlstash.tasks.base_task import BaseTask, TaskStatus
from pixlstash.tasks.description_task import DescriptionTask
from pixlstash.tasks.missing_description_finder import MissingDescriptionFinder
from pixlstash.tasks.task_outcome import Reason, RetryHold, read_outcomes
from pixlstash.utils import unprocessable_image_registry as registry_module
from pixlstash.utils.unprocessable_image_registry import UnprocessableImageRegistry
from pixlstash.vault import Vault


class _StubDB:
    """The VaultDatabase entry points ``DescriptionTask`` and its finder use."""

    image_root = ""

    def __init__(self):
        self._engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self._engine)
        self.unprocessable_images = UnprocessableImageRegistry()

    def run_task(self, func, *args, priority=None, **kwargs):
        with Session(self._engine) as session:
            return func(session, *args, **kwargs)

    def run_immediate_read_task(self, func, *args, **kwargs):
        return self.run_task(func, *args, **kwargs)


def _seed(db, count, paths=None):
    """*count* pictures with a NULL description; *paths* names their files."""

    def insert(session):
        ids = []
        for n in range(count):
            pic = Picture(
                file_path=str(paths[n]) if paths else f"/nonexistent/cancel-{n}.png",
                format="png",
                width=8,
                height=8,
                pixel_sha=f"cancel-{n}-{paths[n] if paths else ''}",
            )
            session.add(pic)
            session.flush()
            ids.append(int(pic.id))
        session.commit()
        return ids

    return db.run_task(insert)


class _CancelledMidBatchWorkflow:
    """Captions the first picture, then takes the cancel the runner would send."""

    def __init__(self):
        self.task = None
        self.stop_event = None

    def generate_batch(self, pictures, engine_override=None, stop_event=None):
        first = pictures[0]
        self.task.on_cancel()  # what TaskRunner.stop() does to an active task
        self.stop_event = stop_event
        return {first.id: "captioned-before-the-cancel"}

    def estimate_vram_mb(self, image_count, plugin_name=None):
        return 0


def _run_cancelled_batch(db, picture_ids):
    workflow = _CancelledMidBatchWorkflow()
    task = DescriptionTask(db, workflow, _handed(db, picture_ids))
    workflow.task = task
    task._run_task()
    return task, workflow


def _descriptions(db, picture_ids):
    return db.run_immediate_read_task(
        lambda s: [s.get(Picture, pid).description for pid in picture_ids]
    )


def _selectable(db):
    return {
        p.id
        for p in db.run_immediate_read_task(
            lambda s: MissingDescriptionFinder._fetch_missing_descriptions(s, 100)
        )
    }


def test_cancelled_batch_leaves_skipped_pictures_uncaptioned_not_blank(tmp_path):
    """The blocker: a picture the cancel skipped keeps a NULL description, so the
    finder picks it up again. An empty string is a permanent, silent exclusion."""
    db = _StubDB()
    # Files that decode, so only the cancel can keep the skipped one unwritten.
    captioned, skipped = _seed(db, 2, paths=_files(tmp_path, 2))

    task, workflow = _run_cancelled_batch(db, [captioned, skipped])
    got_captioned, got_skipped = _descriptions(db, [captioned, skipped])

    assert workflow.stop_event is task._cancel_event, (
        "the task must hand the workflow its own cancel event; a workflow that "
        "owns the event is the wrong object under CPU spillover, which builds a "
        "fresh DescriptionWorkflow on every access"
    )

    assert got_captioned == "captioned-before-the-cancel"
    assert got_skipped is None, (
        f"the cancelled batch wrote {got_skipped!r} to a picture it never "
        "captioned; MissingDescriptionFinder selects only NULL or a "
        "__description:: sentinel, so that picture can never be captioned again"
    )

    selectable = _selectable(db)
    assert skipped in selectable, "skipped picture must be re-queued"
    assert captioned not in selectable, "captioned picture must not be re-queued"


def test_cancelled_batch_does_not_destroy_a_pending_recaption_request(tmp_path):
    """A ``__description::`` sentinel is a user asking for a re-caption. A cancel
    must not consume the request by overwriting it with an empty string."""
    db = _StubDB()
    pending_file, filler_file = _files(tmp_path, 2)
    (pending,) = _seed(db, 1, paths=[pending_file])

    def set_sentinel(session):
        pic = session.get(Picture, pending)
        pic.description = "__description::joycaption"
        session.add(pic)
        session.commit()

    db.run_task(set_sentinel)
    # One picture, and the workflow cancels after captioning pictures[0] - so the
    # batch that carries only this picture returns a caption for it. Give it a
    # second, unrelated picture so this one is the skipped half.
    (filler,) = _seed(db, 1, paths=[filler_file])
    _run_cancelled_batch(db, [filler, pending])

    (got,) = _descriptions(db, [pending])
    assert got == "__description::joycaption", (
        f"the cancel overwrote a pending re-caption request with {got!r}"
    )
    assert pending in _selectable(db)


class _Workflow:
    """A description workflow whose batch result is ``generate(pictures)``."""

    def __init__(self, generate):
        self.generate = generate
        self.batches: list[list[int]] = []

    def generate_batch(self, pictures, engine_override=None, stop_event=None):
        self.batches.append([pic.id for pic in pictures])
        return self.generate(pictures)

    def estimate_vram_mb(self, image_count, plugin_name=None):
        return 0


def _raises(pictures):
    raise RuntimeError("test-endpoint refused the connection")


def _returns_nothing(pictures):
    return {}


def _captions_nobody(pictures):
    # What a captioning model that failed to load produces: the Florence-2
    # service logs the load error and then answers None for every picture.
    return {pic.id: None for pic in pictures}


def _captions_all(pictures):
    return {pic.id: f"fake-caption-{pic.id}" for pic in pictures}


def _files(tmp_path, count):
    """*count* small PNGs that decode (the doubles never open them)."""
    paths = []
    for n in range(count):
        path = tmp_path / f"fake-{n}.png"
        Image.new("RGB", (8, 8)).save(path)
        paths.append(path)
    return paths


def _handed(db, picture_ids):
    """The pictures as the finder hands them to a task: detached row values."""

    def read(session):
        return [
            types.SimpleNamespace(
                id=pic.id, description=pic.description, file_path=pic.file_path
            )
            for pic in (session.get(Picture, pid) for pid in picture_ids)
        ]

    return db.run_immediate_read_task(read)


def _set_description(db, picture_id, description):
    def write(session):
        pic = session.get(Picture, picture_id)
        pic.description = description
        session.add(pic)
        session.commit()

    db.run_task(write)


def _set_fields(db, picture_id, **fields):
    def write(session):
        pic = session.get(Picture, picture_id)
        for name, value in fields.items():
            setattr(pic, name, value)
        session.add(pic)
        session.commit()

    db.run_task(write)


@pytest.mark.parametrize(
    "generate",
    [_raises, _returns_nothing, _captions_nobody],
    ids=["raises", "returns-nothing", "captions-nobody"],
)
def test_a_batch_that_captions_nobody_changes_no_description(tmp_path, generate):
    """The backend failed, not the pictures. Writing ``""`` here turned an
    endpoint that was down for a minute into a permanently uncaptioned batch,
    and a captioning model that failed to load into a blanked library."""
    db = _StubDB()
    never, requested = _seed(db, 2, paths=_files(tmp_path, 2))
    sentinel = make_description_sentinel(None)
    _set_description(db, requested, sentinel)

    result = DescriptionTask(
        db, _Workflow(generate), _handed(db, [never, requested])
    )._run_task()

    assert _descriptions(db, [never, requested]) == [None, sentinel]
    assert {never, requested} <= _selectable(db)
    assert result["changed"] == []
    summary = read_outcomes(result)
    assert summary.failed_retryable == {never, requested}
    assert {summary.reasons[pid] for pid in (never, requested)} == {
        Reason.BACKEND_UNAVAILABLE
    }


def test_a_picture_the_model_cannot_caption_is_cleared_when_the_batch_works(
    tmp_path,
):
    """Guard against over-correcting. Blanking a picture the model left
    uncaptioned while it captioned the rest of the batch is deliberate - it is
    what stops that picture being retried for ever."""
    db = _StubDB()
    captioned, uncaptionable = _seed(db, 2, paths=_files(tmp_path, 2))

    def generate(pictures):
        return {captioned: "fake-caption", uncaptionable: None}

    result = DescriptionTask(
        db, _Workflow(generate), _handed(db, [captioned, uncaptionable])
    )._run_task()

    assert _descriptions(db, [captioned, uncaptionable]) == ["fake-caption", ""]
    assert uncaptionable not in _selectable(db)
    summary = read_outcomes(result)
    assert summary.succeeded == {captioned}
    assert summary.failed_permanent == {uncaptionable}
    assert summary.reasons[uncaptionable] is Reason.INFERENCE_ERROR


def test_a_picture_whose_file_cannot_be_read_is_retried_not_cleared(tmp_path):
    """The picture's own file failed, not the model. A file that is missing for
    now (a dropped share) or whose whole folder is unplugged is retried later;
    clearing its description would lose it for good."""
    db = _StubDB()
    (readable,) = _files(tmp_path, 1)
    missing = tmp_path / "fake-missing.png"
    unplugged = tmp_path / "fake-unplugged-drive" / "fake-1.png"
    ok, gone, unreachable = _seed(db, 3, paths=[readable, missing, unplugged])

    def generate(pictures):
        return {ok: "fake-caption"}  # the workflow could not open the other two

    result = DescriptionTask(
        db, _Workflow(generate), _handed(db, [ok, gone, unreachable])
    )._run_task()

    assert _descriptions(db, [ok, gone, unreachable]) == ["fake-caption", None, None]
    summary = read_outcomes(result)
    assert summary.failed_retryable == {gone, unreachable}
    assert summary.reasons[gone] is Reason.IO_TRANSIENT
    assert summary.reasons[unreachable] is Reason.UNREACHABLE
    assert db.unprocessable_images.is_suppressed(unreachable)
    assert not db.unprocessable_images.is_suppressed(gone)


def test_a_corrupt_file_is_not_taken_for_a_backend_that_is_down(tmp_path):
    """A batch of files that read but do not decode (one broken import) captions
    nobody, and that is the files, not the backend. Taken for the backend, the
    same batch would be the next probe for ever and stall the backlog behind
    it. It is held out of the finders until the file changes, and its
    description is left alone."""
    db = _StubDB()
    corrupt = [tmp_path / f"fake-corrupt-{n}.png" for n in range(2)]
    for path in corrupt:
        path.write_bytes(b"fake-image-bytes")
    first, second = _seed(db, 2, paths=corrupt)

    result = DescriptionTask(
        db, _Workflow(_captions_nobody), _handed(db, [first, second])
    )._run_task()

    assert _descriptions(db, [first, second]) == [None, None]
    summary = read_outcomes(result)
    assert summary.failed_permanent == {first, second}
    assert {summary.reasons[pid] for pid in (first, second)} == {Reason.UNDECODABLE}
    assert db.unprocessable_images.active_suppressed_ids() == {first, second}


def test_marking_a_corrupt_file_does_not_decode_it_again(tmp_path, monkeypatch):
    """The settle's classified load is the registry's evidence: the corrupt
    file is decoded once, still marked, and pinned to the signature taken
    before that one decode."""
    db = _StubDB()
    corrupt = tmp_path / "fake-corrupt-once.png"
    corrupt.write_bytes(b"fake-image-bytes")
    (pid,) = _seed(db, 1, paths=[corrupt])
    decodes = []
    real = registry_module.load_pil_classified

    def counting(path, *, multiframe_count):
        decodes.append(path)
        return real(path, multiframe_count=multiframe_count)

    monkeypatch.setattr(registry_module, "load_pil_classified", counting)

    result = DescriptionTask(
        db, _Workflow(_captions_nobody), _handed(db, [pid])
    )._run_task()

    assert read_outcomes(result).reasons[pid] is Reason.UNDECODABLE
    assert decodes == [str(corrupt)], "one classified decode, shared with the gate"
    assert db.unprocessable_images.is_suppressed(pid)
    stat = os.stat(corrupt)
    _, mtime_ns, size = db.unprocessable_images.snapshot()[pid]
    assert (mtime_ns, size) == (stat.st_mtime_ns, stat.st_size)


@pytest.mark.parametrize(
    ("persons_value", "model_gives_it"),
    [
        ("fake-person-caption", "fake-caption"),
        ("fake-person-caption", None),
        (make_description_sentinel("fake-engine"), "fake-caption"),
    ],
    ids=["caption-over-caption", "blank-over-caption", "caption-over-request"],
)
def test_a_description_a_person_writes_while_the_batch_runs_is_kept(
    tmp_path, persons_value, model_gives_it
):
    """The batch reads the description before inference and writes after it,
    minutes later on a slow model. A person who types a caption in between, or
    asks for a caption from another engine, keeps what they did: the batch
    writes only where the description is still the one it read."""
    db = _StubDB()
    edited, untouched = _seed(db, 2, paths=_files(tmp_path, 2))
    handed = _handed(db, [edited, untouched])

    def generate(pictures):
        _set_description(db, edited, persons_value)  # lands mid-inference
        return {edited: model_gives_it, untouched: "fake-caption"}

    result = DescriptionTask(db, _Workflow(generate), handed)._run_task()

    assert _descriptions(db, [edited, untouched]) == [persons_value, "fake-caption"]
    assert [pid for _, pid, _, _ in result["changed"]] == [untouched]
    summary = read_outcomes(result)
    assert summary.succeeded == {untouched}
    assert summary.deferred == {edited}


def test_a_picture_moved_to_the_scrapheap_while_the_batch_runs_is_not_written(
    tmp_path,
):
    """The write needs a live row. A picture the owner moves to the Scrapheap
    while its caption is being generated keeps the description it had, and is
    not announced as changed; restored, it is captioned on a later sweep."""
    db = _StubDB()
    moved, other = _seed(db, 2, paths=_files(tmp_path, 2))
    handed = _handed(db, [moved, other])

    def generate(pictures):
        _set_fields(db, moved, deleted=True)  # lands mid-inference
        return {moved: "fake-caption", other: "fake-caption"}

    result = DescriptionTask(db, _Workflow(generate), handed)._run_task()

    assert _descriptions(db, [moved, other]) == [None, "fake-caption"]
    assert [pid for _, pid, _, _ in result["changed"]] == [other]
    summary = read_outcomes(result)
    assert summary.deferred == {moved}
    assert summary.succeeded == {other}


def _awaiting_captions(db) -> int:
    """The "awaiting captions" count, read through a vault that owns *db*.

    The count is a method of the vault because it leaves out the pictures the
    vault's unprocessable-image registry is holding back.
    """
    vault = types.SimpleNamespace(db=db)
    return db.run_immediate_read_task(
        lambda session: Vault._count_missing_descriptions(vault, session)
    )


def test_the_finder_and_the_count_leave_out_scrapheap_and_fileless_pictures(
    tmp_path,
):
    """Captioning selects only pictures outside the Scrapheap that have a file,
    and the "awaiting captions" count is that same selection: a picture the
    finder never picks must not sit in the count for ever. A picture restored
    from the Scrapheap is selected and counted again."""
    db = _StubDB()
    live, scrapped, fileless = _seed(db, 3, paths=_files(tmp_path, 3))
    _set_fields(db, scrapped, deleted=True)
    _set_fields(db, fileless, file_path=None)

    assert _selectable(db) == {live}
    assert _awaiting_captions(db) == 1

    _set_fields(db, scrapped, deleted=False)
    assert scrapped in _selectable(db)
    assert _awaiting_captions(db) == 2


class _EditAheadDB(_StubDB):
    """A writer queue on which a person's edit reaches the writer just ahead of
    the next write, after anything the task read before queuing it."""

    def __init__(self):
        super().__init__()
        self._ahead = None
        self.edits_landed = 0

    def edit_ahead_of_next_write(self, picture_id, description):
        self._ahead = (picture_id, description)

    def run_task(self, func, *args, priority=None, **kwargs):
        if self._ahead is not None:
            picture_id, description = self._ahead
            self._ahead = None
            _set_description(self, picture_id, description)
            self.edits_landed += 1
        return super().run_task(func, *args, priority=priority, **kwargs)

    def run_immediate_read_task(self, func, *args, **kwargs):
        return super().run_task(func, *args, **kwargs)  # a read is not a write


def test_a_person_edit_that_reaches_the_writer_just_ahead_of_the_batch_is_kept(
    tmp_path,
):
    """The batch decides inside its own write, not before queuing it: every
    write goes through one writer, so an edit queued just ahead of the batch's
    write is in the row by the time the batch compares. A picture in the same
    batch with a pending request for a caption, unchanged, gets its caption."""
    db = _EditAheadDB()
    edited, requested = _seed(db, 2, paths=_files(tmp_path, 2))
    sentinel = make_description_sentinel(None)
    _set_description(db, requested, sentinel)
    handed = _handed(db, [edited, requested])

    def generate(pictures):
        db.edit_ahead_of_next_write(edited, "fake-person-caption")
        return {edited: "fake-caption", requested: "fake-caption"}

    result = DescriptionTask(db, _Workflow(generate), handed)._run_task()

    assert db.edits_landed == 1
    assert _descriptions(db, [edited, requested]) == [
        "fake-person-caption",
        "fake-caption",
    ]
    summary = read_outcomes(result)
    assert summary.succeeded == {requested}
    assert summary.deferred == {edited}


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _finder(db, workflow, clock, batch_size=2):
    """A finder over *db* whose holds run on *clock* and wait the full delay."""
    engine = types.SimpleNamespace(
        tagger_settings={"active_description_plugin": "florence2"},
        description_batch_size=lambda: batch_size,
        description_workflow=workflow,
    )
    return MissingDescriptionFinder(
        db,
        lambda: engine,
        retry_hold=RetryHold(clock=clock, rng=lambda: 1.0),
        backend_hold=RetryHold(cap_s=60.0, clock=clock, rng=lambda: 1.0),
    )


def _run(finder):
    """Find a task, run it as the runner would, and report it complete."""
    task = finder.find_task()
    if task is None:
        return None
    error = None
    try:
        task.run()
    except Exception as exc:
        error = exc
    finder.on_task_complete(task, error)
    return task.params["picture_ids"]


def test_the_finder_does_not_select_a_failed_batch_on_the_next_sweep(tmp_path):
    """The planner sweeps every few seconds. Without a hold, a batch whose
    backend is down is selected, fails and is selected again on each sweep."""
    db = _StubDB()
    _seed(db, 4, paths=_files(tmp_path, 4))
    finder = MissingDescriptionFinder(
        db,
        lambda: types.SimpleNamespace(
            tagger_settings={"active_description_plugin": "florence2"},
            description_batch_size=lambda: 2,
            description_workflow=_Workflow(_raises),
        ),
    )

    assert _run(finder) is not None
    assert finder.find_task() is None, (
        "a batch that just failed because the backend is down was followed at "
        "once by another one"
    )


def test_holds_grow_while_the_backend_stays_down_and_a_success_clears_them(
    tmp_path,
):
    db = _StubDB()
    a, b = _seed(db, 2, paths=_files(tmp_path, 2))
    workflow = _Workflow(_returns_nothing)
    clock = _Clock()
    finder = _finder(db, workflow, clock)

    assert _run(finder) == [a, b]  # first failure: held 4 s
    assert finder.find_task() is None
    clock.now += 5
    assert _run(finder) == [a, b]  # second failure: held 8 s
    clock.now += 5
    assert finder.find_task() is None, "the second hold must be longer"
    clock.now += 4

    workflow.generate = _captions_all
    assert _run(finder) == [a, b]
    assert _descriptions(db, [a, b]) == [f"fake-caption-{a}", f"fake-caption-{b}"]

    # A person asks for a new caption and the backend is down again: the
    # backoff starts from the beginning, both for the picture and the backend.
    _set_description(db, a, make_description_sentinel(None))
    workflow.generate = _returns_nothing
    assert _run(finder) == [a]
    clock.now += 5
    assert _run(finder) == [a], (
        "a success must clear the holds: the next failure is held for the "
        "first delay again, not the third"
    )


def test_a_held_picture_does_not_crowd_out_the_rest_of_the_backlog(tmp_path):
    """The finder reads three batches' worth of candidates. Held pictures must
    be left out of that read, or enough of them stall every other picture."""
    db = _StubDB()
    unreadable = [tmp_path / f"fake-missing-{n}.png" for n in range(3)]
    (readable,) = _files(tmp_path, 1)
    *held, waiting = _seed(db, 4, paths=[*unreadable, readable])
    workflow = _Workflow(_returns_nothing)
    # One picture per batch; a batch whose only picture is unreadable fails on
    # that picture's file, not on the backend, so no backend hold is taken.
    finder = _finder(db, workflow, _Clock(), batch_size=1)

    for pid in held:
        assert _run(finder) == [pid]
    workflow.generate = _captions_all

    assert _run(finder) == [waiting]


def test_pictures_on_an_unplugged_drive_do_not_crowd_out_the_rest(tmp_path):
    """The registry holds a picture whose drive is unplugged for as long as the
    drive is gone, long after any retry hold has run out. Those pictures must
    be left out of the read as well."""
    db = _StubDB()
    unplugged = [tmp_path / "fake-unplugged-drive" / f"fake-{n}.png" for n in range(3)]
    (readable,) = _files(tmp_path, 1)
    *suppressed, waiting = _seed(db, 4, paths=[*unplugged, readable])
    for pid, path in zip(suppressed, unplugged):
        assert db.unprocessable_images.mark_unreachable(pid, str(path))
    finder = _finder(db, _Workflow(_captions_all), _Clock(), batch_size=1)

    assert _run(finder) == [waiting]


def test_a_batch_that_always_fails_does_not_stall_the_rest_of_the_backlog(tmp_path):
    """One picture can make the model fail the whole batch it is in, every
    time. Every such failure reads as the backend being down, so the batch is
    held picture by picture as well as backend-wide: the picture holds outgrow
    the backend hold, and the next probe is a different batch."""
    db = _StubDB()
    poison, neighbour, *rest = _seed(db, 4, paths=_files(tmp_path, 4))

    def generate(pictures):
        if poison in {pic.id for pic in pictures}:
            raise RuntimeError("test-model fails every batch holding this picture")
        return _captions_all(pictures)

    clock = _Clock()
    finder = _finder(db, _Workflow(generate), clock)
    for _ in range(300):  # five minutes, a second at a time
        if _run(finder) == rest:
            break
        clock.now += 1
    else:
        pytest.fail("the batch that keeps failing stalled the rest of the backlog")

    assert _descriptions(db, [poison, neighbour, *rest]) == [
        None,
        None,
        *(f"fake-caption-{pid}" for pid in rest),
    ]


def test_a_picture_held_while_the_finder_reads_is_not_claimed(tmp_path):
    """A task can finish between the finder's read and its claim. Its failed
    pictures are held and released then, so the claim check alone lets them
    straight back into a batch."""
    db = _StubDB()
    racing, other = _seed(db, 2, paths=_files(tmp_path, 2))
    clock = _Clock()
    finder = _finder(db, _Workflow(_captions_all), clock)
    read = db.run_immediate_read_task

    def read_then_hold(func, *args, **kwargs):
        rows = read(func, *args, **kwargs)
        finder._retry_hold.hold(racing)  # the other task's failure lands now
        return rows

    db.run_immediate_read_task = read_then_hold
    task = finder.find_task()
    db.run_immediate_read_task = read

    assert task.params["picture_ids"] == [other]
    assert racing not in finder._claimed_picture_ids


def test_a_failed_task_holds_its_pictures_and_a_cancelled_one_does_not(tmp_path):
    db = _StubDB()
    a, b = _seed(db, 2, paths=_files(tmp_path, 2))
    clock = _Clock()
    finder = _finder(db, _Workflow(_captions_all), clock)

    task = finder.find_task()
    finder.on_task_complete(task, TaskCancelledError("test-shutdown"))
    task = finder.find_task()
    assert task.params["picture_ids"] == [a, b], "a cancel says nothing about them"

    finder.on_task_complete(task, RuntimeError("test-out-of-memory"))
    assert finder.find_task() is None


def test_a_picture_cleared_as_uncaptionable_starts_its_backoff_afresh(tmp_path):
    """A permanent outcome ends the picture's retries, so its hold is forgotten
    too: a person who later asks for a new caption does not wait out the backoff
    of the failures before it."""
    db = _StubDB()
    a, b = _seed(db, 2, paths=_files(tmp_path, 2))
    workflow = _Workflow(_returns_nothing)
    clock = _Clock()
    finder = _finder(db, workflow, clock)

    assert _run(finder) == [a, b]  # held 4 s
    clock.now += 5
    assert _run(finder) == [a, b]  # held 8 s
    clock.now += 9
    workflow.generate = lambda pictures: {b: "fake-caption"}
    assert _run(finder) == [a, b]
    assert _descriptions(db, [a]) == [""]

    _set_description(db, a, make_description_sentinel(None))
    workflow.generate = _returns_nothing
    assert _run(finder) == [a]  # held for the first delay, 4 s
    clock.now += 5
    assert _run(finder) == [a], "the failures before the clearing still count"


@pytest.mark.parametrize(
    "result",
    [{"changed": []}, {"outcomes": [("fake-not-a-picture-id",)]}],
    ids=["no-outcomes", "malformed-outcomes"],
)
def test_a_task_that_reports_no_outcomes_holds_its_pictures(tmp_path, result):
    """Every DescriptionTask reports outcomes, so a result without them is a bug.
    Its pictures are not known to have succeeded; they are held, not reselected
    at once, and their claims are still released."""
    db = _StubDB()
    _seed(db, 2, paths=_files(tmp_path, 2))
    finder = _finder(db, _Workflow(_captions_all), _Clock())

    task = finder.find_task()
    task.result = result
    finder.on_task_complete(task, None)

    assert finder.find_task() is None
    assert not finder._claimed_picture_ids


def test_a_failure_while_settling_still_releases_the_batch(tmp_path):
    db = _StubDB()
    _seed(db, 2, paths=_files(tmp_path, 2))
    finder = _finder(db, _Workflow(_captions_all), _Clock())
    task = finder.find_task()

    def settle(task, error):
        raise RuntimeError("test-settle-bug")

    finder._settle = settle
    with pytest.raises(RuntimeError, match="test-settle-bug"):
        finder.on_task_complete(task, None)
    assert not finder._claimed_picture_ids, "the batch's claims were stranded"


def test_a_cancel_stops_the_florence_video_loop_before_the_next_video():
    """Videos are captioned one at a time in ``_generate_batch_florence``'s first
    loop, and a video is the slowest single item there is - so the check has to
    be in that loop, not only on the still-image chunks after it."""
    stop = threading.Event()
    captioned: list[str] = []

    def generate_caption(path, _retry_on_cpu=True):
        captioned.append(path)
        stop.set()  # the cancel lands while the first video is being captioned
        return "first-video"

    engine = types.SimpleNamespace(
        tagger_settings={"active_description_plugin": "florence2"},
        ensure_captioning_ready=lambda: None,
        florence_service=types.SimpleNamespace(
            generate_caption=generate_caption,
            description_batch_size=lambda: 8,
            generate_captions_batch=lambda paths, stop_event=None: {},
        ),
    )
    pictures = [
        types.SimpleNamespace(id=1, file_path="/nonexistent/one.mp4"),
        types.SimpleNamespace(id=2, file_path="/nonexistent/two.mp4"),
    ]

    results = DescriptionWorkflow(engine, image_root=None).generate_batch(
        pictures, stop_event=stop
    )

    assert captioned == ["/nonexistent/one.mp4"], (
        "the cancelled batch captioned a second video: the stop event is not "
        "checked in the per-picture loop"
    )
    assert 2 not in results


def test_run_refuses_a_task_cancelled_before_it_started():
    """``BaseTask.on_cancel`` sets the event and ``run()`` reads it before the
    first attempt, so a task cancelled while queued and then picked up by a
    worker does no work at all."""

    class _CountingTask(BaseTask):
        def __init__(self):
            super().__init__(task_type="CountingTask")
            self.attempts = 0

        def _run_task(self):
            self.attempts += 1
            return "did the work"

    task = _CountingTask()
    task.on_cancel()

    assert task.run() is None
    assert task.attempts == 0
    assert task.status == TaskStatus.CANCELLED

    # And a requeue is a fresh run: on_queued clears the event, or the task
    # would refuse its work for the rest of the process.
    task.on_queued()
    assert task.run() == "did the work"
    assert task.attempts == 1
    assert task.status == TaskStatus.COMPLETED


def test_a_cancel_landing_after_the_work_still_reports_completed():
    """The cancel event is set by the shutdown thread and can land at any
    instant, including after ``_run_task`` has committed its rows. Reporting
    CANCELLED there would race a finished task - and ``Vault._on_task_complete``
    fires only for COMPLETED, so it would swallow the notification for work that
    is already in the database."""

    class _SelfCancellingTask(BaseTask):
        def _run_task(self):
            self.on_cancel()  # what TaskRunner.stop() does mid-flight
            return "committed before the cancel"

    task = _SelfCancellingTask(task_type="SelfCancellingTask")

    assert task.run() == "committed before the cancel"
    assert task.status == TaskStatus.COMPLETED
