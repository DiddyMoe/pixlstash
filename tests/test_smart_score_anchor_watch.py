"""The smart-score anchor watch clears every cached score when, and only when, the anchors move.

Every smart score is a distance from the user's best- and worst-rated pictures
(the anchors, capped per side at ``ANCHOR_LIMIT``). A rating change that moves
that set makes every cached score stale; one that leaves it alone makes none
stale. These tests drive real writer sessions of a real vault, so they fail if
the watch is not attached to the sessions, not only if its comparison is wrong.

An embedding landing on a rated picture moves the set too, but the watch
checks only once such writes go quiet (``EMBEDDING_QUIET_S``). The test that
needs the timer to fire shortens it; the ones about what arms it lengthen it
and fire it by hand, so no assertion races a clock.

The last group pins the scorer's own anchor read: the same rows, in the same
order, as the queries the watch compares, with the embeddings fetched by
primary key rather than carried through the sort.

The server runs without background workers: nothing but the test writes
``smart_score``, so a NULL read back here was written by the watch.
"""

import json
import os
import tempfile
import threading
import time
from concurrent.futures import Future
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, update
from sqlmodel import select

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.db_models.tag import Tag
from pixlstash.db_models.tag_prediction import TagPrediction
from pixlstash.scoring import smart_score
from pixlstash.server import Server
from pixlstash.tasks.smart_score_task import SmartScoreTask
from pixlstash.utils.service import smart_score_invalidation
from pixlstash.utils.service.smart_score_invalidation import (
    SmartScoreAnchorWatch,
    note_anchor_inputs_changing,
)
from pixlstash.vault import Vault
from tests.utils import API_PREFIX, wipe_tables

SEEDED_SMART_SCORE = 0.5


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "server-config.json")
        with open(config_path, "w") as fh:
            json.dump({"port": 0, "disable_background_workers": True}, fh)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(scope="module")
def client(server):
    client = TestClient(server.api)
    resp = client.post(
        f"{API_PREFIX}/login",
        json={"username": "testuser", "password": "testpassword"},
    )
    assert resp.status_code == 200
    return client


@pytest.fixture(autouse=True)
def clean_library(server):
    """Start every test with no pictures, no queued check and no baseline.

    ``forget_baseline`` also cancels a quiet-period check a previous test left
    armed.
    """
    _drain(server)
    server.vault.db.run_task(
        wipe_tables, [TagPrediction, Tag, Picture], priority=DBPriority.HIGH
    )
    server.vault.smart_score_anchor_watch.forget_baseline()
    yield


@pytest.fixture
def count_checks(server, monkeypatch):
    """Return a starter that counts, from the moment it is called, the checks that run.

    Called after seeding, so the check the seed's own inserts queue is not counted.
    """
    watch = server.vault.smart_score_anchor_watch

    def start() -> list:
        original = watch._check
        ran = []

        def counting_check(session):
            ran.append(1)
            return original(session)

        monkeypatch.setattr(watch, "_check", counting_check)
        return ran

    return start


def _embedding() -> bytes:
    return np.random.rand(128).astype(np.float32).tobytes()


def _seed(server, specs) -> list[int]:
    """Insert pictures carrying a cached smart score.

    Each spec may give ``score``, ``embedded`` (default True), ``deleted`` and
    ``created_at``.
    """

    def insert(session):
        ids = []
        for index, spec in enumerate(specs):
            picture = Picture(
                file_path=f"/vault/anchor_{index}.png",
                format="png",
                score=spec.get("score"),
                image_embedding=_embedding() if spec.get("embedded", True) else None,
                deleted=spec.get("deleted", False),
                created_at=spec.get("created_at"),
            )
            session.add(picture)
            session.flush()
            ids.append(int(picture.id))
        session.commit()
        return ids

    def cache_scores(session, ids):
        # Core, so the watch does not see it; a cached score is not an anchor input.
        session.exec(
            update(Picture)
            .where(Picture.id.in_(ids))
            .values(smart_score=SEEDED_SMART_SCORE)
        )
        session.commit()

    ids = server.vault.db.run_task(insert)
    # Inserting anchors through the ORM queues a check of its own. Let it run
    # before the scores are cached, then forget its baseline, so every test
    # starts from its seeded library with the watch holding nothing.
    _drain(server)
    server.vault.db.run_task(cache_scores, ids)
    server.vault.smart_score_anchor_watch.forget_baseline()
    return ids


def _set(server, picture_id: int, **values) -> None:
    """Write *values* onto a picture through the ORM, as the routes and undo do."""

    def write(session):
        picture = session.get(Picture, picture_id)
        for name, value in values.items():
            setattr(picture, name, value)
        session.add(picture)
        session.commit()

    server.vault.db.run_task(write)


def _drain(server) -> None:
    """Return once every LOW task queued so far, the anchor check included, has run.

    The writer runs tasks in (priority, submission) order, and a check is queued
    from inside the commit that asks for it, before that commit's task returns.
    """
    server.vault.db.run_task(lambda session: None, priority=DBPriority.LOW)


def _smart_scores(server, ids) -> list:
    def read(session):
        rows = dict(
            session.exec(
                select(Picture.id, Picture.smart_score).where(Picture.id.in_(ids))
            ).all()
        )
        return [rows[pid] for pid in ids]

    return server.vault.db.run_task(read)


def _epoch(server) -> int:
    return server.vault.smart_score_anchor_watch.epoch


def _all_cleared(server, ids) -> bool:
    return _smart_scores(server, ids) == [None] * len(ids)


def _all_kept(server, ids) -> bool:
    return _smart_scores(server, ids) == [SEEDED_SMART_SCORE] * len(ids)


def _wait_for_epoch_after(server, epoch: int, timeout_s: float = 10.0) -> None:
    """Return once a library-wide clear has bumped the epoch past *epoch*."""
    deadline = time.monotonic() + timeout_s
    while _epoch(server) == epoch:
        assert time.monotonic() < deadline, "the quiet-period check never cleared"
        time.sleep(0.01)


# ── which writes move the anchors ─────────────────────────────────────────────


def test_rating_a_picture_into_the_good_set_clears_every_cached_score(server):
    rated, bystander = _seed(server, [{"score": 3}, {"score": None}])
    epoch = _epoch(server)

    _set(server, rated, score=5)
    _drain(server)

    # The bystander's score was computed without the new anchor, so it is
    # stale too: the clear is library-wide, not per picture.
    assert _all_cleared(server, [rated, bystander])
    assert _epoch(server) == epoch + 1


def test_rating_a_picture_into_the_bad_set_clears_every_cached_score(server):
    rated, bystander = _seed(server, [{"score": 3}, {"score": None}])

    _set(server, rated, score=1)
    _drain(server)

    assert _all_cleared(server, [rated, bystander])


def test_rerating_an_anchor_within_the_good_band_clears(server):
    # A 5 weighs more than a 4, so the scorer's inputs moved although the
    # picture stays an anchor.
    anchor, bystander = _seed(server, [{"score": 4}, {"score": None}])

    _set(server, anchor, score=5)
    _drain(server)

    assert _all_cleared(server, [anchor, bystander])


def test_an_unembedded_picture_never_queues_a_check(server, count_checks):
    # The scorer only reads anchors that have an embedding, so this picture
    # cannot move the set however it is rated.
    rated, bystander = _seed(server, [{"score": 3, "embedded": False}, {"score": 2}])
    checks = count_checks()

    _set(server, rated, score=5)
    _drain(server)

    assert checks == []
    assert _all_kept(server, [rated, bystander])


def test_a_rating_between_the_bands_never_queues_a_check(server, count_checks):
    rated, bystander = _seed(server, [{"score": 2}, {"score": None}])
    checks = count_checks()

    _set(server, rated, score=3)
    _drain(server)

    assert checks == []
    assert _all_kept(server, [rated, bystander])


def test_a_rating_that_misses_a_full_anchor_set_keeps_every_cached_score(
    server, count_checks, monkeypatch
):
    # Two fives fill a two-anchor set; a new four ranks below both and never
    # enters it, so no score is stale. The old route cleared the library here.
    monkeypatch.setattr(smart_score, "ANCHOR_LIMIT", 2)
    first, second, rated = _seed(server, [{"score": 5}, {"score": 5}, {"score": 3}])
    checks = count_checks()
    epoch = _epoch(server)

    _set(server, rated, score=4)
    _drain(server)

    assert checks == [1]
    assert _all_kept(server, [first, second, rated])
    assert _epoch(server) == epoch


@pytest.mark.parametrize("score", [5, 1], ids=["good-anchor", "bad-anchor"])
def test_moving_an_anchor_to_the_scrapheap_clears(server, score):
    anchor, bystander = _seed(server, [{"score": score}, {"score": None}])

    _set(server, anchor, deleted=True)
    _drain(server)

    assert _all_cleared(server, [anchor, bystander])


def test_restoring_an_anchor_from_the_scrapheap_clears(server):
    anchor, bystander = _seed(server, [{"score": 5, "deleted": True}, {"score": None}])

    _set(server, anchor, deleted=False)
    _drain(server)

    assert _all_cleared(server, [anchor, bystander])


def test_a_core_update_announced_beforehand_is_checked(server):
    # The batch-rating route writes scores with one Core UPDATE, which the
    # before_flush hook cannot see; it calls note_anchor_inputs_changing first.
    rated, bystander = _seed(server, [{"score": 3}, {"score": None}])

    def write(session):
        note_anchor_inputs_changing(session)
        session.exec(update(Picture).where(Picture.id == rated).values(score=5))
        session.commit()

    server.vault.db.run_task(write)
    _drain(server)

    assert _all_cleared(server, [rated, bystander])


def test_a_burst_of_rating_commits_queues_one_check(server, count_checks):
    ids = _seed(server, [{"score": 3}, {"score": 3}, {"score": 3}, {"score": None}])
    checks = count_checks()
    epoch = _epoch(server)

    # Hold the writer so the three commits queue up behind it, then run them
    # back to back ahead of the LOW check the first one asks for.
    started, release = threading.Event(), threading.Event()
    gate = server.vault.db.submit_task(
        lambda session: (started.set(), release.wait(10)), priority=DBPriority.HIGH
    )
    assert started.wait(10)
    writes = [
        server.vault.db.submit_task(_rate, pid, 5, priority=DBPriority.HIGH)
        for pid in ids[:3]
    ]
    release.set()
    gate.result(10)
    for future in writes:
        future.result(10)
    _drain(server)

    assert checks == [1]
    assert _epoch(server) == epoch + 1
    assert _all_cleared(server, ids)


def _rate(session, picture_id: int, score: int) -> None:
    picture = session.get(Picture, picture_id)
    picture.score = score
    session.add(picture)
    session.commit()


def test_a_forgotten_baseline_does_not_read_as_an_anchor_change(server, count_checks):
    # After a restore the baseline describes a file that is gone. Forgotten,
    # the next check re-baselines instead of clearing a library whose cached
    # scores match its own anchors.
    watch = server.vault.smart_score_anchor_watch
    anchor, bystander = _seed(server, [{"score": 5}, {"score": None}])
    checks = count_checks()
    _set(server, anchor, score=3)  # baseline now holds the anchor; the check empties it
    _drain(server)
    _set(server, anchor, smart_score=SEEDED_SMART_SCORE)
    _set(server, bystander, smart_score=SEEDED_SMART_SCORE)

    def swap_in_other_library(session):
        session.exec(update(Picture).where(Picture.id == anchor).values(score=5))
        session.commit()

    server.vault.db.run_task(swap_in_other_library)  # Core: the watch does not see it
    watch.forget_baseline()
    watch.request_check()
    _drain(server)

    assert len(checks) == 2
    assert _all_kept(server, [anchor, bystander])


# ── embeddings arriving on and leaving rated pictures ────────────────────────


def test_an_embedding_landing_on_a_rated_picture_clears_once_writes_go_quiet(
    server, count_checks, monkeypatch
):
    # A rated picture is an anchor only once it has an embedding, so the
    # embedding task finishing it moves the set.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 0.05)
    rated, bystander = _seed(server, [{"score": 5, "embedded": False}, {}])
    checks = count_checks()
    epoch = _epoch(server)

    _set(server, rated, image_embedding=_embedding())
    _wait_for_epoch_after(server, epoch)
    _drain(server)

    assert checks == [1]
    assert _epoch(server) == epoch + 1
    assert _all_cleared(server, [rated, bystander])


def test_a_burst_of_embedding_commits_queues_one_check_after_the_last(
    server, count_checks, monkeypatch
):
    # Embeddings land a batch per commit. Checking after each would clear the
    # library once per batch; the watch waits until they stop.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    watch = server.vault.smart_score_anchor_watch
    *rated, bystander = _seed(
        server, [{"score": 5, "embedded": False}] * 3 + [{"score": None}]
    )
    checks = count_checks()
    epoch = _epoch(server)

    timers = []
    for picture_id in rated:
        _set(server, picture_id, image_embedding=_embedding())
        _drain(server)
        assert checks == [], "an embedding commit queued its check at once"
        timers.append(watch._quiet_timer)

    # Each commit restarted the quiet period: only the last timer is armed.
    assert all(timer.finished.is_set() for timer in timers[:-1])
    assert not timers[-1].finished.is_set()
    # Timer.cancel() cannot stop a callback that has already started. One that
    # did, from a superseded timer, must still queue nothing.
    timers[0].function(*timers[0].args)
    _drain(server)
    assert checks == []

    last = timers[-1]
    last.cancel()
    last.function(*last.args)  # the quiet period ends
    _drain(server)

    assert checks == [1]
    assert _epoch(server) == epoch + 1
    assert _all_cleared(server, [*rated, bystander])


def test_turning_an_anchor_clears_nothing_once_it_is_re_embedded(
    server, count_checks, monkeypatch
):
    # A turn clears the embedding and the embedding task restores it. In
    # between the picture is out of the set; counting that would clear the
    # library on the way out and again on the way back.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    watch = server.vault.smart_score_anchor_watch
    anchor, bystander = _seed(server, [{"score": 5}, {}])
    checks = count_checks()
    epoch = _epoch(server)

    _set(server, anchor, image_embedding=None)
    _drain(server)
    assert watch._quiet_timer is None, "removing the embedding armed a check"
    assert checks == []

    _set(server, anchor, image_embedding=_embedding())
    _drain(server)
    timer = watch._quiet_timer
    assert timer is not None, "the re-embed armed no check"
    timer.cancel()
    timer.function(*timer.args)  # the quiet period ends
    _drain(server)

    # The check ran and compared against the set from before the turn.
    assert checks == [1]
    assert _epoch(server) == epoch
    assert _all_kept(server, [anchor, bystander])


@pytest.mark.parametrize("score", [None, 3], ids=["unrated", "between-the-bands"])
def test_an_embedding_on_a_picture_that_cannot_be_an_anchor_arms_nothing(
    server, count_checks, monkeypatch, score
):
    # Most embeddings land on unrated imports; arming a check for each would
    # re-read the anchors after every embedding batch for nothing.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    watch = server.vault.smart_score_anchor_watch
    picture, bystander = _seed(
        server, [{"score": score, "embedded": False}, {"score": None}]
    )
    checks = count_checks()

    _set(server, picture, image_embedding=_embedding())
    _drain(server)

    assert watch._quiet_timer is None
    assert checks == []
    assert _all_kept(server, [picture, bystander])


def test_a_rating_and_an_embedding_in_one_commit_are_checked_at_once(
    server, monkeypatch
):
    # The rating is what the user is waiting on, so it does not wait for the
    # quiet period; the immediate check compares the whole set, embedding included.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    watch = server.vault.smart_score_anchor_watch
    rated, bystander = _seed(server, [{"score": 3, "embedded": False}, {}])

    _set(server, rated, score=5, image_embedding=_embedding())
    _drain(server)

    assert _all_cleared(server, [rated, bystander])
    assert watch._quiet_timer is None


def test_clearing_an_anchors_rating_and_embedding_in_one_commit_clears_at_once(
    server, monkeypatch
):
    # Judged with the embedding the flush leaves behind, the picture never looked
    # like an anchor; it was one until this commit.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    watch = server.vault.smart_score_anchor_watch
    anchor, bystander = _seed(server, [{"score": 5}, {}])
    epoch = _epoch(server)

    _set(server, anchor, score=None, image_embedding=None)
    _drain(server)

    assert _epoch(server) == epoch + 1
    assert _all_cleared(server, [anchor, bystander])
    assert watch._quiet_timer is None


class _RecordingDb:
    """Stands in for ``VaultDatabase``: records the checks a watch queues."""

    def __init__(self) -> None:
        self.queued = []

    def submit_task(self, func, *args, priority=None, **kwargs) -> Future:
        self.queued.append(func)
        future = Future()
        future.set_result(0)
        return future


def test_a_closed_watch_queues_no_quiet_check(monkeypatch):
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    db = _RecordingDb()
    watch = SmartScoreAnchorWatch(db)
    watch._schedule_quiet_check()
    timer = watch._quiet_timer

    watch.close()
    timer.join(5)
    assert not timer.is_alive(), "close() left the quiet timer running"
    # A callback that had already started when close() ran queues nothing.
    timer.function(*timer.args)
    watch._schedule_quiet_check()
    assert watch._quiet_timer is None, "a closed watch armed a new timer"
    assert db.queued == []

    # Positive control: the same callback on an open watch queues the check.
    open_watch = SmartScoreAnchorWatch(db)
    open_watch._schedule_quiet_check()
    live = open_watch._quiet_timer
    live.cancel()
    live.function(*live.args)
    assert db.queued == [open_watch._check]


def test_forgetting_the_baseline_cancels_a_pending_quiet_check(monkeypatch):
    # A restore replaces the database and forgets the baseline; a check armed by
    # the old file's writes would compare the new file against nothing useful.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    db = _RecordingDb()
    watch = SmartScoreAnchorWatch(db)
    watch._schedule_quiet_check()
    timer = watch._quiet_timer

    watch.forget_baseline()

    timer.join(5)
    assert not timer.is_alive(), "forget_baseline() left the quiet timer running"
    timer.function(*timer.args)
    assert db.queued == []
    # Unlike close(), the watch keeps working for the new file.
    watch._schedule_quiet_check()
    assert watch._quiet_timer is not None
    watch.close()


def test_stopping_the_vault_cancels_a_pending_quiet_check(tmp_path, monkeypatch):
    # Vault.stop() closes the database; a timer outliving it would queue a
    # check onto a closed writer.
    monkeypatch.setattr(smart_score_invalidation, "EMBEDDING_QUIET_S", 3600)
    with Vault(image_root=str(tmp_path)) as vault:
        vault.smart_score_anchor_watch._schedule_quiet_check()
        timer = vault.smart_score_anchor_watch._quiet_timer
        assert timer.is_alive()

    timer.join(5)
    assert not timer.is_alive(), "Vault.stop() left the anchor watch's timer running"


# ── the routes ────────────────────────────────────────────────────────────────


def test_patching_a_rating_into_the_good_set_clears(server, client):
    rated, bystander = _seed(server, [{"score": 3}, {"score": None}])

    resp = client.patch(f"{API_PREFIX}/pictures/{rated}", json={"score": 5})
    assert resp.status_code == 200
    _drain(server)

    assert _all_cleared(server, [rated, bystander])


def test_undoing_a_rating_that_made_an_anchor_clears(server, client):
    rated, bystander = _seed(server, [{"score": 3}, {"score": None}])
    assert (
        client.patch(f"{API_PREFIX}/pictures/{rated}", json={"score": 5}).status_code
        == 200
    )
    _drain(server)
    _set(server, rated, smart_score=SEEDED_SMART_SCORE)
    _set(server, bystander, smart_score=SEEDED_SMART_SCORE)

    resp = client.post(f"{API_PREFIX}/operations/undo", json={})
    assert resp.status_code == 200
    _drain(server)

    assert server.vault.db.run_task(lambda s: s.get(Picture, rated).score) == 3
    assert _all_cleared(server, [rated, bystander])


def test_batch_rating_into_the_good_set_clears(server, client):
    rated, bystander = _seed(server, [{"score": None}, {"score": None}])

    resp = client.post(
        f"{API_PREFIX}/pictures/apply-scores",
        json={"scores": {str(rated): 5}, "only_unscored": False},
    )
    assert resp.status_code == 200
    assert resp.json()["reset_triggered"] is True
    _drain(server)

    assert _all_cleared(server, [rated, bystander])


def test_batch_rating_that_misses_a_full_anchor_set_keeps_scores(
    server, client, count_checks, monkeypatch
):
    monkeypatch.setattr(smart_score, "ANCHOR_LIMIT", 1)
    anchor, rated = _seed(server, [{"score": 5}, {"score": None}])
    checks = count_checks()

    resp = client.post(
        f"{API_PREFIX}/pictures/apply-scores",
        json={"scores": {str(rated): 4}, "only_unscored": False},
    )
    assert resp.status_code == 200
    # A check is queued (the rating could have moved the set); it finds no move.
    assert resp.json()["reset_triggered"] is True
    _drain(server)

    assert checks == [1]
    assert _all_kept(server, [anchor, rated])


def test_batch_rating_of_an_unembedded_picture_queues_no_check(
    server, client, count_checks
):
    rated, bystander = _seed(server, [{"score": None, "embedded": False}, {}])
    checks = count_checks()

    resp = client.post(
        f"{API_PREFIX}/pictures/apply-scores",
        json={"scores": {str(rated): 5}, "only_unscored": False},
    )
    assert resp.status_code == 200
    assert resp.json()["reset_triggered"] is False
    _drain(server)

    assert checks == []
    assert _all_kept(server, [rated, bystander])


# ── the scorer drops a batch computed from anchors that moved under it ───────


def test_a_batch_scored_before_an_anchor_clear_is_not_written(server):
    anchor, candidate = _seed(server, [{"score": 5}, {"score": None}])
    _set(server, candidate, smart_score=None)
    before = server.vault.db.run_task(
        lambda s: SmartScoreTask._input_signature(s, [candidate])
    )
    epoch = _epoch(server)  # read before the anchors, as SmartScoreTask does

    _set(server, anchor, score=3)  # the anchor leaves while the batch is scored
    _drain(server)
    assert _epoch(server) == epoch + 1

    persisted = server.vault.db.run_task(
        SmartScoreTask._persist_scores, {candidate: 0.9}, before, anchor_epoch=epoch
    )
    assert persisted == []
    assert _smart_scores(server, [candidate]) == [None]

    # Positive control: the same write against the current epoch lands.
    persisted = server.vault.db.run_task(
        SmartScoreTask._persist_scores,
        {candidate: 0.9},
        before,
        anchor_epoch=_epoch(server),
    )
    assert persisted == [candidate]
    assert _smart_scores(server, [candidate]) == [0.9]


# ── the scorer's anchor read ─────────────────────────────────────────────────


def test_the_scorer_reads_the_anchors_the_watch_compares(server, monkeypatch):
    # Ties on score and created_at put the anchor order and the id order apart,
    # so a read that lost its ORDER BY or its tie-break shows up here.
    monkeypatch.setattr(smart_score, "ANCHOR_LIMIT", 3)
    same_moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _seed(
        server,
        [
            {"score": 4},
            {"score": 5, "created_at": same_moment},
            {"score": 4},
            {"score": 5, "created_at": same_moment},
            {"score": 5},
            {"score": 1},
            {"score": 1, "created_at": same_moment},
            {"score": 1, "created_at": same_moment},
            {"score": 1},
            {"score": 5, "deleted": True},
            {"score": 1, "embedded": False},
        ],
    )

    def read(session):
        good, bad = smart_score.read_scoring_anchors(session)
        plain_good = session.exec(
            smart_score.good_anchor_query(Picture.image_embedding, Picture.score)
        ).all()
        plain_bad = session.exec(
            smart_score.bad_anchor_query(Picture.image_embedding, Picture.score)
        ).all()
        id_by_embedding = dict(
            session.exec(select(Picture.image_embedding, Picture.id)).all()
        )
        return (
            good,
            bad,
            plain_good,
            plain_bad,
            id_by_embedding,
            smart_score.anchor_signature(session),
        )

    good, bad, plain_good, plain_bad, id_by_embedding, signature = (
        server.vault.db.run_task(read)
    )
    limit = smart_score.ANCHOR_LIMIT

    assert len(plain_good) == len(plain_bad) == limit
    assert good[:limit] == list(plain_good)
    assert bad[:limit] == list(plain_bad)
    # The set the scorer reads is the set the watch compares.
    assert {(id_by_embedding[e], s) for e, s in good[:limit]} == signature[0]
    assert {(id_by_embedding[e], s) for e, s in bad[:limit]} == signature[1]
    # Fewer than _BUILTIN_MIN_* user anchors: the built-in ones follow them.
    builtin_good, builtin_bad = smart_score._load_builtin_anchors()
    assert builtin_good and builtin_bad
    assert len(good) - limit == len(builtin_good)
    assert all(a is b for a, b in zip(good[limit:], builtin_good))
    assert len(bad) - limit == len(builtin_bad)
    assert all(a is b for a, b in zip(bad[limit:], builtin_bad))


def test_the_anchor_read_fetches_embeddings_by_primary_key(server):
    # Selected inside the ordered, limited anchor query, every qualifying
    # picture's embedding goes through SQLite's sorter. Keyed, the query that
    # reads the embeddings looks rows up by id from the already-cut list.
    _seed(server, [{"score": 5}, {"score": 4}, {"score": 1}, {"score": None}])
    engine = server.vault.db._engine
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        collapsed = " ".join(statement.split()).lower()
        if collapsed.startswith("select picture.image_embedding"):
            statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", record)
    try:
        server.vault.db.run_task(smart_score.read_scoring_anchors)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert len(statements) == 2, statements  # one per anchor side

    def explain(session):
        return [
            [
                (row[1], row[-1])
                for row in session.connection().exec_driver_sql(
                    f"EXPLAIN QUERY PLAN {statement}", parameters
                )
            ]
            for statement, parameters in statements
        ]

    for plan in server.vault.db.run_task(explain):
        top_level = [detail for parent, detail in plan if parent == 0]
        assert any(
            detail.startswith("SEARCH picture USING INTEGER PRIMARY KEY")
            for detail in top_level
        ), f"the embeddings are not fetched by id; plan was {plan}"


# ── the session-hook extension point ─────────────────────────────────────────


def test_a_failing_session_hook_fails_its_task_and_spares_the_writer(server):
    failing = {"on": False}

    def hook(session):
        if failing["on"]:
            raise RuntimeError("hook refused the session")

    # Registered for the module's lifetime; it is inert once the flag is off.
    server.vault.db.add_session_hook(hook)
    failing["on"] = True
    try:
        future = server.vault.db.submit_task(lambda session: "ran")
        with pytest.raises(RuntimeError, match="hook refused the session"):
            future.result(10)
    finally:
        failing["on"] = False

    # The writer thread survived: the next task runs.
    assert server.vault.db.submit_task(lambda session: "ran").result(10) == "ran"
