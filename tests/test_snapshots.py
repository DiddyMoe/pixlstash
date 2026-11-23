"""Tests for SnapshotService - creation, listing, deletion, and GFS retention."""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone

import pytest
import sqlmodel
from sqlalchemy import event, text, update as sa_update

from pixlstash.database import _compute_picture_metadata_hash
from pixlstash.db_models import Picture, Tag, User
from pixlstash.db_models.picture_likeness import (
    PictureLikeness,
    PictureLikenessFrontier,
    PictureLikenessQueue,
)
from pixlstash.db_models.snapshot import Snapshot
from pixlstash.server import Server
from pixlstash.services import snapshot_service as snapshot_service_module
from pixlstash.services.snapshot_service import (
    GFS_KEEP_MONTHLY,
    GFS_KEEP_OPPORTUNISTIC,
    GFS_KEEP_WEEKLY,
    SnapshotService,
)
from pixlstash.tasks.ensure_gfs_snapshot_task import EnsureGfsSnapshotTask
from pixlstash.utils.snapshot_compression import materialize_snapshot
from tests.utils import wipe_tables


@contextmanager
def _open_snapshot(server, cp):
    """Yield a sqlite3 connection to a snapshot, decompressing if compressed.

    Snapshots are stored zstd-compressed on disk, so tests that want to peek
    at the raw rows must materialize a plain .sqlite first.
    """
    abs_path = os.path.join(server.vault.image_root, cp.relative_path)
    tmp_dir = tempfile.mkdtemp(prefix="pixlstash_test_snap_")
    tmp_sqlite = os.path.join(tmp_dir, "snap.sqlite")
    try:
        materialize_snapshot(abs_path, tmp_sqlite)
        with closing(sqlite3.connect(tmp_sqlite)) as conn:
            yield conn
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        # Disable background workers so finders (QualityTask etc.) don't write
        # to `picture` between a test's last write and the restore call.
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def clean_db(server):
    """Wipe DB rows and snapshot files before each test."""

    server.vault.db.run_task(
        wipe_tables,
        [
            Snapshot,
            PictureLikeness,
            PictureLikenessQueue,
            PictureLikenessFrontier,
            Tag,
            Picture,
        ],
    )

    cp_dir = os.path.join(server.vault.image_root, "snapshots")
    if os.path.isdir(cp_dir):
        shutil.rmtree(cp_dir)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_db_snapshots(server) -> int:
    from sqlmodel import func, select

    return server.vault.db.run_immediate_read_task(
        lambda s: s.exec(select(func.count()).select_from(Snapshot)).one()
    )


def test_new_snapshot_contains_no_portable_identity_and_is_private(server):
    marker = "NEW-SNAPSHOT-PORTABLE-SECRET-19ab"

    def _seed(session):
        session.add(
            User(
                username=f"user-{marker}",
                password_hash=f"password-{marker}",
                hidden_tags=f'["{marker}"]',
            )
        )
        session.commit()

    server.vault.db.run_task(_seed)
    try:
        snapshot = server.vault.snapshot_service.create_snapshot("MANUAL")
        archive = os.path.join(server.vault.image_root, snapshot.relative_path)
        if os.name != "nt":
            # Windows synthesises st_mode from the read-only attribute (0o666
            # for any writable file), so no chmod can ever make this hold
            # there; access is the directory ACL's job. The identity
            # assertions below are the test's point and run everywhere.
            assert os.stat(archive).st_mode & 0o777 == 0o600
        with _open_snapshot(server, snapshot) as connection:
            for table in ("user", "usertoken", "guest_session", "guest_score"):
                assert connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone() == (0,)
            database_path = connection.execute("PRAGMA database_list").fetchone()[2]
            assert marker.encode() not in open(database_path, "rb").read()
    finally:
        server.vault.db.run_task(
            lambda session: (
                session.exec(text("DELETE FROM user")),
                session.commit(),
            )
        )


def _add_pictures(server, count: int = 3):
    def _do(session):
        for i in range(count):
            session.add(Picture(file_path=f"pic_{i}.jpg", filename=f"pic_{i}.jpg"))
        session.commit()

    server.vault.db.run_task(_do)


# ---------------------------------------------------------------------------
# create_snapshot: files and DB row are created
# ---------------------------------------------------------------------------


def test_create_manual_snapshot_creates_files_and_row(server):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL", label="my label")

    assert cp.id is not None
    assert cp.kind == "MANUAL"
    assert cp.label == "my label"

    abs_snapshot = os.path.join(server.vault.image_root, cp.relative_path)
    assert os.path.isfile(abs_snapshot), "Snapshot .sqlite must exist on disk"

    abs_manifest = os.path.join(server.vault.image_root, cp.manifest_relative_path)
    assert os.path.isfile(abs_manifest), "Manifest .json must exist on disk"


def test_snapshot_manifest_contains_expected_keys(server):
    _add_pictures(server, count=2)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    abs_manifest = os.path.join(server.vault.image_root, cp.manifest_relative_path)
    with open(abs_manifest) as fh:
        manifest = json.load(fh)

    assert "picture_count" in manifest
    assert "picture_ids" in manifest
    assert "schema_version" in manifest


def test_snapshot_picture_count_matches(server):
    _add_pictures(server, count=4)
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    assert cp.picture_count == 4

    abs_manifest = os.path.join(server.vault.image_root, cp.manifest_relative_path)
    with open(abs_manifest) as fh:
        manifest = json.load(fh)
    assert manifest["picture_count"] == 4
    assert len(manifest["picture_ids"]) == 4


# ---------------------------------------------------------------------------
# Hash sidecar: pictures whose metadata_hash is NULL
# ---------------------------------------------------------------------------


def _add_tagged_pictures(server, count: int) -> list[int]:
    """Add ``count`` pictures with two tags each; return their ids in order."""

    def _do(session):
        pictures = [
            Picture(file_path=f"tagged_{i}.jpg", filename=f"tagged_{i}.jpg")
            for i in range(count)
        ]
        session.add_all(pictures)
        session.flush()
        for picture in pictures:
            session.add(Tag(picture_id=picture.id, tag=f"test-tag-{picture.id}"))
            session.add(Tag(picture_id=picture.id, tag="test-shared"))
        session.commit()
        return [picture.id for picture in pictures]

    return server.vault.db.run_task(_do)


def _null_metadata_hashes(server, picture_ids) -> None:
    """Clear ``metadata_hash`` with a Core UPDATE, as a never-stamped row has it."""
    table = Picture.__table__

    def _do(session):
        session.execute(
            sa_update(table)
            .where(table.c.id.in_(list(picture_ids)))
            .values(metadata_hash=None)
        )
        session.commit()

    server.vault.db.run_task(_do)


def _record_manifest_statements(monkeypatch) -> list[str]:
    """Record every SQL statement sent through the engine the manifest opens."""
    statements: list[str] = []
    real_engine = snapshot_service_module.snapshot_engine

    def _recording_engine(path):
        engine = real_engine(path)

        def _record(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", _record)
        return engine

    monkeypatch.setattr(snapshot_service_module, "snapshot_engine", _recording_engine)
    return statements


def test_null_hashes_in_the_sidecar_match_the_per_picture_helper(server):
    """A NULL ``metadata_hash`` is computed for the sidecar, and the map is the
    one the per-picture helper gives: the stored hash where there is one, the
    helper's digest where there is not. More NULL rows than one batch,
    interleaved with stored ones, so a batch boundary that drops or misplaces
    an id shows."""
    picture_ids = _add_tagged_pictures(server, 660)
    null_ids = [pid for index, pid in enumerate(picture_ids) if index % 10]
    _null_metadata_hashes(server, null_ids)

    def _per_picture_map(session):
        stored = dict(
            session.exec(sqlmodel.select(Picture.id, Picture.metadata_hash)).all()
        )
        return stored, {
            str(pid): stored[pid] or _compute_picture_metadata_hash(session, pid)
            for pid in picture_ids
        }

    stored, expected = server.vault.db.run_immediate_read_task(_per_picture_map)
    assert {pid for pid, value in stored.items() if value is None} == set(null_ids)
    assert len(null_ids) > 500, "the NULL rows must span more than one batch"

    service = server.vault.snapshot_service
    snapshot = service.create_snapshot("MANUAL")

    assert service.load_picture_hashes(snapshot.id) == expected


def test_null_hash_statements_grow_per_batch_not_per_picture(server, monkeypatch):
    """Hashing the NULL rows for the sidecar costs a handful of statements per
    batch of 500 pictures, not five per picture: 600 and 1,200 NULL rows both
    stay at or under the bound (five fixed manifest reads plus five per batch;
    1,200 rows are three batches)."""
    max_statements = 20
    statements = _record_manifest_statements(monkeypatch)
    picture_ids = []
    counts = {}
    for count in (600, 1200):
        picture_ids += _add_tagged_pictures(server, count - len(picture_ids))
        _null_metadata_hashes(server, picture_ids)
        statements.clear()
        snapshot = server.vault.snapshot_service.create_snapshot("MANUAL")
        counts[count] = len(statements)
        hashes = server.vault.snapshot_service.load_picture_hashes(snapshot.id)
        assert set(hashes) == {str(pid) for pid in picture_ids}

    assert all(n <= max_statements for n in counts.values()), (
        f"statements per NULL-hash population: {counts}"
    )


def test_null_hashes_are_computed_in_batches_of_at_most_500(server, monkeypatch):
    """The batched hash helper holds every picture it was asked for until it
    returns, so the manifest hands it at most 500 ids at a time; every NULL row
    goes in exactly one batch, and no row with a stored hash goes in any."""
    picture_ids = _add_tagged_pictures(server, 1100)
    null_ids = [pid for index, pid in enumerate(picture_ids) if index % 10]
    _null_metadata_hashes(server, null_ids)
    real_helper = snapshot_service_module._compute_picture_metadata_hashes
    batches: list[list[int]] = []

    def _recording_helper(session, ids):
        batches.append(list(ids))
        return real_helper(session, ids)

    monkeypatch.setattr(
        snapshot_service_module, "_compute_picture_metadata_hashes", _recording_helper
    )
    server.vault.snapshot_service.create_snapshot("MANUAL")

    assert len(batches) > 1
    assert max(len(batch) for batch in batches) <= 500
    assert sorted(pid for batch in batches for pid in batch) == sorted(null_ids)


# ---------------------------------------------------------------------------
# list_snapshots and get_snapshot
# ---------------------------------------------------------------------------


def test_list_snapshots_returns_all(server):
    server.vault.snapshot_service.create_snapshot("MANUAL")
    server.vault.snapshot_service.create_snapshot("OPPORTUNISTIC")

    cps = server.vault.snapshot_service.list_snapshots()
    assert len(cps) == 2


def test_list_snapshots_ordered_newest_first(server):
    cp1 = server.vault.snapshot_service.create_snapshot("MANUAL", label="first")
    cp2 = server.vault.snapshot_service.create_snapshot("MANUAL", label="second")

    cps = server.vault.snapshot_service.list_snapshots()
    assert cps[0].id == cp2.id, "Newest snapshot should be first"
    assert cps[1].id == cp1.id


def test_get_snapshot_returns_correct_row(server):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    fetched = server.vault.snapshot_service.get_snapshot(cp.id)

    assert fetched is not None
    assert fetched.id == cp.id
    assert fetched.kind == "MANUAL"


def test_get_snapshot_nonexistent_returns_none(server):
    result = server.vault.snapshot_service.get_snapshot(9999)
    assert result is None


# ---------------------------------------------------------------------------
# delete_snapshot removes row and files
# ---------------------------------------------------------------------------


def test_delete_snapshot_removes_row_and_files(server):
    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    abs_snapshot = os.path.join(server.vault.image_root, cp.relative_path)
    abs_manifest = os.path.join(server.vault.image_root, cp.manifest_relative_path)
    abs_hashes = os.path.join(
        server.vault.image_root,
        server.vault.snapshot_service._hashes_relative_path(cp.manifest_relative_path),
    )
    assert os.path.isfile(abs_snapshot)
    assert os.path.isfile(abs_manifest)
    assert os.path.isfile(abs_hashes), "Hash sidecar should be written"

    deleted = server.vault.snapshot_service.delete_snapshot(cp.id)

    assert deleted is True
    assert server.vault.snapshot_service.get_snapshot(cp.id) is None
    assert not os.path.isfile(abs_snapshot), "Snapshot file should be removed"
    assert not os.path.isfile(abs_manifest), "Manifest file should be removed"
    assert not os.path.isfile(abs_hashes), "Hash sidecar should be removed"


def test_delete_nonexistent_snapshot_returns_false(server):
    result = server.vault.snapshot_service.delete_snapshot(9999)
    assert result is False


# ---------------------------------------------------------------------------
# snapshot_if_due: skips when a recent snapshot exists
# ---------------------------------------------------------------------------


def test_snapshot_if_due_creates_when_none_exist(server):
    result = server.vault.snapshot_service.snapshot_if_due("test")
    assert result is not None
    assert _count_db_snapshots(server) == 1


def test_snapshot_if_due_skips_when_recent(server):
    server.vault.snapshot_service.create_snapshot("OPPORTUNISTIC")
    assert _count_db_snapshots(server) == 1

    result = server.vault.snapshot_service.snapshot_if_due("test")

    assert result is None, "Should skip - a snapshot was just taken"
    assert _count_db_snapshots(server) == 1


# ---------------------------------------------------------------------------
# GFS retention: DAILY snapshots beyond the keep limit are pruned
# ---------------------------------------------------------------------------


def test_gfs_retention_prunes_oldest_daily(server):
    """Creating > GFS_KEEP_DAILY DAILY snapshots prunes the oldest.

    The prune happens at the tail of ``create_snapshot``. We backdate the
    created_at on the older snapshots so the prune has a deterministic
    notion of "oldest" (creating them back-to-back can collapse to the
    same microsecond).
    """
    from datetime import datetime, timedelta, timezone

    from pixlstash.services.snapshot_service import GFS_KEEP_DAILY

    cps = []
    for i in range(GFS_KEEP_DAILY + 2):
        cp = server.vault.snapshot_service.create_snapshot("DAILY")
        cps.append(cp)
        # Backdate this snapshot so the next create's prune sees clear ordering.

        def _backdate(session, cp_id=cp.id, i=i):
            row = session.get(Snapshot, cp_id)
            if row is not None:
                row.created_at = datetime.now(timezone.utc) - timedelta(
                    days=GFS_KEEP_DAILY + 2 - i
                )
                session.add(row)
                session.commit()

        server.vault.db.run_task(_backdate)

    # Trigger one more prune by creating a final DAILY snapshot.
    server.vault.snapshot_service.create_snapshot("DAILY")

    surviving = [
        s for s in server.vault.snapshot_service.list_snapshots() if s.kind == "DAILY"
    ]
    assert len(surviving) == GFS_KEEP_DAILY, (
        f"Expected exactly {GFS_KEEP_DAILY} DAILY snapshots after prune, "
        f"got {len(surviving)}"
    )

    # The two oldest of our original batch must have been pruned.
    surviving_ids = {s.id for s in surviving}
    assert cps[0].id not in surviving_ids
    assert cps[1].id not in surviving_ids


@pytest.mark.parametrize(
    "kind, keep",
    [
        ("WEEKLY", GFS_KEEP_WEEKLY),
        ("MONTHLY", GFS_KEEP_MONTHLY),
    ],
)
def test_gfs_retention_prunes_oldest_weekly_and_monthly(server, kind, keep):
    """The promised GFS caps for WEEKLY (4) and MONTHLY (12) prune the oldest
    once the keep limit is exceeded."""
    from datetime import datetime, timedelta, timezone

    cps = []
    for i in range(keep + 2):
        cp = server.vault.snapshot_service.create_snapshot(kind)
        cps.append(cp)

        def _backdate(session, cp_id=cp.id, i=i):
            row = session.get(Snapshot, cp_id)
            if row is not None:
                row.created_at = datetime.now(timezone.utc) - timedelta(
                    days=(keep + 2 - i) * 7
                )
                session.add(row)
                session.commit()

        server.vault.db.run_task(_backdate)

    # Trigger one more prune.
    server.vault.snapshot_service.create_snapshot(kind)

    surviving = [
        s for s in server.vault.snapshot_service.list_snapshots() if s.kind == kind
    ]
    assert len(surviving) == keep, (
        f"Expected exactly {keep} {kind} snapshots after prune, got {len(surviving)}"
    )
    surviving_ids = {s.id for s in surviving}
    assert cps[0].id not in surviving_ids, f"Oldest {kind} must be pruned"
    assert cps[1].id not in surviving_ids


# ---------------------------------------------------------------------------
# GFS retention: OPPORTUNISTIC snapshots are capped to GFS_KEEP_OPPORTUNISTIC
# ---------------------------------------------------------------------------


def test_gfs_retention_prunes_oldest_opportunistic(server):
    """OPPORTUNISTIC snapshots accumulate from safety-snapshot-before-restore
    and ``snapshot_if_due()``. Without a cap they grow unbounded.
    """
    from datetime import datetime, timedelta, timezone

    from pixlstash.services.snapshot_service import GFS_KEEP_OPPORTUNISTIC

    cps = []
    for i in range(GFS_KEEP_OPPORTUNISTIC + 2):
        cp = server.vault.snapshot_service.create_snapshot("OPPORTUNISTIC")
        cps.append(cp)

        def _backdate(session, cp_id=cp.id, i=i):
            row = session.get(Snapshot, cp_id)
            if row is not None:
                row.created_at = datetime.now(timezone.utc) - timedelta(
                    hours=GFS_KEEP_OPPORTUNISTIC + 2 - i
                )
                session.add(row)
                session.commit()

        server.vault.db.run_task(_backdate)

    # Trigger one more prune by creating a final OPPORTUNISTIC.
    server.vault.snapshot_service.create_snapshot("OPPORTUNISTIC")

    surviving = [
        s
        for s in server.vault.snapshot_service.list_snapshots()
        if s.kind == "OPPORTUNISTIC"
    ]
    assert len(surviving) == GFS_KEEP_OPPORTUNISTIC, (
        f"Expected exactly {GFS_KEEP_OPPORTUNISTIC} OPPORTUNISTIC after prune, "
        f"got {len(surviving)}"
    )
    surviving_ids = {s.id for s in surviving}
    assert cps[0].id not in surviving_ids
    assert cps[1].id not in surviving_ids


def test_gfs_retention_does_not_prune_manual(server):
    """MANUAL snapshots are user-curated and must never be auto-pruned, even
    when many of them exist."""
    cps = [
        server.vault.snapshot_service.create_snapshot("MANUAL", label=f"m{i}")
        for i in range(10)
    ]
    # Triggering another snapshot must not touch the MANUAL ones.
    server.vault.snapshot_service.create_snapshot("MANUAL", label="trigger")

    surviving_manual = [
        s for s in server.vault.snapshot_service.list_snapshots() if s.kind == "MANUAL"
    ]
    assert len(surviving_manual) == 11
    surviving_ids = {s.id for s in surviving_manual}
    for cp in cps:
        assert cp.id in surviving_ids, (
            f"MANUAL snapshot {cp.id} was pruned but MANUAL must never auto-prune"
        )


# ---------------------------------------------------------------------------
# Snapshot regenerable-BLOB stripping
# ---------------------------------------------------------------------------


def test_create_snapshot_keeps_regenerable_picture_columns(server):
    """Snapshots must now CARRY the expensive regenerable Picture blobs
    (CLIP image/text embeddings) and derived scores. They used to be
    stripped to save disk, but that forced a full GPU re-embedding pass on
    every restore; the whole archive is zstd-compressed instead, so keeping
    them is affordable and a restore comes back fully populated.

    Set the columns on a live picture, take a snapshot, then decompress the
    archive and verify those columns survived alongside user-editable fields.
    """
    from sqlalchemy import update as sa_update

    from pixlstash.db_models import Picture

    pic_id = _add_picture_with_blobs(server)

    def _populate_derived(session):
        session.execute(
            sa_update(Picture)
            .where(Picture.id == pic_id)
            .values(
                description="user-set",
                text_embedding=b"\x01" * 4096,
                image_embedding=b"\x02" * 4096,
                smart_score=0.42,
                text_score=0.31,
                aesthetic_score=0.55,
            )
        )
        session.commit()

    server.vault.db.run_task(_populate_derived)

    cp = server.vault.snapshot_service.create_snapshot("MANUAL")
    snap_path = os.path.join(server.vault.image_root, cp.relative_path)
    assert os.path.isfile(snap_path)
    assert snap_path.endswith(".sqlite.zst"), "Snapshot must be compressed on disk"

    with _open_snapshot(server, cp) as conn:
        row = conn.execute(
            "SELECT description, text_embedding, image_embedding, "
            "smart_score, text_score, aesthetic_score "
            "FROM picture WHERE id = ?",
            (pic_id,),
        ).fetchone()

    assert row is not None, "Snapshot must contain the picture row"
    desc, te, ie, ss, ts, aest = row
    assert desc == "user-set", (
        f"user-editable column must survive; got description={desc!r}"
    )
    assert te == b"\x01" * 4096, "text_embedding must be kept in the snapshot"
    assert ie == b"\x02" * 4096, "image_embedding must be kept in the snapshot"
    assert ss == 0.42, "smart_score must be kept in the snapshot"
    assert ts == 0.31, "text_score must be kept in the snapshot"
    assert aest == 0.55, "aesthetic_score must be kept in the snapshot"


def test_create_snapshot_drops_likeness_pipeline_tables(server):
    """The likeness graph (``picturelikeness``) plus its progress-tracking
    siblings (``picturelikenessqueue`` / ``picturelikenessfrontier``)
    must be dropped from the snapshot file. The graph is regenerable
    and O(N²); the progress tables are LIVE pipeline state that the
    restore replays from a fresh capture rather than from the snapshot.
    """
    pic_id = _add_picture_with_blobs(server)

    # Seed the three tables with live rows so we can prove they
    # disappear in the snapshot file.
    pic_id_b = _add_picture_with_blobs_named(server, "blobs_b.jpg")

    def _seed(session):
        a, b = sorted([pic_id, pic_id_b])
        session.add(
            PictureLikeness(
                picture_id_a=a, picture_id_b=b, likeness=0.77, metric="clip_cosine"
            )
        )
        session.add(PictureLikenessQueue(picture_id=pic_id))
        session.add(PictureLikenessFrontier(picture_id_a=pic_id, j_max=pic_id_b))
        session.commit()

    server.vault.db.run_task(_seed)

    cp = server.vault.snapshot_service.create_snapshot("MANUAL")

    with _open_snapshot(server, cp) as conn:
        likeness_n = conn.execute("SELECT COUNT(*) FROM picturelikeness").fetchone()[0]
        queue_n = conn.execute("SELECT COUNT(*) FROM picturelikenessqueue").fetchone()[
            0
        ]
        frontier_n = conn.execute(
            "SELECT COUNT(*) FROM picturelikenessfrontier"
        ).fetchone()[0]
        pic_n = conn.execute("SELECT COUNT(*) FROM picture").fetchone()[0]

    assert pic_n >= 2, "Snapshot must still contain the picture rows"
    assert likeness_n == 0, (
        f"picturelikeness must be empty in snapshot; got {likeness_n}"
    )
    assert queue_n == 0, (
        f"picturelikenessqueue must be empty in snapshot; got {queue_n}"
    )
    assert frontier_n == 0, (
        f"picturelikenessfrontier must be empty in snapshot; got {frontier_n}"
    )


# ---------------------------------------------------------------------------
# Capture and archiving run off the writer, and publish all or nothing
# ---------------------------------------------------------------------------


def _snapshot_files(server) -> list[str]:
    """Every file under the vault's snapshots dir, relative to it."""
    root = os.path.join(server.vault.image_root, "snapshots")
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(found)


def _snapshot_paths(server, snapshot) -> list[str]:
    """The archive and both sidecars a snapshot row points at."""
    root = server.vault.image_root
    return [
        os.path.join(root, snapshot.relative_path),
        os.path.join(root, snapshot.manifest_relative_path),
        os.path.join(
            root,
            SnapshotService._hashes_relative_path(snapshot.manifest_relative_path),
        ),
    ]


def _add_described_picture(server, filename: str, description: str) -> int:
    def _do(session):
        picture = Picture(
            file_path=filename, filename=filename, description=description
        )
        session.add(picture)
        session.commit()
        session.refresh(picture)
        return picture.id

    return server.vault.db.run_task(_do)


def _live_descriptions(server) -> dict[int, str]:
    return server.vault.db.run_immediate_read_task(
        lambda session: {
            picture.id: picture.description
            for picture in session.exec(sqlmodel.select(Picture)).all()
        }
    )


def _run_in_thread(target, name: str):
    """Run ``target`` on a daemon thread; return the thread and its outcome."""
    outcome: dict = {}

    def _run():
        try:
            outcome["result"] = target()
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_run, name=name, daemon=True)
    thread.start()
    return thread, outcome


def test_the_writer_keeps_serving_while_an_archive_is_compressed(server, monkeypatch):
    real_compress = snapshot_service_module.compress_snapshot
    compressing, release = threading.Event(), threading.Event()

    def _paused_compress(src, dst, *args, **kwargs):
        compressing.set()
        release.wait(timeout=30)
        return real_compress(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshot_service_module, "compress_snapshot", _paused_compress)
    thread, outcome = _run_in_thread(
        lambda: server.vault.snapshot_service.create_snapshot("MANUAL"),
        "test-snapshot",
    )
    try:
        assert compressing.wait(timeout=10), "the snapshot never reached compression"
        added_id = server.vault.db.submit_task(
            lambda session: _add_and_return_id(session, "during-compress.jpg")
        ).result(timeout=5)
        assert added_id in _live_descriptions(server)
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome
    with _open_snapshot(server, outcome["result"]) as conn:
        captured = {row[0] for row in conn.execute("SELECT id FROM picture")}
    assert added_id not in captured


def _add_and_return_id(session, filename: str) -> int:
    picture = Picture(file_path=filename, filename=filename)
    session.add(picture)
    session.commit()
    session.refresh(picture)
    return picture.id


def test_writes_during_capture_leave_a_consistent_pre_write_snapshot(server):
    """Edits, inserts and deletes commit on the writer at every progress check
    of the capture. The copy is a valid database equal to exactly one state the
    live database committed, and the writes committed after that state are
    absent from it.

    Where the first check lands is SQLite's business (on a fresh connection it
    can come while the schema is loaded, before the copy's read transaction
    opens), so the test writes at every check and asserts only that the copy
    is one committed state and not the last one."""
    kept_id = _add_described_picture(server, "kept.jpg", "before")
    dropped_id = _add_described_picture(server, "dropped.jpg", "before")
    db = server.vault.db
    max_writes = 50

    def _write_generation(session, generation: int):
        kept = session.get(Picture, kept_id)
        kept.description = f"edit-{generation}"
        session.add(kept)
        if generation == 1:
            session.delete(session.get(Picture, dropped_id))
        name = f"added-{generation}.jpg"
        session.add(Picture(file_path=name))
        session.commit()

    class _WriteDuringCapture(snapshot_service_module.SnapshotCancellation):
        def __init__(self):
            super().__init__()
            self.writes = 0
            self.errors = []

        def should_stop(self):
            if self.writes < max_writes:
                self.writes += 1
                generation = self.writes
                try:
                    db.submit_task(
                        lambda session: _write_generation(session, generation)
                    ).result(timeout=10)
                except Exception as exc:
                    self.errors.append(exc)
                    raise
            return super().should_stop()

    cancellation = _WriteDuringCapture()
    snapshot = server.vault.snapshot_service.create_snapshot(
        "MANUAL", cancellation=cancellation
    )

    assert cancellation.errors == []
    writes = cancellation.writes
    assert writes >= 1, "the capture never polled its cancellation"
    with _open_snapshot(server, snapshot) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        captured = {
            file_path: (picture_id, description)
            for picture_id, file_path, description in conn.execute(
                "SELECT id, file_path, description FROM picture"
            )
        }
    kept_description = captured["kept.jpg"][1]
    seen = 0 if kept_description == "before" else int(kept_description[5:])
    expected = {"kept.jpg"} | {f"added-{n}.jpg" for n in range(1, seen + 1)}
    if seen == 0:
        expected.add("dropped.jpg")
    assert set(captured) == expected, f"the copy mixes states around write {seen}"
    assert seen < writes, "every write landed in the copy; none ran during it"
    with open(
        os.path.join(server.vault.image_root, snapshot.manifest_relative_path)
    ) as fh:
        manifest = json.load(fh)
    assert set(manifest["picture_ids"]) == {pid for pid, _ in captured.values()}
    assert snapshot.picture_count == len(captured)
    live = server.vault.db.run_immediate_read_task(
        lambda session: {
            picture.file_path: picture.description
            for picture in session.exec(sqlmodel.select(Picture)).all()
        }
    )
    assert live["kept.jpg"] == f"edit-{writes}" and "dropped.jpg" not in live
    assert {f"added-{n}.jpg" for n in range(1, writes + 1)} <= set(live)


def test_cancel_inflight_ends_a_running_capture_and_leaves_nothing(server):
    _add_pictures(server, count=2)
    reading = threading.Event()

    class _HoldUntilCancelled(snapshot_service_module.SnapshotCancellation):
        def should_stop(self):
            reading.set()
            deadline = time.monotonic() + 15
            while not self.cancelled and time.monotonic() < deadline:
                time.sleep(0.01)
            return super().should_stop()

    service = server.vault.snapshot_service
    thread, outcome = _run_in_thread(
        lambda: service.create_snapshot("MANUAL", cancellation=_HoldUntilCancelled()),
        "test-snapshot",
    )
    try:
        assert reading.wait(timeout=10), "the capture never started reading"
        assert service.cancel_inflight(timeout=10) is True
        thread.join(timeout=10)
        assert not thread.is_alive(), "the capture outlived its cancellation"
    finally:
        thread.join(timeout=30)
    assert isinstance(
        outcome.get("error"), snapshot_service_module.SnapshotCancelledError
    ), outcome
    assert _snapshot_files(server) == []
    assert service.list_snapshots() == []


def test_cancelling_the_gfs_task_ends_its_snapshot(server, monkeypatch):
    real_compress = snapshot_service_module.compress_snapshot
    compressing = threading.Event()

    def _compress_until_cancelled(src, dst, *args, **kwargs):
        compressing.set()
        check_cancelled = kwargs.get("check_cancelled")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if check_cancelled is not None:
                check_cancelled()
            time.sleep(0.01)
        return real_compress(src, dst, *args, **kwargs)

    monkeypatch.setattr(
        snapshot_service_module, "compress_snapshot", _compress_until_cancelled
    )
    task = EnsureGfsSnapshotTask(server.vault, kind="DAILY")
    thread, outcome = _run_in_thread(task.run, "test-gfs-task")
    try:
        assert compressing.wait(timeout=10), "the task never reached compression"
        task.on_cancel()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the GFS task ignored its cancel"
    finally:
        thread.join(timeout=30)
    assert isinstance(
        outcome.get("error"), snapshot_service_module.SnapshotCancelledError
    ), outcome
    assert _snapshot_files(server) == []
    assert server.vault.snapshot_service.list_snapshots() == []


def test_a_snapshot_publishes_its_archive_and_sidecars_privately(server):
    snapshot = server.vault.snapshot_service.create_snapshot("MANUAL")
    paths = _snapshot_paths(server, snapshot)
    assert all(os.path.isfile(path) for path in paths)
    assert [name for name in _snapshot_files(server) if "partial" in name] == []
    if os.name != "nt":
        for path in paths:
            assert os.stat(path).st_mode & 0o777 == 0o600, path


def test_a_failed_compression_leaves_no_archive_or_sidecars(server, monkeypatch):
    def _fail_midway(src, dst, *args, **kwargs):
        with open(dst, "wb") as fh:
            fh.write(b"test-partial-archive")
        raise OSError("test-disk-full")

    monkeypatch.setattr(snapshot_service_module, "compress_snapshot", _fail_midway)
    with pytest.raises(OSError, match="test-disk-full"):
        server.vault.snapshot_service.create_snapshot("MANUAL")
    assert _snapshot_files(server) == []
    assert server.vault.snapshot_service.list_snapshots() == []


def test_a_failed_registration_removes_the_published_files(server, monkeypatch):
    def _refuse_row(**_fields):
        raise RuntimeError("test-registration-failure")

    with monkeypatch.context() as patch:
        patch.setattr(snapshot_service_module, "Snapshot", _refuse_row)
        with pytest.raises(RuntimeError, match="test-registration-failure"):
            server.vault.snapshot_service.create_snapshot("MANUAL")
    assert _snapshot_files(server) == []
    assert server.vault.snapshot_service.list_snapshots() == []


def test_a_failed_table_strip_publishes_nothing(server, monkeypatch):
    monkeypatch.setattr(
        SnapshotService,
        "_STRIP_TABLES_DELETE_ALL",
        ("picturelikeness", "test_missing_table"),
    )
    with pytest.raises(sqlite3.OperationalError, match="test_missing_table"):
        server.vault.snapshot_service.create_snapshot("MANUAL")
    assert _snapshot_files(server) == []
    assert server.vault.snapshot_service.list_snapshots() == []


def test_retention_deletes_files_only_after_its_commit(server, monkeypatch):
    service = server.vault.snapshot_service
    with monkeypatch.context() as patch:
        patch.setattr(service, "_apply_gfs_retention", lambda now: None)
        snapshots = [
            service.create_snapshot("OPPORTUNISTIC")
            for _ in range(GFS_KEEP_OPPORTUNISTIC + 1)
        ]

    def _backdate(session):
        for age, snapshot in enumerate(reversed(snapshots), start=1):
            row = session.get(Snapshot, snapshot.id)
            row.created_at = datetime.now(timezone.utc) - timedelta(hours=age)
            session.add(row)
        session.commit()

    server.vault.db.run_task(_backdate)
    oldest = snapshots[0]

    def _refuse_commit(self):
        raise RuntimeError("test-commit-failure")

    with monkeypatch.context() as patch:
        patch.setattr(sqlmodel.Session, "commit", _refuse_commit)
        with pytest.raises(RuntimeError, match="test-commit-failure"):
            service._apply_gfs_retention(datetime.now(timezone.utc))
    for snapshot in snapshots:
        for path in _snapshot_paths(server, snapshot):
            assert os.path.isfile(path), f"{path} deleted before the commit"
    assert {s.id for s in snapshots} <= {s.id for s in service.list_snapshots()}

    service._apply_gfs_retention(datetime.now(timezone.utc))
    assert service.get_snapshot(oldest.id) is None
    for path in _snapshot_paths(server, oldest):
        assert not os.path.exists(path), f"{path} survived its row"
    for snapshot in snapshots[1:]:
        assert all(os.path.isfile(p) for p in _snapshot_paths(server, snapshot))


def _add_picture_with_blobs(server) -> int:
    """Helper: add one picture and return its id."""

    def _do(session):
        p = Picture(file_path="blobs.jpg", filename="blobs.jpg")
        session.add(p)
        session.commit()
        session.refresh(p)
        return p.id

    return server.vault.db.run_task(_do)


def _add_picture_with_blobs_named(server, filename: str) -> int:
    """Helper: add one picture with the given filename, return its id."""

    def _do(session):
        p = Picture(file_path=filename, filename=filename)
        session.add(p)
        session.commit()
        session.refresh(p)
        return p.id

    return server.vault.db.run_task(_do)


def test_a_snapshot_cancelled_while_its_row_waits_for_the_writer_leaves_nothing(
    server, monkeypatch
):
    """A cancel that lands after the snapshot queued its row insert, while the
    writer is still busy with earlier work, ends the snapshot: no row and no
    files."""
    _add_pictures(server, count=2)
    service = server.vault.snapshot_service
    db = server.vault.db
    writer_busy, release_writer = threading.Event(), threading.Event()
    registering, row_queued = threading.Event(), threading.Event()
    real_acquire = service._acquire_registration
    real_submit = db.submit_task

    def _occupy_writer(session):
        writer_busy.set()
        release_writer.wait(timeout=30)

    def _busy_writer_then_acquire(cancellation):
        real_submit(_occupy_writer)
        assert writer_busy.wait(timeout=10), "the writer never took the blocker"
        real_acquire(cancellation)
        registering.set()

    def _note_queued(func, *args, **kwargs):
        future = real_submit(func, *args, **kwargs)
        if registering.is_set() and threading.current_thread().name == "test-snapshot":
            row_queued.set()
        return future

    monkeypatch.setattr(service, "_acquire_registration", _busy_writer_then_acquire)
    monkeypatch.setattr(db, "submit_task", _note_queued)
    cancellation = snapshot_service_module.SnapshotCancellation()
    thread, outcome = _run_in_thread(
        lambda: service.create_snapshot("MANUAL", cancellation=cancellation),
        "test-snapshot",
    )
    try:
        assert row_queued.wait(timeout=30), "the snapshot never queued its row"
        cancellation.cancel()
    finally:
        release_writer.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    assert isinstance(
        outcome.get("error"), snapshot_service_module.SnapshotCancelledError
    ), outcome
    assert _snapshot_files(server) == []
    assert service.list_snapshots() == []


def test_a_running_capture_holds_off_the_database_file_swap(server):
    """The restore file swap and ``close()`` take the engine lock's write side;
    while a capture copies the live file the swap must wait, and once the
    capture ends it goes."""
    _add_pictures(server, count=2)
    db = server.vault.db
    reading, release, swapped = threading.Event(), threading.Event(), threading.Event()

    class _HoldWhileReading(snapshot_service_module.SnapshotCancellation):
        def should_stop(self):
            reading.set()
            release.wait(timeout=15)
            return super().should_stop()

    def _swap():
        with db.exclusive_engine_access():
            swapped.set()

    snapshot_thread, outcome = _run_in_thread(
        lambda: server.vault.snapshot_service.create_snapshot(
            "MANUAL", cancellation=_HoldWhileReading()
        ),
        "test-snapshot",
    )
    swap_thread = None
    try:
        assert reading.wait(timeout=10), "the capture never started reading"
        swap_thread, _ = _run_in_thread(_swap, "test-swap")
        assert not swapped.wait(timeout=0.5), "the file swap ran during the capture"
    finally:
        release.set()
        snapshot_thread.join(timeout=30)
    assert not snapshot_thread.is_alive()
    assert "error" not in outcome, outcome
    assert swapped.wait(timeout=10), "the file swap never ran after the capture"
    swap_thread.join(timeout=10)
    assert not swap_thread.is_alive()
