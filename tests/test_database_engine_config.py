"""SQLite connection settings on the vault engine (issue #651, tier 1).

These assert the settings on a *real* connection handed out by
``VaultDatabase._engine``, not that some function was called. Every test here
fails against the pre-fix code, which built the engine with no ``connect_args``
(sqlite3's 5 s busy timeout) and set no ``cache_size`` (SQLite's 2 MiB default).

See docs/backend_architecture.md §13.
"""

import asyncio
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import CancelledError, Future, wait
from contextlib import closing

import pytest
from alembic import command
from fastapi import HTTPException

import pixlstash.database as database
from pixlstash.database import (
    SQLITE_BUSY_TIMEOUT_S,
    SQLITE_CACHE_SIZE_KIB,
    DatabaseUnavailableError,
    VaultDatabase,
    WriterState,
    create_configured_engine,
)
from pixlstash.services.restore.schema_upgrade import _alembic_config
from pixlstash.utils.sqlite_uri import sqlite_url
from tests.utils import URI_HOSTILE_FOLDER


@pytest.fixture
def vault_db(tmp_path):
    """A real VaultDatabase (engine + migrations) on a throwaway file."""
    db = VaultDatabase(str(tmp_path / "vault.db"))
    try:
        yield db
    finally:
        db.close()


def _pragma(engine, name):
    with engine.connect() as conn:
        return conn.exec_driver_sql(f"PRAGMA {name}").scalar()


def test_busy_timeout_is_configured_on_a_real_connection(vault_db):
    """The pool hands out connections with the 30 s busy timeout, not 5 s."""
    assert _pragma(vault_db._engine, "busy_timeout") == SQLITE_BUSY_TIMEOUT_S * 1000
    # Guard against the constant being lowered back to sqlite3's default.
    assert SQLITE_BUSY_TIMEOUT_S > 5


def test_cache_size_is_configured_on_a_real_connection(vault_db):
    """``init_database`` raises the page cache above SQLite's 2 MiB default."""
    cache_size = _pragma(vault_db._engine, "cache_size")
    assert cache_size == SQLITE_CACHE_SIZE_KIB
    # Negative means KiB rather than pages; -2000 (2 MiB) is the default we
    # are moving off, so anything at or above it is a no-op.
    assert cache_size < -2000


def test_settings_apply_to_every_pooled_connection(vault_db):
    """Not just the first connection: the pool opens up to 15 of them."""
    conns = [vault_db._engine.connect() for _ in range(5)]
    try:
        for conn in conns:
            assert (
                conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
                == SQLITE_BUSY_TIMEOUT_S * 1000
            )
            assert (
                conn.exec_driver_sql("PRAGMA cache_size").scalar()
                == SQLITE_CACHE_SIZE_KIB
            )
    finally:
        for conn in conns:
            conn.close()


def test_existing_pragmas_are_untouched(vault_db):
    """Regression guard: WAL, synchronous and FK enforcement still applied."""
    assert _pragma(vault_db._engine, "journal_mode") == "wal"
    assert _pragma(vault_db._engine, "synchronous") == 1  # NORMAL
    assert _pragma(vault_db._engine, "foreign_keys") == 1


def test_writer_waits_past_the_sqlite_default_instead_of_failing(vault_db):
    """Behavioural proof, not a PRAGMA readback.

    A competing connection holds the write lock for longer than sqlite3's 5 s
    default busy timeout. With the default the engine's write raises
    "database is locked"; with the configured 30 s it waits and succeeds.
    """
    hold_seconds = 6.0
    assert hold_seconds > 5, "must outlast sqlite3's default timeout to be a test"
    assert SQLITE_BUSY_TIMEOUT_S > hold_seconds, "must not outlast ours"

    db_path = vault_db._db_path
    assert os.path.exists(db_path)

    lock_taken = threading.Event()
    lock_failed = []

    def hold_write_lock():
        blocker = sqlite3.connect(db_path, timeout=1)
        try:
            blocker.execute("BEGIN EXCLUSIVE")
            lock_taken.set()
            time.sleep(hold_seconds)
            blocker.rollback()
        except Exception as exc:  # pragma: no cover - surfaced via lock_failed
            lock_failed.append(exc)
            lock_taken.set()
        finally:
            blocker.close()

    holder = threading.Thread(target=hold_write_lock, daemon=True)
    holder.start()
    try:
        assert lock_taken.wait(timeout=10), "blocker never acquired the write lock"
        assert not lock_failed, f"blocker failed to take the lock: {lock_failed}"

        started = time.perf_counter()
        with vault_db._engine.begin() as conn:
            # No-op write that still needs the write lock, on a table every
            # migrated vault has.
            conn.exec_driver_sql("UPDATE alembic_version SET version_num=version_num")
        elapsed = time.perf_counter() - started
    finally:
        holder.join(timeout=hold_seconds + 10)

    assert elapsed >= 5.0, (
        f"write returned after {elapsed:.2f}s; it cannot have waited out the "
        f"{hold_seconds}s lock, so the busy timeout is not in effect"
    )


# ---------------------------------------------------------------------------
# create_configured_engine: the single way engines are built (issue #709)
# ---------------------------------------------------------------------------


def test_configured_engine_defaults_match_the_vault_engine(tmp_path, vault_db):
    """A default helper engine is indistinguishable from the vault engine.

    Both halves of the configuration (``connect_args`` and the connect
    listener) have to arrive together; asserting them side by side is what
    catches one of the two being dropped.
    """
    engine = create_configured_engine(tmp_path / "configured.db")
    try:
        for name in ("journal_mode", "synchronous", "foreign_keys", "cache_size"):
            assert _pragma(engine, name) == _pragma(vault_db._engine, name), name
        assert _pragma(engine, "busy_timeout") == SQLITE_BUSY_TIMEOUT_S * 1000
        assert _pragma(engine, "journal_mode") == "wal"
        assert _pragma(engine, "foreign_keys") == 1
    finally:
        engine.dispose()


def test_non_wal_engine_keeps_every_other_setting(tmp_path):
    """``wal=False`` is the snapshot deviation and must deviate in that alone.

    The busy timeout, the page cache and FK enforcement still apply; only the
    journal changes, and ``synchronous`` falls back to FULL because NORMAL is
    only crash-safe under WAL.
    """
    engine = create_configured_engine(tmp_path / "snapshot.sqlite", wal=False)
    try:
        assert _pragma(engine, "journal_mode") == "delete"
        assert _pragma(engine, "synchronous") == 2  # FULL
        assert _pragma(engine, "foreign_keys") == 1
        assert _pragma(engine, "cache_size") == SQLITE_CACHE_SIZE_KIB
        assert _pragma(engine, "busy_timeout") == SQLITE_BUSY_TIMEOUT_S * 1000
    finally:
        engine.dispose()


def test_non_wal_engine_writes_leave_no_wal_sidecar(tmp_path):
    """The reason ``wal=False`` exists: the DB must stay a single file.

    A ``-wal``/``-shm`` companion is invisible to the snapshot paths that copy
    or compress the main file by name, so anything it still held would be lost.
    """
    db_path = tmp_path / "single-file.sqlite"
    engine = create_configured_engine(db_path, wal=False)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE t (x INTEGER)")
            conn.exec_driver_sql("INSERT INTO t VALUES (1)")
    finally:
        engine.dispose()

    for suffix in ("-wal", "-shm"):
        companion = tmp_path / (db_path.name + suffix)
        assert not companion.exists(), f"{companion} must not exist"

    # And the committed row really is in the main file. ``closing`` because
    # sqlite3's context manager commits but does not close the handle.
    with closing(sqlite3.connect(db_path)) as probe:
        assert probe.execute("SELECT x FROM t").fetchall() == [(1,)]


def test_foreign_key_enforcement_can_be_waived_explicitly(tmp_path):
    """The knob exists so a deviation is spelled out, never inherited."""
    engine = create_configured_engine(tmp_path / "no-fk.sqlite", foreign_keys=False)
    try:
        assert _pragma(engine, "foreign_keys") == 0
        assert _pragma(engine, "cache_size") == SQLITE_CACHE_SIZE_KIB
    finally:
        engine.dispose()


@pytest.mark.parametrize("kwargs", [{}, {"wal": False}, {"foreign_keys": False}])
def test_custom_sql_functions_are_registered_on_every_variant(tmp_path, kwargs):
    """The custom SQL functions are part of the configuration, not a side effect
    of the vault engine's listener."""
    engine = create_configured_engine(tmp_path / "fns.sqlite", **kwargs)
    try:
        with engine.connect() as conn:
            # An unregistered function raises "no such function" here; the
            # returned score itself is the search code's business, not ours.
            for sql in (
                "SELECT levenshtein('kitten sitting', 'sitting')",
                "SELECT levenshtein_with_id('kitten sitting', 'sitting', 1)",
            ):
                assert isinstance(conn.exec_driver_sql(sql).scalar(), float), sql
    finally:
        engine.dispose()


def test_vault_opens_and_migrates_in_a_folder_with_uri_characters(tmp_path):
    """A raw path was cut at ``?``: the vault was created beside the folder,
    under a truncated name, and the real file was left without Alembic."""
    folder = tmp_path / URI_HOSTILE_FOLDER
    folder.mkdir()
    db = VaultDatabase(str(folder / "vault.db"))
    try:
        assert _pragma(db._engine, "journal_mode") == "wal"
    finally:
        db.close()

    assert os.listdir(tmp_path) == [URI_HOSTILE_FOLDER]
    with closing(sqlite3.connect(folder / "vault.db")) as probe:
        assert probe.execute("SELECT version_num FROM alembic_version").fetchone()


def test_snapshot_upgrade_url_survives_uri_characters(tmp_path):
    """The snapshot upgrader hands Alembic a URL, which env.py builds an engine
    from; both the ConfigParser and SQLAlchemy's URL parser must pass it intact."""
    folder = tmp_path / URI_HOSTILE_FOLDER
    folder.mkdir()
    snapshot = folder / "snapshot.sqlite"

    command.upgrade(_alembic_config(sqlite_url(snapshot)), "head")

    assert os.listdir(tmp_path) == [URI_HOSTILE_FOLDER]
    with closing(sqlite3.connect(snapshot)) as probe:
        assert probe.execute("SELECT version_num FROM alembic_version").fetchone()


# ---------------------------------------------------------------------------
# The writer's settlement contract: every task the writer accepts has its
# future settled exactly once, whatever fails, and the writer either keeps
# running or fails closed (docs/backend_architecture.md §13).
# ---------------------------------------------------------------------------


@pytest.fixture
def writer_db(vault_db):
    """``vault_db`` with a scratch table the writer tests insert into."""
    with vault_db._engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE test_writer_rows (name TEXT)")
    return vault_db


def _insert(session, name):
    session.connection().exec_driver_sql(
        "INSERT INTO test_writer_rows (name) VALUES (?)", (name,)
    )


def _rows(db):
    """The committed rows, read on the caller's thread (never the writer's)."""
    return db.run_immediate_read_task(
        lambda session: [
            row[0]
            for row in session.connection().exec_driver_sql(
                "SELECT name FROM test_writer_rows"
            )
        ]
    )


def _assert_writer_still_runs(db):
    future = db.submit_task(lambda _session: "test-follower")
    assert future.result(timeout=5) == "test-follower"
    assert db.writer_status()["alive"] is True


def _hold_writer(db):
    """Park the writer inside a task until the returned event is set."""
    started, release = threading.Event(), threading.Event()

    def hold(_session):
        started.set()
        return "test-held" if release.wait(timeout=10) else "test-never-released"

    future = db.submit_task(hold)
    assert started.wait(timeout=5), "the writer never started the holding task"
    return future, release


def _inject_session_fault(monkeypatch, db, step, times=1):
    """Make the writer's session fail at ``step`` for its next ``times`` tasks.

    Only the writer thread is affected, so the test's own reads stay real, and
    the database underneath is a real SQLite file throughout. Returns the list
    the raised errors are appended to, in order.
    """
    raised = []

    def maybe_fail(at):
        if at != step or len(raised) == times:
            return
        if threading.current_thread() is not db._task_worker:
            return
        raised.append(OSError(f"test-{step}-fault-{len(raised)}"))
        raise raised[-1]

    class FaultySession(database.Session):
        def __init__(self, *args, **kwargs):
            maybe_fail("construct")
            super().__init__(*args, **kwargs)

        def rollback(self):
            maybe_fail("rollback")
            super().rollback()

        def close(self):
            maybe_fail("close")
            super().close()

    original_hooks = database._attach_session_hooks

    def faulty_hooks(session):
        maybe_fail("hooks")
        original_hooks(session)

    monkeypatch.setattr(database, "Session", FaultySession)
    monkeypatch.setattr(database, "_attach_session_hooks", faulty_hooks)
    return raised


class _FakeLocationGuard:
    """Stands in for the vault's location guard and records when it closes."""

    def __init__(self):
        self.closed = False

    def verify_after_open(self):
        """Nothing to verify: the test owns the folder."""

    def close(self):
        self.closed = True


def _start(target):
    """Run ``target`` on a daemon thread, so a deadlock fails a bounded join."""
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def guarded_db(tmp_path):
    """A real VaultDatabase with a ``_FakeLocationGuard``; yields both."""
    guard = _FakeLocationGuard()
    db = VaultDatabase(str(tmp_path / "vault.db"), location_guard=guard)
    yield db, guard
    # Bounded: a failing test may leave close() blocked holding _close_lock,
    # and an unbounded teardown close() would then hang the run.
    _start(db.close).join(timeout=2)


class _TestCallbackAbort(BaseException):
    """A BaseException that is neither an Exception nor anything stdlib raises."""


def test_callback_error_reaches_caller_and_writer_continues(writer_db):
    """The baseline: a failing callback's own exception reaches its caller,
    its write is rolled back, and the writer takes the next task."""
    error = ValueError("test-callback-error")

    def insert_then_raise(session):
        _insert(session, "test-rolled-back")
        raise error

    with pytest.raises(ValueError) as caught:
        writer_db.submit_task(insert_then_raise).result(timeout=5)
    assert caught.value is error
    assert "test-rolled-back" not in _rows(writer_db)
    _assert_writer_still_runs(writer_db)


@pytest.mark.parametrize("step", ["construct", "hooks"])
def test_session_setup_failure_settles_and_writer_continues(
    writer_db, monkeypatch, step
):
    """A session that cannot be opened fails that task, not the writer: three
    tasks in a row each get their own error and a fourth still runs."""
    raised = _inject_session_fault(monkeypatch, writer_db, step, times=3)
    futures = [writer_db.submit_task(lambda _session: "test-never") for _ in range(3)]

    for index, future in enumerate(futures):
        with pytest.raises(OSError) as caught:
            future.result(timeout=5)
        assert caught.value is raised[index]
    _assert_writer_still_runs(writer_db)
    failure = writer_db.writer_status()["last_session_failure"]
    assert (failure["step"], failure["error"]) == ("open", repr(raised[2]))


def test_rollback_failure_keeps_original_exception(writer_db, monkeypatch):
    """A rollback that fails must not replace the callback's exception, must
    not leave its write behind, and must not stop the writer."""
    raised = _inject_session_fault(monkeypatch, writer_db, "rollback")
    error = ValueError("test-callback-error")

    def insert_then_raise(session):
        _insert(session, "test-rolled-back")
        raise error

    with pytest.raises(ValueError) as caught:
        writer_db.submit_task(insert_then_raise).result(timeout=5)
    assert caught.value is error
    assert any(
        "Rolling back the session also failed" in note and "test-rollback-fault" in note
        for note in caught.value.__notes__
    )
    # The session was discarded, and SQLite drops an uncommitted transaction
    # with its connection.
    assert "test-rolled-back" not in _rows(writer_db)
    failure = writer_db.writer_status()["last_session_failure"]
    assert (failure["step"], failure["error"]) == ("rollback", repr(raised[0]))
    _assert_writer_still_runs(writer_db)


def test_close_failure_after_commit_returns_result(writer_db, monkeypatch):
    """The commit already happened, so the caller gets the result rather than
    an exception that would make a committed write look retryable."""
    raised = _inject_session_fault(monkeypatch, writer_db, "close")

    def insert_and_commit(session):
        _insert(session, "test-committed")
        session.commit()
        return "test-result"

    assert writer_db.submit_task(insert_and_commit).result(timeout=5) == "test-result"
    assert "test-committed" in _rows(writer_db)
    failure = writer_db.writer_status()["last_session_failure"]
    assert (failure["step"], failure["error"]) == ("close", repr(raised[0]))
    _assert_writer_still_runs(writer_db)


@pytest.mark.parametrize("kind", ["normal", "control"])
def test_base_exception_from_callback_settles(writer_db, kind):
    """asyncio's CancelledError is a BaseException, not an Exception."""
    error = asyncio.CancelledError("test-base-exception")

    def raise_base(*_args):
        raise error

    submit = (
        writer_db.submit_task if kind == "normal" else writer_db.submit_control_task
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        submit(raise_base).result(timeout=5)
    assert caught.value is error
    _assert_writer_still_runs(writer_db)


@pytest.mark.parametrize("kind", ["normal", "control"])
def test_cancel_before_start_never_executes(writer_db, kind):
    held, release = _hold_writer(writer_db)
    ran = threading.Event()

    def mark(*_args):
        ran.set()
        return "test-ran"

    submit = (
        writer_db.submit_task if kind == "normal" else writer_db.submit_control_task
    )
    try:
        future = submit(mark)
        assert future.cancel() is True
    finally:
        release.set()
    assert held.result(timeout=5) == "test-held"
    # The follower queues behind the cancelled task, so once it has run the
    # cancelled task's turn has passed.
    _assert_writer_still_runs(writer_db)
    assert not ran.is_set(), "a task cancelled before it started was executed"
    with pytest.raises(CancelledError):
        future.result(timeout=0)


def test_cancel_while_running_is_refused(writer_db):
    """A running write is never reported as cancelled: it may commit."""
    started, release = threading.Event(), threading.Event()

    def insert_and_wait(session):
        _insert(session, "test-running")
        started.set()
        release.wait(timeout=10)
        session.commit()
        return "test-finished"

    future = writer_db.submit_task(insert_and_wait)
    try:
        assert started.wait(timeout=5)
        assert future.cancel() is False
    finally:
        release.set()
    assert future.result(timeout=5) == "test-finished"
    assert "test-running" in _rows(writer_db)


def test_done_callback_base_exception_does_not_kill_writer(writer_db, caplog):
    held, release = _hold_writer(writer_db)

    def abort(_future):
        raise _TestCallbackAbort("test-done-callback")

    held.add_done_callback(abort)
    with caplog.at_level(logging.ERROR, logger="pixlstash.database"):
        release.set()
        assert held.result(timeout=5) == "test-held"
        _assert_writer_still_runs(writer_db)
    assert any(
        record.levelno == logging.ERROR
        and "done-callback" in record.getMessage()
        and "_hold_writer.<locals>.hold" in record.getMessage()
        for record in caplog.records
    )


def test_loop_failure_fails_closed(writer_db, monkeypatch, caplog):
    """If the writer loop itself breaks, nothing it accepted is left waiting
    and nothing new is accepted; reads, which never touch it, still work."""
    held, release = _hold_writer(writer_db)
    queued = [
        writer_db.submit_task(lambda _session: "test-queued-a"),
        writer_db.submit_control_task(lambda: "test-queued-b"),
    ]
    original_get = writer_db._task_queue.get
    fired = []

    def failing_get(*args, **kwargs):
        if not fired and threading.current_thread() is writer_db._task_worker:
            fired.append(True)
            raise RuntimeError("test-loop-fault")
        return original_get(*args, **kwargs)

    monkeypatch.setattr(writer_db._task_queue, "get", failing_get)
    with caplog.at_level(logging.INFO, logger="pixlstash.database"):
        release.set()
        assert held.result(timeout=5) == "test-held"
        for future in queued:
            with pytest.raises(DatabaseUnavailableError, match="test-loop-fault"):
                future.result(timeout=5)
        writer_db._task_worker.join(timeout=5)

    status = writer_db.writer_status()
    assert status["state"] == WriterState.FAILED
    assert status["alive"] is False
    assert status["failed_reason"] == "RuntimeError: test-loop-fault"
    refused = writer_db.submit_task(lambda _session: "test-refused")
    assert refused.done()
    with pytest.raises(DatabaseUnavailableError, match="test-loop-fault"):
        refused.result(timeout=0)
    assert writer_db.is_open
    assert _rows(writer_db) == []
    assert not any("already settled" in r.getMessage() for r in caplog.records)

    writer_db.close()
    assert writer_db.writer_status()["state"] == WriterState.CLOSED
    assert writer_db._engine is None


def test_close_with_running_task_defers_engine_release(tmp_path, monkeypatch):
    """A writer still inside a task holds a pooled connection: disposing the
    engine or closing the guard under it is the corruption the guard exists
    to prevent, so the writer releases both itself once the task returns."""
    monkeypatch.setattr(database, "_WRITER_JOIN_TIMEOUT_S", 0.2)
    guard = _FakeLocationGuard()
    db = VaultDatabase(str(tmp_path / "vault.db"), location_guard=guard)
    held, release = _hold_writer(db)
    try:
        db.close()
        assert db._engine is not None
        assert guard.closed is False
        assert db.writer_status()["state"] == WriterState.CLOSED
        assert db.writer_status()["current_task"].endswith("_hold_writer.<locals>.hold")
        with pytest.raises(DatabaseUnavailableError):
            db.run_immediate_read_task(lambda _session: "test-read")
        refused = db.submit_task(lambda _session: "test-refused")
        with pytest.raises(DatabaseUnavailableError):
            refused.result(timeout=0)
    finally:
        release.set()
    assert held.result(timeout=5) == "test-held"
    db._task_worker.join(timeout=5)
    assert not db._task_worker.is_alive()
    assert db._engine is None
    assert guard.closed is True


def test_close_waits_for_in_flight_immediate_read(writer_db):
    """close() must not dispose the engine under a read that is still running."""
    inside, release = threading.Event(), threading.Event()
    results = []

    def slow_read(_session):
        inside.set()
        release.wait(timeout=10)
        return "test-read"

    reader = threading.Thread(
        target=lambda: results.append(writer_db.run_immediate_read_task(slow_read))
    )
    closer = threading.Thread(target=writer_db.close)
    reader.start()
    try:
        assert inside.wait(timeout=5)
        closer.start()
        deadline = time.monotonic() + 5
        while (
            writer_db._engine_rwlock._writers_waiting == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert writer_db._engine_rwlock._writers_waiting == 1
        assert writer_db._engine is not None
    finally:
        release.set()
        reader.join(timeout=5)
        if closer.is_alive() or closer.ident is not None:
            closer.join(timeout=5)
    assert results == ["test-read"]
    assert writer_db._engine is None


def test_close_from_inside_an_immediate_read_returns(guarded_db):
    """close() issued inside a read cannot wait for that read: it returns, and
    the read releases the engine and the guard when it ends."""
    db, guard = guarded_db
    seen = {}

    def read_then_close(_session):
        db.close()
        seen["engine_after_close"] = db._engine is not None
        seen["guard_after_close"] = guard.closed
        return "test-read"

    results = []
    reader = _start(lambda: results.append(db.run_immediate_read_task(read_then_close)))
    reader.join(timeout=5)
    assert not reader.is_alive(), "close() inside an immediate read deadlocked"
    assert results == ["test-read"]
    assert seen == {"engine_after_close": True, "guard_after_close": False}
    assert db.writer_status()["state"] == WriterState.CLOSED
    assert db._engine is None
    assert guard.closed is True


def test_close_with_a_stuck_immediate_read_is_bounded(guarded_db, monkeypatch):
    """A read that outlives the timeout keeps the engine and the guard open
    and releases them itself; close() returns and later reads are refused."""
    monkeypatch.setattr(database, "_WRITER_JOIN_TIMEOUT_S", 0.2)
    db, guard = guarded_db
    inside, release = threading.Event(), threading.Event()
    results = []

    def slow_read(_session):
        inside.set()
        release.wait(timeout=10)
        return "test-read"

    reader = _start(lambda: results.append(db.run_immediate_read_task(slow_read)))
    try:
        assert inside.wait(timeout=5)
        closer = _start(db.close)
        closer.join(timeout=5)
        assert not closer.is_alive(), (
            "close() waited on an in-flight read without bound"
        )
        assert db._engine is not None
        assert guard.closed is False
        refused = []

        def late_read():
            try:
                db.run_immediate_read_task(lambda _session: "test-late")
            except DatabaseUnavailableError as exc:
                refused.append(exc)

        late = _start(late_read)
        late.join(timeout=5)
        assert not late.is_alive() and len(refused) == 1
    finally:
        release.set()
    reader.join(timeout=5)
    assert results == ["test-read"]
    assert db._engine is None
    assert guard.closed is True


def test_nested_immediate_read_does_not_deadlock_close(guarded_db):
    """A read callback that nests a second read while close() waits must not
    deadlock: the nested read is served or refused, never parked forever."""
    db, guard = guarded_db
    inside, go_nested = threading.Event(), threading.Event()
    nested = []

    def outer(_session):
        inside.set()
        go_nested.wait(timeout=10)
        try:
            nested.append(db.run_immediate_read_task(lambda _s: "test-inner"))
        except DatabaseUnavailableError as exc:
            nested.append(exc)
        return "test-outer"

    results = []
    reader = _start(lambda: results.append(db.run_immediate_read_task(outer)))
    try:
        assert inside.wait(timeout=5)
        closer = _start(db.close)
        deadline = time.monotonic() + 5
        while db._engine_rwlock._writers_waiting == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert db._engine_rwlock._writers_waiting == 1
    finally:
        go_nested.set()
    reader.join(timeout=5)
    closer.join(timeout=5)
    assert not reader.is_alive(), "nested immediate read deadlocked against close()"
    assert not closer.is_alive()
    assert results == ["test-outer"] and len(nested) == 1
    assert db._engine is None
    assert guard.closed is True


def test_close_settles_every_queued_task(guarded_db, monkeypatch):
    """Work queued behind a busy writer is refused by close(), never run and
    never left pending. Normal tasks sort after close()'s IMMEDIATE sentinel,
    so only the drain settles them; a control task is IMMEDIATE too and was
    queued first, so the writer takes it before the sentinel and refuses it."""
    db, guard = guarded_db
    held, release = _hold_writer(db)
    ran = []
    queued = [
        db.submit_task(lambda _session: ran.append("test-normal")),
        db.submit_task(
            lambda _session: ran.append("test-low"), priority=database.DBPriority.LOW
        ),
        db.submit_control_task(lambda: ran.append("test-control")),
    ]
    original_join = db._task_worker.join

    def release_then_join(*args, **kwargs):
        release.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(db._task_worker, "join", release_then_join)
    closer = _start(db.close)
    closer.join(timeout=8)
    assert not closer.is_alive()
    assert held.result(timeout=0) == "test-held"
    for future in queued:
        assert future.done(), "close() left an accepted task pending"
        assert isinstance(future.exception(timeout=0), DatabaseUnavailableError)
    assert ran == []
    assert db._engine is None
    assert guard.closed is True


def test_close_from_a_queued_tasks_done_callback_settles_every_task(
    guarded_db, monkeypatch
):
    """close()'s drain runs done-callbacks on the closing thread; one that calls
    close() again must not deadlock, and every other accepted task settles."""
    db, guard = guarded_db
    held, release = _hold_writer(db)
    queued = db.submit_task(lambda _session: "test-must-not-run")
    trailing = db.submit_task(lambda _session: "test-must-not-run-either")
    reentered = threading.Event()

    def close_again(_future):
        db.close()
        reentered.set()

    queued.add_done_callback(close_again)
    original_join = db._task_worker.join

    def release_then_join(*args, **kwargs):
        # The writer finishes the held task, takes close()'s IMMEDIATE
        # sentinel and exits, so the queued tasks are settled by the drain.
        release.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(db._task_worker, "join", release_then_join)
    closer = _start(db.close)
    closer.join(timeout=8)
    assert not closer.is_alive(), "re-entrant close() from a done-callback deadlocked"
    assert reentered.is_set()
    assert held.result(timeout=0) == "test-held"
    for future in (queued, trailing):
        assert future.done()
        assert isinstance(future.exception(timeout=0), DatabaseUnavailableError)
    assert db.writer_status()["state"] == WriterState.CLOSED
    assert db._engine is None
    assert guard.closed is True


def test_submit_racing_close_is_settled(writer_db, monkeypatch):
    """A submit that passed the open check while close() ran must still be
    settled: either run or refused, never stranded behind the drain."""
    original_put = writer_db._task_queue.put
    entered, proceed = threading.Event(), threading.Event()

    def gated_put(item, *args, **kwargs):
        if item.func is not None and not entered.is_set():
            entered.set()
            proceed.wait(timeout=10)
        return original_put(item, *args, **kwargs)

    monkeypatch.setattr(writer_db._task_queue, "put", gated_put)
    submitted = []
    submitter = threading.Thread(
        target=lambda: submitted.append(
            writer_db.submit_task(lambda _session: "test-raced")
        )
    )
    closer = threading.Thread(target=writer_db.close)
    submitter.start()
    try:
        assert entered.wait(timeout=5)
        closer.start()
        closer.join(timeout=0.5)
        close_waited_for_admission = closer.is_alive()
    finally:
        proceed.set()
        submitter.join(timeout=5)
        closer.join(timeout=5)

    future = submitted[0]
    done, _ = wait([future], timeout=5)
    assert future in done, "a task admitted while close() ran was never settled"
    if future.exception() is None:
        assert future.result() == "test-raced"
    else:
        assert isinstance(future.exception(), DatabaseUnavailableError)
    assert close_waited_for_admission
    assert not closer.is_alive()


def test_http_4xx_refusal_keeps_info_log(writer_db, caplog):
    """A 4xx refusal reaches the caller as the same object, logged at INFO."""
    refusal = HTTPException(status_code=409, detail="test-conflict")

    def refuse(_session):
        raise refusal

    with caplog.at_level(logging.INFO, logger="pixlstash.database"):
        with pytest.raises(HTTPException) as caught:
            writer_db.run_task(refuse)
    assert caught.value is refusal
    levels = [r.levelno for r in caplog.records if "test-conflict" in r.getMessage()]
    assert levels == [logging.INFO]


def test_failure_of_a_cancelled_future_is_logged(caplog):
    """Re-querying a cancelled future inside the handler raised again and the
    error line was never written."""
    future = Future()
    assert future.cancel()
    with caplog.at_level(logging.ERROR, logger="pixlstash.database"):
        with pytest.raises(CancelledError):
            VaultDatabase.result_or_throw(future)
    assert any(
        r.levelno == logging.ERROR
        and r.getMessage().startswith("Database task failed: CancelledError()")
        for r in caplog.records
    )


def test_writer_status_reports_the_task_in_hand(writer_db):
    held, release = _hold_writer(writer_db)
    try:
        queued = writer_db.submit_task(lambda _session: "test-queued")
        status = writer_db.writer_status()
    finally:
        release.set()
    assert status["state"] == WriterState.OPEN
    assert status["alive"] is True
    assert status["current_task"] == (f"{__name__}._hold_writer.<locals>.hold")
    assert status["current_task_seconds"] >= 0
    assert status["queue_depth"] == 1
    assert status["failed_reason"] is None
    assert held.result(timeout=5) == "test-held"
    assert queued.result(timeout=5) == "test-queued"
    assert writer_db.writer_status()["current_task"] is None
