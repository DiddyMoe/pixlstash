"""A write that leaves a person pending on a picture with faces resolves it.

A pending person (``Picture.pending_character_id``) waits for face extraction,
whose completion hook joins it to a face, but only for the pictures that
extraction just ran on. A write that sets one on a picture whose faces are
already there is never revisited, so each write's committing wrapper hands its
pictures to ``resolve_pending_people_with_faces`` once the write has committed
(docs/backend_architecture.md §24).

The two writers that can do that are a duplicate merge's metadata union and an
undo or redo replaying a picture's pending person. Covered here, on the
wrappers a route or a run calls:

* a stack verdict, a verdict batch, Keep cover only and an auto-stack slice
  (the union);
* a redo that replays the person onto faces extracted since the undo;
* a resolver failure after the commit, which is logged and does not turn the
  committed merge into an error.

One server serves the module; ``reset_library`` puts the library back as it
stood at start-up before every test.
"""

import gc
import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlmodel import SQLModel, select

from pixlstash.database import DBPriority
from pixlstash.db_models import Character, Face, Picture, PictureStack
from pixlstash.db_models.dedup import VERDICT_STACKED
from pixlstash.server import Server
from pixlstash.services import dedup_auto_stack_service
from pixlstash.services import dedup_tier_service as tiers
from pixlstash.services import dedup_verdict_service as verdicts
from pixlstash.services import folder_structure_commit_service
from pixlstash.services import keep_cover_only_service, operation_log_service
from pixlstash.services.dedup_tier_service import DedupScope, ScopeType, TierPolicy
from pixlstash.tasks.dedup_auto_stack_finder import DedupAutoStackFinder


def _vault_tables(server):
    """Every table the vault holds, children before parents."""
    present = set(sa.inspect(server.vault.db._engine).get_table_names())
    return [
        table
        for table in reversed(SQLModel.metadata.sorted_tables)
        if table.name in present
    ]


def _primary_keys(server, tables) -> dict[str, list[tuple]]:
    """Each table's rows by primary key, to compare tables by identity."""

    def read(session):
        return {
            table.name: sorted(
                tuple(row)
                for row in session.execute(sa.select(*table.primary_key.columns))
            )
            for table in tables
        }

    return server.vault.db.run_immediate_read_task(read)


@pytest.fixture(scope="module")
def env():
    """One server for the module, and its library's rows as they were at start-up.

    Building a Server costs more than all of this module's assertions together,
    so it is built once and ``reset_library`` returns the library to *baseline*
    before each test. It runs no task runner or planner
    (``disable_background_workers``): the tests stand in for both by hand.
    """
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    force_cpu = Server.DEFAULT_FORCE_CPU
    Server.DEFAULT_FORCE_CPU = True
    srv = Server(config_path)
    try:
        tables = _vault_tables(srv)
        baseline = _primary_keys(srv, tables)
        yield SimpleNamespace(
            server=srv,
            tables=tables,
            baseline=baseline,
            # Start-up state (settings, the schema version) is never wiped.
            wipe=[table for table in tables if not baseline[table.name]],
        )
    finally:
        srv.close()
        Server.DEFAULT_FORCE_CPU = force_cpu
        temp_dir.cleanup()
        gc.collect()


@pytest.fixture(autouse=True)
def reset_library(env):
    """Put the library back as it stood at start-up before each test.

    The tests seed a person, pictures, faces and stacks, scan, merge, undo and
    redo, then read the result by id, so a previous test's rows would be read
    as this one's: a second pair of duplicates in the queue, an operation the
    redo replays. No finder can hold a claim on a picture id that SQLite hands
    out again after the wipe, because the server has no scheduler to run one;
    that is asserted first, since a planner started here later would make the
    wipe unsafe (``BaseTaskFinder._claimed_picture_ids``,
    ``WorkPlanner._inflight_by_finder``).

    Every table that was empty at start-up is emptied, children before parents,
    so no DELETE leaves a row pointing at one already gone while the foreign
    keys stay enforced. Deriving the list from the schema keeps it complete.
    The check then compares every table's primary keys with start-up: a row
    left behind anywhere, or one added to a start-up table, fails here, before
    the test it would mislead.
    """
    server = env.server
    assert server.vault._task_runner is None and server.vault._work_planner is None, (
        "the module's server runs a scheduler, so a wipe could strand a claim"
    )
    assert "runs_background_work" not in vars(server.vault), (
        "a test's stand-in for the planner outlived the test"
    )

    def wipe(session):
        for table in env.wipe:
            session.execute(table.delete())
        session.commit()

    _run(server, wipe)
    assert _primary_keys(server, env.tables) == env.baseline, (
        "the wipe left the library unlike its start-up state"
    )
    yield


@pytest.fixture
def server(env):
    return env.server


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _seed_pair(server, *, folder="/vault/pair", second_has_face=True, stacked=False):
    """Two exact duplicates; the first has a face named Alice.

    The second has an unnamed face (extraction already ran on it) unless
    *second_has_face* is False. With *stacked* the pair is already one stack,
    the first its cover, for Keep cover only.
    """

    def insert(session):
        alice = Character(name="Alice")
        session.add(alice)
        stack = None
        if stacked:
            stack = PictureStack(name=None)
            session.add(stack)
        session.flush()
        ids = []
        for index in range(2):
            picture = Picture(
                file_path=f"{folder}/{index}.png",
                format="png",
                width=100,
                height=100,
                size_bytes=500,
                pixel_sha=f"{folder}-sha",
                stack_id=int(stack.id) if stack else None,
                stack_position=index if stack else None,
            )
            session.add(picture)
            session.flush()
            ids.append(int(picture.id))
        named = Face(
            picture_id=ids[0], face_index=0, character_id=alice.id, bbox=[0, 0, 40, 40]
        )
        session.add(named)
        unnamed_id = None
        if second_has_face:
            unnamed = Face(picture_id=ids[1], face_index=0, bbox=[0, 0, 30, 30])
            session.add(unnamed)
            session.flush()
            unnamed_id = int(unnamed.id)
        session.commit()
        return {
            "alice": int(alice.id),
            "named": ids[0],
            "other": ids[1],
            "unnamed_face": unnamed_id,
            "stack": int(stack.id) if stack else None,
        }

    return _run(server, insert)


def _scan(server, folder=None):
    scope = DedupScope(ScopeType.FOLDER, folder) if folder else None
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), scope)


def _signature(server, ids) -> str:
    """The queue row whose members are this test's pair, by picture id."""
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    pair = {ids["named"], ids["other"]}
    rows = [
        row
        for row in page
        if {member["picture_id"] for member in row["candidates"]} == pair
    ]
    assert len(rows) == 1, page
    return rows[0]["signature"]


def _person_state(server, ids):
    def read(session):
        other = session.get(Picture, ids["other"])
        face = session.get(Face, ids["unnamed_face"]) if ids["unnamed_face"] else None
        return other.pending_character_id, face.character_id if face else None

    return _run(server, read)


def _merge_single(server, signature):
    verdicts.apply_stack_verdict(server.vault, signature)


def _merge_batch(server, signature):
    verdicts.apply_verdict_batch(
        server.vault, [{"verdict": VERDICT_STACKED, "signature": signature}]
    )


@pytest.mark.parametrize("merge", [_merge_single, _merge_batch], ids=["one", "batch"])
def test_a_duplicate_merge_gives_its_person_to_faces_already_there(server, merge):
    ids = _seed_pair(server)
    _scan(server)

    merge(server, _signature(server, ids))

    pending, face_character = _person_state(server, ids)
    assert pending is None, "the merge's pending person was left for a start-up"
    assert face_character == ids["alice"], "the existing face takes the person"


def test_a_duplicate_merge_leaves_a_picture_not_yet_extracted_pending(server):
    ids = _seed_pair(server, second_has_face=False)
    _scan(server)

    _merge_single(server, _signature(server, ids))

    pending, _face = _person_state(server, ids)
    assert pending == ids["alice"], "no face rows yet: extraction resolves it"


def test_keep_cover_only_gives_its_person_to_faces_already_there(server):
    ids = _seed_pair(server, stacked=True)

    keep_cover_only_service.keep_cover_only(server.vault, stack_ids=[ids["stack"]])

    pending, face_character = _person_state(server, ids)
    assert pending is None, "the collapse's pending person was left for a start-up"
    assert face_character == ids["alice"]


def test_an_auto_stack_slice_gives_its_person_to_faces_already_there(
    server, monkeypatch
):
    folder = "/vault/auto"
    ids = _seed_pair(server, folder=folder)
    _scan(server, folder)
    # These tests stand in for the planner (as tests/test_dedup_auto_stack_runs.py
    # does), so the server is told it has one and the run is worked by hand.
    monkeypatch.setattr(server.vault, "runs_background_work", lambda: True)
    dedup_auto_stack_service.start_apply_run(
        server.vault, DedupScope(ScopeType.FOLDER, folder)
    )
    finder = DedupAutoStackFinder(server.vault)
    while (task := finder.find_task()) is not None:
        task.run()

    pending, face_character = _person_state(server, ids)
    assert pending is None, "the slice's pending person was left for a start-up"
    assert face_character == ids["alice"]


def test_a_redo_replaying_a_person_onto_faces_found_since_resolves_it(server):
    ids = _seed_pair(server, second_has_face=False)
    _scan(server)
    _merge_single(server, _signature(server, ids))
    assert _person_state(server, ids)[0] == ids["alice"], "set pending by the merge"

    operation_log_service.undo(server.vault)
    assert _person_state(server, ids)[0] is None

    def extract(session):
        face = Face(picture_id=ids["other"], face_index=0, bbox=[0, 0, 30, 30])
        session.add(face)
        session.commit()
        return int(face.id)

    ids["unnamed_face"] = _run(server, extract)

    body = operation_log_service.redo(server.vault)

    assert ids["other"] in body["picture_ids"]
    pending, face_character = _person_state(server, ids)
    assert pending is None, "the redo's pending person was left for a start-up"
    assert face_character == ids["alice"]


def test_a_resolver_failure_after_the_merge_commits_is_logged_not_raised(
    server, monkeypatch, caplog
):
    ids = _seed_pair(server)
    _scan(server)

    def broken(session, picture_ids):
        raise RuntimeError("example-injected resolver failure")

    monkeypatch.setattr(folder_structure_commit_service, "_pending_with_faces", broken)
    with caplog.at_level(logging.ERROR):
        result = verdicts.apply_stack_verdict(server.vault, _signature(server, ids))

    assert result.event_picture_ids, "the committed merge still answers"
    stacks = _run(
        server,
        lambda session: {
            session.get(Picture, pid).stack_id for pid in (ids["named"], ids["other"])
        },
    )
    assert len(stacks) == 1 and None not in stacks, "the merge committed"
    pending, _face = _person_state(server, ids)
    assert pending == ids["alice"], "left for the start-up pass"
    failures = [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR and "Duplicate merge" in record.getMessage()
    ]
    assert failures, [record.getMessage() for record in caplog.records]
    assert str(ids["other"]) in failures[0].getMessage()
    assert failures[0].exc_info is not None


def test_nothing_else_writes_a_pending_person_without_a_face_check():
    """Every unguarded writer of a pending person is resolved after its commit.

    A writer that sets ``pending_character_id`` only when the picture has no
    face row needs no resolve: extraction will run on it. The union and the
    op-log replay set it regardless, and are the two resolved above. A new
    writer elsewhere has to be added here and to the wrappers, or it strands
    a person until the next start-up.
    """
    root = Path(__file__).resolve().parents[1] / "pixlstash"
    found = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "migrations" not in path.parts
        and "db_models" not in path.parts
        and any(
            ".pending_character_id = " in line and not line.strip().endswith("= None")
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    )
    assert found == [
        "routes/characters_faces.py",  # guarded: only with no face row
        "services/dedup_verdict_service.py",  # the union: resolved after commit
        "services/folder_structure_commit_service.py",  # resolved after commit
        "services/move_reconciliation_service.py",  # guarded: only with no face row
        "services/operation_log_service.py",  # the replay: resolved after commit
        "tasks/picture_import_task.py",  # a new import: no face rows yet
    ], found
    # Every unguarded writer's wrappers resolve through the one helper.
    for module in (
        "services/dedup_verdict_service.py",
        "services/keep_cover_only_service.py",
        "services/operation_log_service.py",
        "tasks/dedup_auto_stack_task.py",
    ):
        text = (root / module).read_text(encoding="utf-8")
        assert "resolve_pending_people_with_faces(" in text, module


def test_the_merge_assigns_through_one_resolve_per_commit(server, monkeypatch):
    """The resolver is handed the merge's pictures once, after the commit."""
    ids = _seed_pair(server)
    _scan(server)
    calls = []
    original = folder_structure_commit_service.resolve_pending_people_with_faces

    def spy(vault, picture_ids, *, origin):
        state = vault.db.run_task(
            lambda session: session.exec(
                select(Picture.stack_id).where(Picture.id == ids["other"])
            ).first()
        )
        calls.append((sorted(picture_ids), origin, state))
        return original(vault, picture_ids, origin=origin)

    monkeypatch.setattr(verdicts, "resolve_pending_people_with_faces", spy)
    _merge_single(server, _signature(server, ids))

    assert len(calls) == 1, calls
    picture_ids, origin, stack_id_seen = calls[0]
    assert picture_ids == sorted([ids["named"], ids["other"]])
    assert origin == "Duplicate merge"
    assert stack_id_seen is not None, "called only after the merge committed"
