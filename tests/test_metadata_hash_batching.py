"""The batched metadata-hash path must be byte-compatible and O(1) per flush.

``_after_flush_hash_updater`` runs on the single writer thread, inside the open
write transaction, and used to issue five SELECTs plus one UPDATE **per dirty
picture**. A 64-picture tag batch therefore added ~384 statements while holding
the SQLite write lock (issue #651). It now issues a constant number of
statements per flush.

Two things have to hold, and this file asserts both:

* **Byte-identical digests.** ``metadata_hash`` is persisted into snapshot files
  and compared against live rows (``services/snapshot_service.py``,
  ``services/restore/preview.py``). If the canonical string that is hashed
  changes by one byte, every picture in every existing snapshot starts
  comparing as "changed". ``test_batch_hash_equals_legacy_algorithm`` therefore
  compares against a frozen, verbatim copy of the pre-batching implementation
  rather than against the new code's own output.
* **Constant statement count.** ``test_flush_hash_update_is_constant_queries``
  measures the hook's own statements for N=2 and N=20 dirty pictures.

The frozen copy below deliberately still imports ``_HASH_SKIP_COLS`` from
``database``: which *columns* are hashed is a deliberate, migration-backed knob
(see ``0050_reset_metadata_hash_membership``), whereas how the values are
*serialised* is the thing that must never move. Freezing the skip set as well
would make this test fail for an intended column change while telling us
nothing about serialisation.
"""

import hashlib
import json
import re
import tempfile
import threading
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import (
    event,
    inspect as sa_inspect,
    select as sa_select,
    update as sa_update,
)

from pixlstash.database import (
    _HASH_SKIP_COLS,
    _compute_picture_metadata_hash,
    _compute_picture_metadata_hashes,
    _picture_needs_rehash,
)
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Project,
)
from pixlstash.db_models.snapshot import Snapshot
from pixlstash.db_models.tag import Tag
from pixlstash.server import Server
from pixlstash.services.operation_log_service import capture_state_in_session
from pixlstash.tasks.ocr_task import OcrTask
from pixlstash.tasks.text_score_task import TextScoreTask
from tests.utils import wipe_tables

API = "/api/v1"


# ---------------------------------------------------------------------------
# Frozen pre-batching implementation
# ---------------------------------------------------------------------------


def _legacy_compute_picture_metadata_hash(session, picture_id):
    """VERBATIM copy of ``_compute_picture_metadata_hash`` before batching.

    Do NOT "fix" or tidy this function. Its value is that it is the historical
    algorithm, including the fact that ``faces`` holds SQLAlchemy ``Row``
    objects which ``json.dumps(..., default=str)`` serialises as strings such as
    ``"(0, 0, '[1, 2, 3, 4]', 5)"`` rather than as JSON arrays. That accident is
    baked into every stored hash.
    """
    pic = session.get(Picture, picture_id)
    if pic is None:
        return None
    col_vals: dict = {}
    for col_attr in sa_inspect(type(pic)).column_attrs:
        col = col_attr.key
        if col in _HASH_SKIP_COLS:
            continue
        val = getattr(pic, col, None)
        if isinstance(val, np.ndarray):
            continue
        if hasattr(val, "isoformat"):
            val = val.isoformat()
        col_vals[col] = val
    tags = sorted(
        session.execute(sa_select(Tag.tag).where(Tag.picture_id == picture_id))
        .scalars()
        .all()
    )
    faces = sorted(
        session.execute(
            sa_select(
                Face.frame_index, Face.face_index, Face.bbox_, Face.character_id
            ).where(Face.picture_id == picture_id)
        ).all()
    )
    set_ids = sorted(
        session.execute(
            sa_select(PictureSetMember.set_id).where(
                PictureSetMember.picture_id == picture_id
            )
        )
        .scalars()
        .all()
    )
    project_ids = sorted(
        session.execute(
            sa_select(PictureProjectMember.project_id).where(
                PictureProjectMember.picture_id == picture_id
            )
        )
        .scalars()
        .all()
    )
    state = {
        "cols": col_vals,
        "tags": tags,
        "faces": faces,
        "sets": set_ids,
        "projects": project_ids,
    }
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """A vault with the WorkPlanner off.

    Background finders would otherwise run queries against the very tables the
    statement counter watches, on their own threads. The counter also filters by
    thread, so this is belt-and-braces rather than the only defence.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"disable_background_workers": True}, handle)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def clean_db(server):
    server.vault.db.run_task(
        wipe_tables,
        [
            Snapshot,
            Tag,
            Face,
            PictureSetMember,
            PictureProjectMember,
            PictureSet,
            Project,
            Character,
            Picture,
            PictureStack,
        ],
    )
    yield


@pytest.fixture(scope="module")
def client(server):
    """A client logged in as the owner, for the tests that go through a route."""
    client = TestClient(server.api)
    response = client.post(
        f"{API}/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert response.status_code == 200, response.text
    return client


# ---------------------------------------------------------------------------
# Byte-compatibility against the historical algorithm
# ---------------------------------------------------------------------------


def _seed_every_branch(server) -> dict:
    """Create pictures covering every branch of the hash input.

    Returns a mapping of a readable label to the picture id, plus the id of a
    picture that is deleted again so the "missing row" branch is covered.
    """

    def _do(session):
        ids = {}

        # 1. A picture with nothing attached: no tags, faces, sets or projects.
        bare = Picture(file_path="bare.jpg", filename="bare.jpg")
        session.add(bare)

        # 2. Tags only, deliberately inserted out of sort order.
        tagged = Picture(file_path="tagged.jpg", filename="tagged.jpg")
        session.add(tagged)

        # 3. Faces: one fully populated, one with a NULL character_id and a
        #    NULL bbox, so the sorted-tuple stringification is exercised with
        #    and without None members.
        faced = Picture(file_path="faced.jpg", filename="faced.jpg")
        session.add(faced)

        # 4. Set + project membership, multiple of each so the sort matters.
        member = Picture(file_path="member.jpg", filename="member.jpg")
        session.add(member)

        # 5. Datetime columns (``imported_at`` is hashed; ``created_at`` is in
        #    the skip set) and an ndarray column (``likeness_parameters``),
        #    which the digest must skip rather than serialise.
        dated = Picture(
            file_path="dated.jpg",
            filename="dated.jpg",
            imported_at=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
            created_at=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )
        session.add(dated)

        # 6. Everything at once.
        combo = Picture(file_path="combo.jpg", filename="combo.jpg")
        session.add(combo)

        # 7. Created only so it can be deleted, giving us an id with no row.
        doomed = Picture(file_path="doomed.jpg", filename="doomed.jpg")
        session.add(doomed)

        session.commit()

        character = Character(name="Ada")
        session.add(character)
        picture_set_b = PictureSet(name="B set")
        picture_set_a = PictureSet(name="A set")
        session.add(picture_set_a)
        session.add(picture_set_b)
        project_b = Project(name="B project")
        project_a = Project(name="A project")
        session.add(project_a)
        session.add(project_b)
        session.commit()

        session.add(Tag(picture_id=tagged.id, tag="zebra"))
        session.add(Tag(picture_id=tagged.id, tag="apple"))

        session.add(
            Face(
                picture_id=faced.id,
                frame_index=0,
                face_index=0,
                bbox=[10, 20, 30, 40],
                character_id=character.id,
            )
        )
        session.add(
            Face(
                picture_id=faced.id,
                frame_index=0,
                face_index=1,
                character_id=None,
            )
        )

        for set_id in (picture_set_b.id, picture_set_a.id):
            session.add(PictureSetMember(set_id=set_id, picture_id=member.id))
        for project_id in (project_b.id, project_a.id):
            session.add(
                PictureProjectMember(project_id=project_id, picture_id=member.id)
            )

        session.add(Tag(picture_id=combo.id, tag="combo-tag"))
        session.add(
            Face(
                picture_id=combo.id,
                frame_index=1,
                face_index=0,
                bbox=[1, 2, 3, 4],
                character_id=character.id,
            )
        )
        session.add(PictureSetMember(set_id=picture_set_a.id, picture_id=combo.id))
        session.add(PictureProjectMember(project_id=project_a.id, picture_id=combo.id))
        session.commit()

        ids = {
            "bare": bare.id,
            "tagged": tagged.id,
            "faced": faced.id,
            "member": member.id,
            "dated": dated.id,
            "combo": combo.id,
            "missing": doomed.id,
        }

        session.delete(session.get(Picture, doomed.id))
        session.commit()
        return ids

    return server.vault.db.run_task(_do)


def test_batch_hash_equals_legacy_algorithm(server):
    """The batched digest reproduces the pre-batching one, byte for byte.

    This is the whole point of the change: the hashes are persisted in snapshot
    files, so a serialisation difference would silently mark every picture in
    every existing snapshot as changed. Compared against a frozen copy of the
    old function, not against the new implementation's own output.
    """
    ids = _seed_every_branch(server)
    picture_ids = list(ids.values())

    def _compare(session):
        # ``no_autoflush`` so the in-memory ndarray assigned below stays
        # in-memory: ``likeness_parameters`` is a LargeBinary column whose live
        # value on a freshly built Picture is an ``np.ndarray``, and the skip
        # branch only fires while it still is one.
        with session.no_autoflush:
            pic = session.get(Picture, ids["dated"])
            pic.likeness_parameters = np.arange(11, dtype=np.float32)
            legacy = {
                pid: _legacy_compute_picture_metadata_hash(session, pid)
                for pid in picture_ids
            }
            batched = _compute_picture_metadata_hashes(session, picture_ids)
            single = {
                pid: _compute_picture_metadata_hash(session, pid) for pid in picture_ids
            }
        return legacy, batched, single

    legacy, batched, single = server.vault.db.run_immediate_read_task(_compare)

    for label, pid in ids.items():
        if label == "missing":
            assert legacy[pid] is None, "seed failed: doomed picture still exists"
            assert pid not in batched, (
                "a picture id with no row must have no entry, mirroring the "
                "None the per-picture helper returns"
            )
            assert single[pid] is None
            continue
        assert legacy[pid] is not None, f"{label}: legacy produced no hash"
        assert batched.get(pid) == legacy[pid], (
            f"{label}: batched digest differs from the historical algorithm "
            f"(legacy={legacy[pid]!r} batched={batched.get(pid)!r}). Every "
            "stored snapshot hash for this picture would now compare as changed."
        )
        assert single[pid] == legacy[pid], (
            f"{label}: the per-picture wrapper drifted from the historical algorithm"
        )

    # Guard against the digest being trivially constant across the branches.
    assert len(set(batched.values())) == len(batched), (
        "distinct pictures produced identical digests; the branches are not "
        "actually contributing to the hash"
    )


def test_batch_matches_legacy_for_the_stored_hashes(server):
    """The hook's persisted value still equals the historical algorithm's.

    ``test_batch_hash_equals_legacy_algorithm`` proves the two functions agree.
    This proves the value the after-flush hook actually wrote to the column is
    that same value, i.e. the executemany UPDATE persists the right hash to the
    right row rather than smearing one picture's digest across the batch.
    """
    ids = _seed_every_branch(server)
    picture_ids = [pid for label, pid in ids.items() if label != "missing"]

    def _read(session):
        stored = {
            pid: h
            for pid, h in session.execute(
                sa_select(Picture.id, Picture.metadata_hash).where(
                    Picture.id.in_(picture_ids)
                )
            )
        }
        legacy = {
            pid: _legacy_compute_picture_metadata_hash(session, pid)
            for pid in picture_ids
        }
        return stored, legacy

    stored, legacy = server.vault.db.run_immediate_read_task(_read)
    for pid in picture_ids:
        assert stored[pid] is not None, f"picture {pid} has no persisted hash"
        assert stored[pid] == legacy[pid], (
            f"picture {pid}: persisted hash {stored[pid]!r} != historical "
            f"algorithm {legacy[pid]!r}"
        )


# ---------------------------------------------------------------------------
# Statement cost of the after-flush hook
# ---------------------------------------------------------------------------


# The hook's reads all come from these five tables; its single write is the only
# statement in the codebase shaped ``UPDATE picture SET metadata_hash``.
_HASH_INPUT_TABLES = re.compile(
    r"\bfrom (picture|tag|face|picturesetmember|pictureprojectmember)\b"
)


class _HashHookStatementCounter:
    """Count only the statements attributable to the metadata-hash hook.

    Deliberately NOT a count of every statement on the engine. The listener
    fires for every thread, and any background activity would otherwise be
    measured as if it were the hook (see the same trap documented in
    ``tests/test_picture_sets_query_cost.py``). Two filters keep the count
    attributable:

    * **Thread.** ``bind_current_thread`` is called from inside the write task,
      so only statements issued on the single writer thread are counted, and
      only from that point on.
    * **Shape.** SELECTs against the five tables the digest reads from, plus the
      ``UPDATE picture SET metadata_hash`` the hook is alone in issuing. The
      flush's own ``INSERT INTO tag`` and any ``UPDATE picture SET <other>`` are
      excluded, so what remains grows only if the hook goes back to per-picture
      work.

    ``statements`` keeps a truncated copy for failure messages; ``full_statements``
    keeps the whole SQL, for assertions on which columns a SELECT fetches.
    """

    def __init__(self, server):
        self._engine = server.vault.db._engine
        self._thread_ident = None
        self.count = 0
        self.statements: list[str] = []
        self.full_statements: list[str] = []

    def bind_current_thread(self) -> None:
        self._thread_ident = threading.get_ident()

    def _on_execute(self, conn, cursor, statement, parameters, context, executemany):
        if self._thread_ident is None or threading.get_ident() != self._thread_ident:
            return
        collapsed = " ".join(statement.split()).lower()
        is_hash_write = collapsed.startswith("update picture set metadata_hash")
        is_hash_read = collapsed.startswith("select") and _HASH_INPUT_TABLES.search(
            collapsed
        )
        if is_hash_write or is_hash_read:
            self.count += 1
            self.statements.append(collapsed[:140])
            self.full_statements.append(collapsed)

    def __enter__(self):
        event.listen(self._engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *exc_info):
        event.remove(self._engine, "before_cursor_execute", self._on_execute)
        return False


def _create_pictures(server, count: int) -> list[int]:
    def _do(session):
        pictures = [
            Picture(file_path=f"cost_{i}.jpg", filename=f"cost_{i}.jpg")
            for i in range(count)
        ]
        session.add_all(pictures)
        session.commit()
        return [pic.id for pic in pictures]

    return server.vault.db.run_task(_do)


def _tag_pictures_in_one_flush(server, counter, picture_ids: list[int]) -> None:
    """Dirty every picture in a single flush by adding one tag to each."""

    def _do(session):
        counter.bind_current_thread()
        for pid in picture_ids:
            session.add(Tag(picture_id=pid, tag=f"batch-{pid}"))
        session.commit()

    server.vault.db.run_task(_do)


def test_flush_hash_update_is_constant_queries(server):
    """The hook costs the same number of statements for 2 pictures as for 20.

    The old shape was five SELECTs plus one UPDATE per dirty picture, so this
    count was exactly ``6 * N`` - 12 here versus 120 - all of it inside the open
    write transaction on the single writer thread.
    """
    small_ids = _create_pictures(server, 2)
    with _HashHookStatementCounter(server) as small:
        _tag_pictures_in_one_flush(server, small, small_ids)

    large_ids = _create_pictures(server, 20)
    with _HashHookStatementCounter(server) as large:
        _tag_pictures_in_one_flush(server, large, large_ids)

    assert small.count > 0, (
        "counted nothing at all; the filter no longer matches the hook's "
        f"statements. Seen: {small.statements}"
    )
    assert small.count == large.count, (
        f"the hook's statement count grew with the batch size: "
        f"{small.count} for 2 pictures, {large.count} for 20. The per-picture "
        f"loop is back.\n2-picture statements: {small.statements}\n"
        f"20-picture statements: {large.statements}"
    )
    # Five batched reads plus one executemany UPDATE. Asserted as an upper bound
    # rather than an equality so an extra eager-load statement is not a failure,
    # while a return to per-picture work (12 for N=2) still is.
    assert small.count <= 8, (
        f"expected a handful of batched statements, got {small.count}: "
        f"{small.statements}"
    )


def test_all_dirty_pictures_get_their_own_hash_in_one_flush(server):
    """Constant cost must not mean a shared or missing hash.

    The executemany UPDATE binds one parameter set per picture; a bug there
    would either leave hashes NULL or write one picture's digest to all rows.
    """
    picture_ids = _create_pictures(server, 20)

    def _tag(session):
        for pid in picture_ids:
            session.add(Tag(picture_id=pid, tag=f"unique-{pid}"))
        session.commit()

    server.vault.db.run_task(_tag)

    stored = server.vault.db.run_immediate_read_task(
        lambda session: dict(
            session.execute(
                sa_select(Picture.id, Picture.metadata_hash).where(
                    Picture.id.in_(picture_ids)
                )
            ).all()
        )
    )
    assert all(stored[pid] is not None for pid in picture_ids), (
        "some pictures dirtied in the batched flush have no hash"
    )
    assert len(set(stored.values())) == len(picture_ids), (
        "pictures dirtied in one flush share a hash; the executemany UPDATE is "
        "not binding per-row values"
    )


# ---------------------------------------------------------------------------
# Columns the writer-thread picture SELECTs fetch
# ---------------------------------------------------------------------------


# Large columns that neither the digest nor the operation capture reads. An
# image embedding alone is kilobytes per row, and both SELECTs run on the single
# writer thread inside the open write transaction.
_UNREAD_LARGE_COLUMNS = (
    "image_embedding",
    "text_embedding",
    "ocr_text",
    "ocr_words",
    "likeness_parameters",
)


def _picture_entity_selects(statements: list[str]) -> list[str]:
    """The full-entity ``SELECT ... FROM picture`` statements among *statements*.

    ``picture.description`` is a plain hashed column, so it is in the column list
    of every entity select and in none of the narrow Tag / Face / membership
    reads; requiring it also proves the recorder saw the whole column list.
    """
    return [
        sql
        for sql in statements
        if sql.startswith("select")
        and re.search(r"\bfrom picture\b", sql)
        and "picture.description" in sql
    ]


def test_hash_select_does_not_fetch_unread_large_columns(server):
    """The hook's picture SELECT leaves out the large columns the digest skips.

    The digest is unchanged by this (it never reads them, which the two
    byte-compatibility tests above pin); what changes is how many bytes the
    writer reads per dirty picture while it holds the write lock.
    """
    picture_ids = _create_pictures(server, 2)
    with _HashHookStatementCounter(server) as counter:
        _tag_pictures_in_one_flush(server, counter, picture_ids)

    entity_selects = _picture_entity_selects(counter.full_statements)
    assert entity_selects, (
        "the hook issued no picture entity SELECT, so this test checks nothing. "
        f"Seen: {counter.statements}"
    )
    for sql in entity_selects:
        fetched = [col for col in _UNREAD_LARGE_COLUMNS if f"picture.{col}" in sql]
        assert not fetched, (
            f"the metadata-hash SELECT fetches {fetched}, which the digest never "
            f"reads: {sql[:400]}"
        )


def test_operation_capture_select_does_not_fetch_unread_large_columns(server):
    """The operation-log capture reads scalar facets only, so it fetches no BLOBs.

    It runs twice for every recorded operation over every affected picture, on
    the writer thread.
    """
    picture_ids = _create_pictures(server, 2)

    with _HashHookStatementCounter(server) as counter:

        def _capture(session):
            counter.bind_current_thread()
            return capture_state_in_session(session, picture_ids)

        state = server.vault.db.run_task(_capture)

    assert sorted(state) == sorted(str(pid) for pid in picture_ids), (
        f"the capture lost a picture: {sorted(state)}"
    )
    entity_selects = _picture_entity_selects(counter.full_statements)
    assert entity_selects, (
        "the capture issued no picture entity SELECT, so this test checks "
        f"nothing. Seen: {counter.statements}"
    )
    for sql in entity_selects:
        fetched = [col for col in _UNREAD_LARGE_COLUMNS if f"picture.{col}" in sql]
        assert not fetched, (
            f"the operation capture SELECT fetches {fetched}, which it never "
            f"reads: {sql[:400]}"
        )


# ---------------------------------------------------------------------------
# Which picture writes recompute the hash
# ---------------------------------------------------------------------------
#
# The stored hash must always equal a from-scratch recompute; within that, a
# write that changes nothing the digest reads must not pay for one. Every test
# below checks the first property, and the cost tests check the second.


def _create_described_picture(server) -> int:
    def _do(session):
        pic = Picture(
            file_path="rehash.jpg",
            filename="rehash.jpg",
            description="test-description",
        )
        session.add(pic)
        session.commit()
        return pic.id

    return server.vault.db.run_task(_do)


def _stored_and_fresh(server, picture_id: int) -> tuple:
    """The persisted hash, and a from-scratch recompute of the same row."""

    def _read(session):
        stored = session.execute(
            sa_select(Picture.metadata_hash).where(Picture.id == picture_id)
        ).scalar_one()
        return stored, _compute_picture_metadata_hashes(session, [picture_id]).get(
            picture_id
        )

    return server.vault.db.run_immediate_read_task(_read)


def _counted_write(server, picture_id: int, mutate) -> _HashHookStatementCounter:
    """Load the picture, then count the hook's statements for *mutate* + commit.

    The picture is loaded before the counter is bound, so its own SELECT is not
    counted and a writer's ``session.get`` hits the identity map.
    """
    with _HashHookStatementCounter(server) as counter:

        def _do(session):
            pic = session.get(Picture, picture_id)
            counter.bind_current_thread()
            mutate(session, pic)
            session.commit()

        server.vault.db.run_task(_do)
    return counter


def test_same_value_assignment_neither_rehashes_nor_moves_the_hash(server):
    """Assigning a column its current value marks the picture dirty, nothing more."""
    picture_id = _create_described_picture(server)
    before, fresh_before = _stored_and_fresh(server, picture_id)
    assert before is not None and before == fresh_before, "seed is not settled"

    def _reassign(session, pic):
        pic.description = pic.description
        pic.score = pic.score
        session.add(pic)

    counter = _counted_write(server, picture_id, _reassign)

    assert counter.count == 0, (
        f"a same-value assignment recomputed the hash: {counter.statements}"
    )
    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == before == fresh


def _write_text_score(session, pic):
    TextScoreTask._persist_text_scores(session, [(pic.id, 0.42)])


def _write_ocr_text(session, pic):
    OcrTask._persist_text(session, {pic.id: [[{"text": "test-word"}]]})


def _write_text_embedding(session, pic):
    # The shape of ``TextEmbeddingTask.run``'s nested ``update_pictures``.
    db_pic = session.get(Picture, pic.id)
    db_pic.text_embedding = np.ones(8, dtype=np.float32).tobytes()
    session.add(db_pic)


@pytest.mark.parametrize(
    ("writer", "column"),
    [
        (_write_text_score, "text_score"),
        (_write_ocr_text, "ocr_text"),
        (_write_text_embedding, "text_embedding"),
    ],
    ids=["text_score", "ocr_text", "text_embedding"],
)
def test_derived_column_write_issues_no_hash_statements(server, writer, column):
    """A write to columns the digest skips leaves the hash alone, for free.

    These are the background writers that touch every picture in the library,
    each one a ``session.get`` loop on the writer thread.
    """
    picture_id = _create_described_picture(server)
    before, _ = _stored_and_fresh(server, picture_id)

    counter = _counted_write(server, picture_id, writer)

    written = server.vault.db.run_immediate_read_task(
        lambda session: session.execute(
            sa_select(getattr(Picture, column)).where(Picture.id == picture_id)
        ).scalar_one()
    )
    assert written is not None, f"the writer did not write {column}"
    assert counter.count == 0, (
        f"a {column}-only write recomputed the hash: {counter.statements}"
    )
    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == before == fresh


def test_mixed_derived_and_hashed_write_rehashes(server):
    picture_id = _create_described_picture(server)
    before, _ = _stored_and_fresh(server, picture_id)

    def _mixed(session, pic):
        pic.text_score = 0.42
        pic.description = "test-description-edited"

    _counted_write(server, picture_id, _mixed)

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == fresh, "a description edit beside a derived score left a stale hash"
    assert after != before


def _null_stored_hash(session, picture_id: int) -> None:
    # Core, on the table rather than the mapper, so no flush hook sees it: the
    # state a row is left in by a migration that resets the column.
    session.execute(
        sa_update(Picture.__table__)
        .where(Picture.__table__.c.id == picture_id)
        .values(metadata_hash=None)
    )


def test_null_stored_hash_is_initialised_by_a_derived_write(server):
    picture_id = _create_described_picture(server)

    def _reset(session):
        _null_stored_hash(session, picture_id)
        session.commit()

    server.vault.db.run_task(_reset)
    assert _stored_and_fresh(server, picture_id)[0] is None

    _counted_write(
        server, picture_id, lambda session, pic: setattr(pic, "text_score", 0.42)
    )

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after is not None, "a NULL hash survived a write to its picture"
    assert after == fresh


def test_unloaded_stored_hash_is_initialised_by_a_derived_write(server):
    """With the hash attribute unloaded the tracker cannot see it, so it recomputes."""
    picture_id = _create_described_picture(server)

    def _null_then_expire(session, pic):
        _null_stored_hash(session, picture_id)
        session.expire(pic, ["metadata_hash"])
        assert "metadata_hash" in sa_inspect(pic).unloaded
        pic.text_score = 0.42

    _counted_write(server, picture_id, _null_then_expire)

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after is not None, "a NULL hash survived a write to its picture"
    assert after == fresh


def test_explicit_hash_write_is_replaced_by_the_recompute(server):
    picture_id = _create_described_picture(server)

    _counted_write(
        server,
        picture_id,
        lambda session, pic: setattr(pic, "metadata_hash", "test-not-a-digest"),
    )

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == fresh, f"an assigned hash was persisted as is: {after!r}"


def test_stack_assignment_alone_rehashes(server):
    """``pic.stack = ...`` changes only relationship state until the flush.

    The ``stack_id`` column, which the digest reads, is filled in by the flush
    itself, so a tracker that looked only at column history would skip it.
    """
    picture_id = _create_described_picture(server)
    before, _ = _stored_and_fresh(server, picture_id)

    _counted_write(
        server, picture_id, lambda session, pic: setattr(pic, "stack", PictureStack())
    )

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == fresh, "joining a stack left a stale hash"
    assert after != before


def _create_picture_set(session) -> int:
    picture_set = PictureSet(name="test-set")
    session.add(picture_set)
    session.commit()
    return picture_set.id


def test_set_membership_added_from_the_set_side_rehashes(server):
    """``picture_set.members.append(pic)`` with ``pic.picture_sets`` unloaded.

    The link row is written by the relationship, not as a ``PictureSetMember``
    object, so the dirty picture is the only signal. Its pending backref change
    is not visible as attribute history, so a tracker that skips pictures with
    no history would leave the hash stale after a real membership change.
    """
    picture_id = _create_described_picture(server)
    set_id = server.vault.db.run_task(_create_picture_set)
    before, _ = _stored_and_fresh(server, picture_id)

    def _append(session, pic):
        assert "picture_sets" in sa_inspect(pic).unloaded, (
            "the collection is loaded, so this no longer exercises the backref case"
        )
        # Bound to a name: the identity map holds an unmodified object weakly,
        # so a bare ``session.get(...).members`` loses its parent mid-append.
        picture_set = session.get(PictureSet, set_id)
        picture_set.members.append(pic)

    _counted_write(server, picture_id, _append)

    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == fresh, "joining a set from the set side left a stale hash"
    assert after != before


def test_rolled_back_write_leaves_no_stale_hash(server):
    picture_id = _create_described_picture(server)
    before, _ = _stored_and_fresh(server, picture_id)

    def _edit_then_roll_back(session):
        pic = session.get(Picture, picture_id)
        pic.description = "test-description-rolled-back"
        session.flush()
        in_transaction = session.execute(
            sa_select(Picture.metadata_hash).where(Picture.id == picture_id)
        ).scalar_one()
        session.rollback()
        return in_transaction

    in_transaction = server.vault.db.run_task(_edit_then_roll_back)
    assert in_transaction != before, "the flush did not rehash inside the transaction"
    after, fresh = _stored_and_fresh(server, picture_id)
    assert after == before == fresh

    counter = _counted_write(
        server, picture_id, lambda session, pic: setattr(pic, "text_score", 0.42)
    )
    assert counter.count == 0, counter.statements
    assert _stored_and_fresh(server, picture_id) == (before, before)


class _UnmappedStandIn:
    """Not a mapped object, so reading its change state raises."""

    def __init__(self, picture_id: int) -> None:
        self.id = picture_id


def test_unreadable_change_state_recomputes_and_warns(caplog):
    """If the tracker cannot tell what changed, it pays for a recompute and says so."""
    with caplog.at_level("WARNING", logger="pixlstash.database"):
        assert _picture_needs_rehash(_UnmappedStandIn(7)) is True
    warnings = [r for r in caplog.records if r.name == "pixlstash.database"]
    assert any("picture 7" in r.getMessage() for r in warnings), (
        f"no warning naming the picture: {[r.getMessage() for r in warnings]}"
    )


# A Core write of a hashed input never reaches the flush hooks, and a later
# derived write (text score, OCR, embedding) does not recompute a non-NULL hash.
# So each Core writer below must set ``metadata_hash`` to NULL in its own
# statement; otherwise the picture matches a snapshot it no longer equals.


def _no_setup(session, picture_id: int) -> None:
    return None


def _rate(client, server, picture_id: int, _context) -> None:
    response = client.post(
        f"{API}/pictures/apply-scores",
        json={"scores": {str(picture_id): 5}, "only_unscored": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["updated_ids"] == [picture_id]


def _favourite(client, server, picture_id: int, _context) -> None:
    response = client.post(
        f"{API}/pictures/favourite",
        json={"picture_ids": [picture_id], "favourite": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["updated_ids"] == [picture_id]


def _reset_description(client, server, picture_id: int, _context) -> None:
    assert server.vault.reset_descriptions([picture_id]) == 1


def _join_two_projects(session, picture_id: int) -> int:
    """Member of two projects, pointing at the one that survives the delete."""
    deleted = Project(name="test-project-deleted")
    kept = Project(name="test-project-kept")
    session.add(deleted)
    session.add(kept)
    session.flush()
    session.add(PictureProjectMember(picture_id=picture_id, project_id=deleted.id))
    session.add(PictureProjectMember(picture_id=picture_id, project_id=kept.id))
    picture = session.get(Picture, picture_id)
    picture.project_id = kept.id
    session.commit()
    return deleted.id


def _delete_project(client, server, picture_id: int, project_id: int) -> None:
    response = client.delete(f"{API}/projects/{project_id}")
    assert response.status_code == 200, response.text


def _queue_for_character(session, picture_id: int) -> int:
    """Waiting for a face of the character, with none extracted yet."""
    character = Character(name="test-character")
    session.add(character)
    session.flush()
    picture = session.get(Picture, picture_id)
    picture.pending_character_id = character.id
    session.commit()
    return character.id


def _delete_character(client, server, picture_id: int, character_id: int) -> None:
    response = client.delete(f"{API}/characters/{character_id}")
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("setup", "write"),
    [
        (_no_setup, _rate),
        (_no_setup, _favourite),
        (_no_setup, _reset_description),
        (_join_two_projects, _delete_project),
        (_queue_for_character, _delete_character),
    ],
    ids=[
        "rating",
        "favourite",
        "description_reset",
        "project_delete",
        "character_delete",
    ],
)
def test_core_write_of_a_hashed_input_leaves_no_stale_hash(
    server, client, setup, write
):
    picture_id = _create_described_picture(server)
    context = server.vault.db.run_task(setup, picture_id)
    stored, fresh = _stored_and_fresh(server, picture_id)
    assert stored is not None and stored == fresh, "seed is not settled"
    snapshot_id = server.vault.snapshot_service.create_snapshot("MANUAL").id

    write(client, server, picture_id, context)

    stored, fresh = _stored_and_fresh(server, picture_id)
    assert stored in (None, fresh), "the write left a stale stored hash"

    derived = server.vault.db.run_task(
        TextScoreTask._persist_text_scores, [(picture_id, 0.42)]
    )
    assert derived, "the derived write found no picture"

    stored, fresh = _stored_and_fresh(server, picture_id)
    assert stored in (None, fresh), "a derived write kept a stale stored hash"
    compared = server.vault.restore_service.compare_hashes(snapshot_id, [picture_id])
    assert compared["changed_ids"] == [picture_id], compared


# ---------------------------------------------------------------------------
# Likeness parameters are derived, so neither the hash nor the preview reads them
# ---------------------------------------------------------------------------
#
# A new picture carries an in-memory array that the database hands back as
# bytes. Hashing the column made a picture's first hash, computed in the same
# session as its insert, one that no later recompute reproduces.


def test_import_shape_stores_the_hash_a_recompute_gives(server):
    """The import shape: insert, flush for the id, add a tag, commit."""

    def _import(session):
        pic = Picture(file_path="import.jpg", filename="import.jpg")
        session.add_all([pic])
        session.flush()
        session.add(Tag(picture_id=pic.id, tag="test-tag"))
        session.commit()
        return pic.id

    picture_id = server.vault.db.run_task(_import)

    stored, fresh = _stored_and_fresh(server, picture_id)
    assert stored is not None
    assert stored == fresh, "the import stored a hash no recompute reproduces"


def test_restore_preview_does_not_list_likeness_parameters(server):
    """A recomputed likeness vector is not a change the owner made."""
    picture_id = _create_described_picture(server)

    def _set_likeness(session):
        pic = session.get(Picture, picture_id)
        pic.likeness_parameters = np.ones(4, dtype=np.float32).tobytes()
        session.commit()

    server.vault.db.run_task(_set_likeness)
    snapshot_side = server.vault.db.run_immediate_read_task(
        lambda session: session.get(Picture, picture_id)
    )
    snapshot_side.likeness_parameters = np.zeros(4, dtype=np.float32).tobytes()
    snapshot_side.description = "test-description-before"

    changed = server.vault.restore_service._diff_picture(snapshot_side, True)

    assert changed == ["description"], changed
