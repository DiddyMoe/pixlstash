"""Keep cover only: collapsing a stack to its cover (docs/design/keep-cover-only.md).

The action keeps a stack's current cover and soft-deletes every other live
member to the Scrapheap. These cover the contract in **both** directions,
because over-blocking and over-collapsing are each their own regression:

* a hand-made stack has its tags unioned onto the cover **before** anything is
  deleted, and the tag is still there afterwards: the sharpest point in the
  design, because ``apply_metadata_union_in_session`` has only ever been called
  from the dedup verdict, so grid-made stacks have never been unioned;
* a stack with a locked-set member is refused **whole**, while its siblings in
  the **same request** still collapse;
* a stack whose only character link sits on a copy is skipped, counted **and
  named**, and one whose single character the union will propagate is not;
* the survivor keeps its ``stack_id``, no stack row is dissolved, and **one**
  undo restores the stack with its cover, positions and pre-union metadata;
* the preview's stack buckets are disjoint and sum to ``stacks_selected``, and
  the preview and the mutation agree over the same selection;
* both new routes are ``OWNER_ONLY`` at the central gate, a resource-scoped
  READ token is refused through the ``Authorization`` header *and* through
  ``?token=``, and the owner still reaches both.

Background workers are disabled and the pictures are inserted directly, so no
worker can rewrite metadata underneath the assertions.
"""

import gc
import json
import os
import re
import sqlite3
import tempfile
import typing
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as SASession
from sqlmodel import delete, select, update

from pixlstash.database import DBPriority
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Project,
    Tag,
    TagPrediction,
)
from pixlstash.routes.stacks import KeepCoverOnlyPreviewResponse, KeepCoverOnlyResponse
from pixlstash.server import Server
from pixlstash.utils import sql_chunking
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.services import (
    keep_cover_only_service,
    operation_log_service,
    set_lock_service,
    workflow_ghost_service,
)
from tests.authz_guard import no_spa_fallback  # noqa: F401

API = "/api/v1"
PREVIEW_URL = f"{API}/stacks/keep-cover-only/preview"
COLLAPSE_URL = f"{API}/stacks/keep-cover-only"
UNDO_URL = f"{API}/operations/undo"
RESTORE_URL = f"{API}/pictures/scrapheap/restore"

# The SPA module that splits a large selection into requests and merges their
# reports back into one (see the drift guard below).
KCO_REQUESTS_JS = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "utils"
    / "keepCoverOnlyRequests.js"
)

# The SPA catch-all answers unmatched GETs with 200, so a wrong URL could make a
# positive assertion vacuous. See tests/authz_guard.py.
pytestmark = pytest.mark.usefixtures("no_spa_fallback")


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _make_stack(server, members: list[dict]) -> tuple[int, list[int]]:
    """Insert one stack. ``members[0]`` becomes the cover (position 0).

    Each entry may carry ``tags`` (list[str]), ``score`` (int),
    ``size_bytes`` (int), ``reference_folder_id`` and ``deleted`` (bool: the
    member is in the Scrapheap, keeping its ``stack_id`` and position).
    """

    def insert(session):
        stack = PictureStack(name=None)
        session.add(stack)
        session.flush()
        picture_ids = []
        for position, spec in enumerate(members):
            picture = Picture(
                file_path=f"/vault/kco_{int(stack.id)}_{position}.png",
                format="png",
                width=1000,
                height=1000,
                size_bytes=spec.get("size_bytes", 1000),
                score=spec.get("score"),
                reference_folder_id=spec.get("reference_folder_id"),
                deleted=spec.get("deleted", False),
                stack_id=int(stack.id),
                stack_position=position,
            )
            session.add(picture)
            session.flush()
            for tag in spec.get("tags", []):
                session.add(Tag(picture_id=int(picture.id), tag=tag))
            picture_ids.append(int(picture.id))
        session.commit()
        return int(stack.id), picture_ids

    return _run(server, insert)


def _assign_character(server, picture_id: int, character_name: str) -> int:
    """Give *picture_id* a face bound to a (created-on-demand) character."""

    def assign(session):
        character = session.exec(
            select(Character).where(Character.name == character_name)
        ).first()
        if character is None:
            character = Character(name=character_name)
            session.add(character)
            session.flush()
        session.add(
            Face(
                picture_id=picture_id,
                frame_index=0,
                face_index=0,
                character_id=int(character.id),
                x=0,
                y=0,
                width=10,
                height=10,
            )
        )
        session.commit()
        return int(character.id)

    return _run(server, assign)


def _lock_set_over(client, server, name: str, picture_id: int) -> int:
    """Create a locked picture set containing exactly *picture_id*."""
    set_id = client.post(f"{API}/picture_sets", json={"name": name}).json()[
        "picture_set"
    ]["id"]

    def freeze(session):
        session.add(PictureSetMember(set_id=set_id, picture_id=picture_id))
        picture_set = session.get(PictureSet, set_id)
        picture_set.locked = True
        session.add(picture_set)
        session.commit()

    _run(server, freeze)
    return set_id


def _picture_rows(server, picture_ids: list[int]) -> dict[int, tuple]:
    """``{picture_id: (deleted, stack_id, stack_position, score)}``.

    Read in chunks, so it can check a fixture past :func:`_sqlite_variable_ceiling`.
    """

    def read(session):
        rows = [
            row
            for chunk in sql_chunking.chunked(list(picture_ids))
            for row in session.exec(
                select(
                    Picture.id,
                    Picture.deleted,
                    Picture.stack_id,
                    Picture.stack_position,
                    Picture.score,
                ).where(Picture.id.in_(chunk))
            ).all()
        ]
        return {
            int(pid): (bool(deleted), sid, pos, score)
            for pid, deleted, sid, pos, score in rows
        }

    return _run(server, read)


def _tags_of(server, picture_id: int) -> set[str]:
    def read(session):
        return {
            str(tag)
            for tag in session.exec(
                select(Tag.tag).where(Tag.picture_id == picture_id)
            ).all()
        }

    return _run(server, read)


def _env():
    """Owner cookie client plus a resource-scoped READ share token."""
    temp_dir = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(temp_dir.name, "images"), exist_ok=True)
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as handle:
        handle.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    Server.DEFAULT_FORCE_CPU = True
    server = Server(config_path)
    client = TestClient(server.api)
    assert (
        client.post(
            f"{API}/login",
            json={"username": "owner", "password": "example-owner-password"},
        ).status_code
        == 200
    )
    return temp_dir, client, server


def _scoped_token(client, server, picture_id: int) -> str:
    """A resource-scoped READ token over a set holding *picture_id*."""
    set_id = client.post(f"{API}/picture_sets", json={"name": "Scope"}).json()[
        "picture_set"
    ]["id"]

    def add(session):
        session.add(PictureSetMember(set_id=set_id, picture_id=picture_id))
        session.commit()

    _run(server, add)
    return client.post(
        f"{API}/users/me/token",
        json={
            "description": "scope read",
            "scope": "READ",
            "resource_type": "picture_set",
            "resource_id": set_id,
        },
    ).json()["token"]


def _teardown(temp_dir, server):
    server.close()
    temp_dir.cleanup()
    gc.collect()


def _rows_by_stack(body) -> dict:
    return {row["stack_id"]: row for row in body["stacks"]}


def _make_many_stacks(
    server,
    count: int,
    members: int,
    *,
    character_per_copy: bool = False,
    character_on_cover: bool = False,
) -> list[tuple[int, list[int]]]:
    """Insert *count* stacks of *members* pictures in ONE transaction.

    :func:`_make_stack` costs a DB task per stack and the variable-ceiling tests
    need over a thousand. Every copy (never the cover) carries the tag
    ``copy_only``, so every cover gains a tag. With *character_per_copy* each copy
    also gets a face bound to its own character, so the stack references
    ``members - 1`` characters, all of them only on copies. With
    *character_on_cover* each cover gets a face bound to a character of its own
    stack, the one-character case the metadata union passes on to the copies.
    """

    def insert(session):
        # One flush per phase, not per row: a flush per picture made this
        # fixture most of the test's run time.
        stacks = [PictureStack(name=None) for _ in range(count)]
        session.add_all(stacks)
        session.flush()
        pictures = [
            [
                Picture(
                    file_path=f"/vault/kco_{int(stack.id)}_{position}.png",
                    format="png",
                    width=1000,
                    height=1000,
                    size_bytes=1000,
                    stack_id=int(stack.id),
                    stack_position=position,
                )
                for position in range(members)
            ]
            for stack in stacks
        ]
        session.add_all(pic for members_of in pictures for pic in members_of)
        session.flush()
        copies = [int(pic.id) for members_of in pictures for pic in members_of[1:]]
        covers = [int(members_of[0].id) for members_of in pictures]
        session.add_all(Tag(picture_id=pid, tag="copy_only") for pid in copies)
        faces_on = (copies if character_per_copy else []) + (
            covers if character_on_cover else []
        )
        characters = [Character(name=f"KCO {pid}") for pid in faces_on]
        session.add_all(characters)
        session.flush()
        session.add_all(
            Face(
                picture_id=pid,
                frame_index=0,
                face_index=0,
                character_id=int(character.id),
                x=0,
                y=0,
                width=10,
                height=10,
            )
            for pid, character in zip(faces_on, characters)
        )
        session.commit()
        return [
            (int(stack.id), [int(pic.id) for pic in members_of])
            for stack, members_of in zip(stacks, pictures)
        ]

    return _run(server, insert)


@contextmanager
def _counting_vault_flushes(server):
    """Collect one entry per flush of a vault session while the block runs.

    A flush is where the vault pays for the metadata-hash, layout and score hooks,
    so a write path that flushes per picture costs those per picture however the
    rest of it is batched. Only sessions on the vault engine count, so a hub-side
    write during a request is not mistaken for the restore's.
    """
    engine = server.vault.db._engine
    flushes: list[int] = []

    def count(session, _flush_context):
        if session.get_bind() is engine:
            flushes.append(1)

    event.listen(SASession, "after_flush", count)
    try:
        yield flushes
    finally:
        event.remove(SASession, "after_flush", count)


@contextmanager
def _sqlite_variable_ceiling(server, limit: int = 999):
    """Pin every new vault connection to SQLite's pre-3.32 bound-parameter cap.

    Current builds allow 32766 parameters per statement, so a test library big
    enough to cross that would take minutes to build. At the historical 999 the
    same unbatched ``IN`` list fails with a fixture of a thousand rows, and the
    failure is the identical ``too many SQL variables``. The pin is proved to
    have taken effect before the block runs (a 1,000-id ``IN`` must fail), so a
    build that ignored ``setlimit`` could not pass these tests vacuously, and it
    is removed on the way out.
    """
    engine = server.vault.db._engine

    def pin(dbapi_conn, _record):
        dbapi_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)

    event.listen(engine, "connect", pin)
    engine.dispose()
    try:

        def over_the_ceiling(session):
            session.exec(select(Picture.id).where(Picture.id.in_(range(limit + 1))))

        with pytest.raises(OperationalError, match="too many SQL variables"):
            _run(server, over_the_ceiling)
        yield
    finally:
        event.remove(engine, "connect", pin)
        engine.dispose()


# ── the metadata union, the sharpest point ───────────────────────────────────


def test_hand_made_stack_is_unioned_onto_the_cover_before_anything_is_deleted():
    """A grid-made stack's tags land on the cover, and are not lost with the copy.

    ``apply_metadata_union_in_session`` has only ever been called from the dedup
    stack verdict, so a stack made by hand in the grid has never been unioned.
    Both halves are asserted: the tag **arrives** on the cover, and the cover
    still has it once the copy carrying it is in the Scrapheap.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(
            server,
            [
                {"tags": ["cover_only"], "score": 2},
                {"tags": ["copy_only"], "score": 5},
            ],
        )
        cover, copy_id = pics

        # Precondition: this is the un-unioned state the design measured.
        assert _tags_of(server, cover) == {"cover_only"}

        response = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["stacks_collapsed"] == 1
        assert body["pictures_moved"] == 1
        assert body["picture_ids_moved"] == [copy_id]
        assert body["covers_gaining_metadata"] == 1
        assert body["tags_added"] >= 1

        # The union ran BEFORE the delete: the copy's tag is on the cover, and
        # the copy is gone from every live view.
        assert _tags_of(server, cover) == {"cover_only", "copy_only"}
        rows = _picture_rows(server, pics)
        assert rows[cover][0] is False
        assert rows[copy_id][0] is True
        # Score lifts to the stack's best.
        assert rows[cover][3] == 5
    finally:
        _teardown(temp_dir, server)


def test_a_tag_removed_from_tagging_is_not_carried_onto_the_cover():
    """The preview and the union agree: a removed tag on a copy is no gain."""
    temp_dir, client, server = _env()
    try:
        removed, removed_pics = _make_stack(
            server, [{"tags": ["cover_only"]}, {"tags": ["Flux Chin"]}]
        )
        kept, kept_pics = _make_stack(
            server, [{"tags": ["cover_only"]}, {"tags": ["copy_only"]}]
        )
        server.vault.db.set_tag_exclusions(frozenset({"flux chin"}))
        selection = {"stack_ids": [removed, kept]}

        preview = client.post(PREVIEW_URL, json=selection)
        assert preview.status_code == 200, preview.text
        rows = _rows_by_stack(preview.json())
        assert (rows[removed]["cover_gains_tags"], rows[kept]["cover_gains_tags"]) == (
            False,
            True,
        )
        applied = client.post(COLLAPSE_URL, json=selection)
        assert applied.status_code == 200, applied.text
        assert _tags_of(server, removed_pics[0]) == {"cover_only"}
        assert _tags_of(server, kept_pics[0]) == {"cover_only", "copy_only"}
    finally:
        _teardown(temp_dir, server)


def test_survivor_keeps_its_stack_and_one_undo_restores_the_stack():
    """Acceptance criterion 6: nothing is dissolved and undo is a flag flip.

    The cover holds position 0, every member still points at the stack, the
    ``PictureStack`` row survives, and one ``Ctrl+Z`` puts the stack back with
    its positions **and** its pre-union metadata.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(
            server,
            [{"tags": ["a"]}, {"tags": ["b"]}, {"tags": ["c"]}],
        )
        cover = pics[0]

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["pictures_moved"] == 2
        batch_id = body["batch_id"]
        assert batch_id

        rows = _picture_rows(server, pics)
        # NOT dissolved and NOT detached: every member still points at the stack.
        assert {row[1] for row in rows.values()} == {stack_id}
        assert rows[cover][2] == 0, "the cover must hold position 0"
        assert all(rows[pid][0] for pid in pics[1:])
        assert _tags_of(server, cover) == {"a", "b", "c"}

        def stack_row(session):
            return session.get(PictureStack, stack_id) is not None

        assert _run(server, stack_row), "the stack row must survive the collapse"

        # One undo reverses the whole thing, the union included: the cover goes
        # back to carrying only its own tag.
        assert (
            client.post(f"{API}/operations/batches/{batch_id}/undo").status_code == 200
        )
        rows = _picture_rows(server, pics)
        assert not any(rows[pid][0] for pid in pics)
        assert {rows[pid][1] for pid in pics} == {stack_id}
        assert rows[cover][2] == 0
        assert _tags_of(server, cover) == {"a"}
    finally:
        _teardown(temp_dir, server)


def test_restoring_a_copy_from_the_scrapheap_returns_it_to_its_stack():
    """Acceptance criterion 7: a restored copy rejoins, it does not land loose.

    This is what leaving ``stack_id`` alone buys. ``restore_pictures`` already
    clears ``deleted_at`` and calls ``normalize_stack_positions``, so the copy
    comes back behind its cover rather than as a loose picture.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}, {}])

        assert (
            client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).status_code == 200
        )
        assert (
            client.post(RESTORE_URL, json={"picture_ids": [pics[1]]}).status_code == 200
        )

        rows = _picture_rows(server, pics)
        assert rows[pics[1]][0] is False
        assert rows[pics[1]][1] == stack_id
        # The cover is still the leader; the restored copy sorts behind it.
        assert rows[pics[0]][2] == 0
        assert rows[pics[1]][2] == 1
        # The copy that was NOT restored stays in the Scrapheap, still stacked.
        assert rows[pics[2]][0] is True
        assert rows[pics[2]][1] == stack_id
    finally:
        _teardown(temp_dir, server)


# ── locked sets refuse the WHOLE stack, siblings proceed ─────────────────────


def test_locked_member_refuses_the_whole_stack_while_siblings_collapse():
    """One frozen member refuses its stack entirely, and only its stack.

    A partial collapse is the worst available outcome (some copies gone, the
    stack still there, no visible reason), so the refusal is whole-stack. It is
    also *local*: the unlocked sibling in the SAME request still collapses.
    """
    temp_dir, client, server = _env()
    try:
        locked_stack, locked_pics = _make_stack(server, [{}, {}])
        free_stack, free_pics = _make_stack(server, [{}, {}])
        # Freeze a NON-cover member: the lock has to reach the whole stack
        # through the sibling, not only through the row the caller names.
        set_id = _lock_set_over(client, server, "Frozen", locked_pics[1])

        body = client.post(
            COLLAPSE_URL, json={"stack_ids": [locked_stack, free_stack]}
        ).json()

        assert body["stacks_collapsed"] == 1
        assert body["stack_ids_collapsed"] == [free_stack]
        assert body["picture_ids_moved"] == [free_pics[1]]
        skipped = body["stacks_skipped_locked"]
        assert [row["stack_id"] for row in skipped] == [locked_stack]
        assert [entry["id"] for entry in skipped[0]["locked_sets"]] == [set_id]
        assert skipped[0]["copy_picture_ids"] == []

        rows = _picture_rows(server, locked_pics + free_pics)
        # Not one member of the locked stack moved.
        assert not any(rows[pid][0] for pid in locked_pics)
        # The sibling did.
        assert rows[free_pics[1]][0] is True
    finally:
        _teardown(temp_dir, server)


# ── characters: skipped, counted, named, and not over-blocked ───────────────


def test_character_only_on_a_copy_skips_the_stack_and_names_it():
    """A link that would be destroyed skips the stack, and the skip names it."""
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}])
        # Two characters, so the union refuses to guess; the second sits only on
        # the copy, which is exactly the link a collapse would destroy.
        _assign_character(server, pics[0], "Cover Character")
        stranded = _assign_character(server, pics[1], "Copy Character")

        preview = client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).json()
        assert preview["stacks_eligible"] == 0
        assert preview["stacks_skipped_character_on_copy"] == 1
        row = _rows_by_stack(preview)[stack_id]
        assert row["skip_reason"] == keep_cover_only_service.SKIP_CHARACTER_ON_COPY
        assert row["lost_characters"] == [
            {"id": stranded, "name": "Copy Character", "picture_ids": [pics[1]]}
        ]

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["stacks_collapsed"] == 0
        assert body["pictures_moved"] == 0
        assert [r["stack_id"] for r in body["stacks_skipped_character_on_copy"]] == [
            stack_id
        ]
        assert not any(_picture_rows(server, pics)[pid][0] for pid in pics)
    finally:
        _teardown(temp_dir, server)


def test_a_single_character_on_a_copy_is_propagated_not_skipped():
    """Over-blocking is its own regression: the union handles the one-character case.

    With exactly one character across the stack the union assigns it to the
    cover (through ``pending_character_id``, never a fabricated face), so
    nothing is lost and the stack must still collapse.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}])
        character_id = _assign_character(server, pics[1], "Only Character")

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["stacks_collapsed"] == 1
        assert body["stacks_skipped_character_on_copy"] == []

        def read_pending(session):
            return session.get(Picture, pics[0]).pending_character_id

        assert _run(server, read_pending) == character_id
    finally:
        _teardown(temp_dir, server)


# ── the dry run: disjoint buckets, and agreement with the mutation ───────────


def test_preview_buckets_are_disjoint_and_sum_to_the_total():
    """Acceptance criterion 1: no bucket is derived by subtraction."""
    temp_dir, client, server = _env()
    try:
        plain_stack, plain_pics = _make_stack(
            server, [{"size_bytes": 10}, {"size_bytes": 700}]
        )
        locked_stack, locked_pics = _make_stack(server, [{}, {}])
        _lock_set_over(client, server, "Frozen", locked_pics[0])
        character_stack, character_pics = _make_stack(server, [{}, {}])
        _assign_character(server, character_pics[0], "A")
        _assign_character(server, character_pics[1], "B")
        single_stack, _single_pics = _make_stack(server, [{}])

        body = client.post(
            PREVIEW_URL,
            json={
                "stack_ids": [
                    plain_stack,
                    locked_stack,
                    character_stack,
                    single_stack,
                ]
            },
        ).json()

        assert body["stacks_selected"] == 4
        assert body["stacks_eligible"] == 1
        assert body["stacks_skipped_locked"] == 1
        assert body["stacks_skipped_character_on_copy"] == 1
        assert body["stacks_skipped_single_member"] == 1
        assert (
            body["stacks_eligible"]
            + body["stacks_skipped_locked"]
            + body["stacks_skipped_character_on_copy"]
            + body["stacks_skipped_single_member"]
            == body["stacks_selected"]
        )
        # Every row appears exactly once, and only the eligible one moves.
        assert len(body["stacks"]) == 4
        assert len({row["stack_id"] for row in body["stacks"]}) == 4
        assert body["pictures_moving"] == 1
        assert body["picture_ids_moving"] == [plain_pics[1]]
        # Bytes HELD, not freed, and only over what would actually move.
        assert body["bytes_held_by_copies"] == 700
        assert body["originals_deleted_from_disk"] == 0
        # The default install never auto-empties the Scrapheap.
        assert body["scrapheap_retention_days"] is None
    finally:
        _teardown(temp_dir, server)


def test_preview_and_mutation_agree_on_the_same_selection():
    """The dialog's figures are the ones the button acts on."""
    temp_dir, client, server = _env()
    try:
        first, first_pics = _make_stack(server, [{"tags": ["x"]}, {"tags": ["y"]}, {}])
        second, second_pics = _make_stack(server, [{}, {}])
        locked, locked_pics = _make_stack(server, [{}, {}])
        _lock_set_over(client, server, "Frozen", locked_pics[1])
        selection = {"stack_ids": [first, second, locked]}

        preview = client.post(PREVIEW_URL, json=selection).json()
        applied = client.post(COLLAPSE_URL, json=selection).json()

        assert applied["stacks_collapsed"] == preview["stacks_eligible"]
        assert applied["pictures_moved"] == preview["pictures_moving"]
        assert applied["picture_ids_moved"] == preview["picture_ids_moving"]
        assert applied["cover_picture_ids"] == preview["cover_picture_ids"]
        assert applied["covers_gaining_metadata"] == preview["covers_gaining_metadata"]
        assert len(applied["stacks_skipped_locked"]) == preview["stacks_skipped_locked"]
        assert (
            applied["reference_folder_pictures_moved"]
            == (preview["reference_folder_pictures_moving"])
        )
        assert set(applied["picture_ids_moved"]) == {
            first_pics[1],
            first_pics[2],
            second_pics[1],
        }
    finally:
        _teardown(temp_dir, server)


def test_a_partial_selection_inside_a_stack_collapses_the_whole_stack():
    """The unit is the stack, and naming one copy names all of them."""
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}, {}])
        loose_id = _make_stack(server, [{}])[1][0]

        body = client.post(
            PREVIEW_URL, json={"picture_ids": [pics[1], loose_id]}
        ).json()
        # The loose picture's own stack is a stack of one, so it lands in the
        # single-member bucket rather than being silently dropped.
        rows = _rows_by_stack(body)
        assert rows[stack_id]["eligible"] is True
        assert rows[stack_id]["copy_picture_ids"] == pics[1:]
        assert body["pictures_moving"] == 2
    finally:
        _teardown(temp_dir, server)


def test_a_selection_reads_its_stacks_by_stack_id_and_leaves_the_scrapheap_out(
    key_list_plan,
):
    """Resolving the stacks and loading their members search ``ix_picture_stack_id``.

    Two or more stack ids beside ``deleted IS 0`` lose to ``ix_picture_deleted``
    without ``sqlite_stat1`` and walk every kept picture. Whatever the shape,
    a stack whose members are all in the Scrapheap still resolves to unknown,
    and a scrapheaped member at position 0 is neither the cover nor a copy.
    """
    temp_dir, client, server = _env()
    try:
        gone, _gone_pics = _make_stack(server, [{"deleted": True}, {"deleted": True}])
        headless, headless_pics = _make_stack(server, [{"deleted": True}, {}, {}])
        plain, plain_pics = _make_stack(server, [{}, {}])

        with key_list_plan(
            server.vault.db._engine,
            "FROM picture \nWHERE picture.stack_id IN (?",
            "ix_picture_stack_id",
        ) as plans:
            body = client.post(
                PREVIEW_URL,
                json={"stack_ids": [gone, headless], "picture_ids": [plain_pics[1]]},
            ).json()
        # The selection's three stacks resolve, then the two live ones load.
        assert len(plans) == 2, plans

        assert body["unknown_stack_ids"] == [gone]
        assert body["stacks_selected"] == 2
        rows = _rows_by_stack(body)
        assert rows[headless]["cover_picture_id"] == headless_pics[1]
        assert rows[headless]["copy_picture_ids"] == [headless_pics[2]]
        assert rows[plain]["copy_picture_ids"] == [plain_pics[1]]
        assert body["picture_ids_moving"] == sorted([headless_pics[2], plain_pics[1]])

        # The plan re-reports a resolved stack with no members as unknown, so
        # the resolver is asked directly whether it called the Scrapheap live.
        resolve = keep_cover_only_service.resolve_selection_in_session
        assert _run(server, resolve, [gone, headless], [plain_pics[1]]) == (
            sorted([headless, plain]),
            [gone],
        )
        assert _run(server, resolve, [], []) == ([], [])
    finally:
        _teardown(temp_dir, server)


def test_reference_folder_copies_are_counted_separately():
    """Their rows move; their files are user-managed and are never touched."""
    temp_dir, client, server = _env()
    try:
        folder = client.post(
            f"{API}/reference-folders",
            json={"folder": os.path.join(temp_dir.name, "refs"), "label": "Refs"},
        )
        assert folder.status_code in (200, 201), folder.text
        folder_id = folder.json().get("id") or folder.json()["reference_folder"]["id"]

        stack_id, pics = _make_stack(
            server, [{}, {"reference_folder_id": folder_id}, {}]
        )

        body = client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).json()
        assert body["pictures_moving"] == 2
        assert body["reference_folder_pictures_moving"] == 1
        assert body["reference_folder_picture_ids_moving"] == [pics[1]]
        assert body["originals_deleted_from_disk"] == 0

        applied = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert applied["reference_folder_pictures_moved"] == 1
        assert applied["originals_deleted_from_disk"] == 0
    finally:
        _teardown(temp_dir, server)


def test_retention_setting_is_served_live_not_hardcoded():
    """The recovery copy must branch on the configured window, including "never"."""
    temp_dir, client, server = _env()
    try:
        stack_id, _pics = _make_stack(server, [{}, {}])
        assert (
            client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).json()[
                "scrapheap_retention_days"
            ]
            is None
        )
        assert (
            client.patch(
                f"{API}/server-config/scrapheap-retention",
                json={"scrapheap_retention_days": 60},
            ).status_code
            == 200
        )
        assert (
            client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).json()[
                "scrapheap_retention_days"
            ]
            == 60
        )
    finally:
        _teardown(temp_dir, server)


# ── the operation log ────────────────────────────────────────────────────────


def test_the_whole_call_is_one_operation_under_one_batch():
    """Several stacks, one row, one Ctrl+Z."""
    temp_dir, client, server = _env()
    try:
        first, first_pics = _make_stack(server, [{}, {}])
        second, second_pics = _make_stack(server, [{}, {}])

        body = client.post(COLLAPSE_URL, json={"stack_ids": [first, second]}).json()
        batch_id = body["batch_id"]

        operations = client.get(f"{API}/operations", params={"limit": 10}).json()
        rows = operations if isinstance(operations, list) else operations["operations"]
        mine = [row for row in rows if row.get("batch_id") == batch_id]
        assert len(mine) == 1
        assert mine[0]["op_type"] == operation_log_service.OP_STACK_KEEP_COVER_ONLY
        assert "Scrapheap" in (mine[0]["summary"] or "")
        # The summary must not claim any space was freed.
        assert "free" not in (mine[0]["summary"] or "").lower()

        assert client.post(UNDO_URL, json={"batch_id": batch_id}).status_code == 200
        rows = _picture_rows(server, first_pics + second_pics)
        assert not any(row[0] for row in rows.values())
    finally:
        _teardown(temp_dir, server)


# ── request validation ───────────────────────────────────────────────────────


def test_an_empty_selection_is_a_400_on_both_routes():
    temp_dir, client, server = _env()
    try:
        for url in (PREVIEW_URL, COLLAPSE_URL):
            assert client.post(url, json={}).status_code == 400
            assert client.post(url, json={"stack_ids": []}).status_code == 400
            assert client.post(url, json={"stack_ids": ["nope"]}).status_code == 422, (
                "pydantic rejects a non-integer before the service sees it"
            )
    finally:
        _teardown(temp_dir, server)


def test_an_unknown_stack_id_is_reported_outside_the_bucket_arithmetic():
    temp_dir, client, server = _env()
    try:
        stack_id, _pics = _make_stack(server, [{}, {}])
        body = client.post(PREVIEW_URL, json={"stack_ids": [stack_id, 999999]}).json()
        assert body["stacks_selected"] == 1
        assert body["stacks_eligible"] == 1
        assert body["unknown_stack_ids"] == [999999]
    finally:
        _teardown(temp_dir, server)


# ── what the collapse announces ──────────────────────────────────────────────


def _capture_events(server) -> list[tuple]:
    """Record every ``vault.notify`` call, in order."""
    emitted: list[tuple] = []
    original = server.vault.notify

    def record(event_type, data=None):
        emitted.append((event_type, data))
        return original(event_type, data)

    server.vault.notify = record
    return emitted


def _stack_facet_events(emitted) -> list[dict]:
    return [
        data
        for _event, data in emitted
        if isinstance(data, dict) and data.get("fields") == ["stack_count"]
    ]


def test_the_covers_stack_change_is_announced_even_when_the_union_added_nothing():
    """The cover's STACK always changes; its metadata only sometimes does.

    The announcement used to be gated on ``tags_added or scores_lifted``, which
    tests the wrong property: a collapse whose union found nothing new said
    nothing at all about the cover, so every view kept drawing it as a stack of
    three around a picture that was now on its own (the owner's report). The two
    halves stay separate ``change_kind``s, the copies ``removed`` and the covers
    ``updated``, because merging them would tell the grid a scrapheaped picture
    was merely updated and leave a 404-clickable card behind.
    """
    temp_dir, client, server = _env()
    try:
        # Identical tags and scores on every member: the union has nothing to do.
        stack_id, pics = _make_stack(
            server,
            [
                {"tags": ["a"], "score": 4},
                {"tags": ["a"], "score": 4},
                {"tags": ["a"], "score": 4},
            ],
        )
        emitted = _capture_events(server)

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["tags_added"] == 0 and body["scores_lifted"] == 0
        assert body["pictures_moved"] == 2

        removed = [
            data
            for _event, data in emitted
            if isinstance(data, dict) and data.get("change_kind") == "removed"
        ]
        assert removed, "the copies must still be announced as removed"
        assert removed[0]["picture_ids"] == sorted(pics[1:])
        # No `fields` on the removal: a vanished card is not a field change.
        assert not removed[0].get("fields")

        stack_events = _stack_facet_events(emitted)
        assert len(stack_events) == 1, emitted
        assert stack_events[0]["picture_ids"] == [pics[0]]
        assert stack_events[0]["change_kind"] == "updated"
        # And the two are separate events, never one merged announcement.
        assert stack_events[0] is not removed[0]
    finally:
        _teardown(temp_dir, server)


def test_a_union_that_did_something_keeps_its_own_unnarrowed_announcement():
    """The stack event is additive, not a replacement.

    Declaring ``fields=["stack_count"]`` on a cover that also gained a tag and a
    score would tell every client the change cannot affect its sort, which is
    false under a score sort. So the union keeps its own ``fields``-less event
    and the stack change rides beside it.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(
            server,
            [{"tags": ["a"], "score": 1}, {"tags": ["b"], "score": 5}],
        )
        emitted = _capture_events(server)

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["tags_added"] > 0 and body["scores_lifted"] > 0

        cover_events = [
            data
            for _event, data in emitted
            if isinstance(data, dict)
            and data.get("picture_ids") == [pics[0]]
            and data.get("change_kind") == "updated"
        ]
        assert len(cover_events) == 2, cover_events
        assert [ev.get("fields") for ev in cover_events].count(None) == 1
        assert [ev.get("fields") for ev in cover_events].count(["stack_count"]) == 1
    finally:
        _teardown(temp_dir, server)


def test_a_collapse_that_moved_nothing_announces_nothing_about_the_covers():
    """No move, no stack change: a skipped stack's cover still leads its stack."""
    temp_dir, client, server = _env()
    try:
        # A single-member stack is skipped outright, so nothing moves.
        stack_id, pics = _make_stack(server, [{}])
        emitted = _capture_events(server)

        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        assert body["pictures_moved"] == 0
        assert body["stacks_collapsed"] == 0
        assert not _stack_facet_events(emitted)
        assert pics
    finally:
        _teardown(temp_dir, server)


def test_undo_announces_the_covers_stack_change_back():
    """The reverse direction, which is the whole reason the undo window exists.

    After the undo the cover leads a stack of three again, and the only thing
    that says so is the ``stack_count`` announcement over the pictures the undo
    did NOT move: the copies come back as ``restored``, the cover was never
    scrapheaped and so appears in neither lifecycle list.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}, {}])
        body = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]}).json()
        batch_id = body["batch_id"]
        assert batch_id

        emitted = _capture_events(server)
        assert (
            client.post(f"{API}/operations/batches/{batch_id}/undo").status_code == 200
        )

        restored = [
            data
            for _event, data in emitted
            if isinstance(data, dict) and data.get("change_kind") == "restored"
        ]
        assert restored, "the copies must come back as restored"
        assert sorted(restored[0]["picture_ids"]) == sorted(pics[1:])

        stack_events = _stack_facet_events(emitted)
        assert len(stack_events) == 1, emitted
        assert stack_events[0]["picture_ids"] == [pics[0]]
        assert stack_events[0]["change_kind"] == "updated"

        # And back again. A fix that only works one way is half a fix, and redo
        # re-collapses the stack the undo just put back.
        emitted.clear()
        assert client.post(f"{API}/operations/redo").status_code == 200
        assert _stack_facet_events(emitted) == [
            {
                "picture_ids": [pics[0]],
                "origin_client_id": None,
                "change_kind": "updated",
                "source": "ui",
                "fields": ["stack_count"],
            }
        ]
    finally:
        _teardown(temp_dir, server)


# ── authorization, both directions ───────────────────────────────────────────


def test_scoped_read_token_is_denied_on_both_keep_cover_only_routes():
    """Negative direction, through both entry points.

    The ``?token=`` query parameter is a separate reachability path from the
    ``Authorization`` header, and a gate covering only one would be a hole
    rather than a policy.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}])
        token = _scoped_token(client, server, pics[0])
        scoped = TestClient(server.api)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"stack_ids": [stack_id]}

        for url in (PREVIEW_URL, COLLAPSE_URL):
            assert scoped.post(url, json=payload, headers=headers).status_code == 403
            assert (
                scoped.post(url, json=payload, params={"token": token}).status_code
                == 403
            )

        # Fail-closed, not fail-late: the refused call changed nothing.
        assert not any(_picture_rows(server, pics)[pid][0] for pid in pics)
    finally:
        _teardown(temp_dir, server)


def test_the_owner_reaches_both_keep_cover_only_routes():
    """Positive direction: over-blocking would be its own regression."""
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}])
        assert (
            client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).status_code == 200
        )
        response = client.post(COLLAPSE_URL, json={"stack_ids": [stack_id]})
        assert response.status_code == 200
        assert response.json()["pictures_moved"] == 1
        assert _picture_rows(server, pics)[pics[1]][0] is True
    finally:
        _teardown(temp_dir, server)


# ── the batch id is validated, not stored verbatim ───────────────────────────


def test_a_forged_batch_id_cannot_graft_two_collapses_into_one_undo():
    """`batch_id` is the undo handle, so an unvalidated one is an undo forgery.

    Taken verbatim, a client could send `srv-…` so its rows read as a server
    batch, or reuse one id across separate gestures so a single `Ctrl+Z`
    reverses more than the user did. The dedup routes validated it; this one
    did not, which is exactly the kind of second copy the shared helper in
    `utils/request_origin.py` exists to prevent.
    """
    temp_dir, client, server = _env()
    try:
        first, _first_pics = _make_stack(server, [{}, {}])
        second, _second_pics = _make_stack(server, [{}, {}])

        for forged in (
            "srv-deadbeefdeadbeefdeadbeefdeadbeef",  # the server's own namespace
            "cli-" + "a" * 200,  # over the length cap
            "x" * 1_000_000,  # unbounded
            "cli-bad id",  # outside the charset
            "cli-ab",  # under the minimum length
        ):
            response = client.post(
                COLLAPSE_URL, json={"stack_ids": [first], "batch_id": forged}
            )
            assert response.status_code == 400, (forged[:32], response.text)
            assert "batch_id" in response.json()["detail"]

        # Nothing was collapsed by any of the refused calls.
        assert (
            client.post(PREVIEW_URL, json={"stack_ids": [first]}).json()[
                "stacks_eligible"
            ]
            == 1
        )

        # A well-formed client id is still honoured, and deliberately sharing it
        # across two calls is the legitimate use it exists for.
        shared = "cli-one-gesture-01"
        assert (
            client.post(
                COLLAPSE_URL, json={"stack_ids": [first], "batch_id": shared}
            ).json()["batch_id"]
            == shared
        )
        assert (
            client.post(
                COLLAPSE_URL, json={"stack_ids": [second], "batch_id": shared}
            ).json()["batch_id"]
            == shared
        )
    finally:
        _teardown(temp_dir, server)


# ── the selection cap bounds the REQUEST, not the de-duplicated set ──────────


def test_the_selection_cap_bounds_the_request_not_the_deduplicated_set():
    """Two ways the cap used to be no cap at all, and the shape that still works.

    It was applied to the de-duplicated set, so a body of a million repeats of
    one id passed; and it was applied per list, so `stack_ids` plus
    `picture_ids` legitimately carried twice it in one request.
    """
    temp_dir, client, server = _env()
    try:
        stack_id, pics = _make_stack(server, [{}, {}])
        cap = keep_cover_only_service.MAX_SELECTION_IDS

        # (1) Repeats count. This de-duplicates to one id and used to return 200.
        response = client.post(COLLAPSE_URL, json={"picture_ids": [pics[0]] * 200_000})
        assert response.status_code == 400, response.status_code
        assert "picture_ids" in response.json()["detail"]

        # (2) The two lists share one budget.
        response = client.post(
            COLLAPSE_URL,
            json={
                "stack_ids": list(range(1, cap + 1)),
                "picture_ids": list(range(cap + 1, 2 * cap + 1)),
            },
        )
        assert response.status_code == 400, response.status_code
        assert "together" in response.json()["detail"]

        # The preview shares the check, so the dialog cannot be used to sneak past.
        assert (
            client.post(PREVIEW_URL, json={"picture_ids": [pics[0]] * 200_000})
        ).status_code == 400

        # A selection right at the budget still works: over-blocking is its own
        # regression, and the natural gesture here is a whole-library sweep.
        response = client.post(
            COLLAPSE_URL,
            json={
                "stack_ids": [stack_id],
                "picture_ids": list(range(10_000, 10_000 + cap - 1)),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["stacks_collapsed"] == 1
    finally:
        _teardown(temp_dir, server)


# ── the SPA's split of a large selection stays inside the contract ───────────


def _js_string_array(name: str) -> list[str]:
    """Return the strings of ``export const <name> = Object.freeze([...])``.

    A source parse rather than running node, as in
    ``tests/test_install_type_buckets.py``: the backend gate has no frontend
    toolchain, and the point is to catch a hand edit of either side.
    """
    source = KCO_REQUESTS_JS.read_text(encoding="utf-8")
    declaration = f"export const {name} = Object.freeze(["
    assert source.count(declaration) == 1, f"{name} not found once in {KCO_REQUESTS_JS}"
    opening = source.index(declaration) + len(declaration)
    closing = source.index("]", opening)
    return re.findall(r'"([^"]+)"', source[opening:closing])


@pytest.mark.parametrize(
    ("model", "prefix"),
    [
        (KeepCoverOnlyPreviewResponse, "KEEP_COVER_ONLY_PREVIEW"),
        (KeepCoverOnlyResponse, "KEEP_COVER_ONLY_RESULT"),
    ],
)
def test_the_spa_merges_every_field_of_a_split_report(model, prefix):
    """The SPA classifies every response field, and classifies it by its type.

    A selection larger than one request is previewed and collapsed as several
    requests whose reports the SPA merges back into one: counts are summed,
    lists concatenated, settings taken from the last report. A field the
    response gains without being classified there would vanish from the merged
    report, and the dialog would show a figure for the first 500 stacks of a
    1,200-stack selection without a sound. A count classified as a setting is
    the same loss; a setting summed turns a 30-day retention into 90.
    """
    counts = _js_string_array(f"{prefix}_COUNTS")
    lists = _js_string_array(f"{prefix}_LISTS")
    settings = _js_string_array(f"{prefix}_SETTINGS")
    classified = counts + lists + settings

    assert len(classified) == len(set(classified)), "a field is classified twice"
    assert set(classified) == set(model.model_fields), (
        "unclassified: "
        f"{sorted(set(model.model_fields) - set(classified))}; "
        f"not in the model: {sorted(set(classified) - set(model.model_fields))}"
    )
    for name in counts:
        assert model.model_fields[name].annotation is int, (name, "summed")
    for name in lists:
        assert typing.get_origin(model.model_fields[name].annotation) is list, (
            name,
            "concatenated",
        )
    for name in settings:
        annotation = model.model_fields[name].annotation
        assert annotation is not int and typing.get_origin(annotation) is not list, (
            name,
            "taken from the last report",
        )


def test_the_spa_never_sends_more_stacks_than_one_request_may_carry():
    """The SPA's request size is under the server's per-request id cap.

    Raising it past the cap brings the original failure back: every request of
    a large selection is refused with 400 and the dialog has no figures.
    """
    source = KCO_REQUESTS_JS.read_text(encoding="utf-8")
    found = re.findall(
        r"^export const KEEP_COVER_ONLY_STACKS_PER_REQUEST = (\d+);$",
        source,
        flags=re.MULTILINE,
    )
    assert len(found) == 1, found
    assert 0 < int(found[0]) <= keep_cover_only_service.MAX_SELECTION_IDS


# ── a whole-library selection crosses SQLite's bound-parameter ceiling ───────


@pytest.fixture(scope="module")
def past_the_ceiling():
    """One library with every id list past :func:`_sqlite_variable_ceiling`.

    Built once for the module because the build is most of what these tests
    cost. 1,500 stacks and 3,500 members, so the planner's stack and member lists
    cross the ceiling; 500 of those stacks strand two characters each on their
    copies, so it also has 1,000 characters to name; and one stack of 1,001
    members, its cover naming a character, so a single stack's own reads cross
    it too, the metadata union's one-character pass included. Every test that
    collapses something undoes it, so the library is the same for each of them.
    The restore-cost tests below add pictures of their own beside these and read
    only those, so they share this server rather than paying for another.
    """
    temp_dir, client, server = _env()
    try:
        eligible = _make_many_stacks(server, 1000, 2)
        stranding = _make_many_stacks(server, 500, 3, character_per_copy=True)
        ((big_stack, big_pics),) = _make_many_stacks(
            server, 1, 1001, character_on_cover=True
        )
        yield {
            "client": client,
            "server": server,
            "eligible": eligible,
            "stranding": stranding,
            "big_stack": big_stack,
            "big_pics": big_pics,
        }
    finally:
        _teardown(temp_dir, server)


def test_a_selection_over_the_variable_ceiling_previews_whole(past_the_ceiling):
    """``Ctrl+A`` over a stacked library is more ids than one statement may bind.

    The planner bound one parameter per stack, per member and per character it
    had to name, so selecting every stack in a big library failed with ``too
    many SQL variables`` and the dialog could only say the preview could not be
    read. Both ways a selection arrives are covered: the grid sends stacks and a
    multi-select sends pictures.
    """
    client, server = past_the_ceiling["client"], past_the_ceiling["server"]
    everything = past_the_ceiling["eligible"] + past_the_ceiling["stranding"]
    last_stack, last_pics = past_the_ceiling["stranding"][-1]

    with _sqlite_variable_ceiling(server):
        for selection in (
            {"stack_ids": [stack_id for stack_id, _pics in everything]},
            {"picture_ids": [pics[0] for _stack_id, pics in everything]},
        ):
            response = client.post(PREVIEW_URL, json=selection)
            assert response.status_code == 200, response.text[:300]
            preview = response.json()
            assert preview["stacks_selected"] == 1500
            assert preview["stacks_eligible"] == 1000
            assert preview["stacks_skipped_character_on_copy"] == 500
            assert preview["pictures_moving"] == 1000
            assert preview["covers_gaining_tags"] == 1000
            # The last stack's characters are in the last chunk, and they are
            # still named rather than dropped.
            lost = _rows_by_stack(preview)[last_stack]["lost_characters"]
            assert {item["name"]: item["picture_ids"] for item in lost} == {
                f"KCO {pid}": [pid] for pid in last_pics[1:]
            }


def test_one_gesture_over_the_variable_ceiling_collapses_and_undoes_whole(
    past_the_ceiling,
):
    """The grid's shape for a big selection: several requests, one batch, one undo.

    The SPA sends a large selection as several requests sharing one client batch
    id, so a single ``Ctrl+Z`` reverses all of them, and that undo restores every
    copy every request moved. The 1,001-member stack makes the metadata union,
    the soft delete and the undo each read more pictures than the ceiling.
    """
    client, server = past_the_ceiling["client"], past_the_ceiling["server"]
    big_stack, big_pics = past_the_ceiling["big_stack"], past_the_ceiling["big_pics"]
    small = past_the_ceiling["eligible"][:2]
    moved = big_pics[1:] + [pics[1] for _stack_id, pics in small]
    covers = [big_pics[0]] + [pics[0] for _stack_id, pics in small]
    batch_id = "cli-variable-ceiling-gesture"

    with _sqlite_variable_ceiling(server):
        for selection, expected in (
            ([big_stack], 1000),
            ([stack_id for stack_id, _pics in small], 2),
        ):
            response = client.post(
                COLLAPSE_URL, json={"stack_ids": selection, "batch_id": batch_id}
            )
            assert response.status_code == 200, response.text[:300]
            assert response.json()["pictures_moved"] == expected
            assert response.json()["batch_id"] == batch_id
        assert all(row[0] for row in _picture_rows(server, moved).values())
        assert all(_tags_of(server, cover) == {"copy_only"} for cover in covers)

        with _counting_vault_flushes(server) as flushes:
            response = client.post(f"{API}/operations/batches/{batch_id}/undo")
        assert response.status_code == 200, response.text[:300]

    # The restore reads every row it compares against before its first write, so
    # its writes reach the database together. Reading per picture autoflushed per
    # picture, about three flushes for each of these 1,005, each recomputing the
    # flushed pictures' metadata hashes: a whole-library undo outlasted the
    # client's timeout.
    assert len(flushes) <= 6, f"{len(flushes)} flushes to restore 1,005 pictures"
    rows = _picture_rows(server, moved + covers)
    assert len(rows) == 1005
    assert not any(row[0] for row in rows.values())
    # The union is reversed with the move: the covers lost the copies' tag.
    assert all(_tags_of(server, cover) == set() for cover in covers)


def test_one_ctrl_z_undoes_every_request_of_a_split_gesture(past_the_ceiling):
    """``Ctrl+Z`` reverses the whole gesture, not its last request.

    The SPA collapses a large selection as several requests under one client
    batch id, each confirming the moving ids of its own preview. The plain undo
    (``POST /operations/undo`` with no body, which is what ``Ctrl+Z`` sends)
    must take back every request of that gesture in one step, or the user sees
    1,200 stacks collapse and one ``Ctrl+Z`` bring back 200 of them.
    """
    client, server = past_the_ceiling["client"], past_the_ceiling["server"]
    first, second = past_the_ceiling["eligible"][2], past_the_ceiling["eligible"][3]
    batch_id = "cli-split-gesture-ctrl-z"

    for stack_id, _pics in (first, second):
        preview = client.post(PREVIEW_URL, json={"stack_ids": [stack_id]}).json()
        response = client.post(
            COLLAPSE_URL,
            json={
                "stack_ids": [stack_id],
                "batch_id": batch_id,
                "expected_picture_ids": preview["picture_ids_moving"],
            },
        )
        assert response.status_code == 200, response.text[:300]
        assert response.json()["stacks_collapsed"] == 1
    copies = [first[1][1], second[1][1]]
    covers = [first[1][0], second[1][0]]
    assert all(row[0] for row in _picture_rows(server, copies).values())

    response = client.post(UNDO_URL)
    assert response.status_code == 200, response.text[:300]
    undone = response.json()["operations"]
    assert len(undone) == 2
    assert {operation["batch_id"] for operation in undone} == {batch_id}
    assert not any(row[0] for row in _picture_rows(server, copies).values())
    assert all(_tags_of(server, cover) == set() for cover in covers)


def test_an_undo_over_the_variable_ceiling_still_names_the_covers_it_changes(
    past_the_ceiling,
):
    """The undo's last read names more stacks than one statement may bind.

    An undo announces every live member of each stack it changed, so the covers'
    badges redraw (``_stack_siblings_in_session``). Undoing a whole-library
    gesture names a stack per collapsed stack. Read directly, because reaching
    it end to end would mean collapsing a thousand stacks, about ten seconds.
    """
    server = past_the_ceiling["server"]
    eligible = past_the_ceiling["eligible"]
    copies = {pics[1] for _stack_id, pics in eligible}

    with _sqlite_variable_ceiling(server):
        siblings = _run(
            server,
            lambda session: operation_log_service._stack_siblings_in_session(
                session, copies
            ),
        )
    assert siblings == sorted(pics[0] for _stack_id, pics in eligible)


# ── undoing a library-sized gesture costs the same at any size ───────────────

# The facets the restore engine reads rows for before it writes, plus a column on
# the picture row itself.
_PRELOADED_FACETS = (
    operation_log_service.FACET_SCORE,
    operation_log_service.FACET_TAGS,
    operation_log_service.FACET_TAG_PREDICTIONS,
    operation_log_service.FACET_SETS,
    operation_log_service.FACET_PROJECTS,
    operation_log_service.FACET_CHARACTERS,
)


def _seed_every_preloaded_facet(server, count: int) -> list[int]:
    """Insert *count* unstacked pictures, each holding a row of every facet above.

    One set, one project and one character are shared by all of them, so the
    only thing that varies between two calls is the number of pictures.
    """

    def insert(session):
        label = f"Restore cost {count}"
        picture_set = PictureSet(name=label)
        project = Project(name=label)
        character = Character(name=label)
        pictures = [
            Picture(
                file_path=f"/vault/restore_cost_{count}_{n}.png",
                format="png",
                width=10,
                height=10,
                size_bytes=1000,
                score=3,
            )
            for n in range(count)
        ]
        session.add_all([picture_set, project, character, *pictures])
        session.flush()
        for picture in pictures:
            picture_id = int(picture.id)
            session.add_all(
                [
                    Tag(picture_id=picture_id, tag="restore_cost"),
                    TagPrediction(
                        picture_id=picture_id,
                        tag="restore_cost",
                        confidence=0.9,
                        model_version="restore-cost-model",
                        status="CONFIRMED",
                        label_state="POS",
                        label_source="human",
                    ),
                    PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id),
                    PictureProjectMember(
                        picture_id=picture_id, project_id=int(project.id)
                    ),
                    Face(
                        picture_id=picture_id,
                        frame_index=0,
                        face_index=0,
                        character_id=int(character.id),
                        x=0,
                        y=0,
                        width=10,
                        height=10,
                    ),
                ]
            )
        session.commit()
        return [int(picture.id) for picture in pictures]

    return _run(server, insert)


def _preloaded_state(server, picture_ids: list[int]) -> dict:
    """The recorded state of *picture_ids*, narrowed to :data:`_PRELOADED_FACETS`."""

    def capture(session):
        state = operation_log_service.capture_state_in_session(session, picture_ids)
        return {
            picture_id: {facet: facets[facet] for facet in _PRELOADED_FACETS}
            for picture_id, facets in state.items()
        }

    return _run(server, capture)


def _scramble_every_preloaded_facet(server, picture_ids: list[int]) -> None:
    """Change every facet of :data:`_PRELOADED_FACETS` on every picture."""

    def scramble(session):
        for model in (Tag, PictureSetMember, PictureProjectMember):
            session.exec(delete(model).where(model.picture_id.in_(picture_ids)))
        session.exec(
            update(Face)
            .where(Face.picture_id.in_(picture_ids))
            .values(character_id=None)
        )
        session.exec(
            update(TagPrediction)
            .where(TagPrediction.picture_id.in_(picture_ids))
            .values(status="PENDING", label_state="UNKNOWN", label_source=None)
        )
        session.exec(update(Picture).where(Picture.id.in_(picture_ids)).values(score=1))
        session.commit()

    _run(server, scramble)


def _restore_round_trips(server, state: dict) -> tuple[int, int]:
    """Apply *state* in one session and commit; ``(flushes, reads)`` it took.

    Reads, not every statement: the ORM inserts an autoincrement row such as a
    ``Tag`` one statement per row inside a single flush, which is the cost of
    the row itself, not of how the restore is organised.
    """

    def restore(session):
        flushes: list[int] = []
        reads: list[str] = []
        connection = session.connection()

        def on_flush(_session, _flush_context):
            flushes.append(1)

        def on_statement(conn, _cursor, statement, *_rest):
            if conn is connection and statement.lstrip().upper().startswith("SELECT"):
                reads.append(statement)

        event.listen(session, "after_flush", on_flush)
        event.listen(connection.engine, "before_cursor_execute", on_statement)
        try:
            operation_log_service.apply_state_in_session(session, state)
            session.commit()
        finally:
            event.remove(connection.engine, "before_cursor_execute", on_statement)
            event.remove(session, "after_flush", on_flush)
        return len(flushes), len(reads)

    return _run(server, restore)


def test_a_restore_reads_and_writes_in_a_constant_number_of_round_trips(
    past_the_ceiling,
):
    """Restoring four times the pictures costs exactly the same flushes and reads.

    Each applier read its own picture's rows with a query of its own, and a
    query in a session with pending writes flushes them first, so a restore
    paid a read per picture per facet and a flush per picture, each flush
    recomputing the flushed pictures' metadata hashes. Undoing a whole-library
    gesture outlasted the client's timeout. The engine now reads every row it
    compares against before its first write. A facet read per picture again
    adds a read per picture here, which the equality catches; the state check
    proves the restore still restores.
    """
    server = past_the_ceiling["server"]
    cost = {}
    for count in (15, 60):
        picture_ids = _seed_every_preloaded_facet(server, count)
        recorded = _preloaded_state(server, picture_ids)
        _scramble_every_preloaded_facet(server, picture_ids)
        scrambled = _preloaded_state(server, picture_ids)
        # Every facet of every picture moved, so the restore has to write each.
        assert all(
            scrambled[picture_id][facet] != recorded[picture_id][facet]
            for picture_id in recorded
            for facet in _PRELOADED_FACETS
        )

        cost[count] = _restore_round_trips(server, recorded)

        assert _preloaded_state(server, picture_ids) == recorded
    assert cost[60] == cost[15], f"(flushes, reads) by size: {cost}"


def test_undo_and_redo_of_stacking_every_group_flush_once_per_stack_at_most(
    past_the_ceiling,
):
    """The other gesture a deduplication pass undoes at library size.

    Stacking every duplicate group sends one ``stacks.create`` per group under
    one batch, so one ``Ctrl+Z`` replays hundreds of operations. A replay
    flushes once per operation by design: each operation's lock check must see
    the operations replayed before it. Everything else used to flush per stack
    as well: the undo deleted each emptied stack row after reading it, and the
    redo looked each stack row up and recreated it one at a time. Those are now
    one read and one write for the whole restore, so the undo flushes once per
    stack and the redo, which also recreates each row as its operation replays,
    at most twice. Both still put back exactly the rows the gesture made.
    """
    client, server = past_the_ceiling["client"], past_the_ceiling["server"]
    groups = 12

    def insert(session):
        pictures = [
            Picture(
                file_path=f"/vault/stack_every_group_{n}.png",
                format="png",
                width=10,
                height=10,
                size_bytes=1000,
            )
            for n in range(2 * groups)
        ]
        session.add_all(pictures)
        session.commit()
        return [int(picture.id) for picture in pictures]

    picture_ids = _run(server, insert)
    batch_id = "cli-stack-every-group"
    stacks = []
    for group in range(groups):
        response = client.post(
            f"{API}/stacks",
            json={"picture_ids": picture_ids[2 * group : 2 * group + 2]},
            headers={"X-Operation-Batch-Id": batch_id},
        )
        assert response.status_code == 200, response.text[:300]
        stacks.append(int(response.json()["id"]))

    def stack_rows(session):
        return {
            int(stack_id)
            for stack_id in session.exec(
                select(PictureStack.id).where(PictureStack.id.in_(stacks))
            )
        }

    stacked = _picture_rows(server, picture_ids)
    assert _run(server, stack_rows) == set(stacks)

    with _counting_vault_flushes(server) as undo_flushes:
        response = client.post(f"{API}/operations/batches/{batch_id}/undo")
    assert response.status_code == 200, response.text[:300]
    assert all(row[1] is None for row in _picture_rows(server, picture_ids).values())
    assert _run(server, stack_rows) == set(), "undo left emptied stack rows behind"

    with _counting_vault_flushes(server) as redo_flushes:
        response = client.post(f"{API}/operations/redo")
    assert response.status_code == 200, response.text[:300]
    # The same rows under the same ids, the pictures back on them in order.
    assert _run(server, stack_rows) == set(stacks)
    assert _picture_rows(server, picture_ids) == stacked

    assert len(undo_flushes) <= groups + 2, f"{len(undo_flushes)} flushes to undo"
    assert len(redo_flushes) <= 2 * groups + 2, f"{len(redo_flushes)} flushes to redo"


def test_a_library_with_no_locked_set_answers_the_lock_check_in_one_read(
    past_the_ceiling,
):
    """The lock check a restore runs per replayed operation, when nothing is locked.

    Expanding the pictures to their stacks and joining them to the locked sets
    took three temp tables per call, the largest share of undoing a
    stack-every-group gesture, in a library with nothing locked to find. One
    read now answers that. The same read is an ordinary autoflushing query, so
    a set locked earlier in the same, uncommitted session still freezes its
    picture: the positive control, without which a check that always answered
    "nothing" would pass.
    """
    server = past_the_ceiling["server"]
    eligible = past_the_ceiling["eligible"]
    cover, copy = eligible[0][1]
    selection = [pics[0] for _stack_id, pics in eligible]

    def check(session):
        statements: list[str] = []
        connection = session.connection()

        def on_statement(conn, _cursor, statement, *_rest):
            if conn is connection:
                statements.append(statement)

        event.listen(connection.engine, "before_cursor_execute", on_statement)
        try:
            unlocked = set_lock_service.locked_picture_ids(session, selection)
        finally:
            event.remove(connection.engine, "before_cursor_execute", on_statement)

        picture_set = PictureSet(name="Locked in this session", locked=True)
        session.add(picture_set)
        session.flush()
        session.add(PictureSetMember(set_id=int(picture_set.id), picture_id=copy))
        locked = set_lock_service.locked_picture_ids(session, selection)
        session.rollback()
        return unlocked, statements, locked

    unlocked, statements, locked = _run(server, check)
    assert unlocked == set()
    assert len(statements) == 1, statements
    # Stack-expanded: the copy is the member, and its cover is frozen with it.
    assert locked == {cover}


# ── Keep recipes only (#1315) ────────────────────────────────────────────────

_ON_SHELF = "kro_on_shelf.safetensors"
_OFF_SHELF = "kro_not_on_shelf.safetensors"


def _txt2img(prompt: str, ckpt: str = _ON_SHELF) -> dict:
    """A minimal API-format txt2img graph; the prompt and model vary the keys."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": prompt},
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "seed": 7,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
        },
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }


def _file_recipe(server, graph: dict):
    """File *graph* in the hub for the active library; returns its keys."""
    from pixlstash.hub.workflows import record_api_graph

    return record_api_graph(server.vault.hub, graph, server.vault.library_uuid)


def _set_keys(server, picture_id: int, keys, thumbnail: bool = True) -> None:
    """Give a picture workflow keys, a pixel_sha and (optionally) a thumbnail file.

    The file is what the purge reads, so the planner checks the file too.
    """

    def write(session):
        picture = session.get(Picture, picture_id)
        if keys is not None:
            picture.workflow_topology_hash = keys.topology_hash
            picture.workflow_structural_hash = keys.structural_hash
            picture.workflow_instance_hash = keys.instance_hash
        picture.pixel_sha = f"kro-pixels-{picture_id}"
        picture.thumbnail_width = 64 if thumbnail else None
        picture.thumbnail_height = 64 if thumbnail else None
        session.add(picture)
        session.commit()
        return picture.file_path

    file_path = _run(server, write)
    if thumbnail:
        thumb = ImageUtils.get_thumbnail_path(server.vault.image_root, file_path)
        os.makedirs(os.path.dirname(thumb), exist_ok=True)
        with open(thumb, "wb") as handle:
            handle.write(b"thumb")


def _recipes_env():
    """A server with the on-shelf model registered and a stack of six:

    cover + ``covered`` (same instance as the cover) + ``no_recipe`` +
    ``uncovered`` (its own prompt) + ``model_missing`` + ``no_thumbnail``.
    """
    temp_dir, client, server = _env()
    assert server.vault.hub is not None and server.vault.library_uuid
    with server.vault.hub.transaction() as conn:
        conn.execute(
            "INSERT INTO model (file_kind, filename, provenance) "
            "VALUES ('checkpoint', ?, 'scanned')",
            (_ON_SHELF,),
        )
    shared = _file_recipe(server, _txt2img("a lighthouse"))
    own = _file_recipe(server, _txt2img("a harbour"))
    off_shelf = _file_recipe(server, _txt2img("a pier", ckpt=_OFF_SHELF))
    stack_id, ids = _make_stack(server, [{}, {}, {}, {}, {}, {}])
    names = [
        "cover",
        "covered",
        "no_recipe",
        "uncovered",
        "model_missing",
        "no_thumbnail",
    ]
    pics = dict(zip(names, ids))
    _set_keys(server, pics["cover"], shared)
    _set_keys(server, pics["covered"], shared)
    _set_keys(server, pics["no_recipe"], None)
    _set_keys(server, pics["uncovered"], own)
    _set_keys(server, pics["model_missing"], off_shelf)
    _set_keys(server, pics["no_thumbnail"], shared, thumbnail=False)
    return temp_dir, client, server, stack_id, pics


def test_keep_recipes_only_moves_only_what_could_be_made_again():
    """Each copy that stays is counted under exactly one reason, and only the
    covered copy moves at the default retention."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        assert server.vault.ghost_retention == "covered"
        selection = {"stack_ids": [stack_id], "keep_recipes": True}
        preview = client.post(PREVIEW_URL, json=selection).json()
        assert preview["keep_recipes"] is True
        assert preview["ghost_retention"] == "covered"
        assert preview["picture_ids_moving"] == [pics["covered"]]
        assert preview["pictures_staying_no_recipe"] == 1
        assert preview["pictures_staying_model_missing"] == 1
        assert preview["pictures_staying_no_thumbnail"] == 1
        assert preview["pictures_staying_ghost_not_kept"] == 1

        body = client.post(COLLAPSE_URL, json=selection).json()
        assert body["picture_ids_moved"] == [pics["covered"]]
        assert body["pictures_staying"] == 4
        rows = _picture_rows(server, list(pics.values()))
        assert {pid for pid, row in rows.items() if row[0]} == {pics["covered"]}
        # Nothing asked for every ghost, so the setting is untouched.
        assert server.vault.ghost_retention == "covered"

        operations = client.get(f"{API}/operations", params={"limit": 10}).json()
        rows = operations if isinstance(operations, list) else operations["operations"]
        mine = [r for r in rows if r.get("batch_id") == body["batch_id"]]
        assert len(mine) == 1
        assert mine[0]["op_type"] == operation_log_service.OP_STACK_KEEP_RECIPES_ONLY
        assert (
            client.post(UNDO_URL, json={"batch_id": body["batch_id"]}).status_code
            == 200
        )
        rows = _picture_rows(server, list(pics.values()))
        assert not any(row[0] for row in rows.values())
    finally:
        _teardown(temp_dir, server)


def test_coverage_is_read_by_instance_hash_and_the_scrapheap_covers_nothing(
    key_list_plan, monkeypatch
):
    """Whether a copy is covered is read through its instance hash's index.

    The two moving copies' instance hashes beside ``deleted IS 0`` lose to
    ``ix_picture_deleted`` without ``sqlite_stat1``. Whatever the shape, a
    scrapheaped picture carrying the uncovered copy's instance does not cover
    it, in one chunk or several.
    """
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        _other, (scrapheaped,) = _make_stack(server, [{"deleted": True}])

        def share_instance(session):
            source = session.get(Picture, pics["uncovered"])
            picture = session.get(Picture, scrapheaped)
            picture.workflow_structural_hash = source.workflow_structural_hash
            picture.workflow_instance_hash = source.workflow_instance_hash
            session.add(picture)
            session.commit()

        _run(server, share_instance)
        selection = {"stack_ids": [stack_id], "keep_recipes": True}

        with key_list_plan(
            server.vault.db._engine,
            "FROM picture \nWHERE picture.workflow_instance_hash IN (?",
            "ix_picture_workflow_instance_hash",
        ) as plans:
            preview = client.post(PREVIEW_URL, json=selection).json()
        assert len(plans) == 1, plans
        assert preview["picture_ids_moving"] == [pics["covered"]]
        assert preview["pictures_staying_ghost_not_kept"] == 1

        monkeypatch.setattr(
            keep_cover_only_service,
            "chunked",
            lambda seq: sql_chunking.chunked(seq, size=1),
        )
        preview = client.post(PREVIEW_URL, json=selection).json()
        assert preview["picture_ids_moving"] == [pics["covered"]]
        assert preview["pictures_staying_ghost_not_kept"] == 1
    finally:
        _teardown(temp_dir, server)


def test_keeping_every_ghost_moves_the_uncovered_copy_and_turns_the_setting_on():
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        selection = {
            "stack_ids": [stack_id],
            "keep_recipes": True,
            "keep_every_ghost": True,
        }
        preview = client.post(PREVIEW_URL, json=selection).json()
        assert preview["ghost_retention"] == "on"
        assert preview["picture_ids_moving"] == sorted(
            [pics["covered"], pics["uncovered"]]
        )
        assert preview["pictures_staying_ghost_not_kept"] == 0
        # A preview is read-only: the consent applies only on the real call.
        assert server.vault.ghost_retention == "covered"

        body = client.post(COLLAPSE_URL, json=selection).json()
        assert body["picture_ids_moved"] == sorted([pics["covered"], pics["uncovered"]])
        assert server.vault.ghost_retention == "on"
        assert server._server_config["workflow_ghost_retention"] == "on"
    finally:
        _teardown(temp_dir, server)


def test_a_stack_with_nothing_reproducible_is_its_own_bucket():
    temp_dir, client, server = _env()
    try:
        stack_id, _ids = _make_stack(server, [{}, {}])
        other, other_ids = _make_stack(server, [{}, {}])
        preview = client.post(
            PREVIEW_URL, json={"stack_ids": [stack_id, other], "keep_recipes": True}
        ).json()
        assert preview["stacks_selected"] == 2
        assert preview["stacks_skipped_nothing_reproducible"] == 2
        assert preview["stacks_eligible"] == 0
        assert preview["pictures_moving"] == 0
        assert preview["pictures_staying_no_recipe"] == 2
        body = client.post(
            COLLAPSE_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        assert body["pictures_moved"] == 0
        assert len(body["stacks_skipped_nothing_reproducible"]) == 1
    finally:
        _teardown(temp_dir, server)


def test_a_character_on_a_copy_that_stays_is_not_lost():
    """Keep recipes only judges character loss over the copies that leave."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        _assign_character(server, pics["cover"], "Ada")
        # Two characters, so the union will not propagate; the one on a staying
        # copy is not lost, the one on the moving copy is.
        _assign_character(server, pics["no_recipe"], "Grace")
        selection = {"stack_ids": [stack_id], "keep_recipes": True}
        preview = client.post(PREVIEW_URL, json=selection).json()
        assert preview["stacks_eligible"] == 1

        # A third character on the copy that moves is lost; Grace, on the copy
        # that stays, still is not.
        _assign_character(server, pics["covered"], "Linus")
        preview = client.post(PREVIEW_URL, json=selection).json()
        row = _rows_by_stack(preview)[stack_id]
        assert row["skip_reason"] == keep_cover_only_service.SKIP_CHARACTER_ON_COPY
        assert [c["name"] for c in row["lost_characters"]] == ["Linus"]
    finally:
        _teardown(temp_dir, server)


def test_a_copy_that_stays_is_left_out_of_the_metadata_union():
    """The union writes every tag and the best score onto every picture it is
    given; a copy Keep recipes only leaves live must not be one of them."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:

        def seed(session):
            cover = session.get(Picture, pics["cover"])
            cover.score = 5
            session.add(cover)
            session.add(Tag(picture_id=pics["cover"], tag="kro_cover_tag"))
            session.add(Tag(picture_id=pics["covered"], tag="kro_copy_tag"))
            session.commit()

        _run(server, seed)
        body = client.post(
            COLLAPSE_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        assert body["picture_ids_moved"] == [pics["covered"]]
        assert "kro_copy_tag" in _tags_of(server, pics["cover"])
        staying = _picture_rows(server, [pics["no_recipe"]])[pics["no_recipe"]]
        assert staying[0] is False and staying[3] != 5
        assert _tags_of(server, pics["no_recipe"]) == set()
    finally:
        _teardown(temp_dir, server)


def test_retention_off_keeps_every_reproducible_copy_and_names_it():
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        server.vault.set_ghost_retention("off")
        preview = client.post(
            PREVIEW_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        assert preview["ghost_retention"] == "off"
        assert preview["pictures_moving"] == 0
        assert preview["pictures_staying_ghost_not_kept"] == 2
        assert preview["stacks_skipped_nothing_reproducible"] == 1
    finally:
        _teardown(temp_dir, server)


def test_a_missing_thumbnail_file_or_pixel_sha_keeps_the_copy():
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        cover_path = _run(
            server, lambda session: session.get(Picture, pics["covered"]).file_path
        )
        os.remove(ImageUtils.get_thumbnail_path(server.vault.image_root, cover_path))
        preview = client.post(
            PREVIEW_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        # The covered copy joins the one with no thumbnail, and nothing moves.
        assert preview["pictures_staying_no_thumbnail"] == 2
        assert preview["pictures_moving"] == 0

        _set_keys(server, pics["covered"], None)  # rewrites the thumbnail file

        def drop_sha(session):
            picture = session.get(Picture, pics["covered"])
            picture.pixel_sha = None
            session.add(picture)
            session.commit()

        _run(server, drop_sha)
        preview = client.post(
            PREVIEW_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        assert preview["pictures_staying_no_recipe"] == 2
        assert preview["pictures_moving"] == 0
    finally:
        _teardown(temp_dir, server)


def test_every_ghost_is_saved_to_disk_and_untouched_when_nothing_moved():
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        config_path = server._server_config_path
        nothing, _ids = _make_stack(server, [{}, {}])
        body = client.post(
            COLLAPSE_URL,
            json={
                "stack_ids": [nothing],
                "keep_recipes": True,
                "keep_every_ghost": True,
            },
        ).json()
        assert body["pictures_moved"] == 0
        assert server.vault.ghost_retention == "covered"

        body = client.post(
            COLLAPSE_URL,
            json={
                "stack_ids": [stack_id],
                "keep_recipes": True,
                "keep_every_ghost": True,
            },
        ).json()
        assert body["pictures_moved"] == 2
        assert body["ghost_retention"] == "on"
        with open(config_path) as handle:
            assert json.load(handle)["workflow_ghost_retention"] == "on"
    finally:
        _teardown(temp_dir, server)


def test_a_model_the_shelf_cannot_judge_keeps_the_copy():
    """The shelf scans .safetensors only, so a .ckpt is unjudgeable. On a
    destructive action "we do not know" has to keep the picture."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        ckpt = _file_recipe(server, _txt2img("a lighthouse", ckpt="kro_local.ckpt"))
        _set_keys(server, pics["covered"], ckpt)
        preview = client.post(
            PREVIEW_URL, json={"stack_ids": [stack_id], "keep_recipes": True}
        ).json()
        assert preview["pictures_staying_model_missing"] == 2
        assert preview["pictures_moving"] == 0
    finally:
        _teardown(temp_dir, server)


def test_a_plan_that_grew_since_the_preview_is_refused():
    """Thumbnails land on a background timer, so the confirm must act on the
    figure the button showed or refuse."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        selection = {"stack_ids": [stack_id], "keep_recipes": True}
        preview = client.post(PREVIEW_URL, json=selection).json()
        confirmed = preview["picture_ids_moving"]
        assert confirmed == [pics["covered"]]

        # What a finder writing a thumbnail mid-dialog does to the plan.
        _set_keys(server, pics["no_thumbnail"], None)

        refused = client.post(
            COLLAPSE_URL, json={**selection, "expected_picture_ids": confirmed}
        )
        assert refused.status_code == 409
        rows = _picture_rows(server, list(pics.values()))
        assert not any(row[0] for row in rows.values())

        # A plan that shrank is within the consent and still runs.
        body = client.post(
            COLLAPSE_URL,
            json={
                **selection,
                "expected_picture_ids": confirmed + [pics["no_thumbnail"]],
            },
        ).json()
        assert sorted(body["picture_ids_moved"]) == sorted(
            [pics["covered"], pics["no_thumbnail"]]
        )
    finally:
        _teardown(temp_dir, server)


def test_keeping_every_ghost_is_in_force_before_anything_moves():
    """The plan is made under `on`, so a failure to apply it must leave the
    library untouched rather than move pictures under the old position."""
    temp_dir, client, server, stack_id, pics = _recipes_env()
    try:
        called = []
        original = workflow_ghost_service.apply_ghost_retention

        def explode(srv, retention):
            # No DB read in here: this runs ON the vault's writer thread, so
            # asking it for a row would wait on the queue it is draining.
            called.append(retention)
            original(srv, retention)
            raise OSError("server-config is not writable")

        workflow_ghost_service.apply_ghost_retention = explode
        try:
            with pytest.raises(OSError):
                client.post(
                    COLLAPSE_URL,
                    json={
                        "stack_ids": [stack_id],
                        "keep_recipes": True,
                        "keep_every_ghost": True,
                    },
                )
        finally:
            workflow_ghost_service.apply_ghost_retention = original
        assert called == ["on"]
        # Nothing moved, which is only true if the setting was attempted first:
        # a write after the soft deletes would have left them behind.
        rows = _picture_rows(server, list(pics.values()))
        assert not any(row[0] for row in rows.values())
    finally:
        _teardown(temp_dir, server)
