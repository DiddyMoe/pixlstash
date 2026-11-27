"""A write that leaves a person pending on a picture with faces gives it to the face.

Face extraction resolves a pending person (``Picture.pending_character_id``)
only for the pictures it has just extracted. A duplicate merge's metadata
union, Keep cover only, an auto-stack slice and an undo or redo can set one on
a picture whose faces are already there, so each committing wrapper hands its
pictures to ``resolve_pending_people_with_faces``.

Each test seeds its own folder, so tests sharing the module's server never
read each other's pictures.
"""

import gc
import json
import os
import tempfile

import pytest

from pixlstash.database import DBPriority
from pixlstash.db_models import Character, Face, Picture, PictureStack
from pixlstash.server import Server
from pixlstash.services import dedup_auto_stack_service
from pixlstash.services import dedup_tier_service as tiers
from pixlstash.services import dedup_verdict_service as verdicts
from pixlstash.services import keep_cover_only_service, operation_log_service
from pixlstash.services.dedup_tier_service import DedupScope, ScopeType, TierPolicy
from pixlstash.tasks.dedup_auto_stack_finder import DedupAutoStackFinder


@pytest.fixture(scope="module")
def server():
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    force_cpu = Server.DEFAULT_FORCE_CPU
    Server.DEFAULT_FORCE_CPU = True
    srv = Server(config_path)
    try:
        yield srv
    finally:
        srv.close()
        Server.DEFAULT_FORCE_CPU = force_cpu
        temp_dir.cleanup()
        gc.collect()


def _run(server, fn, *args):
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _seed_pair(server, folder, *, second_has_face=True, stacked=False):
    """Two exact duplicates in *folder*; the first has a face named Alice.

    The second has an unnamed face (extraction already ran on it) unless
    *second_has_face* is False. With *stacked* the pair is already one stack,
    the first its cover.
    """

    def insert(session):
        alice = Character(name=f"Alice {folder}")
        session.add(alice)
        stack = PictureStack(name=None) if stacked else None
        if stack is not None:
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
        session.add(
            Face(
                picture_id=ids[0],
                face_index=0,
                character_id=alice.id,
                bbox=[0, 0, 40, 40],
            )
        )
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


def _merge(server, folder, ids):
    """Scan *folder* and stack its one duplicate pair."""
    scope = DedupScope(ScopeType.FOLDER, folder)
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), scope)
    page, _total, _cursor = _run(
        server, tiers.page_queue_in_session, None, scope, 0, 10
    )
    assert [{member["picture_id"] for member in row["candidates"]} for row in page] == [
        {ids["named"], ids["other"]}
    ]
    verdicts.apply_stack_verdict(server.vault, page[0]["signature"])


def _person_state(server, ids):
    """The second picture's pending person, and the person on its face."""

    def read(session):
        other = session.get(Picture, ids["other"])
        face = session.get(Face, ids["unnamed_face"]) if ids["unnamed_face"] else None
        return other.pending_character_id, face.character_id if face else None

    return _run(server, read)


def test_a_duplicate_merge_gives_its_person_to_the_face_already_found(server):
    ids = _seed_pair(server, "/vault/merge")

    _merge(server, "/vault/merge", ids)

    assert _person_state(server, ids) == (None, ids["alice"])


def test_a_duplicate_merge_leaves_a_picture_not_yet_extracted_pending(server):
    ids = _seed_pair(server, "/vault/unextracted", second_has_face=False)

    _merge(server, "/vault/unextracted", ids)

    assert _person_state(server, ids) == (ids["alice"], None), (
        "no face rows yet: extraction resolves it"
    )


def test_keep_cover_only_gives_its_person_to_the_face_already_found(server):
    ids = _seed_pair(server, "/vault/keep-cover", stacked=True)

    keep_cover_only_service.keep_cover_only(server.vault, stack_ids=[ids["stack"]])

    assert _person_state(server, ids) == (None, ids["alice"])


def test_an_auto_stack_gives_its_person_to_the_face_already_found(server):
    folder = "/vault/auto-stack"
    ids = _seed_pair(server, folder)
    scope = DedupScope(ScopeType.FOLDER, folder)
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), scope)

    dedup_auto_stack_service.start_apply_run(server.vault, scope)
    finder = DedupAutoStackFinder(server.vault)
    while (task := finder.find_task()) is not None:
        task.run()

    assert _person_state(server, ids) == (None, ids["alice"])


def test_a_redo_gives_its_person_to_a_face_found_since_the_undo(server):
    ids = _seed_pair(server, "/vault/redo", second_has_face=False)
    _merge(server, "/vault/redo", ids)
    operation_log_service.undo(server.vault)

    def extract(session):
        face = Face(picture_id=ids["other"], face_index=0, bbox=[0, 0, 30, 30])
        session.add(face)
        session.commit()
        return int(face.id)

    ids["unnamed_face"] = _run(server, extract)
    assert _person_state(server, ids) == (None, None)

    operation_log_service.redo(server.vault)

    assert _person_state(server, ids) == (None, ids["alice"])
