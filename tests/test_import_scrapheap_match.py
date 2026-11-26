"""Import must SEE the Scrapheap, and restore a match rather than re-import it.

``Picture.find`` defaults ``include_deleted=False``, and the one-shot import
called it that way, so a scrapheaped picture was invisible to import dedup:
re-importing a file whose picture sits in the Scrapheap created a brand-new
second row while the original was still there. A bulk "Keep cover only" cleanup
puts hundreds of pictures in the Scrapheap in one gesture, all of them copies of
files the user still has on disk, so the next import would silently undo the
cleanup and roughly double the bytes.

These tests pin both directions:

* a scrapheaped match is NOT imported again, is reported in its own bucket, and
  is restored: re-importing a file means the user wants that picture back;
* a genuinely new file still imports: over-matching is its own regression;
* a permanently purged (and ledgered) file is not resurrected: with no row left
  there is nothing to match, so a deliberate re-import is a NEW picture;
* the buckets are disjoint and sum to the file total, on every path.
"""

import io
import os
import sqlite3
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlmodel import select

from pixlstash.db_models import (
    DeletedFileLog,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Project,
    Tag,
)
from pixlstash.event_types import EventType
from pixlstash.image_plugins.base import ImagePlugin
from pixlstash.image_plugins.service import (
    _import_output_images,
    apply_plugin_to_pictures,
)
from pixlstash.routes.pictures._helpers import _parse_sidecar_tags
from pixlstash.routes.pictures._library_move import _copy_into
from pixlstash.server import Server
from pixlstash.services import (
    import_dedup_service,
    import_publish_service,
    scrapheap_service,
)
from pixlstash.services.comfyui_service import _import_comfyui_outputs
from pixlstash.services.import_dedup_service import load_match_candidates_in_session
from pixlstash.tasks import picture_import_task
from pixlstash.tasks.missing_face_extraction_finder import MissingFaceExtractionFinder
from pixlstash.tasks.watch_folder_import_task import WatchFolderImportTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from tests.authz_guard import no_spa_fallback  # noqa: F401


API = "/api/v1"

pytestmark = pytest.mark.usefixtures("no_spa_fallback")


def _png_bytes(color=(100, 149, 237)) -> bytes:
    img = Image.new("RGB", (48, 48), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bmp_bytes(color=(100, 149, 237), size=(48, 48)) -> bytes:
    """Uncompressed images make equal/different byte-size cases deterministic."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


@pytest.fixture
def owner_server():
    """A fresh Server with a claimed owner account and an authed client."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "server-config.json")
        with Server(config_path) as srv:
            client = TestClient(srv.api, raise_server_exceptions=True)
            r = client.post(
                f"{API}/login",
                json={"username": "owner", "password": "example-owner-password"},
            )
            assert r.status_code == 200, r.text
            yield srv, client


# ---------------------------------------------------------------------------
# Helpers: the two import paths
# ---------------------------------------------------------------------------


def _wait_for_stage(client, staging_id, target=("completed", "failed"), timeout_s=30):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/pictures/import/staging/{staging_id}/status")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["stage"] in target:
            return last
        time.sleep(0.1)
    raise AssertionError(f"Staging {staging_id} did not reach {target}: {last}")


def _staging_import(client, files, keys=None, target=("completed",)):
    """Run the streaming-staging import over ``files`` and return its status."""
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", (name, data, "image/png")) for name, data in files],
        data={"key": keys} if keys else None,
    )
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    return _wait_for_stage(client, staging_id, target=target)


def _wait_for_import_task(client, task_id, timeout_s=30):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/pictures/import/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.1)
    raise AssertionError(f"Import task {task_id} did not finish: {last}")


def _one_shot_import(client, files):
    """Run the one-shot ``POST /pictures/import`` and return its final status."""
    r = client.post(
        f"{API}/pictures/import",
        files=[("file", (name, data, "image/png")) for name, data in files],
    )
    assert r.status_code == 200, r.text
    return _wait_for_import_task(client, r.json()["task_id"])


def _picture_ids(srv, deleted=None):
    def _do(session):
        query = select(Picture.id)
        if deleted is not None:
            query = query.where(Picture.deleted.is_(deleted))
        return list(session.exec(query).all())

    return sorted(int(i) for i in srv.vault.db.run_task(_do))


def _row_count(srv) -> int:
    return srv.vault.db.run_task(lambda s: s.query(Picture).count())


def _picture_named(srv, original_file_name) -> int:
    def _do(session):
        return session.exec(
            select(Picture.id).where(Picture.original_file_name == original_file_name)
        ).one()

    return int(srv.vault.db.run_task(_do))


def _vault_path(srv, picture_id) -> str:
    file_path = srv.vault.db.run_task(
        lambda s: s.exec(
            select(Picture.file_path).where(Picture.id == picture_id)
        ).one()
    )
    return os.path.normpath(
        ImageUtils.resolve_picture_path(srv.vault.image_root, file_path)
    )


def _scrapheap(client, picture_id):
    r = client.delete(f"{API}/pictures/{picture_id}")
    assert r.status_code == 200, r.text
    return r.json()


def _purge_forever(client, ids):
    """Preview → redeem the confirm token → delete forever."""
    r = client.post(f"{API}/pictures/scrapheap/delete-preview", json={"ids": ids})
    assert r.status_code == 200, r.text
    token = r.json()["confirm_token"]
    r = client.request(
        "DELETE",
        f"{API}/pictures/scrapheap",
        json={"ids": ids, "confirm_token": token, "include_protected": True},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# The bug: a scrapheaped match must not become a second row
# ---------------------------------------------------------------------------


def test_staging_reimport_of_scrapheaped_picture_creates_no_second_row(owner_server):
    """Streaming-staging path: the match is reported in its own bucket."""
    srv, client = owner_server
    data = _png_bytes(color=(11, 22, 33))

    status = _staging_import(client, [("a.png", data)])
    assert status["imported_count"] == 1, status
    original_id = _picture_ids(srv)[0]

    _scrapheap(client, original_id)
    assert _picture_ids(srv, deleted=True) == [original_id]

    status = _staging_import(client, [("a-again.png", data)])
    assert status["imported_count"] == 0, status
    # NOT an ordinary duplicate: a distinct outcome the user can act on.
    assert status["duplicate_count"] == 0, status
    assert status["scrapheaped_count"] == 1, status
    assert status["scrapheaped_picture_ids"] == [original_id], status

    assert _row_count(srv) == 1, "The scrapheaped picture must not be duplicated"
    assert _picture_ids(srv, deleted=False) == [original_id], (
        "Re-importing a scrapheaped picture must restore it"
    )


def test_one_shot_reimport_of_scrapheaped_picture_creates_no_second_row(owner_server):
    """One-shot ``POST /pictures/import``: the path that carried the bug."""
    srv, client = owner_server
    data = _png_bytes(color=(44, 55, 66))

    status = _one_shot_import(client, [("a.png", data)])
    assert status["imported_count"] == 1, status
    original_id = _picture_ids(srv)[0]

    _scrapheap(client, original_id)

    status = _one_shot_import(client, [("a-again.png", data)])
    assert status["imported_count"] == 0, status
    assert status["duplicate_count"] == 0, status
    assert status["scrapheaped_count"] == 1, status
    assert status["scrapheaped_picture_ids"] == [original_id], status
    assert [entry["status"] for entry in status["results"]] == ["scrapheaped"], status

    assert _row_count(srv) == 1, "The scrapheaped picture must not be duplicated"
    assert _picture_ids(srv, deleted=False) == [original_id], (
        "Re-importing a scrapheaped picture must restore it"
    )


# ---------------------------------------------------------------------------
# The other direction: over-matching is its own regression
# ---------------------------------------------------------------------------


def test_new_file_still_imports_next_to_a_scrapheaped_match(owner_server):
    """A genuinely new file must still import while a sibling matches the heap."""
    srv, client = owner_server
    old = _png_bytes(color=(1, 2, 3))
    fresh = _png_bytes(color=(250, 240, 230))

    _staging_import(client, [("old.png", old)])
    original_id = _picture_ids(srv)[0]
    _scrapheap(client, original_id)

    status = _staging_import(client, [("old-again.png", old), ("fresh.png", fresh)])
    assert status["imported_count"] == 1, status
    assert status["scrapheaped_count"] == 1, status
    assert status["duplicate_count"] == 0, status
    assert status["failed_count"] == 0, status
    assert _row_count(srv) == 2, "The new file must still become a picture"
    assert len(_picture_ids(srv, deleted=False)) == 2


def test_buckets_are_disjoint_and_sum_to_the_file_total(owner_server):
    """new + live duplicate + scrapheaped match, in one batch."""
    srv, client = owner_server
    live = _png_bytes(color=(9, 9, 9))
    heaped = _png_bytes(color=(80, 80, 80))
    fresh = _png_bytes(color=(200, 30, 30))

    _staging_import(client, [("live.png", live), ("heaped.png", heaped)])
    ids = _picture_ids(srv)
    heaped_id = srv.vault.db.run_task(
        lambda s: s.exec(
            select(Picture.id).where(Picture.original_file_name == "heaped.png")
        ).first()
    )
    assert heaped_id in ids
    _scrapheap(client, heaped_id)

    status = _staging_import(
        client,
        [("live2.png", live), ("heaped2.png", heaped), ("fresh.png", fresh)],
    )
    assert status["imported_count"] == 1, status
    assert status["duplicate_count"] == 1, status
    assert status["scrapheaped_count"] == 1, status
    assert status["failed_count"] == 0, status
    assert (
        status["imported_count"]
        + status["duplicate_count"]
        + status["scrapheaped_count"]
        + status["failed_count"]
        + (status["cancelled_count"] or 0)
        == status["total"]
        == 3
    ), status


def test_one_shot_buckets_sum_to_the_file_total(owner_server):
    """Same arithmetic on the one-shot path (imported + duplicate + scrapheaped)."""
    srv, client = owner_server
    live = _png_bytes(color=(7, 7, 7))
    heaped = _png_bytes(color=(70, 70, 70))
    fresh = _png_bytes(color=(170, 30, 30))

    _one_shot_import(client, [("live.png", live), ("heaped.png", heaped)])
    heaped_id = srv.vault.db.run_task(
        lambda s: s.exec(
            select(Picture.id).where(Picture.original_file_name == "heaped.png")
        ).first()
    )
    _scrapheap(client, heaped_id)

    status = _one_shot_import(
        client,
        [("live2.png", live), ("heaped2.png", heaped), ("fresh.png", fresh)],
    )
    assert status["imported_count"] == 1, status
    assert status["duplicate_count"] == 1, status
    assert status["scrapheaped_count"] == 1, status
    assert (
        status["imported_count"]
        + status["duplicate_count"]
        + status["scrapheaped_count"]
        == status["total"]
        == 3
    ), status


# ---------------------------------------------------------------------------
# Edge cases settled deliberately
# ---------------------------------------------------------------------------


def test_several_copies_of_one_scrapheaped_picture_count_per_file(owner_server):
    """Counts are per FILE; the restored ids are per PICTURE."""
    srv, client = owner_server
    data = _png_bytes(color=(123, 45, 67))

    _staging_import(client, [("a.png", data)])
    original_id = _picture_ids(srv)[0]
    _scrapheap(client, original_id)

    status = _staging_import(
        client, [("a1.png", data), ("a2.png", data), ("a3.png", data)]
    )
    assert status["scrapheaped_count"] == 3, status
    assert status["scrapheaped_picture_ids"] == [original_id], status
    assert status["imported_count"] == 0 and status["duplicate_count"] == 0, status
    assert _row_count(srv) == 1
    assert _picture_ids(srv, deleted=False) == [original_id]


def test_purged_and_ledgered_file_imports_as_new_and_is_not_resurrected(owner_server):
    """Delete-forever removes the row, so a re-import is a NEW picture."""
    srv, client = owner_server
    data = _png_bytes(color=(210, 105, 30))

    _staging_import(client, [("gone.png", data)])
    original_id = _picture_ids(srv)[0]
    original_path = srv.vault.db.run_task(
        lambda s: s.get(Picture, original_id).file_path
    )
    _scrapheap(client, original_id)
    outcome = _purge_forever(client, [original_id])
    assert outcome["deleted_count"] == 1, outcome
    assert _row_count(srv) == 0

    ledger_before = srv.vault.db.run_task(lambda s: s.query(DeletedFileLog).count())
    assert ledger_before >= 1, "the purge must have written the permanent-delete ledger"

    status = _staging_import(client, [("gone.png", data)])
    # Nothing to match and nothing to restore: the ledger stops a SNAPSHOT
    # RESTORE resurrecting the destroyed row, not the owner re-importing a file
    # they still have.
    assert status["scrapheaped_count"] == 0, status
    assert status["duplicate_count"] == 0, status
    assert status["imported_count"] == 1, status

    new_ids = _picture_ids(srv)
    assert len(new_ids) == 1
    # SQLite hands the emptied table's rowid back out, so identity is judged on
    # the vault file, not on the integer id: a resurrected row would carry the
    # destroyed picture's path, a fresh import gets its own uuid.
    fresh = srv.vault.db.run_task(lambda s: s.get(Picture, new_ids[0]))
    assert fresh.file_path != original_path, (
        "the purged row must not come back; this is a new picture"
    )
    assert fresh.deleted is False
    assert (
        srv.vault.db.run_task(lambda s: s.query(DeletedFileLog).count())
        == ledger_before
    ), "a fresh import must not retract the permanent-deletion ledger"


def test_live_row_wins_over_a_scrapheaped_row_with_the_same_hash(owner_server):
    """Both rows can exist (the old bug made them). A live match is a duplicate."""
    srv, client = owner_server
    data = _png_bytes(color=(15, 240, 90))

    _staging_import(client, [("a.png", data)])
    live_id = _picture_ids(srv)[0]

    # Forge the pre-fix state: a second, soft-deleted row with the same content.
    def _clone_as_deleted(session):
        live = session.get(Picture, live_id)
        clone = Picture(
            file_path=live.file_path,
            pixel_sha=live.pixel_sha,
            size_bytes=live.size_bytes,
            deleted=True,
        )
        session.add(clone)
        session.commit()
        session.refresh(clone)
        return clone.id

    heaped_id = srv.vault.db.run_task(_clone_as_deleted)

    status = _staging_import(client, [("a-again.png", data)])
    assert status["duplicate_count"] == 1, status
    assert status["scrapheaped_count"] == 0, status
    assert status["scrapheaped_picture_ids"] == [], status
    assert _row_count(srv) == 2, "no third row"
    assert _picture_ids(srv, deleted=True) == [heaped_id]


def test_scrapheaped_member_of_a_locked_set_is_still_matched(owner_server):
    """A lock freezes edits and destruction, not the fact that content exists.

    Detection must not consult locks: hiding the match would put the re-imported
    copy back on disk, which is the exact doubling this change exists to stop.
    """
    srv, client = owner_server
    data = _png_bytes(color=(60, 20, 180))

    _staging_import(client, [("a.png", data)])
    original_id = _picture_ids(srv)[0]
    _scrapheap(client, original_id)

    def _lock_set_around(session, picture_id):
        picture_set = PictureSet(name="frozen")
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        picture_set.locked = True
        session.add(picture_set)
        session.commit()
        return picture_set.id

    srv.vault.db.run_task(_lock_set_around, original_id)

    status = _staging_import(client, [("a-again.png", data)])
    assert status["scrapheaped_count"] == 1, status
    assert status["scrapheaped_picture_ids"] == [original_id], status
    assert _row_count(srv) == 1


def test_scrapheaped_stack_member_refolds_when_the_import_restores_it(owner_server):
    """A stacked match is re-folded into its stack ordering by the restore."""
    srv, client = owner_server
    leader = _png_bytes(color=(5, 5, 5))
    member = _png_bytes(color=(6, 6, 6))

    _staging_import(client, [("leader.png", leader), ("member.png", member)])

    leader_id, member_id = _picture_ids(srv)
    r = client.post(f"{API}/stacks", json={"picture_ids": [leader_id, member_id]})
    assert r.status_code == 200, r.text
    stack_id = srv.vault.db.run_task(lambda s: s.get(Picture, leader_id).stack_id)
    assert stack_id is not None
    _scrapheap(client, member_id)

    status = _staging_import(client, [("member-again.png", member)])
    assert status["scrapheaped_count"] == 1, status
    assert status["scrapheaped_picture_ids"] == [member_id], status
    assert _row_count(srv) == 2, "no third row inside the stack"
    assert _picture_ids(srv, deleted=False) == [leader_id, member_id]

    positions = srv.vault.db.run_task(
        lambda s: {
            int(p.id): p.stack_position
            for p in s.exec(select(Picture).where(Picture.stack_id == stack_id)).all()
        }
    )
    assert set(positions) == {leader_id, member_id}
    assert sorted(positions.values()) == [0, 1], (
        f"restore must re-fold the member into the stack ordering: {positions}"
    )


# ---------------------------------------------------------------------------
# The restore is the shipped one: recorded, and undoable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("importer", ["staging", "one_shot"])
def test_import_restore_is_one_undoable_operation(owner_server, importer):
    """Undoing the import's restore puts the picture back in the Scrapheap."""
    srv, client = owner_server
    run_import = _staging_import if importer == "staging" else _one_shot_import
    data = _png_bytes(color=(33, 99, 200))

    run_import(client, [("a.png", data)])
    original_id = _picture_ids(srv)[0]
    _scrapheap(client, original_id)

    run_import(client, [("a-again.png", data)])
    assert _picture_ids(srv, deleted=False) == [original_id]

    r = client.get(
        f"{API}/operations", params={"op_type": "pictures.scrapheap.restore"}
    )
    assert r.status_code == 200, r.text
    assert [op["target_ids"] for op in r.json()] == [[original_id]], r.json()

    r = client.post(f"{API}/operations/undo")
    assert r.status_code == 200, r.text
    assert _picture_ids(srv, deleted=True) == [original_id]


def test_one_shot_restore_leaves_the_picture_as_it_was(owner_server):
    """A restored match does not join the import's project: it comes back as it
    was, as on the staging path (it may sit in a locked set)."""
    srv, client = owner_server
    data = _png_bytes(color=(150, 10, 220))

    _one_shot_import(client, [("a.png", data)])
    original_id = _picture_ids(srv)[0]
    _scrapheap(client, original_id)

    r = client.post(f"{API}/projects", json={"name": "restored-into"})
    assert r.status_code == 200, r.text
    project_id = r.json()["id"]

    r = client.post(
        f"{API}/pictures/import",
        files=[("file", ("a-again.png", data, "image/png"))],
        data={"project_id": str(project_id)},
    )
    assert r.status_code == 200, r.text
    status = _wait_for_import_task(client, r.json()["task_id"])
    assert status["scrapheaped_picture_ids"] == [original_id], status
    assert _picture_ids(srv, deleted=False) == [original_id]

    members = srv.vault.db.run_task(
        lambda s: list(
            s.exec(
                select(PictureProjectMember.picture_id).where(
                    PictureProjectMember.project_id == project_id
                )
            ).all()
        )
    )
    assert members == []


def test_restore_reports_only_the_pictures_it_restored(owner_server):
    """A match purged before the restore lands is not reported as restored."""
    srv, client = owner_server
    _staging_import(
        client, [("a.png", _png_bytes((1, 1, 200))), ("b.png", _png_bytes((2, 2, 201)))]
    )
    heaped_id, purged_id = _picture_ids(srv)
    _scrapheap(client, heaped_id)
    _scrapheap(client, purged_id)
    _purge_forever(client, [purged_id])

    restored = scrapheap_service.restore_pictures(srv.vault, [heaped_id, purged_id])
    assert restored == [heaped_id]


# ---------------------------------------------------------------------------
# Authz: import status is owner data, not a progress counter
#
# Both status routes were declared ANY_TOKEN on the grounds that they return
# "progress/stage/counts only". They do not: the completed payloads carry
# `results[].picture_id`, `results[].file` (the vault-relative filename) and
# `scrapheaped_picture_ids`, for pictures anywhere in the vault. A resource-
# scoped READ token that is refused a picture's thumbnail was handed that same
# picture's id and filename here. Both are now OWNER_ONLY at the gate.
#
# Both directions, on both entry points a share token has (the Authorization
# header and the ?token= query parameter), because a gate that covered one is a
# hole rather than a policy. The owner side is asserted on a REAL completed task
# so this cannot pass by returning 404 to everyone.
# ---------------------------------------------------------------------------


def _scoped_read_token(client, srv, picture_id) -> str:
    """A READ token scoped to a picture set holding exactly *picture_id*."""
    r = client.post(f"{API}/picture_sets", json={"name": "shared-with-a-guest"})
    assert r.status_code == 200, r.text
    set_id = r.json()["picture_set"]["id"]

    def add_member(session):
        session.add(PictureSetMember(set_id=set_id, picture_id=picture_id))
        session.commit()

    srv.vault.db.run_task(add_member)
    r = client.post(
        f"{API}/users/me/token",
        json={
            "description": "share",
            "scope": "READ",
            "resource_type": "picture_set",
            "resource_id": set_id,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_import_status_is_owner_only_and_leaks_no_out_of_scope_ids(owner_server):
    """Out-of-scope (scoped READ token) → 403; the owner still reads it → 200."""
    srv, owner = owner_server
    r = owner.post(
        f"{API}/pictures/import",
        files=[
            ("file", ("shared.png", _png_bytes((9, 9, 9)), "image/png")),
            ("file", ("private.png", _png_bytes((77, 88, 99)), "image/png")),
        ],
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    status = _wait_for_import_task(owner, task_id)
    assert status["status"] == "completed", status

    shared_id, private_id = _picture_ids(srv)
    token = _scoped_read_token(owner, srv, shared_id)
    scoped = TestClient(srv.api, raise_server_exceptions=True)

    # The scope is real: the token reaches its own picture and not the other.
    assert (
        scoped.get(
            f"{API}/pictures/{shared_id}/metadata",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 200
    )
    assert (
        scoped.get(
            f"{API}/pictures/{private_id}/metadata",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 403
    )

    # ...so it must not read the import receipt naming that same picture, on
    # either entry point a share token has.
    status_url = f"{API}/pictures/import/status"
    by_header = scoped.get(
        status_url,
        params={"task_id": task_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    by_query = scoped.get(status_url, params={"task_id": task_id, "token": token})
    for r in (by_header, by_query):
        assert r.status_code == 403, (
            f"a scoped READ token must not read import status, got "
            f"{r.status_code}: {r.text}"
        )
        assert str(private_id) not in r.text

    # The owner is not over-blocked, and the payload really is per-object data.
    r = owner.get(f"{API}/pictures/import/status", params={"task_id": task_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {entry["picture_id"] for entry in body["results"]} == {
        shared_id,
        private_id,
    }, body
    assert all(entry.get("file") for entry in body["results"]), body

    # A status request that names no task is malformed, not "task not found".
    r = owner.get(status_url)
    assert r.status_code == 422, r.text


def test_staging_status_is_owner_only_and_the_owner_still_polls_it(owner_server):
    """Same contract on the async staging sibling, both directions."""
    srv, owner = owner_server
    staging_id = owner.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    r = owner.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("a.png", _png_bytes((5, 6, 7)), "image/png"))],
    )
    assert r.status_code == 200, r.text
    assert owner.post(f"{API}/pictures/import/staging/{staging_id}/commit").status_code
    _wait_for_stage(owner, staging_id, target=("completed",))

    picture_id = _picture_ids(srv)[0]
    token = _scoped_read_token(owner, srv, picture_id)
    scoped = TestClient(srv.api, raise_server_exceptions=True)
    url = f"{API}/pictures/import/staging/{staging_id}/status"

    r = scoped.get(url, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403, r.text
    r = scoped.get(url, params={"token": token})
    assert r.status_code == 403, r.text

    # The owner is not over-blocked: the same poll the import UI runs still works.
    r = owner.get(url)
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "completed", r.json()


# ---------------------------------------------------------------------------
# Sampled SHA is a candidate key, never final file identity
# ---------------------------------------------------------------------------


def test_one_shot_same_size_sampled_collision_imports_both(owner_server, monkeypatch):
    srv, client = owner_server
    first = _bmp_bytes((1, 2, 3))
    second = _bmp_bytes((4, 5, 6))
    assert len(first) == len(second)
    # The one-shot import spools each upload to a file and hashes that file.
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-sampled-collision"),
    )

    status = _one_shot_import(client, [("a.bmp", first), ("b.bmp", second)])

    assert status["imported_count"] == 2, status
    assert status["duplicate_count"] == 0, status
    assert _row_count(srv) == 2


def test_one_shot_different_size_sampled_collision_imports_both(
    owner_server, monkeypatch
):
    srv, client = owner_server
    first = _bmp_bytes((7, 8, 9), (48, 48))
    second = _bmp_bytes((7, 8, 9), (49, 48))
    assert len(first) != len(second)
    # The one-shot import spools each upload to a file and hashes that file.
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-sampled-collision"),
    )

    status = _one_shot_import(client, [("a.bmp", first), ("b.bmp", second)])

    assert status["imported_count"] == 2, status
    assert status["duplicate_count"] == 0, status
    assert _row_count(srv) == 2


def test_full_hash_mismatch_does_not_restore_from_the_scrapheap(
    owner_server, monkeypatch
):
    srv, client = owner_server
    old = _bmp_bytes((10, 20, 30))
    different = _bmp_bytes((30, 20, 10))
    assert len(old) == len(different)
    # The one-shot import spools each upload to a file and hashes that file.
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-sampled-collision"),
    )

    assert _one_shot_import(client, [("old.bmp", old)])["imported_count"] == 1
    _scrapheap(client, _picture_ids(srv)[0])
    status = _one_shot_import(client, [("different.bmp", different)])

    assert status["imported_count"] == 1, status
    assert status["scrapheaped_count"] == 0, status
    assert status["scrapheaped_picture_ids"] == [], status
    assert _row_count(srv) == 2


def test_staging_and_comfy_batches_keep_sampled_collisions_distinct(
    owner_server, monkeypatch, tmp_path
):
    srv, client = owner_server
    first = _bmp_bytes((40, 50, 60))
    second = _bmp_bytes((60, 50, 40))
    assert len(first) == len(second)
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-staging-collision"),
    )

    status = _staging_import(client, [("a.bmp", first), ("b.bmp", second)])
    assert status["imported_count"] == 2, status
    assert status["duplicate_count"] == 0, status

    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-comfy-collision"),
    )
    third = tmp_path / "test-third.bmp"
    third.write_bytes(_bmp_bytes((70, 80, 90)))
    fourth = tmp_path / "test-fourth.bmp"
    fourth.write_bytes(_bmp_bytes((90, 80, 70)))
    outputs = [(str(third), ".bmp"), (str(fourth), ".bmp")]
    new_ids, duplicate_ids = _import_comfyui_outputs(srv, outputs)
    assert len(new_ids) == 2
    assert duplicate_ids == []

    # A generated picture has no file name of its own: the name it was
    # fetched under is not recorded as one.
    made = srv.vault.db.run_immediate_read_task(
        lambda session: session.exec(
            select(Picture).where(Picture.id.in_(new_ids))
        ).all()
    )
    assert [pic.original_file_name for pic in made] == [None, None]
    published = set()
    for pic in made:
        stored = os.path.join(srv.vault.image_root, *pic.file_path.split("/"))
        with open(stored, "rb") as handle:
            published.add(handle.read())
    assert published == {third.read_bytes(), fourth.read_bytes()}

    repeated_new, repeated_duplicates = _import_comfyui_outputs(srv, outputs)
    assert repeated_new == []
    assert sorted(repeated_duplicates) == sorted(new_ids)


def test_a_comfy_output_matching_a_scrapheaped_picture_brings_it_back(
    owner_server, key_list_plan, tmp_path
):
    """A ComfyUI batch looks its files up by content hash, the Scrapheap
    included, as every other import does: an output matching a scrapheaped
    picture restores that picture rather than adding a second row.

    Wrong if the second run adds a row, or leaves the matched picture in the
    Scrapheap: a repeated seed then doubles the bytes and refills the
    duplicate queue the cleanup emptied.
    """
    srv, client = owner_server
    kept = _png_bytes((11, 22, 33))
    heaped = _png_bytes((44, 55, 66))
    kept_file = tmp_path / "test-kept.png"
    kept_file.write_bytes(kept)
    heaped_file = tmp_path / "test-heaped.png"
    heaped_file.write_bytes(heaped)
    outputs = [(str(kept_file), ".png"), (str(heaped_file), ".png")]
    kept_id, heaped_id = _import_comfyui_outputs(srv, outputs)[0]
    _scrapheap(client, heaped_id)
    assert _picture_ids(srv, deleted=True) == [heaped_id]
    kept_key = (ImageUtils.calculate_hash_from_bytes(kept), len(kept))
    heaped_key = (ImageUtils.calculate_hash_from_bytes(heaped), len(heaped))
    plan = (
        srv.vault.db._engine,
        "picture.deleted \nFROM picture \nWHERE picture.pixel_sha IN",
        "ix_picture_pixel_sha",
    )
    emitted: list[dict] = []
    real_notify = srv.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and data.get("change_kind") == "restored":
            emitted.append(data)
        return real_notify(event_type, data)

    srv.vault.notify = _capture
    try:
        with key_list_plan(*plan):
            new_ids, duplicate_ids = _import_comfyui_outputs(srv, outputs)
    finally:
        srv.vault.notify = real_notify

    assert new_ids == []
    assert duplicate_ids == [kept_id, heaped_id]
    assert _row_count(srv) == 2
    assert _picture_ids(srv, deleted=True) == []
    assert [payload["picture_ids"] for payload in emitted] == [[heaped_id]]

    def matched(include_deleted):
        candidates = srv.vault.db.run_immediate_read_task(
            load_match_candidates_in_session,
            [kept_key, heaped_key, (None, 0)],
            include_deleted,
        )
        return {key: [match.id for match in rows] for key, rows in candidates.items()}

    expected = {kept_key: [kept_id], heaped_key: [heaped_id]}
    with key_list_plan(*plan):
        assert matched(False) == expected
    assert matched(True) == expected
    assert (
        srv.vault.db.run_immediate_read_task(
            load_match_candidates_in_session, [(None, 0)], False
        )
        == {}
    )


def test_restore_announces_exactly_the_pictures_it_restored(owner_server):
    """The ``restored`` event names what came back; a no-op restore says nothing."""
    srv, client = owner_server
    _staging_import(
        client,
        [("a.png", _png_bytes((3, 3, 202))), ("b.png", _png_bytes((3, 3, 101)))],
    )
    heaped_id, live_id = _picture_ids(srv)
    _scrapheap(client, heaped_id)

    emitted: list[dict] = []
    real_notify = srv.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and data.get("change_kind") == "restored":
            emitted.append(data)
        return real_notify(event_type, data)

    srv.vault.notify = _capture
    try:
        # The request names a live picture too; the event names only the one
        # that came back.
        r = client.post(
            f"{API}/pictures/scrapheap/restore",
            json={"picture_ids": [heaped_id, live_id]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["restored_count"] == 1, r.text
        assert [event["picture_ids"] for event in emitted] == [[heaped_id]]

        emitted.clear()
        r = client.post(
            f"{API}/pictures/scrapheap/restore", json={"picture_ids": [heaped_id]}
        )
        assert r.json()["restored_count"] == 0, r.text
        assert emitted == []
    finally:
        srv.vault.notify = real_notify


# ---------------------------------------------------------------------------
# Generated outputs (plugin runs, edit copies) match on the full hash
# ---------------------------------------------------------------------------


def _capture_restored(srv):
    """Collect the ``restored`` payloads; returns ``(emitted, undo)``."""
    emitted: list[dict] = []
    real_notify = srv.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and data.get("change_kind") == "restored":
            emitted.append(data)
        return real_notify(event_type, data)

    srv.vault.notify = _capture

    def undo():
        srv.vault.notify = real_notify

    return emitted, undo


def test_plugin_outputs_keep_a_sampled_collision_distinct(owner_server, monkeypatch):
    """A plugin output is a duplicate only on the full hash, as every import is.

    Wrong if the different output with the colliding sampled hash comes back
    as a duplicate of the earlier picture: the plugin's result is then lost
    and the grid says "already in your library".
    """
    srv, _client = owner_server
    earlier = _bmp_bytes((10, 20, 30))
    colliding = _bmp_bytes((30, 20, 10))
    assert len(earlier) == len(colliding)
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_bytes",
        staticmethod(lambda _data: "forced-plugin-collision"),
    )
    (earlier_id,), _dups, _ordered = _import_output_images(srv, [(earlier, ".bmp")])

    new_ids, duplicate_ids, ordered = _import_output_images(
        srv, [(colliding, ".bmp"), (earlier, ".bmp")]
    )

    assert len(new_ids) == 1 and new_ids[0] != earlier_id
    assert duplicate_ids == [earlier_id]
    assert ordered == [new_ids[0], earlier_id]
    assert _row_count(srv) == 2


def test_a_plugin_output_matching_a_scrapheaped_picture_brings_it_back(owner_server):
    """An output matching a scrapheaped picture restores it, as ComfyUI does.

    Wrong if a second row appears, or the matched picture stays in the
    Scrapheap, or the restore is not announced.
    """
    srv, client = owner_server
    data = _png_bytes((12, 34, 56))
    (heaped_id,), _dups, _ordered = _import_output_images(srv, [(data, ".png")])
    _scrapheap(client, heaped_id)
    emitted, undo = _capture_restored(srv)
    try:
        new_ids, duplicate_ids, ordered = _import_output_images(srv, [(data, ".png")])
    finally:
        undo()

    assert (new_ids, duplicate_ids, ordered) == ([], [heaped_id], [heaped_id])
    assert _picture_ids(srv) == [heaped_id]
    assert _picture_ids(srv, deleted=True) == []
    assert [payload["picture_ids"] for payload in emitted] == [[heaped_id]]


def test_identical_plugin_outputs_in_one_batch_add_one_picture(owner_server):
    """Two identical outputs in one run are one picture, named for both.

    Wrong if two rows appear, or the output list loses its alignment with the
    sources (the run copies sets and the source link by position).
    """
    srv, _client = owner_server
    data = _png_bytes((65, 43, 21))
    other = _png_bytes((21, 43, 65))

    new_ids, duplicate_ids, ordered = _import_output_images(
        srv, [(data, ".png"), (other, ".png"), (data, ".png")]
    )

    assert len(new_ids) == 2
    assert duplicate_ids == [new_ids[0]]
    assert ordered == [new_ids[0], new_ids[1], new_ids[0]]
    assert _row_count(srv) == 2


def test_a_plugin_output_whose_heaped_match_was_purged_meanwhile_imports_as_new(
    owner_server, monkeypatch
):
    """The match was purged between the lookup and the restore: no row holds
    the content any more, so the output is a new picture.

    Wrong if the output is reported as the purged id (a dead id in the run's
    result), or dropped so the output list is one short.
    """
    srv, client = owner_server
    data = _png_bytes((77, 66, 55))
    (heaped_id,), _dups, _ordered = _import_output_images(srv, [(data, ".png")])
    _scrapheap(client, heaped_id)
    real_restore = scrapheap_service.restore_pictures

    def _purge_first(vault, ids, **kwargs):
        _purge_forever(client, list(ids))
        return real_restore(vault, ids, **kwargs)

    monkeypatch.setattr(scrapheap_service, "restore_pictures", _purge_first)

    new_ids, duplicate_ids, ordered = _import_output_images(srv, [(data, ".png")])

    # SQLite hands the purged row's id to the next insert, so the id cannot
    # tell the new row from the purged one; the duplicate list can.
    assert len(new_ids) == 1
    assert duplicate_ids == [], "reported as the purged picture"
    assert ordered == new_ids
    assert _picture_ids(srv) == new_ids


def test_a_plugin_output_whose_heaped_match_came_back_meanwhile_is_a_duplicate(
    owner_server, monkeypatch
):
    """The owner restored the match first, so this restore brought nothing
    back: the picture is live and the output is its duplicate.

    Wrong if a second row appears.
    """
    srv, client = owner_server
    data = _png_bytes((55, 66, 77))
    (heaped_id,), _dups, _ordered = _import_output_images(srv, [(data, ".png")])
    _scrapheap(client, heaped_id)
    real_restore = scrapheap_service.restore_pictures

    def _restored_by_the_owner_first(vault, ids, **kwargs):
        assert real_restore(vault, ids, **kwargs) == list(ids)
        return real_restore(vault, ids, **kwargs)

    monkeypatch.setattr(
        scrapheap_service, "restore_pictures", _restored_by_the_owner_first
    )

    new_ids, duplicate_ids, ordered = _import_output_images(srv, [(data, ".png")])

    assert (new_ids, duplicate_ids, ordered) == ([], [heaped_id], [heaped_id])
    assert _picture_ids(srv) == [heaped_id]


# ---------------------------------------------------------------------------
# Generated outputs (plugin runs, ComfyUI): the restore decides, and a failed
# insert leaves no file behind
# ---------------------------------------------------------------------------

_GENERATED_IMPORTERS = ["plugin", "comfyui"]


def _import_generated(srv, importer, datas, tmp_path):
    """Import *datas* as one run's outputs; return ``(new ids, duplicate ids)``."""
    if importer == "plugin":
        new_ids, duplicate_ids, _ordered = _import_output_images(
            srv, [(data, ".png") for data in datas]
        )
        return new_ids, duplicate_ids
    entries = []
    for data in datas:
        path = tmp_path / f"test-output-{time.monotonic_ns()}.png"
        path.write_bytes(data)
        entries.append((str(path), ".png"))
    return _import_comfyui_outputs(srv, entries)


def _fail_the_output_commit(srv, monkeypatch):
    """Make the outputs' insert fail at its commit, as a full disk would."""
    real_run_task = srv.vault.db.run_task

    def run_task(func, *args, **kwargs):
        if getattr(func, "__name__", "") in ("persist", "import_task"):
            real_insert = func

            def func(session, *inner_args, **inner_kwargs):
                def commit():
                    raise RuntimeError("test-insert-failure")

                session.commit = commit
                return real_insert(session, *inner_args, **inner_kwargs)

        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(srv.vault.db, "run_task", run_task)


def _library_files_with_bytes(srv, data: bytes) -> list[str]:
    """Every file under the image root with the content *data*, row or not."""
    sha = ImageUtils.calculate_full_hash_from_bytes(data)
    found = []
    for dirpath, _dirs, names in os.walk(srv.vault.image_root):
        for name in names:
            path = os.path.join(dirpath, name)
            if ImageUtils.calculate_full_hash_from_file_path(path) == sha:
                found.append(path)
    return found


@pytest.mark.parametrize("importer", _GENERATED_IMPORTERS)
def test_a_generated_output_whose_insert_fails_leaves_no_file(
    owner_server, importer, tmp_path, monkeypatch
):
    """Left in the image root without its row, the next root scan would
    index the output as a picture without its run's tags and sets."""
    srv, _client = owner_server
    datas = [_png_bytes((91, 92, 93)), _png_bytes((93, 92, 91))]
    _fail_the_output_commit(srv, monkeypatch)

    with pytest.raises(RuntimeError, match="test-insert-failure"):
        _import_generated(srv, importer, datas, tmp_path)

    assert _row_count(srv) == 0
    for data in datas:
        assert _library_files_with_bytes(srv, data) == []


@pytest.mark.parametrize("failing", [1, 2], ids=["logo", "tagger-test-picture"])
def test_a_default_picture_whose_insert_fails_leaves_no_file(
    owner_server, monkeypatch, failing
):
    """The seed pictures are published before their rows: one whose insert
    does not commit has its file removed, and the ones before it keep theirs."""
    srv, _client = owner_server
    real_run_task = srv.vault.db.run_task
    seen = {"inserts": 0}
    failed_paths: list[str] = []

    def run_task(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "add_picture":
            seen["inserts"] += 1
            if seen["inserts"] == failing:
                failed_paths.append(args[0].file_path)
                real_insert = func

                def func(session, *inner_args, **inner_kwargs):
                    def commit():
                        raise RuntimeError("test-insert-failure")

                    session.commit = commit
                    return real_insert(session, *inner_args, **inner_kwargs)

        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(srv.vault.db, "run_task", run_task)

    with pytest.raises(RuntimeError, match="test-insert-failure"):
        srv.vault.import_default_data(add_tagger_test_images=True)

    (failed_path,) = failed_paths
    assert not os.path.exists(
        ImageUtils.resolve_picture_path(srv.vault.image_root, failed_path)
    ), "the file of the seed picture that was not inserted was left"
    kept = _picture_ids(srv)
    assert len(kept) == failing - 1
    for picture_id in kept:
        assert os.path.exists(_vault_path(srv, picture_id))


@pytest.mark.parametrize("importer", _GENERATED_IMPORTERS)
def test_a_generated_output_whose_heaped_match_was_purged_meanwhile_is_new(
    owner_server, importer, tmp_path, monkeypatch
):
    """Purged between the lookup and the restore: no row holds the content,
    so the output is a new picture, not a dead id."""
    srv, client = owner_server
    data = _png_bytes((78, 67, 56))
    (heaped_id,), _dups = _import_generated(srv, importer, [data], tmp_path)
    _scrapheap(client, heaped_id)
    real_restore = scrapheap_service.restore_pictures

    def _purge_first(vault, ids, **kwargs):
        _purge_forever(client, list(ids))
        return real_restore(vault, ids, **kwargs)

    monkeypatch.setattr(scrapheap_service, "restore_pictures", _purge_first)

    new_ids, duplicate_ids = _import_generated(srv, importer, [data], tmp_path)

    assert len(new_ids) == 1
    assert duplicate_ids == [], "reported as the purged picture"
    assert _picture_ids(srv) == new_ids
    assert len(_library_files_with_bytes(srv, data)) == 1


@pytest.mark.parametrize("importer", _GENERATED_IMPORTERS)
def test_a_generated_output_whose_heaped_match_came_back_meanwhile_is_a_duplicate(
    owner_server, importer, tmp_path, monkeypatch
):
    srv, client = owner_server
    data = _png_bytes((56, 67, 78))
    (heaped_id,), _dups = _import_generated(srv, importer, [data], tmp_path)
    _scrapheap(client, heaped_id)
    real_restore = scrapheap_service.restore_pictures

    def _restored_by_the_owner_first(vault, ids, **kwargs):
        assert real_restore(vault, ids, **kwargs) == [heaped_id]
        return real_restore(vault, ids, **kwargs)

    monkeypatch.setattr(
        scrapheap_service, "restore_pictures", _restored_by_the_owner_first
    )

    new_ids, duplicate_ids = _import_generated(srv, importer, [data], tmp_path)

    assert (new_ids, duplicate_ids) == ([], [heaped_id])
    assert _picture_ids(srv) == [heaped_id]


@pytest.mark.parametrize("importer", _GENERATED_IMPORTERS)
def test_a_generated_output_whose_live_match_was_binned_after_the_lookup_restores_it(
    owner_server, importer, tmp_path, monkeypatch
):
    """The lookup saw the match live; the owner binned it before the restore.
    The output still maps to it, so it is brought back, not left in the
    Scrapheap as the picture a live output names."""
    srv, client = owner_server
    data = _png_bytes((34, 45, 56))
    (match_id,), _dups = _import_generated(srv, importer, [data], tmp_path)
    real_partition = import_dedup_service.partition_confirmed_matches

    def _binned_after_the_lookup(*args, **kwargs):
        live, scrapheaped = real_partition(*args, **kwargs)
        assert [match.id for match in live.values()] == [match_id]
        _scrapheap(client, match_id)
        return live, scrapheaped

    monkeypatch.setattr(
        import_dedup_service, "partition_confirmed_matches", _binned_after_the_lookup
    )
    emitted, undo = _capture_restored(srv)
    try:
        new_ids, duplicate_ids = _import_generated(srv, importer, [data], tmp_path)
    finally:
        undo()

    assert (new_ids, duplicate_ids) == ([], [match_id])
    assert _picture_ids(srv, deleted=False) == [match_id]
    assert [payload["picture_ids"] for payload in emitted] == [[match_id]]


@pytest.mark.parametrize("importer", _GENERATED_IMPORTERS)
def test_a_restore_before_a_failed_output_insert_is_harmless(
    owner_server, importer, tmp_path, monkeypatch
):
    """The restore commits before the new outputs' insert. When that insert
    fails, the restored picture stays live, the new output leaves no file, and
    the retried run adds each content once."""
    srv, client = owner_server
    heaped = _png_bytes((23, 34, 45))
    fresh = _png_bytes((45, 34, 23))
    (heaped_id,), _dups = _import_generated(srv, importer, [heaped], tmp_path)
    _scrapheap(client, heaped_id)
    with monkeypatch.context() as failing:
        _fail_the_output_commit(srv, failing)
        with pytest.raises(RuntimeError, match="test-insert-failure"):
            _import_generated(srv, importer, [heaped, fresh], tmp_path)

    assert _picture_ids(srv, deleted=False) == [heaped_id]
    assert _picture_ids(srv, deleted=True) == []
    assert _library_files_with_bytes(srv, fresh) == []

    new_ids, duplicate_ids = _import_generated(srv, importer, [heaped, fresh], tmp_path)

    assert len(new_ids) == 1 and duplicate_ids == [heaped_id]
    assert _picture_ids(srv) == sorted([heaped_id, *new_ids])
    assert len(_library_files_with_bytes(srv, heaped)) == 1
    assert len(_library_files_with_bytes(srv, fresh)) == 1


def test_a_plugin_run_of_more_outputs_than_sqlite_binds_marks_them_all_imported(
    owner_server,
):
    """The plugin request has no cap on its outputs. On a connection pinned to
    SQLite's historical 999 bound parameters, a run of 1000 new outputs still
    inserts them all and stamps each one ``imported_at``."""
    srv, _client = owner_server
    limit = 999
    engine = srv.vault.db._engine

    def pin(dbapi_conn, _record):
        dbapi_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)

    event.listen(engine, "connect", pin)
    engine.dispose()
    try:

        def over_the_ceiling(session):
            session.exec(select(Picture.id).where(Picture.id.in_(range(limit + 1))))

        with pytest.raises(OperationalError, match="too many SQL variables"):
            srv.vault.db.run_task(over_the_ceiling)

        entries = [
            (_png_bytes((i % 256, i // 256, 7)), ".png") for i in range(limit + 1)
        ]
        new_ids, duplicate_ids, _ordered = _import_output_images(srv, entries)
    finally:
        event.remove(engine, "connect", pin)
        engine.dispose()

    def _unstamped(session):
        return session.exec(
            select(Picture.id).where(Picture.imported_at.is_(None))
        ).all()

    assert len(new_ids) == limit + 1 and duplicate_ids == []
    assert _picture_ids(srv) == sorted(new_ids)
    assert srv.vault.db.run_task(_unstamped) == []


@contextmanager
def _bind_ceiling(srv, limit: int):
    """Pin every new connection to *limit* bound parameters, proven to hold."""
    engine = srv.vault.db._engine

    def pin(dbapi_conn, _record):
        dbapi_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)

    event.listen(engine, "connect", pin)
    engine.dispose()
    try:

        def over_the_ceiling(session):
            session.exec(select(Picture.id).where(Picture.id.in_(range(limit + 1))))

        with pytest.raises(OperationalError, match="too many SQL variables"):
            srv.vault.db.run_task(over_the_ceiling)
        yield
    finally:
        event.remove(engine, "connect", pin)
        engine.dispose()


class _InvertPlugin(ImagePlugin):
    name = "test_invert"

    def parameter_schema(self):
        return []

    def run(self, images, parameters=None, **_kwargs):
        return [ImageOps.invert(image.convert("RGB")) for image in images]


def test_a_plugin_run_over_more_sources_than_sqlite_binds_completes(owner_server):
    """A run's sources are the user's selection, with no cap. On a connection
    pinned to SQLite's historical 999 bound parameters, a run over 1000
    sources loads them all and gives every output its source's set, project,
    source id and stack, and its import time, as a run under the ceiling
    does."""
    srv, _client = owner_server
    limit = 999
    sources, _duplicates, _ordered = _import_output_images(
        srv, [(_png_bytes((i % 256, i // 256, 7)), ".png") for i in range(limit + 1)]
    )

    def seed(session):
        project = Project(name="test-project")
        picture_set = PictureSet(name="test-set")
        session.add(project)
        session.add(picture_set)
        session.flush()
        for picture_id in sources:
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
            session.add(
                PictureProjectMember(picture_id=picture_id, project_id=project.id)
            )
        # The last two sources, past the first chunk of ids, share a stack in
        # the reverse of their run order: their outputs go on top in the
        # sources' stack order, which only reading their positions tells.
        stack = PictureStack()
        session.add(stack)
        session.flush()
        for position, picture_id in enumerate(reversed(sources[-2:])):
            picture = session.get(Picture, picture_id)
            picture.stack_id = stack.id
            picture.stack_position = position
            session.add(picture)
        session.commit()
        return picture_set.id, project.id

    set_id, project_id = srv.vault.db.run_task(seed)

    with _bind_ceiling(srv, limit):
        result = apply_plugin_to_pictures(srv, _InvertPlugin(), sources, None)

    outputs = result["output_picture_ids"]
    assert result["errors"] == []
    assert result["duplicate_picture_ids"] == []
    assert sorted(result["created_picture_ids"]) == sorted(outputs)
    assert len(set(outputs)) == limit + 1 and not set(outputs) & set(sources)

    def read(session):
        in_set = session.exec(
            select(PictureSetMember.picture_id).where(PictureSetMember.set_id == set_id)
        ).all()
        in_project = session.exec(
            select(PictureProjectMember.picture_id).where(
                PictureProjectMember.project_id == project_id
            )
        ).all()
        rows = session.exec(
            select(
                Picture.id,
                Picture.source_picture_id,
                Picture.stack_id,
                Picture.stack_position,
                Picture.imported_at,
            )
        ).all()
        return set(in_set), set(in_project), {row[0]: row[1:] for row in rows}

    in_set, in_project, rows = srv.vault.db.run_task(read)
    assert in_set == set(sources) | set(outputs)
    assert in_project == set(sources) | set(outputs)
    for source_id, output_id in zip(sources, outputs):
        source_of_output, stack_of_output, _position, imported_at = rows[output_id]
        assert source_of_output == source_id
        assert stack_of_output is not None
        assert stack_of_output == rows[source_id][1]
        assert imported_at is not None
    shared_stack = [
        (position, picture_id)
        for picture_id, (_source, stack_id, position, _imported) in rows.items()
        if stack_id == rows[sources[-1]][1]
    ]
    assert [picture_id for _position, picture_id in sorted(shared_stack)] == [
        outputs[-1],
        outputs[-2],
        sources[-1],
        sources[-2],
    ]


def test_a_generation_of_more_outputs_than_sqlite_binds_completes(
    owner_server, tmp_path
):
    """A ComfyUI run's output count is the workflow's. On a connection pinned
    to SQLite's historical 999 bound parameters, a run of 1000 outputs adds
    1000 pictures, each with its import time, as a run under the ceiling does."""
    srv, _client = owner_server
    limit = 999
    fetched = []
    for i in range(limit + 1):
        path = tmp_path / f"test-generated-{i}.png"
        # A size no other test here uses, so nothing is a duplicate.
        Image.new("RGB", (31, 23), color=(i % 256, i // 256, 97)).save(path)
        fetched.append((str(path), ".png"))

    with _bind_ceiling(srv, limit):
        new_ids, duplicate_ids = _import_comfyui_outputs(srv, fetched)

    assert duplicate_ids == []
    assert len(set(new_ids)) == limit + 1

    def read(session):
        return dict(session.exec(select(Picture.id, Picture.imported_at)).all())

    imported_at = srv.vault.db.run_task(read)
    assert [pid for pid in new_ids if imported_at.get(pid) is None] == []


def test_a_one_shot_import_of_more_files_than_sqlite_binds_completes(owner_server):
    """An upload has no file cap. On connections pinned to SQLite's historical
    999 bound parameters, a one-shot import of 1000 files with a caption each
    gives every picture its caption tags and its import time, as an import
    under the ceiling does."""
    srv, client = owner_server
    limit = 999
    caption = "test-sidecar-apple, test-sidecar-sky"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for i in range(limit + 1):
            # A size no other test here uses, so nothing is a duplicate.
            image = Image.new("RGB", (37, 29), color=(i % 256, i // 256, 211))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            zip_file.writestr(f"bulk-{i}.png", buf.getvalue())
            zip_file.writestr(f"bulk-{i}.txt", caption)

    with _bind_ceiling(srv, limit):
        r = client.post(
            f"{API}/pictures/import",
            files=[("file", ("bulk.zip", archive.getvalue(), "application/zip"))],
        )
        assert r.status_code == 200, r.text
        status = _wait_for_import_task(client, r.json()["task_id"], timeout_s=300)

    assert status["status"] == "completed", status.get("error")
    assert status["imported_count"] == limit + 1, status["imported_count"]
    new_ids = {
        result["picture_id"]
        for result in status["results"]
        if result["status"] == "success"
    }
    assert len(new_ids) == limit + 1
    wanted = set(_parse_sidecar_tags(caption))
    assert wanted

    def read(session):
        imported_at = dict(session.exec(select(Picture.id, Picture.imported_at)).all())
        tags: dict[int, set[str]] = {}
        for picture_id, tag in session.exec(
            select(Tag.picture_id, Tag.tag).where(Tag.tag.in_(sorted(wanted)))
        ).all():
            tags.setdefault(picture_id, set()).add(tag)
        return imported_at, tags

    imported_at, tags = srv.vault.db.run_task(read)
    assert [pid for pid in new_ids if imported_at.get(pid) is None] == []
    assert [pid for pid in new_ids if tags.get(pid) != wanted] == []


# ---------------------------------------------------------------------------
# A staged import settles its files in batches
# ---------------------------------------------------------------------------


def test_staging_import_looks_up_matches_before_and_under_the_lock_on_the_read_path(
    owner_server, monkeypatch
):
    """Each batch is looked up twice, once before the import lock and once
    under it, both off the writer queue, and each sees the batches before it:
    the owner's import does not queue behind background writes."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 4, raising=False)
    real_lookup = import_dedup_service.load_match_candidates_in_session
    lookups = []

    def _spy(session, keys, *args, **kwargs):
        keys = list(keys)
        lookups.append(
            {
                "on_writer": threading.current_thread() is srv.vault.db._task_worker,
                "keys": len(keys),
                "rows_visible": len(session.exec(select(Picture.id)).all()),
            }
        )
        return real_lookup(session, keys, *args, **kwargs)

    monkeypatch.setattr(import_dedup_service, "load_match_candidates_in_session", _spy)
    files = [(f"p{i}.png", _png_bytes((20 * i, 90, 200))) for i in range(9)]

    status = _staging_import(client, files)

    assert status["imported_count"] == 9, status
    assert [lookup["keys"] for lookup in lookups] == [4, 4, 4, 4, 1, 1], lookups
    assert not any(lookup["on_writer"] for lookup in lookups), lookups
    assert [lookup["rows_visible"] for lookup in lookups] == [0, 0, 4, 4, 8, 8], lookups


def test_staging_import_reads_each_candidate_once_per_batch(owner_server, monkeypatch):
    """Several staged files sharing one candidate read its vault file once."""
    srv, client = owner_server
    first = _bmp_bytes((15, 25, 35))
    other = _bmp_bytes((35, 25, 15))
    assert len(first) == len(other)
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-staging-collision"),
    )
    assert _staging_import(client, [("first.bmp", first)])["imported_count"] == 1
    candidate_path = _vault_path(srv, _picture_ids(srv)[0])

    real_full_hash = ImageUtils.calculate_full_hash_from_file_path
    candidate_reads = []

    def _spy(path):
        if os.path.normpath(path) == candidate_path:
            candidate_reads.append(path)
        return real_full_hash(path)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_file_path", staticmethod(_spy)
    )
    status = _staging_import(
        client,
        [("copy-1.bmp", first), ("copy-2.bmp", first), ("other.bmp", other)],
    )

    assert status["duplicate_count"] == 2, status
    assert status["imported_count"] == 1, status
    assert len(candidate_reads) == 1, candidate_reads


def test_staging_import_names_the_earlier_batch_picture_a_file_duplicates(
    owner_server, monkeypatch
):
    """A repeat in a later batch is a duplicate OF the picture already added."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 2, raising=False)
    x = _png_bytes((1, 120, 240))
    y = _png_bytes((240, 120, 1))

    status = _staging_import(
        client,
        [("x.png", x), ("y.png", y), ("x-again.png", x)],
        keys=["tile-x", "tile-y", "tile-x-again"],
    )

    assert status["imported_count"] == 2, status
    assert status["duplicate_count"] == 1, status
    assert _row_count(srv) == 2
    assert status["outcomes"] == [
        {"key": "tile-x", "outcome": "new", "picture_id": None},
        {"key": "tile-y", "outcome": "new", "picture_id": None},
        {
            "key": "tile-x-again",
            "outcome": "duplicate",
            "picture_id": _picture_named(srv, "x.png"),
        },
    ], status


def test_staging_batches_keep_a_sampled_collision_split_across_them_distinct(
    owner_server, monkeypatch
):
    """A picture added by an earlier batch is matched on full content, not key."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 1, raising=False)
    first = _bmp_bytes((44, 55, 66))
    second = _bmp_bytes((66, 55, 44))
    assert len(first) == len(second)
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-staging-collision"),
    )

    status = _staging_import(client, [("a.bmp", first), ("b.bmp", second)])

    assert status["imported_count"] == 2, status
    assert status["duplicate_count"] == 0, status
    assert _row_count(srv) == 2


def test_staging_import_failure_after_a_batch_says_what_was_added(
    owner_server, monkeypatch, caplog
):
    """Earlier batches stay imported, and the logged failure says how many."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 2, raising=False)
    real_run_task = srv.vault.db.run_task
    inserts = []

    def _failing_second_insert(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "insert_pictures":
            inserts.append(func)
            if len(inserts) == 2:
                raise RuntimeError("test-insert-failure")
        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(srv.vault.db, "run_task", _failing_second_insert)
    files = [(f"f{i}.png", _png_bytes((70, 70 + 40 * i, 9))) for i in range(3)]

    with caplog.at_level("WARNING", logger="pixlstash"):
        status = _staging_import(
            client,
            files,
            keys=["tile-0", "tile-1", "tile-2"],
            target=("completed", "failed"),
        )

    assert status["stage"] == "failed", status
    # The status says how many pictures came in, in the server's own sentence;
    # the cause's text stays in the log.
    assert status["error"] == "The import stopped after 2 new picture(s) were added."
    logged = [
        r.exc_info[1]
        for r in caplog.records
        if r.name == "pixlstash.tasks.base_task" and r.exc_info
    ]
    assert len(logged) == 1, logged
    assert str(logged[0].__cause__) == "test-insert-failure", logged
    assert [o["outcome"] for o in status["outcomes"]] == ["new", "new", "failed"], (
        status
    )
    assert _row_count(srv) == 2


def _fail_the_second_insert(srv, monkeypatch):
    """Make the second batch's insert raise; the first one runs as usual."""
    real_run_task = srv.vault.db.run_task
    inserts = []

    def _failing_second_insert(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "insert_pictures":
            inserts.append(func)
            if len(inserts) == 2:
                raise RuntimeError("test-insert-failure")
        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(srv.vault.db, "run_task", _failing_second_insert)


def test_staging_import_failure_reports_an_in_batch_copy_as_failed(
    owner_server, monkeypatch
):
    """A copy of a file the failed batch would have added is not in the library."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 2, raising=False)
    _fail_the_second_insert(srv, monkeypatch)
    c = _png_bytes((9, 200, 90))
    files = [
        ("a.png", _png_bytes((200, 9, 90))),
        ("b.png", _png_bytes((90, 9, 200))),
        ("c.png", c),
        ("c-copy.png", c),
    ]

    status = _staging_import(
        client,
        files,
        keys=["a", "b", "c", "c-copy"],
        target=("completed", "failed"),
    )

    assert status["stage"] == "failed", status
    assert [o["outcome"] for o in status["outcomes"]] == [
        "new",
        "new",
        "failed",
        "failed",
    ], status
    assert _row_count(srv) == 2


def test_staging_import_failure_announces_the_pictures_it_added(
    owner_server, monkeypatch
):
    """Open views learn about the earlier batches' pictures, failure or not."""
    srv, client = owner_server
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 2, raising=False)
    _fail_the_second_insert(srv, monkeypatch)
    events = []
    real_notify = srv.vault.notify

    def _capture(event_type, data=None):
        events.append((event_type, data))
        return real_notify(event_type, data)

    monkeypatch.setattr(srv.vault, "notify", _capture)
    files = [(f"g{i}.png", _png_bytes((9, 70 + 40 * i, 70))) for i in range(3)]

    status = _staging_import(client, files, target=("completed", "failed"))

    assert status["stage"] == "failed", status
    added = _picture_ids(srv, deleted=False)
    assert len(added) == 2, added
    assert [
        data["ids"]
        for event_type, data in events
        if event_type == EventType.PICTURE_IMPORTED and isinstance(data, dict)
    ] == [added], events
    assert [
        data["picture_ids"]
        for event_type, data in events
        if event_type == EventType.CHANGED_PICTURES
        and isinstance(data, dict)
        and data.get("change_kind") == "added"
    ] == [added], events


def test_staging_import_counts_a_picture_whose_refresh_fails_after_commit(
    owner_server, monkeypatch
):
    """A picture whose insert committed is new, whatever fails after the commit."""
    srv, client = owner_server
    real_refresh = Session.refresh

    def _refresh(self, instance, *args, **kwargs):
        # Only while an import holds the lock, so other server work is untouched.
        if isinstance(instance, Picture) and import_dedup_service._IMPORT_LOCK.locked():
            raise RuntimeError("test-refresh-failure")
        return real_refresh(self, instance, *args, **kwargs)

    monkeypatch.setattr(Session, "refresh", _refresh)

    status = _staging_import(
        client,
        [("r.png", _png_bytes((33, 66, 99)))],
        target=("completed", "failed"),
    )

    assert status["stage"] == "completed", status
    assert [o["outcome"] for o in status["outcomes"]] == ["new"], status
    assert _row_count(srv) == 1


# ---------------------------------------------------------------------------
# Overlapping imports of the same new content add it once
# ---------------------------------------------------------------------------


class _SignallingLock:
    """The import lock, saying which thread holds it and which had to wait."""

    def __init__(self):
        self._inner = threading.Lock()
        self.owner = None
        self.acquired = 0
        self.contended: list[str] = []

    def __enter__(self):
        if not self._inner.acquire(blocking=False):
            self.contended.append(threading.current_thread().name)
            self._inner.acquire()
        self.owner = threading.current_thread()
        self.acquired += 1
        return self

    def __exit__(self, *exc_info):
        self.owner = None
        self._inner.release()
        return False

    def held_here(self) -> bool:
        return self.owner is threading.current_thread()


def _signalling_import_lock(monkeypatch) -> _SignallingLock:
    lock = _SignallingLock()
    monkeypatch.setattr(import_dedup_service, "_IMPORT_LOCK", lock)
    return lock


def _staged_import_task(srv, tmp_path, name, data):
    staging_dir = tmp_path / f"staging-{name}"
    staging_dir.mkdir()
    file_path = staging_dir / name
    file_path.write_bytes(data)
    return picture_import_task.PictureImportTask(
        srv.vault,
        [{"file_path": str(file_path), "original_file_name": name}],
        staging_id=f"test-{name}",
        staging_dir=str(staging_dir),
    )


def _hold_first_publish(monkeypatch):
    """Pause the first import once it published a file, until the second ends.

    The first import to publish a file into the library waits there for the
    other import to finish. It can finish only if the first one holds no
    import lock while it reads and copies: the lock covers the lookup and the
    insert alone. Returns the hold: its lock, and whether each wait ended
    because the other import finished (``waits``) and with the lock held
    (``waited_holding``).
    """
    hold = SimpleNamespace(
        lock=_signalling_import_lock(monkeypatch),
        first_thread=None,
        first_held=threading.Event(),
        second_done=threading.Event(),
        waits=[],
        waited_holding=[],
    )
    real_add = import_publish_service.PublishedPictures.add

    def _add(self, picture):
        result = real_add(self, picture)
        if hold.first_thread is None:
            hold.first_thread = threading.current_thread()
            hold.first_held.set()
            hold.waited_holding.append(hold.lock.held_here())
            hold.waits.append(hold.second_done.wait(timeout=10))
        return result

    monkeypatch.setattr(import_publish_service.PublishedPictures, "add", _add)
    return hold


def _run_overlapping(hold, first, second):
    results = {}
    errors = []

    def _runner(label, func, done=None):
        def _run():
            try:
                results[label] = func()
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        return threading.Thread(target=_run, name=f"test-import-{label}")

    first_thread = _runner("first", first)
    first_thread.start()
    assert hold.first_held.wait(timeout=30), "the first import published nothing"
    second_thread = _runner("second", second, hold.second_done)
    second_thread.start()
    first_thread.join(timeout=60)
    second_thread.join(timeout=60)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert errors == []
    return results


def _assert_the_second_import_won(srv, hold, data) -> int:
    """The second import added the picture; the first published a copy and
    found the second's row when it looked again under the lock."""
    assert hold.waits == [True], "the second import waited on the first one"
    assert hold.waited_holding == [False]
    assert hold.lock.contended == []
    (picture_id,) = _picture_ids(srv)
    assert len(_library_files_with_bytes(srv, data)) == 1, (
        "the copy the first import published was left in the library"
    )
    return picture_id


def _assert_a_staged_duplicate_of(task, result, picture_id) -> None:
    assert result["imported_count"] == 0, result
    assert result["duplicate_count"] == 1, result
    assert task.outcomes == [
        {"key": None, "outcome": "duplicate", "picture_id": picture_id}
    ]


def test_overlapping_staged_imports_of_new_content_add_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, _client = owner_server
    data = _png_bytes((5, 150, 250))
    first = _staged_import_task(srv, tmp_path, "first.png", data)
    second = _staged_import_task(srv, tmp_path, "second.png", data)
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(hold, first._run_task, second._run_task)

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"]["imported_picture_ids"] == [picture_id], results
    _assert_a_staged_duplicate_of(first, results["first"], picture_id)


def test_generation_import_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, _client = owner_server
    data = _png_bytes((250, 150, 5))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    hold = _hold_first_publish(monkeypatch)
    fetched = tmp_path / "test-fetched.png"
    fetched.write_bytes(data)

    results = _run_overlapping(
        hold,
        staged._run_task,
        lambda: _import_comfyui_outputs(srv, [(str(fetched), ".png")]),
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"] == ([picture_id], []), results
    _assert_a_staged_duplicate_of(staged, results["first"], picture_id)


def test_watch_folder_import_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, _client = owner_server
    data = _png_bytes((250, 5, 150))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(
        hold,
        staged._run_task,
        lambda: _watch_folder_pass(srv, tmp_path, "watched.png", data),
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"]["imported_picture_ids"] == [picture_id], results
    _assert_a_staged_duplicate_of(staged, results["first"], picture_id)


def test_one_shot_import_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, client = owner_server
    data = _png_bytes((5, 250, 150))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(
        hold,
        staged._run_task,
        lambda: _one_shot_import(client, [("one-shot.png", data)]),
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"]["status"] == "completed", results
    assert results["second"]["imported_count"] == 1, results
    _assert_a_staged_duplicate_of(staged, results["first"], picture_id)


def test_plugin_output_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    """Plugin results and Open-with edit copies come in through one funnel."""
    srv, _client = owner_server
    data = _png_bytes((150, 250, 5))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(
        hold,
        staged._run_task,
        lambda: _import_output_images(srv, [(data, ".png")]),
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"] == ([picture_id], [], [picture_id]), results
    _assert_a_staged_duplicate_of(staged, results["first"], picture_id)


def test_library_move_copy_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    """A move's copy into a library is an insert like any import's (I1).

    The move route opens its target beside the active library; the copy is
    the same lookup-to-insert span, so here it targets the server's own vault,
    where a staged import of the same bytes has published its copy and not
    yet looked again.
    """
    srv, _client = owner_server
    data = _png_bytes((150, 5, 250))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    source_root = tmp_path / "other-library"
    source_root.mkdir()
    (source_root / "moved.png").write_bytes(data)
    row = SimpleNamespace(
        id=-1,
        file_path="moved.png",
        original_file_name="moved.png",
        score=3,
        description=None,
        created_at=None,
        tags=[],
    )
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(
        hold,
        staged._run_task,
        lambda: _copy_into(srv.vault, str(source_root), row),
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"] == "new", results
    _assert_a_staged_duplicate_of(staged, results["first"], picture_id)


def _comfy_run(srv, client, tmp_path, data):
    fetched = tmp_path / f"test-fetched-{time.monotonic_ns()}.png"
    fetched.write_bytes(data)
    new_ids, duplicate_ids = _import_comfyui_outputs(srv, [(str(fetched), ".png")])
    return new_ids, duplicate_ids


def _plugin_run(srv, client, tmp_path, data):
    new_ids, duplicate_ids, ordered = _import_output_images(srv, [(data, ".png")])
    assert ordered == (new_ids or duplicate_ids)
    return new_ids, duplicate_ids


def _watch_run(srv, client, tmp_path, data):
    result = _watch_folder_pass(
        srv, tmp_path, f"watched-{time.monotonic_ns()}.png", data
    )
    return result["imported_picture_ids"], None


def _one_shot_run(srv, client, tmp_path, data):
    status = _one_shot_import(client, [("one-shot.png", data)])
    assert status["status"] == "completed", status
    return [None] * status["imported_count"], [None] * status["duplicate_count"]


def _staged_run(srv, client, tmp_path, data):
    task = _staged_import_task(srv, tmp_path, f"staged-{time.monotonic_ns()}.png", data)
    result = task._run_task()
    return result["imported_picture_ids"], [
        o["picture_id"] for o in task.outcomes if o["outcome"] == "duplicate"
    ]


_LOCKED_IMPORTERS = {
    "staged": _staged_run,
    "one-shot": _one_shot_run,
    "watch": _watch_run,
    "plugin": _plugin_run,
    "comfyui": _comfy_run,
}


@pytest.mark.parametrize("importer", ["one-shot", "watch", "plugin", "comfyui"])
def test_an_import_overlapped_by_a_staged_import_withdraws_its_copy(
    owner_server, monkeypatch, tmp_path, importer
):
    """Each path publishes its copy before the lock, looks again under it,
    finds the picture the other import added meanwhile, and removes the copy."""
    srv, client = owner_server
    data = _png_bytes((60, 120, 180))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    hold = _hold_first_publish(monkeypatch)

    results = _run_overlapping(
        hold,
        lambda: _LOCKED_IMPORTERS[importer](srv, client, tmp_path, data),
        staged._run_task,
    )

    picture_id = _assert_the_second_import_won(srv, hold, data)
    assert results["second"]["imported_picture_ids"] == [picture_id], results
    new_ids, duplicate_ids = results["first"]
    assert new_ids == [], results
    if importer in ("plugin", "comfyui"):
        assert duplicate_ids == [picture_id], results
    elif importer == "one-shot":
        assert duplicate_ids == [None], results


_BYTE_READERS = (
    "calculate_hash_from_file_path",
    "calculate_hash_from_open_file",
    "calculate_full_hash_from_file_path",
    "calculate_full_hash_from_open_file",
    "calculate_hash_from_bytes",
    "calculate_full_hash_from_bytes",
    "create_picture_from_file",
    "create_picture_from_bytes",
)


@pytest.mark.parametrize("importer", sorted(_LOCKED_IMPORTERS))
def test_an_import_reads_and_copies_bytes_without_holding_the_import_lock(
    owner_server, monkeypatch, tmp_path, importer
):
    """No import waits on another's disk reads. The lock covers the
    lookup and the insert; the hashing, reading and copying run before it."""
    srv, client = owner_server
    lock = _signalling_import_lock(monkeypatch)
    calls: list[tuple[str, bool]] = []

    def _record(owner, name):
        real = getattr(owner, name)

        def wrapper(*args, **kwargs):
            calls.append((name, lock.held_here()))
            return real(*args, **kwargs)

        return wrapper

    for name in _BYTE_READERS:
        monkeypatch.setattr(ImageUtils, name, staticmethod(_record(ImageUtils, name)))
    real_add = import_publish_service.PublishedPictures.add
    monkeypatch.setattr(
        import_publish_service.PublishedPictures,
        "add",
        lambda self, picture: _record(SimpleNamespace(add=real_add), "add")(
            self, picture
        ),
    )

    new_ids, _duplicates = _LOCKED_IMPORTERS[importer](
        srv, client, tmp_path, _png_bytes((70, 140, 210))
    )

    assert len(new_ids) == 1
    assert lock.acquired >= 1, "the insert ran without the import lock"
    assert ("add", False) in calls, calls
    assert [name for name, held in calls if held] == []


@pytest.mark.parametrize("importer", ["staged", "one-shot", "watch"])
def test_a_collision_compared_before_the_lock_costs_no_read_under_it(
    owner_server, monkeypatch, tmp_path, importer
):
    """A picture the library held before, sharing a file's sampled hash and
    size, is compared once, before the import lock. The look again under the
    lock knows it was compared and reads no file for it: only a candidate
    added since the lookup is read there."""
    srv, client = owner_server
    first = _bmp_bytes((11, 22, 33))
    second = _bmp_bytes((33, 22, 11))
    assert len(first) == len(second)
    for name in ("calculate_hash_from_open_file", "calculate_hash_from_bytes"):
        monkeypatch.setattr(
            ImageUtils,
            name,
            staticmethod(lambda *_args, **_kwargs: "forced-compared-collision"),
        )
    assert _one_shot_import(client, [("first.bmp", first)])["imported_count"] == 1
    candidate_path = _vault_path(srv, _picture_ids(srv)[0])
    lock = _signalling_import_lock(monkeypatch)
    real_full_hash = ImageUtils.calculate_full_hash_from_file_path
    candidate_reads: list[bool] = []

    def _spy(path):
        if os.path.normpath(path) == candidate_path:
            candidate_reads.append(lock.held_here())
        return real_full_hash(path)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_file_path", staticmethod(_spy)
    )

    new_ids, _duplicates = _LOCKED_IMPORTERS[importer](srv, client, tmp_path, second)

    assert len(new_ids) == 1
    assert _row_count(srv) == 2
    assert lock.acquired >= 1, "the insert ran without the import lock"
    # Compared before the lock, and not again under it.
    assert candidate_reads == [False], candidate_reads


# ---------------------------------------------------------------------------
# A watch folder skips only a file whose full content is already there
# ---------------------------------------------------------------------------


def _watch_folder_pass(srv, tmp_path, name, data) -> dict:
    """Run one watch-folder pass over a single file and return its result."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(exist_ok=True)
    file_path = watch_dir / name
    file_path.write_bytes(data)
    task = WatchFolderImportTask(srv.vault.db, [{"file_path": str(file_path)}], 1)
    return task._run_task()


@pytest.mark.parametrize(
    "incoming_size, same_size",
    [((48, 48), True), ((49, 48), False)],
    ids=["same-size", "different-size"],
)
def test_watch_folder_imports_a_file_whose_sampled_hash_collides(
    owner_server, monkeypatch, tmp_path, incoming_size, same_size
):
    """A sampled hash shared with a different picture does not skip the file."""
    srv, client = owner_server
    existing = _bmp_bytes((12, 34, 56))
    incoming = _bmp_bytes((56, 34, 12), incoming_size)
    assert (len(incoming) == len(existing)) is same_size
    # The open-file hash: the watch folder hashes the file it opened, and
    # the by-path hash the staged import uses opens the file and calls it.
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_open_file",
        staticmethod(lambda _handle, source_label=None: "forced-watch-collision"),
    )
    assert _staging_import(client, [("existing.bmp", existing)])["imported_count"] == 1

    result = _watch_folder_pass(srv, tmp_path, "incoming.bmp", incoming)

    assert len(result["imported_picture_ids"]) == 1, result
    assert _row_count(srv) == 2


def test_watch_folder_still_skips_a_file_already_in_the_library(owner_server, tmp_path):
    srv, client = owner_server
    data = _bmp_bytes((21, 43, 65))
    assert _staging_import(client, [("kept.bmp", data)])["imported_count"] == 1

    result = _watch_folder_pass(srv, tmp_path, "kept-copy.bmp", data)

    assert result["imported_picture_ids"] == [], result
    assert _row_count(srv) == 1


def test_watch_folder_skips_a_scrapheaped_match_and_leaves_it_there(
    owner_server, tmp_path
):
    """A kept watch-folder file comes round again after every restart.

    Restoring its scrapheaped picture would undo the owner's cleanup on each
    start, and importing it again would add a second row, so it is skipped.
    """
    srv, client = owner_server
    data = _bmp_bytes((65, 43, 21))
    assert _staging_import(client, [("heaped.bmp", data)])["imported_count"] == 1
    heaped_id = _picture_ids(srv)[0]
    _scrapheap(client, heaped_id)

    result = _watch_folder_pass(srv, tmp_path, "heaped-copy.bmp", data)

    assert result["imported_picture_ids"] == [], result
    assert _picture_ids(srv) == [heaped_id]
    assert _picture_ids(srv, deleted=True) == [heaped_id]


def test_watch_folder_reads_a_file_in_full_only_when_a_picture_shares_its_key(
    owner_server, monkeypatch, tmp_path
):
    """A file no picture shares a sampled hash and size with is never read whole."""
    srv, client = owner_server
    assert (
        _staging_import(client, [("other.bmp", _bmp_bytes((99, 1, 1)))])[
            "imported_count"
        ]
        == 1
    )
    real_full_hash = ImageUtils.calculate_full_hash_from_open_file
    full_reads = []

    def _spy(handle):
        full_reads.append(handle)
        return real_full_hash(handle)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_open_file", staticmethod(_spy)
    )

    result = _watch_folder_pass(srv, tmp_path, "fresh.bmp", _bmp_bytes((1, 1, 99)))

    assert len(result["imported_picture_ids"]) == 1, result
    assert full_reads == []


# ---------------------------------------------------------------------------
# What an import leaves behind for the background workers
# ---------------------------------------------------------------------------


def test_one_shot_import_leaves_no_face_watcher_behind(owner_server, monkeypatch):
    """Nothing waits on a new picture's faces, so the import registers no
    watcher for them: one left behind is only collected by an extraction, and
    an extraction that never comes leaves it in the vault for good. The import
    still brings the inference engine up on a server that has not built it."""
    srv, client = owner_server
    # Registered, or the import is refused; idle, so no extraction collects
    # what the import left behind.
    monkeypatch.setattr(MissingFaceExtractionFinder, "find_task", lambda self: None)
    assert srv.vault._engine is None

    status = _one_shot_import(client, [("test-watch.png", _png_bytes((9, 99, 199)))])

    assert status["imported_count"] == 1, status
    assert srv.vault._planner_watchers == {}
    assert srv.vault._engine is not None


def test_a_duplicate_only_import_does_not_build_the_engine(owner_server, monkeypatch):
    """An import that brings nothing new in gives the workers nothing to do, so
    it does not try to build the inference engine. A library whose engine
    build failed stays open without one; rebuilding it for every re-import of
    files already in the library failed those imports."""
    srv, client = owner_server
    monkeypatch.setattr(MissingFaceExtractionFinder, "find_task", lambda self: None)
    builds = []

    def _failing_build():
        builds.append("build")
        raise RuntimeError("test-engine-build-failed")

    monkeypatch.setattr(srv.vault, "_create_engine", _failing_build)
    assert srv.vault._engine is None
    data = _png_bytes((7, 77, 177))

    _one_shot_import(client, [("test-engineless.png", data)])
    ids = _picture_ids(srv)
    assert ids, "the first import's picture row did not commit"
    builds.clear()

    status = _one_shot_import(client, [("test-engineless.png", data)])

    assert status["status"] == "completed", status
    assert status["duplicate_count"] == 1, status
    assert builds == []
    assert _picture_ids(srv) == ids
