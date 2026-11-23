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
import tempfile
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session
from sqlmodel import select

from pixlstash.db_models import (
    DeletedFileLog,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
)
from pixlstash.event_types import EventType
from pixlstash.server import Server
from pixlstash.services import import_dedup_service, scrapheap_service
from pixlstash.services.comfyui_service import _import_comfyui_outputs
from pixlstash.tasks import picture_import_task
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
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_bytes",
        staticmethod(lambda _data: "forced-sampled-collision"),
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
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_bytes",
        staticmethod(lambda _data: "forced-sampled-collision"),
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
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_bytes",
        staticmethod(lambda _data: "forced-sampled-collision"),
    )

    assert _one_shot_import(client, [("old.bmp", old)])["imported_count"] == 1
    _scrapheap(client, _picture_ids(srv)[0])
    status = _one_shot_import(client, [("different.bmp", different)])

    assert status["imported_count"] == 1, status
    assert status["scrapheaped_count"] == 0, status
    assert status["scrapheaped_picture_ids"] == [], status
    assert _row_count(srv) == 2


def test_staging_and_comfy_batches_keep_sampled_collisions_distinct(
    owner_server, monkeypatch
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
        "calculate_hash_from_bytes",
        staticmethod(lambda _data: "forced-comfy-collision"),
    )
    third = _bmp_bytes((70, 80, 90))
    fourth = _bmp_bytes((90, 80, 70))
    new_ids, duplicate_ids = _import_comfyui_outputs(
        srv, [(third, ".bmp"), (fourth, ".bmp")]
    )
    assert len(new_ids) == 2
    assert duplicate_ids == []

    repeated_new, repeated_duplicates = _import_comfyui_outputs(
        srv, [(third, ".bmp"), (fourth, ".bmp")]
    )
    assert repeated_new == []
    assert sorted(repeated_duplicates) == sorted(new_ids)


def test_restore_announces_exactly_the_pictures_it_restored(owner_server):
    """The ``restored`` event names what came back; a no-op restore says nothing."""
    srv, client = owner_server
    _staging_import(client, [("a.png", _png_bytes((3, 3, 202)))])
    heaped_id = _picture_ids(srv)[0]
    _scrapheap(client, heaped_id)

    emitted: list[dict] = []
    real_notify = srv.vault.notify

    def _capture(event_type, data=None):
        if isinstance(data, dict) and data.get("change_kind") == "restored":
            emitted.append(data)
        return real_notify(event_type, data)

    srv.vault.notify = _capture
    try:
        # Restore-all: no ids in the request, the real one in the event.
        r = client.post(f"{API}/pictures/scrapheap/restore")
        assert r.status_code == 200, r.text
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
# A staged import settles its files in batches
# ---------------------------------------------------------------------------


def test_staging_import_looks_up_matches_once_per_batch_on_the_writer(
    owner_server, monkeypatch
):
    """One writer-queued lookup per batch, and each sees the batches before it."""
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
    assert [lookup["keys"] for lookup in lookups] == [4, 4, 1], lookups
    assert all(lookup["on_writer"] for lookup in lookups), lookups
    assert [lookup["rows_visible"] for lookup in lookups] == [0, 4, 8], lookups


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
    owner_server, monkeypatch
):
    """Earlier batches stay imported, and the failure says how many there are."""
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

    status = _staging_import(
        client,
        files,
        keys=["tile-0", "tile-1", "tile-2"],
        target=("completed", "failed"),
    )

    assert status["stage"] == "failed", status
    assert "2 new picture" in status["error"], status
    assert "test-insert-failure" in status["error"], status
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
    """The import lock, saying when a second import has to wait for it."""

    def __init__(self, on_contended):
        self._inner = threading.Lock()
        self._on_contended = on_contended

    def __enter__(self):
        if not self._inner.acquire(blocking=False):
            self._on_contended()
            self._inner.acquire()
        return self

    def __exit__(self, *exc_info):
        self._inner.release()
        return False


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


def _hold_first_decision(monkeypatch):
    """Pause the first import right after it decided its file is new.

    The first import waits until the second one has gone as far as it can
    without the first one finishing: up to its own decision when nothing
    orders them, or up to the import lock when the lock does. Returns the
    events and the list the first import's wait results land in.
    """
    first_decided = threading.Event()
    second_progressed = threading.Event()
    first_waits: list[bool] = []
    state = {"first_thread": None}
    real_partition = import_dedup_service.partition_confirmed_matches

    def _partition(*args, **kwargs):
        result = real_partition(*args, **kwargs)
        if threading.current_thread() is state["first_thread"] and not first_waits:
            first_decided.set()
            first_waits.append(second_progressed.wait(timeout=30))
        else:
            second_progressed.set()
        return result

    monkeypatch.setattr(import_dedup_service, "partition_confirmed_matches", _partition)
    monkeypatch.setattr(
        import_dedup_service,
        "_IMPORT_LOCK",
        _SignallingLock(second_progressed.set),
        raising=False,
    )
    return state, first_decided, first_waits


def _run_overlapping(state, first_decided, first, second):
    results = {}
    errors = []

    def _runner(label, func):
        def _run():
            try:
                results[label] = func()
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        return threading.Thread(target=_run, name=f"test-import-{label}")

    first_thread = _runner("first", first)
    state["first_thread"] = first_thread
    first_thread.start()
    assert first_decided.wait(timeout=30), "the first import never decided"
    second_thread = _runner("second", second)
    second_thread.start()
    first_thread.join(timeout=60)
    second_thread.join(timeout=60)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert errors == []
    return results


def test_overlapping_staged_imports_of_new_content_add_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, _client = owner_server
    data = _png_bytes((5, 150, 250))
    first = _staged_import_task(srv, tmp_path, "first.png", data)
    second = _staged_import_task(srv, tmp_path, "second.png", data)
    state, first_decided, first_waits = _hold_first_decision(monkeypatch)

    results = _run_overlapping(state, first_decided, first._run_task, second._run_task)

    assert first_waits == [True], "the second import never got going"
    assert _row_count(srv) == 1
    assert results["first"]["imported_count"] == 1, results
    assert results["second"]["imported_count"] == 0, results
    assert results["second"]["duplicate_count"] == 1, results
    assert second.outcomes == [
        {
            "key": None,
            "outcome": "duplicate",
            "picture_id": results["first"]["imported_picture_ids"][0],
        }
    ]


def test_generation_import_overlapping_a_staged_import_adds_one_picture(
    owner_server, monkeypatch, tmp_path
):
    srv, _client = owner_server
    data = _png_bytes((250, 150, 5))
    staged = _staged_import_task(srv, tmp_path, "staged.png", data)
    state, first_decided, first_waits = _hold_first_decision(monkeypatch)

    results = _run_overlapping(
        state,
        first_decided,
        staged._run_task,
        lambda: _import_comfyui_outputs(srv, [(data, ".png")]),
    )

    assert first_waits == [True], "the generation import never got going"
    assert _row_count(srv) == 1
    staged_ids = results["first"]["imported_picture_ids"]
    assert len(staged_ids) == 1, results
    assert results["second"] == ([], staged_ids), results


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
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda _path: "forced-watch-collision"),
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
    real_full_hash = ImageUtils.calculate_full_hash_from_file_path
    full_reads = []

    def _spy(path):
        full_reads.append(path)
        return real_full_hash(path)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_file_path", staticmethod(_spy)
    )

    result = _watch_folder_pass(srv, tmp_path, "fresh.bmp", _bmp_bytes((1, 1, 99)))

    assert len(result["imported_picture_ids"]) == 1, result
    assert full_reads == []
