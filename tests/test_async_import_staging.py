"""Tests for the async streaming-staging import (#459).

Covers the two-phase flow end to end:

* Phase A (unsafe window): open a staging session, stream files into it.
* Phase B (safe window): commit hands off to a background ``PictureImportTask``
  on the shared task runner; progress is polled via the staging status endpoint.

Also covers both authz directions on the owner-only mutating routes: a READ
scoped token is denied (403) and the owner is allowed (200 / works) - over-blocking
the owner would be its own regression.
"""

import asyncio
import errno
import functools
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from pixlstash.db_models import Character, Picture, PictureSet, PictureSetMember, Tag
from pixlstash.routes.pictures import _import as import_mod
from pixlstash.server import Server
from pixlstash.tasks.picture_import_task import PictureImportTask
from tests.authz_guard import no_spa_fallback  # noqa: F401
from tests.utils import wait_for_import_task

API = "/api/v1"

# The SPA catch-all answers unmatched GETs with 200, so a wrong URL can make a
# positive assertion vacuous. See tests/authz_guard.py.
pytestmark = pytest.mark.usefixtures("no_spa_fallback")


def _png_bytes(color=(100, 149, 237)) -> bytes:
    img = Image.new("RGB", (48, 48), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_set(srv, name="drop-set") -> int:
    def _do(session):
        s = PictureSet(name=name)
        session.add(s)
        session.commit()
        session.refresh(s)
        return s.id

    return srv.vault.db.run_task(_do)


def _make_character(srv, name="drop-char") -> int:
    def _do(session):
        c = Character(name=name)
        session.add(c)
        session.commit()
        session.refresh(c)
        return c.id

    return srv.vault.db.run_task(_do)


def _set_member_ids(srv, set_id) -> set:
    return set(
        srv.vault.db.run_task(
            lambda s: [
                m.picture_id
                for m in s.exec(
                    select(PictureSetMember).where(PictureSetMember.set_id == set_id)
                ).all()
            ]
        )
    )


def _tags_for(srv, picture_id) -> set:
    return set(
        srv.vault.db.run_task(
            lambda s: [
                t.tag
                for t in s.exec(select(Tag).where(Tag.picture_id == picture_id)).all()
            ]
        )
    )


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


def _wait_for_stage(client, staging_id, target=("completed", "failed"), timeout_s=30):
    """Poll the staging status until it reaches a terminal stage."""
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


def test_staging_import_happy_path(owner_server):
    """Open → stream one file → commit → the background task imports it."""
    _srv, client = owner_server

    r = client.post(f"{API}/pictures/import/staging", json={"total_files": 1})
    assert r.status_code == 200, r.text
    staging_id = r.json()["staging_id"]
    assert r.json()["safe_threshold"] == 1

    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("photo.png", _png_bytes(), "image/png"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["staged"] == 1
    assert r.json()["received"] == ["photo.png"]

    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    assert r.json()["staged_count"] == 1
    assert r.json()["task_id"]

    status = _wait_for_stage(client, staging_id, target=("completed",))
    assert status["stage"] == "completed", status
    assert status["imported_count"] == 1
    assert status["duplicate_count"] == 0

    # The picture is really in the vault, and the staging dir is cleaned up.
    r = client.get(f"{API}/pictures")
    assert r.status_code == 200
    assert len(r.json()) == 1
    staging_dir = os.path.join(_srv.vault.image_root, ".staging", staging_id)
    assert not os.path.isdir(staging_dir), "Staging dir must be removed after import"


def test_staging_import_dedupes_identical_files(owner_server):
    """Two byte-identical staged files import once; the second is a duplicate.

    Each file is streamed with the client's own ``key``, and the status names
    every file's outcome by that key (#2512), so the grid can mark the tile
    that turned out to be a duplicate rather than report a total at the end.
    """
    _srv, client = owner_server
    data = _png_bytes(color=(10, 200, 30))

    r = client.post(f"{API}/pictures/import/staging", json={})
    staging_id = r.json()["staging_id"]

    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[
            ("file", ("a.png", data, "image/png")),
            ("file", ("b.png", data, "image/png")),
        ],
        data={"key": ["tile-a", "tile-b"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["staged"] == 2

    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text

    status = _wait_for_stage(client, staging_id, target=("completed",))
    assert status["imported_count"] == 1, status
    assert status["duplicate_count"] == 1, status
    assert status["outcomes"] == [
        {"key": "tile-a", "outcome": "new", "picture_id": None},
        {"key": "tile-b", "outcome": "duplicate", "picture_id": None},
    ], status

    # The caller asks only for what it has not seen yet.
    r = client.get(
        f"{API}/pictures/import/staging/{staging_id}/status",
        params={"outcomes_from": 1},
    )
    assert r.status_code == 200, r.text
    assert [o["key"] for o in r.json()["outcomes"]] == ["tile-b"]

    # A second import of the same content is a duplicate OF that picture, and
    # says which one, so the tile can link to it.
    picture_id = _only_picture_id(_srv)
    r = client.post(f"{API}/pictures/import/staging", json={})
    again_id = r.json()["staging_id"]
    r = client.post(
        f"{API}/pictures/import/staging/{again_id}/files",
        files=[("file", ("c.png", data, "image/png"))],
        data={"key": ["tile-c"]},
    )
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/pictures/import/staging/{again_id}/commit")
    assert r.status_code == 200, r.text
    status = _wait_for_stage(client, again_id, target=("completed",))
    assert status["outcomes"] == [
        {"key": "tile-c", "outcome": "duplicate", "picture_id": picture_id},
    ], status

    imported = _srv.vault.db.run_task(lambda s: s.query(Picture).count())
    assert imported == 1


def test_staging_files_rejects_unsupported_extension(owner_server):
    """A non-media file is skipped and reported, not staged."""
    _srv, client = owner_server
    r = client.post(f"{API}/pictures/import/staging", json={})
    staging_id = r.json()["staging_id"]

    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("notes.exe", b"nope", "application/octet-stream"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["staged"] == 0
    assert body["skipped"] == ["notes.exe"]

    # Committing an empty session is a 400.
    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 400, r.text


def test_staging_cancel_discards_without_importing(owner_server):
    """Cancelling before commit removes the staged files and imports nothing."""
    _srv, client = owner_server
    r = client.post(f"{API}/pictures/import/staging", json={})
    staging_id = r.json()["staging_id"]
    staging_dir = os.path.join(_srv.vault.image_root, ".staging", staging_id)

    client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("x.png", _png_bytes(), "image/png"))],
    )
    assert os.path.isdir(staging_dir)

    r = client.delete(f"{API}/pictures/import/staging/{staging_id}")
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "cancelled"
    assert not os.path.isdir(staging_dir), "Cancel must remove the staging dir"

    # Session is gone (404), and no pictures were imported.
    r = client.get(f"{API}/pictures/import/staging/{staging_id}/status")
    assert r.status_code == 404
    assert client.get(f"{API}/pictures").json() == []


def test_staging_status_unknown_session_404(owner_server):
    _srv, client = owner_server
    r = client.get(f"{API}/pictures/import/staging/does-not-exist/status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Authz - both directions on the owner-only mutating routes
# ---------------------------------------------------------------------------


def _read_token(client) -> str:
    r = client.post(
        f"{API}/users/me/token",
        json={"description": "read", "scope": "READ"},
    )
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_staging_open_denied_for_read_token_allowed_for_owner(owner_server):
    """Out-of-scope (READ token) → 403; in-scope (owner) → 200 (not over-blocked)."""
    _srv, owner = owner_server
    token = _read_token(owner)

    scoped = TestClient(_srv.api, raise_server_exceptions=True)
    r = scoped.post(
        f"{API}/pictures/import/staging",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403, (
        f"READ token must not open a staging session, got {r.status_code}: {r.text}"
    )

    # The owner is not over-blocked.
    r = owner.post(f"{API}/pictures/import/staging", json={})
    assert r.status_code == 200, r.text


def test_staging_files_and_commit_denied_for_read_token(owner_server):
    """READ token is blocked from streaming into and committing a session."""
    _srv, owner = owner_server
    token = _read_token(owner)
    auth = {"Authorization": f"Bearer {token}"}

    # Owner opens a real session (so the routes exist to attack).
    staging_id = owner.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]

    scoped = TestClient(_srv.api, raise_server_exceptions=True)
    r = scoped.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("x.png", _png_bytes(), "image/png"))],
        headers=auth,
    )
    assert r.status_code == 403, r.text

    r = scoped.post(f"{API}/pictures/import/staging/{staging_id}/commit", headers=auth)
    assert r.status_code == 403, r.text

    r = scoped.delete(f"{API}/pictures/import/staging/{staging_id}", headers=auth)
    assert r.status_code == 403, r.text


def test_scrapheap_delete_preview_denied_for_read_token_allowed_for_owner(owner_server):
    """The delete-preview route returns per-object absolute on-disk paths, so it is
    OWNER_ONLY: a READ token → 403, the owner → 200 (empty scrapheap, not over-blocked)."""
    _srv, owner = owner_server
    token = _read_token(owner)

    scoped = TestClient(_srv.api, raise_server_exceptions=True)
    r = scoped.post(
        f"{API}/pictures/scrapheap/delete-preview",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403, (
        f"READ token must not preview scrapheap deletions (leaks on-disk paths), "
        f"got {r.status_code}: {r.text}"
    )

    # The owner is not over-blocked - an empty scrapheap previews as zero counts.
    r = owner.post(f"{API}/pictures/scrapheap/delete-preview", json={})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Addition 1 - server-side set / character association
# ---------------------------------------------------------------------------


def _import_one(client, open_body, filename="a.png", data=None):
    """Open → stream one image → commit → wait; return the imported picture id."""
    data = data if data is not None else _png_bytes()
    staging_id = client.post(f"{API}/pictures/import/staging", json=open_body).json()[
        "staging_id"
    ]
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", (filename, data, "image/png"))],
    )
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    _wait_for_stage(client, staging_id, target=("completed",))
    return staging_id


def _only_picture_id(srv):
    return srv.vault.db.run_task(lambda s: s.exec(select(Picture.id)).first())


def test_drop_to_set_associates_imported_pictures(owner_server):
    srv, client = owner_server
    set_id = _make_set(srv)
    _import_one(client, {"set_id": set_id})
    pic_id = _only_picture_id(srv)
    assert pic_id in _set_member_ids(srv, set_id), "Imported picture must join the set"


def test_drop_to_character_sets_pending_assignment(owner_server):
    srv, client = owner_server
    character_id = _make_character(srv)
    _import_one(client, {"character_id": character_id})
    pic_id = _only_picture_id(srv)
    pending = srv.vault.db.run_task(
        lambda s: s.get(Picture, pic_id).pending_character_id
    )
    assert pending == character_id, (
        "Imported picture must carry the pending character assignment "
        "(resolved after face extraction)"
    )


def test_drop_to_set_and_character_together(owner_server):
    srv, client = owner_server
    set_id = _make_set(srv)
    character_id = _make_character(srv)
    _import_one(client, {"set_id": set_id, "character_id": character_id})
    pic_id = _only_picture_id(srv)
    assert pic_id in _set_member_ids(srv, set_id)
    assert (
        srv.vault.db.run_task(lambda s: s.get(Picture, pic_id).pending_character_id)
        == character_id
    )


def test_project_id_still_associates(owner_server):
    srv, client = owner_server
    project_id = srv.vault.db.run_task(lambda s: _make_project(s))
    _import_one(client, {"project_id": project_id})
    pic_id = _only_picture_id(srv)
    assert (
        srv.vault.db.run_task(lambda s: s.get(Picture, pic_id).project_id) == project_id
    )


def _make_project(session):
    from pixlstash.db_models import Project

    p = Project(name="drop-project")
    session.add(p)
    session.commit()
    session.refresh(p)
    return p.id


def test_open_with_nonexistent_set_errors(owner_server):
    _srv, client = owner_server
    r = client.post(f"{API}/pictures/import/staging", json={"set_id": 999999})
    assert r.status_code == 404, r.text


def test_open_with_nonexistent_character_errors(owner_server):
    _srv, client = owner_server
    r = client.post(f"{API}/pictures/import/staging", json={"character_id": 999999})
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# Addition 2 - zip archives + .txt caption sidecars
# ---------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_staging_zip_imports_all_images(owner_server):
    srv, client = owner_server
    zip_data = _zip_bytes(
        {
            "one.png": _png_bytes(color=(10, 20, 30)),
            "two.png": _png_bytes(color=(200, 100, 50)),
            "three.png": _png_bytes(color=(5, 150, 250)),
        }
    )
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", ("bundle.zip", zip_data, "application/zip"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["staged"] == 3, r.json()
    assert r.json()["received"] == ["bundle.zip"]

    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    status = _wait_for_stage(client, staging_id, target=("completed",))
    assert status["imported_count"] == 3, status
    assert srv.vault.db.run_task(lambda s: s.query(Picture).count()) == 3


def test_staging_txt_sidecar_sets_tags(owner_server):
    srv, client = owner_server
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    # Comma-separated .txt = tag list (the legacy single-shot sidecar rule).
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[
            ("file", ("kitten.png", _png_bytes(), "image/png")),
            ("file", ("kitten.txt", b"cat, cute, fluffy", "text/plain")),
        ],
    )
    assert r.status_code == 200, r.text
    assert r.json()["staged"] == 1
    assert r.json()["sidecars"] == 1

    client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    _wait_for_stage(client, staging_id, target=("completed",))
    pic_id = _only_picture_id(srv)
    tags = _tags_for(srv, pic_id)
    assert {"cat", "cute", "fluffy"} <= tags, f"Sidecar tags missing: {tags}"


def test_staging_orphan_sidecar_is_skipped_gracefully(owner_server):
    srv, client = owner_server
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    # Sidecar whose stem matches no staged image.
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[
            ("file", ("photo.png", _png_bytes(), "image/png")),
            ("file", ("nomatch.txt", b"alpha, beta", "text/plain")),
        ],
    )
    assert r.status_code == 200, r.text
    assert r.json()["staged"] == 1
    assert r.json()["sidecars"] == 1

    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    status = _wait_for_stage(client, staging_id, target=("completed",))
    assert status["imported_count"] == 1, status
    # The image imported fine; the orphan sidecar contributed no tags.
    pic_id = _only_picture_id(srv)
    assert {"alpha", "beta"} & _tags_for(srv, pic_id) == set()


# ---------------------------------------------------------------------------
# Round 3 - project_id fail-fast (H2) and the staging-session reaper (H1)
# ---------------------------------------------------------------------------


def test_open_with_nonexistent_project_errors(owner_server):
    _srv, client = owner_server
    r = client.post(f"{API}/pictures/import/staging", json={"project_id": 999999})
    assert r.status_code == 404, r.text


def test_direct_import_with_nonexistent_project_errors(owner_server):
    """The direct multipart endpoint fails fast on a stale project id too.

    It used to skip the check the staging path runs, so the id only failed at
    the membership INSERT - after every file had been imported and committed.
    The caller (the ComfyUI saver node, which stores the project as
    "<name> #<id>" and so replays whatever id was current when the workflow was
    authored) was told the import failed while the pictures were already in the
    vault.
    """
    srv, client = owner_server
    r = client.post(
        f"{API}/pictures/import",
        files=[("file", ("a.png", _png_bytes(), "image/png"))],
        data={"project_id": "999999"},
    )
    assert r.status_code == 404, r.text
    assert "999999" in r.text
    # And nothing was imported on the way to the refusal.
    assert srv.vault.db.run_task(lambda s: s.exec(select(Picture.id)).first()) is None


def test_reaper_evicts_stale_session_and_dir(owner_server):
    srv, client = owner_server
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    staging_dir = os.path.join(srv.vault.image_root, ".staging", staging_id)
    # Drop a marker so we can prove the dir is really removed.
    with open(os.path.join(staging_dir, "marker.png"), "wb") as fh:
        fh.write(_png_bytes())
    assert os.path.isdir(staging_dir)

    # Backdate the session well past the idle TTL, then run the reaper.
    session = srv.staging_sessions[staging_id]
    session["last_update_epoch_ms"] = int(time.time() * 1000) - (
        (import_mod._STAGING_SESSION_TTL_S + 60) * 1000
    )
    import_mod._reap_staging_sessions(srv)

    assert staging_id not in srv.staging_sessions, "Stale session must be evicted"
    assert not os.path.isdir(staging_dir), "Reaper must remove the staging dir"


def test_reaper_keeps_fresh_session(owner_server):
    srv, client = owner_server
    staging_id = client.post(f"{API}/pictures/import/staging", json={}).json()[
        "staging_id"
    ]
    # A just-opened session is within TTL and must survive the reaper.
    import_mod._reap_staging_sessions(srv)
    assert staging_id in srv.staging_sessions, "Fresh session must not be evicted"


# ---------------------------------------------------------------------------
# Ownership of a staging session: an upload, a commit, a cancel and the reaper
# each take the session's files only when nothing else holds them.
# ---------------------------------------------------------------------------

_STAGING = f"{API}/pictures/import/staging"


@pytest.fixture(scope="module")
def shared():
    """One server for the ownership tests below; each opens its own session.

    The tests hold a request at a known point inside the server (an upload
    parked between two files, a commit parked before the task runner) while
    they send others, so every assertion is about the test's own session or a
    difference it measured, never a total across the module.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with Server(os.path.join(tmp, "server-config.json")) as srv:
            client = TestClient(srv.api, raise_server_exceptions=True)
            r = client.post(
                f"{API}/login",
                json={"username": "owner", "password": "example-owner-password"},
            )
            assert r.status_code == 200, r.text
            yield srv, client


class _Gate:
    """Parks the first call of a wrapped function until the test releases it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._used = False

    def wrap(self, original):
        @functools.wraps(original)
        def gated(*args, **kwargs):
            with self._lock:
                first, self._used = not self._used, True
            if first:
                self.entered.set()
                if not self.release.wait(60):
                    raise RuntimeError("test gate was never released")
            return original(*args, **kwargs)

        return gated

    def wait_entered(self):
        assert self.entered.wait(30), "the gated call was never reached"


def _park_uploads_between_files(monkeypatch) -> _Gate:
    """Hold the next upload after its image is written, before its caption.

    Send the batch as ``[image, caption]``: the image is on disk but not yet
    staged when the upload stops.
    """
    gate = _Gate()
    monkeypatch.setattr(
        import_mod,
        "_normalise_sidecar_stem",
        gate.wrap(import_mod._normalise_sidecar_stem),
    )
    return gate


def _open_session(client) -> str:
    r = client.post(_STAGING, json={})
    assert r.status_code == 200, r.text
    return r.json()["staging_id"]


def _staging_dir(srv, staging_id) -> str:
    return os.path.join(srv.vault.image_root, ".staging", staging_id)


def _png_part(name, color):
    return ("file", (name, _png_bytes(color), "image/png"))


def _caption_part(name):
    return ("file", (name, b"test-tag", "text/plain"))


def _key(value) -> dict:
    return {"Idempotency-Key": f'"{value}"'}


def _stage(client, staging_id, parts, headers=None):
    return client.post(
        f"{_STAGING}/{staging_id}/files", files=parts, headers=headers or {}
    )


def _staged_paths(srv, staging_id) -> list:
    session = srv.staging_sessions[staging_id]
    with session["lock"]:
        return [entry["file_path"] for entry in session["staged_files"]]


def _staged_names(srv, staging_id) -> list:
    session = srv.staging_sessions[staging_id]
    with session["lock"]:
        return [entry["original_file_name"] for entry in session["staged_files"]]


def _files_on_disk(staging_dir) -> set:
    return {os.path.join(staging_dir, name) for name in os.listdir(staging_dir)}


def _detail_code(response):
    return response.json()["detail"]["code"]


def _readers(srv) -> int:
    """Read leases the active library has out right now."""
    coordinator = srv.library_coordinator
    return coordinator._readers.get(coordinator.generation, 0)


def _wait_for_readers(srv, expected, timeout_s=10):
    deadline = time.time() + timeout_s
    while _readers(srv) != expected:
        assert time.time() < deadline, (
            f"library has {_readers(srv)} reader(s), expected {expected}"
        )
        time.sleep(0.05)


def _backdate(srv, staging_id):
    """Make a session look idle for longer than the reaper's TTL."""
    session = srv.staging_sessions[staging_id]
    with session["lock"]:
        session["last_update_epoch_ms"] = int(time.time() * 1000) - (
            (import_mod._STAGING_SESSION_TTL_S + 60) * 1000
        )


def test_a_retried_batch_is_staged_once(shared):
    srv, client = shared
    staging_id = _open_session(client)

    first = _stage(
        client,
        staging_id,
        [_png_part("one.png", (1, 2, 3))],
        headers=_key("test-batch-1"),
    )
    assert first.status_code == 200, first.text
    assert first.json()["replayed"] is False
    staged = _staged_paths(srv, staging_id)

    again = _stage(
        client,
        staging_id,
        [_png_part("one.png", (1, 2, 3))],
        headers=_key("test-batch-1"),
    )
    assert again.status_code == 200, again.text
    assert again.json() == {**first.json(), "replayed": True}
    assert _staged_paths(srv, staging_id) == staged
    assert _files_on_disk(_staging_dir(srv, staging_id)) == set(staged)

    # A new key is a new batch.
    other = _stage(
        client,
        staging_id,
        [_png_part("two.png", (4, 5, 6))],
        headers=_key("test-batch-2"),
    )
    assert other.status_code == 200, other.text
    assert other.json()["replayed"] is False
    assert _staged_names(srv, staging_id) == ["one.png", "two.png"]


def test_an_unusable_idempotency_key_is_refused(shared):
    """A caller that asked for idempotency is never treated as one that did not."""
    srv, client = shared
    staging_id = _open_session(client)

    r = _stage(
        client,
        staging_id,
        [_png_part("blank-key.png", (7, 7, 7))],
        headers={"Idempotency-Key": '""'},
    )
    assert r.status_code == 400, r.text
    assert _detail_code(r) == "invalid_idempotency_key"
    assert _staged_paths(srv, staging_id) == []

    r = _stage(
        client,
        staging_id,
        [_png_part("blank-key.png", (7, 7, 7))],
        headers=_key("test-batch-ok"),
    )
    assert r.status_code == 200, r.text
    assert _staged_names(srv, staging_id) == ["blank-key.png"]


def test_a_reused_key_with_different_files_is_refused(shared):
    srv, client = shared
    staging_id = _open_session(client)
    r = _stage(
        client,
        staging_id,
        [_png_part("first.png", (8, 9, 10))],
        headers=_key("test-reused"),
    )
    assert r.status_code == 200, r.text
    staged = _staged_paths(srv, staging_id)

    r = _stage(
        client,
        staging_id,
        [_png_part("second.png", (11, 12, 13))],
        headers=_key("test-reused"),
    )
    assert r.status_code == 422, r.text
    assert _detail_code(r) == "idempotency_key_mismatch"
    assert _staged_paths(srv, staging_id) == staged
    assert _files_on_disk(_staging_dir(srv, staging_id)) == set(staged)


def test_a_retry_while_its_batch_is_still_uploading_is_told_to_wait(
    shared, monkeypatch
):
    srv, client = shared
    staging_id = _open_session(client)
    gate = _park_uploads_between_files(monkeypatch)

    def batch():
        return [_png_part("c.png", (14, 15, 16)), _caption_part("c.txt")]

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_stage, client, staging_id, batch(), _key("test-batch-c"))
        try:
            gate.wait_entered()
            retry = _stage(client, staging_id, batch(), headers=_key("test-batch-c"))
            assert retry.status_code == 409, retry.text
            assert _detail_code(retry) == "batch_in_progress"
        finally:
            gate.release.set()
        r = first.result(timeout=30)
    assert r.status_code == 200, r.text
    assert _staged_names(srv, staging_id) == ["c.png"]

    # Once the first attempt has finished, a retry gets its answer.
    replay = _stage(client, staging_id, batch(), headers=_key("test-batch-c"))
    assert replay.json() == {**r.json(), "replayed": True}


def test_a_cancel_during_an_upload_leaves_the_files_to_that_upload(shared, monkeypatch):
    """The upload still writing owns the directory; the last one out removes it."""
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    gate = _park_uploads_between_files(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as pool:
        upload = pool.submit(
            _stage,
            client,
            staging_id,
            [_png_part("d.png", (17, 18, 19)), _caption_part("d.txt")],
        )
        try:
            gate.wait_entered()
            cancel = client.delete(f"{_STAGING}/{staging_id}")
            assert cancel.status_code == 200, cancel.text
            assert cancel.json()["cleanup"] == "deferred"
            assert os.listdir(staging_dir), "the parked upload's file must survive"
        finally:
            gate.release.set()
        r = upload.result(timeout=30)

    assert r.status_code == 409, r.text
    assert _detail_code(r) == "staging_closed"
    assert r.json()["detail"]["stage"] == "cancelled"
    assert not os.path.lexists(staging_dir)
    assert staging_id not in srv.staging_sessions


def test_a_commit_waits_for_uploads_still_being_received(shared, monkeypatch):
    srv, client = shared
    staging_id = _open_session(client)
    r = _stage(client, staging_id, [_png_part("e1.png", (20, 21, 22))])
    assert r.status_code == 200, r.text
    gate = _park_uploads_between_files(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as pool:
        upload = pool.submit(
            _stage,
            client,
            staging_id,
            [_png_part("e2.png", (23, 24, 25)), _caption_part("e2.txt")],
        )
        try:
            gate.wait_entered()
            commit = client.post(f"{_STAGING}/{staging_id}/commit")
            assert commit.status_code == 409, commit.text
            assert _detail_code(commit) == "uploads_in_flight"
            assert commit.json()["detail"]["stage"] == "staging"
        finally:
            gate.release.set()
        r = upload.result(timeout=30)
    assert r.status_code == 200, r.text

    commit = client.post(f"{_STAGING}/{staging_id}/commit")
    assert commit.status_code == 200, commit.text
    assert commit.json()["staged_count"] == 2
    _wait_for_stage(client, staging_id, target=("completed",))


def test_concurrent_commits_start_one_import_that_holds_the_library(
    shared, monkeypatch
):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    commit_url = f"{_STAGING}/{staging_id}/commit"
    status_url = f"{_STAGING}/{staging_id}/status"
    r = _stage(client, staging_id, [_png_part("f.png", (26, 27, 28))])
    assert r.status_code == 200, r.text
    baseline = _readers(srv)

    handing_over = _Gate()
    monkeypatch.setattr(
        srv.vault, "submit_task", handing_over.wrap(srv.vault.submit_task)
    )
    running = _Gate()
    monkeypatch.setattr(
        PictureImportTask,
        "_import_staged_files",
        running.wrap(PictureImportTask._import_staged_files),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, commit_url)
            try:
                handing_over.wait_entered()
                second = client.post(commit_url)
                assert second.status_code == 409, second.text
                assert _detail_code(second) == "commit_in_progress"
                assert client.get(status_url).json()["stage"] == "committing"

                cancel = client.delete(f"{_STAGING}/{staging_id}")
                assert cancel.status_code == 409, cancel.text
                assert _detail_code(cancel) == "import_committed"

                late = _stage(client, staging_id, [_png_part("late.png", (1, 1, 1))])
                assert late.status_code == 409, late.text
                assert _detail_code(late) == "staging_closed"
            finally:
                handing_over.release.set()
            committed = first.result(timeout=30)
        assert committed.status_code == 200, committed.text
        assert committed.json()["replayed"] is False

        repeat = client.post(commit_url)
        assert repeat.status_code == 200, repeat.text
        assert repeat.json() == {**committed.json(), "replayed": True}

        running.wait_entered()
        # The import holds the library while it runs.
        _wait_for_readers(srv, baseline + 1)
        status = client.get(status_url).json()
        assert status["stage"] == "importing"
        assert status["phase"] == "processing"
    finally:
        handing_over.release.set()
        running.release.set()

    _wait_for_stage(client, staging_id, target=("completed",))
    _wait_for_readers(srv, baseline)
    assert not os.path.lexists(staging_dir)


def _face_worker_down(srv, patch):
    patch.setattr(
        srv.vault,
        "worker_unavailable_reason",
        lambda task_type: "test-face-worker is unavailable",
    )


def _free_space_unreadable(srv, patch):
    real_disk_usage = import_mod.shutil.disk_usage

    def disk_usage(path):
        if path == srv.vault.image_root:
            raise OSError(errno.EIO, "test-unreadable volume", path)
        return real_disk_usage(path)

    patch.setattr(import_mod.shutil, "disk_usage", disk_usage)


def _no_free_space(srv, patch):
    real_disk_usage = import_mod.shutil.disk_usage

    def disk_usage(path):
        usage = real_disk_usage(path)
        return usage._replace(free=0) if path == srv.vault.image_root else usage

    patch.setattr(import_mod.shutil, "disk_usage", disk_usage)


def _runner_down(srv, patch):
    patch.setattr(srv.vault, "submit_task", lambda task: None)


@pytest.mark.parametrize(
    ("refuse", "status_code", "message"),
    [
        (_face_worker_down, 400, "test-face-worker is unavailable"),
        (_free_space_unreadable, 503, "could not check free space"),
        (_no_free_space, 507, "Not enough disk space"),
        (_runner_down, 503, "Task runner is not available"),
    ],
    ids=["face-worker-down", "free-space-unreadable", "disk-full", "runner-down"],
)
def test_a_refused_commit_leaves_the_session_open_and_the_library_free(
    shared, monkeypatch, refuse, status_code, message
):
    srv, client = shared
    staging_id = _open_session(client)
    commit_url = f"{_STAGING}/{staging_id}/commit"
    r = _stage(client, staging_id, [_png_part("g.png", (29, 30, status_code % 256))])
    assert r.status_code == 200, r.text
    baseline = _readers(srv)

    with monkeypatch.context() as patch:
        refuse(srv, patch)
        r = client.post(commit_url)
    assert r.status_code == status_code, r.text
    assert message in r.json()["detail"]
    status = client.get(f"{_STAGING}/{staging_id}/status").json()
    assert status["stage"] == "staging"
    _wait_for_readers(srv, baseline, timeout_s=5)

    # With the cause gone, the same session commits.
    r = client.post(commit_url)
    assert r.status_code == 200, r.text
    _wait_for_stage(client, staging_id, target=("completed",))
    _wait_for_readers(srv, baseline)


def test_a_commit_racing_shutdown_leaves_the_session_open_and_the_library_free(
    shared, monkeypatch
):
    """A commit the stopped task runner refuses gives the session back."""
    srv, client = shared
    staging_id = _open_session(client)
    commit_url = f"{_STAGING}/{staging_id}/commit"
    r = _stage(client, staging_id, [_png_part("s.png", (41, 43, 47))])
    assert r.status_code == 200, r.text
    baseline = _readers(srv)

    def stopped_runner(task):
        raise RuntimeError("TaskRunner test-runner is stopped.")

    with monkeypatch.context() as patch:
        patch.setattr(srv.vault, "submit_task", stopped_runner)
        with pytest.raises(RuntimeError, match="test-runner is stopped"):
            client.post(commit_url)
    status = client.get(f"{_STAGING}/{staging_id}/status").json()
    assert status["stage"] == "staging"
    _wait_for_readers(srv, baseline, timeout_s=5)

    # With the runner back, the same session commits.
    r = client.post(commit_url)
    assert r.status_code == 200, r.text
    _wait_for_stage(client, staging_id, target=("completed",))
    _wait_for_readers(srv, baseline)


def test_an_empty_commit_is_refused_until_a_file_is_staged(shared):
    srv, client = shared
    staging_id = _open_session(client)
    commit_url = f"{_STAGING}/{staging_id}/commit"

    r = client.post(commit_url)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "No staged files to import"
    status = client.get(f"{_STAGING}/{staging_id}/status").json()
    assert status["stage"] == "staging"

    r = _stage(client, staging_id, [_png_part("h0.png", (31, 32, 33))])
    assert r.status_code == 200, r.text
    r = client.post(commit_url)
    assert r.status_code == 200, r.text
    _wait_for_stage(client, staging_id, target=("completed",))


def test_the_file_cap_counts_batches_uploaded_at_the_same_time(shared, monkeypatch):
    srv, client = shared
    monkeypatch.setattr(import_mod, "_STAGING_MAX_FILES", 3)
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    r = _stage(
        client,
        staging_id,
        [_png_part("h1.png", (34, 35, 36)), _png_part("h2.png", (37, 38, 39))],
    )
    assert r.status_code == 200, r.text
    gate = _park_uploads_between_files(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as pool:
        late = pool.submit(
            _stage,
            client,
            staging_id,
            [_png_part("h-late.png", (40, 41, 42)), _caption_part("h-late.txt")],
        )
        try:
            gate.wait_entered()
            # This batch fills the session to exactly the cap.
            fills = _stage(client, staging_id, [_png_part("h3.png", (43, 44, 45))])
            assert fills.status_code == 200, fills.text
            assert fills.json()["staged"] == 3
        finally:
            gate.release.set()
        r = late.result(timeout=30)

    assert r.status_code == 413, r.text
    assert _detail_code(r) == "staging_file_limit"
    assert r.json()["detail"]["limit"] == 3
    assert r.json()["detail"]["staged"] == 3
    assert _staged_names(srv, staging_id) == ["h1.png", "h2.png", "h3.png"]
    assert _files_on_disk(staging_dir) == set(_staged_paths(srv, staging_id))


def _fail_second_write_with_enospc(patch, staging_dir):
    """Make the second file written into *staging_dir* hit a full volume."""
    writes = []

    def fake_open(path, mode="r", *args, **kwargs):
        if "w" in mode and os.path.dirname(os.fspath(path)) == staging_dir:
            writes.append(path)
            if len(writes) == 2:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)
        return open(path, mode, *args, **kwargs)

    patch.setattr(import_mod, "open", fake_open, raising=False)


@pytest.mark.parametrize("as_zip", [False, True], ids=["files", "zip"])
def test_a_batch_that_runs_out_of_disk_is_refused_whole(shared, monkeypatch, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    images = {
        "i1.png": _png_bytes((46, 47, 48)),
        "i2.png": _png_bytes((49, 50, 51)),
        "i3.png": _png_bytes((52, 53, 54)),
    }
    if as_zip:
        parts = [("file", ("i.zip", _zip_bytes(images), "application/zip"))]
    else:
        parts = [("file", (name, data, "image/png")) for name, data in images.items()]

    with monkeypatch.context() as patch:
        _fail_second_write_with_enospc(patch, staging_dir)
        r = _stage(client, staging_id, parts)
    assert r.status_code == 507, r.text
    assert _detail_code(r) == "insufficient_storage"
    assert os.listdir(staging_dir) == []
    assert _staged_paths(srv, staging_id) == []

    # With room again, the same batch stages whole.
    r = _stage(client, staging_id, parts)
    assert r.status_code == 200, r.text
    assert sorted(_staged_names(srv, staging_id)) == sorted(images)


def test_the_reaper_leaves_a_session_whose_files_are_in_use(shared, monkeypatch):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    session = srv.staging_sessions[staging_id]
    r = _stage(client, staging_id, [_png_part("j1.png", (55, 56, 57))])
    assert r.status_code == 200, r.text

    # An upload is still being received.
    gate = _park_uploads_between_files(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        upload = pool.submit(
            _stage,
            client,
            staging_id,
            [_png_part("j2.png", (58, 59, 60)), _caption_part("j2.txt")],
        )
        try:
            gate.wait_entered()
            _backdate(srv, staging_id)
            import_mod._reap_staging_sessions(srv)
            assert srv.staging_sessions.get(staging_id) is session
            assert session["stage"] == "staging"
        finally:
            gate.release.set()
        r = upload.result(timeout=30)
    assert r.status_code == 200, r.text

    # A commit is handing the session to the import.
    handing_over = _Gate()
    monkeypatch.setattr(
        srv.vault, "submit_task", handing_over.wrap(srv.vault.submit_task)
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        commit = pool.submit(client.post, f"{_STAGING}/{staging_id}/commit")
        try:
            handing_over.wait_entered()
            _backdate(srv, staging_id)
            import_mod._reap_staging_sessions(srv)
            assert srv.staging_sessions.get(staging_id) is session
            assert os.path.isdir(staging_dir)
        finally:
            handing_over.release.set()
        r = commit.result(timeout=30)
    assert r.status_code == 200, r.text
    _wait_for_stage(client, staging_id, target=("completed",))


def test_the_reaper_keeps_a_session_whose_files_it_could_not_remove(
    shared, monkeypatch
):
    """Nothing else knows the files are there, so the record waits for a retry."""
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    session = srv.staging_sessions[staging_id]
    r = _stage(client, staging_id, [_png_part("k.png", (61, 62, 63))])
    assert r.status_code == 200, r.text
    _backdate(srv, staging_id)
    real_rmtree = import_mod.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if os.fspath(path) == staging_dir:
            raise PermissionError(errno.EACCES, "test-refused", path)
        return real_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(import_mod.shutil, "rmtree", rmtree)
        import_mod._reap_staging_sessions(srv)
    assert srv.staging_sessions.get(staging_id) is session
    assert os.path.isdir(staging_dir)

    import_mod._reap_staging_sessions(srv)
    assert staging_id not in srv.staging_sessions
    assert not os.path.lexists(staging_dir)


def test_staging_dirs_no_session_owns_are_swept(shared):
    """A restart forgets every session; their directories are removed once old."""
    srv, client = shared
    live_dir = _staging_dir(srv, _open_session(client))
    staging_root = os.path.join(srv.vault.image_root, ".staging")
    old_orphan = os.path.join(staging_root, "test-orphan-old")
    fresh_orphan = os.path.join(staging_root, "test-orphan-fresh")
    for path in (old_orphan, fresh_orphan):
        os.makedirs(path)
        with open(os.path.join(path, "left.png"), "wb") as fh:
            fh.write(_png_bytes())
    stale = time.time() - import_mod._STAGING_SESSION_TTL_S - 60
    for path in (old_orphan, live_dir):
        os.utime(path, (stale, stale))

    _open_session(client)

    assert not os.path.lexists(old_orphan)
    assert os.path.isdir(fresh_orphan), "a fresh directory may be a session opening"
    assert os.path.isdir(live_dir), "a live session's directory is its own"
    shutil.rmtree(fresh_orphan)


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating a symlink needs privileges on Windows"
)
def test_the_sweep_unlinks_a_symlink_without_following_it(shared):
    srv, client = shared
    _open_session(client)
    staging_root = os.path.join(srv.vault.image_root, ".staging")
    with tempfile.TemporaryDirectory() as outside:
        kept = os.path.join(outside, "keep.png")
        with open(kept, "wb") as fh:
            fh.write(_png_bytes())
        link = os.path.join(staging_root, "test-orphan-link")
        os.symlink(outside, link)
        stale = time.time() - import_mod._STAGING_SESSION_TTL_S - 60
        os.utime(link, (stale, stale), follow_symlinks=False)

        _open_session(client)

        assert not os.path.lexists(link)
        assert os.path.isfile(kept)


def test_a_failed_import_removes_its_staged_files_and_frees_the_library(
    shared, monkeypatch
):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    r = _stage(client, staging_id, [_png_part("l.png", (64, 65, 66))])
    assert r.status_code == 200, r.text
    baseline = _readers(srv)

    def fail(task):
        raise RuntimeError("test-import-failure")

    monkeypatch.setattr(PictureImportTask, "_import_staged_files", fail)
    r = client.post(f"{_STAGING}/{staging_id}/commit")
    assert r.status_code == 200, r.text

    status = _wait_for_stage(client, staging_id)
    assert status["stage"] == "failed", status
    assert "test-import-failure" in status["error"]
    assert not os.path.lexists(staging_dir)
    _wait_for_readers(srv, baseline)


# ---------------------------------------------------------------------------
# An upload's disk work runs on a worker thread, never on the event loop; a
# zip entry is copied a bounded chunk at a time; a caption file has a size cap.
# ---------------------------------------------------------------------------

_MIB = 1024 * 1024


def _on_event_loop() -> bool:
    """True when the caller runs on a thread that is running an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _record_writes_into(patch, directory) -> list:
    """For each file opened for writing in *directory*, note whether it was
    opened on the event loop."""
    on_loop = []

    def recording_open(path, mode="r", *args, **kwargs):
        if "w" in mode and os.path.dirname(os.fspath(path)) == directory:
            on_loop.append(_on_event_loop())
        return open(path, mode, *args, **kwargs)

    patch.setattr(import_mod, "open", recording_open, raising=False)
    return on_loop


class _ReadRecorder:
    """A zip entry open for reading that notes each read's size and whether
    the read ran on the event loop."""

    def __init__(self, handle, reads):
        self._handle = handle
        self._reads = reads

    def read(self, n=-1):
        self._reads.append((n, _on_event_loop()))
        return self._handle.read(n)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._handle.__exit__(*exc_info)


def _record_zip_entry_reads(patch) -> list:
    reads = []
    real_open = zipfile.ZipFile.open

    def recording_open(self, name, mode="r", *args, **kwargs):
        handle = real_open(self, name, mode, *args, **kwargs)
        return _ReadRecorder(handle, reads) if mode == "r" else handle

    patch.setattr(zipfile.ZipFile, "open", recording_open)
    return reads


def _caption(size: int) -> bytes:
    """A comma-separated tag list padded to exactly *size* bytes."""
    head = b"test-tag-a, test-tag-b, "
    return head + b"x" * (size - len(head))


def _parts(files: dict, zip_name=None) -> list:
    """Multipart ``file`` parts for *files*, or one zip holding them all."""
    if zip_name:
        return [("file", (zip_name, _zip_bytes(files), "application/zip"))]
    return [
        ("file", (name, data, "text/plain" if name.endswith(".txt") else "image/png"))
        for name, data in files.items()
    ]


@pytest.mark.parametrize("as_zip", [False, True], ids=["files", "zip"])
def test_staging_writes_files_on_a_worker_thread(shared, monkeypatch, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    images = {"m1.png": _png_bytes((67, 68, 69)), "m2.png": _png_bytes((70, 71, 72))}

    with monkeypatch.context() as patch:
        on_loop = _record_writes_into(patch, staging_dir)
        r = _stage(client, staging_id, _parts(images, "m.zip" if as_zip else None))
    assert r.status_code == 200, r.text
    assert sorted(_staged_names(srv, staging_id)) == sorted(images)
    assert on_loop == [False, False]


def test_a_zip_entry_is_copied_in_bounded_chunks_on_a_worker_thread(
    shared, monkeypatch
):
    srv, client = shared
    staging_id = _open_session(client)
    # Deflated and several chunks long, so a whole-entry read would show.
    payload = bytes(range(256)) * (3 * _MIB // 256)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.png", payload)
    parts = [("file", ("big.zip", buf.getvalue(), "application/zip"))]

    with monkeypatch.context() as patch:
        reads = _record_zip_entry_reads(patch)
        r = _stage(client, staging_id, parts)
    assert r.status_code == 200, r.text
    assert reads, "the entry was never read"
    assert all(0 < size <= _MIB for size, _ in reads), reads
    assert not any(on_loop for _, on_loop in reads), reads
    [staged] = _staged_paths(srv, staging_id)
    with open(staged, "rb") as fh:
        assert fh.read() == payload


def test_a_corrupt_zip_entry_leaves_no_partial_file(shared):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    marker = b"test-corrupt-entry"
    archive = bytearray(
        _zip_bytes({"q1.png": _png_bytes((73, 74, 75)), "q2.png": marker * 1000})
    )
    # _zip_bytes stores entries uncompressed: flip a byte of the second
    # entry's data so it no longer matches its CRC.
    archive[archive.index(marker)] ^= 0xFF

    r = _stage(
        client, staging_id, [("file", ("q.zip", bytes(archive), "application/zip"))]
    )
    assert r.status_code == 200, r.text
    assert r.json()["skipped"] == ["q.zip"]
    assert _files_on_disk(staging_dir) == set(_staged_paths(srv, staging_id))


@pytest.mark.parametrize("as_zip", [False, True], ids=["files", "zip"])
def test_staging_keeps_no_caption_over_one_mib(shared, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    files = {
        "n.png": _png_bytes((76, 77, 78)),
        "n-fits.txt": _caption(_MIB),
        "n-over.txt": _caption(_MIB + 1),
    }

    r = _stage(client, staging_id, _parts(files, "n.zip" if as_zip else None))
    assert r.status_code == 200, r.text
    # A caption the client sent is reported back; one inside a zip is left
    # out like any other entry the import cannot use.
    assert r.json()["skipped"] == ([] if as_zip else ["n-over.txt"])
    session = srv.staging_sessions[staging_id]
    with session["lock"]:
        kept = {
            stem: len(text) for stem, text in session["sidecar_text_by_stem"].items()
        }
    assert kept == {"n-fits": _MIB}


def _record_parsed_caption_sizes(patch) -> list:
    sizes = []
    real_parse = import_mod._parse_sidecar_tags

    def recording_parse(raw_text):
        sizes.append(len(raw_text))
        return real_parse(raw_text)

    patch.setattr(import_mod, "_parse_sidecar_tags", recording_parse)
    return sizes


@pytest.mark.parametrize("as_zip", [False, True], ids=["files", "zip"])
def test_the_one_shot_import_leaves_out_a_caption_over_one_mib(
    shared, monkeypatch, as_zip
):
    _srv, client = shared
    files = {
        "p-fits.png": _png_bytes((79, 80, 81 + as_zip)),
        "p-fits.txt": _caption(_MIB),
        "p-over.png": _png_bytes((83, 84, 85 + as_zip)),
        "p-over.txt": _caption(_MIB + 1),
    }

    with monkeypatch.context() as patch:
        parsed = _record_parsed_caption_sizes(patch)
        r = client.post(
            f"{API}/pictures/import",
            files=_parts(files, "p.zip" if as_zip else None),
        )
    assert r.status_code == 200, r.text
    assert wait_for_import_task(client, r.json()["task_id"])["status"] == "completed"
    assert parsed == [_MIB]


def test_the_one_shot_import_unpacks_a_zip_on_a_worker_thread(shared, monkeypatch):
    _srv, client = shared
    archive = _zip_bytes(
        {"r.png": _png_bytes((87, 88, 89)), "r.txt": b"test-tag-a, test-tag-b"}
    )

    with monkeypatch.context() as patch:
        reads = _record_zip_entry_reads(patch)
        r = client.post(
            f"{API}/pictures/import",
            files=[("file", ("r.zip", archive, "application/zip"))],
        )
    assert r.status_code == 200, r.text
    assert wait_for_import_task(client, r.json()["task_id"])["status"] == "completed"
    assert reads, "the zip's entries were never read"
    assert not any(on_loop for _, on_loop in reads), reads


# ---------------------------------------------------------------------------
# A cancelled upload keeps its session and its files until the worker thread
# copying it has stopped, and that copy stops within one chunk. These call the
# upload route's coroutine directly so the test can cancel it the way a server
# shutdown does.
# ---------------------------------------------------------------------------

_CANCELLED_BATCH = "test-cancelled-upload"


class _GatedUploadFile(io.BytesIO):
    """An upload's spooled file that counts the reads made off the event loop
    and holds the first one *holds_at* picks until the test releases it."""

    def __init__(self, data: bytes, holds_at):
        super().__init__(data)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reads_off_loop = 0
        self.reads_when_held = None
        self._holds_at = holds_at
        self._held = False

    def read(self, size=-1):
        if not _on_event_loop():
            self.reads_off_loop += 1
            if not self._held and self._holds_at(self):
                self._held = True
                self.reads_when_held = self.reads_off_loop
                self.entered.set()
                if not self.release.wait(60):
                    raise RuntimeError("test gate was never released")
        return super().read(size)


def _gated_upload(as_zip, staging_dir):
    """A 3 MiB file held at its second chunk, or a stored zip of three entries
    held at the first chunk of the second, a 3 MiB entry, once it is being
    written into *staging_dir*."""
    if as_zip:
        data = _zip_bytes(
            {
                "s1.png": b"\x01" * 300_000,
                "s2.png": b"\x02" * (3 * _MIB),
                "s3.png": b"\x03" * 300_000,
            }
        )
        source = _GatedUploadFile(
            data, lambda _source: len(os.listdir(staging_dir)) >= 2
        )
        name = "s.zip"
    else:
        data = bytes(range(256)) * (3 * _MIB // 256)
        source = _GatedUploadFile(data, lambda source: source.reads_off_loop == 2)
        name = "s.png"
    return source, UploadFile(file=source, filename=name, size=len(data))


def _stage_files_route(srv):
    """The upload route's coroutine, bound to *srv* the way the server binds it."""
    router = APIRouter()
    import_mod.register_routes(router, srv)
    [endpoint] = [
        route.endpoint
        for route in router.routes
        if route.path == "/pictures/import/staging/{staging_id}/files"
        and "POST" in route.methods
    ]
    return endpoint


def _call_stage_files(srv, staging_id, upload):
    """The upload route's coroutine for one file, under the session's library."""
    session = srv.staging_sessions[staging_id]
    lease = SimpleNamespace(
        library_uuid=session["library_uuid"], generation=session["generation"]
    )
    request = SimpleNamespace(state=SimpleNamespace(library_lease=lease))
    return _stage_files_route(srv)(
        request,
        staging_id,
        file=[upload],
        key=[],
        idempotency_key=f'"{_CANCELLED_BATCH}"',
    )


def _upload_state(session) -> dict:
    with session["lock"]:
        return {
            "inflight": session["inflight"],
            "batches": {key: rec["state"] for key, rec in session["batches"].items()},
            "staged": [entry["file_path"] for entry in session["staged_files"]],
        }


async def _wait_for_the_gate(source, task):
    deadline = time.monotonic() + 30
    while not source.entered.is_set():
        if task.done():
            task.result()
            raise AssertionError("the upload finished without reaching the gate")
        assert time.monotonic() < deadline, "the upload never reached the gate"
        await asyncio.sleep(0.01)


async def _cancel_at_the_gate(srv, staging_id, source, upload) -> dict:
    """Start the upload, cancel it while its copy is held, give the cancel
    half a second, then release the copy. Returns what the session and the
    request looked like just before the release."""
    session = srv.staging_sessions[staging_id]
    task = asyncio.create_task(_call_stage_files(srv, staging_id, upload))
    try:
        await _wait_for_the_gate(source, task)
        task.cancel()
        await asyncio.wait({task}, timeout=0.5)
        held = {"request_done": task.done(), **_upload_state(session)}
    finally:
        source.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    return held


@pytest.mark.parametrize("as_zip", [False, True], ids=["plain", "zip"])
def test_a_cancelled_upload_keeps_its_session_until_the_copy_stops(shared, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    session = srv.staging_sessions[staging_id]
    staging_dir = _staging_dir(srv, staging_id)
    before = _upload_state(session)
    source, upload = _gated_upload(as_zip, staging_dir)

    held = asyncio.run(_cancel_at_the_gate(srv, staging_id, source, upload))

    assert held == {
        "request_done": False,
        "inflight": 1,
        "batches": {_CANCELLED_BATCH: "in_progress"},
        "staged": before["staged"],
    }
    assert _upload_state(session) == {
        "inflight": 0,
        "batches": {},
        "staged": before["staged"],
    }
    assert os.listdir(staging_dir) == []


@pytest.mark.parametrize("as_zip", [False, True], ids=["plain", "zip"])
def test_a_torn_down_loop_waits_for_the_upload_copy(shared, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    session = srv.staging_sessions[staging_id]
    staging_dir = _staging_dir(srv, staging_id)
    source, upload = _gated_upload(as_zip, staging_dir)
    at_release = {}

    def release_later():
        if source.entered.wait(30):
            time.sleep(0.5)
            at_release.update(_upload_state(session))
        source.release.set()

    async def leave_while_copying():
        task = asyncio.create_task(_call_stage_files(srv, staging_id, upload))
        await _wait_for_the_gate(source, task)
        # Returning with the upload still running makes asyncio.run cancel it.

    timer = threading.Thread(target=release_later)
    timer.start()
    try:
        asyncio.run(leave_while_copying())
    finally:
        source.release.set()
        timer.join()

    assert at_release["inflight"] == 1, at_release
    state = _upload_state(session)
    assert (state["inflight"], state["batches"]) == (0, {}), state
    assert os.listdir(staging_dir) == []


def _record_zip_members_opened(patch, source) -> list:
    """The names of the entries opened in the zip read from *source*."""
    opened = []
    real_open = zipfile.ZipFile.open

    def recording_open(self, name, *args, **kwargs):
        if self.fp is source:
            opened.append(getattr(name, "filename", name))
        return real_open(self, name, *args, **kwargs)

    patch.setattr(zipfile.ZipFile, "open", recording_open)
    return opened


def _signal_when_the_copy_ends(patch, as_zip, source) -> threading.Event:
    """An event set once the worker thread is done with *source*, however its
    copy ended: the plain copy returns, or the zip around *source* closes."""
    ended = threading.Event()
    if as_zip:
        real_exit = zipfile.ZipFile.__exit__

        def exit_and_signal(self, *exc_info):
            ours = self.fp is source
            try:
                return real_exit(self, *exc_info)
            finally:
                if ours:
                    ended.set()

        patch.setattr(zipfile.ZipFile, "__exit__", exit_and_signal)
    else:
        real_copy = import_mod._copy_upload_part

        def copy_and_signal(*args, **kwargs):
            try:
                return real_copy(*args, **kwargs)
            finally:
                ended.set()

        patch.setattr(import_mod, "_copy_upload_part", copy_and_signal)
    return ended


@pytest.mark.parametrize("as_zip", [False, True], ids=["plain", "zip"])
def test_a_cancelled_upload_stops_copying_within_one_chunk(shared, monkeypatch, as_zip):
    srv, client = shared
    staging_id = _open_session(client)
    staging_dir = _staging_dir(srv, staging_id)
    source, upload = _gated_upload(as_zip, staging_dir)

    with monkeypatch.context() as patch:
        opened = _record_zip_members_opened(patch, source)
        copy_ended = _signal_when_the_copy_ends(patch, as_zip, source)
        asyncio.run(_cancel_at_the_gate(srv, staging_id, source, upload))
        assert copy_ended.wait(30), "the copy never ended"

    # The copy read nothing more from the upload after the held chunk.
    if as_zip:
        assert source.reads_off_loop == source.reads_when_held
        assert opened == ["s1.png", "s2.png"]
    else:
        # The held read was the second 1 MiB chunk of three.
        assert source.reads_off_loop == source.reads_when_held == 2
    assert os.listdir(staging_dir) == []
