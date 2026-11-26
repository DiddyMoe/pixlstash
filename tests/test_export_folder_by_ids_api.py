"""POST /pictures/export/folder/selection: the folder export of named pictures.

``POST /pictures/export/folder`` names a selection as one ``?id=`` per picture,
and a large one makes a request line the server refuses. This route takes the
selection and the destination in a JSON body and hands them to the same
destination checks and the same ``generate_folder_export``. These assert:

* both directions of its policy, which is the view route's
  (``loopback_owner_only``): a loopback owner passes, while a READ token, a
  share token, a ComfyUI token and every non-loopback caller (LAN and
  Tailscale even with ``allow_remote_host_ops`` on, public with it off) are
  refused before any task starts;
* it writes exactly the named pictures, Scrapheap ones only with
  ``only_deleted``, and every one of a selection past SQLite's bound-variable
  limit;
* the body is bounded and typed like the ZIP body.
"""

import io
import json
import os
import tempfile
import time
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from pixlstash import auth
from pixlstash.db_models.picture import Picture
from pixlstash.routes.pictures._export import EXPORT_MAX_IDS
from pixlstash.server import Server
from tests.authz_guard import assert_real_route
from tests.network_vectors import LAN_IPV4
from tests.test_export_by_ids_api import ABSENT_IDS, _sqlite_variable_ceiling
from tests.utils import (
    API_PREFIX,
    bearer_client,
    mint_read_token,
    upload_pictures_and_wait,
)

EXPORT = f"{API_PREFIX}/pictures/export"
FOLDER_SELECTION = f"{EXPORT}/folder/selection"
# Cannot exist, so a caller that passes the gate gets the handler's 404 and
# nothing is written or opened.
MISSING_DESTINATION = "/nonexistent-export-folder-selection-destination"
TAILSCALE_IPV4 = "100.64.0.5"
PUBLIC_IPV4 = "8.8.8.8"


def _picture_files():
    """Four small, distinct generated pictures (the importer keeps all four)."""
    files = []
    for index in range(4):
        image = Image.new(
            "RGB", (40 + index * 8, 36 + index * 4), color=(90, 30 + index * 50, 140)
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        files.append(("file", (f"folder_{index}.png", buf.getvalue(), "image/png")))
    return files


@pytest.fixture(scope="module")
def env():
    """One server, owner cookie, three live pictures and one in the Scrapheap.

    ``trusted_proxies=["testclient"]`` lets a test name the caller's address
    with ``X-Forwarded-For``; without the header the in-process client is a
    loopback caller.
    """
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        json.dump({"port": 8000, "trusted_proxies": ["testclient"]}, fh)
    server = Server(config_path)
    try:
        owner = TestClient(server.api)
        r = owner.post(
            f"{API_PREFIX}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert r.status_code == 200, r.text
        status = upload_pictures_and_wait(owner, _picture_files(), timeout_s=30)
        assert status["status"] == "completed", status
        uploaded = sorted(result["picture_id"] for result in status["results"])
        assert len(set(uploaded)) == 4, status
        ids, deleted_id = uploaded[:3], uploaded[3]

        def soft_delete(session):
            pic = session.get(Picture, deleted_id)
            pic.deleted = True
            session.add(pic)
            session.commit()

        server.vault.db.run_task(soft_delete)
        yield server, owner, ids, deleted_id, temp_dir.name
    finally:
        server.close()
        temp_dir.cleanup()


@contextmanager
def _remote_host_ops(server, enabled):
    """Set the live ``allow_remote_host_ops`` flag for the block."""
    cfg = server.auth._server_config
    previous = cfg.get("allow_remote_host_ops")
    cfg["allow_remote_host_ops"] = enabled
    try:
        yield
    finally:
        if previous is None:
            cfg.pop("allow_remote_host_ops", None)
        else:
            cfg["allow_remote_host_ops"] = previous


def _body(picture_ids, destination=MISSING_DESTINATION, **options):
    return {
        "destination": destination,
        "picture_ids": picture_ids,
        "caption_mode": "none",
        "use_original_file_names": True,
        **options,
    }


def _empty_destination(root: str) -> str:
    destination = tempfile.mkdtemp(prefix="destination-", dir=root)
    assert os.listdir(destination) == []
    return destination


def _finish(client, task_id, timeout_s=30):
    """Poll the export to a terminal status and return the status body."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"{EXPORT}/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"export {task_id} did not finish within {timeout_s}s")


def _written_pictures(client, root, picture_ids, **options) -> list[str]:
    """Export *picture_ids* to a new empty folder and list the pictures written."""
    destination = _empty_destination(root)
    with mock.patch(
        "pixlstash.utils.service.export_utils.open_in_file_manager",
        return_value=True,
    ) as opener:
        r = client.post(
            FOLDER_SELECTION, json=_body(picture_ids, destination, **options)
        )
        assert r.status_code == 200, r.text
        status = _finish(client, r.json()["task_id"])
    assert status["status"] == "completed", status
    opener.assert_called_once_with(os.path.realpath(destination))
    return sorted(
        name
        for _dirpath, _dirs, names in os.walk(destination)
        for name in names
        if name.endswith(".png")
    )


def _file_names(server, picture_ids) -> list[str]:
    def read(session):
        rows = session.exec(
            select(Picture.id, Picture.original_file_name).where(
                Picture.id.in_(picture_ids)
            )
        ).all()
        return {pid: name for pid, name in rows}

    by_id = server.vault.db.run_task(read)
    return sorted(by_id[pid] for pid in picture_ids)


def _assert_refused(server, client, status_code=403, detail=None, **kwargs):
    """POST the selection as *client* and assert it is refused with no task."""
    before = set(server.export_tasks)
    r = client.post(FOLDER_SELECTION, json=_body([1]), **kwargs)
    assert r.status_code == status_code, r.text
    if detail is not None:
        assert detail in r.text, r.text
    assert set(server.export_tasks) == before
    return r


def test_the_route_is_mounted_and_the_gate_enforces(env):
    """Every negative below is a refusal of this route, not a 404/405 for a
    route that is not there or a gate that only reports."""
    server, owner, _ids, _deleted, _root = env
    assert_real_route(server.api, "POST", FOLDER_SELECTION, template=FOLDER_SELECTION)
    assert server.authz.enforcing is True
    r = owner.post(FOLDER_SELECTION, json={"destination": MISSING_DESTINATION})
    assert r.status_code == 422, r.text


def test_a_loopback_owner_passes_the_gate(env):
    """The handler past the gate answers the missing destination with 404."""
    server, owner, _ids, _deleted, _root = env
    _assert_refused(server, owner, status_code=404, detail="Destination folder")


def test_the_owner_writes_exactly_the_named_pictures(env):
    server, owner, ids, _deleted, root = env
    assert _written_pictures(owner, root, ids[:2]) == _file_names(server, ids[:2])
    assert _written_pictures(owner, root, [ids[2]]) == _file_names(server, [ids[2]])


@pytest.mark.parametrize("only_deleted", [True, False])
def test_only_deleted_picks_the_scrapheap_side_of_a_selection(env, only_deleted):
    """A Scrapheap selection writes its soft-deleted pictures and never a live
    one, as the Scrapheap view never shows one; any other selection the
    reverse."""
    server, owner, ids, deleted_id, root = env
    selection = [ids[1], deleted_id, ids[0]]
    expected = [deleted_id] if only_deleted else [ids[1], ids[0]]
    written = _written_pictures(owner, root, selection, only_deleted=only_deleted)
    assert written == _file_names(server, expected)


def test_a_selection_past_the_sqlite_variable_limit_writes_every_picture(env):
    server, owner, ids, _deleted, root = env
    with _sqlite_variable_ceiling(server):
        written = _written_pictures(owner, root, ABSENT_IDS + ids)
    assert written == _file_names(server, ids)


def test_a_non_empty_destination_is_refused_as_the_view_route_refuses_it(env):
    """One set of destination checks serves both routes."""
    server, owner, ids, _deleted, root = env
    destination = _empty_destination(root)
    with open(os.path.join(destination, "already-here.txt"), "w") as fh:
        fh.write("test-occupant")
    before = set(server.export_tasks)
    by_body = owner.post(FOLDER_SELECTION, json=_body([ids[0]], destination))
    by_query = owner.post(
        f"{EXPORT}/folder", params={"destination": destination, "id": ids[0]}
    )
    assert by_body.status_code == by_query.status_code == 409, (
        by_body.text,
        by_query.text,
    )
    assert by_body.json() == by_query.json()
    assert set(server.export_tasks) == before


def test_a_read_token_is_refused(env):
    server, owner, _ids, _deleted, _root = env
    reader = bearer_client(server, mint_read_token(owner, "test-folder-read-token"))
    _assert_refused(server, reader)


def test_a_share_token_is_refused_even_for_its_own_picture(env):
    server, owner, ids, _deleted, _root = env
    token = mint_read_token(
        owner,
        "test-folder-share-token",
        resource_type="picture",
        resource_id=ids[0],
    )
    share = bearer_client(server, token)
    before = set(server.export_tasks)
    r = share.post(FOLDER_SELECTION, json=_body([ids[0]]))
    assert r.status_code == 403, r.text
    assert set(server.export_tasks) == before


def test_a_comfyui_token_is_refused(env):
    server, owner, _ids, _deleted, _root = env
    r = owner.post(
        f"{API_PREFIX}/users/me/token",
        json={"description": "test-folder-comfyui-token", "scope": "COMFYUI"},
    )
    assert r.status_code == 200, r.text
    token, token_id = r.json()["token"], r.json()["token_id"]
    try:
        _assert_refused(
            server, bearer_client(server, token), detail=auth.COMFYUI_SCOPE_REFUSAL
        )
    finally:
        owner.delete(f"{API_PREFIX}/users/me/token/{token_id}")


@pytest.mark.parametrize(
    ("address", "flag"),
    [
        pytest.param(LAN_IPV4, True, id="lan-with-the-flag-on"),
        pytest.param(TAILSCALE_IPV4, True, id="tailscale-with-the-flag-on"),
        pytest.param(PUBLIC_IPV4, False, id="public"),
    ],
)
def test_a_caller_off_this_machine_is_refused(env, address, flag):
    server, owner, _ids, _deleted, _root = env
    with _remote_host_ops(server, flag):
        _assert_refused(
            server,
            owner,
            detail="restricted to loopback",
            headers={"X-Forwarded-For": address},
        )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_body([]), id="empty"),
        pytest.param(_body([1] * (EXPORT_MAX_IDS + 1)), id="over-the-cap"),
        pytest.param(_body([2**63]), id="not-a-sqlite-int"),
        pytest.param(_body([1], destination=""), id="empty-destination"),
        pytest.param({"picture_ids": [1]}, id="no-destination"),
        pytest.param(_body([1], query="cat"), id="unknown-field"),
        pytest.param(_body([1], resolution="huge"), id="bad-option"),
    ],
)
def test_a_malformed_body_is_refused_before_an_export_starts(env, body):
    server, owner, _ids, _deleted, _root = env
    before = set(server.export_tasks)
    r = owner.post(FOLDER_SELECTION, json=body)
    assert r.status_code == 422, r.text
    assert set(server.export_tasks) == before
