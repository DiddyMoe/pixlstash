"""Project attachments: a file and its row live and die together.

Invariants pinned here:

* an attachment is only ever filed on the project it was checked against,
  even when that project's id is reused by a new project mid-upload;
* deleting a project never removes a file that a live row references;
* no failure path leaves a file without a row, or a row without its file.

Each race is staged deterministically by wrapping the writer's ``run_task``,
so it fires at the one point the window opens.
"""

import errno
import gc
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

import pixlstash.routes.projects as projects_routes
from pixlstash.db_models.project import ProjectAttachment
from pixlstash.server import Server


@pytest.fixture
def env():
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as handle:
        handle.write(json.dumps({"port": 8000}))
    server = Server(config_path)
    client = TestClient(server.api, raise_server_exceptions=False)
    r = client.post("/login", json={"username": "testuser", "password": "testpassword"})
    assert r.status_code == 200, r.text
    real_run_task = server.vault.db.run_task
    try:
        yield client, server, real_run_task
    finally:
        server.vault.db.__dict__.pop("run_task", None)
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _rows(real_run_task, pid):
    return real_run_task(
        lambda s: [
            (a.id, a.stored_path)
            for a in s.exec(
                select(ProjectAttachment).where(ProjectAttachment.project_id == pid)
            ).all()
        ]
    )


def _files(server, pid):
    folder = os.path.join(server.vault.image_root, "projects", str(pid), "attachments")
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def _new_project(client, name):
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _upload(client, pid, name="a.txt", body=b"attachment bytes"):
    return client.post(
        f"/projects/{pid}/attachments", files={"file": (name, body, "text/plain")}
    )


def test_an_upload_is_not_filed_on_a_project_that_reused_the_id(env):
    """The checked project is deleted and a new one takes its id while the
    file is written: the upload is refused, not filed on the new project."""
    client, server, real_run_task = env
    pid = _new_project(client, "Checked")
    fired = []

    def run_task(fn, *args, **kwargs):
        if not fired and fn.__name__ == "insert_record":
            fired.append(True)
            assert client.delete(f"/projects/{pid}").status_code == 200
            fired.append(_new_project(client, "Recreated"))
        return real_run_task(fn, *args, **kwargs)

    server.vault.db.run_task = run_task
    r = _upload(client, pid)
    del server.vault.db.run_task
    assert fired and fired[1] == pid, "the id was not reused: the race was not staged"
    assert r.status_code == 404, (r.status_code, r.text)
    assert _rows(real_run_task, pid) == []
    assert _files(server, pid) == []


def test_deleting_a_project_keeps_a_recreated_projects_file(env):
    """A new project takes the deleted one's id and uploads before the
    delete's file cleanup runs: the cleanup removes only the old files."""
    client, server, real_run_task = env
    pid = _new_project(client, "Old")
    assert _upload(client, pid, "old.txt").status_code == 200
    (old_file,) = _files(server, pid)
    fired = []

    def run_task(fn, *args, **kwargs):
        result = real_run_task(fn, *args, **kwargs)
        if not fired and fn.__name__ == "do_delete":
            fired.append(True)
            new_pid = _new_project(client, "New")
            assert new_pid == pid, "the id was not reused: the race was not staged"
            r = _upload(client, new_pid, "new.txt", b"new project bytes")
            assert r.status_code == 200, r.text
            fired.append(r.json()["id"])
        return result

    server.vault.db.run_task = run_task
    r = client.delete(f"/projects/{pid}")
    del server.vault.db.run_task
    assert r.status_code == 200, r.text
    assert len(fired) == 2
    assert old_file not in _files(server, pid)
    download = client.get(f"/projects/{pid}/attachments/{fired[1]}")
    assert download.status_code == 200, download.text
    assert download.content == b"new project bytes"


def test_a_failed_write_leaves_no_partial_file(env, monkeypatch):
    """The disk fills mid-write: the partial file goes, and no row is made."""
    client, server, real_run_task = env
    pid = _new_project(client, "Full disk")
    real_open = open

    class HalfWritten:
        def __init__(self, path, mode):
            self._handle = real_open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def write(self, data):
            self._handle.write(data[: len(data) // 2])
            self._handle.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(projects_routes, "open", HalfWritten, raising=False)
    r = _upload(client, pid, body=b"x" * 4096)
    monkeypatch.undo()
    assert r.status_code == 500, (r.status_code, r.text)
    assert _files(server, pid) == []
    assert _rows(real_run_task, pid) == []


def test_a_failure_after_the_commit_keeps_the_rows_file(env):
    """The insert commits and the request then fails: the committed row keeps
    its file (the caller's cleanup is only for a row that never committed)."""
    client, server, real_run_task = env
    pid = _new_project(client, "Committed")

    def run_task(fn, *args, **kwargs):
        result = real_run_task(fn, *args, **kwargs)
        if fn.__name__ == "insert_record":
            raise RuntimeError("example-failure after the commit")
        return result

    server.vault.db.run_task = run_task
    r = _upload(client, pid)
    del server.vault.db.run_task
    assert r.status_code == 500, (r.status_code, r.text)
    ((_, stored_path),) = _rows(real_run_task, pid)
    assert _files(server, pid) == [os.path.basename(stored_path)]


def test_a_cleanup_error_does_not_replace_the_404(env, monkeypatch):
    """The project is deleted mid-upload and removing the orphan file fails
    with EACCES: the caller still gets the 404, and the failure is logged."""
    client, server, real_run_task = env
    pid = _new_project(client, "Locked")
    fired = []

    def run_task(fn, *args, **kwargs):
        if not fired and fn.__name__ == "insert_record":
            fired.append(True)
            assert client.delete(f"/projects/{pid}").status_code == 200
        return real_run_task(fn, *args, **kwargs)

    def refuse(path):
        raise PermissionError(errno.EACCES, "Permission denied", path)

    server.vault.db.run_task = run_task
    monkeypatch.setattr(projects_routes.os, "remove", refuse)
    r = _upload(client, pid)
    monkeypatch.undo()
    del server.vault.db.run_task
    assert fired
    assert r.status_code == 404, (r.status_code, r.text)
