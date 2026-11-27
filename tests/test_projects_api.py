"""Tests for the Projects API: CRUD, picture assignment, export, and attachments."""

import gc
import json
import os
import tempfile
import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from pixlstash.server import Server
from tests.utils import upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")


def _setup():
    temp_dir = tempfile.TemporaryDirectory()
    image_root = os.path.join(temp_dir.name, "images")
    os.makedirs(image_root, exist_ok=True)
    server_config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(server_config_path, "w") as f:
        f.write(json.dumps({"port": 8000}))
    server = Server(server_config_path)
    client = TestClient(server.api)
    resp = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert resp.status_code == 200
    return temp_dir, client, server


def _upload_picture(client, filename="Bad1.png"):
    img_path = os.path.join(PICTURES_DIR, filename)
    with open(img_path, "rb") as f:
        result = upload_pictures_and_wait(
            client, [("file", (filename, f, "image/png"))]
        )
    assert result["status"] == "completed"
    return result["results"][0]["picture_id"]


def _attach(client, project_id, filename, content=b"attachment"):
    resp = client.post(
        f"/projects/{project_id}/attachments",
        files={"file": (filename, content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _project_dir(server, project_id):
    return os.path.join(server.vault.image_root, "projects", str(project_id))


def test_create_and_get_project():
    temp_dir, client, server = _setup()
    try:
        resp = client.post(
            "/projects", json={"name": "MyProject", "description": "Test"}
        )
        assert resp.status_code == 200
        project = resp.json()
        project_id = project["id"]
        assert project["name"] == "MyProject"

        resp = client.get(f"/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id
        assert resp.json()["name"] == "MyProject"
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_list_projects():
    temp_dir, client, server = _setup()
    try:
        client.post("/projects", json={"name": "ListProject"})

        resp = client.get("/projects")
        assert resp.status_code == 200
        projects = resp.json()
        assert isinstance(projects, list)
        assert any(p["name"] == "ListProject" for p in projects)
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_update_project():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "UpdateMe"})
        project_id = resp.json()["id"]

        resp = client.put(f"/projects/{project_id}", json={"name": "UpdatedName"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "UpdatedName"

        resp = client.get(f"/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "UpdatedName"
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_delete_project_pictures_survive():
    temp_dir, client, server = _setup()
    try:
        pic_id = _upload_picture(client)

        resp = client.post("/projects", json={"name": "DeleteMeProject"})
        project_id = resp.json()["id"]

        client.patch(
            "/pictures/project",
            json={"picture_ids": [pic_id], "project_id": project_id, "mode": "add"},
        )

        resp = client.delete(f"/projects/{project_id}")
        assert resp.status_code == 200

        resp = client.get(f"/projects/{project_id}")
        assert resp.status_code == 404

        resp = client.get(f"/pictures/{pic_id}/metadata")
        assert resp.status_code == 200
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_assign_pictures_to_project():
    temp_dir, client, server = _setup()
    try:
        pic_id = _upload_picture(client)

        resp = client.post("/projects", json={"name": "AssignProject"})
        project_id = resp.json()["id"]

        resp = client.patch(
            "/pictures/project",
            json={"picture_ids": [pic_id], "project_id": project_id, "mode": "add"},
        )
        assert resp.status_code == 200

        resp = client.get(f"/projects/{project_id}/summary")
        assert resp.status_code == 200
        assert resp.json()["image_count"] >= 1
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_project_export_is_valid_zip():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "ExportProject"})
        project_id = resp.json()["id"]

        resp = client.get(f"/projects/{project_id}/export")
        assert resp.status_code == 200
        assert "zip" in resp.headers.get("content-type", "").lower()

        buf = BytesIO(resp.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
        assert any("project.json" in n for n in names)
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_project_attachment_upload_list_delete():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "AttachProject"})
        project_id = resp.json()["id"]

        attachment_content = b"hello attachment"
        resp = client.post(
            f"/projects/{project_id}/attachments",
            files={"file": ("note.txt", attachment_content, "text/plain")},
        )
        assert resp.status_code == 200
        att_id = resp.json()["id"]

        resp = client.get(f"/projects/{project_id}/attachments")
        assert resp.status_code == 200
        attachments = resp.json()
        assert any(a["id"] == att_id for a in attachments)

        resp = client.delete(f"/projects/{project_id}/attachments/{att_id}")
        assert resp.status_code == 200

        resp = client.get(f"/projects/{project_id}/attachments")
        assert resp.status_code == 200
        assert not any(a["id"] == att_id for a in resp.json())

        # URL bookmarks: every viewer opens these, so only web links are stored.
        # Same server, to avoid paying for another environment.
        resp = client.post(
            f"/projects/{project_id}/attachments/url",
            json={"url": "https://example.com/ref", "title": "Ref"},
        )
        assert resp.status_code == 200
        assert resp.json()["url"] == "https://example.com/ref"
        for bad in (
            "javascript:alert(1)",
            " JavaScript:alert(1)",
            "java\tscript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "//example.com/no-scheme",
            "http://[::1",  # urlsplit raises on this; must be a 400, not a 500
        ):
            resp = client.post(
                f"/projects/{project_id}/attachments/url", json={"url": bad}
            )
            assert resp.status_code == 400, bad
        resp = client.get(f"/projects/{project_id}/attachments")
        assert [a["url"] for a in resp.json() if a.get("url")] == [
            "https://example.com/ref"
        ]

        # Deleting a project removes its files and its emptied folder, and
        # leaves a file no row of it names, with the folder holding it.
        other_id = client.post("/projects", json={"name": "Emptied"}).json()["id"]
        _attach(client, other_id, "own.txt")
        assert client.delete(f"/projects/{other_id}").status_code == 200
        assert not os.path.exists(_project_dir(server, other_id))

        _attach(client, project_id, "own.txt")
        att_dir = os.path.join(_project_dir(server, project_id), "attachments")
        with open(os.path.join(att_dir, "stray.txt"), "wb") as f:
            f.write(b"not this project's")
        assert client.delete(f"/projects/{project_id}").status_code == 200
        assert os.listdir(att_dir) == ["stray.txt"]
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_deleting_a_project_keeps_the_files_of_a_project_reusing_its_id():
    """A new project takes the deleted one's id (ids are reused) and uploads
    before the delete's file cleanup runs: its file stays."""
    temp_dir, client, server = _setup()
    try:
        old_id = client.post("/projects", json={"name": "Old"}).json()["id"]
        _attach(client, old_id, "old.txt")
        real_run_task = server.vault.db.run_task
        reused = []

        def run_task(fn, *args, **kwargs):
            result = real_run_task(fn, *args, **kwargs)
            if fn.__name__ == "do_delete":
                new_id = client.post("/projects", json={"name": "New"}).json()["id"]
                reused.append((new_id, _attach(client, new_id, "new.txt", b"new")))
            return result

        server.vault.db.run_task = run_task
        try:
            assert client.delete(f"/projects/{old_id}").status_code == 200
        finally:
            del server.vault.db.run_task
        ((new_id, att_id),) = reused
        assert new_id == old_id, "the id was not reused"
        resp = client.get(f"/projects/{new_id}/attachments/{att_id}")
        assert resp.status_code == 200
        assert resp.content == b"new"
        att_dir = os.path.join(_project_dir(server, new_id), "attachments")
        assert len(os.listdir(att_dir)) == 1
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_an_attachment_whose_project_is_deleted_mid_upload_leaves_no_file():
    """The project is deleted after the file is written and before its row
    is inserted: the insert fails and the written file goes with it."""
    temp_dir, client, server = _setup()
    try:
        project_id = client.post("/projects", json={"name": "Gone"}).json()["id"]
        att_dir = os.path.join(_project_dir(server, project_id), "attachments")
        real_run_task = server.vault.db.run_task

        def run_task(fn, *args, **kwargs):
            if fn.__name__ == "insert_record":
                assert len(os.listdir(att_dir)) == 1, "the file was not written"
                assert client.delete(f"/projects/{project_id}").status_code == 200
            return real_run_task(fn, *args, **kwargs)

        server.vault.db.run_task = run_task
        try:
            with pytest.raises(IntegrityError):
                client.post(
                    f"/projects/{project_id}/attachments",
                    files={"file": ("late.txt", b"late", "text/plain")},
                )
        finally:
            del server.vault.db.run_task
        assert not os.path.exists(att_dir) or os.listdir(att_dir) == []
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_project_get_by_name():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "NamedProject"})
        project_id = resp.json()["id"]

        resp = client.get("/projects/NamedProject")
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_project_duplicate_name_rejected():
    temp_dir, client, server = _setup()
    try:
        client.post("/projects", json={"name": "DupProject"})
        resp = client.post("/projects", json={"name": "DupProject"})
        assert resp.status_code == 409
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()
