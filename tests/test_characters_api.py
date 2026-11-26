"""Tests for the Characters API: create, update, delete, and reference pictures."""

import asyncio
import gc
import glob
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from pixlstash.db_models import Character, Face, Picture, PictureSetMember
from pixlstash.event_types import EventType
from pixlstash.routes.characters import (
    _face_thumbnail_crop_box,
    create_router as create_characters_router,
)
from pixlstash.tasks.face_extraction_task import CROP_EXPAND_SCALE
from pixlstash.server import Server
from tests.utils import upload_pictures_and_wait


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


def test_create_character():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/characters", json={"name": "Alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        char = data["character"]
        assert char["name"] == "Alice"
        assert char["id"] is not None
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_get_character_by_id():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/characters", json={"name": "Bob"})
        char_id = resp.json()["character"]["id"]

        resp = client.get(f"/characters/{char_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == char_id
        assert resp.json()["name"] == "Bob"
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_patch_character_name_and_description():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/characters", json={"name": "Charlie"})
        char_id = resp.json()["character"]["id"]

        resp = client.patch(
            f"/characters/{char_id}",
            json={"name": "Charles", "description": "Updated description"},
        )
        assert resp.status_code == 200

        resp = client.get(f"/characters/{char_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Charles"
        assert resp.json()["description"] == "Updated description"

        resp = client.get(f"/characters/{char_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Charles"
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _patch_character_endpoint(server):
    [endpoint] = [
        route.endpoint
        for route in create_characters_router(server).routes
        if route.path == "/characters/{id}" and "PATCH" in route.methods
    ]
    return endpoint


async def _cancel_patch_mid_write(server, char_id, body, entered, release):
    """Start the PATCH, cancel it while its write is held in the worker
    thread, then let the write finish. The thread commits regardless; the
    cancellation surfaces at the route's ``await``."""

    async def read_body():
        return body

    request = SimpleNamespace(
        state=SimpleNamespace(origin_client_id="test-origin"), json=read_body
    )
    task = asyncio.create_task(
        _patch_character_endpoint(server)(id=char_id, request=request)
    )
    deadline = time.monotonic() + 30
    try:
        while not entered.is_set():
            if task.done():
                task.result()
                raise AssertionError("the PATCH finished without reaching its write")
            assert time.monotonic() < deadline, "the PATCH never reached its write"
            await asyncio.sleep(0.01)
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_a_cancelled_patch_still_announces_the_write_that_landed():
    """A client that goes away mid-PATCH cancels the route at its ``await``,
    but the worker thread still commits. Every listener (the grid, the
    sidebar, other tabs) must hear about the write that landed."""
    temp_dir, client, server = _setup()
    db = server.vault.db
    try:
        char_id = client.post("/characters", json={"name": "test-before"}).json()[
            "character"
        ]["id"]
        events = []
        heard = threading.Event()

        def listener(event_type, data):
            if event_type == EventType.CHANGED_CHARACTERS:
                events.append(data)
                heard.set()

        server.vault.add_event_listener(listener)
        entered = threading.Event()
        release = threading.Event()
        real_run_task = db.run_task

        def gated_run_task(func, *args, **kwargs):
            if getattr(func, "__name__", "") == "alter_char":
                entered.set()
                assert release.wait(30), "the test never released the write"
            return real_run_task(func, *args, **kwargs)

        db.run_task = gated_run_task
        asyncio.run(
            _cancel_patch_mid_write(
                server, char_id, {"name": "test-after"}, entered, release
            )
        )

        assert heard.wait(30), "the committed rename was never announced"
        assert events == [{"origin_client_id": "test-origin"}]
        del db.run_task
        assert client.get(f"/characters/{char_id}").json()["name"] == "test-after"
    finally:
        db.__dict__.pop("run_task", None)
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_delete_character_returns_success():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/characters", json={"name": "DeleteMe"})
        char_id = resp.json()["character"]["id"]

        resp = client.delete(f"/characters/{char_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted_id"] == char_id

        resp = client.get(f"/characters/{char_id}")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert resp.json() is None
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_character_reference_pictures_empty_without_faces():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/characters", json={"name": "NoFaces"})
        char_id = resp.json()["character"]["id"]

        resp = client.get(f"/characters/{char_id}/reference_pictures")
        assert resp.status_code == 200
        assert resp.json()["reference_picture_ids"] == []
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_delete_nonexistent_character_returns_404():
    temp_dir, client, server = _setup()
    try:
        resp = client.delete("/characters/99999")
        assert resp.status_code == 404
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_get_characters_filtered_by_numeric_project_id():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "CharFilter Project"})
        assert resp.status_code == 200
        project_id = resp.json()["id"]

        resp = client.post(
            "/characters", json={"name": "InProject", "project_id": project_id}
        )
        assert resp.status_code == 200

        resp = client.post("/characters", json={"name": "NotInProject"})
        assert resp.status_code == 200

        resp = client.get(f"/characters?project_id={project_id}")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert "InProject" in names
        assert "NotInProject" not in names
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_get_characters_filtered_by_unassigned():
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "UnassignedFilter Project"})
        assert resp.status_code == 200
        project_id = resp.json()["id"]

        resp = client.post(
            "/characters", json={"name": "Assigned", "project_id": project_id}
        )
        assert resp.status_code == 200

        resp = client.post("/characters", json={"name": "Unassigned"})
        assert resp.status_code == 200

        resp = client.get("/characters?project_id=UNASSIGNED")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert "Unassigned" in names
        assert "Assigned" not in names
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_get_characters_invalid_project_id_returns_400():
    temp_dir, client, server = _setup()
    try:
        resp = client.get("/characters?project_id=not-a-number")
        assert resp.status_code == 400
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_a_failed_character_listing_answers_a_fixed_sentence_and_logs_the_cause(
    caplog,
):
    """A fault in ``GET /characters`` is a 500 whose body is a sentence the
    server wrote, never the exception's text; the exception itself goes to the
    log, with its traceback and the request's filters."""
    leak = "test-leak-detail-from-the-database"

    def failing_read(*_args, **_kwargs):
        raise RuntimeError(leak)

    server = SimpleNamespace(
        vault=SimpleNamespace(db=SimpleNamespace(run_immediate_read_task=failing_read))
    )
    app = FastAPI()
    # The suite's TestClient puts every path under /api/v1 (conftest), as the
    # server mounts its routers.
    app.include_router(create_characters_router(server), prefix="/api/v1")
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="pixlstash.routes.characters"):
        resp = client.get("/characters", params={"project_id": "UNASSIGNED"})

    assert resp.status_code == 500, resp.text
    assert resp.json() == {"detail": "Internal Server Error"}
    assert leak not in resp.text
    [record] = [
        r for r in caplog.records if r.getMessage().startswith("Listing characters")
    ]
    assert record.levelno == logging.ERROR
    # ``exc_info=False`` leaves ``False`` here, ``None`` when omitted.
    assert isinstance(record.exc_info, tuple), "logged without its traceback"
    assert isinstance(record.exc_info[1], RuntimeError)
    assert str(record.exc_info[1]) == leak
    assert "project_id='UNASSIGNED'" in record.getMessage()


def test_character_scoped_token_may_not_filter_by_project():
    """Issue #708 F3 - this test previously pinned the vulnerability.

    It used to assert that a character-scoped token filtering
    ``/characters?project_id=<id>`` got its character back, and that the same
    request with ``UNASSIGNED`` did not. Comparing those two answers *is* the
    disclosure: it tells a token that is 403'd on ``GET /projects/{id}`` that the
    project exists and that its character is filed under it. Both requests are
    now refused by the authz gate (``enforce_project_filter_scope``), which is
    the behaviour this test pins instead. The unfiltered listing, which is what
    the share UI actually issues, must keep working - over-blocking would be its
    own regression.
    """
    temp_dir, client, server = _setup()
    try:
        resp = client.post("/projects", json={"name": "ScopedTokenProject"})
        assert resp.status_code == 200
        project_id = resp.json()["id"]

        resp = client.post(
            "/characters", json={"name": "CharInProject", "project_id": project_id}
        )
        assert resp.status_code == 200
        char_id = resp.json()["character"]["id"]

        resp = client.post(
            "/users/me/token",
            json={
                "description": "char token",
                "scope": "READ",
                "resource_type": "character",
                "resource_id": char_id,
            },
        )
        assert resp.status_code == 200
        char_token = resp.json()["token"]

        token_client = TestClient(server.api)
        headers = {"Authorization": f"Bearer {char_token}"}

        # Naming the project it is filed under, the UNASSIGNED sentinel, and a
        # project id that does not exist all get the same 403 - so the refusal
        # itself answers nothing.
        for probe in (str(project_id), "UNASSIGNED", "99999999"):
            resp = token_client.get(f"/characters?project_id={probe}", headers=headers)
            assert resp.status_code == 403, (
                f"project_id={probe} must be refused for a character token; got "
                f"{resp.status_code} {resp.text}"
            )

        # In-scope direction: without the filter the token still reads its own
        # character, and learns no project ids from the payload.
        resp = token_client.get("/characters", headers=headers)
        assert resp.status_code == 200, resp.text
        listed = {c["id"]: c for c in resp.json()}
        assert char_id in listed
        assert listed[char_id]["project_ids"] == []
        assert listed[char_id]["project_id"] is None

        # The owner is never restricted by any of it.
        resp = client.get(f"/characters?project_id={project_id}")
        assert resp.status_code == 200, resp.text
        assert char_id in [c["id"] for c in resp.json()]
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _import_one_picture(client):
    candidates = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "pictures", "*.png")
    ) + glob.glob(os.path.join(os.path.dirname(__file__), "..", "pictures", "*.jpg"))
    assert candidates, "No test images found in pictures/ directory"
    path = candidates[0]
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as image_file:
        import_resp = upload_pictures_and_wait(
            client, [("file", (os.path.basename(path), image_file, mime))]
        )
    return import_resp["results"][0]["picture_id"]


def _link_face(server, pic_id, char_id, face_index=0):
    def _add(session):
        session.add(
            Face(
                picture_id=pic_id,
                frame_index=0,
                face_index=face_index,
                character_id=char_id,
                bbox=[0, 0, 10, 10],
            )
        )
        session.commit()

    server.vault.db.run_task(_add)


def _project_picture_ids(client, project_id):
    resp = client.get("/pictures", params={"project_id": str(project_id)})
    assert resp.status_code == 200
    return {row.get("id") for row in resp.json()}


def test_moving_character_to_new_project_disassociates_pictures_from_old():
    """Moving a character out of project A removes the character's pictures from
    A (the project it was dragged out of) and into the new project B."""
    temp_dir, client, server = _setup()
    try:
        project_a = client.post("/projects", json={"name": "Char Move A"}).json()["id"]
        project_b = client.post("/projects", json={"name": "Char Move B"}).json()["id"]

        pic_id = _import_one_picture(client)
        char_id = client.post("/characters", json={"name": "Mover"}).json()[
            "character"
        ]["id"]
        _link_face(server, pic_id, char_id)

        # Assigning the character to A cascades the picture into A.
        assert (
            client.patch(
                f"/characters/{char_id}", json={"project_id": project_a}
            ).status_code
            == 200
        )
        assert pic_id in _project_picture_ids(client, project_a)

        # Moving the character to B should pull the picture out of A.
        assert (
            client.patch(
                f"/characters/{char_id}", json={"project_id": project_b}
            ).status_code
            == 200
        )
        assert pic_id in _project_picture_ids(client, project_b)
        assert pic_id not in _project_picture_ids(client, project_a)
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_moving_character_keeps_pictures_shared_with_another_character_in_old_project():
    """A picture also containing a second character still in project A is kept in
    A when the first character is moved out - only orphaned pictures leave."""
    temp_dir, client, server = _setup()
    try:
        project_a = client.post("/projects", json={"name": "Shared Char A"}).json()[
            "id"
        ]
        project_b = client.post("/projects", json={"name": "Shared Char B"}).json()[
            "id"
        ]

        pic_id = _import_one_picture(client)
        char_one = client.post("/characters", json={"name": "First"}).json()[
            "character"
        ]["id"]
        char_two = client.post("/characters", json={"name": "Second"}).json()[
            "character"
        ]["id"]
        _link_face(server, pic_id, char_one, face_index=0)
        _link_face(server, pic_id, char_two, face_index=1)

        for char_id in (char_one, char_two):
            assert (
                client.patch(
                    f"/characters/{char_id}", json={"project_id": project_a}
                ).status_code
                == 200
            )
        assert pic_id in _project_picture_ids(client, project_a)

        # Move only the first character to B.
        assert (
            client.patch(
                f"/characters/{char_one}", json={"project_id": project_b}
            ).status_code
            == 200
        )

        # Added to B, but retained in A because the second character anchors it.
        assert pic_id in _project_picture_ids(client, project_b)
        assert pic_id in _project_picture_ids(client, project_a)
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_batch_character_membership_returns_assignments():
    """POST /characters/membership must return a 200 with the picture's character
    assignment. Regression: the handler built character_assignments with integer
    keys while CharacterMembershipResponse declares dict[str, list[int]]; pydantic
    v2 rejected the int keys, the response 500'd, and the AddToCharacter menu
    received no membership data (every character shown unchecked)."""
    temp_dir, client, server = _setup()
    try:
        pic_id = _import_one_picture(client)
        char_id = client.post("/characters", json={"name": "Member"}).json()[
            "character"
        ]["id"]
        _link_face(server, pic_id, char_id)

        resp = client.post("/characters/membership", json={"picture_ids": [pic_id]})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["character_assignments"] == {str(char_id): [pic_id]}
        assert data["pictures_with_faces"] == [pic_id]
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _import_two_pictures(client):
    """Import two DISTINCT images and return their picture ids."""
    candidates = sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..", "pictures", "*.png"))
        + glob.glob(os.path.join(os.path.dirname(__file__), "..", "pictures", "*.jpg"))
    )
    assert len(candidates) >= 2, "Need two test images in pictures/"
    ids = []
    for path in candidates[:2]:
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as image_file:
            import_resp = upload_pictures_and_wait(
                client, [("file", (os.path.basename(path), image_file, mime))]
            )
        ids.append(import_resp["results"][0]["picture_id"])
    return ids


def _link_face_with_bbox(server, pic_id, char_id):
    """Assign a face with a REAL bbox, which the thumbnail crop needs.

    ``_link_face`` above writes a 10 px box at index 0, which is enough for the
    assignment tests that use it and too small for a render.

    ``face_index`` sits above anything detection produces: import runs the face
    detector, and index 0 is whichever face it found, so writing there races the
    import for the (picture, frame, index) unique constraint.
    """

    def _add(session):
        session.add(
            Face(
                picture_id=pic_id,
                frame_index=0,
                face_index=99,
                character_id=char_id,
                bbox=[0, 0, 32, 32],
            )
        )
        session.commit()

    server.vault.db.run_task(_add)


def _thumbnail_bytes(client, char_id):
    """The rendered thumbnail itself.

    The BYTES, not the render's cache metadata: that metadata records the
    selection query's winner, which is not always the picture the route ends up
    cropping, so asserting on it passes with the whole pin implementation
    deleted. The image is the only thing the user sees and the only thing worth
    asserting on.
    """
    resp = client.get(f"/characters/{char_id}/thumbnail")
    assert resp.status_code == 200, resp.text
    return hashlib.sha256(resp.content).hexdigest()


def test_thumbnail_crop_undoes_the_stored_face_box_padding():
    """The avatar crops the detector's face, not the padded box faces are stored
    with, so the face fills the thumbnail instead of sitting small in it."""
    # 256 * 1.25 = 320 is already a multiple of 64: the crop is the raw box.
    raw = [400, 300, 656, 556]
    stored = Face.expand_face_bbox(raw, 1000, 1000, CROP_EXPAND_SCALE - 1)
    assert stored == [368, 268, 688, 588]
    assert _face_thumbnail_crop_box(stored, 1000, 1000) == tuple(raw)

    # 128 * 1.25 = 160 rounds up to 192: tighter than stored, the face still whole.
    raw = [200, 200, 328, 328]
    stored = Face.expand_face_bbox(raw, 1000, 1000, CROP_EXPAND_SCALE - 1)
    x1, y1, x2, y2 = _face_thumbnail_crop_box(stored, 1000, 1000)
    assert x2 - x1 < stored[2] - stored[0]
    assert x1 <= raw[0] and y1 <= raw[1] and x2 >= raw[2] and y2 >= raw[3]

    # Clamped at the image edge, the stored box lost its centre: shrinking it
    # would cut into the face, so it is cropped as stored.
    for raw in ([0, 0, 100, 100], [900, 900, 1000, 1000]):
        stored = Face.expand_face_bbox(raw, 1000, 1000, CROP_EXPAND_SCALE - 1)
        x1, y1, x2, y2 = _face_thumbnail_crop_box(stored, 1000, 1000)
        assert x1 <= raw[0] and y1 <= raw[1] and x2 >= raw[2] and y2 >= raw[3]

    # Degenerate box: nothing to crop.
    assert _face_thumbnail_crop_box([10, 10, 10, 40], 100, 100) is None


def test_pinned_thumbnail_picture_overrides_the_automatic_choice():
    """A pinned reference picture decides the crop, and clearing it restores the
    automatic pick. Both directions, because a pin that cannot be undone is its
    own defect."""
    temp_dir, client, server = _setup()
    try:
        first_pic, second_pic = _import_two_pictures(client)
        char_id = client.post("/characters", json={"name": "Pinned"}).json()[
            "character"
        ]["id"]
        _link_face_with_bbox(server, first_pic, char_id)
        _link_face_with_bbox(server, second_pic, char_id)

        # Score them apart rather than relying on the id tie-break: a scoring
        # sweep landing mid-test would otherwise decide which picture the
        # automatic path picks, and the test would flake instead of failing.
        automatic, pinned = max(first_pic, second_pic), min(first_pic, second_pic)
        resp = client.post(
            "/pictures/apply-scores",
            json={"scores": {str(automatic): 5, str(pinned): 1}},
        )
        assert resp.status_code == 200
        automatic_thumbnail = _thumbnail_bytes(client, char_id)

        # Pin each of the two in turn and compare the RENDERED bytes. The two
        # crops come from different source images, so equal bytes mean the pin
        # did not decide the render.
        #
        # This order is deliberate: `automatic` is the picture the selection
        # query names (and therefore what the cache metadata already held), so
        # pinning it first is the case where a cache keyed on that id alone
        # would hit and keep serving the previous crop. Under that bug both
        # pins render the same picture and the assertion below fails.
        assert (
            client.patch(
                f"/characters/{char_id}", json={"thumbnail_picture_id": automatic}
            ).json()["character"]["thumbnail_picture_id"]
            == automatic
        )
        pinned_to_automatic = _thumbnail_bytes(client, char_id)

        resp = client.patch(
            f"/characters/{char_id}", json={"thumbnail_picture_id": pinned}
        )
        assert resp.status_code == 200
        assert resp.json()["character"]["thumbnail_picture_id"] == pinned
        pinned_thumbnail = _thumbnail_bytes(client, char_id)
        assert pinned_thumbnail != pinned_to_automatic

        # And the pin is stable: going back to the first one renders the first
        # one again rather than whatever was cached last.
        assert (
            client.patch(
                f"/characters/{char_id}", json={"thumbnail_picture_id": automatic}
            ).status_code
            == 200
        )
        assert _thumbnail_bytes(client, char_id) == pinned_to_automatic

        assert (
            client.patch(
                f"/characters/{char_id}", json={"thumbnail_picture_id": pinned}
            ).status_code
            == 200
        )

        # The list endpoint carries it too - that is where the editor reads the
        # current pin from when it opens.
        listed = [c for c in client.get("/characters").json() if c["id"] == char_id]
        assert listed[0]["thumbnail_picture_id"] == pinned

        # null clears it, and an untouched PATCH leaves it alone.
        assert (
            client.patch(f"/characters/{char_id}", json={"name": "Pinned"}).status_code
            == 200
        )
        assert client.get(f"/characters/{char_id}").json()["thumbnail_picture_id"] == (
            pinned
        )
        resp = client.patch(
            f"/characters/{char_id}", json={"thumbnail_picture_id": None}
        )
        assert resp.status_code == 200
        assert _thumbnail_bytes(client, char_id) == automatic_thumbnail
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_pinning_a_thumbnail_does_not_invalidate_derived_picture_fields():
    """The pin is not identity data.

    Renaming a person nulls the description and text embedding of every picture
    they appear in, because their name is baked into both. Choosing which
    existing crop to show is not that, and letting it share the flag deleted
    hand-written descriptions and queued a library-wide re-derive on one click.
    """
    temp_dir, client, server = _setup()
    try:
        pic_id = _import_one_picture(client)
        char_id = client.post("/characters", json={"name": "Keeper"}).json()[
            "character"
        ]["id"]
        _link_face_with_bbox(server, pic_id, char_id)

        def _seed(session):
            picture = session.get(Picture, pic_id)
            picture.description = "a description somebody typed"
            picture.text_embedding = b"embedding-bytes"
            session.add(picture)
            session.commit()

        server.vault.db.run_task(_seed)

        assert (
            client.patch(
                f"/characters/{char_id}", json={"thumbnail_picture_id": pic_id}
            ).status_code
            == 200
        )

        def _read(session):
            picture = session.get(Picture, pic_id)
            return picture.description, picture.text_embedding is not None

        description, has_embedding = server.vault.db.run_immediate_read_task(_read)
        assert description == "a description somebody typed"
        assert has_embedding is True

        # The control: a RENAME still throws both away, so the assertion above
        # is about the pin and not about a wipe that stopped working.
        assert (
            client.patch(f"/characters/{char_id}", json={"name": "Renamed"}).status_code
            == 200
        )
        description, has_embedding = server.vault.db.run_immediate_read_task(_read)
        assert description is None
        assert has_embedding is False
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_pinning_a_picture_without_this_persons_face_is_rejected():
    temp_dir, client, server = _setup()
    try:
        first_pic, second_pic = _import_two_pictures(client)
        char_id = client.post("/characters", json={"name": "NoFace"}).json()[
            "character"
        ]["id"]
        _link_face_with_bbox(server, first_pic, char_id)

        resp = client.patch(
            f"/characters/{char_id}", json={"thumbnail_picture_id": second_pic}
        )
        assert resp.status_code == 400
        resp = client.patch(
            f"/characters/{char_id}", json={"thumbnail_picture_id": "not-an-id"}
        )
        assert resp.status_code == 400

        # A scrapheaped picture keeps its faces, so it passes a face-only check
        # - and the renderer would still skip it, leaving a pin nothing can
        # honour. Refused for the same reason and with the same status.
        resp = client.delete(f"/pictures/{first_pic}")
        assert resp.status_code == 200
        resp = client.patch(
            f"/characters/{char_id}", json={"thumbnail_picture_id": first_pic}
        )
        assert resp.status_code == 400
        # And nothing was written: the person still has no pin.
        assert (
            client.get(f"/characters/{char_id}").json()["thumbnail_picture_id"] is None
        )
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


# ── the character routes read no face embedding ────────────────────────────
#
# A face's ``features`` column is its 2 KiB ArcFace embedding. The thumbnail
# render, the faces listing and a character edit need a face's box, frame or
# picture id, never the embedding; loading whole rows read it for every face.


@contextmanager
def _request_statements(server):
    """The SQL the request handlers run while the block is open.

    Statements on the threads that serve the TestClient's sync routes, and the
    ones on the database's writer thread that select faces by character: a
    route's ``run_task`` work (the character edit's) runs there, beside the
    background workers', and only the route filters faces by character. The
    workers keep reading faces of their own while the test runs, which is why
    every other statement is left out.
    """
    statements = []
    writer = server.vault.db._task_worker

    def record(conn, cursor, statement, parameters, context, executemany):
        thread = threading.current_thread()
        if thread.name == "AnyIO worker thread" or (
            thread is writer and "face.character_id = " in statement
        ):
            statements.append(statement)

    engine = server.vault.db._engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def _add_face(server, pic_id, char_id, face_index):
    def _add(session):
        session.add(
            Face(
                picture_id=pic_id,
                frame_index=0,
                face_index=face_index,
                character_id=char_id,
                bbox=[0, 0, 32, 32],
            )
        )
        session.commit()

    server.vault.db.run_task(_add)


def _assert_no_embedding_read(statements, what):
    face_reads = [s for s in statements if "FROM face" in s]
    assert face_reads, f"{what}: the request read no face at all"
    embedding_reads = [s for s in face_reads if "face.features" in s]
    assert not embedding_reads, f"{what} read face embeddings: {embedding_reads}"


def test_character_routes_read_no_face_embedding():
    """Each way the thumbnail finds its face (the pin, the reference set, the
    character's other faces), the faces listing and an edit of the character
    read the face rows without their embedding, and the thumbnail still
    renders."""
    temp_dir, client, server = _setup()
    try:
        first_pic, second_pic = _import_two_pictures(client)
        ids = {}
        for index, name in enumerate(("Pinned", "Referenced", "Fallback")):
            ids[name] = client.post("/characters", json={"name": name}).json()[
                "character"
            ]["id"]
            _add_face(server, first_pic, ids[name], 90 + index)
        _add_face(server, second_pic, ids["Referenced"], 95)

        def add_reference(session):
            character = session.get(Character, ids["Referenced"])
            session.add(
                PictureSetMember(
                    set_id=character.reference_picture_set_id, picture_id=second_pic
                )
            )
            session.commit()

        server.vault.db.run_task(add_reference)
        resp = client.patch(
            f"/characters/{ids['Pinned']}", json={"thumbnail_picture_id": first_pic}
        )
        assert resp.status_code == 200, resp.text

        for name, char_id in ids.items():
            with _request_statements(server) as statements:
                _thumbnail_bytes(client, char_id)
            _assert_no_embedding_read(statements, f"the {name} thumbnail")

        with _request_statements(server) as statements:
            resp = client.get(f"/characters/{ids['Fallback']}/faces")
        assert resp.status_code == 200
        assert [face["picture_id"] for face in resp.json()["faces"]] == [first_pic]
        _assert_no_embedding_read(statements, "the faces listing")

        project_id = client.post("/projects", json={"name": "Faces Move"}).json()["id"]
        with _request_statements(server) as statements:
            resp = client.patch(
                f"/characters/{ids['Fallback']}",
                json={"name": "Renamed", "project_id": project_id},
            )
        assert resp.status_code == 200, resp.text
        assert first_pic in _project_picture_ids(client, project_id)
        _assert_no_embedding_read(statements, "the character edit")
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()
