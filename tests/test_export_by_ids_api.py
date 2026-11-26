"""POST /pictures/export: the ZIP export with its selection in a JSON body.

The GET route names a selection as one ``?id=`` per picture, and a large one
makes a URL a reverse proxy refuses (414). The POST takes the same selection as
``picture_ids`` and runs the same ``generate_zip``, so these assert what that
sameness must keep: only the named pictures are exported, a share token's
export keeps only its in-scope ids (both directions), a READ token is let
through as it is for the GET, and the body is bounded and typed.
"""

import io
import json
import os
import sqlite3
import tempfile
import time
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlmodel import select

import pixlstash.routes.pictures as pictures_routes
from pixlstash.auth import COMFYUI_SCOPE_REFUSAL
from pixlstash.db_models.detection import Detection
from pixlstash.db_models.face import Face
from pixlstash.db_models.picture import Picture
from pixlstash.routes.pictures._export import EXPORT_MAX_IDS, ExportByIdsRequest
from pixlstash.server import Server
from pixlstash.utils.service.export_utils import ExportUtils
from tests.authz_guard import assert_real_route
from tests.utils import (
    API_PREFIX,
    bearer_client,
    mint_read_token,
    upload_pictures_and_wait,
)

EXPORT = f"{API_PREFIX}/pictures/export"
# Pinned on the vault's connections by ``_sqlite_variable_ceiling``; a selection
# of more ids than this fails in one IN with "too many SQL variables".
VARIABLE_CEILING = 999
# Ids no picture has: they pad a selection past the ceiling.
ABSENT_IDS = list(range(10**9, 10**9 + VARIABLE_CEILING + 200))


def _picture_files():
    """Four small, distinct generated pictures (the importer keeps all four)."""
    files = []
    for index in range(4):
        image = Image.new(
            "RGB", (32 + index * 8, 32 + index * 4), color=(30 + index * 60, 70, 180)
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        files.append(("file", (f"export_{index}.png", buf.getvalue(), "image/png")))
    return files


@pytest.fixture(scope="module")
def env():
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000}))
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
        # The last one is in the Scrapheap; ``ids`` are the three live ones.
        ids, deleted_id = uploaded[:3], uploaded[3]

        def soft_delete(session):
            pic = session.get(Picture, deleted_id)
            pic.deleted = True
            session.add(pic)
            session.commit()

        server.vault.db.run_task(soft_delete)
        yield server, owner, ids
    finally:
        server.close()
        temp_dir.cleanup()


@pytest.fixture
def scrapheap_id(env):
    """The fixture's one soft-deleted picture."""
    server, _owner, _ids = env

    def read(session):
        return session.exec(select(Picture.id).where(Picture.deleted.is_(True))).one()

    return server.vault.db.run_task(read)


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


def _exported_pictures(client, body) -> int:
    """Start an export, wait for it, and count the pictures in its ZIP."""
    r = client.post(EXPORT, json={"caption_mode": "none", **body})
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    status = _finish(client, task_id)
    assert status["status"] == "completed", status
    r = client.get(f"{EXPORT}/download/{task_id}")
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        return sum(1 for name in archive.namelist() if name.endswith(".png"))


def _names_in_order(client, method, selection, **options) -> list[str]:
    """Start an export by *method*, wait for it, and list its pictures in order.

    The pictures are named by their original file names, so the list says which
    pictures the ZIP holds and, in archive order, the order it packaged them.
    """
    options = {"caption_mode": "none", "use_original_file_names": True, **options}
    if method == "POST":
        r = client.post(EXPORT, json={"picture_ids": selection, **options})
    else:
        params = [("id", str(pid)) for pid in selection]
        params += [(k, str(v).lower()) for k, v in options.items()]
        r = client.get(EXPORT, params=params)
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    status = _finish(client, task_id)
    assert status["status"] == "completed", status
    r = client.get(f"{EXPORT}/download/{task_id}")
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        return [name for name in archive.namelist() if name.endswith(".png")]


def _file_names(server, picture_ids) -> list[str]:
    def read(session):
        rows = session.exec(
            select(Picture.id, Picture.original_file_name).where(
                Picture.id.in_(picture_ids)
            )
        ).all()
        return {pid: name for pid, name in rows}

    by_id = server.vault.db.run_task(read)
    return [by_id[pid] for pid in picture_ids]


@contextmanager
def _sqlite_variable_ceiling(server, limit: int = VARIABLE_CEILING):
    """Pin every new vault connection to SQLite's pre-3.32 bound-parameter cap.

    Current builds allow 32,766 or more parameters per statement, and a
    selection that large takes long to build; at the historical 999 the same
    plain IN fails with a selection of a thousand ids, with the identical
    "too many SQL variables". The pin is proved to have taken effect before the
    block runs, so a build that ignored ``setlimit`` could not pass vacuously.
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
            server.vault.db.run_task(over_the_ceiling)
        yield
    finally:
        event.remove(engine, "connect", pin)
        engine.dispose()


def _share_client(server, owner, picture_id):
    token = mint_read_token(
        owner,
        "test-export-share-token",
        resource_type="picture",
        resource_id=picture_id,
    )
    return bearer_client(server, token)


def test_the_route_is_mounted(env):
    server, _owner, _ids = env
    assert_real_route(server.api, "POST", EXPORT, template=EXPORT)


def test_the_owner_exports_exactly_the_named_pictures(env):
    _server, owner, ids = env
    assert _exported_pictures(owner, {"picture_ids": ids[:2]}) == 2
    assert _exported_pictures(owner, {"picture_ids": [ids[2]]}) == 1


def test_a_share_token_exports_only_its_in_scope_pictures(env):
    server, owner, ids = env
    share = _share_client(server, owner, ids[0])

    # Positive control: the in-scope picture is exported, the others dropped.
    assert _exported_pictures(share, {"picture_ids": ids}) == 1

    # Negative: naming only out-of-scope pictures exports nothing at all.
    r = share.post(EXPORT, json={"picture_ids": ids[1:]})
    assert r.status_code == 200, r.text
    assert _finish(share, r.json()["task_id"])["status"] == "failed"


def test_a_read_token_may_start_the_export(env):
    server, owner, ids = env
    reader = bearer_client(server, mint_read_token(owner, "test-export-read-token"))
    r = reader.post(EXPORT, json={"picture_ids": [ids[0]]})
    assert r.status_code == 200, r.text
    assert _finish(reader, r.json()["task_id"])["status"] == "completed"


def test_a_comfyui_token_is_refused_the_export(env):
    server, owner, ids = env
    assert_real_route(server.api, "POST", EXPORT, template=EXPORT)
    r = owner.post(
        f"{API_PREFIX}/users/me/token",
        json={"description": "test-export-comfyui-token", "scope": "COMFYUI"},
    )
    assert r.status_code == 200, r.text
    token, token_id = r.json()["token"], r.json()["token_id"]
    try:
        before = set(server.export_tasks)
        r = bearer_client(server, token).post(EXPORT, json={"picture_ids": [ids[0]]})
        assert r.status_code == 403, r.text
        assert r.json() == {"detail": COMFYUI_SCOPE_REFUSAL}
        assert set(server.export_tasks) == before
        # Positive control: the owner starts the same export.
        r = owner.post(EXPORT, json={"picture_ids": [ids[0]]})
        assert r.status_code == 200, r.text
        assert _finish(owner, r.json()["task_id"])["status"] == "completed"
    finally:
        owner.delete(f"{API_PREFIX}/users/me/token/{token_id}")


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"picture_ids": []}, id="empty"),
        pytest.param({"picture_ids": [1] * (EXPORT_MAX_IDS + 1)}, id="over-the-cap"),
        pytest.param({"picture_ids": [2**63]}, id="not-a-sqlite-int"),
        pytest.param({"picture_ids": [1], "query": "cat"}, id="unknown-field"),
        pytest.param({"picture_ids": [1], "resolution": "huge"}, id="bad-option"),
    ],
)
def test_a_malformed_body_is_refused_before_an_export_starts(env, body):
    server, owner, _ids = env
    before = set(server.export_tasks)
    r = owner.post(EXPORT, json=body)
    assert r.status_code == 422, r.text
    assert set(server.export_tasks) == before


def test_the_cap_itself_is_accepted():
    """The bound is inclusive: a body of exactly EXPORT_MAX_IDS ids validates.

    Checked on the model, not through an export of 200,000 pictures: the
    export's read of a selection past SQLite's variable limit is
    covered by the ceiling tests above.
    """
    body = ExportByIdsRequest(picture_ids=[1] * EXPORT_MAX_IDS)
    assert len(body.picture_ids) == EXPORT_MAX_IDS


@pytest.mark.parametrize("method", ["POST", "GET"])
def test_a_selection_past_the_sqlite_variable_limit_exports_every_picture(env, method):
    server, owner, ids = env
    with _sqlite_variable_ceiling(server):
        names = _names_in_order(owner, method, ABSENT_IDS + ids)
    assert sorted(names) == sorted(_file_names(server, ids))


@pytest.mark.parametrize("method", ["POST", "GET"])
def test_a_selection_is_exported_in_the_order_it_names(env, method):
    server, owner, ids = env
    order = [ids[2], ids[0], ids[1]]
    assert _names_in_order(owner, method, order) == _file_names(server, order)


@pytest.mark.parametrize("only_deleted", [True, False])
def test_the_post_and_the_get_pick_the_same_pictures_for_one_selection(
    env, scrapheap_id, only_deleted
):
    """``only_deleted`` means what it means to the listing: the Scrapheap only.

    A Scrapheap export packages the soft-deleted pictures it names and a live
    one never, as the Scrapheap view never shows one; any other export the
    reverse.
    """
    server, owner, ids = env
    selection = [ids[1], scrapheap_id, ids[0]]
    by_post = _names_in_order(owner, "POST", selection, only_deleted=only_deleted)
    by_get = _names_in_order(owner, "GET", selection, only_deleted=only_deleted)
    assert by_post == by_get
    expected = [scrapheap_id] if only_deleted else [ids[1], ids[0]]
    assert by_post == _file_names(server, expected)


def test_a_filter_export_past_the_sqlite_variable_limit_exports_every_picture(
    env, monkeypatch
):
    """The listing branch reads the view's ids past the ceiling, in its order.

    A real view past the ceiling takes a thousand pictures, so the listing's
    answer is stood in for: the ids it hands the export, with the three real
    pictures among a ceiling's worth that match nothing.
    """
    server, owner, ids = env
    view = [ids[1]] + ABSENT_IDS + [ids[2], ids[0]]
    monkeypatch.setattr(
        pictures_routes, "select_pictures_for_listing", lambda **_kwargs: view
    )
    with _sqlite_variable_ceiling(server):
        r = owner.get(
            EXPORT, params={"caption_mode": "none", "use_original_file_names": True}
        )
        assert r.status_code == 200, r.text
        status = _finish(owner, r.json()["task_id"])
    assert status["status"] == "completed", status
    r = owner.get(f"{EXPORT}/download/{r.json()['task_id']}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".png")]
    assert names == _file_names(server, [ids[1], ids[2], ids[0]])


@pytest.mark.parametrize("bad", ["not-a-number", str(2**63)])
def test_a_get_export_refuses_an_id_sqlite_cannot_store(env, bad):
    server, owner, _ids = env
    before = set(server.export_tasks)
    r = owner.get(EXPORT, params={"id": bad})
    assert r.status_code == 422, r.text
    assert set(server.export_tasks) == before


def test_stack_deduplication_keeps_the_order_it_is_given():
    """A stack's cover takes the place of the stack's first picture.

    The export packages pictures in the view's order and numbers them by it,
    so the deduplication may drop pictures but never move one.
    """

    class P:
        def __init__(self, id, stack_id=None, stack_position=None):
            self.id = id
            self.stack_id = stack_id
            self.stack_position = stack_position
            self.created_at = None

    pics = [
        P(1),
        P(2, stack_id=7, stack_position=1),
        P(3),
        P(4, stack_id=7, stack_position=0),
        P(5),
    ]
    assert [p.id for p in ExportUtils._deduplicate_stacks(pics)] == [1, 4, 3, 5]


@pytest.fixture
def named_face_and_detection(env):
    """One picture with a detection and a face named as a second character.

    Yields ``(picture_id, face_id, detection_id, character_id)``, where
    *character_id* is the character a training run would be for, so the face
    counts as "somebody else". Removed afterwards: the module is shared.
    """
    server, owner, ids = env
    character_ids = []
    for name in ("test-export-trained", "test-export-other"):
        r = owner.post(f"{API_PREFIX}/characters", json={"name": name})
        assert r.status_code == 200, r.text
        character_ids.append(r.json()["character"]["id"])
    trained, other = character_ids

    def add(session):
        face = Face(picture_id=ids[0], face_index=900, character_id=other)
        detection = Detection(picture_id=ids[0], label="test-label")
        session.add(face)
        session.add(detection)
        session.commit()
        return face.id, detection.id

    face_id, detection_id = server.vault.db.run_task(add)
    try:
        yield ids[0], face_id, detection_id, trained
    finally:

        def remove(session):
            session.delete(session.get(Face, face_id))
            session.delete(session.get(Detection, detection_id))
            session.commit()

        server.vault.db.run_task(remove)
        for character_id in character_ids:
            r = owner.delete(f"{API_PREFIX}/characters/{character_id}")
            assert r.status_code == 200, r.text


def test_an_export_reads_faces_and_detections_for_any_number_of_pictures(
    env, named_face_and_detection
):
    """A face export with a bbox sidecar looks both up past the ceiling.

    The pictures stand in for an export's: a ceiling's worth that have no rows
    and the one that has both, so one plain IN would fail.
    """
    server, _owner, _ids = env
    picture_id, face_id, detection_id, _trained = named_face_and_detection
    pics = [
        SimpleNamespace(id=pid, file_path=None) for pid in ABSENT_IDS + [picture_id]
    ]
    with _sqlite_variable_ceiling(server):
        faces, _tags, detections, _total = ExportUtils._prepare_feature_maps_and_total(
            server,
            pics,
            {
                "export_type_d": Picture.ExportType.FACE,
                "bbox_mode_d": "coco-json",
            },
        )
    assert set(faces) == {picture_id}
    assert face_id in {face.id for face in faces[picture_id]}
    assert set(detections) == {picture_id}
    assert detection_id in {det.id for det in detections[picture_id]}


def test_training_facts_count_other_people_for_any_number_of_pictures(
    env, named_face_and_detection, monkeypatch
):
    """The Train dialog's "other people" count reads faces past the ceiling."""
    server, _owner, _ids = env
    picture_id, _face_id, _detection_id, trained = named_face_and_detection
    rows = [
        SimpleNamespace(id=pid, file_path="still.png", description="")
        for pid in ABSENT_IDS + [picture_id]
    ]
    monkeypatch.setattr(
        ExportUtils, "_training_rows", staticmethod(lambda *_args, **_kw: rows)
    )
    monkeypatch.setattr(
        ExportUtils,
        "_captions_and_tags",
        staticmethod(lambda pics, _exclusions: ([""] * len(pics), [[]] * len(pics))),
    )
    with _sqlite_variable_ceiling(server):
        facts = ExportUtils.training_facts(server, character_id=trained)
    assert facts["count"] == len(rows)
    assert facts["other_people"] == 1
