"""Batch face assignment by picture_ids: best-face selection heuristics.

Covers the bootstrap heuristic for a character with no reference faces yet
(issue #645): single-face pictures in the batch are assigned first and then
serve as the comparison set for the multi-face pictures, so group shots pick
the same identity instead of simply the largest face.

Seeding follows the deterministic pattern of ``test_likeness_and_face_search``:
faces are inserted directly with explicit unit-ish feature vectors, so every
similarity in the assertions is arithmetic rather than a property of a real
face model.
"""

import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from pixlstash.database import create_configured_engine
from pixlstash.db_models import Character, Picture, Tag
from pixlstash.db_models.face import Face
from pixlstash.scoring import (
    compute_character_likeness_for_faces,
    match_faces_to_characters,
    select_reference_faces_for_character,
)
from pixlstash.server import Server
from pixlstash.services import operation_log_service
from pixlstash.tasks.missing_text_embedding_finder import MissingTextEmbeddingFinder
from tests.test_server import random_images
from tests.utils import API_PREFIX, upload_pictures_and_wait

ALONG_X = np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).tobytes()
ALONG_Y = np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32).tobytes()
# 0.9 of the way toward +x: a strong match for ALONG_X, weak for ALONG_Y.
NEAR_X = np.asarray([0.9, 0.436, 0, 0, 0, 0, 0, 0], dtype=np.float32).tobytes()

SMALL_BBOX = [0, 0, 10, 10]
LARGE_BBOX = [0, 0, 200, 200]


def _add_face(server, pic_id, features, bbox, face_index=0, character_id=None):
    """Insert one Face row with explicit features and bbox; return its id."""
    holder = {}

    def _add(session):
        face = Face(
            picture_id=pic_id,
            frame_index=0,
            face_index=face_index,
            character_id=character_id,
            bbox=bbox,
            features=features,
        )
        session.add(face)
        session.commit()
        session.refresh(face)
        holder["id"] = face.id

    server.vault.db.run_task(_add)
    return holder["id"]


def _face_character_ids(server, face_ids):
    """Map face id to its current character_id, read straight from the DB."""

    def _fetch(session):
        rows = session.exec(
            select(Face.id, Face.character_id).where(Face.id.in_(list(face_ids)))
        ).all()
        return {int(fid): cid for fid, cid in rows}

    return server.vault.db.run_task(_fetch)


def _upload(client, count):
    """Upload ``count`` noise pictures and return their picture ids."""
    images = [
        ("file", (f"img{i}.png", random_images[i], "image/png")) for i in range(count)
    ]
    import_status = upload_pictures_and_wait(client, images)
    assert import_status["status"] == "completed"
    return [r["picture_id"] for r in import_status["results"]]


def _create_character(client, name):
    resp = client.post(f"{API_PREFIX}/characters", json={"name": name})
    assert resp.status_code == 200, resp.text
    return resp.json()["character"]["id"]


def _assign_pictures(client, character_id, picture_ids):
    resp = client.post(
        f"{API_PREFIX}/characters/{character_id}/faces",
        json={"picture_ids": picture_ids},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    return body


def _stack(client, picture_ids):
    response = client.post(f"{API_PREFIX}/stacks", json={"picture_ids": picture_ids})
    assert response.status_code == 200, response.text
    return response.json()


def test_record_failure_rolls_back_face_assignment(monkeypatch):
    """A face-to-character join cannot survive without its undo receipt."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            login = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert login.status_code == 200
            picture_id = _upload(client, 1)[0]
            character_id = _create_character(client, "Atomic")
            face_id = _add_face(server, picture_id, ALONG_X, SMALL_BBOX)
            monkeypatch.setattr(
                operation_log_service,
                "record_operation_in_session",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("face receipt failed")
                ),
            )

            with pytest.raises(RuntimeError, match="face receipt failed"):
                client.post(
                    f"{API_PREFIX}/characters/{character_id}/faces",
                    json={"face_ids": [face_id]},
                )

            assert _face_character_ids(server, [face_id]) == {face_id: None}
            assert (
                operation_log_service.list_operations(
                    server.vault, op_type="characters.assign"
                )
                == []
            )


def test_authoritative_face_assignment_rejects_a_mismatched_picture():
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            assert (
                client.post(
                    "/login",
                    json={"username": "testuser", "password": "testpassword"},
                ).status_code
                == 200
            )
            first, second = _upload(client, 2)
            character_id = _create_character(client, "Mismatch")
            first_face_id = _add_face(server, first, ALONG_X, SMALL_BBOX)
            second_face_id = _add_face(server, second, ALONG_X, SMALL_BBOX)
            _stack(client, [first, second])

            response = client.post(
                f"{API_PREFIX}/characters/{character_id}/faces",
                json={
                    "face_assignments": [
                        {"picture_id": second, "face_id": first_face_id}
                    ]
                },
            )

            assert response.status_code == 422, response.text
            assert _face_character_ids(server, [first_face_id, second_face_id]) == {
                first_face_id: None,
                second_face_id: None,
            }


def test_authoritative_face_assignment_is_exact_stack_atomic_and_undoable():
    """The reviewed face wins verbatim; stack siblings use normal selection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            assert (
                client.post(
                    "/login",
                    json={"username": "testuser", "password": "testpassword"},
                ).status_code
                == 200
            )
            reference_id, reviewed_id, sibling_id = _upload(client, 3)
            character_id = _create_character(client, "Stack atomic")
            _add_face(
                server,
                reference_id,
                ALONG_X,
                SMALL_BBOX,
                character_id=character_id,
            )

            reviewed_winner = _add_face(
                server, reviewed_id, ALONG_Y, SMALL_BBOX, face_index=0
            )
            reviewed_rerank_winner = _add_face(
                server, reviewed_id, ALONG_X, LARGE_BBOX, face_index=1
            )
            sibling_bystander = _add_face(
                server, sibling_id, ALONG_Y, LARGE_BBOX, face_index=0
            )
            sibling_target = _add_face(
                server, sibling_id, NEAR_X, SMALL_BBOX, face_index=1
            )
            _stack(client, [reviewed_id, sibling_id])

            assigned = client.post(
                f"{API_PREFIX}/characters/{character_id}/faces",
                json={
                    "face_assignments": [
                        {"picture_id": reviewed_id, "face_id": reviewed_winner}
                    ]
                },
            )

            assert assigned.status_code == 200, assigned.text
            assert set(assigned.json()["face_ids"]) == {
                reviewed_winner,
                sibling_target,
            }
            assert _face_character_ids(
                server,
                [
                    reviewed_winner,
                    reviewed_rerank_winner,
                    sibling_bystander,
                    sibling_target,
                ],
            ) == {
                reviewed_winner: character_id,
                reviewed_rerank_winner: None,
                sibling_bystander: None,
                sibling_target: character_id,
            }

            undone = client.post(f"{API_PREFIX}/operations/undo")
            assert undone.status_code == 200, undone.text
            assert set(undone.json()["picture_ids"]) == {reviewed_id, sibling_id}
            assert _face_character_ids(
                server,
                [
                    reviewed_winner,
                    reviewed_rerank_winner,
                    sibling_bystander,
                    sibling_target,
                ],
            ) == {
                reviewed_winner: None,
                reviewed_rerank_winner: None,
                sibling_bystander: None,
                sibling_target: None,
            }


def test_face_search_mixed_legacy_reference_widths_returns_422():
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            assert (
                client.post(
                    "/login",
                    json={"username": "testuser", "password": "testpassword"},
                ).status_code
                == 200
            )
            first, second, candidate = _upload(client, 3)
            character_id = _create_character(client, "Mixed widths")
            _add_face(
                server,
                first,
                np.asarray([1, 0, 0, 0], dtype=np.float32).tobytes(),
                SMALL_BBOX,
                character_id=character_id,
            )
            _add_face(
                server,
                second,
                np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).tobytes(),
                SMALL_BBOX,
                character_id=character_id,
            )
            _add_face(
                server,
                candidate,
                np.asarray([1, 0, 0, 0], dtype=np.float32).tobytes(),
                SMALL_BBOX,
            )

            response = client.post(
                f"{API_PREFIX}/pictures/face-search",
                params={
                    "source_character_id": character_id,
                    "exclude_character_id": character_id,
                },
            )

            assert response.status_code == 422, response.text
            assert "incompatible embedding widths [4, 8]" in response.json()["detail"]


def test_suggest_more_max_winner_is_assigned_without_softmax_reranking():
    """Search and assignment share one winning-face decision."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            assert (
                client.post(
                    "/login",
                    json={"username": "testuser", "password": "testpassword"},
                ).status_code
                == 200
            )
            picture_ids = _upload(client, 11)
            reference_ids, candidate_id = picture_ids[:10], picture_ids[10]
            character_id = _create_character(client, "Reducer parity")

            ref_a = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
            ref_b = np.asarray([0.5, 0.8660254], dtype=np.float32).tobytes()
            for index, picture_id in enumerate(reference_ids):
                _add_face(
                    server,
                    picture_id,
                    ref_a if index == 0 else ref_b,
                    SMALL_BBOX,
                    character_id=character_id,
                )
            candidate_a = _add_face(
                server, candidate_id, ref_a, SMALL_BBOX, face_index=0
            )
            candidate_b = _add_face(
                server,
                candidate_id,
                np.asarray([0.8660254, 0.5], dtype=np.float32).tobytes(),
                LARGE_BBOX,
                face_index=1,
            )

            def old_softmax_winner(session):
                refs = session.exec(
                    select(Face).where(Face.picture_id.in_(reference_ids))
                ).all()
                candidates = session.exec(
                    select(Face).where(Face.picture_id == candidate_id)
                ).all()
                scores = compute_character_likeness_for_faces(refs, candidates)
                return max(scores, key=scores.get)

            assert server.vault.db.run_task(old_softmax_winner) == candidate_b

            search = client.post(
                f"{API_PREFIX}/pictures/face-search",
                params={
                    "source_character_id": character_id,
                    "exclude_character_id": character_id,
                    "combine": "max",
                    "top_n": 100,
                },
            )
            assert search.status_code == 200, search.text
            match = next(
                row for row in search.json() if row["picture_id"] == candidate_id
            )
            assert match["face_id"] == candidate_a

            assigned = client.post(
                f"{API_PREFIX}/characters/{character_id}/faces",
                json={
                    "face_assignments": [
                        {
                            "picture_id": match["picture_id"],
                            "face_id": match["face_id"],
                        }
                    ]
                },
            )
            assert assigned.status_code == 200, assigned.text
            assert assigned.json()["face_ids"] == [candidate_a]
            assert _face_character_ids(server, [candidate_a, candidate_b]) == {
                candidate_a: character_id,
                candidate_b: None,
            }


def test_new_character_bootstraps_multi_face_pictures_from_solo_shots():
    """A brand-new character assigned a mixed batch: the solo shots are
    assigned directly and their faces become the comparison set for the group
    shot, so the group shot assigns the matching (smaller) face instead of the
    largest bystander. The multi-face picture comes first in the request to
    prove the two-pass ordering, not the request order, decides."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200

            group_pid, solo1_pid, solo2_pid = _upload(client, 3)
            character_id = _create_character(client, "Newcomer")

            solo1_fid = _add_face(server, solo1_pid, ALONG_X, SMALL_BBOX)
            solo2_fid = _add_face(server, solo2_pid, ALONG_X, SMALL_BBOX)
            bystander_fid = _add_face(
                server, group_pid, ALONG_Y, LARGE_BBOX, face_index=0
            )
            target_fid = _add_face(server, group_pid, NEAR_X, SMALL_BBOX, face_index=1)

            body = _assign_pictures(
                client, character_id, [group_pid, solo1_pid, solo2_pid]
            )

            assigned = _face_character_ids(
                server, [solo1_fid, solo2_fid, bystander_fid, target_fid]
            )
            assert assigned[solo1_fid] == character_id
            assert assigned[solo2_fid] == character_id
            assert assigned[target_fid] == character_id, (
                f"group shot assigned the wrong face: {assigned}, response {body}"
            )
            assert assigned[bystander_fid] is None, (
                f"largest bystander face was assigned: {assigned}"
            )
            assert set(body["face_ids"]) == {solo1_fid, solo2_fid, target_fid}


def test_new_character_all_multi_face_keeps_largest_face_fallback():
    """With no reference faces and no single-face pictures to bootstrap from,
    the largest-face fallback still applies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200

            (group_pid,) = _upload(client, 1)
            character_id = _create_character(client, "Newcomer")

            largest_fid = _add_face(
                server, group_pid, ALONG_Y, LARGE_BBOX, face_index=0
            )
            smaller_fid = _add_face(server, group_pid, NEAR_X, SMALL_BBOX, face_index=1)

            body = _assign_pictures(client, character_id, [group_pid])

            assigned = _face_character_ids(server, [largest_fid, smaller_fid])
            assert assigned[largest_fid] == character_id, (
                f"largest-face fallback did not apply: {assigned}, response {body}"
            )
            assert assigned[smaller_fid] is None
            assert body["face_ids"] == [largest_fid]


def test_existing_reference_faces_still_drive_selection():
    """A character that already has reference faces keeps the pre-existing
    behaviour: the multi-face picture assigns the face most like the
    references, even when it is not the largest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = os.path.join(temp_dir, "server_config.json")
        with Server(server_config_path=server_config_path) as server:
            client = TestClient(server.api)
            resp = client.post(
                "/login", json={"username": "testuser", "password": "testpassword"}
            )
            assert resp.status_code == 200

            ref_pid, group_pid = _upload(client, 2)
            character_id = _create_character(client, "Established")

            # Pre-assigned reference face: makes
            # select_reference_faces_for_character return a non-empty set.
            ref_fid = _add_face(
                server, ref_pid, ALONG_X, SMALL_BBOX, character_id=character_id
            )
            bystander_fid = _add_face(
                server, group_pid, ALONG_Y, LARGE_BBOX, face_index=0
            )
            target_fid = _add_face(server, group_pid, NEAR_X, SMALL_BBOX, face_index=1)

            body = _assign_pictures(client, character_id, [group_pid])

            assigned = _face_character_ids(server, [ref_fid, bystander_fid, target_fid])
            assert assigned[ref_fid] == character_id
            assert assigned[target_fid] == character_id, (
                f"reference-likeness selection regressed: {assigned}, response {body}"
            )
            assert assigned[bystander_fid] is None
            assert body["face_ids"] == [target_fid]


# --- unassigning a person's faces -------------------------------------------


@pytest.fixture(scope="module")
def unassign_env():
    """One server for the unassign tests: ``(server, client, picture ids)``.

    Each test takes its own slice of the eight pictures and creates its own
    people, so the tests share the server without sharing what they assert on.
    The text-embedding finder is kept idle so a cleared embedding is not
    refilled under the assertion.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MissingTextEmbeddingFinder, "_guard", lambda self: False)
        with tempfile.TemporaryDirectory() as temp_dir:
            server_config_path = os.path.join(temp_dir, "server_config.json")
            with Server(server_config_path=server_config_path) as server:
                client = TestClient(server.api)
                login = client.post(
                    "/login", json={"username": "testuser", "password": "testpassword"}
                )
                assert login.status_code == 200
                yield server, client, _upload(client, 8)


def _seed_text_embeddings(server, picture_ids):
    """Give each picture a stand-in text embedding an unassign may clear."""

    def _seed(session):
        pictures = session.exec(
            select(Picture).where(Picture.id.in_(list(picture_ids)))
        ).all()
        for picture in pictures:
            picture.text_embedding = b"example-text-embedding"
            session.add(picture)
        session.commit()

    server.vault.db.run_task(_seed)


def _has_text_embedding(server, picture_ids):
    """Map picture id to whether it still has a text embedding."""

    def _read(session):
        rows = session.exec(
            select(Picture.id, Picture.text_embedding).where(
                Picture.id.in_(list(picture_ids))
            )
        ).all()
        return {int(pid): embedding is not None for pid, embedding in rows}

    return server.vault.db.run_immediate_read_task(_read)


def _unassign(client, character_id, body):
    return client.request(
        "DELETE", f"{API_PREFIX}/characters/{character_id}/faces", json=body
    )


def _unassign_operation_ids(server):
    return [
        operation["id"]
        for operation in operation_log_service.list_operations(
            server.vault, op_type="characters.unassign"
        )
    ]


def test_unassigning_by_face_id_clears_that_pictures_text_embedding(unassign_env):
    """A face-id request names no pictures, so the route has to find them.

    It used to clear the text embedding of no picture at all, which left the
    person's name in the picture's search text.
    """
    server, client, pictures = unassign_env
    target, unrelated = pictures[0:2]
    character_id = _create_character(client, "Unassigned by face")
    target_face = _add_face(
        server, target, ALONG_X, SMALL_BBOX, character_id=character_id
    )
    unrelated_face = _add_face(
        server, unrelated, ALONG_X, SMALL_BBOX, character_id=character_id
    )
    _seed_text_embeddings(server, [target, unrelated])

    response = _unassign(client, character_id, {"face_ids": [target_face]})

    assert response.status_code == 200, response.text
    assert _face_character_ids(server, [target_face, unrelated_face]) == {
        target_face: None,
        unrelated_face: character_id,
    }
    assert _has_text_embedding(server, [target, unrelated]) == {
        target: False,
        unrelated: True,
    }
    newest = operation_log_service.list_operations(
        server.vault, op_type="characters.unassign"
    )[0]
    assert newest["target_ids"] == [target]

    undone = client.post(f"{API_PREFIX}/operations/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json()["picture_ids"] == [target]
    assert _face_character_ids(server, [target_face]) == {target_face: character_id}


def test_unassigning_by_picture_id_clears_only_pictures_that_lost_the_person(
    unassign_env,
):
    """A named picture whose face is someone else's keeps its text embedding."""
    server, client, pictures = unassign_env
    target, someone_elses, unrelated = pictures[2:5]
    character_id = _create_character(client, "Unassigned by picture")
    other_character_id = _create_character(client, "Somebody else")
    target_face = _add_face(
        server, target, ALONG_X, SMALL_BBOX, character_id=character_id
    )
    other_face = _add_face(
        server, someone_elses, ALONG_X, SMALL_BBOX, character_id=other_character_id
    )
    unrelated_face = _add_face(
        server, unrelated, ALONG_X, SMALL_BBOX, character_id=character_id
    )
    _seed_text_embeddings(server, [target, someone_elses, unrelated])

    response = _unassign(client, character_id, {"picture_ids": [target, someone_elses]})

    assert response.status_code == 200, response.text
    assert _face_character_ids(server, [target_face, other_face, unrelated_face]) == {
        target_face: None,
        other_face: other_character_id,
        unrelated_face: character_id,
    }
    assert _has_text_embedding(server, [target, someone_elses, unrelated]) == {
        target: False,
        someone_elses: True,
        unrelated: True,
    }
    newest = operation_log_service.list_operations(
        server.vault, op_type="characters.unassign"
    )[0]
    assert newest["target_ids"] == [target]


def test_an_unassign_that_matches_nothing_answers_200_and_changes_nothing(
    unassign_env,
):
    """Each of these used to end the request in a 500."""
    server, client, pictures = unassign_env
    held, bare = pictures[5:7]
    character_id = _create_character(client, "Matches nothing")
    other_character_id = _create_character(client, "Not them")
    own_face = _add_face(
        server, held, ALONG_X, SMALL_BBOX, face_index=0, character_id=character_id
    )
    other_face = _add_face(
        server,
        held,
        ALONG_Y,
        SMALL_BBOX,
        face_index=1,
        character_id=other_character_id,
    )
    unknown_face = max(own_face, other_face) + 1_000_000
    _seed_text_embeddings(server, [held, bare])
    recorded = _unassign_operation_ids(server)

    for body in (
        {"face_ids": [unknown_face]},
        {"face_ids": [other_face]},
        {"picture_ids": [bare]},
        {"picture_ids": []},
    ):
        response = _unassign(client, character_id, body)

        assert response.status_code == 200, (body, response.text)
        assert _face_character_ids(server, [own_face, other_face]) == {
            own_face: character_id,
            other_face: other_character_id,
        }, body
        assert _has_text_embedding(server, [held, bare]) == {
            held: True,
            bare: True,
        }, body
    assert _unassign_operation_ids(server) == recorded


def test_an_unknown_face_id_beside_a_real_one_still_unassigns_the_real_one(
    unassign_env,
):
    server, client, pictures = unassign_env
    target = pictures[7]
    character_id = _create_character(client, "Beside an unknown face")
    target_face = _add_face(
        server, target, ALONG_X, SMALL_BBOX, character_id=character_id
    )
    _seed_text_embeddings(server, [target])

    response = _unassign(
        client, character_id, {"face_ids": [target_face, target_face + 1_000_000]}
    )

    assert response.status_code == 200, response.text
    assert _face_character_ids(server, [target_face]) == {target_face: None}
    assert _has_text_embedding(server, [target]) == {target: False}


# --- a picture of several people (a run loading their LoRAs) ---------------


def _vec(*values):
    return np.asarray(values, dtype=np.float32).tobytes()


@pytest.fixture
def library():
    """An in-memory vault: matching reads faces, no server needed."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _person(session, name, *references):
    """A person whose reference faces (one picture each) have *references*."""
    character = Character(name=name)
    session.add(character)
    session.commit()
    for features in references:
        picture = Picture(file_path=f"{name}.png", deleted=False, score=5)
        session.add(picture)
        session.commit()
        session.add(
            Face(
                picture_id=picture.id,
                face_index=0,
                character_id=character.id,
                bbox=SMALL_BBOX,
                features=features,
            )
        )
    session.commit()
    return character.id


def _group(session, *faces):
    """A picture with *faces*, each ``(features, bbox)`` or with a character."""
    picture = Picture(file_path="group.png", deleted=False)
    session.add(picture)
    session.commit()
    rows = []
    for index, (features, bbox, *named) in enumerate(faces):
        face = Face(
            picture_id=picture.id,
            face_index=index,
            bbox=bbox,
            features=features,
            character_id=named[0] if named else None,
        )
        session.add(face)
        rows.append(face)
    session.commit()
    return rows


def test_two_peoples_faces_are_matched_by_likeness_not_size(library):
    a = _person(library, "a", ALONG_X)
    b = _person(library, "b", ALONG_Y)
    like_b, like_a = _group(library, (ALONG_Y, LARGE_BBOX), (NEAR_X, SMALL_BBOX))
    assert match_faces_to_characters(library, [like_b, like_a], [a, b]) == {
        like_a.id: a,
        like_b.id: b,
    }


def test_the_strongest_pair_is_taken_first(library):
    """f1 is most like a (0.83), so a gets it, and b takes f2 though it is
    unlike them (0.00): b is in the run, so b gets the face left."""
    a = _person(library, "a", _vec(-0.293, 0.035, -0.956))
    b = _person(library, "b", _vec(-0.174, -0.918, -0.356))
    f1, f2 = _group(
        library,
        (_vec(-0.506, -0.452, -0.734), SMALL_BBOX),
        (_vec(0.524, 0.219, -0.823), SMALL_BBOX),
    )
    likeness = compute_character_likeness_for_faces(
        [library.exec(select(Face).where(Face.character_id == a)).one()], [f1, f2]
    )
    assert likeness[f1.id] > likeness[f2.id]
    assert match_faces_to_characters(library, [f1, f2], [a, b]) == {
        f1.id: a,
        f2.id: b,
    }


def test_the_less_obvious_person_still_gets_the_face_left(library):
    """b's face came out unlike b, but a takes the obvious one and b the rest."""
    a = _person(library, "a", ALONG_X)
    b = _person(library, "b", ALONG_Y)
    like_a, stranger = _group(
        library, (NEAR_X, SMALL_BBOX), (_vec(0, 0, 1, 0, 0, 0, 0, 0), LARGE_BBOX)
    )
    assert match_faces_to_characters(library, [like_a, stranger], [a, b]) == {
        like_a.id: a,
        stranger.id: b,
    }


def test_a_named_face_and_its_person_are_left_out(library):
    a = _person(library, "a", ALONG_X)
    b = _person(library, "b", ALONG_Y)
    named, free = _group(library, (NEAR_X, SMALL_BBOX, b), (ALONG_X, SMALL_BBOX))
    assert match_faces_to_characters(library, [named, free], [a, b]) == {free.id: a}


def test_one_person_with_no_references_gets_the_largest_face_left(library):
    a = _person(library, "a", ALONG_X)
    newcomer = _person(library, "newcomer")
    like_a, small, large = _group(
        library,
        (ALONG_X, LARGE_BBOX),
        (ALONG_Y, SMALL_BBOX),
        (ALONG_Y, [0, 0, 100, 100]),
    )
    assert match_faces_to_characters(
        library, [like_a, small, large], [newcomer, a]
    ) == {like_a.id: a, large.id: newcomer}


def test_people_with_no_references_take_the_largest_faces_in_turn(library):
    """Nothing to score them by, so in order, until the faces run out."""
    first = _person(library, "first")
    second = _person(library, "second")
    third = _person(library, "third")
    small, large = _group(library, (ALONG_X, SMALL_BBOX), (ALONG_Y, LARGE_BBOX))
    assert match_faces_to_characters(
        library, [small, large], [first, second, third]
    ) == {large.id: first, small.id: second}


# --- reference faces for a person with no reference cluster ----------------

EARLIEST = datetime(2024, 1, 1, tzinfo=timezone.utc)
T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 2, 1, tzinfo=timezone.utc)
T2 = datetime(2025, 3, 1, tzinfo=timezone.utc)
T3 = datetime(2025, 4, 1, tzinfo=timezone.utc)


@pytest.fixture(params=[False, True], ids=["scan-order", "reversed-scan-order"])
def any_scan_order(request, library):
    """The vault read in SQLite's own row order, then in reverse.

    ``reverse_unordered_selects`` flips the order rows come back in wherever
    no ORDER BY fixes it, so a choice that leans on that order rather than
    on its sort key changes between the two runs.
    """
    dbapi = library.connection().connection.dbapi_connection
    dbapi.execute(f"PRAGMA reverse_unordered_selects = {int(request.param)}")
    try:
        yield library
    finally:
        dbapi.execute("PRAGMA reverse_unordered_selects = 0")


@contextmanager
def _variable_limit(session, limit):
    """Lower how many parameters SQLite binds per statement, then restore it."""
    dbapi = session.connection().connection.dbapi_connection
    previous = dbapi.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    try:
        yield
    finally:
        dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)


def _picture(session, created_at, *tags, score=None, deleted=False):
    """A picture taken at *created_at* carrying *tags*; return its id."""
    picture = Picture(
        file_path="ref.png", created_at=created_at, score=score, deleted=deleted
    )
    session.add(picture)
    session.commit()
    for tag in tags:
        session.add(Tag(picture_id=picture.id, tag=tag))
    session.commit()
    return picture.id


def _face(session, character_id, picture_id, face_index=0, features=ALONG_X):
    """A face of *character_id* on *picture_id*; return its id."""
    face = Face(
        picture_id=picture_id,
        face_index=face_index,
        character_id=character_id,
        bbox=SMALL_BBOX,
        features=features,
    )
    session.add(face)
    session.commit()
    return face.id


def _reference_ids(session, character_id, max_refs=10):
    faces = select_reference_faces_for_character(session, character_id, max_refs)
    return [face.id for face in faces]


def test_unrated_references_leave_flawed_pictures_for_last(any_scan_order):
    """Penalised tags count summed and in any case; age breaks the ties."""
    session = any_scan_order
    person = _person(session, "a")
    # Seeded in reverse of the expected order, so row order alone is wrong.
    _face(session, person, _picture(session, T0, "bad anatomy"))  # 5: left out
    double = _face(session, person, _picture(session, T0, "noise", "Waxy Skin"))
    three = _face(session, person, _picture(session, T0, "Incorrect Reflection"))
    light = _face(session, person, _picture(session, T0, "WATERMARK"))  # 1
    undated = _face(session, person, _picture(session, None))
    unflawed = _face(
        session, person, _picture(session, T1, "Silicone Breasts", "smile")
    )

    assert _reference_ids(session, person) == [unflawed, undated, light, three, double]
    assert _reference_ids(session, person, max_refs=3) == [unflawed, undated, light]


def test_unrated_reference_ties_go_by_picture_then_face(any_scan_order):
    session = any_scan_order
    person = _person(session, "a")
    someone_else = _person(session, "b")
    # Older than everything below, and never a reference of "a".
    _face(session, person, _picture(session, EARLIEST, deleted=True))
    _face(session, person, _picture(session, EARLIEST), features=None)
    _face(session, someone_else, _picture(session, EARLIEST))
    _face(session, person, None)
    oldest = _face(session, person, _picture(session, T0))
    shared = _picture(session, T1)
    first_on_shared = _face(session, person, shared, face_index=0)
    second_on_shared = _face(session, person, shared, face_index=1)
    lower, higher = _picture(session, T2), _picture(session, T2)
    on_higher = _face(session, person, higher)  # the lower face id
    on_lower = _face(session, person, lower)

    assert _reference_ids(session, person) == [
        oldest,
        first_on_shared,
        second_on_shared,
        on_lower,
        on_higher,
    ]


def test_rated_references_come_first_and_only_once(any_scan_order):
    session = any_scan_order
    person = _person(session, "a")
    _face(session, person, _picture(session, EARLIEST, score=5, deleted=True))
    late_five = _face(session, person, _picture(session, T3, score=5))
    early_five = _face(session, person, _picture(session, T1, score=5))
    # The oldest picture: first in line again once the unrated ones are read.
    four = _face(session, person, _picture(session, EARLIEST, score=4))
    _face(session, person, _picture(session, T0, "noise", score=0))  # left out
    three = _face(session, person, _picture(session, T0, score=3))
    unrated = _face(session, person, _picture(session, T2))

    assert _reference_ids(session, person) == [
        early_five,
        late_five,
        four,
        three,
        unrated,
    ]
    assert _reference_ids(session, person, max_refs=2) == [early_five, late_five]
    assert _reference_ids(session, person, max_refs=3) == [
        early_five,
        late_five,
        four,
    ]
    assert _reference_ids(session, person, max_refs=4) == [
        early_five,
        late_five,
        four,
        three,
    ]


def test_a_person_with_few_faces_gets_each_of_them_once(any_scan_order):
    session = any_scan_order
    person = _person(session, "a")
    four = _face(session, person, _picture(session, T1, score=4))
    unrated = _face(session, person, _picture(session, T0, "noise"))
    _face(session, person, _picture(session, EARLIEST), features=None)

    assert _reference_ids(session, person) == [four, unrated]
    assert _reference_ids(session, _person(session, "nobody")) == []


def test_references_for_a_person_on_more_pictures_than_sqlite_binds(library):
    """Lowered here to 20 parameters a statement; a desktop SQLite stops at
    32766, which a person on that many pictures used to hit."""
    person = _person(library, "a")
    faces = [
        _face(library, person, _picture(library, T0 + timedelta(days=day)))
        for day in range(40)
    ]
    library.add(Tag(picture_id=library.get(Face, faces[0]).picture_id, tag="noise"))
    library.commit()

    with _variable_limit(library, 20):
        assert _reference_ids(library, person) == faces[1:6]


def test_references_load_the_chosen_faces_and_no_pictures(library):
    person = _person(library, "a")
    five = _face(library, person, _picture(library, T0, score=5))
    four = _face(library, person, _picture(library, T0, score=4))
    unrated = [
        _face(library, person, _picture(library, T1 + timedelta(days=day)))
        for day in range(12)
    ]
    library.expunge_all()
    loaded = Counter()

    def count(target, _context):
        loaded[type(target).__name__] += 1

    event.listen(Face, "load", count)
    event.listen(Picture, "load", count)
    try:
        ids = _reference_ids(library, person)
    finally:
        event.remove(Face, "load", count)
        event.remove(Picture, "load", count)

    assert ids == [five, four, *unrated[:3]]
    assert loaded == {"Face": 5}


def test_a_reference_deleted_while_references_are_chosen_leaves_five(tmp_path):
    """Face search, character sheets and training previews read beside the
    writer, which can delete a face (a face-model refresh, a picture deleted
    for good) while the references are being chosen. The person still gets
    five: the face as it was read, or the next one in line."""
    path = str(tmp_path / "vault.db")
    engine = create_configured_engine(path)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        person = _person(session, "a")
        faces = [
            _face(session, person, _picture(session, T0 + timedelta(days=day)))
            for day in range(7)
        ]
    deleted = []

    def delete_once_ranked(conn, cursor, statement, parameters, context, many):
        # The ranking read of the unrated pictures, the one that ranks on
        # penalised tags: delete the first face it put in line.
        ranking = " ".join(statement.split())
        if deleted or not ranking.startswith("SELECT face.id"):
            return
        if "JOIN picture" not in ranking or "FROM tag" not in ranking:
            return
        writer = sqlite3.connect(path)
        try:
            writer.execute("DELETE FROM face WHERE id = ?", (faces[0],))
            writer.commit()
        finally:
            writer.close()
        deleted.append(faces[0])

    event.listen(engine, "after_cursor_execute", delete_once_ranked)
    try:
        with Session(engine) as session:
            ids = _reference_ids(session, person)
    finally:
        event.remove(engine, "after_cursor_execute", delete_once_ranked)
        engine.dispose()

    assert deleted == [faces[0]]
    assert ids in (faces[:5], faces[1:6])
