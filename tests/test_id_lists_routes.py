"""Caller-sized id lists hold past SQLite's bound-parameter limit.

An ``IN`` list binds one parameter per id, and SQLite refuses a statement that
binds more than 999 before 3.32 (32,766 since). ``_sqlite_variable_ceiling``
pins every vault connection to 999 for one block, so each test here sends one
id more than that and would fail with "too many SQL variables" on any build if
its site went back to one plain ``IN``.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureSet,
    PictureSetMember,
    Project,
    Tag,
)
from pixlstash.server import Server
from pixlstash.utils.sql_chunking import in_json
from tests.test_keep_cover_only import _sqlite_variable_ceiling
from tests.utils import API_PREFIX

# One id past the pinned ceiling.
OVER = 1000
POOL_TAG = "idlist-pool-tag"


@pytest.fixture(scope="module")
def env():
    """A library of OVER pictures, all in one set and one tagged, with one person.

    The person's only face is on the last picture, so a read that stopped one
    chunk short would miss it.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "server-config.json")
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as server:
            owner = TestClient(server.api)
            r = owner.post(
                f"{API_PREFIX}/login",
                json={"username": "owner", "password": "example-owner-password"},
            )
            assert r.status_code == 200, r.text

            def seed(session):
                now = datetime.now(timezone.utc)
                pool = [
                    Picture(
                        file_path=f"/nonexistent/idlist-{n}.png",
                        format="png",
                        width=64,
                        height=64,
                        pixel_sha=f"idlist-{n}",
                        imported_at=now,
                        smart_score=1.0,
                        image_embedding=np.full(8, 1.0 + n, np.float32).tobytes(),
                    )
                    for n in range(OVER)
                ]
                picture_set = PictureSet(name="idlist-set")
                person = Character(name="idlist-person")
                session.add_all([*pool, picture_set, person])
                session.flush()
                for pic in pool:
                    session.add(
                        PictureSetMember(set_id=picture_set.id, picture_id=pic.id)
                    )
                    session.add(Tag(picture_id=pic.id, tag=POOL_TAG))
                session.commit()
                return (
                    sorted(int(pic.id) for pic in pool),
                    int(picture_set.id),
                    int(person.id),
                )

            pool_ids, set_id, person_id = server.vault.db.run_task(seed)
            yield server, owner, pool_ids, set_id, person_id


def test_in_json_matches_a_list_past_the_ceiling(env):
    """``in_json`` binds the whole list as one parameter; text matches by case."""
    server, _owner, pool_ids, _set_id, _person_id = env

    def read(session):
        ids = session.exec(select(Picture.id).where(in_json(Picture.id, pool_ids)))
        tags = session.exec(
            select(Tag.tag).where(in_json(Tag.tag, [POOL_TAG.upper()])).limit(1)
        )
        return sorted(ids.all()), tags.all()

    with _sqlite_variable_ceiling(server):
        ids, upper = server.vault.db.run_task(read)
    assert ids == pool_ids
    assert upper == []


def test_a_flush_touching_every_picture_resets_every_smart_score(env):
    """A face added to each of OVER pictures in one flush clears all their scores."""
    server, _owner, pool_ids, _set_id, _person_id = env

    def add_faces(session):
        for pid in pool_ids:
            session.add(
                Face(picture_id=pid, frame_index=0, face_index=1, bbox=[0, 0, 4, 4])
            )
        session.commit()

    with _sqlite_variable_ceiling(server):
        server.vault.db.run_task(add_faces)

    scored = server.vault.db.run_task(
        lambda session: session.exec(
            select(Picture.id).where(Picture.smart_score.is_not(None))
        ).all()
    )
    assert scored == []


def test_likeness_search_from_a_set_past_the_ceiling(env):
    """A set of OVER pictures is read whole as the query, the scope and its tags."""
    server, owner, pool_ids, set_id, _person_id = env
    params = {
        "source_set_id": set_id,
        "set_id": set_id,
        "include_tag_counts": "true",
        "top_n": 5,
    }
    with _sqlite_variable_ceiling(server):
        r = owner.post(f"{API_PREFIX}/pictures/likeness-search", params=params)
    assert r.status_code == 200, r.text
    matches = r.json()
    assert len(matches) == 5
    assert {m["tags_total"] for m in matches} == {1}


def test_a_set_filter_past_the_ceiling_reads_every_set(env):
    """``GET /pictures?set_ids=..`` naming OVER sets lists the last one's pictures."""
    server, owner, pool_ids, _set_id, _person_id = env

    def seed(session):
        small = PictureSet(name="idlist-small-set")
        session.add(small)
        session.flush()
        for pid in pool_ids[:3]:
            session.add(PictureSetMember(set_id=small.id, picture_id=pid))
        session.commit()
        return int(small.id)

    small_id = server.vault.db.run_task(seed)
    set_ids = [small_id + 1 + n for n in range(OVER - 1)] + [small_id]
    with _sqlite_variable_ceiling(server):
        r = owner.get(
            f"{API_PREFIX}/pictures", params={"set_ids": set_ids, "fields": "grid"}
        )
    assert r.status_code == 200, r.text
    assert sorted(pic["id"] for pic in r.json()) == pool_ids[:3]


def test_a_people_filter_past_the_ceiling_reads_every_person(env, monkeypatch):
    """``GET /pictures/search?character_ids=..`` finds the last person's picture."""
    server, owner, pool_ids, _set_id, person_id = env

    def assign(session):
        face = Face(
            picture_id=pool_ids[-1],
            frame_index=0,
            face_index=2,
            character_id=person_id,
            bbox=[0, 0, 8, 8],
        )
        session.add(face)
        session.commit()

    server.vault.db.run_task(assign)
    candidates = []
    real_semantic_search = Picture.semantic_search

    def spy(cls, session, *args, **kwargs):
        candidates.append(sorted(kwargs["candidate_ids"]))
        return real_semantic_search(session, *args, **kwargs)

    monkeypatch.setattr(
        server.vault, "generate_text_embedding", lambda _q: np.ones(384, np.float32)
    )
    monkeypatch.setattr(
        server.vault, "generate_clip_text_embedding", lambda _q: np.ones(8, np.float32)
    )
    monkeypatch.setattr(Picture, "semantic_search", classmethod(spy))
    people = [person_id + 1 + n for n in range(OVER - 1)] + [person_id]
    with _sqlite_variable_ceiling(server):
        r = owner.get(
            f"{API_PREFIX}/pictures/search",
            params={"query": POOL_TAG, "character_ids": people},
        )
    assert r.status_code == 200, r.text
    assert candidates == [[pool_ids[-1]]]


def test_a_project_list_past_the_ceiling_is_checked_whole(env):
    """A character joined to OVER projects in one PATCH checks they all exist."""
    server, owner, _pool_ids, _set_id, person_id = env

    def seed(session):
        projects = [Project(name=f"idlist-project-{n}") for n in range(OVER)]
        session.add_all(projects)
        session.commit()
        return sorted(int(p.id) for p in projects)

    project_ids = server.vault.db.run_task(seed)
    with _sqlite_variable_ceiling(server):
        r = owner.patch(
            f"{API_PREFIX}/characters/{person_id}", json={"project_ids": project_ids}
        )
    assert r.status_code == 200, r.text
