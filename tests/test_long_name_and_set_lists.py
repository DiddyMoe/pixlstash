"""Name and set lists that grow with the owner's data bind as one JSON array.

The hidden tags, the formats a request names, the removed-from-tagging
spellings and the listed sets have no cap, and one plain ``IN`` binds a
parameter per value. SQLite refuses more than ``SQLITE_LIMIT_VARIABLE_NUMBER``
(999 before 3.32). Each of these sites binds the list once as a JSON array
(``in_json`` / ``not_in_json``), so the statement holds one parameter however
long the list is.

Every test runs the read under ``_sqlite_variable_ceiling``, which pins the
vault connections to 999 and first proves the pin took, and compares the result
with the same read made without the pin, so a site bounded by dropping values
would fail too. Each list is one real value padded with made-up ones to just
past the ceiling.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete

from pixlstash.database import DBPriority
from pixlstash.db_models.character import Character
from pixlstash.db_models.entity_project import (
    CharacterProjectMember,
    PictureSetProjectMember,
)
from pixlstash.db_models.face import Face
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_likeness import PictureLikeness
from pixlstash.db_models.picture_project import PictureProjectMember
from pixlstash.db_models.picture_set import PictureSet, PictureSetMember
from pixlstash.db_models.project import Project
from pixlstash.db_models.tag import Tag
from pixlstash.db_models.tag_suggestion import TagSuggestion
from pixlstash.server import Server
from pixlstash.services.tag_suggestion_service import list_suggestions
from pixlstash.tasks.tag_exclusion_cleanup_task import pictures_carrying
from tests.test_export_by_ids_api import VARIABLE_CEILING, _sqlite_variable_ceiling
from tests.utils import API_PREFIX

HIDDEN_TAG = "test-hidden"
# One real value and enough made-up ones to pass the ceiling.
PADDING = [f"example-name-{n:04d}" for n in range(VARIABLE_CEILING)]
HIDDEN_TAGS = [HIDDEN_TAG, *PADDING]
FORMATS = ["png", *(f"example-format-{n:04d}" for n in range(VARIABLE_CEILING))]
# Stored tag spellings removed from tagging: each one is a stored value, so the
# list the queries read is this long.
EXCLUDED = [f"example-excluded-{n:04d}" for n in range(VARIABLE_CEILING + 1)]
KEPT_SUGGESTION = "test-kept"
# Sets past the ceiling: all smart, all in the project, all children of one set.
CROWD = VARIABLE_CEILING + 1
SMART = {"filter": "format=png", "summary": None, "removed": []}
# A set whose one member carries this many tags has as many signature tags.
SIGNATURE = [f"example-signature-{n:04d}" for n in range(VARIABLE_CEILING + 1)]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("long-name-and-set-lists")
    config = root / "server-config.json"
    # No background work: every row here is written by hand.
    config.write_text(json.dumps({"port": 8000, "disable_background_workers": True}))
    server = Server(str(config))
    try:
        # The smart-set refresher runs two seconds after a library event and
        # would give the crowd's rule its matching pictures between the
        # unpinned and the pinned read. The crowd's rows are seeded by hand.
        server.smart_set_refresher.close()
        owner = TestClient(server.api)
        resp = owner.post(
            f"{API_PREFIX}/login",
            json={"username": "testuser", "password": "testpassword"},
        )
        assert resp.status_code == 200, resp.text
        resp = owner.post(f"{API_PREFIX}/characters", json={"name": "test-person"})
        assert resp.status_code == 200, resp.text
        character_id = resp.json()["character"]["id"]

        def seed(session):
            def picture(name):
                pic = Picture(
                    file_path=f"/nonexistent/{name}.png",
                    width=64,
                    height=48,
                    format="png",
                )
                session.add(pic)
                session.flush()
                return int(pic.id)

            ids = {name: picture(name) for name in ("shown", "hidden", "other")}
            excluded_id = picture("excluded")
            session.add(Tag(picture_id=ids["hidden"], tag=HIDDEN_TAG))
            # One likeness group of the three.
            for a, b in combinations(sorted(ids.values()), 2):
                session.add(
                    PictureLikeness(
                        picture_id_a=a, picture_id_b=b, likeness=0.9, metric="test"
                    )
                )
            for name in EXCLUDED:
                session.add(Tag(picture_id=excluded_id, tag=name))
                session.add(
                    TagSuggestion(
                        picture_id=excluded_id,
                        tag=name,
                        direction="add",
                        source="test",
                        score=0.5,
                    )
                )
            session.add(
                TagSuggestion(
                    picture_id=excluded_id,
                    tag=KEPT_SUGGESTION,
                    direction="add",
                    source="test",
                    score=0.4,
                )
            )

            project = Project(name="test-project")
            session.add(project)
            session.flush()
            for pic_id in ids.values():
                session.add(
                    PictureProjectMember(picture_id=pic_id, project_id=project.id)
                )
            group = PictureSet(name="test-group")
            session.add(group)
            parent = PictureSet(name="test-parent")
            session.add(parent)
            session.flush()
            for pic_id in ids.values():
                session.add(PictureSetMember(set_id=group.id, picture_id=pic_id))

            crowd = []
            for n in range(CROWD):
                child = PictureSet(
                    name=f"example-set-{n:04d}",
                    parent_set_id=parent.id,
                    smart=dict(SMART),
                )
                session.add(child)
                crowd.append(child)
            session.flush()
            for child in crowd:
                session.add(
                    PictureSetProjectMember(set_id=child.id, project_id=project.id)
                )
            # The owner's own row makes the first crowd set custom; the second
            # is custom by what it took out.
            session.add(
                PictureSetMember(
                    set_id=crowd[0].id, picture_id=ids["shown"], smart_auto=False
                )
            )
            crowd[1].smart = {**SMART, "removed": [ids["other"]]}
            session.commit()
            return {
                **ids,
                "excluded": excluded_id,
                "project": int(project.id),
                "group": int(group.id),
                "parent": int(parent.id),
                "crowd": [int(child.id) for child in crowd],
            }

        ids = server.vault.db.run_task(seed, priority=DBPriority.IMMEDIATE)
        yield server, owner, character_id, ids
    finally:
        server.close()


def _pinned_matches_unpinned(server, call):
    """Run *call* without and then under the pin; return the unpinned body."""
    unpinned = call()
    assert unpinned.status_code == 200, unpinned.text
    with _sqlite_variable_ceiling(server):
        pinned = call()
    assert pinned.status_code == 200, pinned.text
    assert pinned.json() == unpinned.json()
    return unpinned.json()


@contextmanager
def _hidden(owner, tags):
    resp = owner.patch(f"{API_PREFIX}/users/me/config", json={"hidden_tags": tags})
    assert resp.status_code == 200, resp.text
    try:
        yield
    finally:
        resp = owner.patch(f"{API_PREFIX}/users/me/config", json={"hidden_tags": []})
        assert resp.status_code == 200, resp.text


@contextmanager
def _excluded(server, names):
    server.vault.db.set_tag_exclusions(frozenset(names))
    try:
        yield
    finally:
        server.vault.db.set_tag_exclusions(frozenset())


def _get(owner, path, **params):
    return lambda: owner.get(f"{API_PREFIX}{path}", params=params)


# The owner's hidden tags.


def test_likeness_groups_leave_out_a_hidden_picture(env):
    """``_fetch_hidden_picture_ids`` (routes/pictures/_helpers.py)."""
    server, owner, _character_id, ids = env
    trio = sorted((ids["shown"], ids["hidden"], ids["other"]))
    unfiltered = owner.get(f"{API_PREFIX}/pictures/likeness-groups")
    assert unfiltered.status_code == 200, unfiltered.text
    assert sorted(row["id"] for row in unfiltered.json()) == trio
    with _hidden(owner, HIDDEN_TAGS):
        body = _pinned_matches_unpinned(
            server, _get(owner, "/pictures/likeness-groups", apply_tag_filter="true")
        )
    assert sorted(row["id"] for row in body) == sorted((ids["shown"], ids["other"]))


def test_the_project_summary_leaves_out_a_hidden_picture(env):
    server, owner, _character_id, ids = env
    path = f"/projects/{ids['project']}/summary"
    with _hidden(owner, HIDDEN_TAGS):
        assert _get(owner, path)().json() == {"image_count": 3}
        body = _pinned_matches_unpinned(
            server, _get(owner, path, apply_tag_filter="true")
        )
    assert body == {"image_count": 2}


def test_the_all_summary_leaves_out_a_hidden_picture(env):
    server, owner, _character_id, _ids = env
    with _hidden(owner, HIDDEN_TAGS):
        everything = _get(owner, "/characters/ALL/summary")().json()["image_count"]
        body = _pinned_matches_unpinned(
            server, _get(owner, "/characters/ALL/summary", apply_tag_filter="true")
        )
    assert body["image_count"] == everything - 1


def test_the_set_list_counts_leave_out_a_hidden_picture(env):
    """``_hidden_tag_condition`` in the sets' sidebar list."""
    server, owner, _character_id, ids = env
    with _hidden(owner, HIDDEN_TAGS):
        body = _pinned_matches_unpinned(
            server, _get(owner, "/picture_sets", apply_tag_filter="true")
        )
    group = next(s for s in body if s["id"] == ids["group"])
    assert group["picture_count"] == 2
    assert ids["hidden"] not in group["top_picture_ids"]


def test_a_set_leaves_out_a_hidden_picture(env):
    """``_filter_hidden_picture_ids`` on one set's pictures."""
    server, owner, _character_id, ids = env
    path = f"/picture_sets/{ids['group']}"
    with _hidden(owner, HIDDEN_TAGS):
        body = _pinned_matches_unpinned(
            server, _get(owner, path, apply_tag_filter="true")
        )
    assert sorted(pic["id"] for pic in body["pictures"]) == sorted(
        (ids["shown"], ids["other"])
    )


# The formats a request names.


def test_a_set_sorted_by_smart_score_reads_every_named_format(env):
    server, owner, _character_id, ids = env
    path = f"/picture_sets/{ids['group']}"
    body = _pinned_matches_unpinned(
        server, _get(owner, path, sort="SMART_SCORE", format=FORMATS)
    )
    assert sorted(pic["id"] for pic in body["pictures"]) == sorted(
        (ids["shown"], ids["hidden"], ids["other"])
    )
    # The real format is what admits them.
    none = _get(owner, path, sort="SMART_SCORE", format=FORMATS[1:])()
    assert none.status_code == 200, none.text
    assert none.json()["pictures"] == []


# The stored spellings removed from tagging.


def test_the_suggestion_list_leaves_out_every_removed_spelling(env):
    server, _owner, _character_id, _ids = env

    def listed():
        return [s.tag for s in list_suggestions(server.vault, limit=len(EXCLUDED) + 5)]

    assert len(listed()) == len(EXCLUDED) + 1
    with _excluded(server, EXCLUDED):
        unpinned = listed()
        with _sqlite_variable_ceiling(server):
            pinned = listed()
    assert pinned == unpinned == [KEPT_SUGGESTION]


def test_the_backfill_count_leaves_out_a_picture_with_only_removed_tags(env):
    server, _owner, _character_id, _ids = env
    vault = server.vault

    def count():
        return vault.db.run_immediate_read_task(vault._count_missing_tag_predictions)

    everything = count()
    with _excluded(server, EXCLUDED):
        unpinned = count()
        with _sqlite_variable_ceiling(server):
            pinned = count()
    assert pinned == unpinned == everything - 1


def test_the_cleanup_finds_the_pictures_carrying_every_removed_spelling(env):
    server, _owner, _character_id, ids = env

    def carrying():
        return server.vault.db.run_immediate_read_task(pictures_carrying, EXCLUDED)

    unpinned = carrying()
    with _sqlite_variable_ceiling(server):
        pinned = carrying()
    assert pinned == unpinned == {ids["excluded"]}


# A set's signature tags.


def test_the_signature_tag_counts_read_every_signature_tag(env):
    """``fetch_tag_counts_for_pictures`` for a likeness search from a set."""
    server, owner, _character_id, _ids = env
    embedding = (np.ones(8, dtype=np.float32) / np.sqrt(8)).tobytes()

    def seed(session):
        member, match = (
            Picture(
                file_path=f"/nonexistent/{name}.png",
                width=64,
                height=48,
                format="png",
                image_embedding=embedding,
            )
            for name in ("signature-member", "signature-match")
        )
        session.add_all([member, match])
        source = PictureSet(name="test-signature")
        session.add(source)
        session.flush()
        session.add(PictureSetMember(set_id=source.id, picture_id=member.id))
        session.add_all(Tag(picture_id=member.id, tag=name) for name in SIGNATURE)
        # The match carries all but the last signature tag.
        session.add_all(Tag(picture_id=match.id, tag=name) for name in SIGNATURE[:-1])
        session.commit()
        return int(member.id), int(match.id), int(source.id)

    def unseed(session):
        session.exec(delete(Tag).where(Tag.picture_id.in_(seeded[:2])))
        session.exec(
            delete(PictureSetMember).where(PictureSetMember.set_id == seeded[2])
        )
        session.exec(delete(PictureSet).where(PictureSet.id == seeded[2]))
        session.exec(delete(Picture).where(Picture.id.in_(seeded[:2])))
        session.commit()

    seeded = server.vault.db.run_task(seed, priority=DBPriority.IMMEDIATE)
    _member_id, match_id, source_id = seeded
    try:
        body = _pinned_matches_unpinned(
            server,
            lambda: owner.post(
                f"{API_PREFIX}/pictures/likeness-search",
                params={
                    "source_set_id": source_id,
                    "exclude_set_id": source_id,
                    "include_tag_counts": True,
                    "top_n": 500,
                },
            ),
        )
        assert [row["picture_id"] for row in body] == [match_id]
        assert body[0]["tags_total"] == len(SIGNATURE)
        assert body[0]["tags_matched"] == len(SIGNATURE) - 1
    finally:
        server.vault.db.run_task(unseed, priority=DBPriority.IMMEDIATE)


# The listed sets.


def test_the_set_list_reads_every_listed_set(env):
    """The project membership, counts and custom flags of every listed set."""
    server, owner, _character_id, ids = env
    body = _pinned_matches_unpinned(server, _get(owner, "/picture_sets"))
    by_id = {s["id"]: s for s in body}
    crowd = ids["crowd"]
    assert all(by_id[sid]["project_ids"] == [ids["project"]] for sid in crowd)
    custom = {sid for sid in crowd if by_id[sid]["smart"]["custom"]}
    assert custom == {crowd[0], crowd[1]}
    assert by_id[crowd[0]]["picture_count"] == 1


def test_a_project_set_list_reads_every_listed_set(env):
    server, owner, _character_id, ids = env
    body = _pinned_matches_unpinned(
        server, _get(owner, f"/projects/{ids['project']}/picture_sets")
    )
    assert sorted(s["id"] for s in body) == ids["crowd"]
    assert all(s["project_ids"] == [ids["project"]] for s in body)


def test_a_set_with_many_child_sets_moves_under_a_person(env):
    """``_check_depth`` walks every child set of the set it moves."""
    server, owner, character_id, ids = env
    path = f"{API_PREFIX}/picture_sets/{ids['parent']}"
    try:
        with _sqlite_variable_ceiling(server):
            resp = owner.patch(path, json={"parent_character_id": character_id})
        assert resp.status_code == 200, resp.text
        moved = owner.get(path, params={"info": "true"})
        assert moved.status_code == 200, moved.text
        assert moved.json()["parent_character_id"] == character_id
    finally:
        resp = owner.patch(path, json={"parent_character_id": None})
        assert resp.status_code == 200, resp.text


# The listed people and projects.


@pytest.fixture(scope="module")
def crowds(env):
    """People and projects past the ceiling, seeded once for the tests below.

    Every person is in a project of its own and in the module's project; one
    person has a face on a picture of its own, and that picture is in one of
    the crowd's projects, so the counts the listings read are not all zero.
    """
    server, _owner, _character_id, ids = env

    def seed(session):
        projects = [Project(name=f"example-project-{n:04d}") for n in range(CROWD)]
        people = [Character(name=f"example-person-{n:04d}") for n in range(CROWD)]
        session.add_all(projects + people)
        counted = Picture(
            file_path="/nonexistent/crowd-counted.png",
            width=64,
            height=48,
            format="png",
        )
        session.add(counted)
        session.flush()
        for person, project in zip(people, projects):
            session.add(
                CharacterProjectMember(character_id=person.id, project_id=project.id)
            )
            session.add(
                CharacterProjectMember(
                    character_id=person.id, project_id=ids["project"]
                )
            )
        session.add(Face(picture_id=counted.id, character_id=people[-1].id))
        session.add(
            PictureProjectMember(picture_id=counted.id, project_id=projects[-1].id)
        )
        session.commit()
        return {
            "people": [int(p.id) for p in people],
            "projects": [int(p.id) for p in projects],
        }

    return server.vault.db.run_task(seed, priority=DBPriority.IMMEDIATE)


def test_the_people_list_reads_every_listed_person(env, crowds, key_list_plan):
    """``get_characters``: each person's projects and picture count."""
    server, owner, _character_id, ids = env
    people, projects = crowds["people"], crowds["projects"]
    body = _pinned_matches_unpinned(
        server, _get(owner, "/characters", include_counts="true")
    )
    by_id = {row["id"]: row for row in body}
    assert all(
        sorted(by_id[pid]["project_ids"]) == sorted((own, ids["project"]))
        for pid, own in zip(people, projects)
    )
    assert by_id[people[-1]]["image_count"] == 1
    assert sum(by_id[pid]["image_count"] for pid in people) == 1
    # One JSON array still searches each person's rows by key.
    engine = server.vault.db._engine
    with key_list_plan(
        engine,
        "WHERE characterprojectmember.character_id IN (SELECT",
        "sqlite_autoindex_characterprojectmember_1 (character_id=?)",
    ):
        _get(owner, "/characters")()
    with key_list_plan(
        engine, "WHERE face.character_id IN (SELECT", "ix_face_character_"
    ) as plans:
        _get(owner, "/characters", include_counts="true")()
    assert all("SCAN face" not in plan for plan in plans), plans


def test_the_project_list_counts_every_listed_project(env, crowds, key_list_plan):
    """``list_projects``: one grouped count for every listed project."""
    server, owner, _character_id, ids = env
    projects = crowds["projects"]
    body = _pinned_matches_unpinned(
        server, _get(owner, "/projects", include_counts="true")
    )
    counts = {row["id"]: row["image_count"] for row in body}
    assert counts[ids["project"]] == 3
    assert counts[projects[-1]] == 1
    assert sum(counts[pid] for pid in projects) == 1
    with key_list_plan(
        server.vault.db._engine,
        "WHERE pictureprojectmember.project_id IN (SELECT",
        "ix_pictureprojectmember_project_id",
    ):
        _get(owner, "/projects", include_counts="true")()


def test_a_set_joins_every_project_at_once(env, crowds):
    """``_ensure_unique_set_name`` and ``_auto_assign_icon_color``."""
    server, owner, _character_id, _ids = env
    projects = crowds["projects"]
    with _sqlite_variable_ceiling(server):
        resp = owner.post(
            f"{API_PREFIX}/picture_sets",
            json={"name": "test-everywhere", "project_ids": projects},
        )
    assert resp.status_code == 200, resp.text
    set_id = resp.json()["picture_set"]["id"]
    try:
        info = owner.get(f"{API_PREFIX}/picture_sets/{set_id}", params={"info": "true"})
        assert info.status_code == 200, info.text
        assert sorted(info.json()["project_ids"]) == projects
        # The name is now taken in every one of them.
        with _sqlite_variable_ceiling(server):
            clash = owner.post(
                f"{API_PREFIX}/picture_sets",
                json={"name": "test-everywhere", "project_ids": projects},
            )
        assert clash.status_code == 409, clash.text
    finally:
        resp = owner.delete(f"{API_PREFIX}/picture_sets/{set_id}")
        assert resp.status_code == 200, resp.text


def test_a_person_joins_every_project_at_once(env, crowds):
    """``_ensure_unique_character_name`` and ``_require_projects_exist``."""
    server, owner, _character_id, ids = env
    people, projects = crowds["people"], crowds["projects"]
    path = f"{API_PREFIX}/characters/{people[0]}"
    try:
        with _sqlite_variable_ceiling(server):
            resp = owner.patch(path, json={"project_ids": projects})
        assert resp.status_code == 200, resp.text
        listed = _get(owner, "/characters")().json()
        joined = next(row for row in listed if row["id"] == people[0])
        assert sorted(joined["project_ids"]) == projects
        # A second person may not take the name in any of them.
        with _sqlite_variable_ceiling(server):
            clash = owner.patch(
                f"{API_PREFIX}/characters/{people[1]}",
                json={"name": "example-person-0000", "project_ids": projects},
            )
        assert clash.status_code == 409, clash.text
    finally:
        resp = owner.patch(
            path, json={"project_ids": sorted((projects[0], ids["project"]))}
        )
        assert resp.status_code == 200, resp.text
