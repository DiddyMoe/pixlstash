"""Picture-id lists in the routes hold past SQLite's bound-parameter limit.

An ``IN`` list binds one parameter per id. SQLite refuses a statement that
binds more than ``SQLITE_LIMIT_VARIABLE_NUMBER``: 999 before 3.32, 32,766 on
the python.org Windows and macOS builds, more on some Linux builds. A list
whose length grows with a selection, a set, a character or the library must
therefore be read from a temp table (``scope_id_subquery``) or chunked
(``chunked``). Which one depends on the statement: chunking returns the same
result only when each id stands alone (a plain SELECT, a per-id write, a
GROUP BY on the listed column); ORDER BY with LIMIT, or an aggregate over the
list, needs the temp table.

``_sqlite_variable_ceiling`` pins every vault connection to the historical 999
for one block, and first proves the pin took, so a site that went back to one
plain ``IN`` fails with "too many SQL variables" on any build. Each test
compares the result under the pin with the same call made without it, so a
site that was bounded by dropping rows would fail as well.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import pixlstash.routes.characters_faces as characters_faces_routes
import pixlstash.routes.pictures._misc as misc_routes
import pixlstash.routes.pictures._thumbnails as thumbnails_routes
import pixlstash.routes.reference_folders as reference_folders_routes
import pixlstash.routes.tags as tags_routes
from pixlstash.db_models.character import Character
from pixlstash.db_models.deleted_file_log import DeletedFileLog
from pixlstash.db_models.face import Face
from pixlstash.db_models.operation import Operation
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.picture_likeness import PictureLikeness
from pixlstash.db_models.picture_move import PictureMove
from pixlstash.db_models.picture_project import PictureProjectMember
from pixlstash.db_models.picture_set import PictureSet, PictureSetMember
from pixlstash.db_models.picture_stack import PictureStack
from pixlstash.db_models.project import Project
from pixlstash.db_models.reference_folder import ReferenceFolder, ReferenceFolderStatus
from pixlstash.db_models.tag import Tag
from pixlstash.routes.pictures._listing import LIST_MAX_IDS
from pixlstash.scoring.character_likeness import (
    count_pictures_by_character_likeness,
    find_pictures_by_character_likeness,
    find_pictures_by_character_likeness_sql,
)
from pixlstash.server import Server
from pixlstash.services.layout_move_service import claim_own_moves
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask
from pixlstash.utils.query.predicate_filter import MAX_FILTER_TERMS
from pixlstash.utils.service.picture_stats import clear_stats_cache
from tests.test_export_by_ids_api import VARIABLE_CEILING, _sqlite_variable_ceiling
from sqlmodel import delete, select
from tests.utils import API_PREFIX

# Pictures with one unassigned face each: past the pinned ceiling by enough
# that a chunk of 900 cannot hold them in one statement.
POOL_SIZE = VARIABLE_CEILING + 150
# The first STACKED pool pictures form stacks of two, so the stack-leader
# clause has members to collapse.
STACKED = 20
REFERENCE_FACES = 3
PAGE = 40
# Every pool picture carries this tag, so the tag counts cover the whole pool.
POOL_TAG = "idlist-pool-tag"
# The detection and batch-likeness routes cap their lists at 1000 ids, which
# is still past the historical limit.
ROUTE_CAP = 1000


def _features(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.normal(size=8).astype(np.float32).tobytes()


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
        # The pictures have no files, so the background stages would mark
        # their quality failed and then recompute their likeness, deleting the
        # seeded likeness rows while the tests read them. Nothing here needs a
        # worker: every row a test reads is seeded.
        server.vault._work_planner.stop()

        def seed(session):
            now = datetime.now(timezone.utc)
            reference = Character(name="Reference")
            session.add(reference)
            session.flush()

            def picture(name):
                pic = Picture(
                    file_path=f"/nonexistent/idlist-{name}.png",
                    format="png",
                    width=64,
                    height=64,
                    pixel_sha=f"idlist-{name}",
                    imported_at=now,
                )
                session.add(pic)
                return pic

            refs = [picture(f"ref-{n}") for n in range(REFERENCE_FACES)]
            pool = [picture(f"pool-{n}") for n in range(POOL_SIZE)]
            stacks = [PictureStack(name=f"stack-{n}") for n in range(STACKED // 2)]
            session.add_all(stacks)
            session.flush()
            for n, pic in enumerate(pool[:STACKED]):
                pic.stack_id = stacks[n // 2].id
                pic.stack_position = n % 2
            for n, pic in enumerate(refs):
                session.add(
                    Face(
                        picture_id=pic.id,
                        frame_index=0,
                        face_index=0,
                        character_id=reference.id,
                        bbox=[0, 0, 10, 10],
                        features=_features(n),
                    )
                )
            for pic in pool:
                session.add(Tag(picture_id=pic.id, tag=POOL_TAG))
            for n, pic in enumerate(pool):
                session.add(
                    Face(
                        picture_id=pic.id,
                        frame_index=0,
                        face_index=0,
                        character_id=None,
                        bbox=[1, 2, 11, 12],
                        features=_features(1000 + n),
                    )
                )
            session.commit()
            return int(reference.id), sorted(int(pic.id) for pic in pool)

        reference_id, pool_ids = server.vault.db.run_task(seed)
        assert len(pool_ids) == POOL_SIZE
        yield server, owner, reference_id, pool_ids
    finally:
        server.close()
        temp_dir.cleanup()


def test_thumbnail_batch_reads_faces_past_the_ceiling(env, monkeypatch):
    """``POST /pictures/thumbnails`` returns every picture's faces for a long list.

    The body's ``ids`` has no cap, and the grid sends a whole page or a
    select-all through it. The penalised-tag read is a separate site (it binds
    the raw request list), so it is switched off here: this test is about the
    face read in ``read_pictures_and_faces``.
    """
    server, owner, _reference_id, pool_ids = env
    monkeypatch.setattr(
        thumbnails_routes,
        "get_smart_score_penalised_tags_from_request",
        lambda _server, _request: {},
    )
    with _sqlite_variable_ceiling(server):
        r = owner.post(f"{API_PREFIX}/pictures/thumbnails", json={"ids": pool_ids})
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(int(pid) for pid in body) == pool_ids
    assert all(len(entry["faces"]) == 1 for entry in body.values()), (
        "a picture lost its face overlay"
    )


@pytest.mark.parametrize("stack_leaders_only", [False, True])
def test_likeness_listing_with_a_long_candidate_list(env, stack_leaders_only):
    """The SQL likeness sort keeps its order and page over a long candidate list.

    The candidates (a set's members, a face cluster, a share's scope) have no
    cap. Every unassigned face is in the pool, so naming the whole pool as the
    candidates must give exactly the page the unrestricted query gives.
    """
    server, _owner, reference_id, pool_ids = env

    def page(candidate_ids):
        rows = find_pictures_by_character_likeness_sql(
            server,
            "UNASSIGNED",
            reference_id,
            offset=PAGE,
            limit=PAGE,
            descending=True,
            candidate_ids=candidate_ids,
            stack_leaders_only=stack_leaders_only,
        )
        return [row["id"] for row in rows]

    expected = page(None)
    assert len(expected) == PAGE
    with _sqlite_variable_ceiling(server):
        assert page(pool_ids) == expected


@pytest.mark.parametrize("stack_leaders_only", [False, True])
def test_likeness_count_with_a_long_candidate_list(env, stack_leaders_only):
    """The likeness count over a long candidate list matches the unrestricted one."""
    server, _owner, _reference_id, pool_ids = env
    expected = count_pictures_by_character_likeness(
        server, "UNASSIGNED", stack_leaders_only=stack_leaders_only
    )
    leaders = POOL_SIZE - STACKED // 2 if stack_leaders_only else POOL_SIZE
    assert expected == leaders
    with _sqlite_variable_ceiling(server):
        assert (
            count_pictures_by_character_likeness(
                server,
                "UNASSIGNED",
                candidate_ids=pool_ids,
                stack_leaders_only=stack_leaders_only,
            )
            == expected
        )


def test_python_likeness_sort_with_a_long_candidate_list(env):
    """The Python likeness sort reads long candidate and unassigned lists.

    It reads the candidates' faces, then keeps the pictures with no assigned
    face; with every pool picture a candidate both lists are past the ceiling.
    """
    server, _owner, reference_id, pool_ids = env

    def page(candidate_ids):
        rows = find_pictures_by_character_likeness(
            server,
            "UNASSIGNED",
            reference_id,
            PAGE,
            PAGE,
            True,
            candidate_ids=candidate_ids,
        )
        return [row["id"] for row in rows]

    expected = page(None)
    assert len(expected) == PAGE
    with _sqlite_variable_ceiling(server):
        assert page(pool_ids) == expected


def test_character_membership_past_the_ceiling(env):
    """``POST /characters/membership`` reports every picture of a select-all."""
    server, owner, _reference_id, pool_ids = env
    with _sqlite_variable_ceiling(server):
        r = owner.post(
            f"{API_PREFIX}/characters/membership", json={"picture_ids": pool_ids}
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pictures_with_faces"] == pool_ids
    assert body["character_assignments"] == {}


def test_project_membership_past_the_ceiling(env):
    """``POST /projects/membership`` reports every picture of a select-all.

    The last pool picture, in the read's last chunk, is in a project, so a read
    that stopped short would call it unassigned.
    """
    server, owner, _reference_id, pool_ids = env
    last = pool_ids[-1]

    def seed(session):
        project = Project(name="idlist-membership-project")
        session.add(project)
        session.flush()
        session.add(PictureProjectMember(picture_id=last, project_id=project.id))
        session.commit()
        return int(project.id)

    def drop(session, project_id):
        session.exec(
            delete(PictureProjectMember).where(
                PictureProjectMember.project_id == project_id
            )
        )
        session.exec(delete(Project).where(Project.id == project_id))
        session.commit()

    project_id = server.vault.db.run_task(seed)
    try:
        with _sqlite_variable_ceiling(server):
            r = owner.post(
                f"{API_PREFIX}/projects/membership", json={"picture_ids": pool_ids}
            )
    finally:
        server.vault.db.run_task(drop, project_id)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unassigned_picture_ids"] == pool_ids[:-1]
    assert body["project_assignments"] == {str(project_id): [last]}


def test_tag_counts_over_a_long_share_scope(env, monkeypatch):
    """``GET /tags`` counts tags over a share scope past the ceiling.

    A share's scope (a set, a project) scales with the library, and the counts
    are grouped and ordered over all of it, so the scope comes from a temp
    table. The scope is the whole pool here, so the counts must equal the
    unscoped ones for the pool tag.
    """
    server, owner, _reference_id, pool_ids = env
    monkeypatch.setattr(
        tags_routes,
        "fetch_scope_allowed_picture_ids",
        lambda _server, _request: set(pool_ids),
    )
    with _sqlite_variable_ceiling(server):
        r = owner.get(f"{API_PREFIX}/tags")
    assert r.status_code == 200, r.text
    assert r.json() == [{"tag": POOL_TAG, "count": POOL_SIZE}]


def test_detect_reads_the_whole_capped_list(env, monkeypatch):
    """``POST /pictures/detect`` queues every picture of a full 1000-id batch."""
    server, owner, _reference_id, pool_ids = env
    queued = []
    monkeypatch.setattr(server.vault, "_engine", object(), raising=False)

    def submit_task(task):
        queued.append(task)
        return "test-detect-task"

    monkeypatch.setattr(server.vault, "submit_task", submit_task)
    batch = pool_ids[:ROUTE_CAP]
    with _sqlite_variable_ceiling(server):
        r = owner.post(f"{API_PREFIX}/pictures/detect", json={"picture_ids": batch})
    assert r.status_code == 200, r.text
    assert sorted(r.json()["picture_ids"]) == batch
    assert len(queued) == 1


def test_batch_likeness_past_the_ceiling(env):
    """``POST /pictures/character_likeness/batch`` answers a full 1000-id batch.

    The batch must give the same answer for every id under the pin as without
    it; each pool picture has one unassigned face and no set, so each is
    eligible and ready.
    """
    server, owner, reference_id, pool_ids = env
    body = {
        "reference_character_id": reference_id,
        "picture_ids": pool_ids[:ROUTE_CAP],
        "character_id": "UNASSIGNED",
    }
    unpinned = owner.post(f"{API_PREFIX}/pictures/character_likeness/batch", json=body)
    assert unpinned.status_code == 200, unpinned.text
    results = unpinned.json()["results"]
    assert [entry["picture_id"] for entry in results] == pool_ids[:ROUTE_CAP]
    assert all(entry["eligible"] and entry["ready"] for entry in results)
    with _sqlite_variable_ceiling(server):
        pinned = owner.post(
            f"{API_PREFIX}/pictures/character_likeness/batch", json=body
        )
    assert pinned.status_code == 200, pinned.text
    assert pinned.json() == unpinned.json()


@pytest.fixture(scope="module")
def reference_pool(env, tmp_path_factory):
    """POOL_SIZE tagged pictures indexed in a reference folder that is offline.

    The folder's root does not exist, so the scanner leaves it in
    ``mount_error`` and never touches its pictures. A second, empty folder is
    an active move destination.
    """
    server, _owner, _reference_id, _pool_ids = env
    base = tmp_path_factory.mktemp("idlist-reference")
    root = str(base / "offline-reference-root")
    destination = str(base / "move-destination")
    os.makedirs(destination)

    def seed(session):
        now = datetime.now(timezone.utc)
        source = ReferenceFolder(
            folder=root,
            status=ReferenceFolderStatus.MOUNT_ERROR,
            last_scanned=now.timestamp(),
        )
        target = ReferenceFolder(
            folder=destination,
            status=ReferenceFolderStatus.ACTIVE,
            last_scanned=now.timestamp(),
        )
        session.add_all([source, target])
        session.flush()
        pictures = [
            Picture(
                file_path=os.path.join(root, f"idlist-ref-{n}.png"),
                format="png",
                width=64,
                height=64,
                pixel_sha=f"idlist-ref-{n}",
                imported_at=now,
                reference_folder_id=source.id,
            )
            for n in range(POOL_SIZE)
        ]
        session.add_all(pictures)
        session.flush()
        for pic in pictures:
            session.add(Tag(picture_id=pic.id, tag=POOL_TAG))
        session.commit()
        return (
            int(source.id),
            int(target.id),
            sorted(int(pic.id) for pic in pictures),
        )

    return server.vault.db.run_task(seed)


def test_reference_metadata_export_reads_tags_past_the_ceiling(
    env, reference_pool, monkeypatch
):
    """The metadata export reads the tags of a folder larger than the ceiling.

    The folder is offline, so its files are reported present and the sidecar
    writer only records what it is handed: every picture must reach it with its
    tag, which a tag read that stopped short would leave out for the last ones.
    """
    server, owner, _reference_id, _pool_ids = env
    source_id, _target_id, _ref_ids = reference_pool
    root = server.vault.db.run_task(
        lambda session: session.get(ReferenceFolder, source_id).folder
    )
    real_isfile = os.path.isfile
    monkeypatch.setattr(
        os.path,
        "isfile",
        lambda path: os.path.dirname(path) == root or real_isfile(path),
    )
    written: dict[str, str] = {}

    def record_sidecar(target, text):
        # No mtime: the export leaves the picture rows as they were.
        written[target] = text
        return None

    monkeypatch.setattr(reference_folders_routes, "write_sidecar", record_sidecar)
    with _sqlite_variable_ceiling(server):
        r = owner.post(
            f"{API_PREFIX}/reference-folders/{source_id}/metadata/export",
            json={"types": ["tags"]},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skipped_count"] == 0
    assert body["tags_count"] == 0
    assert len(written) == POOL_SIZE
    assert set(written.values()) == {POOL_TAG}


def test_reference_move_reads_a_long_selection(env, reference_pool):
    """``move-pictures`` reads every picture of a selection past the ceiling.

    The files are absent, so the move is refused with one failure per picture;
    a picture the read dropped would be reported ``picture_not_found``.
    """
    server, owner, _reference_id, _pool_ids = env
    _source_id, target_id, ref_ids = reference_pool
    with _sqlite_variable_ceiling(server):
        r = owner.post(
            f"{API_PREFIX}/reference-folders/{target_id}/move-pictures",
            json={"picture_ids": ref_ids},
        )
    assert r.status_code == 400, r.text
    failures = r.json()["detail"]["failures"]
    assert sorted(entry["picture_id"] for entry in failures) == ref_ids
    assert {entry["reason"] for entry in failures} == {"source_file_missing"}


# Stacks of two in the crowd: one stack per id past the pinned ceiling, so a
# read over the stack ids alone, or over the leaders alone, overflows it.
CROWD_STACKS = VARIABLE_CEILING + 1
# Loose crowd pictures that also sit in the crowd set, carrying the hidden tag.
HIDDEN_MEMBERS = 5
HIDDEN_TAG = "idlist-hidden"


@pytest.fixture(scope="module")
def crowd(env):
    """A character, a project and a set that each scale past the ceiling.

    ``CROWD_STACKS`` stacks of two and as many loose pictures, every one with
    a face assigned to the "Crowd" character. The stacked pictures are the
    members of one project and of one set; the first ``HIDDEN_MEMBERS`` loose
    pictures join the set too and carry ``HIDDEN_TAG``. Each stack leader is
    paired with one loose picture by a picture likeness, so every likeness
    group holds one stack and one loose picture.
    """
    server, _owner, _reference_id, _pool_ids = env

    def seed(session):
        now = datetime.now(timezone.utc)
        character = Character(name="Crowd")
        project = Project(name="idlist-crowd-project")
        picture_set = PictureSet(name="idlist-crowd-set")
        session.add_all([character, project, picture_set])
        stacks = [PictureStack(name=f"crowd-stack-{n}") for n in range(CROWD_STACKS)]
        session.add_all(stacks)
        session.flush()

        def picture(name, **fields):
            pic = Picture(
                file_path=f"/nonexistent/idlist-crowd-{name}.png",
                format="png",
                width=64,
                height=64,
                pixel_sha=f"idlist-crowd-{name}",
                imported_at=now,
                **fields,
            )
            session.add(pic)
            return pic

        leaders = []
        stacked = []
        for n, stack in enumerate(stacks):
            for position in range(2):
                pic = picture(
                    f"stack-{n}-{position}",
                    stack_id=stack.id,
                    stack_position=position,
                )
                stacked.append(pic)
                if position == 0:
                    leaders.append(pic)
        loose = [picture(f"loose-{n}") for n in range(CROWD_STACKS)]
        session.flush()
        for pic in stacked + loose:
            session.add(
                Face(
                    picture_id=pic.id,
                    frame_index=0,
                    face_index=0,
                    character_id=character.id,
                    bbox=[0, 0, 10, 10],
                )
            )
        for pic in stacked:
            session.add(PictureProjectMember(picture_id=pic.id, project_id=project.id))
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=pic.id))
        for pic in loose[:HIDDEN_MEMBERS]:
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=pic.id))
            session.add(Tag(picture_id=pic.id, tag=HIDDEN_TAG))
        for leader, partner in zip(leaders, loose):
            low, high = sorted((int(leader.id), int(partner.id)))
            session.add(
                PictureLikeness(
                    picture_id_a=low, picture_id_b=high, likeness=0.9, metric="test"
                )
            )
        session.commit()
        return {
            "character_id": int(character.id),
            "project_id": int(project.id),
            "set_id": int(picture_set.id),
            "leaders": sorted(int(pic.id) for pic in leaders),
            "stacked": sorted(int(pic.id) for pic in stacked),
            "loose": sorted(int(pic.id) for pic in loose),
            "hidden": sorted(int(pic.id) for pic in loose[:HIDDEN_MEMBERS]),
        }

    return server.vault.db.run_task(seed)


def _pinned_matches_unpinned(server, call):
    """Run *call* without and then under the pin; return the unpinned body."""
    unpinned = call()
    assert unpinned.status_code == 200, unpinned.text
    with _sqlite_variable_ceiling(server):
        pinned = call()
    assert pinned.status_code == 200, pinned.text
    assert pinned.json() == unpinned.json()
    return unpinned.json()


def test_set_membership_of_a_select_all_past_the_ceiling(env, crowd):
    """``POST /picture_sets/membership`` expands a select-all of stack leaders.

    The leaders, their stacks and the expanded pictures each pass the ceiling.
    """
    server, owner, _reference_id, _pool_ids = env
    with _sqlite_variable_ceiling(server):
        r = owner.post(
            f"{API_PREFIX}/picture_sets/membership",
            json={"picture_ids": crowd["leaders"]},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {str(crowd["set_id"])}
    assert sorted(body[str(crowd["set_id"])]) == crowd["stacked"]


def test_set_listing_with_hidden_tags_past_the_ceiling(env, crowd):
    """``GET /picture_sets/{id}`` hides, expands and counts a whole set.

    The set's members, its stacks and its stack leaders each pass the ceiling;
    the hidden members must be gone and every leader counts its stack of two.
    """
    server, owner, _reference_id, _pool_ids = env
    r = owner.patch(f"{API_PREFIX}/users/me/config", json={"hidden_tags": [HIDDEN_TAG]})
    assert r.status_code == 200, r.text
    try:
        body = _pinned_matches_unpinned(
            server,
            lambda: owner.get(
                f"{API_PREFIX}/picture_sets/{crowd['set_id']}",
                params={"apply_tag_filter": "true"},
            ),
        )
    finally:
        r = owner.patch(f"{API_PREFIX}/users/me/config", json={"hidden_tags": []})
        assert r.status_code == 200, r.text
    pictures = body["pictures"]
    assert sorted(pic["id"] for pic in pictures) == crowd["leaders"]
    assert {pic["stack_count"] for pic in pictures} == {2}


def test_set_members_with_stacks_past_the_ceiling(env, crowd):
    """``GET /picture_sets/{id}/members?expand_stacks=true`` lists every member.

    A second set holds only the stack leaders and the hidden pictures, so each
    other stacked picture is reached through its stack alone: a stack read, or
    a sibling read, that stopped short would leave the last stacks unexpanded.
    """
    server, owner, _reference_id, _pool_ids = env
    with _sqlite_variable_ceiling(server):
        r = owner.get(
            f"{API_PREFIX}/picture_sets/{crowd['set_id']}/members",
            params={"expand_stacks": "true"},
        )
    assert r.status_code == 200, r.text
    assert sorted(r.json()["picture_ids"]) == sorted(crowd["stacked"] + crowd["hidden"])

    def seed(session):
        picture_set = PictureSet(name="idlist-leaders-set")
        session.add(picture_set)
        session.flush()
        for picture_id in crowd["leaders"] + crowd["hidden"]:
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.commit()
        return int(picture_set.id)

    def drop(session, set_id):
        session.exec(delete(PictureSetMember).where(PictureSetMember.set_id == set_id))
        session.exec(delete(PictureSet).where(PictureSet.id == set_id))
        session.commit()

    leaders_set_id = server.vault.db.run_task(seed)
    try:
        with _sqlite_variable_ceiling(server):
            r = owner.get(
                f"{API_PREFIX}/picture_sets/{leaders_set_id}/members",
                params={"expand_stacks": "true"},
            )
    finally:
        server.vault.db.run_task(drop, leaders_set_id)
    assert r.status_code == 200, r.text
    assert sorted(r.json()["picture_ids"]) == sorted(crowd["stacked"] + crowd["hidden"])


@pytest.fixture(scope="module")
def big_stack(env):
    """POOL_SIZE pictures in one stack, and an empty set to add it to."""
    server, _owner, _reference_id, _pool_ids = env

    def seed(session):
        now = datetime.now(timezone.utc)
        stack = PictureStack(name="idlist-big-stack")
        picture_set = PictureSet(name="idlist-big-stack-set")
        session.add_all([stack, picture_set])
        session.flush()
        pictures = [
            Picture(
                file_path=f"/nonexistent/idlist-big-{n}.png",
                format="png",
                width=64,
                height=64,
                pixel_sha=f"idlist-big-{n}",
                imported_at=now,
                stack_id=stack.id,
                stack_position=n,
            )
            for n in range(POOL_SIZE)
        ]
        session.add_all(pictures)
        session.commit()
        return int(picture_set.id), sorted(int(pic.id) for pic in pictures)

    return server.vault.db.run_task(seed)


def test_set_add_and_remove_a_stack_past_the_ceiling(env, big_stack):
    """Adding or removing one stacked picture moves its whole stack."""
    server, owner, _reference_id, _pool_ids = env
    set_id, stack_ids = big_stack

    def members():
        r = owner.get(f"{API_PREFIX}/picture_sets/{set_id}/members")
        assert r.status_code == 200, r.text
        return sorted(r.json()["picture_ids"])

    with _sqlite_variable_ceiling(server):
        r = owner.post(f"{API_PREFIX}/picture_sets/{set_id}/members/{stack_ids[0]}")
        assert r.status_code == 200, r.text
        assert members() == stack_ids
        r = owner.delete(f"{API_PREFIX}/picture_sets/{set_id}/members/{stack_ids[0]}")
        assert r.status_code == 200, r.text
        assert members() == []


def _listing(owner, **params):
    return lambda: owner.get(
        f"{API_PREFIX}/pictures", params={"fields": "grid", **params}
    )


def test_character_listing_unassigned_project_past_the_ceiling(env, crowd):
    """``GET /pictures?character_id=..&project_id=UNASSIGNED`` narrows a crowd."""
    server, owner, _reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        _listing(owner, character_id=crowd["character_id"], project_id="UNASSIGNED"),
    )
    assert sorted(pic["id"] for pic in body) == crowd["loose"]


def test_character_listing_in_a_project_past_the_ceiling(env, crowd):
    """``GET /pictures?character_id=..&project_id=P`` lists a project's leaders.

    The character's pictures, the leaders and their stacks each pass the
    ceiling, and each leader counts its stack of two.
    """
    server, owner, _reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        _listing(
            owner, character_id=crowd["character_id"], project_id=crowd["project_id"]
        ),
    )
    assert sorted(pic["id"] for pic in body) == crowd["leaders"]
    assert {pic["stack_count"] for pic in body} == {2}


def test_characters_listing_unassigned_project_past_the_ceiling(env, crowd):
    """``character_ids=`` with ``project_id=UNASSIGNED`` narrows a crowd."""
    server, owner, reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        _listing(
            owner,
            character_ids=[crowd["character_id"], reference_id],
            project_id="UNASSIGNED",
        ),
    )
    assert crowd["loose"] == sorted(
        pic["id"] for pic in body if pic["id"] in set(crowd["loose"])
    )


def test_characters_listing_in_a_project_past_the_ceiling(env, crowd):
    """``character_ids=`` with ``project_id=P`` lists a project's leaders.

    A second project holds only the newest crowd picture, which sorts into the
    membership read's last chunk: a read that stopped short would list nothing.
    """
    server, owner, reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        _listing(
            owner,
            character_ids=[crowd["character_id"], reference_id],
            project_id=crowd["project_id"],
        ),
    )
    assert sorted(pic["id"] for pic in body) == crowd["leaders"]

    newest = max(crowd["loose"])

    def seed(session):
        project = Project(name="idlist-newest-project")
        session.add(project)
        session.flush()
        session.add(PictureProjectMember(picture_id=newest, project_id=project.id))
        session.commit()
        return int(project.id)

    def drop(session, project_id):
        session.exec(
            delete(PictureProjectMember).where(
                PictureProjectMember.project_id == project_id
            )
        )
        session.exec(delete(Project).where(Project.id == project_id))
        session.commit()

    project_id = server.vault.db.run_task(seed)
    try:
        body = _pinned_matches_unpinned(
            server,
            _listing(
                owner,
                character_ids=[crowd["character_id"], reference_id],
                project_id=project_id,
            ),
        )
    finally:
        server.vault.db.run_task(drop, project_id)
    assert [pic["id"] for pic in body] == [newest]


def test_likeness_groups_of_a_crowd_past_the_ceiling(env, crowd):
    """``GET /pictures/likeness-groups`` groups a whole character.

    The character's pictures, the grouped pictures and their stacks each pass
    the ceiling, and ``unscored`` sends every grouped picture through the
    post-filter. Each group is one loose picture and one stack of two.
    """
    server, owner, _reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.get(
            f"{API_PREFIX}/pictures/likeness-groups",
            params={
                "character_id": crowd["character_id"],
                "threshold": 0.5,
                "unscored": "true",
            },
        ),
    )
    assert sorted(pic["id"] for pic in body) == sorted(
        crowd["stacked"] + crowd["loose"]
    )
    assert len({pic["stack_index"] for pic in body}) == CROWD_STACKS


# Values in one ``format`` or ``hidden_tag`` list: past the pinned ceiling.
# Neither list has a cap, and none of the made-up values names anything.
LONG_TERM_LIST = VARIABLE_CEILING + 1


def test_likeness_groups_read_a_long_format_list(env, crowd):
    """``GET /pictures/likeness-groups?format=..`` keeps every grouped picture.

    One real format among made-up ones: the groups are the crowd's own.
    """
    server, owner, _reference_id, _pool_ids = env
    formats = ["png"] + [f"example-format-{n}" for n in range(LONG_TERM_LIST - 1)]
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.get(
            f"{API_PREFIX}/pictures/likeness-groups",
            params={
                "character_id": crowd["character_id"],
                "threshold": 0.5,
                "format": formats,
            },
        ),
    )
    assert sorted(pic["id"] for pic in body) == sorted(
        crowd["stacked"] + crowd["loose"]
    )


@pytest.mark.parametrize("route", ["/pictures", "/pictures/count", "/pictures/stats"])
def test_a_long_hidden_tag_list_hides_its_tags(env, crowd, route):
    """The owner's hidden tags past the ceiling hide what the one real tag hides.

    ``GET /pictures``, its count and stats over the crowd set, with
    ``apply_tag_filter``: the made-up names hide nothing more, and the hidden
    members are gone.
    """
    server, owner, _reference_id, _pool_ids = env
    params = {"set_ids": [crowd["set_id"]], "apply_tag_filter": "true"}
    if route == "/pictures":
        params["fields"] = "grid"

    def answer(hidden_tags):
        r = owner.patch(
            f"{API_PREFIX}/users/me/config", json={"hidden_tags": hidden_tags}
        )
        assert r.status_code == 200, r.text

        def read():
            clear_stats_cache()
            return owner.get(f"{API_PREFIX}{route}", params=params)

        try:
            return _pinned_matches_unpinned(server, read)
        finally:
            clear_stats_cache()
            r = owner.patch(f"{API_PREFIX}/users/me/config", json={"hidden_tags": []})
            assert r.status_code == 200, r.text

    shown = answer([HIDDEN_TAG])
    long_list = [HIDDEN_TAG] + [f"example-tag-{n}" for n in range(LONG_TERM_LIST - 1)]
    assert answer(long_list) == shown
    if route == "/pictures":
        assert not set(crowd["hidden"]) & {pic["id"] for pic in shown}
    else:
        total = shown["count"] if route == "/pictures/count" else shown["total"]
        assert total == len(crowd["stacked"])


def test_thumbnail_batch_reads_penalised_tags_past_the_ceiling(env, monkeypatch):
    """``POST /pictures/thumbnails`` marks the penalised tags of a long batch.

    The penalised-tag read binds the body's ids, which have no cap. Every pool
    picture carries ``POOL_TAG``, so every entry must list it.
    """
    server, owner, _reference_id, pool_ids = env
    monkeypatch.setattr(
        thumbnails_routes,
        "get_smart_score_penalised_tags_from_request",
        lambda _server, _request: {POOL_TAG: 3},
    )
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.post(f"{API_PREFIX}/pictures/thumbnails", json={"ids": pool_ids}),
    )
    assert sorted(int(pid) for pid in body) == pool_ids
    assert all(entry["penalised_tags"] == [POOL_TAG] for entry in body.values())


# Pictures that each show a different person, so one thumbnail batch names
# more people than the pinned ceiling binds.
CAST_SIZE = VARIABLE_CEILING + 50


@pytest.fixture(scope="module")
def cast(env):
    """CAST_SIZE pictures, each with one face assigned to a character of its own.

    Returns ``{picture_id: character name}``.
    """
    server, _owner, _reference_id, _pool_ids = env

    def seed(session):
        now = datetime.now(timezone.utc)
        characters = [Character(name=f"idlist-cast-{n}") for n in range(CAST_SIZE)]
        pictures = [
            Picture(
                file_path=f"/nonexistent/idlist-cast-{n}.png",
                format="png",
                width=64,
                height=64,
                pixel_sha=f"idlist-cast-{n}",
                imported_at=now,
            )
            for n in range(CAST_SIZE)
        ]
        session.add_all(characters + pictures)
        session.flush()
        for character, pic in zip(characters, pictures):
            session.add(
                Face(
                    picture_id=pic.id,
                    frame_index=0,
                    face_index=0,
                    character_id=character.id,
                    bbox=[0, 0, 10, 10],
                )
            )
        session.commit()
        return {
            int(pic.id): character.name for character, pic in zip(characters, pictures)
        }

    return server.vault.db.run_task(seed)


def test_thumbnail_batch_names_the_people_of_a_long_batch(env, cast, monkeypatch):
    """``POST /pictures/thumbnails`` names the character on every face.

    The names are read for every character on the batch's faces, as many as
    the batch has pictures here. The penalised-tag read is switched off: this
    test is about the name read.
    """
    server, owner, _reference_id, _pool_ids = env
    monkeypatch.setattr(
        thumbnails_routes,
        "get_smart_score_penalised_tags_from_request",
        lambda _server, _request: {},
    )
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.post(
            f"{API_PREFIX}/pictures/thumbnails", json={"ids": sorted(cast)}
        ),
    )
    assert {
        int(pid): [face["character_name"] for face in entry["faces"]]
        for pid, entry in body.items()
    } == {pid: [name] for pid, name in cast.items()}


def test_search_reads_a_full_id_filter(env, monkeypatch):
    """``GET /pictures/search`` narrows to a full ``id`` filter.

    The listing caps ``id`` at ``LIST_MAX_IDS``, past the historical limit,
    and the search reads the filter beside its other clauses. Every requested
    picture must reach the ranking as a candidate.
    """
    assert LIST_MAX_IDS > VARIABLE_CEILING
    server, owner, _reference_id, pool_ids = env
    batch = pool_ids[:LIST_MAX_IDS]
    candidates = []
    real_semantic_search = Picture.semantic_search

    def spy_semantic_search(cls, session, *args, **kwargs):
        candidates.append(sorted(kwargs["candidate_ids"]))
        return real_semantic_search(session, *args, **kwargs)

    monkeypatch.setattr(
        server.vault,
        "generate_text_embedding",
        lambda _query: np.full(384, 0.5, dtype=np.float32),
    )
    monkeypatch.setattr(
        server.vault,
        "generate_clip_text_embedding",
        lambda _query: np.full(512, 0.25, dtype=np.float32),
    )
    monkeypatch.setattr(Picture, "semantic_search", classmethod(spy_semantic_search))
    params = [("query", POOL_TAG), ("threshold", 0.0)]
    params += [("id", pid) for pid in batch]
    _pinned_matches_unpinned(
        server, lambda: owner.get(f"{API_PREFIX}/pictures/search", params=params)
    )
    assert candidates == [batch, batch]


def _pool_face_ids(server, pool_ids):
    pool = set(pool_ids)

    def read(session):
        return sorted(
            int(face_id)
            for face_id, picture_id in session.exec(select(Face.id, Face.picture_id))
            if picture_id in pool
        )

    return server.vault.db.run_immediate_read_task(read)


def test_face_assignment_scope_reads_every_face(env, monkeypatch):
    """A scoped face assignment resolves every face of a long ``face_ids`` list.

    The scope check reads the picture behind each face. One picture of the
    pool is outside the scope, and its face is the last of the list: the
    request is refused only if every face was resolved.
    """
    server, owner, reference_id, pool_ids = env
    face_ids = _pool_face_ids(server, pool_ids)
    assert len(face_ids) == POOL_SIZE
    monkeypatch.setattr(
        characters_faces_routes,
        "fetch_scope_allowed_picture_ids",
        lambda _server, _request: set(pool_ids[:-1]),
    )

    def call():
        return owner.post(
            f"{API_PREFIX}/characters/{reference_id}/faces",
            json={"face_ids": face_ids},
        )

    assert call().status_code == 403
    with _sqlite_variable_ceiling(server):
        refused = call()
    assert refused.status_code == 403, refused.text


def test_face_assign_and_unassign_record_every_picture(env, monkeypatch):
    """Assigning and unassigning a long ``face_ids`` list records every picture.

    The request names faces, not pictures; the scope check and the operation
    log each resolve the pictures behind them. A picture the log missed would
    be left out of the undo.
    """
    server, owner, _reference_id, pool_ids = env
    face_ids = _pool_face_ids(server, pool_ids)
    monkeypatch.setattr(
        characters_faces_routes,
        "fetch_scope_allowed_picture_ids",
        lambda _server, _request: set(pool_ids),
    )

    def seed(session):
        character = Character(name="idlist-assignee")
        session.add(character)
        session.commit()
        return int(character.id)

    character_id = server.vault.db.run_task(seed)

    def assigned():
        return server.vault.db.run_immediate_read_task(
            lambda session: sorted(
                int(face_id)
                for face_id in session.exec(
                    select(Face.id).where(Face.character_id == character_id)
                )
            )
        )

    def logged(op_type):
        def read(session):
            operation = session.exec(
                select(Operation)
                .where(Operation.op_type == op_type)
                .order_by(Operation.id.desc())
            ).first()
            return sorted(json.loads(operation.target_ids))

        return server.vault.db.run_immediate_read_task(read)

    url = f"{API_PREFIX}/characters/{character_id}/faces"
    with _sqlite_variable_ceiling(server):
        r = owner.post(url, json={"face_ids": face_ids})
    assert r.status_code == 200, r.text
    assert assigned() == face_ids
    assert logged("characters.assign") == pool_ids

    with _sqlite_variable_ceiling(server):
        r = owner.request("DELETE", url, json={"face_ids": face_ids})
    assert r.status_code == 200, r.text
    assert assigned() == []
    assert logged("characters.unassign") == pool_ids


def test_training_suggestion_for_a_character_of_many_stacks(env, crowd):
    """``GET /training/character_set_suggestion`` keeps one cover per stack.

    The crowd spans ``CROWD_STACKS`` stacks, past the ceiling. Training keeps
    each stack's cover only, so the candidates are the leaders and the loose
    pictures, not the stacks' second members.
    """
    server, owner, _reference_id, _pool_ids = env
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.get(
            f"{API_PREFIX}/training/character_set_suggestion",
            params={"character_id": crowd["character_id"]},
        ),
    )
    assert body["stills"]["candidates"] == len(crowd["leaders"]) + len(crowd["loose"])


def test_move_journal_claim_past_the_ceiling(env):
    """The scan's move-journal claim reads every pair of a large folder rename.

    One journal row per pair, past the ceiling, and the last pair journaled
    twice: the claim consumes that pair's older row and leaves the newer one.
    The claim is rolled back each time, so both runs see the same journal.
    """
    server, _owner, _reference_id, _pool_ids = env
    prefix = "/nonexistent/idlist-move-"
    pairs = [(f"{prefix}{n}.png", f"{prefix}moved/{n}.png") for n in range(POOL_SIZE)]

    def seed(session):
        rows = [PictureMove(old_path=old, new_path=new) for old, new in pairs]
        rows.append(PictureMove(old_path=pairs[-1][0], new_path=pairs[-1][1]))
        session.add_all(rows)
        session.commit()
        return [int(row.id) for row in rows]

    row_ids = server.vault.db.run_task(seed)
    unjournaled = (f"{prefix}unjournaled.png", f"{prefix}moved/unjournaled.png")

    def claim(session):
        claimed = claim_own_moves(session, [*pairs, unjournaled])
        consumed = sorted(
            int(row_id)
            for row_id in session.exec(
                select(PictureMove.id).where(
                    PictureMove.consumed.is_(True),
                    PictureMove.old_path.startswith(prefix),
                )
            )
        )
        session.rollback()
        return claimed, consumed

    unpinned = server.vault.db.run_task(claim)
    with _sqlite_variable_ceiling(server):
        pinned = server.vault.db.run_task(claim)
    assert pinned == unpinned
    assert pinned == (set(pairs), row_ids[:-1])


def test_explicit_reimport_clears_the_ledger_past_the_ceiling(env, tmp_path):
    """Re-adding a folder clears the deletion-ledger row of every file in it.

    The scan deletes the ledger rows of the files it re-imports, and a re-added
    folder can hold more of them than the ceiling binds.
    """
    server, _owner, _reference_id, _pool_ids = env
    count = VARIABLE_CEILING + 1
    folder = tmp_path / "idlist-reimport"
    folder.mkdir()
    paths = []
    for n in range(count):
        path = str(folder / f"reimport-{n}.png")
        Image.new("RGB", (8, 8), color=(n % 256, n // 256, 90)).save(path)
        paths.append(path)
    shas = {DeletedFileLog.hash_path(path) for path in paths}

    def seed(session):
        now = datetime.now(timezone.utc)
        session.add_all(
            DeletedFileLog(path_sha=sha, deleted_at=now, file_removed=False)
            for sha in shas
        )
        reference_folder = ReferenceFolder(
            folder=str(folder),
            status=ReferenceFolderStatus.ACTIVE,
            pending_reimport=True,
        )
        session.add(reference_folder)
        session.commit()
        return int(reference_folder.id)

    folder_id = server.vault.db.run_task(seed)

    def ledger(session):
        return {
            sha for sha in session.exec(select(DeletedFileLog.path_sha)) if sha in shas
        }

    assert server.vault.db.run_immediate_read_task(ledger) == shas
    with _sqlite_variable_ceiling(server):
        result = ReferenceFolderScanTask(
            database=server.vault.db,
            folder_id=folder_id,
            folder_path=str(folder),
            resolved_path=str(folder),
        )._run_task()
    assert result["new_count"] == count
    assert server.vault.db.run_immediate_read_task(ledger) == set()


# Penalised names in one thumbnail batch: beside a full chunk of ids they pass
# the pinned ceiling, unless the names bind as one value.
PENALISED_IN_A_BATCH = 150


def test_thumbnail_batch_binds_a_long_penalised_list(env, monkeypatch):
    """``POST /pictures/thumbnails`` reads a long penalised list beside a chunk.

    The owner's penalised names have no cap and ride with every id chunk. Each
    pool picture carries ``POOL_TAG`` and none of the filler names, so every
    entry lists ``POOL_TAG`` alone.
    """
    server, owner, _reference_id, pool_ids = env
    penalised = {POOL_TAG: 3}
    penalised.update({f"idlist-filler-{n}": 1 for n in range(PENALISED_IN_A_BATCH - 1)})
    monkeypatch.setattr(
        thumbnails_routes,
        "get_smart_score_penalised_tags_from_request",
        lambda _server, _request: penalised,
    )
    body = _pinned_matches_unpinned(
        server,
        lambda: owner.post(f"{API_PREFIX}/pictures/thumbnails", json={"ids": pool_ids}),
    )
    assert sorted(int(pid) for pid in body) == pool_ids
    assert all(entry["penalised_tags"] == [POOL_TAG] for entry in body.values())


# Penalised names in one stats read: past the pinned ceiling on their own, so
# each clause that tests them fails unless it binds them as one value.
PENALISED_IN_STATS = VARIABLE_CEILING + 1
# A penalised tag spelled with capitals (the owner's list is lower-cased), and
# a tag that is not penalised, both on the last few pool pictures.
STATS_PENALISED_TAG = "Idlist-Pen-B"
STATS_PLAIN_TAG = "idlist-plain"
STATS_TAGGED = 7


@pytest.mark.parametrize(
    ("only_penalised", "pairs"),
    [
        # One penalised tag per pair is enough.
        (
            "true",
            [
                ((STATS_PENALISED_TAG, STATS_PLAIN_TAG), STATS_TAGGED),
                ((STATS_PENALISED_TAG, POOL_TAG), STATS_TAGGED),
                ((STATS_PLAIN_TAG, POOL_TAG), STATS_TAGGED),
            ],
        ),
        # Both tags of a pair must be penalised.
        ("both", [((STATS_PENALISED_TAG, POOL_TAG), STATS_TAGGED)]),
    ],
)
def test_stats_bind_a_long_penalised_list(env, monkeypatch, only_penalised, pairs):
    """``GET /pictures/stats?only_penalised=..`` reads a long penalised list.

    The top tags and both sides of the co-occurrence join each test the list,
    under a GROUP BY ... LIMIT that chunks could not add up. The cache key does
    not carry the list, so the cache is cleared before every read.
    """
    server, owner, _reference_id, pool_ids = env
    tagged = pool_ids[-STATS_TAGGED:]
    penalised = {STATS_PENALISED_TAG.lower(): 3, POOL_TAG: 3}
    penalised.update(
        {f"idlist-filler-{n}": 1 for n in range(PENALISED_IN_STATS - len(penalised))}
    )
    assert len(penalised) == PENALISED_IN_STATS
    monkeypatch.setattr(
        misc_routes,
        "get_smart_score_penalised_tags_from_request",
        lambda _server, _request: penalised,
    )

    def seed(session):
        for picture_id in tagged:
            session.add(Tag(picture_id=picture_id, tag=STATS_PENALISED_TAG))
            session.add(Tag(picture_id=picture_id, tag=STATS_PLAIN_TAG))
        session.commit()

    def drop(session):
        session.exec(
            delete(Tag).where(
                Tag.picture_id.in_(tagged),
                Tag.tag.in_([STATS_PENALISED_TAG, STATS_PLAIN_TAG]),
            )
        )
        session.commit()

    def read():
        clear_stats_cache()
        return owner.get(
            f"{API_PREFIX}/pictures/stats",
            params={
                "tag": STATS_PENALISED_TAG,
                "include": "cooc",
                "only_penalised": only_penalised,
            },
        )

    server.vault.db.run_task(seed)
    try:
        body = _pinned_matches_unpinned(server, read)
    finally:
        server.vault.db.run_task(drop)
        clear_stats_cache()
    assert body["total"] == STATS_TAGGED
    assert sorted((row["tag"], row["count"]) for row in body["top_tags"]) == [
        (STATS_PENALISED_TAG, STATS_TAGGED),
        (POOL_TAG, STATS_TAGGED),
    ]
    assert (
        sorted((tuple(row["tags"]), row["count"]) for row in body["top_cooccurrences"])
        == pairs
    )


# People in one ``character_ids`` filter: past the pinned ceiling, one chunk
# and part of another.
PEOPLE = VARIABLE_CEILING + 1


def _people(server, cast):
    """``{character id: picture id}`` for the first PEOPLE cast members."""
    picture_by_name = {name: picture_id for picture_id, name in cast.items()}

    def read(session):
        return session.exec(
            select(Character.id, Character.name).where(
                Character.name.like("idlist-cast-%")
            )
        ).all()

    rows = sorted(server.vault.db.run_task(read))
    assert len(rows) == CAST_SIZE
    return {int(cid): picture_by_name[name] for cid, name in rows[:PEOPLE]}


def test_search_reads_a_long_people_filter(env, cast, monkeypatch):
    """``GET /pictures/search?character_ids=..`` reads every person named.

    The list repeats a few people, as a client may. A union must reach the
    ranking with every person's picture, and a difference keeps its first
    person as the base even when that person is named again later.
    """
    server, owner, _reference_id, _pool_ids = env
    people = _people(server, cast)
    cids = list(people)
    candidates = []
    real_semantic_search = Picture.semantic_search

    def spy_semantic_search(cls, session, *args, **kwargs):
        candidates.append(sorted(kwargs["candidate_ids"]))
        return real_semantic_search(session, *args, **kwargs)

    monkeypatch.setattr(
        server.vault,
        "generate_text_embedding",
        lambda _query: np.full(384, 0.5, dtype=np.float32),
    )
    monkeypatch.setattr(
        server.vault,
        "generate_clip_text_embedding",
        lambda _query: np.full(512, 0.25, dtype=np.float32),
    )
    monkeypatch.setattr(Picture, "semantic_search", classmethod(spy_semantic_search))
    search = [("query", POOL_TAG), ("threshold", 0.0)]

    union = search + [("character_ids", cid) for cid in cids + cids[:5]]
    _pinned_matches_unpinned(
        server, lambda: owner.get(f"{API_PREFIX}/pictures/search", params=union)
    )
    assert candidates == [sorted(people.values())] * 2

    candidates.clear()
    difference = search + [("character_mode", "difference")]
    difference += [("character_ids", cid) for cid in cids + cids[:1]]
    _pinned_matches_unpinned(
        server, lambda: owner.get(f"{API_PREFIX}/pictures/search", params=difference)
    )
    assert candidates == [[people[cids[0]]]] * 2


def test_listing_reads_a_long_people_filter(env, cast):
    """``GET /pictures?character_ids=..`` lists every person's picture."""
    server, owner, _reference_id, _pool_ids = env
    people = _people(server, cast)
    cids = list(people)
    body = _pinned_matches_unpinned(
        server, _listing(owner, character_ids=cids + cids[:5])
    )
    assert sorted(pic["id"] for pic in body) == sorted(people.values())


def test_stats_read_a_long_people_filter(env, cast):
    """``GET /pictures/stats?character_ids=..`` counts every person's picture."""
    server, owner, _reference_id, _pool_ids = env
    people = _people(server, cast)
    cids = list(people)

    def read():
        clear_stats_cache()
        return owner.get(
            f"{API_PREFIX}/pictures/stats", params={"character_ids": cids + cids[:5]}
        )

    try:
        body = _pinned_matches_unpinned(server, read)
    finally:
        clear_stats_cache()
    assert body["total"] == PEOPLE


# Sets in one ``set_ids`` filter: past the pinned ceiling, one chunk and part
# of another.
SHELF_SIZE = VARIABLE_CEILING + 1


@pytest.fixture(scope="module")
def shelf(env, cast):
    """SHELF_SIZE sets, set n holding cast picture n.

    The first set also holds the last set's picture, so a difference based on
    the first set answers its own picture only when the last set is read too.
    Returns the set ids and each set's own picture id, in set order.
    """
    server, _owner, _reference_id, _pool_ids = env
    picture_ids = sorted(cast)[:SHELF_SIZE]

    def seed(session):
        sets = [PictureSet(name=f"idlist-shelf-{n}") for n in range(SHELF_SIZE)]
        session.add_all(sets)
        session.flush()
        for picture_set, picture_id in zip(sets, picture_ids):
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=picture_id))
        session.add(PictureSetMember(set_id=sets[0].id, picture_id=picture_ids[-1]))
        session.commit()
        return [int(picture_set.id) for picture_set in sets]

    return server.vault.db.run_task(seed), picture_ids


@pytest.mark.parametrize("route", ["/pictures", "/pictures/count", "/pictures/stats"])
def test_a_long_set_filter_reads_every_set(env, shelf, route):
    """``?set_ids=..`` past the ceiling: a union and a difference of every set.

    The list repeats a few sets, as a client may.
    """
    server, owner, _reference_id, _pool_ids = env
    set_ids, picture_ids = shelf

    def answer(set_mode):
        def read():
            clear_stats_cache()
            return owner.get(
                f"{API_PREFIX}{route}",
                params={
                    "set_ids": set_ids + set_ids[:5],
                    "set_mode": set_mode,
                    **({"fields": "grid"} if route == "/pictures" else {}),
                },
            )

        try:
            body = _pinned_matches_unpinned(server, read)
        finally:
            clear_stats_cache()
        if route == "/pictures":
            return sorted(pic["id"] for pic in body)
        return body["count"] if route == "/pictures/count" else body["total"]

    if route == "/pictures":
        assert answer("union") == picture_ids
        assert answer("difference") == [picture_ids[0]]
    else:
        assert answer("union") == SHELF_SIZE
        assert answer("difference") == 1


def test_apply_scores_past_the_ceiling(env):
    """``POST /pictures/apply-scores`` rates every picture of a select-all.

    Runs last in the module: it writes scores onto the pool.
    """
    server, owner, _reference_id, pool_ids = env
    scores = {str(pid): 1 + pid % 5 for pid in pool_ids}
    with _sqlite_variable_ceiling(server):
        r = owner.post(
            f"{API_PREFIX}/pictures/apply-scores",
            json={"scores": scores, "only_unscored": False},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["updated_ids"] == pool_ids
    assert body["missing_ids"] == []

    def read_scores(session):
        return dict(
            session.exec(
                select(Picture.id, Picture.score).where(Picture.score.is_not(None))
            ).all()
        )

    assert server.vault.db.run_task(read_scores) == {
        pid: 1 + pid % 5 for pid in pool_ids
    }


# Between 498 and SQLITE_ID_CHUNK: a list this long fits in a statement once
# but not twice, and the stack-leader collapse reads an id list twice (once to
# filter, once to rank each stack inside the list).
FITS_ONCE = 600


def test_grid_listing_of_an_id_list_that_fits_once_but_not_twice(env, crowd):
    """``GET /pictures?fields=grid`` with ``FITS_ONCE`` ids.

    Every stack in the list is named by its second member only, so each must
    still show, represented by that member.
    """
    server, owner, _reference_id, _pool_ids = env
    non_covers = sorted(set(crowd["stacked"]) - set(crowd["leaders"]))
    ids = (
        non_covers[: FITS_ONCE // 2]
        + crowd["loose"][HIDDEN_MEMBERS : HIDDEN_MEMBERS + FITS_ONCE // 2]
    )
    assert len(ids) == FITS_ONCE

    body = _pinned_matches_unpinned(
        server,
        lambda: owner.get(
            f"{API_PREFIX}/pictures",
            params=[("fields", "grid"), ("limit", "2000")]
            + [("id", str(pid)) for pid in ids],
        ),
    )

    assert sorted(pic["id"] for pic in body) == sorted(ids)


def test_set_listing_of_a_set_that_fits_once_but_not_twice(env, crowd):
    """``GET /picture_sets/{id}`` of a ``FITS_ONCE``-member set collapses its
    stacks by default, which reads the member list twice."""
    server, owner, _reference_id, _pool_ids = env
    members = crowd["stacked"][:FITS_ONCE]

    def seed(session):
        picture_set = PictureSet(name="idlist-fits-once-set")
        session.add(picture_set)
        session.flush()
        for pid in members:
            session.add(PictureSetMember(set_id=picture_set.id, picture_id=pid))
        session.commit()
        stacks = session.exec(
            select(Picture.stack_id, Picture.id, Picture.stack_position).where(
                Picture.id.in_(members)
            )
        ).all()
        return int(picture_set.id), stacks

    set_id, rows = server.vault.db.run_task(seed)
    stack_ids = {stack_id for stack_id, _pid, _position in rows}

    def covers(session):
        return sorted(
            session.exec(
                select(Picture.id).where(
                    Picture.stack_id.in_(sorted(stack_ids)),
                    Picture.stack_position == 0,
                )
            ).all()
        )

    expected = server.vault.db.run_task(covers)
    assert len(set(members)) == FITS_ONCE

    body = _pinned_matches_unpinned(
        server, lambda: owner.get(f"{API_PREFIX}/picture_sets/{set_id}")
    )

    pictures = body["pictures"]
    assert sorted(pic["id"] for pic in pictures) == expected
    assert {pic["stack_count"] for pic in pictures} == {2}


# The term lists a request sends one value at a time, each value binding its
# own variables in the statement that filters by it.
TERM_LISTS = (
    "tag",
    "rejected_tag",
    "tag_confidence_above",
    "tag_confidence_below",
    "comfyui_model",
    "comfyui_lora",
)
# Every route that reads them: the listing and its count, stream and stats,
# the search, a set's listing and the likeness groups.
TERM_ROUTES = (
    ("/pictures", {}),
    ("/pictures/count", {}),
    ("/pictures/stream", {}),
    ("/pictures/stats", {}),
    ("/pictures/search", {"query": "example query"}),
    ("/picture_sets/1", {}),
    ("/pictures/likeness-groups", {}),
)


def _term_values(name, count):
    """``count`` values for the term list ``name``, none of them a real term."""
    if name.startswith("tag_confidence_"):
        return [(name, f"example-term-{n}:0.5") for n in range(count)]
    return [(name, f"example-term-{n}") for n in range(count)]


@pytest.mark.parametrize("name", TERM_LISTS)
@pytest.mark.parametrize("route, params", TERM_ROUTES, ids=[r for r, _ in TERM_ROUTES])
def test_a_term_list_past_the_cap_is_refused_by_name(env, route, params, name):
    """A list longer than ``MAX_FILTER_TERMS`` is a 422 naming the list.

    The sentence is fixed: none of the values comes back.
    """
    _server, owner, _reference_id, _pool_ids = env
    r = owner.get(
        f"{API_PREFIX}{route}",
        params=list(params.items()) + _term_values(name, MAX_FILTER_TERMS + 1),
    )
    assert r.status_code == 422, (r.status_code, r.text[:400])
    assert r.json()["detail"] == (f"Too many {name} values: at most {MAX_FILTER_TERMS}")
    assert "example-term" not in r.text


# The routes whose answer is one JSON body, so the pinned answer can be
# compared with the unpinned one.
TERM_JSON_ROUTES = [TERM_ROUTES[i] for i in (0, 1, 3, 4)]


@pytest.mark.parametrize(
    "route, params", TERM_JSON_ROUTES, ids=[r for r, _ in TERM_JSON_ROUTES]
)
def test_every_term_list_at_the_cap_fits_under_the_ceiling(env, route, params):
    """All six lists at ``MAX_FILTER_TERMS`` at once still bind inside 999.

    The cap keeps half the ceiling for the rest of the statement. Each value
    is also one more ``AND``, so a cap without that margin fails here first
    with "Expression tree is too large (maximum depth 1000)".
    """
    server, owner, _reference_id, _pool_ids = env
    terms = [
        pair for name in TERM_LISTS for pair in _term_values(name, MAX_FILTER_TERMS)
    ]

    def read():
        clear_stats_cache()
        return owner.get(f"{API_PREFIX}{route}", params=list(params.items()) + terms)

    try:
        _pinned_matches_unpinned(server, read)
    finally:
        clear_stats_cache()


# The two routes that take a selection in a JSON body. Their options are the
# view routes', less the search ``threshold``: a selection runs no search, and
# the threshold is the server's policy, so a client's value is refused.
SELECTION_ROUTES = ("/pictures/export", "/pictures/export/folder/selection")


@pytest.mark.parametrize("route", SELECTION_ROUTES)
def test_a_selection_body_threshold_is_refused_by_name(env, route, tmp_path):
    """A body ``threshold`` is a 422 naming the field, and no export starts.

    The same body without it passes validation, so the 422 is the threshold's.
    """
    server, owner, _reference_id, pool_ids = env
    # Missing, so the folder route's handler answers 404 rather than writing.
    body = {"picture_ids": [pool_ids[0]]}
    if route.endswith("/selection"):
        body["destination"] = str(tmp_path / "example-missing-destination")
    before = set(server.export_tasks)

    refused = owner.post(f"{API_PREFIX}{route}", json={**body, "threshold": 0.01})

    assert refused.status_code == 422, refused.text
    assert [(e["loc"], e["type"]) for e in refused.json()["detail"]] == [
        (["body", "threshold"], "extra_forbidden")
    ]
    assert set(server.export_tasks) == before

    accepted = owner.post(f"{API_PREFIX}{route}", json=body)
    assert accepted.status_code != 422, accepted.text
