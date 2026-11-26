"""Tests for the GET /pictures/stats endpoint."""

import gc
import json
import os
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

import pixlstash.routes.pictures as pictures_module  # noqa: F401  (kept for backward compat with other tests)
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Project,
    ReferenceFolder,
    Tag,
)
from pixlstash.db_models.tag import TAG_PENDING_SENTINEL, make_tag_sentinel
from pixlstash.tasks.missing_tag_finder import MissingTagFinder
from pixlstash.tasks.tag_task import TagTask
from pixlstash.utils.service import picture_stats as picture_stats_module
from pixlstash.utils.service.picture_stats import STATS_TTL, clear_stats_cache
from pixlstash.server import Server
from tests.authz_guard import assert_real_route
from tests.utils import bearer_client, mint_read_token, upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")
# A face embedding: every face shares it, so likeness ranks them all alike.
FACE_FEATURES = np.ones(8, dtype=np.float32).tobytes()


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
    results = result.get("results") or []
    assert results, "No pictures imported"
    return results[0]["picture_id"]


def test_stats_basic_counts():
    """Total, tagged, untagged counts are accurate and response shape is correct."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id1 = _upload_picture(client, "Bad1.png")
        _upload_picture(client, "Bad2.png")

        # Tag one picture only.
        resp = client.post(f"/pictures/{pic_id1}/tags", json={"tag": "solo_tag"})
        assert resp.status_code == 200

        clear_stats_cache()
        resp = client.get("/pictures/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total"] == 2
        assert data["tagged"] == 1
        assert data["untagged"] == 1
        assert data["total_tags"] == 1
        assert isinstance(data["avg_tags_per_image"], float)
        assert isinstance(data["top_tags"], list)
        assert any(t["tag"] == "solo_tag" for t in data["top_tags"])
        # By default these expensive sections are empty.
        assert data["top_cooccurrences"] == []
        assert data["confidence_histogram"] == []
        assert data["score_distribution"] == []
        assert data["smart_score_distribution"] == []
        assert data["resolution_distribution"] == []
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_tag_filter_reduces_total():
    """Passing tag= restricts the picture population used for counting."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id1 = _upload_picture(client, "Bad1.png")
        _upload_picture(client, "Bad2.png")

        client.post(f"/pictures/{pic_id1}/tags", json={"tag": "rare_filter_tag"})

        clear_stats_cache()
        resp = client.get("/pictures/stats?tag=rare_filter_tag")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

        clear_stats_cache()
        resp = client.get("/pictures/stats?tag=nonexistent_tag_xyz")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_score_filter():
    """min_score/max_score params restrict the picture population."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id1 = _upload_picture(client, "Bad1.png")
        pic_id2 = _upload_picture(client, "Bad2.png")

        client.patch(f"/pictures/{pic_id1}", json={"score": 5})
        client.patch(f"/pictures/{pic_id2}", json={"score": 2})

        clear_stats_cache()
        resp = client.get("/pictures/stats?min_score=4")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        clear_stats_cache()
        resp = client.get("/pictures/stats?max_score=3")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        clear_stats_cache()
        resp = client.get("/pictures/stats?min_score=1&max_score=5")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_include_picture():
    """include=picture returns score_distribution, smart_score_distribution, resolution_distribution."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id = _upload_picture(client, "Bad1.png")
        client.patch(f"/pictures/{pic_id}", json={"score": 3})

        clear_stats_cache()
        resp = client.get("/pictures/stats?include=picture")
        assert resp.status_code == 200
        data = resp.json()

        score_dist = data["score_distribution"]
        assert isinstance(score_dist, list)
        labels = [entry["label"] for entry in score_dist]
        assert "3" in labels
        entry_3 = next(e for e in score_dist if e["label"] == "3")
        assert entry_3["count"] == 1

        assert isinstance(data["smart_score_distribution"], list)
        assert len(data["smart_score_distribution"]) > 0
        assert isinstance(data["resolution_distribution"], list)
        assert len(data["resolution_distribution"]) > 0
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_unscored_histogram_bucket_matches_what_the_unscored_filter_returns():
    """The Unscored bar and ``unscored=1`` must agree, score 0 included.

    Clicking the current star again writes a literal 0 via ``POST
    /pictures/apply-scores``, and nothing normalises it back to NULL. The
    histogram used to read only the NULL group, so a score-0 picture fell into no
    bucket at all: the bars did not sum to the library and the sidebar count
    disagreed with the grid on the very first click.
    """
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        cleared = _upload_picture(client, "Bad1.png")
        never = _upload_picture(client, "Bad2.png")
        rated = _upload_picture(client, "Changed1.png")

        resp = client.post(
            "/pictures/apply-scores",
            json={"scores": {str(cleared): 0, str(rated): 4}, "only_unscored": False},
        )
        assert resp.status_code == 200

        clear_stats_cache()
        data = client.get("/pictures/stats?include=picture").json()
        score_dist = {e["label"]: e["count"] for e in data["score_distribution"]}
        assert score_dist["Unscored"] == 2, "score 0 belongs in Unscored with NULL"
        assert score_dist["4"] == 1
        assert sum(score_dist.values()) == data["total"], (
            "every picture lands in exactly one bar"
        )

        resp = client.get("/pictures?unscored=1&limit=100")
        assert resp.status_code == 200
        returned = {p["id"] for p in resp.json()}
        assert returned == {cleared, never}, "both NULL and 0, and nothing rated"
        assert len(returned) == score_dist["Unscored"]

        # The over-blocking direction: the rated picture is still reachable, and
        # an unfiltered listing is unaffected.
        resp = client.get("/pictures?min_score=1&limit=100")
        assert {p["id"] for p in resp.json()} == {rated}
        resp = client.get("/pictures?limit=100")
        assert {p["id"] for p in resp.json()} == {cleared, never, rated}

        # The param reaches the stats population too, so the sidebar totals track
        # the grid rather than the whole library.
        clear_stats_cache()
        assert client.get("/pictures/stats?unscored=1").json()["total"] == 2
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_include_cooc():
    """include=cooc returns top_cooccurrences when two tags share a picture."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id = _upload_picture(client, "Bad1.png")
        client.post(f"/pictures/{pic_id}/tags", json={"tag": "cooc_a"})
        client.post(f"/pictures/{pic_id}/tags", json={"tag": "cooc_b"})

        clear_stats_cache()
        resp = client.get("/pictures/stats?include=cooc")
        assert resp.status_code == 200
        data = resp.json()

        cooc = data["top_cooccurrences"]
        assert isinstance(cooc, list)
        assert len(cooc) > 0
        pair = cooc[0]
        assert set(pair["tags"]) == {"cooc_a", "cooc_b"}
        assert pair["count"] == 1
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_include_conf():
    """include=conf returns non-empty confidence_histogram and regular_tags."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        pic_id = _upload_picture(client, "Bad1.png")
        client.post(f"/pictures/{pic_id}/tags", json={"tag": "conf_tag"})

        clear_stats_cache()
        resp = client.get("/pictures/stats?include=conf")
        assert resp.status_code == 200
        data = resp.json()

        assert isinstance(data["confidence_histogram"], list)
        assert isinstance(data["regular_tags"], list)
        assert "conf_tag" in data["regular_tags"]
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_cache_expires_after_ttl(monkeypatch):
    """After the TTL elapses the result is recomputed from the database."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        _upload_picture(client, "Bad1.png")

        clear_stats_cache()
        resp1 = client.get("/pictures/stats")
        assert resp1.status_code == 200
        assert resp1.json()["total"] == 1

        # Expire the cache entry by back-dating its timestamp.
        expired_ts = time.monotonic() - (STATS_TTL + 1)
        for key in list(picture_stats_module._stats_cache.keys()):
            _, data = picture_stats_module._stats_cache[key]
            picture_stats_module._stats_cache[key] = (expired_ts, data)

        _upload_picture(client, "Bad2.png")

        resp2 = client.get("/pictures/stats")
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 2, (
            "Stats should be recomputed after TTL expires"
        )
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_scrapheap_view_reads_deleted_only():
    """``only_deleted=true`` (what the Scrapheap panel sends) counts the heap.

    The endpoint used to read only the legacy ``character_id=SCRAPHEAP``, so the
    Scrapheap panel showed the live library's totals beside a sidebar badge of 0.
    """
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        live = _upload_picture(client, "Bad1.png")
        gone = _upload_picture(client, "Bad2.png")
        assert (
            client.post(f"/pictures/{live}/tags", json={"tag": "live_tag"}).status_code
            == 200
        )
        assert (
            client.post(f"/pictures/{gone}/tags", json={"tag": "gone_tag"}).status_code
            == 200
        )
        assert client.delete(f"/pictures/{gone}").status_code == 200

        for params in ({"only_deleted": "true"}, {"character_id": "SCRAPHEAP"}):
            clear_stats_cache()
            data = client.get("/pictures/stats", params=params).json()
            assert data["total"] == 1, params
            assert data["tagged"] == 1, params
            assert [t["tag"] for t in data["top_tags"]] == ["gone_tag"], params

        # The live view is untouched: the heap picture is not in it.
        clear_stats_cache()
        data = client.get("/pictures/stats").json()
        assert data["total"] == 1
        assert [t["tag"] for t in data["top_tags"]] == ["live_tag"]
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_empty_scrapheap_is_zero():
    """An empty heap reports zero, not the live library."""
    temp_dir, client, server = _setup()
    clear_stats_cache()
    try:
        _upload_picture(client, "Bad1.png")

        clear_stats_cache()
        data = client.get("/pictures/stats", params={"only_deleted": "true"}).json()
        assert data["total"] == 0
        assert data["tagged"] == 0
        assert data["untagged"] == 0
        assert data["top_tags"] == []
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_untagged_breakdown_and_tag_state_filter(tmp_path):
    """``untagged_breakdown`` says why each untagged picture has no tags.

    ``pending`` is what the tagger will take up, ``unprocessable`` is held
    out by the unreadable-file registry, and ``none`` was tagged and found
    nothing (or lost its tags since). The three sum to ``untagged``, and
    ``tag_state`` narrows the stats to one of them. A scrapheaped picture's
    sentinel is not ``pending``: the tagger does not take up the Scrapheap.
    """
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps({"port": 8000, "disable_background_workers": True})
    )
    server = Server(str(config_path))
    client = TestClient(server.api)
    assert (
        client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        ).status_code
        == 200
    )
    clear_stats_cache()
    try:

        def seed(session):
            ids = {}
            for key, tag, deleted in (
                ("tagged", "cat", False),
                ("pending", TAG_PENDING_SENTINEL, False),
                ("engine_pending", make_tag_sentinel("wd14"), False),
                ("none", None, False),
                ("broken", TAG_PENDING_SENTINEL, False),
                ("gone_pending", TAG_PENDING_SENTINEL, True),
            ):
                (tmp_path / f"{key}.png").write_bytes(b"test-not-an-image")
                pic = Picture(
                    file_path=str(tmp_path / f"{key}.png"),
                    width=8,
                    height=8,
                    deleted=deleted,
                )
                session.add(pic)
                session.flush()
                if tag is not None:
                    session.add(Tag(picture_id=pic.id, tag=tag))
                ids[key] = pic.id
            session.commit()
            return ids

        ids = server.vault.db.run_task(seed)
        assert server.vault.db.unprocessable_images.mark_unprocessable(
            ids["broken"], str(tmp_path / "broken.png"), reason="test-corrupt fixture"
        )

        data = client.get("/pictures/stats").json()
        assert data["total"] == 5
        assert data["tagged"] == 1
        assert data["untagged"] == 4
        assert data["untagged_breakdown"] == {
            "pending": 2,
            "unprocessable": 1,
            "none": 1,
        }

        heap = client.get("/pictures/stats", params={"only_deleted": "true"}).json()
        assert heap["untagged_breakdown"] == {
            "pending": 0,
            "unprocessable": 0,
            "none": 1,
        }

        # Each state, as a filter, is exactly the pictures its count counted.
        for state, expected in (("pending", 2), ("unprocessable", 1), ("none", 1)):
            sliced = client.get("/pictures/stats", params={"tag_state": state}).json()
            assert sliced["total"] == expected, state
            assert sliced["tagged"] == 0, state
            assert sliced["untagged_breakdown"][state] == expected, state

        resp = client.get("/pictures/stats", params={"tag_state": "bogus"})
        assert resp.status_code == 422
        assert "tag_state" in resp.json()["detail"]
    finally:
        server.close()
        gc.collect()


def test_grid_tag_state_lists_exactly_the_pictures_the_breakdown_counts(tmp_path):
    """``GET /pictures?tag_state=<reason>`` is the grid a breakdown row opens.

    For every reason, live and in the Scrapheap, the listed ids are the very
    pictures the stats breakdown counts under it, and ``/pictures/count``
    agrees. The face-sorted path filters the same way, and so do the Inbox and
    the Unassigned view, which build their query apart from the main listing:
    a picture a person claims is left out of both, under every reason.

    ``pending`` is what the tag finder hands out and what the tagger's
    remaining count counts, by one shared definition: a sentinel in the
    Scrapheap or on a locked-set member is not waiting for the tagger.
    """
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps({"port": 8000, "disable_background_workers": True})
    )
    server = Server(str(config_path))
    client = TestClient(server.api)
    assert (
        client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        ).status_code
        == 200
    )
    clear_stats_cache()
    try:

        def seed(session):
            ids = {}
            claimant = Character(name="test-claimant")
            session.add(claimant)
            session.flush()
            for key, tag, deleted in (
                ("tagged", "cat", False),
                ("pending", TAG_PENDING_SENTINEL, False),
                ("engine_pending", make_tag_sentinel("wd14"), False),
                ("none", None, False),
                ("broken", TAG_PENDING_SENTINEL, False),
                ("broken_no_sentinel", None, False),
                ("claimed_pending", TAG_PENDING_SENTINEL, False),
                ("claimed_none", None, False),
                ("claimed_broken", None, False),
                ("claimed_locked_pending", TAG_PENDING_SENTINEL, False),
                ("retagged_kept", "cat", False),
                ("gone_pending", TAG_PENDING_SENTINEL, True),
                ("gone_none", None, True),
                ("gone_broken", TAG_PENDING_SENTINEL, True),
            ):
                (tmp_path / f"{key}.png").write_bytes(b"test-not-an-image")
                pic = Picture(
                    file_path=str(tmp_path / f"{key}.png"),
                    width=8,
                    height=8,
                    deleted=deleted,
                )
                session.add(pic)
                session.flush()
                if tag is not None:
                    session.add(Tag(picture_id=pic.id, tag=tag))
                # A face on each, so the face-sorted listing sees them all.
                # A ``claimed_*`` face names a person, which takes the picture
                # out of the Inbox and the Unassigned view.
                session.add(
                    Face(
                        picture_id=pic.id,
                        face_index=0,
                        bbox=[1, 1, 7, 7],
                        character_id=(
                            claimant.id if key.startswith("claimed_") else None
                        ),
                    )
                )
                ids[key] = pic.id
            # Re-tagged with a real tag kept (as ``retag_face_crops`` leaves
            # one): tagger work, but not untagged.
            session.add(Tag(picture_id=ids["retagged_kept"], tag=TAG_PENDING_SENTINEL))
            locked = PictureSet(name="test-locked-set", locked=True)
            session.add(locked)
            session.flush()
            session.add(
                PictureSetMember(
                    set_id=locked.id, picture_id=ids["claimed_locked_pending"]
                )
            )
            session.commit()
            return ids

        ids = server.vault.db.run_task(seed)
        for key in ("broken", "broken_no_sentinel", "claimed_broken", "gone_broken"):
            assert server.vault.db.unprocessable_images.mark_unprocessable(
                ids[key], str(tmp_path / f"{key}.png"), reason="test-corrupt fixture"
            )

        unassigned = {
            "pending": {ids["pending"], ids["engine_pending"]},
            "unprocessable": {ids["broken"], ids["broken_no_sentinel"]},
            "none": {ids["none"]},
        }
        expected = {
            False: {
                "pending": unassigned["pending"] | {ids["claimed_pending"]},
                "unprocessable": unassigned["unprocessable"] | {ids["claimed_broken"]},
                "none": unassigned["none"]
                | {ids["claimed_none"], ids["claimed_locked_pending"]},
            },
            True: {
                "pending": set(),
                "unprocessable": {ids["gone_broken"]},
                "none": {ids["gone_none"], ids["gone_pending"]},
            },
        }
        for deleted, by_state in expected.items():
            scope = {"only_deleted": "true"} if deleted else {}
            breakdown = client.get("/pictures/stats", params=scope).json()[
                "untagged_breakdown"
            ]
            for state, want in by_state.items():
                params = {**scope, "tag_state": state}
                listed = client.get("/pictures", params=params)
                assert listed.status_code == 200, listed.text
                listed_ids = {item["id"] for item in listed.json()}
                assert listed_ids == want, (deleted, state)
                assert breakdown[state] == len(listed_ids), (deleted, state)
                counted = client.get("/pictures/count", params=params)
                assert counted.status_code == 200, counted.text
                assert counted.json()["count"] == len(want), (deleted, state)

        # The "Waiting for the tagger" row is what the finder hands out and
        # what the tagger's remaining count counts, with the one deliberate
        # difference that a re-tag with a real tag kept is not untagged.
        suppressed = server.vault.db.unprocessable_images.active_suppressed_ids()
        window = server.vault.db.run_immediate_read_task(
            lambda s: MissingTagFinder._fetch_missing_tags(s, 1000, suppressed)
        )
        remaining = server.vault.db.run_immediate_read_task(
            lambda s: TagTask.count_missing_tags(s, suppressed)
        )
        tagger_work = expected[False]["pending"] | {ids["retagged_kept"]}
        assert {pic.id for pic in window} == tagger_work
        assert remaining == len(tagger_work)

        for state, want in expected[False].items():
            counted = client.get(
                "/pictures/count",
                params={"sort": "CHARACTER_LIKENESS", "tag_state": state},
            )
            assert counted.status_code == 200, counted.text
            assert counted.json()["count"] == len(want), state

        # The Inbox and the Unassigned view list, count and break down the
        # same unclaimed pictures for each reason.
        for scope in ({"character_id": "UNASSIGNED"}, {"inbox": "all"}):
            clear_stats_cache()
            breakdown = client.get("/pictures/stats", params=scope).json()[
                "untagged_breakdown"
            ]
            for state, want in unassigned.items():
                params = {**scope, "tag_state": state}
                listed = client.get("/pictures", params=params)
                assert listed.status_code == 200, listed.text
                listed_ids = {item["id"] for item in listed.json()}
                assert listed_ids == want, (scope, state)
                assert breakdown[state] == len(want), (scope, state)
                counted = client.get("/pictures/count", params=params)
                assert counted.status_code == 200, counted.text
                assert counted.json()["count"] == len(want), (scope, state)

        refused = client.get("/pictures", params={"tag_state": "bogus"})
        assert refused.status_code == 422
        assert "tag_state" in refused.json()["detail"]
    finally:
        server.close()
        gc.collect()


def test_header_grid_and_stats_agree_for_each_view(tmp_path):
    """The header count, the grid's list and the Details panel's total agree.

    One case per view whose count, list and stats were built apart: the
    Unassigned view's collapsed stacks, a folder in the Scrapheap, the Scrapheap
    with the unassigned view or the Inbox on, the unassigned view in a project,
    a set difference with its base named, the unassigned view and the Inbox
    narrowed to one reference folder, and the filters the panel used to leave
    out. The folder views list the same pictures under the likeness sort,
    whose candidates once took neither folder filter. Two differences are deliberate and documented in
    ``picture_stats``: the panel counts pictures, never stack leaders, so the
    collapsed-stack cases compare only the header with the grid; and it counts
    a folder with its subfolders, which the folder case states.
    """
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps({"port": 8000, "disable_background_workers": True})
    )
    server = Server(str(config_path))
    client = TestClient(server.api)
    assert (
        client.post(
            "/login", json={"username": "testuser", "password": "testpassword"}
        ).status_code
        == 200
    )
    clear_stats_cache()
    try:

        def seed(session):
            ids = {}
            stack = PictureStack(name="test-stack")
            project = Project(name="test-project")
            # A person in no project: inside the project, a picture they claim
            # is still unassigned.
            outsider = Character(name="test-outsider")
            reference = ReferenceFolder(folder=str(tmp_path / "ref"), label="test-ref")
            session.add_all([stack, project, outsider, reference])
            session.flush()
            for key, rel, extra in (
                ("in_reference", "ref/r.png", {"reference_folder_id": reference.id}),
                ("stack_lead", "a/s1.png", {"stack_id": stack.id, "stack_position": 0}),
                (
                    "stack_member",
                    "a/s2.png",
                    {"stack_id": stack.id, "stack_position": 1},
                ),
                ("in_project", "c/p.png", {"comfyui_models": '["test-model"]'}),
                ("claimed_outside", "c/o.png", {}),
                ("set_one", "c/m1.png", {}),
                ("set_both", "c/m2.png", {}),
                ("gone_a", "a/g1.png", {"deleted": True}),
                ("gone_sub", "a/sub/g2.png", {"deleted": True}),
                ("gone_b", "b/g3.png", {"deleted": True}),
            ):
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test-not-an-image")
                pic = Picture(file_path=str(path), width=8, height=8, **extra)
                session.add(pic)
                session.flush()
                ids[key] = pic.id
            first = PictureSet(name="test-first")
            second = PictureSet(name="test-second")
            session.add_all([first, second])
            session.flush()
            session.add_all(
                [
                    PictureSetMember(set_id=first.id, picture_id=ids["set_one"]),
                    PictureSetMember(set_id=first.id, picture_id=ids["set_both"]),
                    PictureSetMember(set_id=second.id, picture_id=ids["set_both"]),
                    PictureProjectMember(
                        picture_id=ids["in_project"], project_id=project.id
                    ),
                    PictureProjectMember(
                        picture_id=ids["claimed_outside"], project_id=project.id
                    ),
                    # The outsider's face is the likeness sort's reference.
                    Face(
                        picture_id=ids["claimed_outside"],
                        face_index=0,
                        bbox=[1, 1, 7, 7],
                        character_id=outsider.id,
                        features=FACE_FEATURES,
                    ),
                ]
            )
            # Faces nobody claims, so the likeness sort can list these
            # pictures; they leave every unassigned view as it was.
            session.add_all(
                [
                    Face(
                        picture_id=ids[key],
                        face_index=0,
                        bbox=[1, 1, 7, 7],
                        features=FACE_FEATURES,
                    )
                    for key in (
                        "in_reference",
                        "in_project",
                        "set_one",
                        "set_both",
                        "gone_a",
                        "gone_sub",
                        "gone_b",
                    )
                ]
            )
            session.commit()
            ids["first"], ids["second"], ids["project"] = (
                first.id,
                second.id,
                project.id,
            )
            ids["reference"] = reference.id
            ids["outsider"] = outsider.id
            return ids

        ids = server.vault.db.run_task(seed)

        def agree(params, want, *, with_stats=True, stats_total=None):
            listed = client.get("/pictures", params={**params, "fields": "grid"})
            assert listed.status_code == 200, (params, listed.text)
            assert {item["id"] for item in listed.json()} == want, params
            counted = client.get(
                "/pictures/count", params={**params, "stack_leaders_only": "true"}
            )
            assert counted.status_code == 200, (params, counted.text)
            assert counted.json()["count"] == len(want), params
            if with_stats:
                clear_stats_cache()
                stats = client.get("/pictures/stats", params=params)
                assert stats.status_code == 200, (params, stats.text)
                expected_total = len(want) if stats_total is None else stats_total
                assert stats.json()["total"] == expected_total, params

        heap = {ids["gone_a"], ids["gone_sub"], ids["gone_b"]}
        # Unassigned, stacks collapsed: the header counts the stack once.
        agree(
            {"character_id": "UNASSIGNED"},
            {ids["stack_lead"], ids["in_project"], ids["in_reference"]},
            with_stats=False,
        )
        # One reference folder, alone, in the unassigned view and in the Inbox
        # (which lists through the unassigned view).
        reference_only = {"reference_folder_id": str(ids["reference"])}
        agree(reference_only, {ids["in_reference"]})
        agree({**reference_only, "character_id": "UNASSIGNED"}, {ids["in_reference"]})
        agree({**reference_only, "inbox": "all"}, {ids["in_reference"]})
        # A folder in the Scrapheap: the grid and the header take its own
        # pictures, not its subfolders'. The panel summarises the folder with
        # its subfolders on purpose (``picture_stats``, where the folder
        # predicate is built), so it counts both.
        agree(
            {"only_deleted": "true", "file_path_prefix": str(tmp_path / "a")},
            {ids["gone_a"]},
            stats_total=2,
        )
        # The Scrapheap wins over the unassigned view and the Inbox.
        agree({"only_deleted": "true", "character_id": "UNASSIGNED"}, heap)
        agree({"only_deleted": "true", "inbox": "all"}, heap)
        # The unassigned view within one project, where a person from outside
        # the project claims nothing.
        agree(
            {"character_id": "UNASSIGNED", "project_id": str(ids["project"])},
            {ids["in_project"], ids["claimed_outside"]},
        )
        # A difference is its base minus the rest, whichever order they come in.
        for base, want in (
            (ids["second"], set()),
            (ids["first"], {ids["set_one"]}),
        ):
            agree(
                {
                    "set_ids": [str(ids["first"]), str(ids["second"])],
                    "set_mode": "difference",
                    "base_set_id": str(base),
                },
                want,
            )
        # Filters the panel used to drop.
        agree({"comfyui_model": "test-model"}, {ids["in_project"]})
        agree(
            {"character_id": "UNASSIGNED", "unassigned_by": "character"},
            {
                ids["stack_lead"],
                ids["in_project"],
                ids["in_reference"],
                ids["set_one"],
                ids["set_both"],
            },
            with_stats=False,
        )
        # The folder views under the likeness sort list what they list
        # without it: its candidates take both folder filters.
        likeness = {
            "sort": "CHARACTER_LIKENESS",
            "reference_character_id": str(ids["outsider"]),
        }
        for params, want, stats_total in (
            (
                {"file_path_prefix": str(tmp_path / "c")},
                {
                    ids["in_project"],
                    ids["claimed_outside"],
                    ids["set_one"],
                    ids["set_both"],
                },
                None,
            ),
            (reference_only, {ids["in_reference"]}, None),
            (
                {**reference_only, "character_id": "UNASSIGNED"},
                {ids["in_reference"]},
                None,
            ),
            (
                {"only_deleted": "true", "file_path_prefix": str(tmp_path / "a")},
                {ids["gone_a"]},
                2,
            ),
            (
                {**reference_only, "character_id": str(ids["outsider"])},
                set(),
                None,
            ),
        ):
            agree(params, want, stats_total=stats_total)
            agree({**params, **likeness}, want, stats_total=stats_total)
    finally:
        server.close()
        gc.collect()


def test_stats_computed_across_a_clear_are_not_cached(monkeypatch):
    """A result read before a change must not outlive the clear that announced it.

    The clear lands while the stats are being computed, as a tag write does
    when it finishes during a slow read. The control run, with no clear, shows
    the same call does store its result.
    """
    clear_stats_cache()

    def compute_with_clear(vault, params):
        clear_stats_cache()
        return {"total": 1}

    monkeypatch.setattr(
        picture_stats_module, "compute_picture_stats", compute_with_clear
    )
    assert picture_stats_module.get_cached_picture_stats(None, None, "test-raced") == {
        "total": 1
    }
    assert "test-raced" not in picture_stats_module._stats_cache

    monkeypatch.setattr(
        picture_stats_module,
        "compute_picture_stats",
        lambda vault, params: {"total": 2},
    )
    picture_stats_module.get_cached_picture_stats(None, None, "test-calm")
    assert picture_stats_module._stats_cache["test-calm"][1] == {"total": 2}
    clear_stats_cache()


def test_stats_cache_digest_names_a_set_whatever_its_order():
    """The same ids give the same key part in any order; other ids do not."""
    digest = picture_stats_module.stats_cache_digest
    assert digest([3, 1, 2]) == digest({1, 2, 3})
    assert digest([1, 2]) != digest([1, 2, 3])
    assert digest(["1,2"]) != digest(["1", "2"])


def _stats(client, **params):
    clear_stats_cache()
    return client.get("/pictures/stats", params=params)


def test_stats_shared_only_counts_just_the_pictures_the_owner_shared():
    """``shared_only=true`` totals the owner's shared pictures, not the library."""
    temp_dir, client, server = _setup()
    try:
        assert_real_route(
            server.api, "GET", "/api/v1/pictures/stats", "/api/v1/pictures/stats"
        )
        shared, *_ = [
            _upload_picture(client, name)
            for name in ("Bad1.png", "Bad2.png", "Changed1.png")
        ]
        mint_read_token(
            client, "test-share", resource_type="picture", resource_id=shared
        )

        resp = _stats(client)
        assert resp.status_code == 200, resp.text
        assert resp.json()["total"] == 3
        resp = _stats(client, shared_only="true")
        assert resp.status_code == 200, resp.text
        assert resp.json()["total"] == 1
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def test_stats_face_cluster_is_for_the_owner_only():
    """A face cluster spans the person's whole library, as ``GET /face-clusters``
    does, so a READ token and a share link are refused it; each still reads the
    stats it is scoped to."""
    temp_dir, client, server = _setup()
    try:
        assert_real_route(
            server.api, "GET", "/api/v1/pictures/stats", "/api/v1/pictures/stats"
        )
        picture_id = _upload_picture(client, "Bad1.png")
        resp = client.post("/characters", json={"name": "test-person"})
        assert resp.status_code == 200, resp.text
        person = {"character_id": resp.json()["character"]["id"]}
        cluster = {**person, "face_cluster": 1}
        read = bearer_client(server, mint_read_token(client, "test-read"))
        share = bearer_client(
            server,
            mint_read_token(
                client, "test-share", resource_type="picture", resource_id=picture_id
            ),
        )

        resp = _stats(client, **cluster)
        assert resp.status_code == 200, resp.text
        for caller in (read, share):
            resp = _stats(caller, **person)
            assert resp.status_code == 200, resp.text
            resp = _stats(caller, **cluster)
            assert resp.status_code == 403, resp.text
            assert resp.json() == {
                "detail": "Owner-level (full, unscoped) access required"
            }
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()
