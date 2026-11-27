"""``similarity_character`` belongs to the library, not to the person.

It is a row id in one vault's character table, so a per-user copy silently names
a different person after a library switch: character 7 in Family Photos and
character 7 in Client Work are not the same face. This is the one setting from
the §5 candidate list that moved to the vault (decided 2026-08-02); hidden tags,
the tag filter and the penalised-tag weights stay in the hub as user preferences.

The API surface is unchanged, so the tests assert the seam is invisible from
outside: one flat config object, two databases behind it.
"""

import hashlib
import hmac
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from pixlstash.db_models import Picture, User
from pixlstash.db_models.library_settings import LibrarySettings
from pixlstash.db_models.tag import DEFAULT_SMART_SCORE_PENALIZED_TAGS
from pixlstash.scoring.smart_score import (
    get_smart_score_penalised_tags_from_request,
    resolve_penalised_tag_weights,
)
from pixlstash.routes import config as config_routes
from pixlstash.server import Server
from pixlstash.services import library_settings_service
from pixlstash.utils.quality.smart_score_utils import resolve_owner_penalised_tags
from tests.utils import reset_owner_auth

API = "/api/v1"


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = f"{temp_dir}/server-config.json"
        with open(config_path, "w") as handle:
            json.dump({}, handle)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture(autouse=True)
def clean(server):
    reset_owner_auth(server)
    library_settings_service.set_similarity_character(server.vault.db, None)
    with server.hub.transaction() as conn:
        conn.execute("DELETE FROM user_tag_preference")
    yield


def _owner(server) -> TestClient:
    client = TestClient(server.api)
    response = client.post(
        "/login", json={"username": "libset", "password": "libsetpass1"}
    )
    assert response.status_code == 200, response.text
    return client


class TestStorage:
    def test_it_is_stored_in_the_vault(self, server):
        library_settings_service.set_similarity_character(server.vault.db, 7)

        stored = server.vault.db.run_immediate_read_task(
            lambda session: session.exec(select(LibrarySettings)).first()
        )
        assert stored.similarity_character == 7

    def test_it_is_not_stored_in_the_hub(self, server):
        """The hub keeps the column for model compatibility but must not own it."""
        library_settings_service.set_similarity_character(server.vault.db, 7)

        in_hub = server.hub_engine.run_immediate_read_task(
            lambda session: session.exec(select(User)).first().similarity_character
        )
        assert in_hub is None

    def test_clearing_it_works(self, server):
        library_settings_service.set_similarity_character(server.vault.db, 7)
        library_settings_service.set_similarity_character(server.vault.db, None)

        assert (
            library_settings_service.get_similarity_character(server.vault.db) is None
        )

    def test_there_is_exactly_one_row(self, server):
        library_settings_service.set_similarity_character(server.vault.db, 3)
        library_settings_service.set_similarity_character(server.vault.db, 4)

        rows = server.vault.db.run_immediate_read_task(
            lambda session: session.exec(select(LibrarySettings)).all()
        )
        assert len(rows) == 1


class TestTheApiSeamIsInvisible:
    def test_the_config_get_returns_it(self, server):
        library_settings_service.set_similarity_character(server.vault.db, 11)

        body = _owner(server).get(f"{API}/users/me/config").json()

        assert body["similarity_character"] == 11

    def test_the_config_patch_writes_it_to_the_vault(self, server):
        client = _owner(server)

        response = client.patch(
            f"{API}/users/me/config", json={"similarity_character": 5}
        )

        assert response.status_code == 200, response.text
        assert library_settings_service.get_similarity_character(server.vault.db) == 5

    def test_patching_it_does_not_touch_the_hub_copy(self, server):
        client = _owner(server)

        client.patch(f"{API}/users/me/config", json={"similarity_character": 5})

        in_hub = server.hub_engine.run_immediate_read_task(
            lambda session: session.exec(select(User)).first().similarity_character
        )
        assert in_hub is None, "the hub must not become a second, stale home for it"

    def test_it_can_be_cleared_through_the_api(self, server):
        client = _owner(server)
        client.patch(f"{API}/users/me/config", json={"similarity_character": 5})

        client.patch(f"{API}/users/me/config", json={"similarity_character": None})

        assert (
            library_settings_service.get_similarity_character(server.vault.db) is None
        )

    def test_patching_other_settings_still_works(self, server):
        """The split must not break the settings that stayed in the hub."""
        client = _owner(server)

        response = client.patch(
            f"{API}/users/me/config",
            json={"theme_mode": "dark", "similarity_character": 9},
        )

        assert response.status_code == 200, response.text
        body = client.get(f"{API}/users/me/config").json()
        assert body["theme_mode"] == "dark"
        assert body["similarity_character"] == 9


class TestItFollowsTheLibrary:
    def test_each_library_keeps_its_own_selection(self, server, tmp_path):
        """The bug this move exists to prevent, end to end.

        Two libraries, two selections. Before the move both would have read the
        same per-user number, so switching would have sorted against whoever
        happened to hold that id in the new library.
        """
        client = _owner(server)
        original = server.library_registry.active_library()
        client.patch(f"{API}/users/me/config", json={"similarity_character": 4})

        other = server.library_registry.create(str(tmp_path / "other"), "Other")
        try:
            server.library_switch.switch_to(other.uuid)

            # A fresh library has made no selection, and must not inherit one.
            assert (
                client.get(f"{API}/users/me/config").json()["similarity_character"]
                is None
            )

            client.patch(f"{API}/users/me/config", json={"similarity_character": 12})
            assert (
                client.get(f"{API}/users/me/config").json()["similarity_character"]
                == 12
            )
        finally:
            server.library_switch.switch_to(original.uuid)

        # Back on the first library, its own selection is intact.
        assert client.get(f"{API}/users/me/config").json()["similarity_character"] == 4

    def test_a_user_preference_does_follow_the_switch(self, server, tmp_path):
        """The other direction: theme is the user's, so it must not be per-library."""
        client = _owner(server)
        original = server.library_registry.active_library()
        client.patch(f"{API}/users/me/config", json={"theme_mode": "dark"})

        other = server.library_registry.create(str(tmp_path / "theme-other"), "Theme")
        try:
            server.library_switch.switch_to(other.uuid)
            assert client.get(f"{API}/users/me/config").json()["theme_mode"] == "dark"
        finally:
            server.library_switch.switch_to(original.uuid)


class TestTheOwnersPenalisedTable:
    """An emptied penalised table is the owner's choice, not a missing value.

    A missing or unreadable value still means the shipped defaults. A saved
    ``{}`` means "penalise nothing", and every reader has to agree: the settings
    response, request scoring and background scoring.
    """

    def test_a_missing_or_unreadable_value_reads_as_the_defaults(self, server):
        assert resolve_owner_penalised_tags(None) == DEFAULT_SMART_SCORE_PENALIZED_TAGS
        assert (
            resolve_owner_penalised_tags("not json")
            == DEFAULT_SMART_SCORE_PENALIZED_TAGS
        )

    def test_an_emptied_table_stays_empty_for_every_reader(self, server):
        client = _owner(server)

        response = client.patch(
            f"{API}/users/me/config", json={"smart_score_penalised_tags": {}}
        )

        assert response.status_code == 200, response.text
        config = client.get(f"{API}/users/me/config").json()
        assert config["smart_score_penalised_tags"] == {}
        assert client.get(f"{API}/users/me/penalised-tags").json() == {
            "smart_score_penalised_tags": {}
        }
        owner_request = SimpleNamespace(
            state=SimpleNamespace(auth_user_id=server.auth.user.id), cookies={}
        )
        assert get_smart_score_penalised_tags_from_request(server, owner_request) == {}
        assert resolve_penalised_tag_weights(server.auth) == {}


class TestTheSettingsFingerprint:
    """The library records *that* the owner's weights changed, never what they are.

    Penalised and hidden tags say what someone collects and what they hide. A
    library folder is made to be copied, moved and handed to other people, so
    nothing derived from those settings may be recoverable from it.
    """

    def test_the_vault_stores_a_hash_and_not_the_tags(self, server):
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        secret_tags = {"a-very-distinctive-tag": 0.5, "another-private-one": 0.9}
        library = server.library_registry.active_library()
        reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, secret_tags
        )

        stored = server.vault.db.run_immediate_read_task(
            lambda session: session.exec(select(LibrarySettings)).first()
        )
        assert stored.settings_fingerprint
        for tag in secret_tags:
            assert tag not in stored.settings_fingerprint

    def test_the_whole_vault_file_contains_no_tag_names(self, server, tmp_path):
        """The real test of the claim: grep the bytes a recipient would receive."""
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        marker = "zzz-private-marker-tag"
        library = server.library_registry.active_library()
        reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {marker: 0.7}
        )

        vault_bytes = open(
            os.path.join(server.vault.image_root, "vault.db"), "rb"
        ).read()
        assert marker.encode() not in vault_bytes

    def test_the_key_is_not_in_the_library(self, server):
        """Without the salt the hash is unguessable, so the salt must stay in the hub."""
        library = server.library_registry.active_library()
        vault_bytes = open(
            os.path.join(server.vault.image_root, "vault.db"), "rb"
        ).read()

        assert library.settings_salt
        assert library.settings_salt.encode() not in vault_bytes

    def test_identical_settings_hash_differently_per_library(self, server, tmp_path):
        """The salt is per library, so one fingerprint says nothing about another."""
        from pixlstash.services.library_settings_service import (
            compute_settings_fingerprint,
        )

        first = server.library_registry.active_library()
        second = server.library_registry.create(str(tmp_path / "salted"), "Salted")
        settings = {"blurry": 0.5}

        assert compute_settings_fingerprint(
            first.settings_salt, settings
        ) != compute_settings_fingerprint(second.settings_salt, settings)

    def test_the_same_settings_hash_the_same_way(self, server):
        from pixlstash.services.library_settings_service import (
            compute_settings_fingerprint,
        )

        salt = "a-salt"
        # Order must not matter: a reordered payload is not a change.
        assert compute_settings_fingerprint(salt, {"a": 1.0, "b": 2.0}) == (
            compute_settings_fingerprint(salt, {"b": 2.0, "a": 1.0})
        )

    def test_a_changed_weight_changes_the_hash(self, server):
        from pixlstash.services.library_settings_service import (
            compute_settings_fingerprint,
        )

        salt = "a-salt"
        assert compute_settings_fingerprint(salt, {"a": 1.0}) != (
            compute_settings_fingerprint(salt, {"a": 1.5})
        )

    def test_removing_a_tag_from_tagging_changes_the_hash(self, server):
        """A removed tag stops counting, so the cached scores are stale too."""
        from pixlstash.services.library_settings_service import (
            compute_settings_fingerprint,
        )

        salt = "a-salt"
        weights = {"flux chin": 1.0, "waxy skin": 0.5}
        assert compute_settings_fingerprint(salt, weights) != (
            compute_settings_fingerprint(salt, weights, frozenset({"flux chin"}))
        )

    def test_nothing_removed_keeps_the_hash_libraries_already_store(self, server):
        """Otherwise every library would re-score in full on its next open."""
        from pixlstash.services.library_settings_service import (
            compute_settings_fingerprint,
        )

        salt = "a-salt"
        stored_before = hmac.new(
            salt.encode("utf-8"), b'{"a":1.0,"b":0.5}', hashlib.sha256
        ).hexdigest()
        assert (
            compute_settings_fingerprint(salt, {"b": 0.5, "a": 1.0}, frozenset())
            == stored_before
        )


class TestCatchingUpADormantLibrary:
    def test_a_weights_change_while_closed_invalidates_on_open(self, server):
        """The gap this closes: a dormant library never hears about the change."""
        from pixlstash.db_models import Picture
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        library = server.library_registry.active_library()
        reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}
        )

        def _add_scored_picture(session: Session):
            picture = Picture(file_path="/tmp/dormant.png", smart_score=4.5)
            session.add(picture)
            session.commit()
            session.refresh(picture)
            return picture.id

        picture_id = server.vault.db.run_task(_add_scored_picture)

        # The owner re-weights while this library is closed; on reopening, the
        # fingerprint no longer matches.
        changed = reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.9}
        )

        assert changed is True
        score = server.vault.db.run_immediate_read_task(
            lambda session: session.get(Picture, picture_id).smart_score
        )
        assert score is None, "the stale cached score should have been cleared"

    def test_no_change_leaves_scores_alone(self, server):
        """Over-invalidating would re-score the whole library on every open."""
        from pixlstash.db_models import Picture
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        library = server.library_registry.active_library()
        reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}
        )

        def _add(session: Session):
            picture = Picture(file_path="/tmp/unchanged.png", smart_score=3.3)
            session.add(picture)
            session.commit()
            session.refresh(picture)
            return picture.id

        picture_id = server.vault.db.run_task(_add)

        changed = reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}
        )

        assert changed is False
        score = server.vault.db.run_immediate_read_task(
            lambda session: session.get(Picture, picture_id).smart_score
        )
        assert score == pytest.approx(3.3)

    def test_emptying_the_table_while_closed_invalidates_on_open(self, server):
        """Removing every penalised tag is a change, not a return to the defaults."""
        client = _owner(server)
        library = server.library_registry.active_library()
        library_settings_service.reconcile_settings_fingerprint(
            server.vault.db,
            library.settings_salt,
            resolve_owner_penalised_tags(None),
        )

        def _add(session: Session):
            picture = Picture(file_path="/tmp/emptied.png", smart_score=4.4)
            session.add(picture)
            session.commit()
            session.refresh(picture)
            return picture.id

        picture_id = server.vault.db.run_task(_add)
        response = client.patch(
            f"{API}/users/me/config", json={"smart_score_penalised_tags": {}}
        )
        assert response.status_code == 200, response.text

        server.reconcile_library_settings(server.vault, library)

        score = server.vault.db.run_immediate_read_task(
            lambda session: session.get(Picture, picture_id).smart_score
        )
        assert score is None, "an emptied table must not hash like the defaults"

    def test_a_tag_removed_while_closed_invalidates_on_open(self, server):
        """Opening a library reconciles against the tags removed from tagging too."""
        server.reconcile_library_settings(server.vault)

        def _add(session: Session):
            picture = Picture(file_path="/tmp/removed-tag.png", smart_score=2.2)
            session.add(picture)
            session.commit()
            session.refresh(picture)
            return picture.id

        picture_id = server.vault.db.run_task(_add)
        try:
            server.vault.db.set_tag_exclusions(frozenset({"flux chin"}))
            server.reconcile_library_settings(server.vault)
            score = server.vault.db.run_immediate_read_task(
                lambda session: session.get(Picture, picture_id).smart_score
            )
            assert score is None, "removing a tag should have cleared the score"
        finally:
            server.vault.db.set_tag_exclusions(frozenset())
            server.reconcile_library_settings(server.vault)

    def test_the_first_ever_reconcile_invalidates_nothing(self, server):
        """A library that has never recorded one has nothing to repair."""
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        library = server.library_registry.active_library()

        def _clear(session: Session):
            row = session.exec(select(LibrarySettings)).first()
            row.settings_fingerprint = None
            session.add(row)
            session.commit()

        server.vault.db.run_task(_clear)

        assert (
            reconcile_settings_fingerprint(
                server.vault.db, library.settings_salt, {"blurry": 0.5}
            )
            is False
        )

    def test_a_missing_salt_is_skipped_rather_than_hashed_unkeyed(self, server):
        """An unkeyed hash in a portable folder is the thing being avoided."""
        from pixlstash.services.library_settings_service import (
            reconcile_settings_fingerprint,
        )

        assert (
            reconcile_settings_fingerprint(server.vault.db, None, {"blurry": 0.5})
            is False
        )


def _stored_fingerprint(server):
    return server.vault.db.run_immediate_read_task(
        lambda session: (
            session.exec(select(LibrarySettings)).first().settings_fingerprint
        )
    )


def _scored_picture(server, path: str, score: float) -> int:
    def _add(session: Session):
        picture = Picture(file_path=path, smart_score=score)
        session.add(picture)
        session.commit()
        session.refresh(picture)
        return picture.id

    return server.vault.db.run_task(_add)


def _score_of(server, picture_id: int):
    return server.vault.db.run_immediate_read_task(
        lambda session: session.get(Picture, picture_id).smart_score
    )


class TestAChangeMadeWhileOpen:
    """The open library repairs a weights change at once; the next open must not again.

    Without this, every penalised-tag edit re-scored the whole library at the
    following start, which is the "Smart score N / N" nobody asked for.
    """

    def test_the_next_open_does_not_invalidate_after_a_patch(self, server):
        """End to end: the setting is saved, then the library is opened again."""
        client = _owner(server)
        # Bring the fingerprint in line with the owner's current weights first,
        # so the only change the reopen can see is the one this PATCH makes.
        server.reconcile_library_settings(server.vault)
        before = _stored_fingerprint(server)
        current = client.get(f"{API}/users/me/config").json()[
            "smart_score_penalised_tags"
        ]
        weight = 0.25 if (current or {}).get("blurry") != 0.25 else 0.75
        untouched = _scored_picture(server, "/tmp/while-open.png", 2.7)

        response = client.patch(
            f"{API}/users/me/config",
            json={"smart_score_penalised_tags": {"blurry": weight}},
        )
        assert response.status_code == 200, response.text
        assert _stored_fingerprint(server) != before, "the PATCH changed nothing"

        server.reconcile_library_settings(server.vault)

        assert _score_of(server, untouched) == pytest.approx(2.7)

    def test_it_moves_from_the_previous_weights_to_the_new(self, server):
        library = server.library_registry.active_library()
        library_settings_service.reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}
        )

        advanced = library_settings_service.advance_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}, {"blurry": 0.9}
        )

        assert advanced is True
        assert _stored_fingerprint(server) == (
            library_settings_service.compute_settings_fingerprint(
                library.settings_salt, {"blurry": 0.9}
            )
        )

    def test_a_repair_already_owed_is_not_erased(self, server):
        """A stale fingerprint is a debt; advancing past it would forgive it."""
        library = server.library_registry.active_library()
        # The library last saw 0.1, so it missed the move to 0.5 as well.
        library_settings_service.reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.1}
        )
        owed = _stored_fingerprint(server)

        advanced = library_settings_service.advance_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}, {"blurry": 0.9}
        )

        assert advanced is False
        assert _stored_fingerprint(server) == owed
        stale = _scored_picture(server, "/tmp/owed.png", 1.9)
        assert library_settings_service.reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.9}
        )
        assert _score_of(server, stale) is None

    def test_reaching_the_same_weights_again_writes_nothing(self, server, caplog):
        """Already at the new weights: no write, and no report of a debt.

        The stored value matches neither the old weights nor nothing, so the
        compare-and-set alone would read it as a repair owed and say so.
        """
        library = server.library_registry.active_library()
        library_settings_service.reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.9}
        )
        recorded = _stored_fingerprint(server)
        caplog.clear()

        assert (
            library_settings_service.advance_settings_fingerprint(
                server.vault.db,
                library.settings_salt,
                {"blurry": 0.5},
                {"blurry": 0.9},
            )
            is False
        )
        assert _stored_fingerprint(server) == recorded
        assert not [
            record for record in caplog.records if "owed a repair" in record.message
        ]

    def test_a_missing_salt_writes_nothing(self, server):
        library = server.library_registry.active_library()
        library_settings_service.reconcile_settings_fingerprint(
            server.vault.db, library.settings_salt, {"blurry": 0.5}
        )
        recorded = _stored_fingerprint(server)

        assert (
            library_settings_service.advance_settings_fingerprint(
                server.vault.db, None, {"blurry": 0.5}, {"blurry": 0.9}
            )
            is False
        )
        assert _stored_fingerprint(server) == recorded


class TestValueTypes:
    """A value a setting cannot hold is refused with a fixed sentence, unsaved."""

    @pytest.mark.parametrize(
        "key, value, detail",
        [
            ("columns", {}, "columns must be an integer"),
            ("columns", "example-big", "columns must be an integer"),
            ("columns", 2**70, "columns must be an integer"),
            ("stack_strictness", "nan", "stack_strictness must be a number"),
            ("max_vram_gb", [1], "max_vram_gb must be a number"),
            (
                "similarity_character",
                "example-word",
                "similarity_character must be an integer",
            ),
        ],
    )
    def test_a_value_of_the_wrong_type_is_refused(self, server, key, value, detail):
        client = _owner(server)
        before = client.get(f"{API}/users/me/config").json()[key]
        response = client.patch(f"{API}/users/me/config", json={key: value})
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == detail
        assert client.get(f"{API}/users/me/config").json()[key] == before

    def test_a_number_sent_as_text_or_a_float_is_stored(self, server):
        client = _owner(server)
        response = client.patch(
            f"{API}/users/me/config", json={"columns": "5", "stack_strictness": 0.5}
        )
        assert response.status_code == 200, response.text
        body = client.get(f"{API}/users/me/config").json()
        assert (body["columns"], body["stack_strictness"]) == (5, 0.5)


class TestTagChipSize:
    """The tag chip size is the owner's, kept in its own hub table."""

    def test_default_when_nothing_is_stored(self, server):
        body = _owner(server).get(f"{API}/users/me/config").json()
        assert body["tag_chip_size"] == "comfortable"

    def test_a_patch_round_trips(self, server):
        client = _owner(server)
        response = client.patch(
            f"{API}/users/me/config", json={"tag_chip_size": "large"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["updated"] is True
        assert client.get(f"{API}/users/me/config").json()["tag_chip_size"] == "large"

    def test_an_unknown_size_is_refused(self, server):
        client = _owner(server)
        response = client.patch(
            f"{API}/users/me/config", json={"tag_chip_size": "huge"}
        )
        assert response.status_code == 400, response.text
        body = client.get(f"{API}/users/me/config").json()
        assert body["tag_chip_size"] == "comfortable"

    def test_a_patch_refused_elsewhere_saves_no_chip_size(self, server):
        client = _owner(server)
        response = client.patch(
            f"{API}/users/me/config",
            json={"tag_chip_size": "large", "no_such_setting": 1},
        )
        assert response.status_code == 400, response.text
        body = client.get(f"{API}/users/me/config").json()
        assert body["tag_chip_size"] == "comfortable"


class TestVramSliderBound:
    """The card bounding ``max_vram_gb`` arrives in the same response as it.

    The slider took its bound from ``GET /workers/progress``, and sat at its
    2 GB floor until that heaviest of polls answered.
    """

    def test_the_card_comes_with_the_budget_it_bounds(self, server, monkeypatch):
        monkeypatch.setattr(config_routes, "query_total_vram_mb", lambda: 16384)
        body = _owner(server).get(f"{API}/users/me/config").json()
        assert body["vram_total_gb"] == 16.0
        assert body["max_vram_gb"] is not None

    def test_no_card_sends_no_total(self, server, monkeypatch):
        monkeypatch.setattr(config_routes, "query_total_vram_mb", lambda: 0)
        body = _owner(server).get(f"{API}/users/me/config").json()
        assert body["vram_total_gb"] is None
