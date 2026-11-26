"""Tests for tagger_settings serialisation and patching in user_settings_utils."""

import json
import tempfile
import types

import pytest
from fastapi.testclient import TestClient

from pixlstash.server import Server
from pixlstash.utils.service.user_settings_utils import (
    serialize_user_config,
    apply_user_config_patch,
)


def _make_user(**kwargs):
    """Return a minimal fake User object with tagger_settings support."""
    user = types.SimpleNamespace(
        id=1,
        username="testuser",
        description=None,
        sort="date",
        descending=True,
        columns=4,
        sidebar_thumbnail_size=None,
        show_stars=True,
        show_face_bboxes=False,
        show_hand_bboxes=False,
        show_format=False,
        show_resolution=False,
        show_problem_icon=False,
        compact_mode=False,
        sidebar_docked=False,
        sidebar_pinned=True,
        hide_purge_snapshot_warning=False,
        sidebar_width=240,
        date_format="relative",
        theme_mode="dark",
        check_for_updates=True,
        hidden_tags=None,
        apply_tag_filter=False,
        smart_score_penalised_tags=None,
        keep_models_in_memory=True,
        max_vram_gb=None,
        tagger_settings=None,
        watermark_image=None,
        watermark_opacity=0.5,
        watermark_position="bottom_right",
        watermark_size_percent=10,
        export_caption_mode=None,
        embed_watermark=False,
        comfyui_url=None,
        public_url=None,
        stack_strictness=None,
        show_keyboard_hint=True,
        similarity_character=None,
        show_stacks=True,
    )
    for k, v in kwargs.items():
        setattr(user, k, v)
    return user


# ---------------------------------------------------------------------------
# serialize_user_config
# ---------------------------------------------------------------------------


def test_serialize_includes_tagger_settings_key():
    user = _make_user()
    config = serialize_user_config(user)
    assert "tagger_settings" in config


def test_serialize_tagger_settings_has_all_plugins():
    user = _make_user()
    config = serialize_user_config(user)
    ts = config["tagger_settings"]
    assert "plugins" in ts
    assert "active_description_plugin" in ts
    plugin_names = set(ts["plugins"].keys())
    assert {"wd14", "pixlstash_tagger", "florence2"}.issubset(plugin_names)


def test_serialize_fills_defaults_when_tagger_settings_null():
    user = _make_user(tagger_settings=None)
    config = serialize_user_config(user)
    ts = config["tagger_settings"]
    assert ts["active_description_plugin"] == "florence2"
    assert "wd14" in ts["plugins"]
    assert "enabled" in ts["plugins"]["wd14"]


def test_serialize_respects_stored_tagger_settings():
    stored = json.dumps(
        {
            "active_description_plugin": "florence2",
            "plugins": {
                "wd14": {"enabled": True, "params": {"threshold": 0.72}},
            },
        }
    )
    user = _make_user(tagger_settings=stored)
    config = serialize_user_config(user)
    ts = config["tagger_settings"]
    assert ts["plugins"]["wd14"]["enabled"] is True
    assert ts["plugins"]["wd14"]["params"]["threshold"] == 0.72


# ---------------------------------------------------------------------------
# apply_user_config_patch - tagger_settings
# ---------------------------------------------------------------------------


def test_patch_sets_plugin_enabled_flag():
    user = _make_user()
    patch = {"tagger_settings": {"plugins": {"wd14": {"enabled": True}}}}
    changed = apply_user_config_patch(user, patch)
    assert changed
    stored = json.loads(user.tagger_settings)
    assert stored["plugins"]["wd14"]["enabled"] is True


def test_patch_updates_plugin_param():
    user = _make_user()
    patch = {"tagger_settings": {"plugins": {"wd14": {"params": {"threshold": 0.42}}}}}
    apply_user_config_patch(user, patch)
    stored = json.loads(user.tagger_settings)
    assert stored["plugins"]["wd14"]["params"]["threshold"] == 0.42


def test_patch_sets_active_description_plugin():
    user = _make_user()
    patch = {"tagger_settings": {"active_description_plugin": "florence2"}}
    apply_user_config_patch(user, patch)
    stored = json.loads(user.tagger_settings)
    assert stored["active_description_plugin"] == "florence2"


def test_patch_sets_active_description_plugin_to_none():
    user = _make_user()
    patch = {"tagger_settings": {"active_description_plugin": None}}
    apply_user_config_patch(user, patch)
    stored = json.loads(user.tagger_settings)
    assert stored["active_description_plugin"] is None


def test_patch_returns_false_when_no_change():
    user = _make_user()
    # First patch to set a known state.
    apply_user_config_patch(
        user, {"tagger_settings": {"plugins": {"wd14": {"enabled": False}}}}
    )
    # Repeat the same patch - no change.
    changed = apply_user_config_patch(
        user, {"tagger_settings": {"plugins": {"wd14": {"enabled": False}}}}
    )
    assert not changed


def test_patch_rejects_unknown_plugin():
    user = _make_user()
    patch = {"tagger_settings": {"plugins": {"ghost_plugin": {"enabled": True}}}}
    with pytest.raises(ValueError, match="Unknown plugin"):
        apply_user_config_patch(user, patch)


def test_patch_rejects_unknown_param():
    user = _make_user()
    patch = {
        "tagger_settings": {"plugins": {"wd14": {"params": {"nonexistent_param": 99}}}}
    }
    with pytest.raises(ValueError, match="Unknown parameter"):
        apply_user_config_patch(user, patch)


def test_patch_rejects_unknown_active_description_plugin():
    user = _make_user()
    patch = {"tagger_settings": {"active_description_plugin": "ghost_plugin"}}
    with pytest.raises(ValueError):
        apply_user_config_patch(user, patch)


def test_patch_rejects_tag_only_plugin_as_active_description():
    user = _make_user()
    patch = {"tagger_settings": {"active_description_plugin": "wd14"}}
    with pytest.raises(ValueError):
        apply_user_config_patch(user, patch)


def test_patch_merges_without_overwriting_other_plugins():
    user = _make_user()
    # Enable pixlstash_tagger first.
    apply_user_config_patch(
        user,
        {"tagger_settings": {"plugins": {"pixlstash_tagger": {"enabled": True}}}},
    )
    # Now patch wd14 - pixlstash_tagger entry should be untouched.
    apply_user_config_patch(
        user, {"tagger_settings": {"plugins": {"wd14": {"enabled": True}}}}
    )
    stored = json.loads(user.tagger_settings)
    assert stored["plugins"]["pixlstash_tagger"]["enabled"] is True
    assert stored["plugins"]["wd14"]["enabled"] is True


def test_patch_accepts_json_string_as_tagger_settings():
    user = _make_user()
    patch_value = json.dumps({"plugins": {"wd14": {"enabled": True}}})
    apply_user_config_patch(user, {"tagger_settings": patch_value})
    stored = json.loads(user.tagger_settings)
    assert stored["plugins"]["wd14"]["enabled"] is True


def test_patch_refuses_malformed_json_string_with_a_fixed_sentence(caplog):
    # The decoder's message names where the text broke; it is logged for the
    # owner, and the 400 the settings route returns says only what is wrong.
    user = _make_user()
    with caplog.at_level(
        "WARNING", logger="pixlstash.utils.service.user_settings_utils"
    ):
        with pytest.raises(ValueError) as refused:
            apply_user_config_patch(user, {"tagger_settings": '{"plugins": ['})
    assert str(refused.value) == "tagger_settings must be a JSON object"
    assert "char 13" in caplog.text
    assert user.tagger_settings is None


@pytest.fixture(scope="module")
def owner():
    """A real server and the owner signed in, for the settings route itself."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = f"{tmp}/server-config.json"
        with open(config_path, "w") as fh:
            json.dump({"disable_background_workers": True}, fh)
        with Server(config_path) as server:
            client = TestClient(server.api, raise_server_exceptions=False)
            login = client.post(
                "/api/v1/login",
                json={"username": "owner", "password": "example-owner-password"},
            )
            assert login.status_code == 200, login.text
            yield client


def test_the_settings_route_refuses_deeply_nested_json_like_any_malformed_json(
    owner,
):
    """CWE-674: CPython's decoder raises RecursionError, not JSONDecodeError,
    on text nested past the recursion limit. The route answers it as it
    answers any other malformed tagger_settings string, not with a 500."""
    before = owner.get("/api/v1/users/me/config")
    assert before.status_code == 200, before.text
    accepted = owner.patch(
        "/api/v1/users/me/config",
        json={"tagger_settings": json.dumps({"plugins": {"wd14": {"enabled": True}}})},
    )
    assert accepted.status_code == 200, accepted.text
    stored = owner.get("/api/v1/users/me/config").json()["tagger_settings"]

    for malformed in ('{"plugins": [', "[" * 100000):
        refused = owner.patch(
            "/api/v1/users/me/config", json={"tagger_settings": malformed}
        )
        assert refused.status_code == 400, refused.text[:200]
        assert refused.json() == {"detail": "tagger_settings must be a JSON object"}

    after = owner.get("/api/v1/users/me/config").json()["tagger_settings"]
    assert after == stored
