"""Every read of the owner's saved ComfyUI address goes through the normaliser.

A ``comfyui_url`` saved by an older build, or written into the settings by hand,
may no longer pass ``normalize_comfyui_url``. ``comfyui_connections.current_base``
is the one read path: it raises ``SavedComfyUIAddressInvalid`` for such a value,
``routes.comfyui._comfyui_url`` answers that with a 409 whose detail is
``SAVED_ADDRESS_INVALID``, and nothing is sent. This module holds

* the guardrail: no module outside the owner reads the attribute directly;
* each route that builds a request on it answering the 409 without asking;
* the vault's training watcher sending nothing, not raising, and saying so once.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pixlstash.routes import comfyui as comfyui_routes
from pixlstash.routes import model_files as model_files_routes
from pixlstash.routes import model_folders as model_folders_routes
from pixlstash.routes import model_shelf as model_shelf_routes
from pixlstash.routes import test_hooks
from pixlstash.routes import training as training_routes
from pixlstash.server import Server
from pixlstash.services import comfyui_connections
from pixlstash.utils.comfyui_url import SAVED_ADDRESS_INVALID
from pixlstash.vault import Vault
from pixlstash import vault as vault_module
from tests.utils import quiesce_background_work
from tests.package_sources import package_sources

API = "/api/v1"
PACKAGE = Path(__file__).resolve().parent.parent / "pixlstash"

# The address an older build could save: the backslash makes urlsplit and
# urllib3 disagree on the host, so the normaliser refuses it today.
LEGACY_ADDRESS = "http://evil.example\\@127.0.0.1:18188/"
VALID_ADDRESS = "http://127.0.0.1:18188/"

# Modules that own the attribute: the connection list reads and writes it, and
# the settings writer compares the value it is about to store.
OWNERS = {
    "services/comfyui_connections.py",
    "utils/service/user_settings_utils.py",
}

# (module, enclosing function) reads still raw, each with the lane that owns
# the fix. Empty is the goal, and it is empty: a new entry needs a reason.
ALLOWED_RAW_READS: set[tuple[str, str]] = set()


def _is_user_like(node: ast.expr) -> bool:
    text = ast.unparse(node).lower()
    return "user" in text or "owner" in text


def raw_reads(source: str) -> list[tuple[str, int]]:
    """``[(enclosing function, line)]`` of each direct read of a saved address.

    A read is ``getattr(<anything>, "comfyui_url", ...)`` or a load of
    ``<receiver>.comfyui_url`` where the receiver names a user or owner (a run
    plan's ``plan.comfyui_url`` is a value already read through the
    normaliser, and an assignment is a write).
    """
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else scope
            )
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in {"getattr", "hasattr"}
                and len(child.args) >= 2
                and isinstance(child.args[1], ast.Constant)
                and child.args[1].value == "comfyui_url"
            ):
                found.append((scope, child.lineno))
            elif (
                isinstance(child, ast.Attribute)
                and child.attr == "comfyui_url"
                and isinstance(child.ctx, ast.Load)
                and _is_user_like(child.value)
            ):
                found.append((scope, child.lineno))
            visit(child, inner)

    visit(tree, "<module>")
    return found


def test_the_detector_sees_each_kind_of_raw_read_and_nothing_else():
    """The guardrail below is only as good as this: each form it must catch,
    and the forms that are not a read of the saved address."""
    source = (
        "def a(user):\n"
        "    return getattr(user, 'comfyui_url', None)\n"
        "def b(server):\n"
        "    return server._user.comfyui_url.rstrip('/')\n"
        "def c(owner):\n"
        "    return owner.comfyui_url\n"
        "def d(plan):\n"
        "    return plan.comfyui_url\n"
        "def e(user, url):\n"
        "    user.comfyui_url = url\n"
    )
    assert raw_reads(source) == [("a", 2), ("b", 4), ("c", 6)]


def test_no_module_reads_the_saved_address_around_the_normaliser():
    """A new ``getattr(user, "comfyui_url")`` would build a request on a value
    the normaliser was never asked about. Read it with
    ``comfyui_connections.current_base`` (``_comfyui_url`` in a route), or ask
    ``comfyui_connections.is_configured`` when only "is one saved" matters."""
    found: set[tuple[str, str]] = set()
    lines: list[str] = []
    for path in package_sources(PACKAGE):
        module = path.relative_to(PACKAGE).as_posix()
        if module in OWNERS:
            continue
        for scope, line in raw_reads(path.read_text(encoding="utf-8")):
            found.add((module, scope))
            if (module, scope) not in ALLOWED_RAW_READS:
                lines.append(f"pixlstash/{module}:{line} in {scope}")
    assert lines == [], "raw reads of the saved ComfyUI address:\n" + "\n".join(lines)
    # An allowance whose read is gone is removed, so the list only shrinks.
    assert ALLOWED_RAW_READS <= found, ALLOWED_RAW_READS - found


# ---------------------------------------------------------------------------
# The current_base contract
# ---------------------------------------------------------------------------


def test_with_nothing_saved_a_none_default_reads_as_none_and_raises_nothing():
    user = SimpleNamespace(comfyui_url=None, comfyui_connections=None)
    assert comfyui_connections.current_base(user, None) is None
    assert comfyui_connections.current_base(None, None) is None
    assert comfyui_connections.is_configured(user) is False
    assert comfyui_connections.is_configured(None) is False


def test_with_nothing_saved_a_route_reads_the_default_address_when_called(
    monkeypatch,
):
    """The e2e start-up points the default at its fake ComfyUI. A default read
    once, when the module loaded, would keep a test server asking ComfyUI's own
    port, where a developer's real ComfyUI may answer or take a run."""
    user = SimpleNamespace(comfyui_url=None, comfyui_connections=None)
    monkeypatch.setattr(
        comfyui_routes, "DEFAULT_COMFYUI_URL", comfyui_routes.DEFAULT_COMFYUI_URL
    )
    test_hooks.point_comfyui_default(VALID_ADDRESS)
    assert comfyui_routes._comfyui_url(user) == VALID_ADDRESS.rstrip("/")
    assert comfyui_routes._comfyui_url(user, None) is None


def test_is_configured_says_saved_even_for_an_address_that_will_be_refused():
    """It answers "is one saved" only; the read that follows is the 409."""
    user = SimpleNamespace(comfyui_url=LEGACY_ADDRESS, comfyui_connections=None)
    assert comfyui_connections.is_configured(user) is True


# ---------------------------------------------------------------------------
# Routes that build a request on it answer the 409 and send nothing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """One Server and one owner login, for every test in the module."""
    tmp = tempfile.TemporaryDirectory()
    config_path = f"{tmp.name}/server-config.json"
    with open(config_path, "w") as handle:
        json.dump({"port": 8000}, handle)
    server = Server(config_path)
    server.__enter__()
    try:
        owner = TestClient(server.api, raise_server_exceptions=True)
        r = owner.post(
            f"{API}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert r.status_code == 200, r.text
        quiesce_background_work(server)
        yield SimpleNamespace(server=server, owner=owner)
    finally:
        server.__exit__(None, None, None)
        tmp.cleanup()


def _set_saved(server, url):
    def change(user):
        user.comfyui_url = url
        user.comfyui_connections = None
        return True

    comfyui_connections.save(server, server.auth.get_user().id, change)


@pytest.fixture
def asked(env, monkeypatch):
    """Every way these routes reach ComfyUI, recorded instead of sent; the
    saved address put back to none after."""
    calls: list[str] = []

    def record(name):
        def sender(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was asked")

        return sender

    monkeypatch.setattr(model_shelf_routes, "probe_comfyui", record("version"))
    monkeypatch.setattr(
        model_folders_routes.comfyui_link_service, "locate", record("locate")
    )
    monkeypatch.setattr(model_files_routes, "fetch_object_info", record("object_info"))
    monkeypatch.setattr(training_routes, "comfyui_probe", record("probe"))
    yield calls
    _set_saved(env.server, None)


REFUSING_ROUTES = [
    ("GET", "/models/workflow-templates", None),
    ("GET", "/model-folders/comfyui-reads", None),
    (
        "POST",
        "/model-files/merge",
        {"keep": [{"model_id": 1, "folder_id": 1, "relpath": "x.safetensors"}]},
    ),
    ("POST", "/training/resolve", {"subject": "person", "picture_count": 10}),
]


@pytest.mark.parametrize("method, path, body", REFUSING_ROUTES)
def test_a_route_reading_an_invalid_saved_address_answers_409_and_asks_nothing(
    env, asked, method, path, body
):
    _set_saved(env.server, LEGACY_ADDRESS)
    r = env.owner.request(method, f"{API}{path}", json=body)
    assert (r.status_code, r.json()["detail"]) == (409, SAVED_ADDRESS_INVALID)
    assert asked == []
    assert "evil" not in r.text


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/models/workflow-templates", {"comfyui_version": None}),
        ("/model-folders/comfyui-reads", {"known": False}),
    ],
)
def test_with_no_address_saved_those_routes_answer_without_asking(
    env, asked, path, expected
):
    """No address is not an invalid one: the default is None, not ComfyUI's
    default port, so nothing is asked and the answer says ComfyUI is unknown."""
    _set_saved(env.server, None)
    r = env.owner.get(f"{API}{path}")
    assert r.status_code == 200, r.text
    assert {key: r.json()[key] for key in expected} == expected
    assert asked == []


# ---------------------------------------------------------------------------
# The vault's training watcher
# ---------------------------------------------------------------------------


def _watcher(user):
    return SimpleNamespace(
        auth_service=SimpleNamespace(get_user=lambda: user),
        _comfyui_address_refused=False,
    )


def test_the_training_watcher_sends_nothing_to_an_invalid_address_and_says_so_once(
    monkeypatch, caplog
):
    probed: list[str] = []
    monkeypatch.setattr(
        vault_module, "comfyui_probe", lambda client, url: probed.append(url)
    )
    user = SimpleNamespace(comfyui_url=LEGACY_ADDRESS, comfyui_connections=None)
    watcher = _watcher(user)

    with caplog.at_level(logging.INFO, logger=vault_module.logger.name):
        Vault._watch_comfyui(watcher)
        Vault._watch_comfyui(watcher)
    assert probed == []
    refused = [
        r
        for r in caplog.records
        if r.name == vault_module.logger.name and "is not valid" in r.getMessage()
    ]
    assert [r.levelno for r in refused] == [logging.WARNING]

    caplog.clear()
    user.comfyui_url = VALID_ADDRESS
    with caplog.at_level(logging.INFO, logger=vault_module.logger.name):
        Vault._watch_comfyui(watcher)
        Vault._watch_comfyui(watcher)
    assert probed == [VALID_ADDRESS.rstrip("/")] * 2
    recovered = [r for r in caplog.records if "no longer refused" in r.getMessage()]
    assert len(recovered) == 1


# ---------------------------------------------------------------------------
# The connection list says which saved address is the bad one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url, valid", [(LEGACY_ADDRESS, False), (VALID_ADDRESS, True)])
def test_the_connection_list_marks_a_saved_address_that_no_longer_passes(
    env, asked, url, valid
):
    _set_saved(env.server, url)
    r = env.owner.get(f"{API}/comfyui/connections")
    assert r.status_code == 200, r.text
    (connection,) = r.json()["connections"]
    assert (connection["url"], connection["address_valid"]) == (url, valid)
    assert asked == []


def test_the_response_schema_declares_address_valid_as_always_present():
    """The ComfyUI router is left out of the served OpenAPI document
    (server.py mounts it with ``include_in_schema=False``), so the contract is
    checked on the model's own serialization schema: the boolean is in every
    entry, not optional."""
    schema = comfyui_routes.ComfyUIConnection.model_json_schema(mode="serialization")
    assert schema["properties"]["address_valid"]["type"] == "boolean"
    assert "address_valid" in schema["required"]


def test_the_frontend_recognises_the_409_by_the_detail_the_backend_sends():
    """The API client prompts on this one 409 by comparing its detail, so a
    reworded sentence on either side would leave the prompt silently dead."""
    source = (PACKAGE.parent / "frontend/src/utils/apiClient.js").read_text(
        encoding="utf-8"
    )
    match = re.search(r"const SAVED_ADDRESS_INVALID =((?:\s*'[^']*'\s*\+?)+);", source)
    assert match, "SAVED_ADDRESS_INVALID not found in apiClient.js"
    assert "".join(re.findall(r"'([^']*)'", match.group(1))) == SAVED_ADDRESS_INVALID
