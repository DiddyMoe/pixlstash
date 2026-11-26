"""The HTTP contract the app owes every client, whatever route it reaches.

Invariants pinned here:

* a path under the API prefix that no API route matches answers a JSON 404,
  for every method: never the SPA document with a 200, never a 405;
* the SPA catch-all serves files from the built frontend only, never from a
  sibling whose name starts with it, however the ``..`` is spelled;
* an unhandled 500 says ``Connection: close`` when the server is about to
  close the connection, so a keep-alive client never sends a request into a
  connection the server has dropped (checked on a real uvicorn);
* a client on an allowed origin can read ``Retry-After`` and
  ``Content-Disposition``; a refused origin is not let in;
* a handler's own 5xx says a fixed sentence and logs the cause: the
  exception's text (an OS error, a path, an upstream answer) is never in the
  body, whichever shape the handler has (an ``OSError`` caught around one
  call, a catch-all that rolls back first, a service refusal passed on, an
  upstream read that failed);
* the vault holds the server's library coordinator before it starts, so the
  start-up training reconcile can take a read lease for the runs it imports;
* a refused request's 422 answers each error's type, location and message,
  and in ``ctx`` only the bounds the route declared: never the value sent,
  a validator's exception or anything measured from what was sent;
* a handler that reads its own body answers a body that is not a JSON object
  (not JSON, not UTF-8, an array, a string) with a 422 naming which, as a
  route whose body FastAPI parses does: never a 500, and never a 200 that
  dropped the body;
* a client that leaves before its request body arrives is logged as having
  left, not as a server fault (checked on a real uvicorn).
"""

import errno
import json
import os
import socket
import tempfile
import threading
import time
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Union

import pytest
import uvicorn
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError

from pixlstash.db_models.picture import Picture
from pixlstash.db_models.reference_folder import ReferenceFolder
from pixlstash.routes import library_layout as library_layout_routes
from pixlstash.routes import reference_folders as reference_folder_routes
from pixlstash.routes.model_folders import move_refusal_error
from pixlstash.routes.workflows import _require_object_info
from pixlstash.server import Server
from pixlstash.services.model_mover import MoveRefused
from pixlstash.services.training_service import TrainingService

API = "/api/v1"
SPA_DOCUMENT = "<!doctype html><title>example-spa</title>"
METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")


def _write_config(directory: str) -> str:
    config_path = os.path.join(directory, "server-config.json")
    with open(config_path, "w") as handle:
        json.dump({"port": 8000}, handle)
    return config_path


@pytest.fixture(scope="module")
def env():
    """One Server with a stand-in built frontend, an owner session and a
    route that raises, as a real handler bug would."""
    tmp = tempfile.TemporaryDirectory()
    config_path = _write_config(tmp.name)
    dist_dir = os.path.join(tmp.name, "dist")
    os.makedirs(dist_dir)
    with open(os.path.join(dist_dir, "index.html"), "w") as handle:
        handle.write(SPA_DOCUMENT)
    server = Server(config_path)
    # The SPA fallback only answers when a built frontend exists; a checkout
    # without frontend/dist would make every unknown path a 404 and hide the
    # bug, so the stand-in is given explicitly.
    server._get_frontend_dist_dir = lambda: dist_dir

    def raise_example_failure():
        raise RuntimeError("example-unhandled failure")

    server.api.add_api_route("/example-unhandled", raise_example_failure)
    # Ahead of the SPA catch-all, which would otherwise match the path first.
    server.api.router.routes.insert(0, server.api.router.routes.pop())
    owner = TestClient(server.api, raise_server_exceptions=False)
    r = owner.post(
        f"{API}/login",
        json={"username": "owner", "password": "example-owner-password"},
    )
    assert r.status_code == 200, r.text
    try:
        yield server, owner
    finally:
        server.close()
        tmp.cleanup()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "path",
    [
        f"{API}/example-no-such-route",
        f"{API}/pictures/1/example-no-such-field/deeper",
        API,
        f"{API}/ws/example-no-such-socket",
    ],
)
def test_an_unknown_api_path_is_a_json_404(env, method, path):
    _server, owner = env
    r = owner.request(method, path)
    assert r.status_code == 404, (method, path, r.status_code, r.text[:200])
    if method != "HEAD":
        assert r.headers["content-type"].startswith("application/json"), r.headers
        assert r.json() == {"detail": "Not Found"}


def test_the_spa_and_the_api_still_answer(env):
    """Positive controls: the SPA document still serves a client-side route,
    a real API route still answers, and a real route's wrong method keeps its
    405 with the methods it does allow."""
    _server, owner = env
    r = owner.request("GET", "/example-client-side/route")
    assert r.status_code == 200, r.text[:200]
    assert r.text == SPA_DOCUMENT
    r = owner.request("GET", f"{API}/check-session")
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"].startswith("application/json")
    r = owner.request("PATCH", f"{API}/check-session")
    assert r.status_code == 405, r.text[:200]
    assert "GET" in r.headers["allow"]


def test_the_spa_never_serves_a_prefix_sibling_of_the_built_frontend(env):
    """A ``..`` sent percent-encoded is not folded away by the client and
    reaches the catch-all decoded, so it must not serve a file from a sibling
    whose name merely starts with the built frontend's (``dist-old`` next to
    ``dist``). The route needs no session, so the request carries none. A
    file in the build, reached directly or through a ``..`` that stays
    inside, is still served."""
    server, _owner = env
    anonymous = TestClient(server.api, raise_server_exceptions=False)
    dist_dir = server._get_frontend_dist_dir()
    # ``request``, not ``get``: the suite's conftest prefixes a ``get`` path
    # with the API prefix, which would never reach the catch-all.
    sibling = dist_dir + "-old"
    os.makedirs(sibling, exist_ok=True)
    with open(os.path.join(sibling, "example-sibling.txt"), "w") as handle:
        handle.write("example-sibling-file")
    with open(os.path.join(dist_dir, "example-asset.txt"), "w") as handle:
        handle.write("example-asset-file")
    for path in (
        "/%2e%2e/dist-old/example-sibling.txt",
        "/example-deeper/%2e%2e/%2e%2e/dist-old/example-sibling.txt",
    ):
        r = anonymous.request("GET", path)
        assert r.status_code == 200, (path, r.status_code)
        assert r.text == SPA_DOCUMENT, (path, r.text[:200])
    for path in ("/example-asset.txt", "/example-deeper/%2e%2e/example-asset.txt"):
        r = anonymous.request("GET", path)
        assert r.status_code == 200, (path, r.status_code)
        assert r.text == "example-asset-file", (path, r.text[:200])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_response(sock):
    """One HTTP/1.1 response off *sock* as ``(status_line, headers, body)``,
    or ``None`` when the server closed the connection before answering."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            return None
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status_line, *lines = head.decode("latin-1").split("\r\n")
    headers = {}
    for line in lines:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    while len(body) < int(headers.get("content-length", "0")):
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return status_line, headers, body


def _get(sock, path):
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    return _read_response(sock)


def _is_closed(sock, path):
    """True when the server has closed *sock*: a request on it gets no answer
    (end of stream) or the connection reset."""
    try:
        return _get(sock, path) is None
    except (ConnectionResetError, BrokenPipeError):
        return True


@pytest.fixture(scope="module")
def live_port(env):
    """The app on a real uvicorn, configured as ``Server.run`` builds it."""
    server, _owner = env
    port = _free_port()
    listener = uvicorn.Server(
        uvicorn.Config(
            server.api,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    thread = threading.Thread(target=listener.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not listener.started:
        assert thread.is_alive() and time.monotonic() < deadline, (
            "uvicorn did not start"
        )
        time.sleep(0.05)
    try:
        yield port
    finally:
        listener.should_exit = True
        thread.join(10)


def test_keep_alive_holds_across_ordinary_requests(live_port):
    """Control: two requests on one connection both answer, so the next test
    is about the 500, not about a server that never keeps a connection."""
    with socket.create_connection(("127.0.0.1", live_port), timeout=10) as sock:
        for _ in range(2):
            status_line, headers, _body = _get(sock, "/version")
            assert status_line.startswith("HTTP/1.1 200"), status_line
            assert headers.get("connection") != "close", headers


def test_an_unhandled_500_says_the_connection_closes(live_port):
    """The server closes the connection after an unhandled 500 (Starlette
    re-raises, uvicorn closes). The 500 must say so, and the client's next
    request, on a new connection as it was told, is answered."""
    with socket.create_connection(("127.0.0.1", live_port), timeout=10) as sock:
        status_line, headers, body = _get(sock, "/example-unhandled")
        assert status_line.startswith("HTTP/1.1 500"), status_line
        assert json.loads(body) == {"detail": "Internal Server Error"}
        assert b"example-unhandled" not in body
        assert headers.get("connection") == "close", (
            "the 500 left the client believing the connection persists",
            headers,
        )
        # A second request on the same connection: the server has closed it.
        # Without the header above this is the request a keep-alive client
        # loses.
        assert _is_closed(sock, "/version")
    with socket.create_connection(("127.0.0.1", live_port), timeout=10) as sock:
        status_line, _headers, _body = _get(sock, "/version")
        assert status_line.startswith("HTTP/1.1 200"), status_line


NOT_JSON = "The request body is not valid JSON."
NOT_AN_OBJECT = "The request body must be a JSON object."
BODIES_THAT_ARE_NOT_JSON_OBJECTS = [
    pytest.param(b"{example-not-json", NOT_JSON, id="malformed"),
    # A UTF-16 byte-order mark with an odd byte after it: UnicodeDecodeError.
    pytest.param(b"\xff\xfe\xfd", NOT_JSON, id="not-utf"),
    pytest.param(b"[1, 2]", NOT_AN_OBJECT, id="array"),
    pytest.param(b'"example-text"', NOT_AN_OBJECT, id="string"),
]
OWN_BODY_ROUTES = ("config", "character", "picture", "guest-scores")


@pytest.fixture(scope="module")
def own_body_routes(env, tmp_path_factory):
    """Every handler that reads its own body instead of declaring a model,
    as ``name -> (client, method, path, a body it accepts)``."""
    server, owner = env
    r = owner.post(f"{API}/characters", json={"name": "example-own-body"})
    assert r.status_code == 200, r.text
    character_id = r.json()["character"]["id"]
    picture_file = tmp_path_factory.mktemp("example-own-body") / "example.png"
    picture_file.write_bytes(b"example-bytes")

    def insert_picture(session):
        pic = Picture(
            file_path=str(picture_file),
            pixel_sha="example-own-body-sha",
            width=8,
            height=8,
        )
        session.add(pic)
        session.commit()
        return pic.id

    picture_id = server.vault.db.run_task(insert_picture)
    r = owner.post(
        f"{API}/users/me/token",
        json={"description": "example-guest", "scope": "READ"},
    )
    assert r.status_code == 200, r.text
    guest = TestClient(
        server.api,
        raise_server_exceptions=False,
        headers={"Authorization": f"Bearer {r.json()['token']}"},
    )
    described = {"description": "example-description"}
    return {
        "config": (owner, "PATCH", f"{API}/users/me/config", {}),
        "character": (owner, "PATCH", f"{API}/characters/{character_id}", described),
        "picture": (owner, "PATCH", f"{API}/pictures/{picture_id}", described),
        "guest-scores": (
            guest,
            "POST",
            f"{API}/pictures/guest-scores",
            {"session_id": "example-session", "scores": {}},
        ),
    }


@pytest.mark.parametrize("body, detail", BODIES_THAT_ARE_NOT_JSON_OBJECTS)
@pytest.mark.parametrize("route", OWN_BODY_ROUTES)
def test_a_body_that_is_not_a_json_object_is_a_422(
    own_body_routes, route, body, detail, caplog
):
    """Each used to answer 500 and log a server fault (config, characters,
    guest scores), or drop the body and answer 200 having changed nothing it
    was sent (a picture). The positive control is the same route taking a
    body that is an object."""
    client, method, path, accepted = own_body_routes[route]
    with caplog.at_level("ERROR", logger="pixlstash"):
        r = client.request(
            method, path, content=body, headers={"Content-Type": "application/json"}
        )
    assert r.status_code == 422, (route, r.status_code, r.text[:200])
    assert r.json() == {"detail": detail}, r.text[:200]
    # Only the 500 handler's line: the planner reads the stand-in picture's
    # bytes in the background and may log its own error meanwhile.
    messages = [rec.getMessage() for rec in caplog.records]
    assert not [m for m in messages if "Unhandled exception" in m], messages
    r = client.request(method, path, json=accepted)
    assert r.status_code == 200, (route, r.status_code, r.text[:200])


def test_a_picture_patch_by_query_with_a_json_header_and_no_body_still_applies(
    own_body_routes,
):
    """A client may send the JSON content type on a PATCH whose fields are
    all in the query; an empty body is no body, not a malformed one."""
    client, _method, path, _accepted = own_body_routes["picture"]
    r = client.patch(
        path,
        params={"description": "example-from-the-query"},
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text[:200]
    assert r.json()["picture"]["description"] == "example-from-the-query"


LEFT_EARLY = "the client closed the connection before the request body arrived"


def test_a_client_that_leaves_before_its_body_arrives_is_not_a_server_fault(
    env, live_port, caplog
):
    """A PATCH that promises a body and closes the connection part-way
    through it, as a closed tab does. The handler's body read raises
    ClientDisconnect; it must be logged as the client leaving, never reach
    the 500 handler as an unhandled exception with a traceback."""
    _server, owner = env
    cookie = "; ".join(f"{name}={value}" for name, value in owner.cookies.items())
    request = (
        f"PATCH {API}/users/me/config HTTP/1.1\r\nHost: testserver\r\n"
        f"Cookie: {cookie}\r\nContent-Type: application/json\r\n"
        'Content-Length: 4096\r\n\r\n{"example'
    )
    with caplog.at_level("INFO", logger="pixlstash"):
        with socket.create_connection(("127.0.0.1", live_port), timeout=10) as sock:
            sock.sendall(request.encode())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            messages = [rec.getMessage() for rec in caplog.records]
            if any(LEFT_EARLY in m or "Unhandled exception" in m for m in messages):
                break
            time.sleep(0.05)
    assert not [m for m in messages if "Unhandled exception" in m], messages
    assert [m for m in messages if LEFT_EARLY in m and "/users/me/config" in m], (
        "the handler never read the body, so the test proved nothing",
        messages,
    )


def test_an_allowed_origin_reads_the_download_name_and_a_refused_one_is_not_let_in(
    env,
):
    _server, owner = env
    r = owner.request(
        "GET", f"{API}/check-session", headers={"Origin": "http://localhost:5173"}
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    exposed = {
        name.strip().lower()
        for name in r.headers["access-control-expose-headers"].split(",")
    }
    assert {"retry-after", "content-disposition"} <= exposed, exposed
    r = owner.request(
        "GET", f"{API}/check-session", headers={"Origin": "http://example.invalid"}
    )
    assert "access-control-allow-origin" not in r.headers, r.headers


def test_the_vault_has_the_coordinator_before_the_startup_reconcile(monkeypatch):
    """A training run that finished while PixlStash was down is imported by
    the reconcile inside ``vault.start()``, and the import takes a read lease
    from the coordinator. So the vault must hold the server's coordinator
    when the reconcile runs, not only once the server is built."""
    seen = []
    reconcile = TrainingService.reconcile_on_startup

    def record_then_reconcile(service):
        vault = service._import_run.__self__
        seen.append(getattr(vault, "library_coordinator", None))
        return reconcile(service)

    monkeypatch.setattr(TrainingService, "reconcile_on_startup", record_then_reconcile)
    with tempfile.TemporaryDirectory() as directory:
        server = Server(_write_config(directory))
        try:
            assert server.vault.library_coordinator is server.library_coordinator
            assert len(seen) == 1, seen
            assert seen[0] is server.library_coordinator, (
                "the start-up reconcile ran before the vault had the coordinator"
            )
        finally:
            server.close()


# The text a failure carries that must reach the log and never the body: an
# OS error naming a server path, as a real one would.
CAUSE = "example-cause /example/server/path"


def _caused(records) -> bool:
    """True when a log record carries the cause, with its traceback."""
    return any(CAUSE in r.getMessage() and r.exc_info for r in records)


def _fail_for(real, target, error):
    """*real*, except that it raises *error* for *target*: the forced failure
    hits the handler's own call and nothing else in the process."""

    def failing(path, *args, **kwargs):
        if os.path.abspath(path) == os.path.abspath(target):
            raise error
        return real(path, *args, **kwargs)

    return failing


def _reference_folder(server, folder: str) -> int:
    os.makedirs(folder, exist_ok=True)

    def insert(session):
        row = ReferenceFolder(
            folder=folder, label="example", status="active", last_scanned=time.time()
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id

    return server.vault.db.run_task(insert)


def test_an_os_error_around_one_call_answers_a_fixed_500(
    env, tmp_path, monkeypatch, caplog
):
    """Shape: ``except OSError`` around the one call that failed."""
    server, owner = env
    monkeypatch.setattr(server, "running_in_docker", lambda: False)
    target = str(tmp_path / "example-new-folder")
    monkeypatch.setattr(
        os, "mkdir", _fail_for(os.mkdir, target, OSError(errno.EIO, CAUSE))
    )
    r = owner.post(f"{API}/filesystem/folders", json={"path": target})
    assert r.status_code == 500, r.text
    assert r.json() == {"detail": "Could not create the folder."}
    assert _caused(caplog.records), [rec.getMessage() for rec in caplog.records]


def test_a_failed_relocation_answers_a_fixed_500_after_the_rollback(
    env, tmp_path, monkeypatch, caplog
):
    """Shapes: an ``OSError`` creating the destination, and the catch-all that
    moves the files back before it answers."""
    server, owner = env
    monkeypatch.setattr(server, "running_in_docker", lambda: False)
    old_root = tmp_path / "example-old"
    folder_id = _reference_folder(server, str(old_root))
    (old_root / "example-note.txt").write_text("example", encoding="utf-8")

    new_root = str(tmp_path / "example-new-a")
    monkeypatch.setattr(
        os, "makedirs", _fail_for(os.makedirs, new_root, OSError(errno.EIO, CAUSE))
    )
    r = owner.post(
        f"{API}/reference-folders/{folder_id}/relocate",
        json={"destination_folder": new_root},
    )
    assert r.status_code == 500, r.text
    assert r.json() == {"detail": "Could not create the destination folder."}
    assert _caused(caplog.records)
    monkeypatch.undo()
    caplog.clear()

    monkeypatch.setattr(server, "running_in_docker", lambda: False)
    run_task = server.vault.db.run_task

    def failing_apply(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "apply_relocation":
            raise RuntimeError(CAUSE)
        return run_task(func, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", failing_apply)
    r = owner.post(
        f"{API}/reference-folders/{folder_id}/relocate",
        json={"destination_folder": str(tmp_path / "example-new-b")},
    )
    assert r.status_code == 500, r.text
    assert r.json() == {
        "detail": (
            "Relocation failed; the files were moved back. The server log says why."
        )
    }
    assert _caused(caplog.records)
    assert (old_root / "example-note.txt").is_file()


def test_a_space_refusal_says_a_fixed_sentence_and_a_4xx_keeps_its_own(caplog):
    """Shape: a service refusal passed on with its status. A 507 names a server
    folder and its free space; a 4xx refusal is the mover's own sentence."""
    try:
        raise MoveRefused(f"Not enough space in {CAUSE}.", status_code=507)
    except MoveRefused as exc:
        error = move_refusal_error(exc, "Nothing was moved.")
    assert error.status_code == 507
    assert error.detail == (
        "There is not enough free space at the destination, or its free space "
        "could not be read. Nothing was moved."
    )
    assert _caused(caplog.records)
    error = move_refusal_error(
        MoveRefused("PixlStash does not delete files from that folder.", 409),
        "Nothing was moved.",
    )
    assert (error.status_code, error.detail) == (
        409,
        "PixlStash does not delete files from that folder.",
    )


def test_comfyui_out_of_reach_is_a_fixed_503(caplog):
    """Shape: an upstream read that failed. Its reason names the saved
    ComfyUI address and the transport error, so it goes to the log."""
    with pytest.raises(HTTPException) as raised:
        _require_object_info((None, CAUSE), "which nodes it has")
    assert raised.value.status_code == 503
    assert raised.value.detail == (
        "PixlStash could not ask ComfyUI which nodes it has: ComfyUI did not answer."
    )
    assert any(CAUSE in r.getMessage() for r in caplog.records)


def test_a_refused_layout_or_folder_names_only_the_servers_own_words(env):
    """Shape: a 4xx about text the caller sent. The detail is the server's
    sentence and vocabulary; the caller's text is not repeated back."""
    _server, owner = env
    sent = "example-not-a-facet"
    r = owner.patch(f"{API}/server-config/layout", json={"layout": sent})
    assert r.status_code == 400, r.text
    assert r.json() == {
        "detail": (
            "The layout names something that is not a layout facet; expected "
            "one of project, person, set, tag."
        )
    }
    sent = "../example-unfiled"
    r = owner.patch(f"{API}/server-config/layout", json={"layout_unfiled": sent})
    assert r.status_code == 400, r.text
    # Not even the cleaned-up suggestion: it is the caller's text again.
    assert "example-unfiled" not in r.text, r.text
    assert r.json() == {"detail": library_layout_routes.UNFILED_NOT_SAFE}
    assert owner.get(f"{API}/server-config/layout").json()["layout_unfiled"] != sent

    sent = "example-unlisted/folder"
    r = owner.post(f"{API}/library/folders/open", json={"path": sent})
    assert r.status_code == 404, r.text
    assert r.json() == {"detail": "No picture is filed in that folder."}


def test_a_reference_folder_layout_refusal_names_only_the_servers_own_words(
    env, tmp_path, caplog
):
    """Shape: the library layout's refusals, on a reference folder. The sent
    unfiled name, and a stored layout that no longer parses, go to the log."""
    server, owner = env
    folder_id = _reference_folder(server, str(tmp_path / "example-ref-layout"))
    sent = "../example-unfiled"
    with caplog.at_level("INFO", logger="pixlstash.routes.reference_folders"):
        r = owner.patch(
            f"{API}/reference-folders/{folder_id}", json={"layout_unfiled": sent}
        )
    assert r.status_code == 400, r.text
    assert "example-unfiled" not in r.text, r.text
    assert r.json() == {"detail": library_layout_routes.UNFILED_NOT_SAFE}
    assert any(
        sent in rec.getMessage() and str(folder_id) in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]
    caplog.clear()

    def store_layout(session):
        row = session.get(ReferenceFolder, folder_id)
        row.layout = "example-not-a-facet"
        session.add(row)
        session.commit()

    server.vault.db.run_task(store_layout)
    with caplog.at_level("INFO", logger="pixlstash.routes.reference_folders"):
        r = owner.patch(
            f"{API}/reference-folders/{folder_id}",
            json={"layout_unfiled": "Example"},
        )
    assert r.status_code == 400, r.text
    assert "example-not-a-facet" not in r.text, r.text
    assert r.json() == {"detail": library_layout_routes.NOT_A_LAYOUT_FACET}
    assert any("example-not-a-facet" in rec.getMessage() for rec in caplog.records), [
        rec.getMessage() for rec in caplog.records
    ]


def test_a_relocated_file_that_cannot_be_read_is_reported_in_a_fixed_sentence(
    env, tmp_path, monkeypatch, caplog
):
    """Shape: an OS error inside a 200 body. The relocation report says the
    file could not be read; the OS's text goes to the log with its traceback."""
    server, owner = env
    monkeypatch.setattr(server, "running_in_docker", lambda: False)
    old_root = tmp_path / "example-ref-old"
    new_root = tmp_path / "example-ref-new"
    folder_id = _reference_folder(server, str(old_root))
    new_root.mkdir()
    (new_root / "example.png").write_bytes(b"example-bytes")

    def insert_picture(session):
        pic = Picture(
            file_path=str(old_root / "example.png"),
            reference_folder_id=folder_id,
            pixel_sha="example-sha",
            width=8,
            height=8,
        )
        session.add(pic)
        session.commit()
        return pic.id

    picture_id = server.vault.db.run_task(insert_picture)
    cause = OSError(errno.EIO, CAUSE)
    monkeypatch.setattr(
        reference_folder_routes.ImageUtils,
        "calculate_hash_from_file_path",
        _fail_for(
            reference_folder_routes.ImageUtils.calculate_hash_from_file_path,
            str(new_root / "example.png"),
            cause,
        ),
    )
    with caplog.at_level("WARNING", logger="pixlstash.routes.reference_folders"):
        r = owner.patch(
            f"{API}/reference-folders/{folder_id}", json={"folder": str(new_root)}
        )
    assert r.status_code == 200, r.text
    unmatched = r.json()["relocation"]["unmatched"]
    assert [(u["picture_id"], u["error"]) for u in unmatched] == [
        (picture_id, reference_folder_routes.RELOCATED_FILE_UNREADABLE)
    ]
    assert CAUSE not in r.text, r.text
    assert any(
        rec.exc_info
        and rec.exc_info[1] is cause
        and str(picture_id) in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_a_metadata_import_that_breaks_a_constraint_is_a_fixed_409(
    env, tmp_path, monkeypatch, caplog
):
    """Shape: a driver error on commit. Its text names the statement and the
    bound values, so the 409 is the server's sentence and the cause is logged."""
    server, owner = env
    folder_id = _reference_folder(server, str(tmp_path / "example-ref-meta"))
    cause = IntegrityError("INSERT INTO example", {"example": CAUSE}, Exception(CAUSE))
    run_task = server.vault.db.run_task

    def failing_commit(func, *args, **kwargs):
        if getattr(func, "__name__", "") != "import_metadata":
            return run_task(func, *args, **kwargs)

        def import_metadata(session):
            def commit():
                raise cause

            monkeypatch.setattr(session, "commit", commit)
            return func(session)

        return run_task(import_metadata, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", failing_commit)
    with caplog.at_level("WARNING", logger="pixlstash.routes.reference_folders"):
        r = owner.post(
            f"{API}/reference-folders/{folder_id}/metadata/import",
            json={"types": ["tags"]},
        )
    assert r.status_code == 409, r.text
    assert r.json() == {"detail": reference_folder_routes.METADATA_IMPORT_FAILED}
    assert CAUSE not in r.text and "INSERT" not in r.text, r.text
    assert any(
        rec.exc_info and rec.exc_info[1] is cause and str(folder_id) in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


class _ExampleTagA(BaseModel):
    kind: Literal["example-a"]


class _ExampleTagB(BaseModel):
    kind: Literal["example-b"]


class _ExampleBounded(BaseModel):
    """One field per kind of check a request model declares."""

    count: Annotated[int, Field(le=5)]
    share: Annotated[float, Field(gt=0.5)]
    mode: Literal["example-x", "example-y"]
    name: Annotated[str, Field(pattern=r"^[a-z]+$")]
    ids: Annotated[list[int], Field(max_length=2)]
    since: Annotated[date, Field(gt=date(2000, 1, 1))]
    price: Annotated[Decimal, Field(gt=Decimal("1.5"))]
    tagged: Annotated[Union[_ExampleTagA, _ExampleTagB], Field(discriminator="kind")]
    note: str

    @field_validator("note")
    @classmethod
    def _refuse(cls, value):
        raise ValueError(f"{value!r} is not a note")


def test_a_refused_request_answers_the_declared_bounds_never_the_value_sent(env):
    """Shape: a 422 keeps FastAPI's ``type``, ``loc`` and ``msg`` and, in
    ``ctx``, only what the route declared (a bound, a pattern, the accepted
    values). Never ``input``, a validator's exception, the tag sent or the
    length of what was sent."""
    server, owner = env
    sent = "example-sent-text"

    def accept(body: _ExampleBounded):
        return {"ok": True}

    # Under the API prefix, which the suite's TestClient adds to a bare path.
    server.api.add_api_route(f"{API}/example-bounded", accept, methods=["POST"])
    # Ahead of the SPA catch-all, which would otherwise match the path first.
    server.api.router.routes.insert(0, server.api.router.routes.pop())
    r = owner.post(
        f"{API}/example-bounded",
        json={
            "count": 9,
            "share": 0.25,
            "mode": sent,
            "name": sent,
            "ids": [1, 2, 3],
            "since": "1999-12-31",
            "price": "1.0",
            "tagged": {"kind": sent},
            "note": sent,
        },
    )
    assert r.status_code == 422, r.text
    assert sent not in r.text, r.text
    answered = {error["loc"][-1]: error for error in r.json()["detail"]}
    assert {name: set(error) for name, error in answered.items()} == {
        "count": {"type", "loc", "msg", "ctx"},
        "share": {"type", "loc", "msg", "ctx"},
        "mode": {"type", "loc", "msg", "ctx"},
        "name": {"type", "loc", "msg", "ctx"},
        "ids": {"type", "loc", "msg", "ctx"},
        "since": {"type", "loc", "msg", "ctx"},
        # A Decimal bound is no JSON value: the message states it instead.
        "price": {"type", "loc", "msg"},
        "tagged": {"type", "loc", "msg"},
        "note": {"type", "loc", "msg"},
    }
    assert {name: error.get("ctx") for name, error in answered.items()} == {
        "count": {"le": 5},
        "share": {"gt": 0.5},
        "mode": {"expected": "'example-x' or 'example-y'"},
        "name": {"pattern": "^[a-z]+$"},
        "ids": {"max_length": 2},
        "since": {"gt": "2000-01-01"},
        "price": None,
        "tagged": None,
        "note": None,
    }
    assert answered["price"]["msg"] == "Input should be greater than 1.5"
    assert answered["note"]["msg"] == "Value error, '*' is not a note"

    # A real route's bound is answered the same way.
    r = owner.get(f"{API}/operations/{2**70}")
    assert r.status_code == 422, r.text
    assert str(2**70) not in r.text, r.text
    assert r.json()["detail"] == [
        {
            "type": "less_than_equal",
            "loc": ["path", "operation_id"],
            "msg": f"Input should be less than or equal to {2**63 - 1}",
            "ctx": {"le": 2**63 - 1},
        }
    ]
