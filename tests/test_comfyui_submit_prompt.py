"""Payload tests for ``_submit_comfyui_prompt`` (issue #628).

The POST /prompt payload must NOT carry ``extra_data.extra_pnginfo.workflow``:
that PNG chunk is where the ComfyUI frontend stores the *UI* node graph, and
it feeds the chunk to ``loadGraphData`` unguarded when an image is dropped on
the canvas. Embedding the API-format graph there breaks drag-back-in. ComfyUI
writes the correct ``prompt`` chunk itself, so nothing needs to be embedded.
"""

import io
import json as stdlib_json
import logging
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import requests
import urllib3

import pixlstash.services.comfyui_service as comfyui_service
from pixlstash.resources import comfyui_tenant
from pixlstash.resources.comfyui_tenant import ComfyUIProbe
from pixlstash.utils import bounded_http
from pixlstash.utils.bounded_http import BoundedResponse
from tests.test_bounded_http import _rest_of_body

WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 12345, "steps": 20, "model": ["4", 0]},
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    # PixlStash-internal key: must be stripped before submission, because
    # ComfyUI iterates top-level entries as nodes and crashes on non-dicts.
    "pixlstash_output_nodes": ["3"],
}


def _accepted(url: str) -> BoundedResponse:
    return BoundedResponse(url, 200, b'{"prompt_id": "abc123"}')


def _capture_submit_payload(monkeypatch, **submit_kwargs) -> dict:
    captured = {}

    def fake_request(method, url, json=None, proxies=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = json
        captured["redirects"] = kwargs.get("redirects", 0)
        captured["proxies"] = proxies
        return _accepted(url)

    monkeypatch.setattr(comfyui_service, "bounded_request", fake_request)
    result = comfyui_service._submit_comfyui_prompt(
        "http://comfy.test", WORKFLOW, **submit_kwargs
    )
    assert result == {"prompt_id": "abc123"}
    return captured


class TestSubmitPayload:
    def test_no_extra_pnginfo_workflow_is_embedded(self, monkeypatch):
        # The regression under test: the API graph must not be placed in the
        # PNG's ``workflow`` chunk via extra_pnginfo (issue #628).
        captured = _capture_submit_payload(monkeypatch)
        payload = captured["payload"]
        extra_pnginfo = payload.get("extra_data", {}).get("extra_pnginfo", {})
        assert "workflow" not in extra_pnginfo
        assert "extra_data" not in payload

    def test_prompt_carries_the_cleaned_workflow(self, monkeypatch):
        captured = _capture_submit_payload(monkeypatch)
        payload = captured["payload"]
        assert captured["url"] == "http://comfy.test/prompt"
        # Straight to the address asked, never through an environment proxy,
        # and never on to wherever a redirect points.
        assert captured["method"] == "POST"
        assert captured["proxies"] is comfyui_service.NO_PROXY
        assert captured["redirects"] == 0
        assert payload["prompt"] == {
            k: v for k, v in WORKFLOW.items() if not k.startswith("pixlstash_")
        }
        assert "pixlstash_output_nodes" not in payload["prompt"]

    def test_client_id_is_forwarded(self, monkeypatch):
        captured = _capture_submit_payload(monkeypatch, client_id="tab-1")
        assert captured["payload"]["client_id"] == "tab-1"

    def test_a_nan_is_changed_from_a_prompt_chunk_is_dropped(self, monkeypatch):
        # ComfyUI writes `is_changed: NaN` for a loader whose file it could not
        # hash, and requests refuses to serialise a NaN, so the run never left.
        sent = {}

        def strict_request(method, url, json=None, **_kwargs):
            sent["body"] = stdlib_json.dumps(json, allow_nan=False)
            return _accepted(url)

        monkeypatch.setattr(comfyui_service, "bounded_request", strict_request)
        graph = {
            "81": {
                "class_type": "LoadImage",
                "inputs": {"image": "a.png"},
                "is_changed": float("nan"),
            }
        }
        comfyui_service._submit_comfyui_prompt("http://comfy.test", graph)
        assert stdlib_json.loads(sent["body"])["prompt"] == {
            "81": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}
        }


# --- the runs in flight, and the one socket that follows them (#1862) --------

_REAL_LISTENER = comfyui_service._listen_for_progress
COMFY = "https://comfy.test"


def _progress(prompt_id, value, top=4) -> str:
    return stdlib_json.dumps(
        {
            "type": "progress",
            "data": {"prompt_id": prompt_id, "value": value, "max": top},
        }
    )


class _Upstream:
    """A ComfyUI socket that says *messages*, then calls *then* and is silent."""

    def __init__(self, messages, then):
        self.messages = list(messages)
        self.then = then
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True
        return False

    def recv(self, timeout=None):
        if self.messages:
            return self.messages.pop(0)
        self.then()
        raise TimeoutError


@pytest.fixture
def listening(monkeypatch):
    """The registries empty, the address already marked as listened on (so
    ``follow_run`` starts no thread: each test runs the loop itself), and no
    queue, sleep or socket reaching a real ComfyUI or a real clock.

    Returns the dials: each is ``(address, timeout, socket)``, the socket an
    unconnected one of the test's own, which is what the listener hands to
    the stand-in ``ws_connect``."""
    monkeypatch.setattr(comfyui_service, "_live_runs", {})
    monkeypatch.setattr(comfyui_service, "_listening", {COMFY})
    monkeypatch.setattr(comfyui_service, "_run_outcomes", {})
    monkeypatch.setattr(comfyui_service, "_note_running_prompts", lambda url: None)
    monkeypatch.setattr(comfyui_service.time, "sleep", lambda seconds: None)
    dialled = []

    def dial(address, timeout=None):
        raw = socket.socket()
        # As create_connection leaves it: the connect's timeout still set.
        raw.settimeout(timeout)
        dialled.append((address, timeout, raw))
        return raw

    monkeypatch.setattr(
        comfyui_service, "socket", SimpleNamespace(create_connection=dial)
    )
    return dialled


def test_the_listener_reads_progress_until_no_run_is_in_flight(listening, monkeypatch):
    """One socket per ComfyUI, opened as the client every run is submitted
    under, up only while a run is in flight."""
    opened = []
    seen: dict = {}

    def all_heard():
        seen.update({run["prompt_id"]: run for run in comfyui_service.live_runs("lib")})
        comfyui_service.record_run_outcome("run-a", "completed")
        comfyui_service.record_run_outcome("run-b", "failed", message="boom")

    upstream = _Upstream(
        [
            b"\x00\x00\x00\x01 a preview",
            "not json",
            stdlib_json.dumps({"type": "status", "data": {"status": {}}}),
            stdlib_json.dumps(
                {"type": "executing", "data": {"prompt_id": "run-b", "node": "3"}}
            ),
            _progress("run-b", 3),
            # A step count no float holds, and a stranger's run: neither stops it.
            _progress("run-b", 10**400),
            _progress("stranger", 1),
        ],
        all_heard,
    )

    def fake_connect(url, **kwargs):
        opened.append(url)
        return upstream

    monkeypatch.setattr(comfyui_service, "ws_connect", fake_connect)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")
    comfyui_service.follow_run("run-b", COMFY, "Portrait", "wf-1", "lib")

    _REAL_LISTENER(COMFY)

    assert opened == [f"wss://comfy.test/ws?clientId={comfyui_service.RUN_CLIENT_ID}"]
    assert seen == {
        "run-a": {
            "run_id": None,
            "prompt_id": "run-a",
            "label": "ComfyUI",
            "workflow_id": None,
            "status": "queued",
            "percent": 0.0,
            "generated": False,
        },
        "run-b": {
            "run_id": None,
            "prompt_id": "run-b",
            "label": "Portrait",
            "workflow_id": "wf-1",
            "status": "running",
            "percent": 75.0,
            # Still on node 3: ComfyUI has not finished it.
            "generated": False,
        },
    }
    # Struck off, and only once its socket was closed: the next run starts a
    # listener of its own, and never beside this one.
    assert upstream.closed
    assert comfyui_service._listening == set()
    assert comfyui_service.live_runs("lib") == []


def test_a_message_nested_past_the_parser_keeps_the_socket(
    listening, monkeypatch, caplog
):
    """JSON nested deeper than the parser recurses raises RecursionError, not
    ValueError. The message is unreadable like any other, and the socket goes
    on to the progress after it rather than being dropped and dialled again."""
    opened = []
    seen: dict = {}

    def all_heard():
        seen.update({run["prompt_id"]: run for run in comfyui_service.live_runs("lib")})
        comfyui_service.record_run_outcome("run-a", "completed")

    upstream = _Upstream(
        ["[" * 100_000 + "]" * 100_000, _progress("run-a", 2)], all_heard
    )

    def fake_connect(url, **kwargs):
        opened.append(url)
        return upstream

    monkeypatch.setattr(comfyui_service, "ws_connect", fake_connect)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")

    with caplog.at_level("DEBUG", logger=comfyui_service.logger.name):
        _REAL_LISTENER(COMFY)

    assert len(opened) == 1
    assert (seen["run-a"]["status"], seen["run-a"]["percent"]) == ("running", 50.0)
    assert any(
        r.getMessage().startswith("Unreadable message on ComfyUI's socket")
        for r in caplog.records
    )


def test_a_run_queued_as_the_last_one_ends_keeps_the_same_listener(
    listening, monkeypatch
):
    """ComfyUI keys its clients by id, so two sockets under one would cost
    the second its messages when the first closes. A run queued while the
    listener is closing is followed by that thread, on a new socket."""
    sockets = []

    def first_ends_as_another_is_queued():
        comfyui_service.record_run_outcome("run-a", "completed")

    def second_ends():
        comfyui_service.record_run_outcome("run-b", "completed")

    def fake_connect(url, **kwargs):
        if not sockets:
            sockets.append(_Upstream([], first_ends_as_another_is_queued))
        else:
            # Queued after the first socket closed and before the listener
            # looked again: it is still listed as listening, so no thread.
            assert comfyui_service._listening == {COMFY}
            sockets.append(_Upstream([_progress("run-b", 1)], second_ends))
        return sockets[-1]

    real_still_listening = comfyui_service._still_listening

    def still_listening(base_url):
        if len(sockets) == 1 and sockets[0].closed and not comfyui_service._live_runs:
            # Still listed while its socket was closing, or this run would
            # start a second listener beside it.
            assert COMFY in comfyui_service._listening
            comfyui_service.follow_run("run-b", COMFY, library_uuid="lib")
        return real_still_listening(base_url)

    monkeypatch.setattr(comfyui_service, "ws_connect", fake_connect)
    monkeypatch.setattr(comfyui_service, "_still_listening", still_listening)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")

    _REAL_LISTENER(COMFY)

    assert len(sockets) == 2 and all(socket.closed for socket in sockets)
    assert comfyui_service._listening == set()


def test_the_listener_outlives_a_socket_it_cannot_open(listening, monkeypatch):
    """The socket is only the step progress: whatever goes wrong with it, the
    listener says so, tries again, and is struck off when it does stop."""
    attempts = []

    def fake_connect(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise TypeError("an argument this websockets does not take")
        if len(attempts) == 2:
            raise OSError("connection refused")
        return _Upstream(
            [_progress("run-a", 2)],
            lambda: comfyui_service.record_run_outcome("run-a", "completed"),
        )

    monkeypatch.setattr(comfyui_service, "ws_connect", fake_connect)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")
    _REAL_LISTENER(COMFY)
    assert len(attempts) == 3
    assert comfyui_service._listening == set()

    # And one that dies outright still leaves the address free for the next.
    comfyui_service._listening.add(COMFY)
    comfyui_service.follow_run("run-b", COMFY, library_uuid="lib")
    monkeypatch.setattr(
        comfyui_service,
        "ws_connect",
        lambda url, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        _REAL_LISTENER(COMFY)
    assert comfyui_service._listening == set()


def test_the_listeners_socket_goes_straight_to_comfyui(listening, monkeypatch):
    """The listener dials ComfyUI's own host and port and hands the socket to
    ``websockets``. Left to open its own, ``websockets`` goes through the
    environment's proxy, which would hear every run's progress, and follows a
    redirect to whatever host the answer names."""
    asked = []

    def fake_connect(url, **kwargs):
        sock = kwargs.get("sock")
        asked.append((kwargs, "no socket" if sock is None else sock.gettimeout()))
        return _Upstream(
            [], lambda: comfyui_service.record_run_outcome("run-a", "completed")
        )

    monkeypatch.setattr(comfyui_service, "ws_connect", fake_connect)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")
    _REAL_LISTENER(COMFY)
    assert [(address, timeout) for address, timeout, _raw in listening] == [
        (("comfy.test", 443), 10)
    ]
    raw = listening[0][2]
    assert len(asked) == 1 and asked[0][0].get("sock") is raw
    # Handed over blocking: websockets clears no timeout on a plain socket it
    # is given, and the connect's 10 s would end a socket that ComfyUI,
    # loading a model, leaves silent for minutes.
    assert asked[0][1] is None
    # And closed once the listener is done with it, whatever websockets did.
    assert raw.fileno() == -1


@pytest.mark.parametrize("status", ["302 Found", "307 Temporary Redirect"])
def test_a_redirect_on_the_listeners_socket_is_not_followed(
    monkeypatch, caplog, status
):
    """ComfyUI can answer the socket's opening with a redirect, and
    ``websockets`` followed one to any host and port it named, on a socket of
    its own. Handed the listener's socket, it refuses, and the host named is
    never dialled. The stand-in is real, so this is ``websockets``' own
    behaviour, not a stub's."""
    elsewhere = socket.create_server(("127.0.0.1", 0))
    try:
        target = f"ws://127.0.0.1:{elsewhere.getsockname()[1]}/ws"
        answer = _redirect_to(target).replace(
            b"307 Temporary Redirect", status.encode(), 1
        )
        assert status.encode() in answer
        with _TricklingComfyUI(answer, [], 0.0) as comfyui:
            monkeypatch.setattr(comfyui_service, "_live_runs", {})
            monkeypatch.setattr(comfyui_service, "_listening", {comfyui.url})
            monkeypatch.setattr(comfyui_service, "_run_outcomes", {})
            monkeypatch.setattr(
                comfyui_service, "_note_running_prompts", lambda url: None
            )
            # The run ends in the wait before a second try, so there is one.
            monkeypatch.setattr(
                comfyui_service.time,
                "sleep",
                lambda seconds: comfyui_service.record_run_outcome(
                    "run-a", "completed"
                ),
            )
            comfyui_service.follow_run("run-a", comfyui.url, library_uuid="lib")
            with caplog.at_level(logging.WARNING, logger=comfyui_service.logger.name):
                _REAL_LISTENER(comfyui.url)
        # The stand-in was asked and answered: websockets read its redirect.
        assert f"cannot follow redirect to {target}" in caplog.text, caplog.text
        assert _nobody_connected(elsewhere), "the redirect was followed"
        assert comfyui_service._listening == set()
    finally:
        elsewhere.close()


def test_a_run_already_started_is_read_off_the_queue(monkeypatch):
    """The run that starts the listener was submitted before its socket was
    open, so what ComfyUI said of it then is read from the queue."""
    monkeypatch.setattr(comfyui_service, "_live_runs", {})
    monkeypatch.setattr(comfyui_service, "_listening", {COMFY})
    monkeypatch.setattr(comfyui_service, "_run_outcomes", {})
    monkeypatch.setattr(comfyui_service, "_queue_reads", {})
    monkeypatch.setattr(comfyui_service, "_queue_waits", {})

    queue = {
        "queue_running": [[0, "run-a", {}]],
        "queue_pending": [[1, "run-b", {}]],
    }
    asked = []

    def _bounded_request(
        method, url, *, deadline_s, max_bytes, timeout, proxies=None, redirects=0
    ):
        asked.append((method, url, redirects, proxies))
        return BoundedResponse(
            url=url, status_code=200, content=stdlib_json.dumps(queue).encode()
        )

    monkeypatch.setattr(comfyui_service, "bounded_request", _bounded_request)
    comfyui_service.follow_run("run-a", COMFY, library_uuid="lib")
    comfyui_service.follow_run("run-b", COMFY, library_uuid="lib")
    comfyui_service._note_running_prompts(COMFY)
    assert asked == [("GET", f"{COMFY}/queue", 0, comfyui_service.NO_PROXY)]
    # Nobody waits on that address, so the read is not kept for anyone.
    assert comfyui_service._queue_reads == {}
    assert comfyui_service._queue_waits == {}
    assert {
        run["prompt_id"]: run["status"] for run in comfyui_service.live_runs("lib")
    } == {
        "run-a": "running",
        "run-b": "queued",
    }


def test_a_run_still_going_keeps_its_run_id_past_newer_ended_runs(monkeypatch):
    monkeypatch.setattr(comfyui_service, "MAX_RUN_OUTCOMES", 2)
    for name in ("_run_outcomes", "_run_ids", "_live_runs", "_held_cancels"):
        monkeypatch.setattr(comfyui_service, name, {})
    monkeypatch.setattr(comfyui_service, "_listening", {COMFY})
    comfyui_service.follow_run("p-long", COMFY, run_id="run-long")
    for n in (1, 2):
        comfyui_service.follow_run(f"p-{n}", COMFY, run_id=f"run-{n}")
        comfyui_service.record_run_outcome(f"p-{n}", "completed")
    comfyui_service.follow_run("p-3", COMFY, run_id="run-3")
    assert comfyui_service.run_outcome("run-long")["status"] == "running"
    assert comfyui_service.run_outcome("run-3")["status"] == "running"
    assert len(comfyui_service._run_ids) == 2


def test_following_a_run_never_drops_its_own_run_id(monkeypatch):
    for name in ("_run_outcomes", "_run_ids", "_live_runs", "_held_cancels"):
        monkeypatch.setattr(comfyui_service, name, {})
    monkeypatch.setattr(comfyui_service, "_listening", {COMFY})
    comfyui_service.follow_run("p-a", COMFY, run_id="run-a")
    comfyui_service.follow_run("p-b", COMFY, run_id="run-b")
    monkeypatch.setattr(comfyui_service, "MAX_RUN_OUTCOMES", 1)
    # Followed again, so first in the map, with every run still going.
    comfyui_service.follow_run("p-a2", COMFY, run_id="run-a")
    assert comfyui_service._run_ids == {"run-a": "p-a2"}


def test_an_ending_dropped_by_retention_is_still_announced(monkeypatch):
    """Wrong if the ending raises KeyError because retention evicted it."""
    monkeypatch.setattr(comfyui_service, "MAX_RUN_OUTCOMES", 1)
    for name in ("_run_outcomes", "_run_ids", "_live_runs", "_held_cancels"):
        monkeypatch.setattr(comfyui_service, name, {})
    monkeypatch.setattr(comfyui_service, "_listening", {COMFY})
    heard = []
    monkeypatch.setattr(comfyui_service, "_ending_listener", heard.append)
    comfyui_service.follow_run("p-long", COMFY, run_id="run-long")
    comfyui_service.record_run_outcome("p-x", "failed", run_id="run-x")
    assert [(o["run_id"], o["status"]) for o in heard] == [("run-x", "failed")]


def _fresh_registries(monkeypatch, listening: set) -> list[dict]:
    """The runs in flight empty, and *listening* the addresses listened on;
    the endings announced, in order."""
    for name in ("_run_outcomes", "_run_ids", "_live_runs", "_held_cancels"):
        monkeypatch.setattr(comfyui_service, name, {})
    monkeypatch.setattr(comfyui_service, "_listening", listening)
    heard = []
    monkeypatch.setattr(comfyui_service, "_ending_listener", heard.append)
    return heard


def test_a_listener_that_cannot_start_leaves_the_address_to_the_next_run(
    monkeypatch,
):
    """Raised to the route; and the address is not left marked as listened
    on, or no run there would hear its progress again."""
    _fresh_registries(monkeypatch, set())
    started = []

    class _Thread:
        refuse = True

        def __init__(self, target, args, daemon):
            self.call = (target, args)

        def start(self):
            if _Thread.refuse:
                raise RuntimeError("can't start new thread")
            started.append(self.call)

    monkeypatch.setattr(comfyui_service, "threading", SimpleNamespace(Thread=_Thread))
    with pytest.raises(RuntimeError, match="can't start new thread"):
        comfyui_service.follow_run("p-a", COMFY, library_uuid="lib", run_id="run-a")
    assert comfyui_service._listening == set()

    # The next run on the address starts a listener of its own.
    _Thread.refuse = False
    comfyui_service.follow_run("p-b", COMFY, library_uuid="lib", run_id="run-b")
    assert started == [(comfyui_service._listen_for_progress, (COMFY,))]
    assert comfyui_service._listening == {COMFY}


def test_every_request_of_a_run_goes_straight_to_comfyui(monkeypatch, tmp_path):
    """Never through a proxy set in the environment, which would see the
    pictures and prompts, or answer for a ComfyUI on this machine; and never
    on to wherever a redirect points, which is another host's address."""
    _fresh_registries(monkeypatch, {COMFY})
    monkeypatch.setattr(comfyui_service, "_queue_cleared", 0)
    loader = f"/object_info/{comfyui_service.PIXLSTASH_CHECKPOINT_LOADER}"
    bodies = {
        "/upload/image": {"name": "test-in.png"},
        "/prompt": {"prompt_id": "p-1"},
        "/history/p-1": {},
        "/queue": {"queue_running": [], "queue_pending": []},
        "/extensions": [],
        loader: {},
    }
    sent = []

    def request(method, url, proxies=None, sink=None, **kwargs):
        path = url.removeprefix(COMFY)
        sent.append((method, path, kwargs.get("redirects", 0), proxies))
        if sink is not None:
            sink(b"fake-bytes")
            return BoundedResponse(url, 200, b"")
        return BoundedResponse(url, 200, stdlib_json.dumps(bodies.get(path)).encode())

    monkeypatch.setattr(comfyui_service, "bounded_request", request)
    picture = tmp_path / "test-in.png"
    picture.write_bytes(b"fake-png")

    comfyui_service._upload_image_to_comfyui(COMFY, str(picture))
    comfyui_service._submit_comfyui_prompt(COMFY, WORKFLOW)
    comfyui_service.follow_run("p-1", COMFY, run_id="run-1")
    comfyui_service._read_comfyui_queue(COMFY)
    comfyui_service._fetch_comfyui_history(COMFY, "p-1")
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    comfyui_service._download_comfyui_image(
        COMFY, {"filename": "test-out.png"}, str(fetched)
    )
    comfyui_service._live_runs["p-1"]["status"] = "running"
    comfyui_service.cancel_runs(["run-1"])
    comfyui_service._comfyui_abort(COMFY)
    comfyui_service.comfyui_pixlstash_scripts(COMFY)
    comfyui_service.comfyui_shelf_loader_current(COMFY)

    assert [(method, path) for method, path, *_how in sent] == [
        ("POST", "/upload/image"),
        ("POST", "/prompt"),
        ("GET", "/queue"),
        ("GET", "/history/p-1"),
        ("GET", "/view"),
        ("POST", "/interrupt"),
        ("POST", "/queue"),
        ("POST", "/interrupt"),
        ("POST", "/queue"),
        ("GET", "/extensions"),
        ("GET", loader),
    ]
    assert [
        (method, path)
        for method, path, redirects, proxies in sent
        if redirects != 0 or proxies is not comfyui_service.NO_PROXY
    ] == []
    assert [f.read_bytes() for f in fetched.iterdir()] == [b"fake-bytes"]


PLAIN = "http://comfy.invalid"


def _comfyui_redirects_once(monkeypatch, status: int, body: bytes) -> list[tuple]:
    """``requests`` sends for real, up to the socket: its first answer is a
    *status* redirect to another path, every later one a 200 with *body*.
    The environment names a proxy. Returns each request ``requests`` sends,
    as ``(method, url, the proxies it would go through)``."""
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:9999")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    sent = []

    def send(_adapter, request, **kwargs):
        proxies = {key: url for key, url in kwargs["proxies"].items() if url}
        sent.append((request.method, request.url, proxies))
        response = requests.Response()
        response.request, response.url = request, request.url
        # A body to stream, as the socket would give it: the reader takes
        # nothing ``requests`` has already read.
        response.raw = urllib3.HTTPResponse(
            body=io.BytesIO(body), preload_content=False
        )
        response.status_code = 200
        if len(sent) == 1:
            response.status_code = status
            response.headers["Location"] = f"{PLAIN}/test-elsewhere"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return sent


def _outcome(call):
    """What *call* returned, or what it raised."""
    try:
        return call()
    except Exception as exc:  # How the call ended is what is tested.
        return exc


def test_a_prompt_comfyui_redirects_is_not_sent_on(monkeypatch):
    """``requests`` would send the prompt again to wherever a 307 points,
    and through the environment's proxy. It is sent once, to the address
    asked, and the redirect is a refusal, not an accepted run."""
    sent = _comfyui_redirects_once(monkeypatch, 307, b'{"prompt_id": "test-p"}')
    outcome = _outcome(lambda: comfyui_service._submit_comfyui_prompt(PLAIN, WORKFLOW))
    assert sent == [("POST", f"{PLAIN}/prompt", {})]
    assert isinstance(outcome, comfyui_service.HTTPException), outcome
    assert outcome.detail.startswith("ComfyUI prompt failed: 307")


def test_a_queue_comfyui_redirects_is_unread(monkeypatch):
    """A redirect is not ComfyUI's queue: each run waited on there reads its
    own history instead, as for any queue that cannot be read."""
    queue = b'{"queue_running": [], "queue_pending": []}'
    sent = _comfyui_redirects_once(monkeypatch, 302, queue)
    outcome = _outcome(lambda: comfyui_service._read_comfyui_queue(PLAIN))
    assert sent == [("GET", f"{PLAIN}/queue", {})]
    assert outcome is None


GUARD_S = 10.0
"""How long a test waits on its stand-in server before calling it stuck."""
logger = logging.getLogger(__name__)


class _TricklingComfyUI:
    """A stand-in ComfyUI on a loopback socket of this test's own: it answers
    one request with *head*, then *pieces* one at a time, *gap_s* apart, then
    closes. Nothing here reaches a real ComfyUI."""

    def __init__(self, head: bytes, pieces: list[bytes], gap_s: float):
        self._head, self._pieces, self._gap_s = head, pieces, gap_s
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.settimeout(GUARD_S)
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(GUARD_S)
        self._listener.close()
        assert not self._thread.is_alive(), "the stand-in ComfyUI did not stop"
        return False

    @staticmethod
    def _read_request(conn):
        """The whole request, head and body to its Content-Length: bytes left
        unread when the answer closes the connection reset it, which can lose
        the answer (a multipart upload's body comes after its head)."""
        head = b""
        while b"\r\n\r\n" not in head:
            piece = conn.recv(65536)
            if not piece:
                return
            head += piece
        _rest_of_body(conn, head)

    def _serve(self):
        try:
            conn, _addr = self._listener.accept()
        except OSError as exc:
            logger.warning("the stand-in ComfyUI was never asked: %s", exc)
            return
        with conn:
            conn.settimeout(GUARD_S)
            self._read_request(conn)
            conn.sendall(self._head)
            for piece in self._pieces:
                if self._stop.wait(self._gap_s):
                    return
                try:
                    conn.sendall(piece)
                except OSError as exc:
                    # The reader gave up, which is what the test wants of it.
                    logger.info("the stand-in ComfyUI's reader left: %s", exc)
                    return


def test_a_queue_trickled_past_its_deadline_is_unread(monkeypatch):
    """``requests``' timeout is per receive, so a ComfyUI sending one byte a
    tenth of a second would hold the queue read, and every run waited on
    there, for as long as it liked. The read has a total deadline; past it
    the queue is unread and each run reads its own history instead."""
    monkeypatch.setattr(comfyui_service, "QUEUE_READ_DEADLINE_S", 0.5)
    queue = b'{"queue_running": [], "queue_pending": []}'
    head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [bytes([b]) for b in queue], 0.1) as comfyui:
        started = time.monotonic()
        outcome = comfyui_service._read_comfyui_queue(comfyui.url)
        took = time.monotonic() - started
    # The whole answer would take 4.3 s to arrive.
    assert outcome is None
    assert took < 2.0, took


def test_a_queue_answer_past_the_cap_is_unread(monkeypatch):
    """An answer without end would fill memory; past the cap it is dropped,
    whether it says its length up front or not."""
    monkeypatch.setattr(comfyui_service, "QUEUE_READ_MAX_BYTES", 1024)
    filler = b'{"queue_running": [], "queue_pending": [], "x": "' + b"a" * 4096
    head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [filler, b'"}'], 0.0) as comfyui:
        assert comfyui_service._read_comfyui_queue(comfyui.url) is None


def test_a_declared_length_past_the_cap_is_refused_unread():
    head = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Content-Length: 4096\r\nConnection: close\r\n\r\n"
    )
    with _TricklingComfyUI(head, [b"{" * 4096], 0.0) as comfyui:
        with pytest.raises(bounded_http.ResponseTooLarge, match="declared 4096"):
            bounded_http.bounded_get(
                comfyui.url, deadline_s=5.0, max_bytes=1024, timeout=5.0
            )


def test_a_bounded_read_under_its_limits_is_the_whole_body():
    """The positive control: a slow but finished answer within the deadline
    and the cap is read whole, chunked encoding included."""
    head = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )
    pieces = [b"5\r\n[1, 2\r\n", b"3\r\n, 3\r\n", b"1\r\n]\r\n", b"0\r\n\r\n"]
    with _TricklingComfyUI(head, pieces, 0.05) as comfyui:
        answer = bounded_http.bounded_get(
            comfyui.url, deadline_s=5.0, max_bytes=1024, timeout=5.0
        )
    assert answer.status_code == 200
    assert answer.json() == [1, 2, 3]


_JSON_HEAD = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
)


def test_the_arbiters_comfyui_read_has_a_total_deadline(monkeypatch):
    """The arbiter's probe reads ``/system_stats`` and ``/queue`` for every
    lease and every progress poll, and its callers wait on one shared read.
    A ComfyUI trickling its answer held them all; past the deadline the read
    is a timeout, which the probe already treats as unread."""
    monkeypatch.setattr(comfyui_tenant, "READ_DEADLINE_S", 0.5)
    stats = b'{"system": {}, "devices": []}'
    with _TricklingComfyUI(_JSON_HEAD, [bytes([b]) for b in stats], 0.1) as comfyui:
        started = time.monotonic()
        with pytest.raises(requests.Timeout):
            comfyui_tenant._read_json(f"{comfyui.url}/system_stats")
        took = time.monotonic() - started
    # The whole answer would take 2.9 s to arrive.
    assert took < 2.0, took


def test_the_arbiters_comfyui_read_has_a_size_cap(monkeypatch):
    monkeypatch.setattr(comfyui_tenant, "READ_MAX_BYTES", 1024)
    filler = b'{"queue_running": [], "queue_pending": [], "x": "' + b"a" * 4096
    with _TricklingComfyUI(_JSON_HEAD, [filler, b'"}'], 0.0) as comfyui:
        with pytest.raises(bounded_http.ResponseTooLarge):
            comfyui_tenant._read_json(f"{comfyui.url}/queue")


def test_the_arbiters_comfyui_read_takes_a_finished_answer():
    """The positive control for the two above."""
    stats = b'{"system": {}, "devices": []}'
    with _TricklingComfyUI(_JSON_HEAD, [stats], 0.0) as comfyui:
        assert comfyui_tenant._read_json(f"{comfyui.url}/system_stats") == {
            "system": {},
            "devices": [],
        }


_COMFYUI_WORDS = b"test-comfyui-own-words <b>Traceback</b>"


def _refusal_head(status: bytes, length: int) -> bytes:
    return (
        b"HTTP/1.1 " + status + b"\r\nContent-Type: text/html\r\n"
        b"Content-Length: " + str(length).encode() + b"\r\nConnection: close\r\n\r\n"
    )


@pytest.mark.parametrize(
    "ask, sentence",
    [
        (
            lambda url, tmp: comfyui_service._fetch_comfyui_history(url, "p-1"),
            "ComfyUI history failed: 500. ComfyUI did not say how the run ended; "
            "its console says why.",
        ),
        (
            lambda url, tmp: comfyui_service._submit_comfyui_prompt(url, WORKFLOW),
            "ComfyUI prompt failed: 500. ComfyUI refused the workflow; its "
            "console says why.",
        ),
        (
            lambda url, tmp: comfyui_service._download_comfyui_image(
                url, {"filename": "test-out.png"}, str(tmp)
            ),
            "ComfyUI image fetch failed: 500. ComfyUI did not hand over the "
            "file the run made.",
        ),
        (
            lambda url, tmp: comfyui_service._upload_image_to_comfyui(
                url, str(tmp / "test-in.png")
            ),
            "ComfyUI upload failed: 500. ComfyUI refused the picture; its "
            "console says why.",
        ),
    ],
    ids=["history", "prompt", "view", "upload"],
)
def test_a_comfyui_refusal_is_a_fixed_sentence_its_body_only_logged(
    ask, sentence, tmp_path, caplog
):
    """A 502 detail reaches the owner's screen. ComfyUI's own body (an HTML
    page, a traceback, anything a host at that address sends) is never in it;
    the log has a bounded excerpt beside the upstream status."""
    (tmp_path / "test-in.png").write_bytes(b"fake-png")
    body = _COMFYUI_WORDS * 4000  # Well past the logged excerpt.
    head = _refusal_head(b"500 Internal Server Error", len(body))
    with caplog.at_level(logging.WARNING, logger=comfyui_service.logger.name):
        with _TricklingComfyUI(head, [body], 0.0) as comfyui:
            with pytest.raises(comfyui_service.HTTPException) as refused:
                ask(comfyui.url, tmp_path)
    assert refused.value.status_code == 502
    assert refused.value.detail == sentence
    logged = [r.getMessage() for r in caplog.records if "status=500" in r.getMessage()]
    assert len(logged) == 1, caplog.text
    assert "test-comfyui-own-words" in logged[0]
    assert logged[0].endswith("(cut short)")
    assert len(logged[0]) < 1000


def test_a_prompt_rejection_is_told_in_its_own_composed_sentence_cut_short():
    """ComfyUI's validation answer is the one account of why a graph will not
    run, so the sentence composed from its fields is kept, to a bound."""
    rejection = stdlib_json.dumps(
        {
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {
                "3": {
                    "class_type": "KSampler",
                    "errors": [{"message": "bad cfg", "details": "x" * 5000}],
                }
            },
        }
    ).encode()
    head = _refusal_head(b"400 Bad Request", len(rejection))
    with _TricklingComfyUI(head, [rejection], 0.0) as comfyui:
        with pytest.raises(comfyui_service.HTTPException) as refused:
            comfyui_service._submit_comfyui_prompt(comfyui.url, WORKFLOW)
    detail = refused.value.detail
    assert detail.startswith("ComfyUI prompt failed: 400 Prompt outputs failed")
    assert "KSampler" in detail
    prefix = len("ComfyUI prompt failed: 400 ")
    assert len(detail) == prefix + comfyui_service.MAX_PROMPT_REJECTION
    assert detail.endswith("\u2026")


def test_an_output_past_the_file_cap_is_a_failed_fetch(monkeypatch, tmp_path):
    """``/view`` is written to disk as it arrives; one without end used to
    fill the disk. Past the cap the fetch fails, with at most the cap written."""
    monkeypatch.setattr(comfyui_service, "COMFYUI_FILE_MAX_BYTES", 1024)
    head = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [b"\x89" * 8192], 0.0) as comfyui:
        with pytest.raises(comfyui_service.HTTPException) as failed:
            comfyui_service._download_comfyui_image(
                comfyui.url, {"filename": "test-out.png"}, str(tmp_path)
            )
    assert failed.value.detail == "ComfyUI image fetch failed"
    assert [f.stat().st_size <= 1024 for f in tmp_path.iterdir()] == [True]


def test_an_output_trickled_past_its_deadline_is_a_failed_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(comfyui_service, "COMFYUI_FILE_DEADLINE_S", 0.5)
    head = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [b"\x89"] * 40, 0.1) as comfyui:
        started = time.monotonic()
        with pytest.raises(comfyui_service.HTTPException) as failed:
            comfyui_service._download_comfyui_image(
                comfyui.url, {"filename": "test-out.png"}, str(tmp_path)
            )
        took = time.monotonic() - started
    assert failed.value.detail == "ComfyUI image fetch failed"
    # The whole file would take 4 s to arrive.
    assert took < 2.0, took


def test_an_output_within_its_bounds_is_written_whole(tmp_path):
    """The positive control for the two above."""
    head = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [b"\x89PNG", b"-test-body"], 0.0) as comfyui:
        path, ext = comfyui_service._download_comfyui_image(
            comfyui.url, {"filename": "test-out.png"}, str(tmp_path)
        )
    assert ext == ".png"
    with open(path, "rb") as handle:
        assert handle.read() == b"\x89PNG-test-body"


def _nobody_connected(listener: socket.socket) -> bool:
    """Whether no client has connected to *listener*: the kernel completes a
    connect into the backlog without ``accept``, so a non-blocking accept
    finds any that did."""
    listener.setblocking(False)
    try:
        conn, _addr = listener.accept()
    except BlockingIOError:
        return True
    conn.close()
    return False


def _redirect_to(target: str) -> bytes:
    return (
        f"HTTP/1.1 307 Temporary Redirect\r\nLocation: {target}\r\n"
        "Content-Length: 0\r\nConnection: close\r\n\r\n"
    ).encode()


def test_the_arbiters_comfyui_read_does_not_follow_a_redirect():
    """A redirect names an address nobody checked: it is no answer, and the
    address it names is never asked."""
    elsewhere = socket.create_server(("127.0.0.1", 0))
    try:
        target = f"http://127.0.0.1:{elsewhere.getsockname()[1]}/system_stats"
        with _TricklingComfyUI(_redirect_to(target), [], 0.0) as comfyui:
            with pytest.raises(requests.HTTPError, match="answered 307"):
                comfyui_tenant._read_json(f"{comfyui.url}/system_stats")
        assert _nobody_connected(elsewhere), "the redirect was followed"
    finally:
        elsewhere.close()


def test_asking_comfyui_to_free_memory_does_not_follow_a_redirect(monkeypatch):
    """``POST /free`` answered with a redirect was followed, resent to the
    address in its ``Location``, and its 200 there read as ComfyUI having
    freed memory. Now it is refused, and the other address is never asked."""
    elsewhere = socket.create_server(("127.0.0.1", 0))
    try:
        target = f"http://127.0.0.1:{elsewhere.getsockname()[1]}/free"
        with _TricklingComfyUI(_redirect_to(target), [], 0.0) as comfyui:
            probe = ComfyUIProbe("test-client")
            probe.watch(comfyui.url)
            monkeypatch.setattr(probe, "idle", lambda max_age_s=None: True)
            monkeypatch.setattr(probe, "usage_mb", lambda: 4000)
            assert probe.free() is None
        assert _nobody_connected(elsewhere), "the redirect was followed"
    finally:
        elsewhere.close()


def test_asking_comfyui_to_free_memory_is_answered_by_a_200(monkeypatch):
    """The positive control: ComfyUI's own 200 is a yes."""
    head = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    with _TricklingComfyUI(head, [], 0.0) as comfyui:
        probe = ComfyUIProbe("test-client")
        probe.watch(comfyui.url)
        monkeypatch.setattr(probe, "idle", lambda max_age_s=None: True)
        monkeypatch.setattr(probe, "usage_mb", lambda: 4000)
        monkeypatch.setattr(comfyui_tenant.time, "sleep", lambda _s: None)
        # The settle loop's fresh read finds nothing (conftest's no-ComfyUI
        # read), so the answer is what was freed by then: nothing measured.
        assert probe.free() == 0


@pytest.mark.parametrize(
    "spelling",
    [
        "http://comfy.invalid:8188/",
        "  http://comfy.invalid:8188  ",
        "http://comfy.invalid:8188",
    ],
)
def test_one_comfyui_spelled_two_ways_is_one_address(spelling):
    """Callers hand the probe the saved address with or without its slash.
    Read as two addresses, every switch between them dropped what was read
    of ComfyUI, its process and its floor included."""
    probe = ComfyUIProbe("test-client")
    probe.watch("http://comfy.invalid:8188")
    generation = probe._generation
    probe.watch(spelling)
    assert probe.base_url == "http://comfy.invalid:8188"
    assert probe._generation == generation


@pytest.mark.parametrize(
    "address",
    [
        "http://comfy.invalid\\@127.0.0.1:8188",
        "http://someone@127.0.0.1:8188",
        "http://127.0.0.1:8188/?next=elsewhere",
        "file:///etc/comfyui",
        "comfy.invalid:8188",
    ],
)
def test_an_ambiguous_comfyui_address_is_not_read(address):
    """An address URL parsers disagree on, or not http(s) at all, points the
    probe at nothing: no read is sent there, and no local process is matched
    to it (which ``quit`` would end)."""
    probe = ComfyUIProbe("test-client")
    probe.watch("http://127.0.0.1:8188")
    probe.watch(address)
    assert probe.base_url is None
    assert probe.stats(max_age_s=0.0) is None
    assert not probe.can_quit()


class _CardProcesses:
    """A ledger reading one card's processes, and nothing else."""

    def __init__(self, processes):
        self._processes = processes

    def processes(self, uuid=None):
        return dict(self._processes)


def test_a_search_for_comfyuis_process_does_not_hold_the_probe(monkeypatch):
    """The search reads every process's sockets on ComfyUI's card, which takes
    as long as there are processes. Under the probe's lock it held every read
    of ComfyUI, and a switch to another address, behind it.

    Wrong if the lock is held while the sockets are read, or if two searches
    run at once (the second waits and reuses the first's answer).
    """
    probe = ComfyUIProbe("test-client")
    probe.watch("http://comfy.invalid:8188")
    probe._mapping = (time.monotonic(), "GPU-test-card")
    monkeypatch.setattr(
        comfyui_tenant, "ledger", lambda: _CardProcesses({4242: 900, 5151: 300})
    )
    held_during_search = []
    searches = []
    searching = threading.Event()
    release = threading.Event()

    def listening(pids):
        searches.append(sorted(pids))
        held_during_search.append(probe._lock.locked())
        searching.set()
        assert release.wait(timeout=10)
        return 4242

    probe._listening_pid = listening
    first = threading.Thread(target=probe._process_mb, name="test-search-1")
    second = threading.Thread(target=probe._process_mb, name="test-search-2")
    first.start()
    try:
        assert searching.wait(timeout=10)
        second.start()
        # A switch-sized step on the lock completes while the search runs.
        assert probe._lock.acquire(timeout=2), "the search held the probe's lock"
        probe._lock.release()
    finally:
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert held_during_search == [False]
    assert searches == [[4242, 5151]], "a second search ran beside the first"
    assert probe._pid == 4242
    assert probe._process_mb() == 900


def test_a_search_answered_after_a_switch_is_not_kept(monkeypatch):
    """Off the lock, a search can finish after the probe moved to another
    ComfyUI: its pid is the old one's and is dropped, not matched to the new
    address (``quit`` would end it)."""
    probe = ComfyUIProbe("test-client")
    probe.watch("http://comfy-a.invalid:7801")
    probe._mapping = (time.monotonic(), "GPU-test-card")
    monkeypatch.setattr(comfyui_tenant, "ledger", lambda: _CardProcesses({4242: 900}))
    switches = []

    def listening(pids):
        # On a thread, so a search still holding the lock fails the test
        # rather than deadlocking it: the switch then lands after the search.
        switch = threading.Thread(
            target=probe.watch, args=("http://comfy-b.invalid:7802",)
        )
        switch.start()
        switches.append(switch)
        switch.join(timeout=2)
        return 4242

    probe._listening_pid = listening
    answered = probe._process_mb()
    for switch in switches:
        switch.join(timeout=10)
        assert not switch.is_alive()
    assert answered is None, "the old ComfyUI's process was read for the new one"
    assert probe._pid is None and not probe.can_quit()


class _PidReadSwitches(ComfyUIProbe):
    """A probe whose first read of its pid by ``quit`` asks for a switch to
    *switch_to* on another thread, and gives it a moment to land."""

    def __init__(self, client_id, switch_to):
        self._quit_pid_reads = 0
        self._switch_to = switch_to
        self.switch: threading.Thread | None = None
        super().__init__(client_id)

    @property
    def _pid(self):
        if threading.current_thread().name == "test-quit" and self._quit_pid_reads == 0:
            self._quit_pid_reads += 1
            self.switch = threading.Thread(
                target=self.watch, args=(self._switch_to,), name="test-switch"
            )
            self.switch.start()
            # Lands at once unless the read holds the lock.
            self.switch.join(timeout=0.5)
        return self.__dict__.get("_pid_value")

    @_pid.setter
    def _pid(self, value):
        self.__dict__["_pid_value"] = value


def test_quit_reads_its_pid_and_its_generation_as_one_pair(monkeypatch):
    """``quit`` takes the pid and the generation it checks against together.
    Read apart, a switch between them paired the old ComfyUI's pid with the
    new one's generation, so the check passed and the old ComfyUI was ended
    on the new one's empty queue.

    Wrong if ``ended`` holds the pid: the old process was ended mid-job.
    """
    old, new = "http://comfy-a.invalid:7801", "http://comfy-b.invalid:7802"
    ended = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            ended.append(self.pid)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(comfyui_tenant.psutil, "Process", FakeProcess)
    probe = _PidReadSwitches("test-client", new)
    probe.watch(old)
    probe._pid = 4242
    probe._listening_pid = lambda pids: next(iter(pids))

    def read(path):
        # The queue read starts once the switch has landed, whichever pid
        # and generation the quit read.
        probe.switch.join(timeout=10)
        assert not probe.switch.is_alive()
        return {"queue_running": [], "queue_pending": []}

    probe._get = read
    quitting = []
    asking = threading.Thread(
        target=lambda: quitting.append(probe.quit()), name="test-quit"
    )
    asking.start()
    asking.join(timeout=10)
    assert not asking.is_alive()
    assert quitting == [False] and ended == [], (
        "the old ComfyUI was ended on the new one's queue"
    )


def test_quit_ends_the_process_of_the_comfyui_it_checked(monkeypatch):
    """The generation check and the terminate are one step: a switch asked
    between them lands after the signal, so the process ended is the one
    whose port and queue were checked, never one checked for another
    address.

    Wrong if the generation seen at terminate is not the one checked.
    """
    old, new = "http://comfy-a.invalid:7801", "http://comfy-b.invalid:7802"
    at_terminate = []
    switches = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            switch = threading.Thread(target=probe.watch, args=(new,))
            switch.start()
            switches.append(switch)
            # Lands at once unless the check still holds the lock.
            switch.join(timeout=0.5)

        def terminate(self):
            at_terminate.append((self.pid, probe._generation))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(comfyui_tenant.psutil, "Process", FakeProcess)
    probe = ComfyUIProbe("test-client")
    probe.watch(old)
    probe._pid = 4242
    probe._listening_pid = lambda pids: next(iter(pids))
    probe._get = lambda path: {"queue_running": [], "queue_pending": []}
    checked = probe._generation
    assert probe.quit()
    for switch in switches:
        switch.join(timeout=10)
        assert not switch.is_alive()
    assert at_terminate == [(4242, checked)]
    assert probe.base_url == new
