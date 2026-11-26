"""No thread may ever be handed a partially initialised ML package.

CPython gives every module its own import lock. When two threads import
modules that import each other, the deadlock detector does not wait: it hands
one thread the other's half-built module (``_lock_unlock_module`` in
``importlib/_bootstrap.py`` swallows ``_DeadlockError``). ``accelerate`` has
such a cycle and both the ``sentence_transformers`` stack (the search route's
first query embedding) and the ``open_clip`` stack (the embedding preload's
first resize) import it, so the first search after start-up failed with
``cannot import name 'AcceleratorState' from partially initialized module
'accelerate.state'``.

The race needs two threads to interleave inside one package, which the real
packages do rarely and not on demand. The reproduction below uses throwaway
stand-ins with the same shape - a package whose ``utils`` and ``state``
import each other, reached by one thread from its top and by the other from
``state`` - and drives the interleaving with a handshake on the import
machinery's own lock state, through the two app entry points that collided.
It runs in a subprocess so the stand-ins never meet the real packages.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from functools import cache
from pathlib import Path

import pytest

from pixlstash.image_plugins.registry import ImagePluginManager
from pixlstash.services.config_service import collect_vram_from_torch
from pixlstash.tagger_plugins.clip_service import ClipService
from pixlstash.tagger_plugins.registry import TaggerPluginManager
from pixlstash.utils import accelerator, model_utils
from pixlstash.utils.model_utils import MODEL_CONSTRUCTION_LOCK, ml_imports

REPO_ROOT = Path(__file__).resolve().parent.parent


def checkout_env() -> dict[str, str]:
    """The environment for a subprocess that must import this checkout.

    A driver script's own directory, not the working directory, heads its
    ``sys.path``, so without this ``pixlstash`` resolves through the editable
    install's finder, which sits after ``PathFinder`` on ``sys.meta_path`` and
    points at whichever tree was installed. A copy of the suite run against
    another tree would then test the installed one. ``PYTHONPATH`` is searched
    by ``PathFinder`` and so wins.
    """
    paths = (str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}


def assert_imported_this_checkout(pixlstash_file: str) -> None:
    """Fail when a subprocess's ``pixlstash`` came from another tree."""
    assert Path(pixlstash_file).resolve().is_relative_to(REPO_ROOT / "pixlstash"), (
        f"the subprocess imported {pixlstash_file}, not this checkout ({REPO_ROOT})"
    )


PARTIAL_MODULE_ERROR = (
    "ImportError: cannot import name 'AcceleratorState' from partially "
    "initialized module 'fakeaccel.state' (most likely due to a circular import)"
)

_FAKE_PACKAGES = {
    # The cyclic package: accelerate's utils <-> state, in miniature. Safe
    # when one thread imports it; only a concurrent import can break it.
    "fakeaccel/__init__.py": "from . import utils  # noqa: F401\n",
    "fakeaccel/utils.py": """
        import fakectl

        x = 1
        fakectl.utils_body()
        from .state import AcceleratorState  # noqa: E402,F401
    """,
    "fakeaccel/state.py": """
        import fakectl

        fakectl.state_body()
        from .utils import x  # noqa: E402,F401


        class AcceleratorState:
            pass
    """,
    # What the search route's first embedding imports: the package's top.
    "sentence_transformers/__init__.py": """
        import fakeaccel  # noqa: F401


        class SentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass
    """,
    "transformers/__init__.py": "",
    "transformers/logging.py": """
        def get_verbosity():
            return 30


        def set_verbosity_error():
            pass


        def set_verbosity(level):
            pass
    """,
    # What the embedding preload's first resize imports: straight into state.
    "open_clip/__init__.py": """
        import fakeaccel.state  # noqa: F401

        from . import pretrained, transform  # noqa: F401


        def get_model_config(name):
            return {"vision_cfg": {"image_size": 224}}
    """,
    "open_clip/pretrained.py": """
        def get_pretrained_cfg(name, tag):
            return {}
    """,
    "open_clip/transform.py": """
        from dataclasses import dataclass


        @dataclass
        class PreprocessCfg:
            size: int = 224


        def merge_preprocess_dict(base, overlay):
            return {**base, **overlay}


        class _Compose:
            def __init__(self, transforms):
                self.transforms = transforms


        def image_transform_v2(cfg, is_train):
            return _Compose([lambda img: img.copy()])
    """,
}

_DRIVER = """
    import contextlib
    import json
    import sys
    import threading
    import time
    import types

    # importlib replaces its _bootstrap with the frozen module the import
    # system actually runs, so this is the live lock table.
    import importlib._bootstrap as bootstrap

    pkgroot, mode = sys.argv[1], sys.argv[2]
    sys.path.insert(0, pkgroot)
    DEADLINE_S = 30.0

    ctl = types.ModuleType("fakectl")
    ctl.in_utils = threading.Event()
    ctl.in_state = threading.Event()
    ctl.preload_ident = None
    ctl.handshake_done = False


    def _preload_waits_on_the_ml_lock():
        frame = sys._current_frames().get(ctl.preload_ident)
        return frame is not None and frame.f_code.co_name == "__enter__" and (
            frame.f_code.co_filename.endswith("model_utils.py")
        )


    def utils_body():
        # Thread 1, inside fakeaccel.utils and holding its module lock.
        if ctl.handshake_done:
            return
        ctl.handshake_done = True
        ctl.in_utils.set()
        deadline = time.monotonic() + DEADLINE_S
        while time.monotonic() < deadline:
            if ctl.in_state.is_set():
                break
            if _preload_waits_on_the_ml_lock():
                return  # serialised: the preload cannot get in, carry on
            time.sleep(0.001)
        else:
            raise AssertionError("the preload thread never reached the import")
        # Thread 2 is inside fakeaccel.state; wait until it is queued on
        # fakeaccel.utils, so thread 1's next import closes the cycle.
        while time.monotonic() < deadline:
            ref = bootstrap._module_locks.get("fakeaccel.utils")
            lock = ref() if ref is not None else None
            if lock is not None and lock.waiters:
                return
            time.sleep(0.001)
        raise AssertionError("the preload thread never waited on fakeaccel.utils")


    ctl.utils_body = utils_body
    ctl.state_body = ctl.in_state.set
    sys.modules["fakectl"] = ctl

    from PIL import Image

    from pixlstash.tagger_plugins import clip_service
    from pixlstash.utils import model_utils

    if mode == "unguarded":
        model_utils.ml_imports = contextlib.nullcontext
        clip_service.ml_imports = contextlib.nullcontext

    results = {"pixlstash": model_utils.__file__}


    def _run(name, fn):
        try:
            fn()
            results[name] = "ok"
        except BaseException as exc:
            results[name] = f"{type(exc).__name__}: {exc}"


    def search():
        model_utils.load_sentence_transformer("example-model")


    def preload():
        clip_service.ClipService("cpu").reduce_for_preprocess(
            [Image.new("RGB", (8, 8))]
        )


    t1 = threading.Thread(target=_run, args=("search", search))
    t1.start()
    if not ctl.in_utils.wait(DEADLINE_S):
        raise SystemExit("the search thread never reached fakeaccel.utils")
    t2 = threading.Thread(target=_run, args=("preload", preload))
    t2.start()
    ctl.preload_ident = t2.ident
    t1.join(DEADLINE_S)
    t2.join(DEADLINE_S)
    results["hung"] = t1.is_alive() or t2.is_alive()
    print(json.dumps(results))
"""


def _race(tmp_path: Path, mode: str) -> dict:
    """Run the two first imports concurrently in a fresh interpreter."""
    pkgroot = tmp_path / "pkgs"
    for rel, body in _FAKE_PACKAGES.items():
        path = pkgroot / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(_DRIVER))
    proc = subprocess.run(
        [sys.executable, str(driver), str(pkgroot), mode],
        cwd=REPO_ROOT,
        env=checkout_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_imported_this_checkout(results.pop("pixlstash"))
    return results


def test_unguarded_first_imports_reproduce_the_partial_module(tmp_path):
    """The harness really produces the production failure without the lock.

    Without this the passing test below could pass because nothing raced.
    """
    results = _race(tmp_path, "unguarded")

    assert results["hung"] is False
    assert results["search"].startswith(PARTIAL_MODULE_ERROR), results


def test_first_imports_from_two_threads_never_see_a_partial_module(tmp_path):
    results = _race(tmp_path, "guarded")

    assert results == {"search": "ok", "preload": "ok", "hung": False}


@pytest.fixture
def fresh_sites(monkeypatch):
    """Forget which import blocks have completed, for this test only."""
    monkeypatch.setattr(model_utils, "_COMPLETED_IMPORT_SITES", set())


def _held_elsewhere():
    """Hold MODEL_CONSTRUCTION_LOCK on another thread until released."""
    taken, release = threading.Event(), threading.Event()

    def _hold():
        with MODEL_CONSTRUCTION_LOCK:
            taken.set()
            release.wait(10)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert taken.wait(5)
    return release, holder


def _import_json():
    with ml_imports():
        import json  # noqa: F401


def _finishes(fn, timeout: float) -> bool:
    worker = threading.Thread(target=fn, daemon=True)
    worker.start()
    worker.join(timeout)
    return not worker.is_alive()


def test_a_first_import_waits_for_the_lock(fresh_sites):
    release, holder = _held_elsewhere()
    try:
        assert not _finishes(_import_json, 0.3)
    finally:
        release.set()
        holder.join(5)


def test_a_completed_import_block_no_longer_takes_the_lock(fresh_sites):
    _import_json()
    release, holder = _held_elsewhere()
    try:
        # The hot paths re-run ``import torch`` per batch; once the block has
        # completed they must not queue behind a model construction.
        assert _finishes(_import_json, 5)
    finally:
        release.set()
        holder.join(5)


def _failing_import():
    with ml_imports():
        import pixlstash_no_such_package  # noqa: F401


def test_a_failed_import_releases_the_lock_and_is_tried_again_locked(fresh_sites):
    with pytest.raises(ModuleNotFoundError):
        _failing_import()

    assert not MODEL_CONSTRUCTION_LOCK._is_owned()
    release, holder = _held_elsewhere()  # free for another thread to take
    release.set()
    holder.join(5)
    # Not marked complete: the next run imports again, under the lock, and
    # raises the same error rather than finding a cached module.
    errors: list[BaseException] = []

    def _record_failure():
        try:
            _failing_import()
        except ModuleNotFoundError as exc:
            errors.append(exc)

    release, holder = _held_elsewhere()
    worker = threading.Thread(target=_record_failure, daemon=True)
    try:
        worker.start()
        worker.join(0.3)
        assert worker.is_alive(), "the second run did not wait for the lock"
    finally:
        release.set()
        holder.join(5)
    worker.join(5)
    assert [type(e) for e in errors] == [ModuleNotFoundError]


def test_a_module_body_block_waits_like_a_function_body(tmp_path, monkeypatch):
    """A plugin or test may build a service at module scope.

    Its import block then runs at import time, and must wait for the lock and
    go on to finish the import, not refuse.
    """
    (tmp_path / "mli_module_body_probe.py").write_text(
        "from pixlstash.utils.model_utils import ml_imports\n"
        "with ml_imports():\n"
        "    import json  # noqa: F401\n"
        "LOADED = True\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mli_module_body_probe", raising=False)
    errors: list[BaseException] = []

    def _import_probe():
        try:
            __import__("mli_module_body_probe")
        except Exception as exc:  # recorded and asserted on below
            errors.append(exc)

    release, holder = _held_elsewhere()
    worker = threading.Thread(target=_import_probe, daemon=True)
    try:
        worker.start()
        worker.join(0.3)
        assert worker.is_alive(), "the module body did not wait for the lock"
    finally:
        release.set()
        holder.join(5)
    worker.join(5)
    assert not worker.is_alive()
    assert errors == []
    assert sys.modules["mli_module_body_probe"].LOADED is True
    assert not MODEL_CONSTRUCTION_LOCK._is_owned()


# --------------------------------------------------------------------------
# A route or a poll never waits on a model load.
# --------------------------------------------------------------------------
#
# The lock is held across every model construction, downloads included, so a
# block that a request runs and that takes the lock waits for the whole of a
# first-launch download: the review's probe had the progress poll stuck for
# as long as the lock was held. A block that names its modules skips the lock
# when they are loaded whole, and still waits while one is being imported.


def _parked_on_the_lock(worker: threading.Thread) -> bool:
    """Wait until *worker* sits in ``_MLImportBlock.__enter__``, waiting.

    The sync point for the negative cases: the worker is seen at the lock
    rather than merely late. ``RLock.acquire`` is C, so the innermost Python
    frame of a thread that waits on it is ``__enter__``.
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(worker.ident)
        if (
            frame is not None
            and frame.f_code.co_name == "__enter__"
            and frame.f_code.co_filename == model_utils.__file__
        ):
            return True
        if not worker.is_alive():
            return False
        time.sleep(0.005)
    return False


def _half_built(name: str) -> types.ModuleType:
    """A stand-in for a module another thread is still importing."""
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(name=name, _initializing=True)
    return module


def _progress_poll_vram():
    collect_vram_from_torch({})


def _text_search_query():
    # Nothing to encode, so it returns straight after the block: no model.
    ClipService("cpu").encode_text("")


def _likeness_search_picture():
    ClipService("cpu").encode_image_batch([])


# The named torch rungs: GET /workers/progress where NVML has no figure
# (macOS, Windows WDDM, non-NVIDIA Linux), device resolution, and the CLIP
# encodes a text search and a likeness search run on the request thread
# (with CLIP loaded, nothing after the block takes the lock).
_TORCH_RUNGS = pytest.mark.parametrize(
    "rung",
    [
        _progress_poll_vram,
        accelerator._torch,
        _text_search_query,
        _likeness_search_picture,
    ],
    ids=[
        "progress-poll-vram",
        "device-resolution",
        "text-search-clip",
        "likeness-search-clip",
    ],
)


@_TORCH_RUNGS
def test_a_request_path_torch_import_never_waits_on_a_model_load(
    rung, fresh_sites, monkeypatch
):
    """The review's probe, gated: torch is in, a model load holds the lock."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    release, holder = _held_elsewhere()  # a download inside from_pretrained
    try:
        assert _finishes(rung, 5), "the request waited on the model load"
    finally:
        release.set()
        holder.join(5)


@_TORCH_RUNGS
def test_a_request_path_torch_import_still_waits_for_torch_itself(
    rung, fresh_sites, monkeypatch
):
    """The race protection stays: a torch mid-import is not whole."""
    monkeypatch.setitem(sys.modules, "torch", _half_built("torch"))
    release, holder = _held_elsewhere()
    worker = threading.Thread(target=rung, daemon=True)
    try:
        worker.start()
        assert _parked_on_the_lock(worker), "it did not wait for the lock"
    finally:
        release.set()
        holder.join(5)
    worker.join(5)
    assert not worker.is_alive(), "it did not finish once the lock was free"


_NAMED_PKG_MODULES = ("mli_named_pkg", "mli_named_pkg.sub")


def _named_pkg_block():
    with ml_imports("mli_named_pkg.sub"):
        import mli_named_pkg.sub  # noqa: F401


@pytest.mark.parametrize(
    "state", ["missing", "half-built", "parent-half-built", "whole"]
)
def test_a_named_block_skips_the_lock_only_over_whole_modules(
    state, fresh_sites, tmp_path, monkeypatch
):
    (tmp_path / "mli_named_pkg").mkdir()
    (tmp_path / "mli_named_pkg" / "__init__.py").write_text("")
    (tmp_path / "mli_named_pkg" / "sub.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    planted = {
        "missing": {},
        "half-built": {
            "mli_named_pkg": types.ModuleType("mli_named_pkg"),
            "mli_named_pkg.sub": _half_built("mli_named_pkg.sub"),
        },
        # ``import a.b`` hands back ``a``, so a half-built parent counts.
        "parent-half-built": {
            "mli_named_pkg": _half_built("mli_named_pkg"),
            "mli_named_pkg.sub": types.ModuleType("mli_named_pkg.sub"),
        },
        "whole": {
            "mli_named_pkg": types.ModuleType("mli_named_pkg"),
            "mli_named_pkg.sub": types.ModuleType("mli_named_pkg.sub"),
        },
    }[state]
    for name in _NAMED_PKG_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, module in planted.items():
        monkeypatch.setitem(sys.modules, name, module)
    errors: list[BaseException] = []

    def _run():
        try:
            _named_pkg_block()
        except Exception as exc:  # recorded and asserted on below
            errors.append(exc)

    release, holder = _held_elsewhere()
    worker = threading.Thread(target=_run, daemon=True)
    try:
        worker.start()
        if state == "whole":
            worker.join(5)
            assert not worker.is_alive(), "a whole module waited for the lock"
        else:
            assert _parked_on_the_lock(worker), "it skipped the lock"
    finally:
        release.set()
        holder.join(5)
        worker.join(5)
        # The "missing" case really imports them; the next case plants its own.
        for name in _NAMED_PKG_MODULES:
            sys.modules.pop(name, None)
    assert not worker.is_alive()
    assert errors == []
    assert not MODEL_CONSTRUCTION_LOCK._is_owned()


# --------------------------------------------------------------------------
# Plugin loaders run a module body under the lock.
# --------------------------------------------------------------------------

# A plugin body that does what a real plugin may: build a model (re-entering
# the lock on its own thread), and hand a plain import to a thread it waits
# for. Neither may hang the loader. The sink records what the body saw.
_PLUGIN_BODY = """
import threading

import mli_plugin_sink as sink
from pixlstash.utils.model_utils import MODEL_CONSTRUCTION_LOCK, ml_imports

with ml_imports():
    import json  # noqa: F401
with MODEL_CONSTRUCTION_LOCK:
    sink.reentered = True
sink.held = MODEL_CONSTRUCTION_LOCK._is_owned()


def _plain_import():
    import mli_plugin_spawned  # noqa: F401

    sink.spawned = True


worker = threading.Thread(target=_plain_import)
worker.start()
worker.join(5)
sink.joined = not worker.is_alive()
"""

_PLUGIN_MODULES = (
    "mli_plugin_spawned",
    "mli_plugin_first_party",
    "pixlstash_user_tagger_mli_plugin_py",
)


def _tagger_user_dir(tmp_path: Path) -> None:
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "mli_plugin.py").write_text(_PLUGIN_BODY)
    TaggerPluginManager(user_dir=str(tmp_path / "plugins"), first_party=[]).reload()


def _tagger_first_party(tmp_path: Path) -> None:
    (tmp_path / "mli_plugin_first_party.py").write_text(_PLUGIN_BODY)
    TaggerPluginManager(
        user_dir=None, first_party=[("mli_plugin_first_party", "Absent")]
    ).reload()


def _image_user_dir(tmp_path: Path) -> None:
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "mli_plugin.py").write_text(_PLUGIN_BODY)
    ImagePluginManager(
        built_in_dir=str(tmp_path / "no-built-ins"),
        user_dir=str(tmp_path / "plugins"),
    ).reload()


@pytest.mark.parametrize(
    "load",
    [_tagger_user_dir, _tagger_first_party, _image_user_dir],
    ids=["tagger-user", "tagger-first-party", "image"],
)
def test_a_plugin_body_runs_under_the_lock_and_cannot_hang_its_loader(
    load, tmp_path, monkeypatch
):
    """Covers ``_import_user_module``, ``_load_plugin`` and ``_load_module``."""
    (tmp_path / "mli_plugin_spawned.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sink = types.ModuleType("mli_plugin_sink")
    monkeypatch.setitem(sys.modules, "mli_plugin_sink", sink)
    for name in _PLUGIN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    errors: list[BaseException] = []

    def _load():
        try:
            load(tmp_path)
        except Exception as exc:  # recorded and asserted on below
            errors.append(exc)

    try:
        # On a worker, so a deadlock fails here instead of hanging the shard.
        assert _finishes(_load, 10), "the plugin loader hung"
    finally:
        for name in _PLUGIN_MODULES:
            sys.modules.pop(name, None)
    assert errors == []
    assert sink.held is True, "the plugin body ran outside the lock"
    assert sink.reentered is True
    # Unset when the spawned thread's import raised: read as "not spawned".
    assert (getattr(sink, "spawned", False), sink.joined) == (True, True), (
        "the plugin body's thread did not finish its plain import"
    )
    assert not MODEL_CONSTRUCTION_LOCK._is_owned()


# --------------------------------------------------------------------------
# The class: every function-local import of the ML stack goes through it.
# --------------------------------------------------------------------------

ML_PACKAGES = frozenset(
    {
        "accelerate",
        "bitsandbytes",
        "diffusers",
        "huggingface_hub",
        "insightface",
        "onnxruntime",
        "open_clip",
        "peft",
        "safetensors",
        "sentence_transformers",
        "timm",
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
    }
)

# The only function-local loads outside a block, by (file, enclosing
# function, imported module). None starts an ML import: the trainer's
# sitecustomize chain-loads the user's own ``sitecustomize`` in a subprocess,
# and the rotate step loads a PIL-only image plugin. (``loaded_module`` once
# sat here; it no longer imports at all, so it has no entry.) There is no
# pending list: a new unguarded ML import fails the sweep below.
NOT_ML = {
    (
        "pixlstash/training/trainer_hook/sitecustomize.py",
        "_start",
        "import_module",
    ),
    (
        "pixlstash/services/operation_log_service.py",
        "_rotate_picture_boxes",
        "import_module",
    ),
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported(node: ast.AST, module: str, is_package: bool) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        return [node.module]
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)]
    return [".".join(base + ([node.module] if node.module else []))]


def _is_type_checking(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and (
        getattr(node.test, "id", None) == "TYPE_CHECKING"
        or getattr(node.test, "attr", None) == "TYPE_CHECKING"
    )


def _ml_imports_call(node: ast.AST) -> ast.Call | None:
    """The ``ml_imports(...)`` call a ``with`` statement enters, if any."""
    if not isinstance(node, ast.With):
        return None
    for item in node.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and getattr(call.func, "id", None) == "ml_imports"
        ):
            return call
    return None


def _is_ml_imports_block(node: ast.AST) -> bool:
    """``with ml_imports():``, or the lock itself, which is stronger.

    A loader whose target varies (``import_module(path)``) takes the lock
    directly: ``ml_imports`` stops locking once its line has run once.
    """
    return _ml_imports_call(node) is not None or (
        isinstance(node, ast.With)
        and any(
            getattr(item.context_expr, "id", None) == "MODEL_CONSTRUCTION_LOCK"
            for item in node.items
        )
    )


def _top_level_imports(tree: ast.Module, module: str, is_package: bool) -> set:
    names: set[str] = set()

    def walk(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(_imported(node, module, is_package))
            elif isinstance(node, ast.If) and not _is_type_checking(node):
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)
                walk(node.finalbody)

    walk(tree.body)
    return names


@cache
def _parsed() -> dict[Path, ast.Module]:
    """Every module under ``pixlstash/``, parsed once for both sweeps."""
    files = sorted((REPO_ROOT / "pixlstash").rglob("*.py"))
    return {path: ast.parse(path.read_text(encoding="utf-8")) for path in files}


def _sweep() -> tuple[list[tuple[str, str, str, bool]], set[str]]:
    """Every function-local import that can load the ML stack.

    Returns (file, enclosing function, module, inside ml_imports) per import,
    and the pixlstash modules that import the stack at their top level, which
    makes a deferred import of one of them a first ML import too.

    Also fails on an ``ml_imports`` block holding anything but import
    statements. Such a statement runs under the lock on the block's first
    pass and outside it on every later one, so whatever it does is serialised
    only by accident; a model call there would hold the lock on a request.
    (A bare ``MODEL_CONSTRUCTION_LOCK`` block is a loader and runs code.)
    """
    trees = _parsed()
    top = {
        _module_name(path): _top_level_imports(
            tree, _module_name(path), path.name == "__init__.py"
        )
        for path, tree in trees.items()
    }
    heavy_modules: set[str] = set()

    def is_ml(name: str) -> bool:
        return name.split(".")[0] in ML_PACKAGES or name in heavy_modules

    changed = True
    while changed:
        changed = False
        for module, names in top.items():
            if module not in heavy_modules and any(is_ml(n) for n in names):
                heavy_modules.add(module)
                changed = True

    found = []
    not_imports = []
    for path, tree in trees.items():
        rel = path.relative_to(REPO_ROOT).as_posix()
        module, is_package = _module_name(path), path.name == "__init__.py"

        def visit(node, function, guarded):
            for child in ast.iter_child_nodes(node):
                if _is_type_checking(child):
                    continue
                inner_function, inner_guarded = function, guarded
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner_function, inner_guarded = child.name, False
                elif _is_ml_imports_block(child):
                    inner_guarded = True
                    if _ml_imports_call(child) is not None:
                        not_imports.extend(
                            f"{rel}:{stmt.lineno}"
                            for stmt in child.body
                            if not isinstance(stmt, (ast.Import, ast.ImportFrom))
                        )
                if function and isinstance(child, (ast.Import, ast.ImportFrom)):
                    for name in _imported(child, module, is_package):
                        if is_ml(name):
                            found.append((rel, function, name, guarded))
                if (
                    function
                    and isinstance(child, ast.Call)
                    and getattr(child.func, "attr", None)
                    in ("import_module", "exec_module")
                ):
                    found.append((rel, function, child.func.attr, guarded))
                visit(child, inner_function, inner_guarded)

        visit(tree, None, False)
    assert not_imports == [], (
        "keep an ml_imports block to import statements; move the rest below "
        f"it: {not_imports}"
    )
    return found, heavy_modules


def test_every_deferred_ml_import_is_in_an_ml_imports_block():
    found, heavy_modules = _sweep()
    # The sweep must see the stack, or an empty result proves nothing.
    assert {"pixlstash.tagger_plugins.wd14"} <= heavy_modules
    assert sum(1 for *_, guarded in found if guarded) >= 50

    unguarded = {tuple(site) for *site, guarded in found if not guarded}
    assert unguarded - NOT_ML == set()
    # An entry that no longer matches anything has moved or gone, and must be
    # removed, or it would silently excuse a new import at that site.
    assert NOT_ML - unguarded == set()


# Blocks a route or a poll runs, by (file, enclosing function):
# GET /workers/progress where NVML has no figure, device resolution, saving
# the VRAM budget, the model-folder listing, and the tagger list, download
# and delete routes. Each names its modules (see ``ml_imports``), or the
# request waits on any model download. The set is exact: naming a block
# anywhere else needs an entry here, so every block that can skip the lock is
# listed and checked below.
REQUEST_PATH_BLOCKS = {
    ("pixlstash/inference/vram_budget.py", "set_budget_gb"),
    ("pixlstash/services/builtin_caches.py", "huggingface_cache_dir"),
    ("pixlstash/services/config_service.py", "collect_vram_from_torch"),
    ("pixlstash/tagger_plugins/clip_service.py", "encode_image_batch"),
    ("pixlstash/tagger_plugins/clip_service.py", "encode_text"),
    ("pixlstash/tagger_plugins/joycaption.py", "delete_artifact"),
    ("pixlstash/tagger_plugins/joycaption.py", "list_downloaded_artifacts"),
    ("pixlstash/tagger_plugins/joycaption.py", "needs_download"),
    ("pixlstash/utils/accelerator.py", "_torch"),
}


def _named_blocks() -> list[tuple[str, int, str | None, list, ast.With]]:
    """Every ``ml_imports`` block that names modules.

    Returns (file, line, enclosing function, the call's arguments, block).
    """
    found = []
    for path, tree in _parsed().items():
        rel = path.relative_to(REPO_ROOT).as_posix()

        def visit(node, function):
            for child in ast.iter_child_nodes(node):
                inner = function
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner = child.name
                call = _ml_imports_call(child)
                if call is not None and call.args:
                    found.append((rel, child.lineno, function, call.args, child))
                visit(child, inner)

        visit(tree, None)
    return found


def test_request_path_blocks_name_the_modules_they_import():
    """A named block skips the lock only if its statements cannot run a body.

    So each statement imports from a module the block names, by its full
    name, and a ``from`` import takes only names that module already
    defines: a lazy package (``huggingface_hub``) imports a submodule when
    one of its names is first read, which would run that body unlocked.
    """
    blocks = _named_blocks()
    assert {(rel, function) for rel, _, function, _, _ in blocks} == (
        REQUEST_PATH_BLOCKS
    )
    problems = []
    for rel, line, _function, args, block in blocks:
        names = [a.value for a in args if isinstance(a, ast.Constant)]
        if len(names) != len(args) or not all(isinstance(n, str) for n in names):
            problems.append(f"{rel}:{line} names a module by a non-literal")
            continue
        for stmt in block.body:
            if isinstance(stmt, ast.Import):
                stray = [a.name for a in stmt.names if a.name not in names]
            elif (
                isinstance(stmt, ast.ImportFrom)
                and stmt.level == 0
                and stmt.module in names
            ):
                defined = vars(importlib.import_module(stmt.module))
                stray = [
                    f"{stmt.module}.{a.name} (not defined there)"
                    for a in stmt.names
                    if a.name not in defined
                ]
            else:
                stray = [ast.unparse(stmt)]
            problems += [f"{rel}:{stmt.lineno} {item}" for item in stray]
    assert problems == []


def _module_scope_blocks() -> list[tuple[str, int]]:
    """Every block that runs while its own module is being imported.

    Module-level statements and class bodies run at import time, nested in an
    ``if`` or ``try`` or not; function bodies do not (a ``with`` is a
    statement, so a decorator or default argument cannot hold one).
    """
    found = []
    for path, tree in _parsed().items():

        def visit(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_ml_imports_block(child):
                    found.append((path.relative_to(REPO_ROOT).as_posix(), child.lineno))
                visit(child)

        visit(tree)
    return found


def test_no_import_block_runs_at_module_scope():
    """The module-body precondition, enforced where it is cheap.

    A module body that enters a block takes the lock while its importer holds
    that module's import lock: the reverse of the order everywhere else (the
    lock, then module locks). A thread that holds the lock and imports the
    same module then waits forever, and CPython's deadlock detection sees only
    module locks. Plugin bodies outside ``pixlstash/`` are the exception the
    loaders make safe by taking the lock first. A body *calling* a function
    that holds a block is the transitive case; deciding it needs a call graph,
    so it is documented in ``docs/backend_architecture.md`` instead.
    """
    assert _module_scope_blocks() == []
