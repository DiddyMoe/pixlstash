"""``loaded_module``: a module another thread is still importing is never
handed out half-built, and nothing in ``pixlstash/`` reads an ML package's
``sys.modules`` entry around it.

A ``sys.modules`` entry exists from the moment a module's body starts
running. Code that only wants torch "if something already imported it" and
reads the entry directly can get a torch whose ``cuda`` attribute is not set
yet, and reading ``torch.cuda`` off it raises AttributeError.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import types

import pytest

from pixlstash.utils.model_utils import loaded_module
from tests.package_sources import package_sources
from tests.test_ml_first_import_race import (
    ML_PACKAGES,
    REPO_ROOT,
    assert_imported_this_checkout,
    checkout_env,
)

# Files allowed to read an ML package's entry directly, with the reason.
RAW_READ_ALLOWED = {
    # Runs inside the trainer's own subprocess, where pixlstash is not
    # importable, and already treats a half-built torch as "not yet"
    # (``except AttributeError: continue`` on its next tick).
    "pixlstash/training/trainer_hook/sitecustomize.py",
}


def _ml_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value.split(".")[0] in ML_PACKAGES:
            return node.value
    return None


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _raw_reads(tree: ast.AST) -> list[tuple[int, str]]:
    """``sys.modules.get("torch")`` and ``sys.modules["torch"]`` loads.

    A membership test (``"torch" in sys.modules``) is not a read: it hands
    out no module.
    """
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_sys_modules(node.func.value)
            and node.args
            and (name := _ml_literal(node.args[0]))
        ):
            found.append((node.lineno, name))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _is_sys_modules(node.value)
            and (name := _ml_literal(node.slice))
        ):
            found.append((node.lineno, name))
    return found


def test_no_ml_package_is_read_off_sys_modules_directly():
    offenders = []
    for path in package_sources(REPO_ROOT / "pixlstash"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in RAW_READ_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        offenders += [f"{rel}:{line} {name}" for line, name in _raw_reads(tree)]
    assert offenders == [], (
        "read through pixlstash.utils.model_utils.loaded_module() instead, "
        f"which never returns a module that is still being imported: {offenders}"
    )


def test_the_guard_sees_both_read_forms_and_passes_membership():
    tree = ast.parse(
        "import sys\n"
        "a = sys.modules.get('torch')\n"
        "b = sys.modules['transformers.utils']\n"
        "c = 'torch' in sys.modules\n"
        "d = sys.modules.get('json')\n"
    )
    assert _raw_reads(tree) == [(2, "torch"), (3, "transformers.utils")]


# --- behaviour ---------------------------------------------------------------


def test_a_module_never_imported_is_none(monkeypatch):
    monkeypatch.delitem(sys.modules, "lm_never_imported", raising=False)
    assert loaded_module("lm_never_imported") is None
    assert "lm_never_imported" not in sys.modules, "loaded_module imported it"


def test_a_blocked_module_is_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_blocked", None)
    assert loaded_module("lm_blocked") is None


def test_a_planted_stand_in_is_returned_as_it_is(monkeypatch):
    stand_in = types.SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "lm_stand_in", stand_in)
    assert loaded_module("lm_stand_in") is stand_in


# The module another thread is importing: its body stops half-way until the
# test lets it finish, or fail. ``runs`` counts how often the body executed,
# so a second import started by a reader shows up.
_SLOW_BODY = """
import lm_gate

lm_gate.runs += 1
lm_gate.entered.set()
assert lm_gate.release.wait(30), "the test never released the import"
if lm_gate.fail:
    raise RuntimeError("lm_slow failed to import")
READY = True
"""


@pytest.fixture
def slow_module(tmp_path, monkeypatch):
    (tmp_path / "lm_slow.py").write_text(_SLOW_BODY)
    monkeypatch.syspath_prepend(str(tmp_path))
    gate = types.ModuleType("lm_gate")
    gate.entered = threading.Event()
    gate.release = threading.Event()
    gate.fail = False
    gate.runs = 0
    monkeypatch.setitem(sys.modules, "lm_gate", gate)
    monkeypatch.delitem(sys.modules, "lm_slow", raising=False)
    importer_errors: list[BaseException] = []

    def _import():
        try:
            import lm_slow  # noqa: F401
        except RuntimeError as exc:  # asserted on by the failing-import test
            importer_errors.append(exc)

    importer = threading.Thread(target=_import, name="lm-importer")
    importer.start()
    assert gate.entered.wait(10)
    try:
        yield gate, importer, importer_errors
    finally:
        gate.release.set()
        importer.join(10)
        sys.modules.pop("lm_slow", None)


def _read_in_thread():
    """Call ``loaded_module`` on a thread; record what it saw on return."""
    seen: dict = {}

    def _read():
        try:
            module = loaded_module("lm_slow")
            seen["ready"] = getattr(module, "READY", None)
            seen["module"] = module
        except BaseException as exc:  # loaded_module must never raise
            seen["error"] = exc

    reader = threading.Thread(target=_read, name="lm-reader")
    reader.start()
    return reader, seen


def test_a_module_mid_import_is_none_at_once_then_the_whole_module(slow_module):
    gate, importer, _errors = slow_module
    partial = sys.modules["lm_slow"]
    assert not hasattr(partial, "READY"), "the import did not stop half-way"

    reader, seen = _read_in_thread()
    reader.join(5)
    assert not reader.is_alive(), "loaded_module waited for the import in flight"
    assert seen == {"ready": None, "module": None}

    gate.release.set()
    importer.join(10)
    assert not importer.is_alive()
    module = loaded_module("lm_slow")
    # A half-built module handed out early would lack READY.
    assert module is sys.modules["lm_slow"]
    assert module.READY is True
    assert gate.runs == 1


def test_a_failed_import_in_flight_is_none_and_never_retried(slow_module):
    gate, importer, importer_errors = slow_module
    reader, seen = _read_in_thread()
    reader.join(5)
    assert not reader.is_alive()
    assert seen == {"ready": None, "module": None}

    gate.fail = True
    gate.release.set()
    importer.join(10)
    assert not importer.is_alive()
    assert len(importer_errors) == 1

    # CPython removed the failed entry. The import's error belongs to the
    # thread that imported; a reader gets None and starts no import of its own.
    assert loaded_module("lm_slow") is None
    assert gate.runs == 1, "loaded_module ran the module's body again"
    assert "lm_slow" not in sys.modules


# The review's probe, as a test: the planner polls ``loaded_module("torch")``
# every cycle while a burst is in flight. A torch whose first import blocks and
# then fails on another thread must neither stall the planner while it runs
# nor end it, nor be imported a second time by the poll. It runs in a fresh
# interpreter so the stand-in torch never meets the real one.
_FAKE_TORCH = """
import lm_gate

lm_gate.runs += 1
lm_gate.entered.set()
assert lm_gate.release.wait(30), "the test never released the import"
raise OSError("example torch DLL failed to load")
"""

_PLANNER_DRIVER = """
import json
import sys
import threading
import time
import types

sys.path.insert(0, sys.argv[1])
gate = types.ModuleType("lm_gate")
gate.entered = threading.Event()
gate.release = threading.Event()
gate.runs = 0
sys.modules["lm_gate"] = gate

import pixlstash.tasks  # noqa: E402,F401  (the app's import order)
from pixlstash.work_planner import WorkPlanner, _PassStats  # noqa: E402


class Runner:
    def replace_dead_workers(self):
        return None

    def submit(self, task):
        raise AssertionError("no finder is registered")


planner = WorkPlanner(task_runner=Runner(), task_finders=[])
planner._pass_stats["MissingTagFinder"] = _PassStats(started_at=time.perf_counter())
importer_error = []


def _import():
    try:
        import torch  # noqa: F401
    except OSError as exc:
        importer_error.append(str(exc))


importer = threading.Thread(target=_import, name="gpu-worker", daemon=True)
importer.start()
assert gate.entered.wait(10)

planner.start()
deadline = time.monotonic() + 5
while not planner._gpu_util_torch_missing_logged and time.monotonic() < deadline:
    time.sleep(0.01)
polled_while_in_flight = planner._gpu_util_torch_missing_logged
gate.release.set()
importer.join(10)
for _ in range(5):
    planner._wake.set()
    time.sleep(0.1)
result = {
    "pixlstash": pixlstash.__file__,
    "polled_while_in_flight": polled_while_in_flight,
    "alive_after_failure": planner.is_running(),
    "torch_bodies": gate.runs,
    "importer_error": importer_error,
    "torch_loaded": "torch" in sys.modules,
}
planner.stop()
print(json.dumps(result))
"""


def test_a_failed_torch_import_in_flight_neither_stalls_nor_ends_the_planner(
    tmp_path,
):
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "torch.py").write_text(_FAKE_TORCH)
    driver = tmp_path / "driver.py"
    driver.write_text(_PLANNER_DRIVER)
    proc = subprocess.run(
        [sys.executable, str(driver), str(fakes)],
        cwd=REPO_ROOT,
        env=checkout_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_imported_this_checkout(result.pop("pixlstash"))
    assert result == {
        "polled_while_in_flight": True,
        "alive_after_failure": True,
        "torch_bodies": 1,
        "importer_error": ["example torch DLL failed to load"],
        "torch_loaded": False,
    }, proc.stderr[-4000:]
