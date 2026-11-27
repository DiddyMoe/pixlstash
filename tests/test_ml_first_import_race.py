"""Two threads making the first ML import at once both get whole modules.

CPython's per-module import locks do not wait out a cycle between two
threads: one of them is handed the other's half-initialised module.
``accelerate``'s ``utils`` and ``state`` import each other, and both the first
search (``sentence_transformers``) and the first embedding pass
(``open_clip``) import it, so the first search could fail with "cannot import
name 'AcceleratorState' from partially initialized module".

The stand-in packages below have that shape. The driver holds the search
thread inside ``fakeaccel.utils`` until the second thread is either inside
``fakeaccel.state`` (the cycle closes and one thread fails) or waiting on
``MODEL_CONSTRUCTION_LOCK``. It runs in a subprocess so the stand-ins never
meet the real packages.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_FAKE_PACKAGES = {
    "fakeaccel/__init__.py": "from . import utils  # noqa: F401\n",
    "fakeaccel/utils.py": """
        import fakectl

        x = 1
        fakectl.utils_body()
        from .state import AcceleratorState  # noqa: E402,F401
    """,
    "fakeaccel/state.py": """
        import fakectl

        fakectl.in_state.set()
        from .utils import x  # noqa: E402,F401


        class AcceleratorState:
            pass
    """,
    # The search's import enters the cycle from the package's top.
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
    # The embedding pass's import and the plugin enter it from state.
    "open_clip/__init__.py": """
        import fakeaccel.state  # noqa: F401

        from . import pretrained, transform  # noqa: F401


        class _Model:
            def to(self, device):
                return self

            def half(self):
                return self


        def get_model_config(name):
            return {"vision_cfg": {"image_size": 224}}


        def create_model_and_transforms(name, pretrained=None):
            return _Model(), None, transform.image_transform_v2(None, False)


        def get_tokenizer(name):
            return str.split
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
    "fakeplugin.py": """
        import fakeaccel.state  # noqa: F401


        class FakePlugin:
            name = "fake"
    """,
}

_DRIVER = """
    import importlib._bootstrap as bootstrap
    import json
    import linecache
    import sys
    import threading
    import time
    import types

    pkgroot, case = sys.argv[1], sys.argv[2]
    sys.path.insert(0, pkgroot)
    DEADLINE_S = 30.0

    ctl = types.ModuleType("fakectl")
    ctl.in_utils = threading.Event()
    ctl.in_state = threading.Event()
    ctl.second = None
    ctl.held_once = False
    sys.modules["fakectl"] = ctl


    def _second_waits_on_the_lock():
        frame = sys._current_frames().get(ctl.second)
        if frame is None:
            return False
        line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
        return "with MODEL_CONSTRUCTION_LOCK" in line


    def utils_body():
        # The search thread, inside fakeaccel.utils and holding its lock.
        if ctl.held_once:
            return
        ctl.held_once = True
        ctl.in_utils.set()
        deadline = time.monotonic() + DEADLINE_S
        while not ctl.in_state.is_set():
            if _second_waits_on_the_lock():
                return
            if time.monotonic() > deadline:
                raise AssertionError("the second thread never reached the import")
            time.sleep(0.001)
        # Return once the second thread queues on fakeaccel.utils, so the
        # next import here closes the cycle.
        while time.monotonic() < deadline:
            ref = bootstrap._module_locks.get("fakeaccel.utils")
            lock = ref() if ref is not None else None
            if lock is not None and lock.waiters:
                return
            time.sleep(0.001)
        raise AssertionError("the second thread never waited on fakeaccel.utils")


    ctl.utils_body = utils_body

    from PIL import Image

    from pixlstash.tagger_plugins.clip_service import ClipService
    from pixlstash.tagger_plugins.registry import TaggerPluginManager
    from pixlstash.utils import model_utils

    results = {"pixlstash": model_utils.__file__}


    def search():
        model_utils.load_sentence_transformer("example-model")


    def tagger_plugin():
        manager = TaggerPluginManager(
            user_dir=None, first_party=[("fakeplugin", "FakePlugin")]
        )
        manager.reload()
        if manager.list_errors():
            raise ImportError(manager.list_errors()[0]["message"])


    SECOND = {
        "clip-load": lambda: ClipService("cpu").ensure_ready(),
        "clip-first-resize": lambda: ClipService("cpu").reduce_for_preprocess(
            [Image.new("RGB", (8, 8))]
        ),
        "tagger-plugin": tagger_plugin,
    }


    def _run(name, fn):
        try:
            fn()
            results[name] = "ok"
        except BaseException as exc:
            results[name] = f"{type(exc).__name__}: {exc}"


    t1 = threading.Thread(target=_run, args=("search", search))
    t1.start()
    if not ctl.in_utils.wait(DEADLINE_S):
        raise SystemExit("the search thread never reached fakeaccel.utils")
    t2 = threading.Thread(target=_run, args=("second", SECOND[case]))
    t2.start()
    ctl.second = t2.ident
    t1.join(DEADLINE_S)
    t2.join(DEADLINE_S)
    results["hung"] = t1.is_alive() or t2.is_alive()
    print(json.dumps(results))
"""


@pytest.mark.parametrize("case", ["clip-load", "clip-first-resize", "tagger-plugin"])
def test_two_first_imports_at_once_both_get_whole_modules(case, tmp_path):
    pkgroot = tmp_path / "pkgs"
    for rel, body in _FAKE_PACKAGES.items():
        path = pkgroot / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(_DRIVER))
    # PYTHONPATH, so the driver imports this checkout's pixlstash rather than
    # whichever tree the editable install points at.
    proc = subprocess.run(
        [sys.executable, str(driver), str(pkgroot), case],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(results.pop("pixlstash")).is_relative_to(REPO_ROOT / "pixlstash")

    assert results == {"search": "ok", "second": "ok", "hung": False}
