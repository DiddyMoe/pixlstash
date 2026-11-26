"""Tests for ``scripts/sync-node-versions.mjs``.

The script mirrors pyproject.toml's version into the Node manifests. It is run
against a copy of itself in a throwaway tree, because it finds the files it
edits from its own location and the real ones must not move.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync-node-versions.mjs"

PEP440 = "1.6.0.dev1"
SEMVER = "1.6.0-dev.1"


def _missing(message: str) -> None:
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.fail(message)
    pytest.skip(message)


def _write_manifests(root: Path, dir_name: str, version: str, lock_root: str) -> None:
    folder = root / dir_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "package.json").write_text(
        json.dumps({"name": dir_name, "version": version}, indent=2) + "\n"
    )
    (folder / "package-lock.json").write_text(
        json.dumps(
            {
                "name": dir_name,
                "version": version,
                "packages": {
                    "": {"name": dir_name, "version": lock_root},
                    "node_modules/dep": {"version": "9.9.9"},
                },
            },
            indent=2,
        )
        + "\n"
    )


@pytest.fixture
def tree(tmp_path):
    """A repository shape holding the script, pyproject.toml and both apps'
    manifests, all at the synced version."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts")
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "pixlstash"\nversion = "{PEP440}"\n'
    )
    for dir_name in ("electron", "frontend"):
        _write_manifests(tmp_path, dir_name, SEMVER, SEMVER)
    return tmp_path


def _manifests(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.glob("*/package*.json"))
    }


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    node = shutil.which("node")
    if not node:
        _missing("node is not installed, so the script cannot be run.")
    return subprocess.run(
        [node, str(root / "scripts" / SCRIPT.name), *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_check_passes_a_synced_tree_and_writes_nothing(tree):
    before = _manifests(tree)
    done = _run(tree, "--check")
    assert done.returncode == 0, done.stderr
    assert _manifests(tree) == before


def test_check_names_the_drifted_manifest_exits_1_and_writes_nothing(tree):
    """The lockfile's self-entry is the one that drifts unseen: package.json
    shows the version, the lock's ``packages[""]`` is a second copy."""
    _write_manifests(tree, "frontend", SEMVER, "1.5.0")
    before = _manifests(tree)
    done = _run(tree, "--check")
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert "frontend/package-lock.json" in done.stderr
    assert "frontend/package.json" not in done.stderr
    assert "electron/" not in done.stderr
    assert _manifests(tree) == before


def test_check_reads_only_the_directories_named(tree):
    _write_manifests(tree, "electron", "1.5.0", "1.5.0")
    done = _run(tree, "--check", "frontend")
    assert done.returncode == 0, done.stderr
    done = _run(tree, "electron", "--check")
    assert done.returncode == 1
    assert "electron/package.json and electron/package-lock.json" in done.stderr


@pytest.mark.parametrize("option", ["--chek", "--dry-run", "-c"])
def test_an_unknown_option_is_refused_not_read_as_a_directory(tree, option):
    _write_manifests(tree, "electron", "1.5.0", "1.5.0")
    before = _manifests(tree)
    done = _run(tree, option)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert f"unknown option {option}" in done.stderr
    assert _manifests(tree) == before


def test_without_check_a_drifted_tree_is_synced(tree):
    """Positive control: the sync itself, dependency versions left alone."""
    _write_manifests(tree, "electron", "1.5.0", "1.5.0")
    done = _run(tree)
    assert done.returncode == 0, done.stderr
    assert f"electron -> {SEMVER} (from {PEP440})" in done.stdout
    lock = json.loads((tree / "electron" / "package-lock.json").read_text())
    assert lock["version"] == SEMVER
    assert lock["packages"][""]["version"] == SEMVER
    assert lock["packages"]["node_modules/dep"]["version"] == "9.9.9"
    package = json.loads((tree / "electron" / "package.json").read_text())
    assert package["version"] == SEMVER
    assert _run(tree, "--check").returncode == 0
