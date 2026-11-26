"""The modules a test that reads PixlStash's own source should walk.

``setup.py`` copies the ComfyUI-PixlStash nodes into ``pixlstash/data/comfyui-pack/``
on any non-editable install, CI's ``pip install .`` included. That copy is another
project's code, run inside ComfyUI's own server and held to its own rules in its
own repository (gitignored here, and ruff excludes it too). A walk of
``pixlstash/`` that reads it judges ComfyUI code by PixlStash's rules.
"""

from pathlib import Path

# Where setup.py stages the nodes, relative to the package root.
STAGED_COMFYUI_PACK = ("data", "comfyui-pack")


def package_sources(root: Path) -> list[Path]:
    """Every ``*.py`` under the package root *root*, sorted, minus the staged nodes.

    Args:
        root: A ``pixlstash`` package directory: the checkout's, or the one an
            installed ``pixlstash.__file__`` names.

    Returns:
        The paths, sorted.
    """
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[: len(STAGED_COMFYUI_PACK)]
        != STAGED_COMFYUI_PACK
    )
