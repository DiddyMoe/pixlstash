"""Every database write a background task queues names its priority.

``VaultDatabase.run_task`` defaults to ``DBPriority.IMMEDIATE``, the level of
a request the user is waiting on. A task under ``pixlstash/tasks/`` that
leaves the argument out queues its write at that level, ahead of the clicks
the writer queue exists to keep responsive, and nothing at the call site says
it was a choice. So each call there names one: ``LOW`` for the bulk of
background work, and a higher level only where the line says why.
"""

import ast
from pathlib import Path

_TASKS = Path(__file__).resolve().parents[1] / "pixlstash" / "tasks"


def _run_task_calls(source: str):
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_task"
        ):
            yield node


def test_the_scan_sees_the_task_writes():
    """The walk finds the calls, so an empty result below is not vacuous."""
    found = sum(
        1
        for path in _TASKS.glob("*.py")
        for _call in _run_task_calls(path.read_text(encoding="utf-8"))
    )
    assert found >= 50, f"only {found} run_task calls under pixlstash/tasks/"


def test_every_task_write_names_its_priority():
    unnamed = sorted(
        f"{path.name}:{call.lineno}"
        for path in _TASKS.glob("*.py")
        for call in _run_task_calls(path.read_text(encoding="utf-8"))
        if not any(keyword.arg == "priority" for keyword in call.keywords)
    )
    assert not unnamed, (
        "These run_task calls take the IMMEDIATE default, the level of a "
        "request the user is waiting on. Name the priority (DBPriority.LOW "
        "for background work):\n" + "\n".join(f"  {site}" for site in unnamed)
    )
