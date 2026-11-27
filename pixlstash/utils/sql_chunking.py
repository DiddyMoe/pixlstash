"""One definition of the SQLite bound-parameter cap and the ways to stay under it.

SQLite limits how many bound parameters a single statement may carry
(``SQLITE_LIMIT_VARIABLE_NUMBER``: 32766 on current builds, 999 on older ones).
Any ``WHERE col IN (:a, :b, ...)`` built from a caller-sized id list can cross
that cap and fail at execution time. A load is chunked; an exclusion cannot be
(``NOT IN`` over one chunk says nothing about the others), so it binds the
whole set as one JSON array instead, and so does a membership test that must
stay in one statement (:func:`in_json`).

The limit and the chunker used to be redeclared in five modules, each with its
own copy of this rationale and no shared value to change. They live here now.
"""

import json
from typing import Collection, Iterator, Sequence, TypeVar

from sqlalchemy import func, select
from sqlalchemy.sql.elements import ColumnElement

T = TypeVar("T")

# Chunk size for ``IN`` lists built from ids. Sits under the 999-parameter floor
# with room for the other bound values a statement carries alongside the list.
SQLITE_ID_CHUNK = 900


def chunked(seq: Sequence[T], size: int = SQLITE_ID_CHUNK) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of *seq* of at most *size* items.

    Args:
        seq: Sequence to split. An empty sequence yields nothing.
        size: Maximum items per slice; defaults to :data:`SQLITE_ID_CHUNK`.

    Yields:
        Slices of *seq* in order, each at most *size* long.

    Raises:
        ValueError: *size* is not positive (which would loop forever).
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def not_in_json(column, values: Collection[int]) -> ColumnElement:
    """``column NOT IN values`` with *values* bound as ONE JSON array.

    Compiles to ``column NOT IN (SELECT value FROM json_each(:param))``, so the
    statement carries a single parameter however many values there are. SQLite
    evaluates the uncorrelated subquery once into an ephemeral index and probes
    it per row, which keeps the exclusion ahead of an ``ORDER BY ... LIMIT``
    without a scan per value.

    Args:
        column: The column to test, e.g. ``Picture.id``.
        values: Integer ids to exclude. Sorted before encoding so equal sets
            bind byte-identical text.

    Returns:
        The predicate. An empty *values* excludes nothing and is a valid
        clause, so callers need no special case for it.
    """
    rows = func.json_each(json.dumps(sorted(values))).table_valued("value")
    return column.not_in(select(rows.c.value))


def in_json(column, values: Collection) -> ColumnElement:
    """``column IN values`` with *values* bound as ONE JSON array.

    The membership twin of :func:`not_in_json`, for a list that has to stay in
    one statement: under a ``GROUP BY ... LIMIT`` (where :func:`chunked` would
    change the answer) or a one-line swap at a caller-sized ``.in_()``. The
    match is SQLite's ``=``, so text compares case-sensitively.

    Args:
        column: The column or expression to test, e.g. ``Picture.id``.
        values: Integers or strings to match. Sorted before encoding so equal
            sets bind byte-identical text.

    Returns:
        The predicate. An empty *values* matches nothing.
    """
    rows = func.json_each(json.dumps(sorted(values))).table_valued("value")
    return column.in_(select(rows.c.value))
