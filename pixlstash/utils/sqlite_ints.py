"""The integer range SQLite can store, as one type a request input can declare.

SQLite's INTEGER is a signed 64-bit value. A larger Python int does not
wrap or truncate: the driver raises ``OverflowError`` while binding it, which
reaches the client as a 500. An id, a count or a cursor read from a request
is therefore declared ``SqliteInt`` (or ``list[SqliteInt]``), so validation
answers 422 naming the input before any query is built, and the bound shows
in the OpenAPI schema as ``minimum``/``maximum``.

``annotated_types.Interval`` rather than a pydantic ``Field``: the same
metadata works on a path, query or form parameter and on a model field.

A route parameter keeps the bound only when ``Query``/``Path``/``Form``
sits inside the ``Annotated`` or the parameter has no such default:
``x: Annotated[SqliteInt, Query()]``, ``x: SqliteInt = 0`` and
``x: Optional[SqliteInt] = Query(None)`` are bounded, while
``x: SqliteInt = Query(...)`` silently drops the Interval (FastAPI 0.165), so
``2**63`` is accepted and overflows in the driver. Either use the
``Annotated`` form or give the ``Query`` its own ``le=SQLITE_INT_MAX``.
``tests/test_route_int_bounds.py`` reads the runtime parameter, so it catches
this where reading the source would not.
"""

from dataclasses import dataclass
from typing import Annotated

import annotated_types

#: The smallest integer SQLite stores.
SQLITE_INT_MIN = -(2**63)
#: The largest integer SQLite stores.
SQLITE_INT_MAX = 2**63 - 1

#: An ``int`` that SQLite can store; anything outside is a 422.
SqliteInt = Annotated[
    int, annotated_types.Interval(ge=SQLITE_INT_MIN, le=SQLITE_INT_MAX)
]


@dataclass(frozen=True)
class NotASqliteInteger:
    """Metadata for a request int that is never bound as a SQLite INTEGER.

    A ComfyUI widget value or a sampler seed goes to ComfyUI and is kept as
    TEXT, and a seed's legal range (up to ``2**64 - 1``) is wider than
    SQLite's. This marker says so at the declaration and in the OpenAPI
    schema (``x-not-sqlite-integer``, with the reason).
    ``tests/test_route_int_bounds.py`` accepts an integer outside SQLite's
    range only when it carries this marker, so the exception is made where the
    input is declared, with its reason, never in an allowlist in the test.
    Code that later binds such a value as an INTEGER must re-validate it as
    :data:`SqliteInt` first.
    """

    reason: str

    def __get_pydantic_json_schema__(self, core_schema, handler):
        schema = handler(core_schema)
        schema["x-not-sqlite-integer"] = self.reason
        return schema


def parse_sqlite_int(value) -> int:
    """``int(value)`` for an id a handler parses itself, inside SQLite's range.

    Some routes take an id as a ``str`` path segment or read it out of a
    ``dict`` body, so validation never sees it as an integer. ``int()`` of a
    long digit string succeeds and the value overflows later, in the driver.
    This raises ``ValueError`` for such a value, the same exception ``int()``
    raises for text that is not a number, so the caller's existing
    ``except (TypeError, ValueError)`` answers both.

    Raises:
        TypeError: *value* is not a number or a string (as ``int()``).
        ValueError: *value* is not an integer, or SQLite cannot store it.
    """
    number = int(value)
    if not SQLITE_INT_MIN <= number <= SQLITE_INT_MAX:
        raise ValueError("integer outside SQLite's 64-bit range")
    return number


def is_sqlite_int(value) -> bool:
    """True when *value* is an ``int`` (not a ``bool``) that SQLite can store."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and SQLITE_INT_MIN <= value <= SQLITE_INT_MAX
    )
