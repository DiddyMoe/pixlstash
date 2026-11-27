"""The integer range SQLite can store, as a type a request input can declare.

SQLite's INTEGER is signed 64-bit; binding a larger Python int raises
``OverflowError`` in the driver, which reaches the client as a 500.

A route parameter keeps the bound only when ``Query``/``Path`` sits inside the
``Annotated`` or is not the default: ``x: SqliteInt = Query(...)`` drops it,
so write ``x: Annotated[SqliteInt, Query()]`` or give the ``Query`` its own
``le=SQLITE_INT_MAX``.
"""

from typing import Annotated

from pydantic import Field

SQLITE_INT_MIN = -(2**63)
SQLITE_INT_MAX = 2**63 - 1

SqliteInt = Annotated[int, Field(ge=SQLITE_INT_MIN, le=SQLITE_INT_MAX)]


def parse_sqlite_int(value) -> int:
    """``int(value)`` that also raises ``ValueError`` outside SQLite's range."""
    try:
        number = int(value)
    except OverflowError as exc:
        # JSON reads 1e999 as float("inf"), which int() cannot convert.
        raise ValueError("integer outside SQLite's 64-bit range") from exc
    if not SQLITE_INT_MIN <= number <= SQLITE_INT_MAX:
        raise ValueError("integer outside SQLite's 64-bit range")
    return number
