"""One spelling of "this path is in folder F" for SQL and for Python.

A folder holds the paths that continue past its name with a separator:
``/lib/photos`` holds ``/lib/photos/a.jpg`` and ``/lib/photos/2020/b.jpg``, and
does NOT hold ``/lib/photos-old/c.jpg`` or ``/lib/photosX/d.jpg``. A bare
``startswith`` or ``LIKE '<folder>%'`` gets that wrong: it walks into every
prefix sibling. SQLite's ``LIKE`` is also case-insensitive for ASCII by default
(https://www.sqlite.org/lang_expr.html#like), so on a case-sensitive
filesystem (Linux, and macOS only on a case-sensitive volume) it also pulled in
``/lib/PHOTOS``, which is a different folder there.

**The predicate is a half-open range per separator**, ``col >= F + sep AND
col < F + chr(ord(sep) + 1)``. Every string that starts with ``F + sep`` sorts
inside that range and nothing else does, so the range is exact on its own, needs
no escaping (``%`` and ``_`` are ordinary characters in a comparison), and is
the form SQLite can serve from an index on the column: the right-most column of
an index may carry up to two inequality constraints
(https://www.sqlite.org/optoverview.html#where_clause_analysis). ``LIKE`` gets
the same index use only under the conditions of the LIKE optimisation
(https://www.sqlite.org/optoverview.html#the_like_optimization), and only for
a pattern with no leading wildcard.

**Case follows the platform.** Where ``os.path.normcase`` folds case
(Windows; the rule :func:`pixlstash.utils.path_utils.path_is_within` already
uses) the comparison is ``COLLATE NOCASE``. macOS's default volume folds case
too, but ``normcase`` does not say so there, so the comparison stays exact and
a path stored in another case is not in the folder. An explicit ``COLLATE`` on an operand decides
the comparison's collation
(https://sqlite.org/datatype3.html#assigning_collating_sequences_from_sql), and
NOCASE folds only the 26 ASCII letters
(https://sqlite.org/datatype3.html#collating_sequences). The range bounds stay
exact under NOCASE because every bound character (``/ 0 \\ ]``) is outside
the letters, so folding cannot move a path across one. The Python mirror
(:meth:`FolderPrefix.contains`) folds the same 26 letters and nothing else.

**Separators follow the platform too:** ``os.sep`` and ``os.altsep``, which
on Windows are ``\\`` and ``/`` and on POSIX only ``/``, because ``\\`` is an
ordinary file-name character there. The alternate separator is accepted only
right after the folder name; the folder's own spelling is compared as written.

**Callers normalise first.** The folder and every path compared with it must
be normalised (``os.path.normpath`` / ``os.path.abspath``) with the folder's own
separator. The checks are lexical: they do not collapse ``..`` (``F/../x``
counts as below ``F``) and do not resolve links.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass

from sqlalchemy import and_, func, or_
from sqlalchemy.sql.elements import ColumnElement

PATHS_FOLD_CASE: bool = os.path.normcase("A") == "a"
"""True where the platform's paths compare case-insensitively (Windows)."""

PATH_SEPARATORS: tuple[str, ...] = tuple(
    sep for sep in (os.sep, os.altsep) if sep is not None
)
"""The characters that end a folder name on this platform."""

NOCASE = "NOCASE"
"""SQLite's built-in ASCII case-folding collation."""

_ASCII_FOLD = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


@dataclass(frozen=True)
class FolderPrefix:
    """A folder, and the checks for "this path is in it or below it".

    Build it with :meth:`of`, which strips trailing separators. The defaults
    are this platform's; tests pass the other platform's rules explicitly.

    Attributes:
        folder: The folder, without trailing separators. ``""`` is the
            filesystem root (``/``): every absolute POSIX path is below it.
        separators: The characters that end a folder name.
        fold_case: Compare ASCII letters case-insensitively.
    """

    folder: str
    separators: tuple[str, ...] = PATH_SEPARATORS
    fold_case: bool = PATHS_FOLD_CASE

    @classmethod
    def of(
        cls,
        folder: str,
        *,
        separators: tuple[str, ...] = PATH_SEPARATORS,
        fold_case: bool = PATHS_FOLD_CASE,
    ) -> "FolderPrefix":
        """The prefix for ``folder``, with its trailing separators stripped."""
        return cls(str(folder).rstrip("".join(separators)), separators, fold_case)

    def bounds(self) -> list[tuple[str, str]]:
        """``(low, high)`` per separator: below the folder is ``low <= p < high``."""
        return [
            (self.folder + sep, self.folder + chr(ord(sep) + 1))
            for sep in self.separators
        ]

    def under(self, column) -> ColumnElement:
        """SQL: ``column`` names a path anywhere below the folder."""
        col = self._collated(column)
        return or_(*(and_(col >= low, col < high) for low, high in self.bounds()))

    def directly_in(self, column) -> ColumnElement:
        """SQL: ``column`` names a path directly in the folder, not a sub-folder."""
        rest = func.substr(column, len(self.folder) + 2)
        return and_(
            self.under(column),
            *(func.instr(rest, sep) == 0 for sep in self.separators),
        )

    def under_sql(self, column_sql: str) -> tuple[str, list[str]]:
        """Raw-SQL :meth:`under` for ``sqlite3``: a fragment and its ``?`` params.

        ``column_sql`` is spliced into the fragment, so it must be a column
        name the caller wrote, never a value.
        """
        col = f"{column_sql} COLLATE {NOCASE}" if self.fold_case else column_sql
        fragment = " OR ".join(f"({col} >= ? AND {col} < ?)" for _ in self.bounds())
        params = [bound for pair in self.bounds() for bound in pair]
        return f"({fragment})", params

    def contains(self, path: str) -> bool:
        """Python: ``path`` is below the folder (the folder itself is not)."""
        folded = self._fold(path)
        return any(folded.startswith(self._fold(low)) for low, _ in self.bounds())

    def covers(self, folder: str) -> bool:
        """Python: ``folder`` is this folder or a folder below it."""
        return self._fold(folder) == self._fold(self.folder) or self.contains(folder)

    def _collated(self, column):
        return column.collate(NOCASE) if self.fold_case else column

    def _fold(self, text: str) -> str:
        return text.translate(_ASCII_FOLD) if self.fold_case else text
