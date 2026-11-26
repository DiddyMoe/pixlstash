"""FolderPrefix: one spelling of "this path is in folder F", in SQL and Python.

Each rule set (POSIX: ``/`` and case-sensitive; Windows: ``\\`` and ``/`` and
ASCII case folded) is passed explicitly, so both run on every platform, and
every SQL answer is checked against the Python mirror and against a reference
written the slow, obvious way.
"""

import random
import sqlite3
import string

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

from pixlstash.utils.query.folder_prefix import FolderPrefix

POSIX = {"separators": ("/",), "fold_case": False}
WINDOWS = {"separators": ("\\", "/"), "fold_case": True}

_metadata = MetaData()
_paths = Table(
    "paths",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("path", String),
)


def _reference_under(folder: str, path: str, rules: dict) -> bool:
    """The obvious, slow rule: the path continues past the folder with a separator."""

    def fold(text):
        if not rules["fold_case"]:
            return text
        return "".join(c.lower() if c in string.ascii_uppercase else c for c in text)

    return any(fold(path).startswith(fold(folder) + sep) for sep in rules["separators"])


def _reference_directly_in(folder: str, path: str, rules: dict) -> bool:
    if not _reference_under(folder, path, rules):
        return False
    rest = path[len(folder) + 1 :]
    return not any(sep in rest for sep in rules["separators"])


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'paths.db'}")
    _metadata.create_all(engine)
    yield engine
    engine.dispose()


def _load(engine, paths):
    with engine.begin() as conn:
        conn.execute(_paths.delete())
        conn.execute(
            _paths.insert(), [{"id": i, "path": p} for i, p in enumerate(paths)]
        )


def _selected(engine, predicate, paths):
    with engine.connect() as conn:
        ids = conn.execute(select(_paths.c.id).where(predicate)).scalars().all()
    return {paths[i] for i in ids}


POSIX_PATHS = [
    "/lib/photos/a.jpg",
    "/lib/photos/2020/b.jpg",
    "/lib/photos-old/c.jpg",
    "/lib/photosX/d.jpg",
    "/lib/photos0/e.jpg",
    "/lib/photos.jpg",
    "/lib/photos",
    "/lib/PHOTOS/f.jpg",
    "/lib/Photos/2020/g.jpg",
    "/lib/photos\\h.jpg",
    "/lib/pho%os/i.jpg",
    "/lib/pho_os/j.jpg",
    "/lib/photos/\u00e9t\u00e9/k.jpg",
    "relative/lib/photos/l.jpg",
    None,
]


def test_posix_rules_take_the_folder_and_below_and_no_sibling(engine):
    _load(engine, POSIX_PATHS)
    prefix = FolderPrefix.of("/lib/photos/", **POSIX)

    under = _selected(engine, prefix.under(_paths.c.path), POSIX_PATHS)
    assert under == {
        "/lib/photos/a.jpg",
        "/lib/photos/2020/b.jpg",
        "/lib/photos/\u00e9t\u00e9/k.jpg",
    }
    directly = _selected(engine, prefix.directly_in(_paths.c.path), POSIX_PATHS)
    assert directly == {"/lib/photos/a.jpg"}


def test_windows_rules_fold_ascii_case_and_take_either_separator(engine):
    paths = [
        "C:\\Lib\\Photos\\a.jpg",
        "c:\\lib\\photos\\2020\\b.jpg",
        "C:\\LIB\\PHOTOS/c.jpg",
        "C:\\Lib\\Photos-old\\d.jpg",
        "C:\\Lib\\PhotosX\\e.jpg",
        "C:\\Lib\\Photos]\\f.jpg",
        "C:\\Lib\\Photos0\\g.jpg",
        "C:\\Lib\\Photos",
        "D:\\Lib\\Photos\\h.jpg",
    ]
    _load(engine, paths)
    prefix = FolderPrefix.of("C:\\Lib\\Photos\\", **WINDOWS)

    assert _selected(engine, prefix.under(_paths.c.path), paths) == {
        "C:\\Lib\\Photos\\a.jpg",
        "c:\\lib\\photos\\2020\\b.jpg",
        "C:\\LIB\\PHOTOS/c.jpg",
    }
    assert _selected(engine, prefix.directly_in(_paths.c.path), paths) == {
        "C:\\Lib\\Photos\\a.jpg",
        "C:\\LIB\\PHOTOS/c.jpg",
    }


def test_like_wildcards_in_a_folder_name_are_plain_characters(engine):
    _load(engine, POSIX_PATHS)
    for folder, expected in (
        ("/lib/pho%os", {"/lib/pho%os/i.jpg"}),
        ("/lib/pho_os", {"/lib/pho_os/j.jpg"}),
    ):
        prefix = FolderPrefix.of(folder, **POSIX)
        assert _selected(engine, prefix.under(_paths.c.path), POSIX_PATHS) == expected


def test_the_root_folder_takes_every_absolute_path_and_no_relative_one(engine):
    _load(engine, POSIX_PATHS)
    prefix = FolderPrefix.of("/", **POSIX)

    assert prefix.folder == ""
    under = _selected(engine, prefix.under(_paths.c.path), POSIX_PATHS)
    assert under == {p for p in POSIX_PATHS if p and p.startswith("/")}


@pytest.mark.parametrize("rules", [POSIX, WINDOWS], ids=["posix", "windows"])
def test_sql_python_and_the_reference_agree_on_random_paths(engine, rules):
    # Paths and folders are drawn from an alphabet holding every character the
    # range bounds and the case fold could trip on: both separators, the bound
    # characters `0` and `]`, letters in both cases, the characters that sort
    # just below a separator, and one non-ASCII letter whose case NOCASE does
    # not fold.
    alphabet = "/\\0]aAbB.-_%\u00e9\u00c9"
    rng = random.Random(20261125)
    paths = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 7)))
        for _ in range(400)
    ]
    _load(engine, paths)
    for _ in range(60):
        folder = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 3)))
        prefix = FolderPrefix(folder, **rules)
        expected_under = {p for p in paths if _reference_under(folder, p, rules)}
        expected_direct = {p for p in paths if _reference_directly_in(folder, p, rules)}

        assert _selected(engine, prefix.under(_paths.c.path), paths) == expected_under
        assert {p for p in paths if prefix.contains(p)} == expected_under
        assert (
            _selected(engine, prefix.directly_in(_paths.c.path), paths)
            == expected_direct
        )

        fragment, params = prefix.under_sql("path")
        with engine.connect() as conn:
            raw = conn.exec_driver_sql(
                f"SELECT path FROM paths WHERE {fragment}", tuple(params)
            ).scalars()
            assert set(raw) == expected_under


def test_covers_is_the_folder_itself_or_below_it():
    prefix = FolderPrefix.of("/lib/photos", **POSIX)
    assert prefix.covers("/lib/photos")
    assert prefix.covers("/lib/photos/2020")
    assert not prefix.covers("/lib/photos-old")
    assert not prefix.covers("/lib/PHOTOS")

    folding = FolderPrefix.of("C:\\Lib", **WINDOWS)
    assert folding.covers("c:\\lib")
    assert folding.covers("C:\\LIB/2020")
    assert not folding.covers("C:\\Lib2")


@pytest.mark.parametrize("rules", [POSIX, WINDOWS], ids=["posix", "windows"])
def test_the_range_is_served_by_an_index_of_the_matching_collation(tmp_path, rules):
    # picture.file_path has no index today; this pins that the predicate is the
    # form one would serve, so adding the index is all it takes. A range under
    # NOCASE needs a NOCASE index, and a BINARY range a BINARY one
    # (https://sqlite.org/datatype3.html#assigning_collating_sequences_from_sql).
    collation = "NOCASE" if rules["fold_case"] else "BINARY"
    db = tmp_path / "plan.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE paths (id INTEGER PRIMARY KEY, path TEXT)")
        conn.execute(f"CREATE INDEX ix_paths_path ON paths (path COLLATE {collation})")
        fragment, params = FolderPrefix.of("/lib", **rules).under_sql("path")
        plan = [
            row[3]
            for row in conn.execute(
                f"EXPLAIN QUERY PLAN SELECT id FROM paths WHERE {fragment}", params
            )
        ]
    finally:
        conn.close()

    # Every arm of the OR is an index range on the path; nothing scans the table.
    searches = [line for line in plan if "ix_paths_path" in line]
    assert len(searches) == len(rules["separators"]), plan
    assert all("(path>? AND path<?)" in line for line in searches), plan
    assert not any(line.startswith("SCAN") for line in plan), plan
