"""``Picture.find`` and ``Picture.fetch_features`` take an id list of any length.

An ``IN`` list binds one parameter per id, and SQLite refuses a statement past
``SQLITE_LIMIT_VARIABLE_NUMBER``: 999 before 3.32, 32,766 or more after. A list
longer than ``SQLITE_ID_CHUNK`` is therefore read from a temp table
(``scope_id_subquery``) inside the same statement, so the sort, offset and limit
still apply over the whole list rather than per chunk.

Every connection here is pinned to the historical 999, and
``test_the_ceiling_is_pinned`` proves the pin took, so a long list that went
back to a plain ``IN`` fails with "too many SQL variables" on any build.
"""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import event, insert, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, select

# Importing the models registers them on SQLModel.metadata.
from pixlstash.db_models import Face, Picture, PictureStack, SortMechanism
from pixlstash.utils.sql_chunking import SQLITE_ID_CHUNK

VARIABLE_CEILING = 999
PICTURE_COUNT = 1_400
STACK_COUNT = 100
STACK_MEMBERS = 3
# Ids no picture has: they pad a list without matching anything.
ABSENT_IDS = list(range(10**9, 10**9 + 50))


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("find_id_lists") / "find_id_lists.db"
    engine = create_engine(f"sqlite:///{path}", echo=False, poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection, _record):
        dbapi_connection.setlimit(
            sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, VARIABLE_CEILING
        )
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(
            insert(PictureStack.__table__),
            [{"name": f"stack-{i}"} for i in range(STACK_COUNT)],
        )
        session.commit()
        stack_ids = list(
            session.exec(select(PictureStack.id).order_by(PictureStack.id)).all()
        )
        stacked = STACK_COUNT * STACK_MEMBERS
        rows = [
            {
                "file_path": f"pic_{index}.jpg",
                "deleted": False,
                "score": index % 6,
                "stack_id": stack_ids[index // STACK_MEMBERS]
                if index < stacked
                else None,
                "stack_position": index % STACK_MEMBERS if index < stacked else None,
            }
            for index in range(PICTURE_COUNT)
        ]
        session.execute(insert(Picture.__table__), rows)
        session.commit()
        ids = list(session.exec(select(Picture.id).order_by(Picture.id)).all())
        session.execute(
            insert(Face.__table__),
            [{"picture_id": pid, "face_index": 0} for pid in ids],
        )
        session.commit()
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _all_pictures(session) -> list[Picture]:
    return list(session.exec(select(Picture).order_by(Picture.id)).all())


def _temp_tables(session) -> set[str]:
    return set(
        session.execute(
            text("SELECT name FROM sqlite_temp_master WHERE type = 'table'")
        ).scalars()
    )


def test_the_ceiling_is_pinned(session):
    """A plain IN over more ids than the pin allows fails, so every test below
    exercises the long-list path rather than passing on a roomy build."""
    ids = [pic.id for pic in _all_pictures(session)]
    assert len(ids) > VARIABLE_CEILING
    with pytest.raises(OperationalError, match="too many SQL variables"):
        session.exec(select(Picture.id).where(Picture.id.in_(ids))).all()


def test_find_reads_every_named_picture_past_the_ceiling(session):
    pictures = _all_pictures(session)
    ids = [pic.id for pic in pictures]
    # Digit strings match as the IN matches them; junk and absent ids match
    # nothing, as they never did.
    named = [str(pid) for pid in ids[:200]] + ids[200:] + ABSENT_IDS + ["junk"]

    found = Picture.find(session, id=named)

    assert sorted(pic.id for pic in found) == ids


def test_count_only_counts_the_whole_long_list(session):
    ids = [pic.id for pic in _all_pictures(session)]

    assert Picture.find(session, id=ids + ABSENT_IDS, count_only=True) == len(ids)


def test_sort_offset_and_limit_hold_over_the_whole_long_list(session):
    """One window over the whole list, not one per chunk: the page is the
    slice of the full sorted list."""
    pictures = _all_pictures(session)
    named = [pic.id for pic in pictures if pic.id % 7 != 0]
    assert len(named) > SQLITE_ID_CHUNK
    by_score = sorted(
        (pic for pic in pictures if pic.id in set(named)),
        key=lambda pic: (pic.score, pic.id),
        reverse=True,
    )
    expected = [pic.id for pic in by_score[950:1150]]

    found = Picture.find(
        session,
        id=named,
        sort_mech=SortMechanism(SortMechanism.Keys.SCORE, descending=True),
        offset=950,
        limit=200,
    )

    assert [pic.id for pic in found] == expected


@pytest.mark.parametrize(
    ("length", "reads_a_temp_table"),
    [(SQLITE_ID_CHUNK, False), (SQLITE_ID_CHUNK + 1, True)],
)
def test_the_boundary_is_sqlite_id_chunk(session, length, reads_a_temp_table):
    """At ``SQLITE_ID_CHUNK`` ids the list is still bound as an IN (well under
    the 999 floor); one more moves it to the temp table. Both answer alike."""
    ids = [pic.id for pic in _all_pictures(session)][:length]

    found = Picture.find(session, id=ids)

    assert sorted(pic.id for pic in found) == ids
    assert ("_pixlstash_find_id_ids" in _temp_tables(session)) is reads_a_temp_table


def test_two_long_lists_on_different_columns_keep_their_own_tables(session):
    pictures = _all_pictures(session)
    ids = [pic.id for pic in pictures]
    stack_ids = sorted({pic.stack_id for pic in pictures if pic.stack_id})
    # stack_id needs a long list of its own: pad it past the chunk with ids no
    # stack has.
    stacks_named = stack_ids + list(range(10**9, 10**9 + SQLITE_ID_CHUNK))

    found = Picture.find(session, id=ids, stack_id=stacks_named)

    assert sorted(pic.id for pic in found) == [
        pic.id for pic in pictures if pic.stack_id is not None
    ]
    assert {"_pixlstash_find_id_ids", "_pixlstash_find_stack_id_ids"} <= (
        _temp_tables(session)
    )


def _expected_leaders(pictures, scope: set[int]) -> list[int]:
    """Each loose picture in scope, and each stack's lowest-placed member in
    scope (#670: the global cover may be outside the scope)."""
    leaders = []
    best: dict[int, Picture] = {}
    for pic in pictures:
        if pic.id not in scope:
            continue
        if pic.stack_id is None:
            leaders.append(pic.id)
        elif (
            pic.stack_id not in best
            or pic.stack_position < best[pic.stack_id].stack_position
        ):
            best[pic.stack_id] = pic
    return sorted(leaders + [pic.id for pic in best.values()])


def test_stack_leaders_over_a_long_list_rank_within_the_list(session):
    """Leaving each stack's cover (position 0) out of the list must still show
    the stack, through its next member in the list."""
    pictures = _all_pictures(session)
    scope = {pic.id for pic in pictures if pic.stack_position != 0}
    assert len(scope) > SQLITE_ID_CHUNK
    expected = _expected_leaders(pictures, scope)

    found = Picture.find(session, id=sorted(scope), stack_leaders_only=True)
    count = Picture.find(
        session, id=sorted(scope), stack_leaders_only=True, count_only=True
    )

    assert sorted(pic.id for pic in found) == expected
    assert count == len(expected)


def test_stack_leaders_rank_within_a_select_id_filter(session):
    """A SELECT id filter scopes the leader rank as a list does."""
    pictures = _all_pictures(session)
    scope = {pic.id for pic in pictures if pic.stack_position != 0}
    expected = _expected_leaders(pictures, scope)

    found = Picture.find(
        session,
        id=select(Picture.id).where(
            (Picture.stack_position.is_(None)) | (Picture.stack_position != 0)
        ),
        stack_leaders_only=True,
    )

    assert sorted(pic.id for pic in found) == expected


@pytest.mark.parametrize("length", [SQLITE_ID_CHUNK, PICTURE_COUNT])
def test_fetch_features_reads_a_list_of_any_length(session, length):
    ids = [pic.id for pic in _all_pictures(session)][:length]

    faces_by_pic, *_ = Picture.fetch_features(session, ids + ABSENT_IDS)

    assert sorted(faces_by_pic) == ids
    assert all(len(faces) == 1 for faces in faces_by_pic.values())


@pytest.mark.parametrize("count_only", [False, True])
@pytest.mark.parametrize(
    ("length", "reads_a_temp_table"),
    [
        (SQLITE_ID_CHUNK // 2, False),
        (SQLITE_ID_CHUNK // 2 + 1, True),
        (498, True),
        (SQLITE_ID_CHUNK, True),
    ],
)
def test_stack_leaders_over_a_list_that_fits_once_but_not_twice(
    session, length, reads_a_temp_table, count_only
):
    """The stack-leader collapse binds the id list a second time, in the
    subquery that ranks each stack's members, so a list of ``SQLITE_ID_CHUNK``
    or fewer can still pass the ceiling: 498 ids bound twice plus the
    statement's other values are 1,001. Under ``stack_leaders_only`` a list
    past half the chunk is read from the temp table, which both reads share.
    """
    pictures = _all_pictures(session)
    # The non-covers first, so the rank inside the list is what is tested.
    scope = sorted(pic.id for pic in pictures if pic.stack_position != 0)[:length]
    assert len(scope) == length
    expected = _expected_leaders(pictures, set(scope))

    result = Picture.find(
        session, id=scope, stack_leaders_only=True, count_only=count_only
    )

    if count_only:
        assert result == len(expected)
    else:
        assert sorted(pic.id for pic in result) == expected
    assert ("_pixlstash_find_id_ids" in _temp_tables(session)) is reads_a_temp_table


@pytest.mark.parametrize("count_only", [False, True])
@pytest.mark.parametrize(
    ("length", "reads_a_temp_table"),
    [
        (SQLITE_ID_CHUNK // 2, False),
        (SQLITE_ID_CHUNK // 2 + 1, True),
        (498, True),
        (SQLITE_ID_CHUNK, True),
    ],
)
def test_unassigned_stack_leaders_over_a_list_that_fits_once_but_not_twice(
    session, length, reads_a_temp_table, count_only
):
    """``find_unassigned`` binds ``picture_ids`` twice under the collapse too:
    once to narrow the rows and once to rank each stack inside the list."""
    pictures = _all_pictures(session)
    scope = sorted(pic.id for pic in pictures if pic.stack_position != 0)[:length]
    assert len(scope) == length
    expected = _expected_leaders(pictures, set(scope))

    result = Picture.find_unassigned(
        session, picture_ids=scope, stack_leaders_only=True, count_only=count_only
    )

    if count_only:
        assert result == len(expected)
    else:
        assert sorted(pic.id for pic in result) == expected
    assert ("_pixlstash_unassigned_ids" in _temp_tables(session)) is (
        reads_a_temp_table
    )
