"""Unit tests for the id-scoped stack-leader collapse in :meth:`Picture.find`.

``stack_leaders_only`` renders one tile per stack. On the unscoped grid the
leader is simply ``stack_position == 0``. When the caller passes an explicit id
filter, a picture set, a share token's scope, a split, the global position-0
leader may not be inside that filter, and requiring it rendered NEITHER picture:
a set holding only a non-cover stack member showed five tiles for its six
members and no stack at all (#670 / #1746). The id-scoped branch therefore
represents each stack by its lowest-positioned member INSIDE the id filter.

That rule was first implemented as a correlated ``EXISTS`` over an aliased
picture which re-tested the entire id list once per candidate row. Measured on a
19,822-picture vault, a 6,641-id set view cost 102 ms against 4.6 ms for the
unscoped fast path, and seconds once every row in scope was stacked. Each stack
is now ranked ONCE by ``ROW_NUMBER() OVER (PARTITION BY stack_id)`` in a derived
table, and every candidate row probes that one-row-per-stack result: 9.7 ms for
the same set view, same ids returned.

These tests pin both halves of that change:

* ``_legacy_scoped_leader_ids`` reproduces the correlated formulation verbatim
  and is used as an oracle: the id sets must be identical, not merely the same
  size, across every scenario (stack wholly inside the scope, wholly outside it,
  straddling it, unstacked rows, NULL ``stack_position``, deleted members, and
  all three lifecycle modes: live, ``include_deleted``, ``only_deleted``).
* ``test_id_scoped_leader_count_is_single_pass`` fails if the per-row shape ever
  comes back: it counts an 8,000-id scope that is half stacked, which the old
  formulation cannot do inside the budget.
"""

from __future__ import annotations

import importlib
import itertools
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy import event, func, or_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import aliased
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, exists, insert, select

# Importing the models registers them on SQLModel.metadata.
from pixlstash.db_models import (
    DedupGroup,
    DedupGroupMember,
    Picture,
    PictureProjectMember,
    PictureStack,
    Project,
    ReferenceFolder,
    SortMechanism,
)
from pixlstash.utils.sql_chunking import SQLITE_ID_CHUNK

_file_names = itertools.count()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'stack_leader.db'}",
        echo=False,
        poolclass=NullPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _new_stack(session) -> int:
    stack = PictureStack()
    session.add(stack)
    session.commit()
    session.refresh(stack)
    return stack.id


def _new_stack_at(session, updated_at) -> int:
    stack = PictureStack(created_at=updated_at, updated_at=updated_at)
    session.add(stack)
    session.commit()
    session.refresh(stack)
    return stack.id


def _add_picture(session, **kwargs) -> Picture:
    kwargs.setdefault("file_path", f"pic_{next(_file_names)}.jpg")
    pic = Picture(**kwargs)
    session.add(pic)
    session.commit()
    session.refresh(pic)
    return pic


def test_stack_leaders_sort_by_the_stack_updated_time(session):
    older = _new_stack_at(session, datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    newer = _new_stack_at(session, datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    older_cover = _add_picture(session, stack_id=older, stack_position=0)
    _add_picture(session, stack_id=older, stack_position=1)
    newer_cover = _add_picture(session, stack_id=newer, stack_position=0)
    _add_picture(session, stack_id=newer, stack_position=1)

    recent_first = Picture.find(
        session,
        stack_leaders_only=True,
        stack_state="stacked",
        sort_mech=SortMechanism(SortMechanism.Keys.STACK_UPDATED_AT, descending=True),
    )
    oldest_first = Picture.find(
        session,
        stack_leaders_only=True,
        stack_state="stacked",
        sort_mech=SortMechanism(SortMechanism.Keys.STACK_UPDATED_AT, descending=False),
    )

    assert [pic.id for pic in recent_first] == [newer_cover.id, older_cover.id]
    assert [pic.id for pic in oldest_first] == [older_cover.id, newer_cover.id]


def test_unassigned_stack_leaders_support_recently_changed_sort(session):
    older = _new_stack_at(session, datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    newer = _new_stack_at(session, datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    older_cover = _add_picture(session, stack_id=older, stack_position=0)
    _add_picture(session, stack_id=older, stack_position=1)
    newer_cover = _add_picture(session, stack_id=newer, stack_position=0)
    _add_picture(session, stack_id=newer, stack_position=1)

    found = Picture.find_unassigned(
        session,
        stack_leaders_only=True,
        stack_state="stacked",
        sort_mech=SortMechanism(SortMechanism.Keys.STACK_UPDATED_AT, descending=True),
    )

    assert [pic.id for pic in found] == [newer_cover.id, older_cover.id]


def test_unassigned_takes_an_id_list_past_the_sqlite_variable_limit(session):
    """``find_unassigned`` reads an id list of any length, as ``find`` does.

    A plain ``IN`` binds one variable per id and SQLite builds before 3.32
    refuse more than 999, so past ``SQLITE_ID_CHUNK`` the list is read from a
    temp table. The count covers the whole list, and the page is one window
    over the whole sorted list, not one per chunk.
    """
    session.execute(
        insert(Picture.__table__),
        [
            {"file_path": f"named_{index}.jpg", "deleted": False, "score": index % 6}
            for index in range(SQLITE_ID_CHUNK + 300)
        ],
    )
    session.commit()
    pictures = list(session.exec(select(Picture).order_by(Picture.id)).all())
    named = [pic.id for pic in pictures if pic.id % 7 != 0]
    assert len(named) > SQLITE_ID_CHUNK
    by_score = sorted(
        (pic for pic in pictures if pic.id % 7 != 0),
        key=lambda pic: (pic.score, pic.id),
        reverse=True,
    )

    dbapi = session.connection().connection.dbapi_connection
    previous = dbapi.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        count = Picture.find_unassigned(session, picture_ids=named, count_only=True)
        page = Picture.find_unassigned(
            session,
            picture_ids=named,
            sort_mech=SortMechanism(SortMechanism.Keys.SCORE, descending=True),
            offset=950,
            limit=60,
        )
        # The pin is on the connection both calls ran on: a plain IN fails.
        with pytest.raises(OperationalError, match="too many SQL variables"):
            session.exec(select(Picture.id).where(Picture.id.in_(named))).all()
    finally:
        dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)

    assert count == len(named)
    assert [pic.id for pic in page] == [pic.id for pic in by_score[950:1010]]


def _legacy_scoped_leader_ids(
    session,
    id_scope: list[int],
    only_deleted: bool = False,
    include_deleted: bool = False,
) -> set[int]:
    """The pre-optimization correlated-EXISTS formulation, kept as an oracle.

    This is the exact condition that shipped for #670 / #1746, transcribed
    unchanged. It is intentionally NOT shared with the production code: its only
    job is to answer "what did the slow query return?" so the fast query can be
    proved identical rather than merely plausible.

    ``only_deleted`` / ``include_deleted`` mirror the same ``Picture.find``
    arguments, because they decide whether a candidate row can be OUTSIDE the
    ranking set (which is always live rows only): the one place where "compare
    against the best member" and "no sibling outranks me" could have diverged.
    """
    sibling = aliased(Picture)
    cur_pos = func.coalesce(Picture.stack_position, 999999)
    sib_pos = func.coalesce(sibling.stack_position, 999999)
    has_higher_ranked_sibling = exists(
        select(sibling.id).where(
            sibling.stack_id == Picture.stack_id,
            sibling.deleted.is_(False),
            sibling.id.in_(id_scope),
            or_(
                sib_pos < cur_pos,
                (sib_pos == cur_pos) & (sibling.id < Picture.id),
            ),
        )
    )
    stmt = select(Picture.id).where(Picture.id.in_(id_scope))
    if only_deleted:
        stmt = stmt.where(Picture.deleted.is_(True))
    elif not include_deleted:
        stmt = stmt.where(Picture.deleted.is_(False))
    stmt = stmt.where(or_(Picture.stack_id.is_(None), ~has_higher_ranked_sibling))
    return set(session.exec(stmt).all())


def _found_ids(session, id_scope: list[int], **kwargs) -> set[int]:
    pictures = Picture.find(
        session, stack_leaders_only=True, id=list(id_scope), **kwargs
    )
    return {pic.id for pic in pictures}


def _found_count(session, id_scope: list[int], **kwargs) -> int:
    return Picture.find(
        session,
        stack_leaders_only=True,
        count_only=True,
        id=list(id_scope),
        **kwargs,
    )


def _assert_matches_legacy(session, id_scope, expected, **kwargs):
    """Assert find() returns ``expected``, and that the legacy query agreed."""
    found = _found_ids(session, id_scope, **kwargs)
    assert found == expected
    assert found == _legacy_scoped_leader_ids(
        session,
        id_scope,
        only_deleted=kwargs.get("only_deleted", False),
        include_deleted=kwargs.get("include_deleted", False),
    )
    assert _found_count(session, id_scope, **kwargs) == len(expected)


def test_stack_fully_inside_the_scope_is_represented_by_its_cover(session):
    stack = _new_stack(session)
    cover = _add_picture(session, stack_id=stack, stack_position=0)
    second = _add_picture(session, stack_id=stack, stack_position=1)
    third = _add_picture(session, stack_id=stack, stack_position=2)

    scope = [cover.id, second.id, third.id]
    _assert_matches_legacy(session, scope, {cover.id})


def test_stack_fully_outside_the_scope_contributes_nothing(session):
    stack = _new_stack(session)
    _add_picture(session, stack_id=stack, stack_position=0)
    _add_picture(session, stack_id=stack, stack_position=1)
    loose = _add_picture(session)

    _assert_matches_legacy(session, [loose.id], {loose.id})


def test_straddling_stack_is_represented_by_its_best_in_scope_member(session):
    """The #670 / #1746 case: the cover is outside the id filter.

    Exactly one tile must still stand for the stack, and it must be the
    lowest-positioned member that IS in the filter: not nothing, and not one
    tile per member.
    """
    stack = _new_stack(session)
    cover = _add_picture(session, stack_id=stack, stack_position=0)
    second = _add_picture(session, stack_id=stack, stack_position=1)
    third = _add_picture(session, stack_id=stack, stack_position=2)

    scope = [second.id, third.id]
    _assert_matches_legacy(session, scope, {second.id})
    assert cover.id not in _found_ids(session, scope)


def test_single_non_cover_member_in_scope_still_renders_one_tile(session):
    """A set holding ONE non-cover member: one tile, still wearing its stack."""
    stack = _new_stack(session)
    _add_picture(session, stack_id=stack, stack_position=0)
    member = _add_picture(session, stack_id=stack, stack_position=1)

    found = Picture.find(session, stack_leaders_only=True, id=[member.id])
    assert [pic.id for pic in found] == [member.id]
    assert found[0].stack_id is not None


def test_unstacked_pictures_are_never_collapsed(session):
    loose = [_add_picture(session) for _ in range(4)]
    stack = _new_stack(session)
    cover = _add_picture(session, stack_id=stack, stack_position=0)
    _add_picture(session, stack_id=stack, stack_position=1)

    scope = [pic.id for pic in loose] + [cover.id]
    _assert_matches_legacy(session, scope, {pic.id for pic in loose} | {cover.id})


def test_null_stack_position_sorts_last_and_ties_break_on_id(session):
    """NULL positions rank behind every numbered one; NULL vs NULL takes the
    lowest id. This is the tie-break the correlated form used, and changing it
    would silently change which picture covers a stack."""
    numbered_stack = _new_stack(session)
    unpositioned = _add_picture(session, stack_id=numbered_stack, stack_position=None)
    positioned = _add_picture(session, stack_id=numbered_stack, stack_position=3)

    null_stack = _new_stack(session)
    first_null = _add_picture(session, stack_id=null_stack, stack_position=None)
    second_null = _add_picture(session, stack_id=null_stack, stack_position=None)

    scope = [unpositioned.id, positioned.id, first_null.id, second_null.id]
    # `unpositioned` has the lower id but the NULL position, so the numbered
    # member wins; the all-NULL stack falls back to the lowest id.
    _assert_matches_legacy(session, scope, {positioned.id, first_null.id})


def test_deleted_members_never_represent_a_stack(session):
    """A soft-deleted member is not a candidate cover, so the stack is
    represented by its best LIVE in-scope member."""
    stack = _new_stack(session)
    dead_cover = _add_picture(session, stack_id=stack, stack_position=0, deleted=True)
    live_second = _add_picture(session, stack_id=stack, stack_position=1)
    live_third = _add_picture(session, stack_id=stack, stack_position=2)

    scope = [dead_cover.id, live_second.id, live_third.id]
    _assert_matches_legacy(session, scope, {live_second.id})


def test_trash_view_scope_matches_the_legacy_query(session):
    """``only_deleted`` + an id scope (a scoped trash view) is the one case
    where the ranking set can be empty for a stack: every candidate is deleted
    and the ranking set holds only live rows. Whatever the legacy query did
    there, the single-pass query must still do."""
    stack = _new_stack(session)
    dead_cover = _add_picture(session, stack_id=stack, stack_position=0, deleted=True)
    dead_second = _add_picture(session, stack_id=stack, stack_position=1, deleted=True)
    live_third = _add_picture(session, stack_id=stack, stack_position=2)

    scope = [dead_cover.id, dead_second.id, live_third.id]
    found = _found_ids(session, scope, only_deleted=True)
    assert found == _legacy_scoped_leader_ids(session, scope, only_deleted=True)
    assert found == {dead_cover.id, dead_second.id}


def test_mixed_library_matches_the_legacy_query_for_every_scope(session):
    """Sweep the whole shape space at once and compare id SETS, not counts."""
    loose = [_add_picture(session) for _ in range(3)]
    stack_a = _new_stack(session)
    a0 = _add_picture(session, stack_id=stack_a, stack_position=0)
    a1 = _add_picture(session, stack_id=stack_a, stack_position=1)
    a2 = _add_picture(session, stack_id=stack_a, stack_position=2)
    stack_b = _new_stack(session)
    b0 = _add_picture(session, stack_id=stack_b, stack_position=0)
    b1 = _add_picture(session, stack_id=stack_b, stack_position=None)
    stack_c = _new_stack(session)
    c0 = _add_picture(session, stack_id=stack_c, stack_position=0, deleted=True)
    c1 = _add_picture(session, stack_id=stack_c, stack_position=1)

    everything = [pic.id for pic in loose + [a0, a1, a2, b0, b1, c0, c1]]
    scopes = [
        everything,
        [a1.id, a2.id, b1.id],
        [a2.id],
        [b0.id, b1.id, c1.id],
        [pic.id for pic in loose],
        [c0.id, c1.id],
        [a0.id, b0.id, c0.id],
    ]
    # Every lifecycle mode, because each one changes whether a candidate row can
    # sit outside the (always-live) ranking set.
    modes = [{}, {"include_deleted": True}, {"only_deleted": True}]
    for scope in scopes:
        for mode in modes:
            found = _found_ids(session, scope, **mode)
            assert found == _legacy_scoped_leader_ids(session, scope, **mode), (
                scope,
                mode,
            )
            assert _found_count(session, scope, **mode) == len(found), (scope, mode)


def _seed_bulk(session, picture_count: int, stacked_fraction: float, members: int = 2):
    """Insert ``picture_count`` pictures, ``stacked_fraction`` of them stacked."""
    stacked = int(picture_count * stacked_fraction)
    stack_count = stacked // members
    session.execute(
        insert(PictureStack.__table__),
        [{"name": f"stack-{i}"} for i in range(stack_count)],
    )
    session.commit()
    stack_ids = list(
        session.exec(select(PictureStack.id).order_by(PictureStack.id)).all()
    )

    # Every row must carry the same keys: a bulk insert compiles one statement
    # for the whole list.
    rows = [
        {
            "file_path": f"stacked_{index}.jpg",
            "deleted": False,
            "stack_id": stack_ids[index // members],
            "stack_position": index % members,
        }
        for index in range(stacked)
    ]
    rows += [
        {
            "file_path": f"loose_{index}.jpg",
            "deleted": False,
            "stack_id": None,
            "stack_position": None,
        }
        for index in range(picture_count - stacked)
    ]
    session.execute(insert(Picture.__table__), rows)
    session.commit()
    return list(session.exec(select(Picture.id)).all())


def _timed_count(session, id_scope: list[int], expected: int) -> float:
    """Return how long the id-scoped leader COUNT took, asserting its result."""
    started = time.perf_counter()
    assert _found_count(session, id_scope) == expected
    return time.perf_counter() - started


# Wall-clock budget for the id-scoped COUNT below, measured on this fixture:
# the single-pass form takes 12-21 ms, the per-row correlated form it replaced
# takes 1.2-3.6 s (the spread is machine load; the ratio held at 100-300x). The
# budget sits between them with an order of magnitude of headroom over the fast
# one, so a loaded CI runner does not turn it red, while anything quadratic in
# (id-list size x stacked rows) cannot squeeze under it.
_SINGLE_PASS_BUDGET_S = 0.3


def test_id_scoped_leader_count_is_single_pass(session):
    """Performance regression guard for the id-scoped stack-leader branch.

    ``/pictures/count`` runs this exact shape (COUNT(*), no LIMIT) on every grid
    load of a picture set, a share-token scope, or a split. If the leader rank is
    ever resolved per candidate row again, this blows the budget by orders of
    magnitude rather than by a few percent.
    """
    ids = _seed_bulk(session, picture_count=8000, stacked_fraction=0.5, members=2)
    assert len(ids) == 8000

    # 4,000 loose pictures + one tile for each of the 2,000 two-member stacks.
    # The first call also pays SQLAlchemy's statement compilation, which is not
    # what this measures, so it is the warm-up.
    assert _found_count(session, ids) == 6000

    elapsed = min(_timed_count(session, ids, expected=6000) for _ in range(3))
    assert elapsed < _SINGLE_PASS_BUDGET_S, (
        f"id-scoped stack-leader COUNT took {elapsed:.2f}s for {len(ids)} ids "
        f"(budget {_SINGLE_PASS_BUDGET_S}s): the per-row correlated subquery "
        "has probably come back"
    )


# Every narrowing that can hide a stack's cover is in the leader scope.
#
# The collapse ranks each stack's members inside the leader scope and keeps the
# best-ranked one. A narrowing the outer query applies but the leader scope
# leaves out lets the global cover win that rank and then fail the outer
# filter, so the stack shows no tile at all, in the grid and in its count.

_INSIDE_SOURCE = "/example-import/inside"
_OUTSIDE_SOURCE = "/example-import/outside"
_INSIDE_IMPORTED = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_OUTSIDE_IMPORTED = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_INBOX_CUTOFF = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def narrowed(session):
    """Pictures inside and outside every narrowing at once.

    * ``straddling``: a stack whose cover is outside and whose two other
      members are inside. Its position-1 member must represent it.
    * ``inside``: a stack wholly inside. Its cover must represent it.
    * a loose picture inside, and a loose picture and a stack wholly outside.

    Inside and outside differ in every narrowing: folder, import source and
    time, project, format, resolution, workflow, and the inside pictures are
    one unresolved duplicate group.
    """
    folders = {}
    for name in ("inside", "outside"):
        folder = ReferenceFolder(folder=f"/example-library/{name}", label=name)
        session.add(folder)
        session.commit()
        session.refresh(folder)
        folders[name] = folder.id
    projects = {}
    for name in ("inside", "other", "outside"):
        project = Project(name=f"example-{name}")
        session.add(project)
        session.commit()
        session.refresh(project)
        projects[name] = project.id

    def picture(where: str, **kwargs) -> Picture:
        pic = _add_picture(
            session,
            reference_folder_id=folders[where],
            import_source_folder=(
                _INSIDE_SOURCE if where == "inside" else _OUTSIDE_SOURCE
            ),
            imported_at=(_INSIDE_IMPORTED if where == "inside" else _OUTSIDE_IMPORTED),
            format="PNG" if where == "inside" else "JPEG",
            # 4 MP inside ("4-8mp"), a quarter of one outside ("lt1mp").
            width=2000 if where == "inside" else 500,
            height=2000 if where == "inside" else 500,
            workflow_structural_hash=f"test-wf-{where}",
            comfyui_models=json.dumps([f"test-model-{where}"]),
            **kwargs,
        )
        session.add(PictureProjectMember(picture_id=pic.id, project_id=projects[where]))
        session.commit()
        return pic

    straddling = _new_stack(session)
    straddling_cover = picture("outside", stack_id=straddling, stack_position=0)
    straddling_first = picture("inside", stack_id=straddling, stack_position=1)
    straddling_second = picture("inside", stack_id=straddling, stack_position=2)
    inside = _new_stack(session)
    inside_cover = picture("inside", stack_id=inside, stack_position=0)
    inside_member = picture("inside", stack_id=inside, stack_position=1)
    loose_inside = picture("inside")
    picture("outside")
    outside = _new_stack(session)
    picture("outside", stack_id=outside, stack_position=0)
    picture("outside", stack_id=outside, stack_position=1)
    inside_ids = [
        straddling_first.id,
        straddling_second.id,
        inside_cover.id,
        inside_member.id,
        loose_inside.id,
    ]
    group = DedupGroup(signature="test-group", tier="exact", resolved=False)
    session.add(group)
    session.commit()
    session.refresh(group)
    for position, picture_id in enumerate(inside_ids):
        session.add(
            DedupGroupMember(
                group_id=group.id, picture_id=picture_id, position=position
            )
        )
    session.commit()
    return {
        "folders": folders,
        "projects": projects,
        "straddling_cover": straddling_cover.id,
        "straddling_first": straddling_first.id,
        "inside_cover": inside_cover.id,
        "loose_inside": loose_inside.id,
        "inside_ids": inside_ids,
        "expected": {straddling_first.id, inside_cover.id, loose_inside.id},
    }


# The grid's filters that read the picture's own row, beside the scopes. Scores
# and tags stay out of the leader scope (a stack is one tile for its cover's
# score and tags), and ``stacked`` / ``unstacked`` hold or fail for the whole
# stack, so none of those can hide a cover the rest of its stack matches.
_PICTURE_FILTERS = {
    "format": {"format": ["PNG"]},
    "resolution_bucket": {"resolution_bucket": "4-8mp"},
    "workflow": {"workflow_structural_hashes": ["test-wf-inside"]},
    "unresolved": {"stack_state": "unresolved"},
}


def _find_narrowings(narrowed) -> dict[str, dict]:
    return {
        **_PICTURE_FILTERS,
        "reference_folder_id": {"reference_folder_id": narrowed["folders"]["inside"]},
        # The listing hands its query string to ``find`` unparsed.
        "reference_folder_id as text": {
            "reference_folder_id": str(narrowed["folders"]["inside"])
        },
        "import_source_folder": {"import_source_folder": _INSIDE_SOURCE},
        "several projects": {
            "project_id": [
                narrowed["projects"]["inside"],
                narrowed["projects"]["other"],
            ]
        },
    }


def _unassigned_narrowings(narrowed) -> dict[str, dict]:
    return {
        **_PICTURE_FILTERS,
        "reference_folder_id": {"reference_folder_id": narrowed["folders"]["inside"]},
        "import_source_folder": {"import_source_folder": _INSIDE_SOURCE},
        "imported_since": {"imported_since": _INBOX_CUTOFF},
        "picture_ids": {"picture_ids": narrowed["inside_ids"]},
    }


def _assert_collapsed_to(found, count, expected: set[int]):
    assert {pic.id for pic in found} == expected
    assert count == len(expected)


@pytest.mark.parametrize(
    "narrowing",
    [
        "reference_folder_id",
        "reference_folder_id as text",
        "import_source_folder",
        "several projects",
        *_PICTURE_FILTERS,
    ],
)
def test_find_keeps_a_stack_whose_cover_a_filter_hides(session, narrowed, narrowing):
    search = _find_narrowings(narrowed)[narrowing]

    found = Picture.find(session, stack_leaders_only=True, **search)
    count = Picture.find(session, stack_leaders_only=True, count_only=True, **search)

    _assert_collapsed_to(found, count, narrowed["expected"])


def test_find_keeps_the_stack_of_one_named_non_cover(session, narrowed):
    """A single id is a column filter like a list: it scopes the rank too."""
    member = narrowed["straddling_first"]

    found = Picture.find(session, stack_leaders_only=True, id=member)
    count = Picture.find(session, stack_leaders_only=True, count_only=True, id=member)

    _assert_collapsed_to(found, count, {member})


@pytest.mark.parametrize(
    "narrowing",
    [
        "reference_folder_id",
        "import_source_folder",
        "imported_since",
        "picture_ids",
        *_PICTURE_FILTERS,
    ],
)
def test_unassigned_keeps_a_stack_whose_cover_a_filter_hides(
    session, narrowed, narrowing
):
    search = _unassigned_narrowings(narrowed)[narrowing]

    found = Picture.find_unassigned(session, stack_leaders_only=True, **search)
    count = Picture.find_unassigned(
        session, stack_leaders_only=True, count_only=True, **search
    )

    _assert_collapsed_to(found, count, narrowed["expected"])


def test_unassigned_keeps_the_stack_of_an_id_list_of_one_non_cover(session, narrowed):
    """The share-token scope and ``?id=`` reach ``find_unassigned`` as
    ``picture_ids``: a list holding only a non-cover member still shows it."""
    member = narrowed["straddling_first"]

    found = Picture.find_unassigned(
        session, stack_leaders_only=True, picture_ids=[member]
    )
    count = Picture.find_unassigned(
        session, stack_leaders_only=True, count_only=True, picture_ids=[member]
    )

    _assert_collapsed_to(found, count, {member})


# The bound-parameter cap of SQLite builds before 3.32. Current builds allow
# 32,766, which no grid filter reaches in a test that runs quickly.
VARIABLE_CEILING = 999


@contextmanager
def _variable_ceiling(session):
    """Pin the session's connection to ``VARIABLE_CEILING`` bound parameters.

    The pin is proved before the block runs, so a build that ignored
    ``setlimit`` could not pass vacuously.
    """
    dbapi = session.connection().connection.dbapi_connection
    previous = dbapi.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, VARIABLE_CEILING)
    try:
        with pytest.raises(OperationalError, match="too many SQL variables"):
            session.exec(
                select(Picture.id).where(
                    Picture.id.in_(list(range(VARIABLE_CEILING + 1)))
                )
            ).all()
        yield
    finally:
        dbapi.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)


def _plan(session, run) -> list[str]:
    """The query plan of the last statement *run* sends, one string per step."""
    engine = session.get_bind()
    sent = []

    def capture(_conn, _cursor, statement, parameters, _context, _executemany):
        sent.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        run()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    statement, parameters = sent[-1]
    rows = session.connection().exec_driver_sql(
        "EXPLAIN QUERY PLAN " + statement, parameters
    )
    return [row[3] for row in rows]


@pytest.mark.parametrize("narrowing", list(_PICTURE_FILTERS))
@pytest.mark.parametrize("lister", ["find", "find_unassigned"])
def test_a_grid_filter_in_the_leader_scope_reads_no_whole_table(
    session, lister, narrowing
):
    """The collapse ranks the stacked members a filter keeps without reading
    every picture to find them."""
    search = _PICTURE_FILTERS[narrowing]

    plan = _plan(
        session,
        lambda: getattr(Picture, lister)(session, stack_leaders_only=True, **search),
    )

    # "SCAN picture USING INDEX ..." reads every row too, in index order.
    assert not [step for step in plan if step.split()[:2] == ["SCAN", "picture"]], plan


# The grid's partial indexes, as the migration that last built them defines
# them. ``create_all`` does not build them, so a test that needs one builds it.
_GRID_INDEXES = importlib.import_module(
    "pixlstash.migrations.versions.0052_drop_import_excluded_sentinel"
)._GRID_INDEXES_WITHOUT_IMPORT_EXCLUDED


@pytest.mark.parametrize("lister", ["find", "find_unassigned"])
def test_the_collapsed_grid_can_read_a_partial_leader_index(session, lister):
    """The listing states ``deleted = 0``, the leader indexes' own term.

    ``deleted IS 0`` selects the same rows, but SQLite uses a partial index only
    for a query that repeats the index's WHERE, so the newest-first grid read the
    whole-table grid index and skipped every stacked member by hand.
    """
    for name, columns, where in _GRID_INDEXES:
        if where is not None:
            session.connection().exec_driver_sql(
                f"CREATE INDEX {name} ON picture ({', '.join(columns)}) WHERE {where}"
            )

    plan = _plan(
        session,
        lambda: getattr(Picture, lister)(
            session,
            stack_leaders_only=True,
            sort_mech=SortMechanism(SortMechanism.Keys.DATE, descending=True),
            limit=50,
        ),
    )

    assert any("ix_picture_grid_leaders_created_at" in step for step in plan), plan


def _long_lists(narrowed) -> dict[str, dict]:
    """Each narrowing as a list one value past ``VARIABLE_CEILING``."""
    absent = range(VARIABLE_CEILING)
    return {
        "import sources": {
            "import_source_folder": [
                _INSIDE_SOURCE,
                *(f"/example-import/absent-{index}" for index in absent),
            ]
        },
        "formats": {"format": ["PNG", *(f"TEST{index}" for index in absent)]},
        "projects": {
            "project_id": [
                narrowed["projects"]["inside"],
                *(10**9 + index for index in absent),
            ]
        },
    }


@pytest.mark.parametrize("narrowing", ["import sources", "formats", "projects"])
def test_find_binds_a_long_list_once_under_the_collapse(session, narrowed, narrowing):
    """The collapse applies each narrowing twice, outside and in its leader
    scope, so a list bound one variable per value had to fit the statement
    twice. Bound as one JSON array, it costs one variable in each place."""
    search = _long_lists(narrowed)[narrowing]

    with _variable_ceiling(session):
        found = Picture.find(session, stack_leaders_only=True, **search)
        count = Picture.find(
            session, stack_leaders_only=True, count_only=True, **search
        )

    _assert_collapsed_to(found, count, narrowed["expected"])


def test_unassigned_binds_a_long_format_list_once_under_the_collapse(session, narrowed):
    search = _long_lists(narrowed)["formats"]

    with _variable_ceiling(session):
        found = Picture.find_unassigned(session, stack_leaders_only=True, **search)
        count = Picture.find_unassigned(
            session, stack_leaders_only=True, count_only=True, **search
        )

    _assert_collapsed_to(found, count, narrowed["expected"])


@pytest.mark.parametrize(
    "narrowing, probe",
    [
        (
            "import sources",
            "SEARCH picture USING INDEX ix_picture_import_source_folder"
            " (import_source_folder=?)",
        ),
        (
            "projects",
            "SEARCH pictureprojectmember USING COVERING INDEX"
            " sqlite_autoindex_pictureprojectmember_1 (picture_id=? AND project_id=?)",
        ),
    ],
)
def test_a_long_list_still_probes_its_index(session, narrowed, narrowing, probe):
    """A JSON array is read through the column's index, as the values were.

    ``include_deleted`` leaves the list the only narrowing, so the plan shows
    how the list itself is read and not which of two indexes SQLite preferred.
    """
    search = _long_lists(narrowed)[narrowing]

    plan = _plan(session, lambda: Picture.find(session, include_deleted=True, **search))

    assert probe in plan, plan


@pytest.mark.parametrize("lister", ["find", "find_unassigned"])
def test_a_long_comfyui_model_list_binds_once_under_the_collapse(
    session, narrowed, lister
):
    """A stack shows when any member used one of the models, so the collapse
    tests the list against the picture and against its stack: two places, one
    JSON array each. Its tile stays the stack's cover."""
    names = [
        "test-model-inside",
        *(f"test-model-absent-{index}" for index in range(VARIABLE_CEILING)),
    ]
    find = getattr(Picture, lister)

    with _variable_ceiling(session):
        found = find(session, stack_leaders_only=True, comfyui_models_filter=names)
        count = find(
            session,
            stack_leaders_only=True,
            count_only=True,
            comfyui_models_filter=names,
        )

    _assert_collapsed_to(
        found,
        count,
        {
            narrowed["straddling_cover"],
            narrowed["inside_cover"],
            narrowed["loose_inside"],
        },
    )
