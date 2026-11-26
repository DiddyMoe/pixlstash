"""Declarative, serialisable predicate over :class:`Picture` rows.

PixlStash selects ``Picture`` rows by the same vocabulary (score range, smart-score
bucket, resolution bucket, ComfyUI model/LoRA membership, tag include/exclude, hidden
tags, tag-confidence thresholds, format, deleted/imported flags, face presence, path
prefix, …) in several independent places.  Historically the WHERE-clause logic - down
to byte-for-byte copies of raw ``text()`` ``EXISTS`` / ``json_each`` snippets - was
duplicated across all of them.

``PredicateFilter`` is the single source of truth for that logic.  It has three
consumers:

* **compile** - :meth:`predicates` / :meth:`apply` turn the declarative fields into
  SQLAlchemy WHERE clauses (the raw ``text()`` snippets are moved here verbatim, bound
  params intact).
* **match** - :meth:`matches` narrows the same predicate set to a single picture id.
  This is exactly the set predicate restricted to one row; there is no separate
  Python-side re-implementation of the SQL semantics.  Staging auto-triage consumes
  this.
* **parse** - :meth:`from_query_params` builds the filter from request query params
  (one parser, replacing the per-route copies).

The name is ``PredicateFilter`` (not ``Filter``) because "filter" already means an
image-manipulation plugin in this codebase (see ``pixlstash/image_plugins/``).

Membership filters (``set``/``character``/``project``) are intentionally *out of
scope*: they resolve to candidate-id sets and stay in ``filter_helpers.py``.  Compose
with a pre-resolved set via the caller's own ``Picture.id.in_(...)`` clause.
"""

import json
import math
from functools import cache
from datetime import datetime, timedelta, timezone
from typing import Collection, List, Optional, Sequence

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import and_, case, false, not_, or_, text, true
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import select

from pixlstash.db_models.dedup import DedupGroup, DedupGroupMember
from pixlstash.db_models.picture import Picture
from pixlstash.utils.query.folder_prefix import FolderPrefix
from pixlstash.db_models.tag import (
    TAG_PENDING_SENTINEL,
    TAG_SENTINEL_ESCAPE_CHAR,
    TAG_SENTINEL_LIKE_PATTERN,
    Tag,
)
from pixlstash.services.set_lock_service import locked_picture_id_subquery
from pixlstash.utils.service.filter_helpers import is_truthy_flag  # noqa: F401
from pixlstash.utils.sql_chunking import in_json, not_in_json
from pixlstash.utils.sqlite_ints import parse_sqlite_int
from pixlstash.utils.workflow_ids import AUTO_PREFIX, MANUAL_PREFIX
from pixlstash.utils.service.person_tags import (
    FACE_REQUIRING_TAGS,
    OBJECT_META_TAGS,
    PERSON_TAGS,
)

_WORKFLOW_ID_PREFIXES = (AUTO_PREFIX, MANUAL_PREFIX)

# The Inbox's periods (``?inbox=``): how far back "recently added" reaches.
# ``all`` is the Inbox with no cutoff, every picture with no person and no set.
INBOX_PERIODS: dict[str, Optional[timedelta]] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "14d": timedelta(days=14),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "all": None,
}


def inbox_cutoff(period: Optional[str]) -> Optional[datetime]:
    """The earliest ``imported_at`` an Inbox ``period`` admits.

    Args:
        period: A key of :data:`INBOX_PERIODS`, or ``None`` for no Inbox.

    Returns:
        The cutoff in UTC, or ``None`` for no Inbox and for ``all``.

    Raises:
        HTTPException: 422 for a period that is not one of the keys, rather
            than an Inbox silently widened to every picture.
    """
    if period is None:
        return None
    if period not in INBOX_PERIODS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid inbox: must be one of {', '.join(INBOX_PERIODS)}",
        )
    span = INBOX_PERIODS[period]
    return None if span is None else datetime.now(timezone.utc) - span


# In a workflow filter's key list, a workflow whose runs file on it and whose
# pictures are therefore NOT a plain variant's (``!manual:<uuid>``,
# ``!auto:<digest>``). See :func:`workflow_keys_predicate`.
NOT_MADE_BY = "!"
# The workflow a held variant (:data:`RUN_VARIANT`) must have been run by
# (``=manual:<uuid>``): one run-filed workflow's own runs, narrowed by a LoRA.
ONLY_MADE_BY = "="
# A variant counted only in a picture an :data:`ONLY_MADE_BY` workflow's run
# made (``@<structural hash>``), where a plain variant is counted in every
# picture no run-filed workflow made.
RUN_VARIANT = "@"


def made_by_run_filed(live: Sequence[str]) -> ColumnElement:
    """``picture.run_workflow_id`` names one of the *live* run-filed workflows.

    *live* is ``workflow_card_reads.run_filed_workflow_ids``: the manual
    workflows the hub holds and the automatic ones the owner saved over. A
    picture naming a manual workflow that has since been deleted is in none.
    """
    return in_json(Picture.run_workflow_id, live)


def filed_as(live: Sequence[str]) -> ColumnElement:
    """The key a kept picture is filed under: the workflow whose run made it, else its variant.

    A picture made by a run of a workflow whose runs file on it
    (``run_workflow_id`` in *live*: a manual workflow, or an automatic one
    the owner saved over) belongs to THAT workflow while it lives, not to the
    automatic one its variant is in. Every per-workflow read of pictures (the
    grid's counts and covers, a workflow's pictures, its recipes' credit and
    looks) groups or narrows by this, so such a workflow's id is one more key
    beside the variants (``workflow_card_reads.picture_keys``).
    """
    if not live:
        return Picture.workflow_structural_hash
    return case(
        (made_by_run_filed(live), Picture.run_workflow_id),
        else_=Picture.workflow_structural_hash,
    )


def workflow_keys_predicate(keys: Sequence[str]) -> ColumnElement:
    """The pictures a resolved workflow filter names.

    *keys* is what ``routes/pictures/_listing._resolve_workflow_filter``
    resolves ``?workflow=`` to, for one workflow or several (OR'd):

    * a workflow id (``manual:``, ``auto:``): the pictures its own runs made,
      by ``run_workflow_id``. A manual workflow's only key; a saved-over
      automatic one's beside its variants;
    * a variant: its pictures, less those a :data:`NOT_MADE_BY` workflow's
      run made. A run-filed workflow's picture is that workflow's and never
      the one its variant is in, while it lives;
    * a :data:`RUN_VARIANT`: that variant's pictures made by a run of an
      :data:`ONLY_MADE_BY` workflow (its runs, narrowed by a LoRA).

    Empty matches nothing.
    """
    own = [key for key in keys if key.startswith(_WORKFLOW_ID_PREFIXES)]
    excluded = [key[1:] for key in keys if key.startswith(NOT_MADE_BY)]
    only = [key[1:] for key in keys if key.startswith(ONLY_MADE_BY)]
    held = [key[1:] for key in keys if key.startswith(RUN_VARIANT)]
    variants = [
        key
        for key in keys
        if key not in own and key[:1] not in (NOT_MADE_BY, ONLY_MADE_BY, RUN_VARIANT)
    ]
    found = in_json(Picture.workflow_structural_hash, variants)
    if excluded:
        found = and_(
            found,
            or_(Picture.run_workflow_id.is_(None), not_(made_by_run_filed(excluded))),
        )
    if held:
        found = or_(
            found,
            and_(
                in_json(Picture.workflow_structural_hash, held),
                in_json(Picture.run_workflow_id, only),
            ),
        )
    return or_(in_json(Picture.run_workflow_id, own), found) if own else found


@cache
def _unresolved_pictures():
    """The pictures in a duplicate group still awaiting a ruling, as one CTE.

    A CTE so the statement carries it once, at the top: the stack-leader collapse
    compiles this filter a second time inside its ranked subquery, and nested
    there in full (``live_groups_filter`` is itself four subqueries deep) the
    statement reached SQLite's parser-stack limit, so one more level of nesting
    from any caller would fail. One object for every statement, because two
    CTEs of one name in one statement are an error.

    ``live_groups_filter`` is imported lazily for the same reason
    ``Picture.find`` imports this module lazily: ``db_models.picture`` sits
    between the two.
    """
    from pixlstash.services.dedup_tier_service import live_groups_filter

    return (
        select(DedupGroupMember.picture_id)
        .join(DedupGroup, DedupGroup.id == DedupGroupMember.group_id)
        .where(DedupGroup.resolved.is_(False), live_groups_filter())
        .cte("unresolved_pictures")
    )


# Tag vocabularies for the live "Impossible tags" grid filters, lowercased once for
# case-insensitive SQL membership.
_FACE_REQUIRING_LOWER = sorted(FACE_REQUIRING_TAGS)
_PERSON_LOWER = sorted(PERSON_TAGS)
_OBJECT_META_LOWER = sorted(OBJECT_META_TAGS)

# "This picture holds a face the extractor actually found", i.e. not the
# ``face_index == -1`` sentinel it writes when it found none. One spelling,
# because the stack-leader collapse and the row filter must agree exactly.
_HAS_REAL_FACE_SQL = (
    "EXISTS (SELECT 1 FROM face WHERE face.picture_id = picture.id "
    "AND face.face_index != -1)"
)

# The shared no-real-face test (only sentinel faces, ``face_index == -1``).
_NO_REAL_FACE_SQL = (
    "NOT EXISTS (SELECT 1 FROM face WHERE face.picture_id = picture.id "
    "AND face.face_index != -1)"
)


# Why an untagged picture has no tags (``?tag_state=``), for the insights
# panel's untagged legend and the grid it opens. "Untagged" is no tag but the
# pending sentinels (``__tag``/``__tag:<engine>``), the same test the stats count
# uses. The three states partition the untagged pictures:
#
# * ``unprocessable``: the tagger could not read the file and the in-memory
#   unprocessable registry suppresses it, so its sentinel stays and nothing
#   retries it until the server restarts (the registry is not persisted).
# * ``pending``: the tagger will take it up, by the one definition the tag
#   finder and the tagger's remaining count also apply
#   (:func:`awaiting_tagger_clauses`), so this row is what the finder hands out.
# * ``none``: not suppressed and not waiting for the tagger. Usually the tagger
#   ran and found nothing to write, or every tag was removed afterwards (by
#   hand, or by the removed-from-tagging cleanup); nothing re-queues those, a
#   manual Re-tag does. It also holds a picture whose sentinel the tagger will
#   not act on where it is now: in the Scrapheap, frozen by a locked set, or
#   with no file. It keeps the sentinel, so a restore or an unlock makes it
#   ``pending`` again.
TAG_STATES = ("pending", "unprocessable", "none")


def parse_tag_state(raw: Optional[str]) -> Optional[str]:
    """The ``tag_state`` query param, validated. ``None`` or empty is no filter.

    Raises:
        HTTPException: 422 for a value outside :data:`TAG_STATES`. Silently
            ignoring it would show every picture under a filter the request
            says is applied.
    """
    if raw is None or raw == "":
        return None
    if raw not in TAG_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tag_state: must be one of {', '.join(TAG_STATES)}",
        )
    return raw


def _is_sentinel_tag() -> ColumnElement:
    return Tag.tag.like(TAG_SENTINEL_LIKE_PATTERN, escape=TAG_SENTINEL_ESCAPE_CHAR)


def untagged_predicate() -> ColumnElement:
    """The picture has no tag but the pending sentinels."""
    return ~Picture.tags.any(and_(~_is_sentinel_tag(), Tag.tag.is_not(None)))


def tag_state_predicates(
    state: str, unprocessable_ids: Collection[int]
) -> list[ColumnElement]:
    """The clauses for one of :data:`TAG_STATES` (see there for the meaning).

    Args:
        state: One of :data:`TAG_STATES`.
        unprocessable_ids: The registry's suppressed picture ids
            (``unprocessable_images.active_suppressed_ids()``), bound as one
            JSON array so the size of the set is not a parameter-count risk.

    Raises:
        ValueError: *state* is not one of :data:`TAG_STATES`.
    """
    suppressed = ~not_in_json(Picture.id, unprocessable_ids)
    if state == "unprocessable":
        return [untagged_predicate(), suppressed]
    awaiting = awaiting_tagger_clauses(unprocessable_ids)
    if state == "pending":
        return [untagged_predicate(), *awaiting]
    if state == "none":
        # The complement of ``pending`` among the untagged, unsuppressed
        # pictures, so the three states still add up to the untagged count.
        return [untagged_predicate(), ~suppressed, not_(and_(*awaiting))]
    raise ValueError(f"Unknown tag_state {state!r}; expected one of {TAG_STATES}")


# Every sentinel value starts with TAG_PENDING_SENTINEL (``__tag`` or
# ``__tag:<engine>``), so the half-open range [``__tag``, ``__tah``) is the
# sentinel set. Written as a range rather than the LIKE pattern because SQLite
# cannot serve LIKE ... ESCAPE from an index: this form walks ``ix_tag_tag``.
_TAG_SENTINEL_RANGE_END = TAG_PENDING_SENTINEL[:-1] + chr(
    ord(TAG_PENDING_SENTINEL[-1]) + 1
)


def awaiting_tagger_clauses(
    suppressed_ids: Collection[int],
    *,
    sentinel_terms: Sequence[ColumnElement] = (),
) -> tuple[ColumnElement, ...]:
    """The pictures the tagger will take up: one definition for every reader.

    A pending-retag sentinel on a picture tagging may touch (not scrapheaped,
    has a file), not frozen by a locked set, and not suppressed by the
    unprocessable registry. Three readers apply exactly these clauses, so
    their numbers cannot drift apart:

    * ``MissingTagFinder._fetch_missing_tags``: the pictures handed out. It
      adds two waits that end by themselves: a picture's faces must be known,
      and a plugin on a backend hold does not select (``sentinel_terms``).
    * ``TagTask.count_missing_tags``: the tagger's remaining count, which
      also counts a picture re-tagged with real tags kept
      (``retag_face_crops`` keeps every tag the crops did not write).
    * ``tag_state_predicates("pending")``: the "Waiting for the tagger" row
      and the grid it opens. They split the *untagged* pictures only, so such
      a picture is in the count above and not in this row.

    Args:
        suppressed_ids: ``unprocessable_images.active_suppressed_ids()``,
            bound as one JSON array.
        sentinel_terms: Extra conditions on the sentinel ``Tag`` row, applied
            inside the subquery (the finder's held-plugin narrowing).

    Returns:
        WHERE terms on ``Picture``. The sentinel test is one uncorrelated
        ``picture.id IN (SELECT tag.picture_id ...)`` over the sentinel range,
        which ``ix_tag_tag`` serves.
    """
    # Local import: initialising ``pixlstash.tasks`` loads this module part-way
    # (tasks.description_task -> services.scrapheap_service -> ... ->
    # ``pixlstash.utils.query``, whose package init imports this module), so a
    # top-level import of tag_task would be circular.
    from pixlstash.tasks.tag_task import taggable_picture_clauses

    sentinel_picture_ids = select(Tag.picture_id).where(
        Tag.tag >= TAG_PENDING_SENTINEL,
        Tag.tag < _TAG_SENTINEL_RANGE_END,
        *sentinel_terms,
    )
    return (
        Picture.id.in_(sentinel_picture_ids),
        *taggable_picture_clauses(),
        ~Picture.id.in_(locked_picture_id_subquery()),
        not_in_json(Picture.id, suppressed_ids),
    )


# Tag-group terms travel inside the ordinary ``tag`` / ``rejected_tag`` params:
# ``__group:<id>`` is "any tag filed in that group", ``__group:none`` is "any tag
# filed nowhere". Anything else stays a literal tag name, as it always was.
GROUP_TERM_PREFIX = "__group:"
GROUP_TERM_NONE = "__group:none"
# SQLite's INTEGER is 64-bit; an id beyond it cannot name a group, so it binds to
# a value no group can have (ids start at 1) rather than overflowing the driver.
_SQLITE_MAX_INT = 2**63 - 1

# SQLite's smallest cap on the variables one statement binds (older builds;
# current ones allow 32766).
_SQLITE_VARIABLE_FLOOR = 999

#: Most values one request may send in each of the ``tag``, ``rejected_tag``,
#: ``tag_confidence_above``, ``tag_confidence_below``, ``comfyui_model`` and
#: ``comfyui_lora`` lists. Each value binds its own variables, a confidence
#: entry two (its tag and its threshold), so the six lists full bind at most
#: eight per unit of the cap. That keeps them inside half the floor and leaves
#: the other half for what else the statement binds. Each value is also one
#: more ``AND`` in the statement, and SQLite refuses an expression tree deeper
#: than 1000 by default: about 160 values a list already crosses that.
MAX_FILTER_TERMS = _SQLITE_VARIABLE_FLOOR // 2 // 8


def _group_term_id(term: str) -> Optional[int]:
    """The group id a ``__group:<digits>`` term names, else ``None``."""
    if not term.startswith(GROUP_TERM_PREFIX):
        return None
    digits = term[len(GROUP_TERM_PREFIX) :]
    if not digits or not digits.isascii() or not digits.isdigit():
        return None
    # Leading zeros name the same id; past 19 significant digits no 64-bit id
    # fits, and int() of a few thousand digits raises on Python 3.11+.
    significant = digits.lstrip("0") or "0"
    if len(significant) > len(str(_SQLITE_MAX_INT)):
        return -1
    value = int(significant)
    return value if value <= _SQLITE_MAX_INT else -1


def _group_term_clause(
    term: str, index: int, negate: bool
) -> Optional[tuple[str, dict]]:
    """SQL for one group term in a has (``negate=False``) or lacks list.

    Returns ``None`` when *term* is a literal tag, so the caller keeps its own
    exact-name clause. ``has`` means "some tag of the picture matches"; ``lacks``
    is the negation of that same EXISTS, so a picture lacks a group only when it
    carries none of the group's tags.
    """
    # The has and lacks lists are numbered from 0 each, and their clauses land
    # in one statement, so the list is part of every bind name.
    side = "lacks" if negate else "has"
    group_id = _group_term_id(term)
    if group_id is not None:
        name = f"tg_group_{side}_{index}"
        body = (
            "EXISTS (SELECT 1 FROM tag JOIN tag_group_member m ON m.tag = tag.tag "
            f"WHERE tag.picture_id = picture.id AND m.group_id = :{name})"
        )
        params: dict = {name: group_id}
    elif term == GROUP_TERM_NONE:
        prefix = f"tg_unfiled_{side}_{index}"
        # A tag is unfiled when no tag_group_member row names it. Pending-retag
        # sentinels are machine bookkeeping, not user tags, so they never count.
        body = (
            "EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND "
            f"tag.tag NOT LIKE :{prefix}_pattern ESCAPE '{TAG_SENTINEL_ESCAPE_CHAR}' "
            "AND NOT EXISTS (SELECT 1 FROM tag_group_member m WHERE m.tag = tag.tag)"
            ")"
        )
        params = {f"{prefix}_pattern": TAG_SENTINEL_LIKE_PATTERN}
    else:
        return None
    return (f"NOT {body}" if negate else body), params


def _tag_in_exists(prefix: str, tags_lower: list[str]) -> tuple[str, dict]:
    """Return an ``EXISTS (a tag in tags_lower)`` SQL fragment + its bind params.

    ``prefix`` namespaces the bound params so several fragments can coexist in one
    statement without collision.
    """
    placeholders = ", ".join(f":{prefix}_{i}" for i in range(len(tags_lower)))
    sql = (
        f"EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id "
        f"AND LOWER(tag.tag) IN ({placeholders}))"
    )
    return sql, {f"{prefix}_{i}": t for i, t in enumerate(tags_lower)}


def comfyui_leaf_parts(
    models_filter: Optional[List[str]],
    loras_filter: Optional[List[str]],
) -> tuple[list[str], list[str], dict]:
    """Build the leaf ``json_each EXISTS`` SQL fragments for ComfyUI membership.

    Two parallel fragment lists are returned plus a shared bind-param dict:

    * ``self_parts``   – fragments testing the picture row itself.
    * ``member_parts`` – equivalent fragments testing an aliased stack-member row
      (``_m``), used by :meth:`Picture.find` for stack-leader expansion.

    Both lists use the same bind-param names so the dict applies to either.  This is
    the single definition of the ComfyUI leaf snippet; callers decide how to combine
    the fragments (``AND`` of separate clauses vs. ``OR`` within one clause, with or
    without stack expansion).

    Each list is one fragment, "records any of these names", with the names bound
    as ONE JSON array (``:cmf`` / ``:clf``). The stack-expanded form repeats the
    fragment for the member row, so one parameter per name would bind every name
    twice and a long pick would pass SQLite's variable limit.
    """
    self_parts: list[str] = []
    member_parts: list[str] = []
    bind_params: dict = {}
    for names, column, param in (
        (models_filter, "comfyui_models", "cmf"),
        (loras_filter, "comfyui_loras", "clf"),
    ):
        if not names:
            continue
        for row, parts in (("picture", self_parts), ("_m", member_parts)):
            parts.append(
                f"EXISTS (SELECT 1 FROM json_each({row}.{column})"
                f" WHERE value IN (SELECT value FROM json_each(:{param})))"
            )
        bind_params[param] = json.dumps(sorted(set(names)))
    return self_parts, member_parts, bind_params


class PredicateFilter(BaseModel):
    """A declarative, serialisable predicate over ``Picture`` rows.

    All fields are optional / defaulted.  The defaults describe the predicate used by
    a "give me the matching live pictures" query: non-deleted, no other restrictions
    - which is exactly what single-picture :meth:`matches` wants for auto-triage.

    Each builder site sets the subset of fields it needs and toggles the
    flag fields (``include_deleted`` / ``only_deleted`` / ``apply_deleted_filter`` /
    ``include_unimported``) so that the centralised compiler reproduces that site's
    emitted predicate exactly.
    """

    # --- intrinsic attribute predicates ---
    format: Optional[List[str]] = None
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    # "Never rated by the user". Both states are reachable in production: a
    # picture that was never touched keeps ``score IS NULL``, while clicking the
    # current star again writes a literal 0 (``POST /pictures/apply-scores``
    # accepts 0..5 and nothing normalises it back to NULL). To a user those are
    # the same thing, so the predicate covers both. Given with a score range it
    # widens the range to take the unrated in too.
    unscored: bool = False
    # "Favourites only": the owner's hearted pictures (``picture.favourite``).
    favourite_only: bool = False
    smart_score_bucket: Optional[str] = None
    resolution_bucket: Optional[str] = None
    comfyui_models_filter: Optional[List[str]] = None
    comfyui_loras_filter: Optional[List[str]] = None
    # "The pictures made by this workflow": its variants (or a manual one's
    # own id), resolved from the hub BEFORE the query is built, because the
    # workflows live in the hub and the pictures live in the vault - there is
    # no join to write (:func:`workflow_keys_predicate`). ``None`` is no
    # filter; an empty list is a workflow with no filed variant and must match
    # nothing, so every read of this field tests ``is not None`` rather than
    # truthiness.
    workflow_structural_hashes: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None
    tags_rejected_filter: Optional[List[str]] = None
    hidden_tags_filter: Optional[List[str]] = None
    tags_confidence_above_filter: Optional[List[str]] = None
    tags_confidence_below_filter: Optional[List[str]] = None
    face_filter: Optional[str] = None
    # The grid's "Stacks" segment: ``"stacked"`` / ``"unstacked"`` / ``"unresolved"``.
    # "All" is the absence of the filter, so it is spelled as ``None`` rather than as a
    # fourth value the compiler would have to recognise and then ignore.  The first two
    # read the picture's own stack membership; ``"unresolved"`` is dedup-queue state, so
    # it is the one value that leaves the ``picture`` table.  An unrecognised value
    # compiles to no predicate (same as ``face_filter``).
    stack_state: Optional[str] = None
    # "Impossible tags" grid filters - live, computed from the picture's own tags/faces
    # (no precomputed queue). Recognised kinds: ``"no_face"`` (no detected face yet a
    # face-requiring tag) and ``"no_humans"`` (no face, tagged "no humans"/"scenery", yet
    # a person-tag). Multiple kinds are OR'd. Only ever narrows a scope-enforced listing.
    impossible_sources: Optional[List[str]] = None
    # Why the picture is untagged (one of ``TAG_STATES``); ``None`` is no filter.
    # ``unprocessable_ids`` is the in-memory registry's suppressed set, which only
    # the caller can read, so a filter with a state and no set is refused at
    # compile time rather than read as "nothing is unprocessable".
    tag_state: Optional[str] = None
    unprocessable_ids: Optional[frozenset[int]] = None
    file_path_prefix: Optional[str] = None
    # ``find()`` browses a folder and shows direct children only.  The stats
    # sidebar instead aggregates the whole sub-tree under the folder.  ``True``
    # selects the children-only behaviour; ``False`` the sub-tree.  Both are
    # separator-bounded (see ``FolderPrefix``).
    file_path_prefix_children_only: bool = True
    import_source_folder: Optional[str] = None
    # Added at or after this time (the Inbox's period, see ``inbox_cutoff``).
    imported_since: Optional[datetime] = None

    # --- scope / lifecycle flags ---
    include_deleted: bool = False
    only_deleted: bool = False
    # When ``False`` the caller applies the deleted predicate itself (e.g. woven into
    # a membership branch) and the compiler emits no ``deleted`` clause.
    apply_deleted_filter: bool = True
    include_unimported: bool = True

    def _smart_score_bucket_predicates(self) -> list[ColumnElement]:
        bucket = self.smart_score_bucket
        if bucket == "unscored":
            return [Picture.smart_score.is_(None)]
        if bucket == "1-2":
            return [Picture.smart_score.is_not(None), Picture.smart_score < 2.0]
        if bucket == "2-3":
            return [Picture.smart_score >= 2.0, Picture.smart_score < 3.0]
        if bucket == "3-4":
            return [Picture.smart_score >= 3.0, Picture.smart_score < 4.0]
        if bucket == "4-5":
            return [Picture.smart_score >= 4.0]
        return []

    def _resolution_bucket_predicates(self) -> list[ColumnElement]:
        bucket = self.resolution_bucket
        if bucket == "unknown":
            return [or_(Picture.width.is_(None), Picture.height.is_(None))]
        area = Picture.width * Picture.height
        not_null = [Picture.width.is_not(None), Picture.height.is_not(None)]
        if bucket == "lt1mp":
            return [*not_null, area < 1_000_000]
        if bucket == "1-4mp":
            return [*not_null, area >= 1_000_000, area < 4_000_000]
        if bucket == "4-8mp":
            return [*not_null, area >= 4_000_000, area < 8_000_000]
        if bucket == "8-16mp":
            return [*not_null, area >= 8_000_000, area < 16_000_000]
        if bucket == "16plus":
            return [*not_null, area >= 16_000_000]
        return []

    def file_path_prefix_predicates(self) -> list[ColumnElement]:
        """The "in this folder" clauses.

        Public because the stack-leader collapse compiles them too: with a
        folder scope the stack's global position-0 member can live in another
        folder, so the collapse has to know which members are in scope, and a
        second spelling of "in this folder" is how it gets that wrong.
        """
        if self.file_path_prefix is None:
            return []
        # One spelling of "in this folder" (FolderPrefix): separator-bounded, so
        # "/ref/photos" never takes "/ref/photos2/a.jpg"; case and separators
        # as the platform's paths compare; "%" and "_" are plain characters.
        prefix = FolderPrefix.of(self.file_path_prefix)
        if not self.file_path_prefix_children_only:
            # Sub-tree match used by the stats sidebar.
            return [prefix.under(Picture.file_path)]
        return [prefix.directly_in(Picture.file_path)]

    def face_predicates(self) -> list[ColumnElement]:
        """The "has a real face" / "has none" clauses.

        Public and separate from :meth:`predicates` because the stack-leader
        collapse needs the SAME clause: a face facet can hide a stack's global
        position-0 member (the cover is often the tidy shot and the face is on a
        sibling), so the collapse has to know which members are in scope, and a
        second spelling of "has a face" is how it gets that wrong.

        ``face_index == -1`` is the sentinel row the extractor writes for a
        picture it found NO face in, so both directions exclude it.
        """
        if self.face_filter == "with_face":
            return [text(_HAS_REAL_FACE_SQL)]
        if self.face_filter == "without_face":
            return [text(f"NOT {_HAS_REAL_FACE_SQL}")]
        return []

    def favourite_predicates(self) -> list[ColumnElement]:
        """The "Favourites only" clause.

        Public for the stack-leader collapse, like :meth:`face_predicates`: the
        heart is often on a stack member that is not its global cover.
        """
        return [Picture.favourite.is_(True)] if self.favourite_only else []

    def predicates(self) -> list[ColumnElement]:
        """Compile this filter to a list of SQLAlchemy WHERE clauses.

        The clauses are intended to be ANDed together (``stmt.where(*predicates)``).
        ComfyUI membership is emitted as one clause per model/LoRA (AND semantics);
        ``Picture.find`` deliberately keeps its own OR + stack-expansion variant and
        therefore leaves ``comfyui_*_filter`` unset here.
        """
        preds: list[ColumnElement] = []

        if self.apply_deleted_filter:
            # ``deleted = 0``, not ``IS 0``. The column is NOT NULL, so both
            # select the same rows, but SQLite chooses a partial index only when
            # the query states its WHERE term exactly, and the grid's leader
            # indexes (``ix_picture_grid_leaders_*``) say ``deleted = 0``.
            if self.only_deleted:
                preds.append(Picture.deleted == true())
            elif not self.include_deleted:
                preds.append(Picture.deleted == false())

        if not self.include_unimported:
            preds.append(Picture.imported_at.is_not(None))

        preds.extend(self.file_path_prefix_predicates())

        if self.import_source_folder:
            preds.append(Picture.import_source_folder == self.import_source_folder)

        if self.imported_since is not None:
            preds.append(Picture.imported_at >= self.imported_since)

        if self.format:
            preds.append(in_json(Picture.format, self.format))

        score_range: list[ColumnElement] = []
        if self.min_score is not None:
            score_range.append(Picture.score >= self.min_score)
        if self.max_score is not None:
            score_range.append(Picture.score <= self.max_score)
        if self.unscored:
            # With a range, unscored is added to it ("3-4 plus the unrated"),
            # alone it is the unrated only.
            unrated = or_(Picture.score.is_(None), Picture.score == 0)
            preds.append(or_(and_(*score_range), unrated) if score_range else unrated)
        else:
            preds.extend(score_range)

        preds.extend(self.favourite_predicates())

        preds.extend(self._smart_score_bucket_predicates())
        preds.extend(self._resolution_bucket_predicates())

        if self.tags_filter:
            for i, tag in enumerate(self.tags_filter):
                group = _group_term_clause(tag, i, negate=False)
                if group is not None:
                    preds.append(text(group[0]).bindparams(**group[1]))
                    continue
                preds.append(
                    text(
                        f"EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :tag_filter_{i})"
                    ).bindparams(**{f"tag_filter_{i}": tag})
                )

        if self.tags_rejected_filter:
            for i, tag in enumerate(self.tags_rejected_filter):
                group = _group_term_clause(tag, i, negate=True)
                if group is not None:
                    preds.append(text(group[0]).bindparams(**group[1]))
                    continue
                preds.append(
                    text(
                        f"NOT EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :rejected_tag_filter_{i})"
                    ).bindparams(**{f"rejected_tag_filter_{i}": tag})
                )

        if self.hidden_tags_filter:
            # The owner's whole hidden list, so one JSON array, not a
            # parameter per tag.
            preds.append(
                text(
                    "NOT EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id"
                    " AND LOWER(tag.tag) IN (SELECT value FROM json_each(:ht)))"
                ).bindparams(ht=json.dumps(sorted(set(self.hidden_tags_filter))))
            )

        if self.tags_confidence_above_filter:
            for i, entry in enumerate(self.tags_confidence_above_filter):
                tag, threshold = entry.rsplit(":", 1)
                if float(threshold) <= 0.0:
                    # NOTE: the whole disjunction is wrapped in an extra outer pair of
                    # parentheses.  SQL ``AND`` binds tighter than ``OR``, so without
                    # the wrapper this clause leaks its ``OR`` when ANDed with the
                    # other predicates (e.g. the ``Picture.id == id`` narrowing in
                    # ``matches()``), letting the second branch escape every other
                    # filter.  The pre-refactor copies omitted this wrapper.
                    preds.append(
                        text(
                            f"(("
                            f"EXISTS (SELECT 1 FROM tag_prediction WHERE tag_prediction.picture_id = picture.id"
                            f" AND tag_prediction.tag = :ca_tag_{i} AND tag_prediction.confidence >= :ca_thresh_{i})"
                            f" AND NOT EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :ca_tag_{i})"
                            f") OR ("
                            f"EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :ca_tag_{i})"
                            f" AND NOT EXISTS (SELECT 1 FROM tag_prediction WHERE tag_prediction.picture_id = picture.id AND tag_prediction.tag = :ca_tag_{i})"
                            f"))"
                        ).bindparams(
                            **{f"ca_tag_{i}": tag, f"ca_thresh_{i}": float(threshold)}
                        )
                    )
                else:
                    preds.append(
                        text(
                            f"EXISTS (SELECT 1 FROM tag_prediction WHERE tag_prediction.picture_id = picture.id"
                            f" AND tag_prediction.tag = :ca_tag_{i} AND tag_prediction.confidence >= :ca_thresh_{i})"
                            f" AND NOT EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :ca_tag_{i})"
                        ).bindparams(
                            **{f"ca_tag_{i}": tag, f"ca_thresh_{i}": float(threshold)}
                        )
                    )

        if self.tags_confidence_below_filter:
            for i, entry in enumerate(self.tags_confidence_below_filter):
                tag, threshold = entry.rsplit(":", 1)
                preds.append(
                    text(
                        f"EXISTS (SELECT 1 FROM tag_prediction WHERE tag_prediction.picture_id = picture.id"
                        f" AND tag_prediction.tag = :cb_tag_{i} AND tag_prediction.confidence < :cb_thresh_{i})"
                        f" AND EXISTS (SELECT 1 FROM tag WHERE tag.picture_id = picture.id AND tag.tag = :cb_tag_{i})"
                    ).bindparams(
                        **{f"cb_tag_{i}": tag, f"cb_thresh_{i}": float(threshold)}
                    )
                )

        # ComfyUI membership: AND of one EXISTS per model/LoRA (used by
        # semantic_search / listing-candidate / grouped-misc sites).  ``find()`` does
        # not set these fields - it applies its own OR + stack-expansion variant.
        if self.comfyui_models_filter:
            for i, m in enumerate(self.comfyui_models_filter):
                preds.append(
                    text(
                        f"EXISTS (SELECT 1 FROM json_each(picture.comfyui_models) WHERE value = :cmf_{i})"
                    ).bindparams(**{f"cmf_{i}": m})
                )
        if self.comfyui_loras_filter:
            for i, m in enumerate(self.comfyui_loras_filter):
                preds.append(
                    text(
                        f"EXISTS (SELECT 1 FROM json_each(picture.comfyui_loras) WHERE value = :clf_{i})"
                    ).bindparams(**{f"clf_{i}": m})
                )
        if self.workflow_structural_hashes is not None:
            # Each key list is bound as one JSON array (``in_json``), so no
            # number of variants meets SQLite's variable limit.
            preds.append(workflow_keys_predicate(self.workflow_structural_hashes))

        preds.extend(self.face_predicates())

        if self.stack_state in ("stacked", "unstacked"):
            # Both halves ask "is this picture in a stack that is still a stack",
            # not "does this row carry a stack_id". A soft-deleted picture keeps
            # its stack_id so a Scrapheap restore can put it back, so a stack
            # whose other members are all scrapheaped leaves one live picture
            # still carrying one. Keying on the column alone served that survivor
            # under `stacked` while the grid drew it as a plain picture, because
            # `_enrich_stack_counts` counts live members and `StackBadge` hides
            # below two. It also fell out of `unstacked`, so it appeared under
            # neither. Collapsing a stack to its cover produces that state every
            # time, which is how a rare inconsistency became a guaranteed one.
            #
            # Imported lazily for the same reason `live_groups_filter` is:
            # `db_models.picture` sits between this module and that one.
            from pixlstash.services.stack_membership import in_a_live_stack

            live_stack = in_a_live_stack()
            preds.append(live_stack if self.stack_state == "stacked" else ~live_stack)
        elif self.stack_state == "unresolved":
            # A picture the duplicate detector grouped with others and nobody has
            # ruled on yet. Deliberately keyed on ``DedupGroup.resolved`` alone and
            # NOT on the caller's tier policy, so this matches what the sidebar
            # badge counts (see ``DedupGroup.resolved``); filtering by tier here
            # would show a count the grid could not account for.
            #
            # ``live_groups_filter`` is the OTHER half of what the badge counts,
            # and leaving it out is what let a scrapheaped partner keep marking
            # its survivor: the group had left the queue and the badge, but the
            # grid still called the remaining picture unresolved.
            preds.append(Picture.id.in_(select(_unresolved_pictures().c.picture_id)))

        if self.impossible_sources:
            # Each selected kind is a live "impossible combination" predicate over the
            # picture's own tags/faces (no precomputed queue). Multiple selected kinds
            # are OR'd, so the grid shows a picture matching ANY chosen filter.
            clauses: list[ColumnElement] = []
            if "no_face" in self.impossible_sources:
                # No detected face, yet carries a face-requiring tag.
                face_exists, fp = _tag_in_exists("imp_face", _FACE_REQUIRING_LOWER)
                clauses.append(
                    text(f"({_NO_REAL_FACE_SQL} AND {face_exists})").bindparams(**fp)
                )
            if "no_humans" in self.impossible_sources:
                # No detected face, tagged "no humans"/"scenery", yet carries a person-tag.
                meta_exists, mp = _tag_in_exists("imp_meta", _OBJECT_META_LOWER)
                person_exists, pp = _tag_in_exists("imp_person", _PERSON_LOWER)
                clauses.append(
                    text(
                        f"({_NO_REAL_FACE_SQL} AND {meta_exists} AND {person_exists})"
                    ).bindparams(**mp, **pp)
                )
            if clauses:
                preds.append(or_(*clauses))

        if self.tag_state is not None:
            if self.unprocessable_ids is None:
                raise ValueError(
                    "PredicateFilter.tag_state needs unprocessable_ids: the caller "
                    "reads them from vault.db.unprocessable_images"
                )
            preds.extend(tag_state_predicates(self.tag_state, self.unprocessable_ids))

        return preds

    def apply(self, stmt):
        """Apply every compiled predicate to ``stmt`` and return the new statement."""
        for clause in self.predicates():
            stmt = stmt.where(clause)
        return stmt

    def matches(self, session, picture_id: int) -> bool:
        """Return ``True`` if the single picture ``picture_id`` satisfies this filter.

        This is the set predicate narrowed to one id - the hook consumed by staging
        auto-triage.  There is no separate Python-side evaluator; the SQL semantics
        are the same as the set queries.
        """
        stmt = self.apply(select(Picture.id)).where(Picture.id == picture_id)
        return session.exec(stmt).first() is not None

    @classmethod
    def from_query_params(
        cls, request, *, children_only: bool = True
    ) -> "PredicateFilter":
        """Build a filter from the intrinsic-attribute query params on ``request``.

        Covers the vocabulary shared by the picture-listing routes.  Membership
        params (``set``/``character``/``project``) and pagination/sort are *not*
        read here - they are resolved separately by the caller.  Unspecified params
        simply leave their field at the default, so a route that never sends a given
        param is unaffected.
        """
        qp = request.query_params

        def _int_or_none(name: str) -> Optional[int]:
            raw = qp.get(name)
            if raw is None:
                return None
            try:
                return parse_sqlite_int(raw)
            except (TypeError, ValueError) as exc:
                # A malformed numeric query param is a client error, not a
                # server fault. Raise 422 with a clear message instead of
                # letting a bare int() bubble up as an unhandled 500.
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid {name}: must be an integer",
                ) from exc

        def _terms(name: str) -> List[str]:
            # One bound term per value (see ``MAX_FILTER_TERMS``): a list past
            # the cap is refused here, before it reaches SQLite's variable limit.
            values = qp.getlist(name)
            if len(values) > MAX_FILTER_TERMS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Too many {name} values: at most {MAX_FILTER_TERMS}",
                )
            return values

        def _confidence_entries(name: str) -> Optional[List[str]]:
            # ``tag:threshold``, compiled by ``predicates()`` with ``rsplit`` and
            # ``float``: a value they cannot read is the client's error, refused
            # here rather than raised mid-query as a 500.
            entries = _terms(name)
            for entry in entries:
                tag, sep, threshold = entry.rpartition(":")
                try:
                    readable = bool(sep and tag) and math.isfinite(float(threshold))
                except ValueError:
                    readable = False
                if not readable:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid {name}: each value must be tag:number",
                    )
            return entries or None

        return cls(
            format=qp.getlist("format") or None,
            min_score=_int_or_none("min_score"),
            max_score=_int_or_none("max_score"),
            unscored=is_truthy_flag(qp.get("unscored")),
            favourite_only=is_truthy_flag(qp.get("favourite")),
            smart_score_bucket=qp.get("smart_score_bucket") or None,
            resolution_bucket=qp.get("resolution_bucket") or None,
            comfyui_models_filter=_terms("comfyui_model") or None,
            comfyui_loras_filter=_terms("comfyui_lora") or None,
            tags_filter=_terms("tag") or None,
            tags_rejected_filter=_terms("rejected_tag") or None,
            hidden_tags_filter=qp.getlist("hidden_tag") or None,
            tags_confidence_above_filter=_confidence_entries("tag_confidence_above"),
            tags_confidence_below_filter=_confidence_entries("tag_confidence_below"),
            face_filter=qp.get("face_filter") or None,
            stack_state=qp.get("stack_state") or None,
            impossible_sources=qp.getlist("impossible_tag_source") or None,
            file_path_prefix=qp.get("file_path_prefix") or None,
            file_path_prefix_children_only=children_only,
            import_source_folder=qp.get("import_source_folder") or None,
            imported_since=inbox_cutoff(qp.get("inbox")),
        )
