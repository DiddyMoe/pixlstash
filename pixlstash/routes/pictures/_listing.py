import functools
import inspect
import math
import re
import sys
import threading
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import (
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from sqlalchemy import (
    or_,
)
from sqlalchemy.types import TypeDecorator
from sqlmodel import Session, SQLModel, select

from pixlstash.database import DBPriority, VaultDatabase
from pixlstash.hub import workflow_cards
from pixlstash.hub.workflow_card_reads import (
    run_filed_workflow_ids,
    variants_in_workflows,
)
from pixlstash.db_models import (
    Face,
    Picture,
    PictureProjectMember,
    SortMechanism,
)
from pixlstash.db_models.guest_score import GuestScore
from pixlstash.db_models.user_token import UserToken
from pixlstash.pixl_logging import get_logger
from pixlstash.scoring import (
    count_pictures_by_character_likeness,
    find_pictures_by_character_likeness_sql,
)
from pixlstash.utils.serialization_utils import safe_model_dict
from pixlstash.utils.sql_chunking import chunked
from pixlstash.utils.sqlite_ints import SQLITE_INT_MAX, SqliteInt, parse_sqlite_int
from pixlstash.utils.service.filter_helpers import (
    collect_set_filter_ids,
    fetch_scope_allowed_picture_ids,
    fetch_set_candidate_ids,
    is_scrapheap_view_id,
    normalize_set_mode,
    project_membership_exists_clause,
    project_unassigned_clause,
    request_targets_scrapheap,
)
from pixlstash.utils.query.predicate_filter import (
    NOT_MADE_BY,
    ONLY_MADE_BY,
    RUN_VARIANT,
    PredicateFilter,
    inbox_cutoff,
    is_truthy_flag,
    parse_tag_state,
)

from ._helpers import (
    _enrich_scrapheap_retention,
    _enrich_stack_counts,
    _face_cluster_ids,
    _get_hidden_tags_from_request,
)


logger = get_logger(__name__)

# Upper bound on repeated ``id`` params per listing request. Each id is a bound
# parameter of one IN (...) and a row of one response, so an unbounded list lets
# one URL serialise an arbitrarily large read on the DB queue. Mirrors
# BULK_DELETE_MAX_IDS (reject over the limit rather than truncate); the frontend
# chunks larger id lists into several calls.
LIST_MAX_IDS = 1000

# The raw query names the listing reads out of ``query_params`` by the same
# name as a ``Picture.find`` keyword, and pops before any splat.
_ROUTE_READ_FIND_NAMES = frozenset(
    {
        "file_path_prefix",
        "limit",
        "max_score",
        "min_score",
        "offset",
        "only_deleted",
        "resolution_bucket",
        "smart_score_bucket",
        "tag_state",
        "unscored",
    }
)
# Every other parameter name of ``run_immediate_read_task`` or ``Picture.find``
# is dropped from the raw query before the route reads it. The leftovers are
# splatted into ``run_immediate_read_task(Picture.find, <explicit kwargs>,
# **query_params)``, so such a name would either repeat a keyword the route
# passes itself (a TypeError: ``?select_fields=id`` was a 500) or set one no
# client is meant to set (``include_deleted``, ``count_only``, a list filter as
# a bare string). The route sets the derived ones (``tags_filter`` from
# ``tag``, ``stack_state`` once validated, ...) itself afterwards. Read from
# the signatures, so a keyword added to either is covered as well.
_SPLAT_RESERVED_NAMES = (
    frozenset(
        name
        for fn in (Picture.find.__func__, VaultDatabase.run_immediate_read_task)
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )
    - _ROUTE_READ_FIND_NAMES
)


def enforce_list_id_cap(request: Request) -> None:
    """Refuse a listing request that names more than ``LIST_MAX_IDS`` ids.

    Checked at the HTTP routes, not in ``select_pictures_for_listing``: a smart
    set's stored rule may hold a longer id list, and its refresh is not a request
    anyone is waiting on.
    """
    count = len(request.query_params.getlist("id"))
    if count > LIST_MAX_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many id params ({count}); the maximum is {LIST_MAX_IDS} "
                "per request"
            ),
        )


def _storable_picture_ids(values) -> list[int]:
    """The ``id`` filter values that name a picture SQLite can store.

    A value that is not a digit string, or is one past SQLite's range, names
    no picture, so it is dropped as a malformed ``id`` always has been rather
    than bound (where a past-range one overflows in the driver).
    """
    return [int(i) for i in values if str(i).isdigit() and int(i) <= SQLITE_INT_MAX]


def _reference_folder_id(query_params: dict) -> int | None:
    """The ``reference_folder_id`` filter, for the branches that name filters.

    The main listing and the Scrapheap take it out of the ``**query_params``
    splat as a ``Picture.find`` column filter. The unassigned view (the Inbox
    included) and the likeness sort's candidates name each filter instead, so
    they read it here, or a reference folder's view lists the whole library
    while the stats count the folder. A value SQLite cannot store names no
    folder and is refused.
    """
    raw = query_params.get("reference_folder_id")
    if raw in (None, ""):
        return None
    try:
        return parse_sqlite_int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Refusing reference_folder_id %r: not an integer SQLite stores", raw
        )
        raise HTTPException(status_code=400, detail="Invalid reference_folder_id")


# Stored values already answered null, as (table, row id, column), so a damaged
# row the grid reads on every page is logged once and not once per request.
# Past the cap the record starts over and a value still bad is logged once
# more. A library switch starts it over too: the next library reuses row ids.
_REFUSED_STORED_VALUES_LIMIT = 10_000
_refused_stored_values: set[tuple[str, int, str]] = set()
_refused_stored_values_lock = threading.Lock()


def forget_refused_stored_values() -> None:
    """Start the record of stored values answered null over."""
    with _refused_stored_values_lock:
        _refused_stored_values.clear()


def _first_refusal(table: str, row_id, column: str) -> bool:
    """Whether this is the first time *column* of this row is answered null."""
    key = (table, row_id, column)
    with _refused_stored_values_lock:
        if key in _refused_stored_values:
            return False
        if len(_refused_stored_values) >= _REFUSED_STORED_VALUES_LIMIT:
            _refused_stored_values.clear()
        _refused_stored_values.add(key)
    return True


@functools.cache
def _declared_type(table: str, column: str):
    """The Python type *table*'s *column* declares, or None for no such column.

    SQLite does not enforce it: a row written by a tool outside the app, or
    before a check existed, can hold a REAL ``inf`` or a BLOB in an INTEGER
    column, or a BLOB in a TEXT one.
    """
    found = SQLModel.metadata.tables[table].columns.get(column)
    if found is None:
        return None
    declared = found.type
    if isinstance(declared, TypeDecorator):
        declared = declared.impl_instance
    return declared.python_type


def stored_value(value, *, table: str, row_id, column: str, picture_id, log=logger):
    """A stored column value as the column declares it, or None.

    For a response built from a row loaded from the database. An INTEGER
    column's value that is not an integer, a REAL one's that is not a finite
    number, or a TEXT one's that is not text fails the response model, or
    JSON itself, and answered 500 for the whole page. Such a value is logged
    with the picture and the column, once per row and column (a damaged row
    the grid reads on every page does not flood the log), and answered as
    null; the rest of the row, and every other row, is served. NULL is normal
    and passes through. A column of another type (a date, a flag, a blob) is
    converted by the database layer on load and is returned as it is, as is a
    name that is not a column of *table*.

    Args:
        value: The value read from the row.
        table: The table the row is in.
        row_id: The row's id.
        column: The column the value was read from.
        picture_id: The picture the row is or belongs to, named in the log.
        log: The logger of the route answering null.
    """
    declared = _declared_type(table, column)
    if value is None or declared not in (int, float, str):
        return value
    if declared is str:
        carried = isinstance(value, str)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        carried = False
    elif declared is int:
        carried = isinstance(value, int)
    else:
        carried = math.isfinite(value)
    if carried:
        return value
    if _first_refusal(table, row_id, column):
        log.warning(
            "Picture %s has a stored %s that is not a %s: %.80r; answered as null",
            picture_id,
            column if table == "picture" else f"{table} {row_id} {column}",
            {int: "integer", float: "finite number", str: "text"}[declared],
            value,
        )
    return None


def stored_picture_value(value, *, picture_id, column: str):
    """:func:`stored_value` for a ``picture`` column of picture *picture_id*."""
    return stored_value(
        value,
        table="picture",
        row_id=picture_id,
        column=column,
        picture_id=picture_id,
    )


def stored_picture_dict(pic) -> dict:
    """``safe_model_dict`` of a loaded picture, read through the column types.

    Each loaded ``picture`` column is read through :func:`stored_picture_value`
    before ``safe_model_dict`` sees it: that writes a BLOB as base64 text, after
    which a TEXT column's BLOB cannot be told from text, and a row's ``score``
    is compared before any response model reads it.
    """
    raw = vars(pic)
    picture_id = raw.get("id")
    columns = Picture.__table__.columns
    return {
        key: None
        if key in columns
        and key in raw
        and stored_picture_value(raw[key], picture_id=picture_id, column=key) is None
        else value
        for key, value in safe_model_dict(pic).items()
    }


class StoredPictureRow(BaseModel):
    """A response model whose fields are ``picture`` columns by the same name.

    Each such field, and each extra key of a model that keeps extras, is read
    through :func:`stored_picture_value`, so a stored value its column does
    not declare is answered as null rather than failing the response. ``id``
    is declared first by every subclass, so it is validated before the field
    the log line names it for.
    """

    @model_validator(mode="before")
    @classmethod
    def stored_extras_or_null(cls, data):
        if not isinstance(data, dict):
            return data
        return {
            key: value
            if key in cls.model_fields
            else stored_picture_value(value, picture_id=data.get("id"), column=key)
            for key, value in data.items()
        }

    @field_validator("*", mode="before")
    @classmethod
    def stored_or_null(cls, value, info: ValidationInfo):
        return stored_picture_value(
            value, picture_id=info.data.get("id"), column=info.field_name
        )


class GridPicture(StoredPictureRow):
    """A single picture row as returned by the grid listing endpoints.

    Fields are the projection used when ``fields=grid``; all are optional
    because the exact set returned depends on the requested ``fields`` and on
    server-side enrichment (stack counts, guest-score overlay, thumbnail/face
    data added by ``POST /pictures/thumbnails``).

    **This model is enforced, not just documented.** It is the declared
    ``response_model`` of the grid listing routes, so FastAPI validates every row
    against it and - since the config does not set ``extra="allow"`` - **drops any
    key not declared here**. That is load-bearing beyond tidiness: ``fields=full``
    selects ``Picture.metadata_fields()``, which includes ``project_id``, and the
    only reason those rows do not leak a project id to a token that cannot see it
    is that this model has no ``project_id`` field to serialise it into (§16.6).
    Adding a field here widens what the route returns; removing one silently
    stops returning it.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 12345,
                "width": 1024,
                "height": 1536,
                "format": "PNG",
                "score": 4,
                "smart_score": 3.87,
                "created_at": "2026-01-14T09:30:00",
                "imported_at": "2026-01-15T18:02:11",
                "stack_id": 87,
                "stack_position": 0,
                "stack_count": 5,
                "file_path": "01a43b7e-dda4-45bb-a912-2ed4b786260d.png",
            }
        },
    )

    id: int | None = Field(None, description="Picture id (primary key).")
    width: int | None = Field(None, description="Image width in pixels.")
    height: int | None = Field(None, description="Image height in pixels.")
    format: str | None = Field(None, description="File format, e.g. PNG, JPEG, MP4.")
    score: int | None = Field(None, description="Manual star rating (0-5).")
    favourite: bool | None = Field(
        None, description="The owner has hearted this picture."
    )
    smart_score: float | None = Field(
        None, description="Model-predicted aesthetic score."
    )
    smartScore: float | None = Field(
        None, description="camelCase alias of smart_score (legacy clients)."
    )
    created_at: datetime | None = Field(
        None, description="When the source file was created."
    )
    imported_at: datetime | None = Field(
        None, description="When the picture was imported into the vault."
    )
    stack_id: int | None = Field(
        None, description="Id of the stack this picture belongs to, if any."
    )
    stack_position: int | None = Field(
        None, description="Position within its stack; 0 is the stack leader."
    )
    stack_count: int | None = Field(
        None,
        description="Number of pictures in the stack (only set on stack leaders "
        "when stack_leaders_only is used).",
    )
    tag_uncertainty: float | None = Field(
        None, description="Aggregate tag-prediction uncertainty."
    )
    anomaly_tag_uncertainty: float | None = Field(
        None, description="Uncertainty restricted to anomaly/problem tags."
    )
    text_score: float | None = Field(
        None, description="Relevance score for the active text search, if any."
    )
    reference_folder_id: int | None = Field(
        None, description="Source reference-folder id for externally-referenced files."
    )
    orientation: int | None = Field(
        None,
        description=(
            "Stored EXIF orientation, 1-8, or null when it has not been read "
            "yet. Carried in the grid projection because it is the cache-buster "
            "for the full-size media URL: an in-place rotate changes nothing "
            "else about the picture."
        ),
    )
    file_path: str | None = Field(
        None,
        description="Path relative to the image root (omitted when grid_lite=true).",
    )
    # Not in the grid projection: returned only to a caller that names them in
    # `fields` (the stats rail's Properties for several pictures, #2161). Both
    # are what `GET /pictures/{id}/metadata` already returns for each picture.
    size_bytes: int | None = Field(
        None, description="The file's size in bytes (only when asked for in fields)."
    )
    comfyui_models: str | None = Field(
        None,
        description="JSON list of the checkpoint / UNET names the picture was "
        "generated with; '[]' when it was checked and has none (only when asked "
        "for in fields).",
    )
    purge_at: str | None = Field(
        None,
        description=(
            "Scrapheap view only (`only_deleted=true`). ISO 8601 UTC instant at "
            "which this picture will be permanently deleted by the retention "
            "auto-purge, already including the grace floor that a retention "
            "reduction puts under every deadline. `null` means it will never "
            "be auto-purged - because retention is set to Never, because the "
            "picture is `auto_purge_exempt`, or because it carries no "
            "soft-delete timestamp. Render the countdown from this value; do "
            "not recompute it client-side."
        ),
        examples=["2026-08-21T09:15:00+00:00"],
    )
    auto_purge_exempt: bool | None = Field(
        None,
        description=(
            "Scrapheap view only (`only_deleted=true`). True when the retention "
            "sweep will never destroy this picture, so `purge_at` is null. Show "
            "an exemption badge instead of a countdown; see "
            "`auto_purge_exempt_reason` for which badge."
        ),
        examples=[False],
    )
    auto_purge_exempt_reason: str | None = Field(
        None,
        description=(
            "Scrapheap view only (`only_deleted=true`). Why the picture is "
            "exempt, or null when it is not:\n\n"
            "- `protected` - a reference-folder original with "
            "`allow_delete_file=false`. Permanent and intrinsic; only the "
            "manual, consent-gated delete-forever can destroy it.\n"
            "- `locked` - frozen by a locked picture-set (directly, or via a "
            "stack sibling). Clearable: unlock the set and the picture rejoins "
            "the normal retention schedule.\n\n"
            "When both apply, `protected` wins - it is the stronger reason, and "
            "labelling such a picture merely 'locked' would understate why it "
            "is being kept."
        ),
        examples=["locked"],
    )
    character_likeness: float | None = Field(
        None,
        description=(
            "`sort=CHARACTER_LIKENESS` only. Softmax-weighted cosine similarity "
            "of the picture's best face to `reference_character_id`'s reference "
            "faces, 0.0-1.0, the value the rows are ordered by. `null` for "
            "every other sort."
        ),
        examples=[0.83],
    )


class StreamPicturesResponse(BaseModel):
    """One batch of the streaming listing plus pagination state."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pictures": [GridPicture.model_config["json_schema_extra"]["example"]],
                "done": False,
                "next_offset": 1000,
            }
        }
    )

    pictures: list[GridPicture] = Field(
        description="Pictures in this batch, in sort order."
    )
    done: bool = Field(
        description="True when the underlying SQL row count for this batch was "
        "<= batch_limit, i.e. there are no further rows to fetch.",
    )
    next_offset: int = Field(
        description="Offset to pass on the next request to continue paginating.",
    )


class PictureCountResponse(BaseModel):
    """Total number of pictures matching the listing filters.

    ``count`` is an integer for every sort (including CHARACTER_LIKENESS,
    which counts via its candidate/character filter); ``null`` is reserved
    for a future sort whose total genuinely cannot be computed cheaply."""

    model_config = ConfigDict(json_schema_extra={"example": {"count": 28259}})

    count: int | None = Field(
        None,
        description="Total matching pictures, or null when not cheaply computable.",
    )


class PictureListFilters:
    """Shared query-param filters for the picture listing endpoints.

    These were previously parsed directly out of ``request.query_params`` and so
    were invisible in the OpenAPI schema. Declaring them here as a ``Depends``
    dependency surfaces them in the generated docs (and validates the typed
    ones) without changing behaviour: the handlers still read the raw values
    from ``request.query_params``.
    """

    def __init__(
        self,
        character_id: str | None = Query(
            None,
            description="Character filter: a character id, or 'UNASSIGNED'.",
            examples=["42"],
        ),
        unassigned_by: Literal["character", "set"] | None = Query(
            None,
            description=(
                "With character_id=UNASSIGNED, narrow 'unassigned' to one half: "
                "'character' (no named face, sets ignored) or 'set' (in no "
                "set, faces ignored). Omitted keeps both."
            ),
        ),
        character_ids: list[str] = Query(
            default=[], description="Multi-character filter (repeatable)."
        ),
        character_mode: str | None = Query(
            None, description="union | intersection | difference | xor"
        ),
        set_id: str | None = Query(None, description="Single picture-set filter."),
        set_ids: list[str] = Query(
            default=[], description="Multi-set filter (repeatable)."
        ),
        set_mode: str | None = Query(
            None, description="union | intersection | difference"
        ),
        base_set_id: str | None = Query(
            None, description="Base set for set_mode=difference."
        ),
        reference_character_id: str | None = Query(
            None, description="Required reference for sort=CHARACTER_LIKENESS."
        ),
        min_score: SqliteInt | None = Query(
            None, description="Minimum star score.", examples=[3]
        ),
        max_score: SqliteInt | None = Query(
            None, description="Maximum star score.", examples=[5]
        ),
        unscored: bool = Query(
            False,
            description=(
                "Pictures the user never rated: score IS NULL or score = 0. "
                "With min_score/max_score, added to that range rather than "
                "intersected with it, so a range spanning every rating "
                "(min_score=0 or max_score=5) plus unscored is every picture."
            ),
        ),
        favourite: bool = Query(
            False, description="Only the owner's favourite pictures."
        ),
        smart_score_bucket: str | None = Query(
            None, description="unscored | 1-2 | 2-3 | 3-4 | 4-5"
        ),
        resolution_bucket: str | None = Query(
            None,
            description="unknown | lt1mp | 1-4mp | 4-8mp | 8-16mp | 16plus",
        ),
        file_path_prefix: str | None = Query(
            None,
            description=(
                "Restrict to pictures directly in this folder (not a text "
                "prefix; its sub-folders are not included)."
            ),
        ),
        face_filter: str | None = Query(
            None, description="with_face | without_face", examples=["with_face"]
        ),
        stack_state: str | None = Query(
            None,
            description=(
                "stacked | unstacked | unresolved. Omit for every picture. "
                "'unresolved' means the picture is in a duplicate group that "
                "carries no verdict yet."
            ),
            examples=["stacked"],
        ),
        tag_state: str | None = Query(
            None,
            description=(
                "pending | unprocessable | none: only the untagged pictures "
                "under that reason, the ones `GET /pictures/stats` counts in "
                "`untagged_breakdown`, in every scope (the Inbox and "
                "character_id=UNASSIGNED included)."
            ),
            examples=["unprocessable"],
        ),
        shared_only: bool = Query(
            False, description="Only pictures shared with the current user."
        ),
        apply_tag_filter: bool = Query(
            False, description="Apply the user's hidden-tag filter."
        ),
        format: list[str] = Query(
            default=[], description="Filter by file format (repeatable)."
        ),
        tag: list[str] = Query(
            default=[],
            description="Require these tags (repeatable).",
            examples=["sunset"],
        ),
        rejected_tag: list[str] = Query(
            default=[], description="Exclude these tags (repeatable)."
        ),
        tag_confidence_above: list[str] = Query(
            default=[], description="'tag:threshold' prediction filter (repeatable)."
        ),
        tag_confidence_below: list[str] = Query(
            default=[], description="'tag:threshold' prediction filter (repeatable)."
        ),
        comfyui_model: list[str] = Query(
            default=[], description="Filter by ComfyUI model (repeatable)."
        ),
        comfyui_lora: list[str] = Query(
            default=[], description="Filter by ComfyUI LoRA (repeatable)."
        ),
        workflow: list[str] = Query(
            default=[],
            description=(
                "Only pictures made by any of these workflows (repeatable, "
                "OR'd), each named by its id (`auto:<core and families "
                "digest>` or a manual group's id). Resolved to the workflows' "
                "variants; an unknown id matches nothing."
            ),
        ),
        workflow_lora: str | None = Query(
            None,
            description=(
                "Only pictures whose workflow loaded this LoRA file, named by "
                "its stored reference (`asset:<sha256>`, as "
                "`GET /workflows/{workflow_id}/lora-summary` serves it). "
                "Narrows `workflow`, and only exactly one; alone, beside "
                "several workflows, or malformed, it matches nothing."
            ),
        ),
        reference_folder_id: SqliteInt | None = Query(
            None, description="Filter by reference-folder id."
        ),
        import_source_folder: str | None = Query(
            None, description="Filter by import source folder."
        ),
        inbox: Literal["24h", "7d", "14d", "30d", "90d", "all"] | None = Query(
            None,
            description=(
                "Only the Inbox: pictures added (imported) within this period "
                "that have no person and no set. Implies "
                "character_id=UNASSIGNED with both halves; `all` has no "
                "cutoff."
            ),
        ),
        id: list[str] = Query(
            default=[], description="Restrict to specific picture ids (repeatable)."
        ),
        face_cluster: SqliteInt | None = Query(
            None,
            ge=1,
            description=(
                "With one character_id: only the person's 'Cluster N', "
                "numbered as `GET /face-clusters` lists them (reference "
                "cluster first, then largest first). Recomputed per request."
            ),
        ),
        face_cluster_threshold: float | None = Query(
            None,
            ge=0.05,
            le=0.95,
            description="The face_cluster strictness, as `/face-clusters` takes it.",
        ),
    ):
        # Values are intentionally not stored: the handlers read them from
        # request.query_params. This dependency exists purely to document and
        # validate the query surface in OpenAPI.
        pass


_ASSET_RE = re.compile(r"^asset:[0-9a-f]{64}$")


def _resolve_workflow_filter(server, query_params: dict) -> list[str] | None:
    """``workflow`` / ``workflow_lora`` as variants, or ``None``.

    The workflows are in the hub and the pictures are in the vault, so there is
    no join to write: the workflows are resolved to their variants here and the
    listing matches ``picture.workflow_structural_hash`` against them, the way
    the ComfyUI LoRA filter matches a name against the picture's own column.

    ``workflow`` is repeatable and the workflows are OR'd, like
    ``comfyui_model``: the keys are the union of each one's. That union is
    exact because every automatic workflow's variants leave out the same
    thing (the runs of every workflow whose runs file on it: the live manual
    ones and the automatic ones the owner saved over,
    ``run_filed_workflow_ids``), and such a workflow's own runs arrive by its
    id, past that exclusion.

    **An empty list is a filter, not the absence of one.** A workflow with no
    filed variant - or a workflow the pictures of this library never ran - matches no
    picture, and returning ``None`` for it would widen the grid to the whole
    library instead. Given both, the two narrow each other, as every other pair
    of filters on this route does. ``workflow_lora`` narrows exactly one
    workflow; beside several it matches nothing, like a malformed one.

    Args:
        server: The running server, for the hub.
        query_params: The parsed query params; every key is popped, so an
            unresolved one cannot reach ``Picture.find(**query_params)``.
            ``workflow`` is a list of ids, or one id as a string.

    Returns:
        The keys to match (``predicate_filter.workflow_keys_predicate``: the
        ids of the named workflows whose runs file on them, and automatic
        ones' variants with the run-filed ids whose runs the variants leave
        out; narrowed by a LoRA, the variants that load it, a run-filed
        workflow's own runs standing in as their variants held to its runs),
        or ``None`` when neither param is set.
    """
    # Presence, not truthiness: `?workflow=` names no workflow, and dropping
    # the filter for it would answer a request for one workflow with the whole
    # library - the exact "no results means the filter was ignored" reading the
    # contract tells clients is wrong. An empty value matches nothing, like any
    # other workflow no picture was made with.
    # Popped whatever happens: `Picture.find` now declares this name, so a
    # client sending `?workflow_structural_hashes=abc` would otherwise reach
    # `PredicateFilter` as a bare string and 500 on its `List[str]`. It could
    # never widen anything - it only adds an IN, and the resolution below
    # overwrites it whenever a real filter resolves - but a reachable 500 is
    # its own bug. (``parse_request_params`` now drops every such name before
    # this runs, ``_SPLAT_RESERVED_NAMES``; the pop keeps this function's own
    # contract for any other caller.)
    query_params.pop("workflow_structural_hashes", None)
    raw = query_params.pop("workflow", None)
    # Deduplicated, in order: a repeated id names the same workflow once.
    workflow_ids = (
        None
        if raw is None
        else list(dict.fromkeys([raw] if isinstance(raw, str) else raw))
    )
    lora = query_params.pop("workflow_lora", None)
    if workflow_ids is None and lora is None:
        return None
    # The one workflow a LoRA may narrow, else None.
    workflow_id = workflow_ids[0] if workflow_ids and len(workflow_ids) == 1 else None
    hub = getattr(server, "hub", None)
    live: list[str] = []
    # The named workflows whose own runs are wanted (by ``run_workflow_id``),
    # the variants wanted in every picture no run-filed workflow made, and the
    # variants wanted only in one workflow's runs.
    own: set[str] = set()
    variants: set[str] = set()
    held: set[str] = set()
    try:
        if hub is None:
            raise LookupError("no hub is attached to this server")
        if workflow_ids is not None:
            live = run_filed_workflow_ids(hub)
            # A manual workflow's pictures are its runs' (it is on no
            # topology, so it has no variant); an automatic one's are its
            # variants', less every run-filed workflow's runs, and - once the
            # owner has saved over it - its own runs' as well.
            own = {wid for wid in workflow_ids if wid in live}
            variants = variants_in_workflows(hub, workflow_ids)
        if lora is not None:
            # Only ever a narrowing of one workflow, and only of a well-formed
            # reference: alone it would parse every stored graph on the hub
            # per request, and this route is open to scoped tokens.
            if workflow_id is not None and _ASSET_RE.match(lora):
                if workflow_id in own:
                    # Its id is every run of it, not a variant: narrow the
                    # variants its own runs made, and keep those to its runs.
                    held = set(
                        workflow_cards.variants_loading(
                            hub, lora, _variants_run_by(server, workflow_id)
                        )
                    )
                    own = set()
                variants = set(
                    workflow_cards.variants_loading(hub, lora, sorted(variants))
                )
            else:
                own, variants = set(), set()
    except Exception as exc:
        # Fail closed, both for a hub that is absent and for one that will not
        # answer: a filter that cannot be resolved must not silently widen the
        # grid to every picture in the library, and it must not 500 a listing
        # the rest of which is perfectly answerable from the vault.
        logger.warning(
            "Could not resolve the workflow filter (workflow=%r, lora=%r): "
            "%s; listing no pictures rather than every picture.",
            workflow_ids,
            lora,
            exc,
        )
        return []
    keys = sorted(own | variants)
    if held:
        keys += [RUN_VARIANT + variant for variant in sorted(held)]
        keys += [ONLY_MADE_BY + workflow_id]
    if variants:
        keys += [NOT_MADE_BY + wid for wid in live]
    return keys


def _variants_run_by(server, workflow_id: str) -> list[str]:
    """The variants a run-filed workflow's own runs made, sorted."""
    return server.vault.db.run_immediate_read_task(
        lambda session: sorted(
            session.exec(
                select(Picture.workflow_structural_hash)
                .where(Picture.run_workflow_id == workflow_id)
                .where(Picture.workflow_structural_hash.is_not(None))
                .distinct()
            ).all()
        )
    )


def parse_picture_filter(server, request: Request) -> PredicateFilter:
    """The narrowing params of a picture request, parsed once for every route.

    The listing (``select_pictures_for_listing``) and the text search
    (``GET /pictures/search``) both read their filters here, so a filter the
    grid sends narrows a search exactly as it narrows the listing, and a bad
    value is refused with the same 422 on both.

    The result carries the intrinsic vocabulary of
    :meth:`PredicateFilter.from_query_params` and the two filters that need the
    server to resolve: ``tag_state`` with the registry's suppressed ids, and
    ``workflow`` / ``workflow_lora`` as variant keys (see
    :func:`_resolve_workflow_filter`). Membership (people, sets, projects),
    paging and sort are not read here; each route resolves those itself.

    Args:
        server: The running server, for the unprocessable registry and the hub.
        request: The request whose query params are read.

    Returns:
        The parsed filter. Its deleted flags are left at their defaults; the
        caller decides the deleted scope.

    Raises:
        HTTPException: 422 for an integer param that is not one SQLite stores,
            a ``tag_confidence_above`` / ``tag_confidence_below`` that is not
            ``tag:number``, or a ``tag_state`` outside ``TAG_STATES``. The
            detail names the param, never the value sent.
    """
    query = request.query_params
    parsed = PredicateFilter.from_query_params(request)
    tag_state = parse_tag_state(query.get("tag_state"))
    # The reasons split on the in-memory suppressed set, so read it once here,
    # the same read the stats breakdown makes.
    unprocessable_ids = (
        frozenset(server.vault.db.unprocessable_images.active_suppressed_ids())
        if tag_state is not None
        else None
    )
    # Presence, not truthiness, as `_resolve_workflow_filter` reads them:
    # `?workflow=` names a workflow no picture was made with.
    workflow_params: dict = {}
    if "workflow" in query:
        workflow_params["workflow"] = query.getlist("workflow")
    if "workflow_lora" in query:
        workflow_params["workflow_lora"] = query.get("workflow_lora")
    return parsed.model_copy(
        update={
            "tag_state": tag_state,
            "unprocessable_ids": unprocessable_ids,
            "workflow_structural_hashes": _resolve_workflow_filter(
                server, workflow_params
            ),
        }
    )


def shared_picture_ids(server, request: Request) -> set[int] | None:
    """The pictures the caller has shared, for ``shared_only=true``.

    Returns:
        The ids of the pictures with a live READ share token of the signed-in
        user, or ``None`` when the request does not ask (``shared_only`` is
        not ``true``) or names no user, so the caller narrows nothing.
    """
    if request.query_params.get("shared_only") != "true":
        return None
    auth_user_id = getattr(request.state, "auth_user_id", None)
    if auth_user_id is None:
        return None

    def _fetch_shared_ids(session: Session, uid: int) -> list[int]:
        now = datetime.now(timezone.utc)
        return list(
            session.exec(
                select(UserToken.resource_id).where(
                    UserToken.user_id == uid,
                    UserToken.resource_type == "picture",
                    UserToken.scope == "READ",
                    UserToken.resource_id.is_not(None),
                    or_(
                        UserToken.expires_at.is_(None),
                        UserToken.expires_at > now,
                    ),
                )
            ).all()
        )

    # Tokens live in the hub, so the "what has this owner shared" lookup runs
    # there. Reading it from the vault would silently return nothing rather
    # than failing, which is the worse shape.
    return set(
        server.hub_engine.run_task(
            _fetch_shared_ids, auth_user_id, priority=DBPriority.IMMEDIATE
        )
    )


def select_pictures_for_listing(
    *,
    server,
    request: Request,
    sort,
    descending,
    offset,
    limit,
    metadata_fields,
    return_ids_only: bool = False,
    exclude_query_params: set[str] | None = None,
    stack_leaders_only: bool = False,
    project_id: int | None = None,
    scope_set_id: int | None = None,
    scope_character_id: int | None = None,
    scope_picture_id: int | None = None,
    stream_state: dict | None = None,
    count_only: bool = False,
):
    """List pictures for a request.

    When `stream_state` is a dict, the function operates in streaming mode:
    it over-fetches by one row at the SQL layer and records the pre-post-filter
    row count in `stream_state["sql_count"]`. This lets the caller decide
    completion (`done = sql_count <= limit`) without relying on the post-filter
    row count, which the historical implementation incorrectly conflated with
    end-of-stream. CHARACTER_LIKENESS now uses a registered SQLite scalar
    function (character_face_likeness) so sorting and pagination happen fully
    at the SQL layer, meaning the standard over-fetch probe applies.
    """
    effective_limit = limit + 1 if stream_state is not None else limit

    def _record_sql_count(pics):
        if stream_state is not None:
            stream_state["sql_count"] = len(pics)
            if len(pics) > limit:
                return pics[:limit]
        return pics

    def _empty_result():
        # An empty-result early-return is always end-of-stream for a streaming
        # caller: any subsequent fetch with a larger offset would yield the
        # same empty set.
        if stream_state is not None:
            stream_state["sql_count"] = 0
            stream_state["oneshot"] = True
        return 0 if count_only else []

    def serialize_metadata(pictures):
        result = []
        for pic in pictures:
            pic_d = stored_picture_dict(pic)
            d = {
                field: pic_d.get(field) for field in metadata_fields if field != "tags"
            }
            if "tags" in metadata_fields:
                d["tags"] = [t.tag for t in getattr(pic, "tags", [])]
            if "smart_score" in d:
                d["smartScore"] = d["smart_score"]
            result.append(d)
        return result

    def parse_request_params():
        query_params = {}
        format = None
        if request.query_params:
            format = request.query_params.getlist("format")
            query_params = {
                name: value
                for name, value in request.query_params.items()
                if name not in _SPLAT_RESERVED_NAMES
            }
            if exclude_query_params:
                for key in exclude_query_params:
                    query_params.pop(key, None)
            picture_ids = request.query_params.getlist("id")
            if picture_ids:
                query_params["id"] = picture_ids
            workflows = request.query_params.getlist("workflow")
            if workflows:
                query_params["workflow"] = workflows
            comfyui_models = request.query_params.getlist("comfyui_model")
            if comfyui_models:
                query_params["comfyui_models_filter"] = comfyui_models
            query_params.pop("comfyui_model", None)
            comfyui_loras = request.query_params.getlist("comfyui_lora")
            if comfyui_loras:
                query_params["comfyui_loras_filter"] = comfyui_loras
            query_params.pop("comfyui_lora", None)
            tags = request.query_params.getlist("tag")
            if tags:
                query_params["tags_filter"] = tags
            query_params.pop("tag", None)
            rejected_tags = request.query_params.getlist("rejected_tag")
            if rejected_tags:
                query_params["tags_rejected_filter"] = rejected_tags
            query_params.pop("rejected_tag", None)
            confidence_above = request.query_params.getlist("tag_confidence_above")
            if confidence_above:
                query_params["tags_confidence_above_filter"] = confidence_above
            query_params.pop("tag_confidence_above", None)
            confidence_below = request.query_params.getlist("tag_confidence_below")
            if confidence_below:
                query_params["tags_confidence_below_filter"] = confidence_below
            query_params.pop("tag_confidence_below", None)
            set_ids = request.query_params.getlist("set_ids")
            if set_ids:
                query_params["set_ids"] = set_ids
            character_ids_raw = request.query_params.getlist("character_ids")
            if character_ids_raw:
                query_params["character_ids"] = character_ids_raw
            face_filter_param = request.query_params.get("face_filter")
            if face_filter_param in ("with_face", "without_face"):
                query_params["face_filter"] = face_filter_param
            stack_state_param = request.query_params.get("stack_state")
            if stack_state_param in ("stacked", "unstacked", "unresolved"):
                query_params["stack_state"] = stack_state_param
            else:
                # An unrecognised value must not survive into ``query_params``: it
                # would be splatted into ``Picture.find(**query_params)``, where the
                # ``hasattr`` fallthrough drops any key that is not a column and the
                # filter silently does nothing.
                query_params.pop("stack_state", None)
            impossible_sources_param = request.query_params.getlist(
                "impossible_tag_source"
            )
            if impossible_sources_param:
                query_params["impossible_sources"] = impossible_sources_param
            shared_only_param = request.query_params.get("shared_only")
            if shared_only_param == "true":
                query_params["shared_only"] = True
        return format, query_params

    def _character_id(value):
        if value == "ALL":
            return None
        if value is not None and value != "" and str(value).isdigit():
            try:
                return parse_sqlite_int(value)
            except ValueError:
                logger.warning("Refusing a character_id past SQLite's range")
                raise HTTPException(status_code=400, detail="Invalid character_id")
        return value

    format, query_params = parse_request_params()
    # Read before ``character_id`` is popped: the set-candidate lookup below
    # needs the Scrapheap scope, legacy ``character_id=SCRAPHEAP`` included.
    only_deleted = request_targets_scrapheap(query_params)
    if project_id is not None:
        query_params["project_id"] = project_id
    sort = query_params.pop("sort", sort)
    desc_val = query_params.pop("descending", descending)
    descending = (
        desc_val.lower() == "true" if isinstance(desc_val, str) else bool(desc_val)
    )
    try:
        offset = parse_sqlite_int(query_params.pop("offset", offset))
        limit = parse_sqlite_int(query_params.pop("limit", limit))
    except (TypeError, ValueError):
        logger.warning(
            "Refusing an offset or limit that is not an integer SQLite stores "
            "(offset=%r, limit=%r)",
            request.query_params.get("offset"),
            request.query_params.get("limit"),
        )
        raise HTTPException(status_code=400, detail="Invalid offset or limit")
    character_id = _character_id(query_params.pop("character_id", None))
    unassigned_by = query_params.pop("unassigned_by", None)
    # The Inbox is the unassigned view (no person and no set, both halves)
    # narrowed to what was added within its period. A local, not a
    # ``query_params`` entry: ``Picture.find(**query_params)`` would drop a key
    # that is not a column without a word.
    inbox_period = query_params.pop("inbox", None)
    imported_since = inbox_cutoff(inbox_period)
    if inbox_period is not None:
        character_id = "UNASSIGNED"
        unassigned_by = None
    _character_ids_raw = query_params.pop("character_ids", None)
    character_id_list: list[int] = []
    if _character_ids_raw:
        for _v in (
            _character_ids_raw
            if isinstance(_character_ids_raw, list)
            else [_character_ids_raw]
        ):
            try:
                _cid = parse_sqlite_int(_v)
                if _cid > 0:
                    character_id_list.append(_cid)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid character_ids value %r in /pictures request",
                    _v,
                )
        character_id_list = sorted(set(character_id_list))
    character_mode_raw = query_params.pop("character_mode", "union")
    character_mode = (character_mode_raw or "union").strip().lower()
    if character_mode not in {"union", "intersection", "difference", "xor"}:
        character_mode = "union"
    set_id_raw = query_params.pop("set_id", None)
    set_ids_raw = query_params.pop("set_ids", None)
    set_mode_raw = query_params.pop("set_mode", "union")
    base_set_id_raw = query_params.pop("base_set_id", None)
    reference_character_id = query_params.pop("reference_character_id", None)
    if reference_character_id not in (None, ""):
        try:
            reference_character_id = parse_sqlite_int(reference_character_id)
        except (TypeError, ValueError):
            logger.warning(
                "Refusing reference_character_id %r: not an integer SQLite stores",
                reference_character_id,
            )
            raise HTTPException(
                status_code=400, detail="Invalid reference_character_id"
            )
    # The filters the text search honours too, parsed where it parses them.
    narrowing = parse_picture_filter(server, request)
    query_params.pop("min_score", None)
    min_score = narrowing.min_score
    query_params.pop("max_score", None)
    max_score = narrowing.max_score
    # Popped, not left in ``query_params``: the leftovers are splatted into
    # ``Picture.find(**search)``, whose ``hasattr`` fallthrough silently drops any
    # key that is not a Picture column, so the filter would do nothing at all.
    unscored = is_truthy_flag(query_params.pop("unscored", None))
    favourite_only = is_truthy_flag(query_params.pop("favourite", None))
    smart_score_bucket = query_params.pop("smart_score_bucket", None) or None
    resolution_bucket = query_params.pop("resolution_bucket", None) or None
    project_id_raw = query_params.pop("project_id", None)
    file_path_prefix = query_params.pop("file_path_prefix", None) or None
    face_filter = query_params.pop("face_filter", None)
    stack_state = query_params.pop("stack_state", None)
    impossible_sources = query_params.pop("impossible_sources", None)
    query_params.pop("tag_state", None)
    tag_state = narrowing.tag_state
    unprocessable_ids = narrowing.unprocessable_ids
    if tag_state is not None:
        # Back into ``query_params`` for the ``**query_params`` sites, as the
        # workflow filter below does.
        query_params["tag_state"] = tag_state
        query_params["unprocessable_ids"] = unprocessable_ids
    # Resolved by ``parse_picture_filter``; popped so no raw name reaches the
    # ``Picture.find(**query_params)`` splat.
    for name in ("workflow", "workflow_lora", "workflow_structural_hashes"):
        query_params.pop(name, None)
    workflow_structural_hashes = narrowing.workflow_structural_hashes
    if workflow_structural_hashes is not None:
        # Back into ``query_params`` under the name ``Picture.find`` declares,
        # so every ``**query_params`` site gets it; the branches that name each
        # filter instead pass it explicitly.
        query_params["workflow_structural_hashes"] = workflow_structural_hashes
    query_params.pop("shared_only", None)  # read by shared_picture_ids
    query_params.pop("only_deleted", None)  # read above
    set_mode = normalize_set_mode(set_mode_raw)
    set_filter_ids = collect_set_filter_ids(
        set_id_value=set_id_raw,
        set_ids_values=set_ids_raw if isinstance(set_ids_raw, list) else None,
    )
    if set_mode == "difference" and base_set_id_raw is not None:
        try:
            _base_id = int(base_set_id_raw)
            if _base_id in set_filter_ids:
                set_filter_ids = [_base_id] + [
                    s for s in set_filter_ids if s != _base_id
                ]
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid base_set_id value %r in /pictures request",
                base_set_id_raw,
            )

    # Token scope enforcement: if a picture_set token is active, restrict to that set.
    if scope_set_id is not None:
        set_filter_ids = [scope_set_id]
        set_mode = "union"

    # Token scope enforcement: if a character token is active, restrict to that character.
    if scope_character_id is not None:
        character_id = scope_character_id
        character_id_list = [scope_character_id]
        character_mode = "union"

    # Token scope enforcement: a single-picture share token may only ever
    # resolve that one picture id.  Intersect with any caller-supplied ids so
    # a `picture`-scoped token cannot widen the result to the whole vault.
    if scope_picture_id is not None:
        existing_ids = query_params.get("id")
        if existing_ids:
            query_params["id"] = [
                i for i in existing_ids if str(i) == str(scope_picture_id)
            ]
        else:
            query_params["id"] = [str(scope_picture_id)]

    # One of the person's face clusters, numbered as the cluster view numbers
    # them ("Cluster N" at a threshold). Narrows through ``query_params["id"]``
    # like the scopes above, which every branch reads, the likeness sort
    # included.
    face_cluster_ids = _face_cluster_ids(
        server,
        request,
        character_id,
        query_params.pop("face_cluster", None),
        query_params.pop("face_cluster_threshold", None),
    )
    if face_cluster_ids is not None:
        existing_ids = query_params.get("id")
        query_params["id"] = [
            str(i)
            for i in sorted(face_cluster_ids)
            if existing_ids is None or str(i) in {str(e) for e in existing_ids}
        ]

    # Shared-only filter: restrict to pictures that have an active READ token for the current user.
    shared_id_set = shared_picture_ids(server, request)
    if shared_id_set is not None:
        existing_ids = query_params.get("id")
        # `existing_ids` may already be `[]` -- a real "nothing matches"
        # result from scope_picture_id's own narrowing above -- which is
        # falsy but distinct from "no id filter was ever set" (None).
        # Checking `is not None` preserves that distinction; a plain
        # truthiness check would treat an already-narrowed-to-empty scope
        # as unset and hand back every shared picture, discarding the
        # narrowing. A malformed `id` names no picture, so it is dropped.
        if existing_ids is not None:
            query_params["id"] = [
                i for i in existing_ids if str(i).isdigit() and int(i) in shared_id_set
            ]
        else:
            query_params["id"] = [str(i) for i in shared_id_set]

    def _set_candidate_ids_for_session(session: Session):
        return fetch_set_candidate_ids(
            session,
            set_ids=set_filter_ids,
            set_mode=set_mode,
            deleted_only=only_deleted,
        )

    try:
        sort_mech = (
            SortMechanism.from_string(sort, descending=descending) if sort else None
        )
    except ValueError as ve:
        logger.error(f"Invalid sort mechanism: {sort} - {ve}")
        raise HTTPException(status_code=400, detail="Invalid sort mechanism")

    guest_session_id = getattr(request.state, "guest_session_id", None)
    guest_token_public_id = (
        getattr(request.state, "token_public_id", None) if guest_session_id else None
    )
    # Fallback: rejected-consent guests have no HttpOnly cookie, but may pass
    # the in-memory session ID as a query param so scores are overlaid for the
    # current page session.  Only honoured for READ-scoped tokens.
    if not guest_session_id:
        token_scope = getattr(request.state, "token_scope", None)
        if token_scope is not None and token_scope.scope == "READ":
            qp_sid = request.query_params.get("guest_session_id", "")
            if qp_sid and re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", qp_sid):
                guest_session_id = qp_sid
                guest_token_public_id = getattr(request.state, "token_public_id", None)

    hidden_tags = _get_hidden_tags_from_request(server, request) or None

    pics = []
    if is_scrapheap_view_id(character_id):
        character_id = None
    # The Scrapheap wins over the unassigned view and the Inbox: those read
    # live pictures only, so with ``only_deleted`` they would list pictures
    # outside the heap. The whole heap is listed instead, as the likeness sort
    # (``fetch_smart_score_candidate_ids``), the text search and the stats
    # (``get_picture_stats``) read it.
    if only_deleted and character_id == "UNASSIGNED":
        character_id = None

    # The scrapheap (deleted) view never collapses stacks: every deleted picture
    # is listed individually. A soft-deleted stack member (e.g. a former leader,
    # which normalize_stack_positions sorts behind its live siblings) is therefore
    # always visible and restorable, instead of being hidden because it is no
    # longer at stack_position == 0.
    if only_deleted:
        stack_leaders_only = False

    def fetch_smart_score_candidate_ids(
        session: Session,
        character_id_value,
        deleted_only: bool,
        formats: list[str] | None,
        project_id_value: str | None,
        min_score_value: int | None = None,
        max_score_value: int | None = None,
        unscored_value: bool = False,
        favourite_only_value: bool = False,
        smart_score_bucket_value: str | None = None,
        resolution_bucket_value: str | None = None,
        tags_filter_value: list[str] | None = None,
        tags_rejected_filter_value: list[str] | None = None,
        tags_confidence_above_filter_value: list[str] | None = None,
        tags_confidence_below_filter_value: list[str] | None = None,
        hidden_tags_filter_value: list[str] | None = None,
        comfyui_models_filter_value: list[str] | None = None,
        comfyui_loras_filter_value: list[str] | None = None,
        workflow_structural_hashes_value: list[str] | None = None,
        face_filter_value: str | None = None,
        stack_state_value: str | None = None,
        impossible_sources_value: list[str] | None = None,
        import_source_folder_value: str | None = None,
        imported_since_value: datetime | None = None,
        tag_state_value: str | None = None,
        unprocessable_ids_value: frozenset[int] | None = None,
        file_path_prefix_value: str | None = None,
        reference_folder_id_value: int | None = None,
    ):
        if deleted_only:
            query = select(Picture.id).where(
                Picture.deleted.is_(True),
            )
        elif character_id_value == "UNASSIGNED":
            assignment_project_id = None
            assignment_unassigned_project = False
            if project_id_value == "UNASSIGNED":
                assignment_unassigned_project = True
            elif project_id_value is not None:
                try:
                    assignment_project_id = parse_sqlite_int(project_id_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid project_id_raw value %r for UNASSIGNED assignment scope; treating as global scope",
                        project_id_value,
                    )
            unassigned_conditions = Picture.build_unassigned_conditions(
                enforce_stack_assignment=True,
                assignment_project_id=assignment_project_id,
                assignment_unassigned_project=assignment_unassigned_project,
                unassigned_by=unassigned_by,
            )
            query = select(Picture.id).where(
                *unassigned_conditions,
                Picture.deleted.is_(False),
            )
        elif character_id_value is None or character_id_value == "":
            if (
                project_id_value is None
                and not formats
                and min_score_value is None
                and max_score_value is None
                and not unscored_value
                and not favourite_only_value
                and smart_score_bucket_value is None
                and resolution_bucket_value is None
                and not tags_filter_value
                and not tags_rejected_filter_value
                and not tags_confidence_above_filter_value
                and not tags_confidence_below_filter_value
                and not hidden_tags_filter_value
                and not comfyui_models_filter_value
                and not comfyui_loras_filter_value
                # ``is None``, not falsiness: an empty list is a workflow
                # filter that matches nothing, and treating it as "no filter"
                # would widen this branch to every candidate picture.
                and workflow_structural_hashes_value is None
                and not face_filter_value
                and not stack_state_value
                and not impossible_sources_value
                and not import_source_folder_value
                and imported_since_value is None
                and tag_state_value is None
                # None means every picture, so a folder must build a list.
                and file_path_prefix_value is None
                and reference_folder_id_value is None
            ):
                return None
            query = select(Picture.id).where(
                Picture.deleted.is_(False),
            )
        elif isinstance(character_id_value, int):
            query = (
                select(Picture.id)
                .join(Face, Face.picture_id == Picture.id)
                .where(
                    Face.character_id == character_id_value,
                    Picture.deleted.is_(False),
                )
            )
        else:
            return None

        if project_id_value == "UNASSIGNED":
            query = query.where(project_unassigned_clause(Picture))
        elif project_id_value is not None:
            try:
                query = query.where(
                    project_membership_exists_clause(
                        parse_sqlite_int(project_id_value), Picture
                    )
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid project_id_raw value %r for SMART_SCORE candidate filtering; skipping project filter",
                    project_id_value,
                )
        # The listing's two folder filters, in every branch: the reference
        # folder as the column filter ``Picture.find`` and ``find_unassigned``
        # apply here, and the folder prefix in the PredicateFilter below, on
        # the same children-only default the listing uses.
        if reference_folder_id_value is not None:
            query = query.where(
                Picture.reference_folder_id == reference_folder_id_value
            )

        # Intrinsic-attribute predicates via the shared compiler.  Deleted / project
        # / character scoping is handled above (woven into the candidate branches),
        # so the filter applies no lifecycle clauses here.
        query = PredicateFilter(
            format=formats,
            min_score=min_score_value,
            max_score=max_score_value,
            unscored=unscored_value,
            favourite_only=favourite_only_value,
            smart_score_bucket=smart_score_bucket_value,
            resolution_bucket=resolution_bucket_value,
            tags_filter=tags_filter_value,
            tags_rejected_filter=tags_rejected_filter_value,
            hidden_tags_filter=hidden_tags_filter_value,
            tags_confidence_above_filter=tags_confidence_above_filter_value,
            tags_confidence_below_filter=tags_confidence_below_filter_value,
            comfyui_models_filter=comfyui_models_filter_value,
            comfyui_loras_filter=comfyui_loras_filter_value,
            workflow_structural_hashes=workflow_structural_hashes_value,
            face_filter=face_filter_value,
            stack_state=stack_state_value,
            impossible_sources=impossible_sources_value,
            import_source_folder=import_source_folder_value,
            imported_since=imported_since_value,
            tag_state=tag_state_value,
            unprocessable_ids=unprocessable_ids_value,
            file_path_prefix=file_path_prefix_value,
            apply_deleted_filter=False,
        ).apply(query)

        return list(session.exec(query).all())

    logger.info("Getting pictures with project id = %s", project_id_raw)

    if sort_mech and sort_mech.key == SortMechanism.Keys.CHARACTER_LIKENESS:
        # Counting does not need the likeness reference (it only counts the
        # candidate/character filter), so the reference is required for the
        # listing path only.
        if not reference_character_id and not count_only:
            raise HTTPException(
                status_code=400,
                detail="reference_character_id is required for CHARACTER_LIKENESS sort",
            )
        candidate_ids = server.vault.db.run_immediate_read_task(
            fetch_smart_score_candidate_ids,
            character_id,
            only_deleted,
            format,
            project_id_raw,
            min_score_value=min_score,
            max_score_value=max_score,
            unscored_value=unscored,
            favourite_only_value=favourite_only,
            smart_score_bucket_value=smart_score_bucket,
            resolution_bucket_value=resolution_bucket,
            tags_filter_value=query_params.get("tags_filter"),
            tags_rejected_filter_value=query_params.get("tags_rejected_filter"),
            tags_confidence_above_filter_value=query_params.get(
                "tags_confidence_above_filter"
            ),
            tags_confidence_below_filter_value=query_params.get(
                "tags_confidence_below_filter"
            ),
            hidden_tags_filter_value=hidden_tags,
            comfyui_models_filter_value=query_params.get("comfyui_models_filter"),
            comfyui_loras_filter_value=query_params.get("comfyui_loras_filter"),
            workflow_structural_hashes_value=workflow_structural_hashes,
            face_filter_value=face_filter,
            stack_state_value=stack_state,
            impossible_sources_value=impossible_sources,
            import_source_folder_value=query_params.get("import_source_folder"),
            imported_since_value=imported_since,
            tag_state_value=tag_state,
            unprocessable_ids_value=unprocessable_ids,
            file_path_prefix_value=file_path_prefix,
            reference_folder_id_value=_reference_folder_id(query_params),
        )
        if set_filter_ids:
            set_candidate_ids = server.vault.db.run_immediate_read_task(
                _set_candidate_ids_for_session
            )
            candidate_ids = (
                set_candidate_ids
                if candidate_ids is None
                else set(candidate_ids) & set_candidate_ids
            )
        # The ``id`` filter, as every other sort applies it: the caller's
        # ``id=`` list, narrowed above by a picture-scoped token, the face
        # cluster and shared-only. A filter admits the same rows under every
        # sort; this one only orders them. ``[]`` is a filter that matched
        # nothing, not an absent one, and a malformed id names no picture.
        listed_ids = query_params.get("id")
        if listed_ids is not None:
            listed = set(_storable_picture_ids(listed_ids))
            candidate_ids = (
                listed if candidate_ids is None else set(candidate_ids) & listed
            )
        if candidate_ids is not None and not candidate_ids:
            return _empty_result()
        if count_only:
            return count_pictures_by_character_likeness(
                server,
                character_id,
                candidate_ids=list(candidate_ids)
                if candidate_ids is not None
                else None,
                deleted_only=only_deleted,
                stack_leaders_only=stack_leaders_only,
            )
        pics = find_pictures_by_character_likeness_sql(
            server,
            character_id,
            reference_character_id,
            offset,
            effective_limit,
            descending,
            candidate_ids=list(candidate_ids) if candidate_ids is not None else None,
            deleted_only=only_deleted,
            stack_leaders_only=stack_leaders_only,
        )
        pics = _record_sql_count(pics)
        if return_ids_only:
            return [pic.get("id") for pic in pics if pic.get("id") is not None]
        if stack_leaders_only:
            pics = _enrich_stack_counts(server, pics)
        if only_deleted:
            pics = _enrich_scrapheap_retention(server, pics)
        return pics
    if character_id == "UNASSIGNED":
        # Token scope enforcement (BOLA): this branch bypasses the set/character/
        # picture scope filters applied to the main query path, so constrain it
        # explicitly.  fetch_scope_allowed_picture_ids returns None for an
        # owner/unscoped token and the allowed id set otherwise; fail closed when
        # nothing is in scope rather than falling through to an unfiltered query.
        scope_allowed_ids = fetch_scope_allowed_picture_ids(server, request)
        if scope_allowed_ids is not None:
            requested = query_params.get("id")
            # `requested` may be `[]` -- already narrowed to "nothing matches"
            # by scope_picture_id/shared_only/split above -- which is falsy
            # but distinct from "no id filter was ever set" (None). Checking
            # `is not None` preserves that distinction instead of treating an
            # already-empty scope intersection as unset.
            if requested is not None:
                allowed = [
                    str(i)
                    for i in requested
                    if str(i).isdigit() and int(i) in scope_allowed_ids
                ]
            else:
                allowed = [str(i) for i in scope_allowed_ids]
            if not allowed:
                return _empty_result()
            query_params["id"] = allowed
        unassigned_project_id = None
        unassigned_project_only = False
        if project_id_raw == "UNASSIGNED":
            unassigned_project_only = True
        elif project_id_raw is not None:
            try:
                unassigned_project_id = parse_sqlite_int(project_id_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid project_id_raw value %r for UNASSIGNED query; skipping project filter",
                    project_id_raw,
                )
        unassigned_reference_folder_id = _reference_folder_id(query_params)
        if count_only:
            return server.vault.db.run_immediate_read_task(
                Picture.find_unassigned,
                count_only=True,
                # Counts what the list returns: stack leaders only when the
                # list collapses stacks, so the header matches the grid.
                stack_leaders_only=stack_leaders_only,
                project_id=unassigned_project_id,
                only_unassigned_project=unassigned_project_only,
                unassigned_by=unassigned_by,
                format=format,
                min_score=min_score,
                max_score=max_score,
                unscored=unscored,
                favourite_only=favourite_only,
                smart_score_bucket=smart_score_bucket,
                resolution_bucket=resolution_bucket,
                face_filter=face_filter,
                stack_state=stack_state,
                impossible_sources=impossible_sources,
                file_path_prefix=file_path_prefix,
                reference_folder_id=unassigned_reference_folder_id,
                import_source_folder=query_params.get("import_source_folder"),
                imported_since=imported_since,
                tags_filter=query_params.get("tags_filter") or None,
                tags_rejected_filter=query_params.get("tags_rejected_filter") or None,
                tags_confidence_above_filter=query_params.get(
                    "tags_confidence_above_filter"
                )
                or None,
                tags_confidence_below_filter=query_params.get(
                    "tags_confidence_below_filter"
                )
                or None,
                hidden_tags_filter=hidden_tags,
                comfyui_models_filter=query_params.get("comfyui_models_filter"),
                comfyui_loras_filter=query_params.get("comfyui_loras_filter"),
                workflow_structural_hashes=workflow_structural_hashes,
                tag_state=tag_state,
                unprocessable_ids=unprocessable_ids,
                picture_ids=(
                    _storable_picture_ids(query_params["id"])
                    if query_params.get("id") is not None
                    else None
                ),
            )
        pics = server.vault.db.run_immediate_read_task(
            Picture.find_unassigned,
            sort_mech=sort_mech,
            offset=offset,
            limit=effective_limit,
            format=format,
            metadata_fields=metadata_fields,
            stack_leaders_only=stack_leaders_only,
            min_score=min_score,
            max_score=max_score,
            unscored=unscored,
            favourite_only=favourite_only,
            smart_score_bucket=smart_score_bucket,
            resolution_bucket=resolution_bucket,
            project_id=unassigned_project_id,
            only_unassigned_project=unassigned_project_only,
            unassigned_by=unassigned_by,
            tags_filter=query_params.get("tags_filter") or None,
            tags_rejected_filter=query_params.get("tags_rejected_filter") or None,
            tags_confidence_above_filter=query_params.get(
                "tags_confidence_above_filter"
            )
            or None,
            tags_confidence_below_filter=query_params.get(
                "tags_confidence_below_filter"
            )
            or None,
            hidden_tags_filter=hidden_tags,
            face_filter=face_filter,
            stack_state=stack_state,
            impossible_sources=impossible_sources,
            file_path_prefix=file_path_prefix,
            reference_folder_id=unassigned_reference_folder_id,
            import_source_folder=query_params.get("import_source_folder"),
            imported_since=imported_since,
            comfyui_models_filter=query_params.get("comfyui_models_filter"),
            comfyui_loras_filter=query_params.get("comfyui_loras_filter"),
            workflow_structural_hashes=workflow_structural_hashes,
            tag_state=tag_state,
            unprocessable_ids=unprocessable_ids,
            picture_ids=(
                _storable_picture_ids(query_params["id"])
                if query_params.get("id") is not None
                else None
            ),
            guest_session_id=guest_session_id,
            guest_token_public_id=guest_token_public_id,
        )
        pics = _record_sql_count(pics)
    elif only_deleted:
        if count_only:
            return server.vault.db.run_immediate_read_task(
                Picture.find,
                count_only=True,
                only_deleted=True,
                include_unimported=True,
                stack_leaders_only=stack_leaders_only,
                format=format,
                min_score=min_score,
                max_score=max_score,
                unscored=unscored,
                favourite_only=favourite_only,
                smart_score_bucket=smart_score_bucket,
                resolution_bucket=resolution_bucket,
                face_filter=face_filter,
                stack_state=stack_state,
                impossible_sources=impossible_sources,
                file_path_prefix=file_path_prefix,
                hidden_tags_filter=hidden_tags,
                guest_session_id=guest_session_id,
                guest_token_public_id=guest_token_public_id,
                **query_params,
            )
        pics = server.vault.db.run_immediate_read_task(
            Picture.find,
            sort_mech=sort_mech,
            offset=offset,
            limit=effective_limit,
            select_fields=metadata_fields,
            format=format,
            only_deleted=True,
            include_unimported=True,
            stack_leaders_only=stack_leaders_only,
            min_score=min_score,
            max_score=max_score,
            unscored=unscored,
            favourite_only=favourite_only,
            smart_score_bucket=smart_score_bucket,
            resolution_bucket=resolution_bucket,
            face_filter=face_filter,
            stack_state=stack_state,
            impossible_sources=impossible_sources,
            file_path_prefix=file_path_prefix,
            hidden_tags_filter=hidden_tags,
            guest_session_id=guest_session_id,
            guest_token_public_id=guest_token_public_id,
            **query_params,
        )
        pics = _record_sql_count(pics)
    else:
        if set_filter_ids:
            set_candidate_ids = server.vault.db.run_immediate_read_task(
                _set_candidate_ids_for_session
            )
            if not set_candidate_ids:
                return _empty_result()
            existing_ids = query_params.get("id")
            # `existing_ids` may already be `[]` -- a real "nothing matches
            # this token's scope" result computed by scope_picture_id/
            # shared_only/split above -- which is falsy but distinct from "no
            # id filter was ever set" (None). `is not None` preserves that
            # distinction; a plain truthiness check would treat the
            # already-empty scope intersection as unset and hand back every
            # picture in the set, discarding the scope narrowing.
            if existing_ids is not None:
                # existing_ids may be a list of strings (query params, or ids
                # populated by an earlier filter such as scope_picture_id,
                # shared_only, or split -- all of which stringify their ids),
                # while set_candidate_ids is always set[int]. Intersecting the
                # two directly with mismatched types silently yields an empty
                # set every time, which is why a scoped token's ?split= query
                # came back empty instead of narrowed. Normalize to int first.
                existing_id_set = {int(i) for i in existing_ids if str(i).isdigit()}
                query_params["id"] = list(existing_id_set & set_candidate_ids)
            else:
                query_params["id"] = list(set_candidate_ids)

        if character_id is not None and character_id != "":

            def get_picture_ids_for_character(session, character_id):
                faces = session.exec(
                    select(Face).where(Face.character_id == character_id)
                ).all()
                return list({face.picture_id for face in faces})

            picture_ids = server.vault.db.run_immediate_read_task(
                get_picture_ids_for_character, character_id
            )
            if not picture_ids:
                return _empty_result()

            # When a project filter is also present, restrict to pictures that
            # are members of that project so pictures removed from a project no
            # longer appear in its character grid view.
            if project_id_raw is not None:
                if project_id_raw == "UNASSIGNED":

                    def _get_project_unassigned_ids(session, ids):
                        # A character's pictures scale with the library, so
                        # they are read in chunks under SQLite's
                        # bound-parameter limit; the ids are unique.
                        return [
                            pic_id
                            for id_chunk in chunked(ids)
                            for pic_id in session.exec(
                                select(Picture.id).where(
                                    Picture.id.in_(id_chunk),
                                    project_unassigned_clause(Picture),
                                )
                            ).all()
                        ]

                    picture_ids = server.vault.db.run_immediate_read_task(
                        _get_project_unassigned_ids, picture_ids
                    )
                else:
                    try:
                        proj_id_int = parse_sqlite_int(project_id_raw)
                    except (TypeError, ValueError):
                        proj_id_int = None
                    if proj_id_int is not None:

                        def _get_project_member_ids(session, ids, pid):
                            # A character's pictures scale with the library,
                            # so they are read in chunks under SQLite's
                            # bound-parameter limit; the ids are unique.
                            return [
                                pic_id
                                for id_chunk in chunked(ids)
                                for pic_id in session.exec(
                                    select(PictureProjectMember.picture_id).where(
                                        PictureProjectMember.picture_id.in_(id_chunk),
                                        PictureProjectMember.project_id == pid,
                                    )
                                ).all()
                            ]

                        picture_ids = server.vault.db.run_immediate_read_task(
                            _get_project_member_ids, picture_ids, proj_id_int
                        )

            if not picture_ids:
                return _empty_result()
            existing_ids = query_params.get("id")
            # `existing_ids` may already be `[]` -- a real "nothing matches
            # this token's scope" result from scope_picture_id/shared_only/
            # split above -- which is falsy but distinct from "no id filter
            # was ever set" (None). `is not None` preserves that distinction;
            # a plain truthiness check would treat the already-empty scope
            # intersection as unset and hand back every picture for this
            # character, discarding the scope narrowing.
            if existing_ids is not None:
                existing_id_set = {int(i) for i in existing_ids if str(i).isdigit()}
                picture_ids = [i for i in picture_ids if int(i) in existing_id_set]
                if not picture_ids:
                    return _empty_result()
            query_params["id"] = picture_ids
        elif character_id_list:

            def get_picture_ids_for_characters(session, ids, mode):
                # Fetch picture_ids keyed by character for all chars. The
                # caller's list has no cap: read in chunks under SQLite's
                # bound-parameter limit. Each row names its own person; the
                # modes below combine the merged rows in Python.
                rows = [
                    row
                    for id_chunk in chunked(ids)
                    for row in session.exec(
                        select(Face.character_id, Face.picture_id).where(
                            Face.character_id.in_(id_chunk)
                        )
                    ).all()
                ]
                members_by_char: dict[int, set[int]] = {cid: set() for cid in ids}
                for cid, pid in rows:
                    members_by_char.setdefault(int(cid), set()).add(int(pid))
                if mode == "intersection":
                    result: set[int] | None = None
                    for cid in ids:
                        current = members_by_char.get(cid, set())
                        result = set(current) if result is None else result & current
                    return list(result or set())
                if mode == "difference":
                    first = members_by_char.get(ids[0], set())
                    rest: set[int] = set()
                    for cid in ids[1:]:
                        rest |= members_by_char.get(cid, set())
                    return list(first - rest)
                if mode == "xor":
                    xor_u: set[int] = set()
                    for cid in ids:
                        xor_u |= members_by_char.get(cid, set())
                    xor_i: set[int] | None = None
                    for cid in ids:
                        cur = members_by_char.get(cid, set())
                        xor_i = set(cur) if xor_i is None else xor_i & cur
                    return list(xor_u - (xor_i or set()))
                # union
                union: set[int] = set()
                for cid in ids:
                    union |= members_by_char.get(cid, set())
                return list(union)

            picture_ids = server.vault.db.run_immediate_read_task(
                get_picture_ids_for_characters, character_id_list, character_mode
            )
            if not picture_ids:
                return _empty_result()

            if project_id_raw is not None:
                if project_id_raw == "UNASSIGNED":

                    def _get_project_unassigned_ids_multi(session, ids):
                        # A character's pictures scale with the library, so
                        # they are read in chunks under SQLite's
                        # bound-parameter limit; the ids are unique.
                        return [
                            pic_id
                            for id_chunk in chunked(ids)
                            for pic_id in session.exec(
                                select(Picture.id).where(
                                    Picture.id.in_(id_chunk),
                                    project_unassigned_clause(Picture),
                                )
                            ).all()
                        ]

                    picture_ids = server.vault.db.run_immediate_read_task(
                        _get_project_unassigned_ids_multi, picture_ids
                    )
                else:
                    try:
                        proj_id_int = parse_sqlite_int(project_id_raw)
                    except (TypeError, ValueError):
                        proj_id_int = None
                    if proj_id_int is not None:

                        def _get_project_member_ids_multi(session, ids, pid):
                            # A character's pictures scale with the library,
                            # so they are read in chunks under SQLite's
                            # bound-parameter limit; the ids are unique.
                            # Sorted, so each chunk is a fixed id range
                            # rather than the set's iteration order.
                            return [
                                pic_id
                                for id_chunk in chunked(sorted(ids))
                                for pic_id in session.exec(
                                    select(PictureProjectMember.picture_id).where(
                                        PictureProjectMember.picture_id.in_(id_chunk),
                                        PictureProjectMember.project_id == pid,
                                    )
                                ).all()
                            ]

                        picture_ids = server.vault.db.run_immediate_read_task(
                            _get_project_member_ids_multi, picture_ids, proj_id_int
                        )

            if not picture_ids:
                return _empty_result()
            existing_ids = query_params.get("id")
            # See the comment on the analogous check in the character_id
            # branch above: `existing_ids` may already be `[]` (a real
            # "nothing matches this token's scope" result), which is falsy
            # but distinct from "no id filter was ever set" (None).
            if existing_ids is not None:
                existing_id_set = {int(i) for i in existing_ids if str(i).isdigit()}
                picture_ids = [i for i in picture_ids if int(i) in existing_id_set]
                if not picture_ids:
                    return _empty_result()
            query_params["id"] = picture_ids
        elif project_id_raw is not None:
            # Project filter only applies when not already filtering by character/set.
            # "UNASSIGNED" means pictures with no project (project_id IS NULL).
            # A numeric value filters to that specific project.
            if project_id_raw == "UNASSIGNED":

                def get_unassigned_project_ids(session):
                    from pixlstash.db_models.picture import Picture as Pic

                    rows = session.exec(
                        select(Pic.id).where(
                            project_unassigned_clause(Pic),
                            Pic.deleted.is_(False),
                        )
                    ).all()
                    return list(rows)

                project_pic_ids = server.vault.db.run_immediate_read_task(
                    get_unassigned_project_ids
                )
                if not project_pic_ids:
                    return _empty_result()
                existing_ids = query_params.get("id")
                # See the comment on the analogous check in the set_filter_ids
                # branch above: `existing_ids` may already be `[]` (a real
                # "nothing matches this token's scope" result), which is
                # falsy but distinct from "no id filter was ever set" (None).
                if existing_ids is not None:
                    # existing_ids may be a list of strings (query params, or
                    # ids populated by an earlier filter such as
                    # scope_picture_id, shared_only, or split -- all of which
                    # stringify their ids), while project_pic_ids is always
                    # list[int] (from select(Pic.id)). Intersecting the two
                    # directly with mismatched types silently yields an empty
                    # set every time, so `?id=5&project_id=UNASSIGNED` returned
                    # nothing instead of the matching picture. Normalize to int
                    # first, mirroring the set_filter_ids / character branches.
                    existing_id_set = {int(i) for i in existing_ids if str(i).isdigit()}
                    query_params["id"] = list(existing_id_set & set(project_pic_ids))
                else:
                    query_params["id"] = project_pic_ids
            else:
                try:
                    query_params["project_id"] = parse_sqlite_int(project_id_raw)
                except (TypeError, ValueError):
                    # If project_id_raw is not a valid integer, skip applying a project filter.
                    logger.warning(
                        "Invalid project_id_raw value %r; skipping project filter",
                        project_id_raw,
                    )

        if count_only:
            return server.vault.db.run_immediate_read_task(
                Picture.find,
                count_only=True,
                stack_leaders_only=stack_leaders_only,
                format=format,
                include_unimported=True,
                min_score=min_score,
                max_score=max_score,
                unscored=unscored,
                favourite_only=favourite_only,
                smart_score_bucket=smart_score_bucket,
                resolution_bucket=resolution_bucket,
                file_path_prefix=file_path_prefix,
                face_filter=face_filter,
                stack_state=stack_state,
                impossible_sources=impossible_sources,
                hidden_tags_filter=hidden_tags,
                **query_params,
            )
        pics = server.vault.db.run_immediate_read_task(
            Picture.find,
            sort_mech=sort_mech,
            offset=offset,
            limit=effective_limit,
            select_fields=metadata_fields,
            format=format,
            include_unimported=True,
            stack_leaders_only=stack_leaders_only,
            min_score=min_score,
            max_score=max_score,
            unscored=unscored,
            favourite_only=favourite_only,
            smart_score_bucket=smart_score_bucket,
            resolution_bucket=resolution_bucket,
            file_path_prefix=file_path_prefix,
            face_filter=face_filter,
            stack_state=stack_state,
            impossible_sources=impossible_sources,
            hidden_tags_filter=hidden_tags,
            guest_session_id=guest_session_id,
            guest_token_public_id=guest_token_public_id,
            **query_params,
        )
        pics = _record_sql_count(pics)
    if return_ids_only:
        return [pic.id for pic in pics]
    result = serialize_metadata(pics)
    if sort_mech and sort_mech.key == SortMechanism.Keys.TEXT_CONTENT and result:
        for d in result:
            ts = d.get("text_score")
            d["text_score"] = round(ts, 3) if ts is not None and ts >= 0 else None
    if stack_leaders_only:
        result = _enrich_stack_counts(server, result)
    if only_deleted:
        # Scrapheap view: hand the client a server-computed deadline and the
        # protected-exempt flag so the countdown and the badge never re-derive
        # the retention grace maths client-side.
        result = _enrich_scrapheap_retention(server, result)

    if guest_session_id and result:
        picture_ids_set = {d["id"] for d in result if "id" in d}

        def fetch_guest_scores(session: Session) -> dict[int, int]:
            # Fetch all scores for this session; filter to the current page in
            # Python.  Avoids .in_() on sa_column-defined fields which can
            # silently produce no rows in SQLModel.
            stmt = select(GuestScore).where(GuestScore.session_id == guest_session_id)
            if guest_token_public_id is not None:
                stmt = stmt.where(GuestScore.token_public_id == guest_token_public_id)
            rows = session.exec(stmt).all()
            return {row.picture_id: row.score for row in rows}

        try:
            guest_scores: dict[int, int] = server.vault.db.run_immediate_read_task(
                fetch_guest_scores
            )
        except Exception:
            logger.exception("[guest-scores] Failed to fetch guest scores for overlay")
            guest_scores = {}

        if guest_scores:
            for d in result:
                pic_id = d.get("id")
                if pic_id in guest_scores and pic_id in picture_ids_set:
                    d["score"] = guest_scores[pic_id]
    return result


def rows_of_listed_fields(rows: list[dict], fields: list[str]) -> list[dict]:
    """Listing rows as :class:`GridPicture` answers them, cut to ``fields``.

    For a ``fields`` comma list: the model alone would answer every key it
    declares, the ones not asked for as null, so a two-field read cost most of
    a ``fields=grid`` one. Each row is still validated by the model first, so
    a name it does not declare (``project_id``, §16.6) stays out however it is
    asked for, and a stored value its column does not declare is still null.
    A key the server adds to a row (``stack_count``, ``smartScore``) is
    answered only when it is asked for too.
    """
    include = set(fields)
    return [
        GridPicture.model_validate(row).model_dump(mode="json", include=include)
        for row in rows
    ]


# The response header that says which library a listing's picture ids are of
# (#1912). An id means nothing in another library, so what keeps one where it
# may be read back somewhere else keeps this beside it: the ComfyUI-PixlStash
# picture picker lists through this route and writes it into its loader.
LIBRARY_HEADER = "X-PixlStash-Library"


def register_routes(router, server):
    @router.get(
        "/pictures",
        summary="List pictures",
        description=(
            "Lists pictures with filtering, sort, pagination, and optional grid "
            "field projection. For the owner, the `X-PixlStash-Library` "
            "response header names the library the ids are of."
        ),
        response_model=list[GridPicture],
        responses={400: {"description": "Invalid sort mechanism."}},
    )
    def list_pictures(
        request: Request,
        response: Response,
        sort: str = Query(
            None,
            description="Sort mechanism. One of: DATE, IMPORTED_AT, SCORE, "
            "SMART_SCORE, IMAGE_SIZE, TEXT_CONTENT, CHARACTER_LIKENESS. "
            "Omit for natural (id) order.",
            examples=["DATE"],
        ),
        descending: bool = Query(
            True, description="Sort direction; true = newest/highest first."
        ),
        offset: Annotated[
            SqliteInt, Query(description="Number of rows to skip (pagination).")
        ] = 0,
        limit: Annotated[
            SqliteInt, Query(description="Maximum rows to return.", examples=[200])
        ] = sys.maxsize,
        fields: str = Query(
            None,
            description="Field projection: 'grid' for the minimal grid set, a "
            "comma-separated field list (each row then carries only those of "
            "the listed fields this model declares), or omit for full "
            "metadata.",
            examples=["grid"],
        ),
        project_id: str | None = Query(
            None,
            description="Filter by project id or 'UNASSIGNED'",
            examples=["UNASSIGNED"],
        ),
        only_deleted: bool = Query(
            False,
            description="Return only deleted (scrapheap) pictures. "
            "Replaces the deprecated character_id=SCRAPHEAP.",
        ),
        filters: PictureListFilters = Depends(),
    ):
        enforce_list_id_cap(request)
        # The owner only (ComfyUI's link key is the owner here): a share link
        # has no use for it, and two links are not told they are one library.
        library_uuid = getattr(server.vault, "library_uuid", None)
        if library_uuid and server.auth.is_unscoped_owner_request(request):
            response.headers[LIBRARY_HEADER] = library_uuid
        if fields == "grid":
            metadata_fields = list(Picture.grid_fields())
        elif fields:
            metadata_fields = [f.strip() for f in fields.split(",") if f.strip()]
        else:
            metadata_fields = Picture.metadata_fields()
        token_scope = getattr(request.state, "token_scope", None)
        scope_set_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture_set"
            else None
        )
        # For project-scoped tokens, force the project filter regardless of what
        # the caller passes.  For character-scoped tokens, force the character
        # filter by injecting it as a query-param override via project_id /
        # character_id below.
        if token_scope is not None and token_scope.resource_type == "project":
            project_id = str(token_scope.resource_id)
        scope_character_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "character"
            else None
        )
        scope_picture_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture"
            else None
        )
        pictures = select_pictures_for_listing(
            server=server,
            request=request,
            sort=sort,
            descending=descending,
            offset=offset,
            limit=limit,
            metadata_fields=metadata_fields,
            return_ids_only=False,
            stack_leaders_only=(fields == "grid"),
            project_id=project_id,
            scope_set_id=scope_set_id,
            scope_character_id=scope_character_id,
            scope_picture_id=scope_picture_id,
        )
        if fields and fields != "grid":
            # Returned as is, the injected response's headers are not merged
            # in: they are handed over here.
            return JSONResponse(
                rows_of_listed_fields(pictures, metadata_fields),
                headers=response.headers,
            )
        return pictures

    @router.get(
        "/pictures/stream",
        summary="Stream pictures in batches",
        description=(
            "Fetches a single batch of pictures and reports completion via an "
            "explicit `done` flag derived from the underlying SQL row count "
            "(not the post-filter row count). Callers paginate by passing the "
            "returned `next_offset` until `done` is true."
        ),
        response_model=StreamPicturesResponse,
        responses={
            400: {"description": "Invalid sort mechanism."},
        },
    )
    def stream_pictures(
        request: Request,
        sort: str = Query(
            None,
            description="Sort mechanism (see /pictures). Omit for natural order.",
            examples=["DATE"],
        ),
        descending: bool = Query(
            True, description="Sort direction; true = newest/highest first."
        ),
        offset: int = Query(
            0,
            ge=0,
            le=SQLITE_INT_MAX,
            description="Row offset for this batch (use next_offset).",
        ),
        batch_limit: int = Query(
            1000,
            ge=1,
            le=5000,
            description="Rows per batch (1-5000).",
            examples=[1000],
        ),
        fields: str = Query(
            None,
            description="Field projection: 'grid', a comma list (each row then "
            "carries only those of the listed fields the grid model declares), "
            "or omit.",
            examples=["grid"],
        ),
        grid_lite: bool = Query(
            False,
            description=(
                "When `fields=grid`, omit high-cardinality string fields "
                "such as `file_path` to reduce payload size for streaming."
            ),
        ),
        stack_leaders_only: bool = Query(
            False, description="Collapse stacks to their leader only."
        ),
        project_id: str | None = Query(
            None,
            description="Filter by project id or 'UNASSIGNED'",
            examples=["UNASSIGNED"],
        ),
        only_deleted: bool = Query(
            False,
            description="Return only deleted (scrapheap) pictures. "
            "Replaces the deprecated character_id=SCRAPHEAP.",
        ),
        filters: PictureListFilters = Depends(),
    ):
        enforce_list_id_cap(request)
        if fields == "grid":
            metadata_fields = list(Picture.grid_fields())
            if grid_lite:
                metadata_fields = [f for f in metadata_fields if f != "file_path"]
        elif fields:
            metadata_fields = [f.strip() for f in fields.split(",") if f.strip()]
        else:
            metadata_fields = Picture.metadata_fields()
        token_scope = getattr(request.state, "token_scope", None)
        scope_set_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture_set"
            else None
        )
        if token_scope is not None and token_scope.resource_type == "project":
            project_id = str(token_scope.resource_id)
        scope_character_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "character"
            else None
        )
        scope_picture_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture"
            else None
        )
        # Mirror the /pictures endpoint: fields=grid implies stack_leaders_only.
        effective_stack_leaders_only = stack_leaders_only or (fields == "grid")
        stream_state: dict = {}
        pictures = select_pictures_for_listing(
            server=server,
            request=request,
            sort=sort,
            descending=descending,
            offset=offset,
            limit=batch_limit,
            metadata_fields=metadata_fields,
            return_ids_only=False,
            stack_leaders_only=effective_stack_leaders_only,
            project_id=project_id,
            scope_set_id=scope_set_id,
            scope_character_id=scope_character_id,
            scope_picture_id=scope_picture_id,
            stream_state=stream_state,
        )
        sql_count = int(stream_state.get("sql_count", 0))
        oneshot = bool(stream_state.get("oneshot", False))
        # `done` is decided purely on the pre-post-filter SQL row count. This
        # is the crucial property that prevents the historical bug where
        # post-filter shrinkage (hidden tags, stack dedup) was misread as
        # end-of-stream.
        done = oneshot or sql_count <= batch_limit
        next_offset = offset + min(batch_limit, sql_count)
        if fields and fields != "grid":
            return JSONResponse(
                {
                    "pictures": rows_of_listed_fields(pictures, metadata_fields),
                    "done": done,
                    "next_offset": next_offset,
                }
            )
        return {
            "pictures": pictures,
            "done": done,
            "next_offset": next_offset,
        }

    @router.get(
        "/pictures/count",
        summary="Total picture count for a listing filter",
        description=(
            "Returns the total number of pictures matching the same filter "
            "set used by `/pictures` and `/pictures/stream`."
        ),
        response_model=PictureCountResponse,
    )
    def count_pictures(
        request: Request,
        sort: str = Query(
            None,
            description="Sort mechanism; accepted for parity with /pictures. "
            "CHARACTER_LIKENESS counts via its candidate/character filter "
            "(no reference required).",
            examples=["DATE"],
        ),
        descending: bool = Query(
            True, description="Unused for counting; accepted for parity with /pictures."
        ),
        fields: str = Query(
            None, description="Unused for counting; accepted for parity."
        ),
        stack_leaders_only: bool = Query(
            False, description="Count stacks as one (their leader)."
        ),
        project_id: str | None = Query(
            None,
            description="Filter by project id or 'UNASSIGNED'",
            examples=["UNASSIGNED"],
        ),
        only_deleted: bool = Query(
            False,
            description="Return only deleted (scrapheap) pictures. "
            "Replaces the deprecated character_id=SCRAPHEAP.",
        ),
        filters: PictureListFilters = Depends(),
    ):
        enforce_list_id_cap(request)
        token_scope = getattr(request.state, "token_scope", None)
        scope_set_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture_set"
            else None
        )
        if token_scope is not None and token_scope.resource_type == "project":
            project_id = str(token_scope.resource_id)
        scope_character_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "character"
            else None
        )
        scope_picture_id = (
            token_scope.resource_id
            if token_scope is not None and token_scope.resource_type == "picture"
            else None
        )
        # Use count_only=True to run a fast SELECT COUNT(*) rather than fetching all rows.
        # The count may be a small over-estimate for deployments with hidden-tag post-filtering,
        count = select_pictures_for_listing(
            server=server,
            request=request,
            sort=sort,
            descending=descending,
            offset=0,
            limit=sys.maxsize,
            metadata_fields=[],
            count_only=True,
            stack_leaders_only=stack_leaders_only,
            project_id=project_id,
            scope_set_id=scope_set_id,
            scope_character_id=scope_character_id,
            scope_picture_id=scope_picture_id,
        )
        return {"count": count}
