import json
import math
import sys
import threading

from sqlalchemy.orm import validates
from sqlmodel import (
    Column,
    ForeignKey,
    Index,
    Integer,
    select,
    String,
    SQLModel,
    Field,
    Relationship,
    text,
    UniqueConstraint,
)
from typing import List, Optional, TYPE_CHECKING

from pixlstash.pixl_logging import get_logger

if TYPE_CHECKING:
    from .picture import Picture
    from .character import Character

logger = get_logger(__name__)

# The largest finite float. A coordinate is kept only within it: ``json.loads``
# reads ``NaN``, ``Infinity`` and integers of any size, and none of those is a
# position a box can have. Comparing against it never raises, where
# ``math.isfinite`` raises ``OverflowError`` on an int too large for a float.
_FLOAT_MAX = sys.float_info.max


def is_finite_bbox(value) -> bool:
    """Whether *value* is ``[x1, y1, x2, y2]``: four finite, non-bool numbers."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and -_FLOAT_MAX <= v <= _FLOAT_MAX
            for v in value
        )
    )


# Each stored box :func:`read_stored_bbox` refused, so a reader that runs every
# cycle logs a row once and not once per cycle. It gates only that log line:
# the box is left out either way. Bounded: past the limit the set is cleared,
# and a row still bad is logged once more. Keyed by row id, which another
# library reuses, so a library switch clears it (:func:`clear_refused_bboxes`).
_REFUSED_BBOXES: set = set()
_REFUSED_BBOXES_LOCK = threading.Lock()
_REFUSED_BBOXES_LIMIT = 10_000
# How much of a refused text the log line quotes: an integer can be thousands
# of digits long.
_REFUSED_BBOX_REPR_CHARS = 80


def _shown_bbox_text(raw) -> str:
    """*raw* for a log line: its repr cut to :data:`_REFUSED_BBOX_REPR_CHARS`.

    A value that is not text (an assignment of the wrong type, which only the
    write check sees) is named by its type: the repr of a huge int raises.
    """
    if not isinstance(raw, (str, bytes, bytearray)):
        return f"a {type(raw).__name__}, not text"
    shown = repr(raw)
    if len(shown) > _REFUSED_BBOX_REPR_CHARS:
        shown = f"{shown[:_REFUSED_BBOX_REPR_CHARS]}... ({len(raw)} chars)"
    return shown


def clear_refused_bboxes() -> None:
    """Forget every stored box :func:`read_stored_bbox` has refused.

    Called when the server switches libraries. The memory is keyed by row id
    and text, and the next library numbers its rows from 1 again: a copy of
    the old library holds the same bad row under the same id, and its one log
    line would otherwise never be written for the library it is in.
    """
    with _REFUSED_BBOXES_LOCK:
        _REFUSED_BBOXES.clear()


def read_stored_bbox(
    raw: Optional[str], *, owner: str, row_id, picture_id
) -> Optional[list]:
    """Return a stored ``bbox`` column text as ``[x1, y1, x2, y2]``, or None.

    For a reader of a row loaded from the database. The ``bbox_`` validator
    runs when a value is set, not on load, so a row written before that check
    or by a tool outside the app can hold text the ``bbox`` property raises on
    (not JSON) or returns unusable (``NaN``, ``Infinity``, an integer too large
    for a float, the wrong shape). A reader that met one raised, answered 500,
    wrote ``NaN`` into an export, or failed the same picture every cycle.

    None means there is no box to use: ``null`` (a no-face row has none) and
    no text are normal and not logged; any other text that is not four finite
    numbers is logged once per row and text, with the picture and row ids,
    and is never replaced by a made-up box. The caller leaves the box out.

    Args:
        raw: The ``bbox_`` column text, or None.
        owner: ``"Face"`` or ``"Detection"``, named in the log line.
        row_id: The face or detection id.
        picture_id: The picture the row belongs to.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        # ValueError covers JSONDecodeError and the plain ValueError of an
        # integer past Python's int-string limit (4300 digits). RecursionError
        # is a RuntimeError: the parser raises it on arrays or objects nested
        # past the interpreter's recursion limit ("[" * 100000).
        reason = f"is not JSON ({type(exc).__name__})"
    else:
        if parsed is None:
            return None
        if is_finite_bbox(parsed):
            return parsed
        reason = "is not four finite numbers [x1, y1, x2, y2]"
    key = (owner, row_id, hash(raw))
    with _REFUSED_BBOXES_LOCK:
        if key in _REFUSED_BBOXES:
            return None
        if len(_REFUSED_BBOXES) >= _REFUSED_BBOXES_LIMIT:
            _REFUSED_BBOXES.clear()
        _REFUSED_BBOXES.add(key)
    logger.warning(
        "%s %s on picture %s has a stored bbox that %s: %s; leaving the box out",
        owner,
        row_id,
        picture_id,
        reason,
        _shown_bbox_text(raw),
    )
    return None


def check_bbox_text(owner: str, value: Optional[str]) -> Optional[str]:
    """Return *value* if it is a ``bbox`` column text a reader can use.

    The column, on ``Face`` and ``Detection`` alike, is JSON text:
    ``[x1, y1, x2, y2]`` or ``null``.

    Args:
        owner: ``"Face.bbox_"`` or ``"Detection.bbox_"``, named in the error.
        value: The text about to be stored, or None.

    Raises:
        ValueError: the text is not JSON, or not ``null`` or four finite
            numbers. The message is fixed; the refused text goes to the log,
            cut short, since a caller may pass the message on.
    """
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as exc:
        # ValueError, not only JSONDecodeError: an integer longer than
        # Python's int-string limit (4300 digits) fails the parse with a
        # plain ValueError. RecursionError (a RuntimeError) is the parse of
        # arrays or objects nested past the recursion limit.
        logger.warning(
            "%s refused a bbox text that is not JSON (%s): %s",
            owner,
            type(exc).__name__,
            _shown_bbox_text(value),
        )
        raise ValueError(f"{owner} must be JSON text") from exc
    if parsed is None or is_finite_bbox(parsed):
        return value
    logger.warning(
        "%s refused a bbox text that is not null or four finite numbers "
        "[x1, y1, x2, y2]: %s",
        owner,
        _shown_bbox_text(value),
    )
    raise ValueError(f"{owner} must be null or [x1, y1, x2, y2] of finite numbers")


class Face(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)

    picture_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("picture.id", ondelete="CASCADE"),
            index=True,
        ),
        default=None,
    )
    frame_index: int = Field(default=0)
    face_index: int = Field(default=0)

    character_id: Optional[int] = Field(
        sa_column=Column(Integer, ForeignKey("character.id"), default=None, index=True)
    )
    bbox_: Optional[str] = Field(sa_column=Column("bbox", String, default=None))
    features: Optional[bytes] = None
    # Name of the InsightFace model pack that produced this face's embedding
    # (e.g. "buffalo_l" or "auraface"). Nullable for rows created before the
    # column existed; backfilled to "buffalo_l" by the Alembic migration.
    model_pack: Optional[str] = Field(
        sa_column=Column("model_pack", String, default=None)
    )

    # Relationships
    picture: Optional["Picture"] = Relationship(
        back_populates="faces", sa_relationship_kwargs={"overlaps": "character"}
    )
    character: Optional["Character"] = Relationship(
        back_populates="faces", sa_relationship_kwargs={"overlaps": "picture"}
    )

    __table_args__ = (
        UniqueConstraint("picture_id", "frame_index", "face_index"),
        # "Which characters have a face that carries an embedding?", behind
        # ``GET /characters``. Plain ``ix_face_character_id`` covers every face,
        # embedded or not, and needs a table lookup per row to test ``features``;
        # scoped to the embedded faces the answer comes out of the index alone.
        # Issue #651.
        #
        # ONLY RELY ON THIS FOR ONE-PASS SHAPES: ``GROUP BY character_id`` or
        # ``DISTINCT character_id`` over ``features IS NOT NULL``. For those the
        # planner picks it unconditionally. A PER-CHARACTER probe
        # (``character_id = ?`` / ``IN (...)``) is a coin flip against
        # ``ix_face_character_id``: this database never runs ``ANALYZE``, so with
        # no ``sqlite_stat1`` the two indexes tie on cost, and the tie is
        # broken by index-creation order - which ``metadata.create_all()``
        # iterates from a set, so it varies per process. Making the per-character
        # probe deterministic would mean adding ``features`` itself as a second
        # index column, i.e. duplicating every face embedding into the index.
        # Answer the question in one grouped pass instead; that is also the fix
        # for the N+1 the per-character form implies.
        Index(
            "ix_face_character_features",
            "character_id",
            sqlite_where=text("features IS NOT NULL"),
        ),
    )

    def __init__(self, *args, bbox=None, **kwargs):
        super().__init__(*args, **kwargs)
        if bbox is not None:
            self.bbox = bbox

    @validates("bbox_")
    def _validate_bbox_text(self, key: str, value: Optional[str]) -> Optional[str]:
        """Refuse to store a ``bbox`` column value that the reader cannot use.

        The column is JSON text: ``[x1, y1, x2, y2]`` or ``null`` (the setter
        writes ``null`` for a frame with no face). Every app writer goes
        through the ``bbox`` setter, but ``bbox_`` is itself a public model
        field, so ``Face(bbox_=...)`` and ``face.bbox_ = ...`` skip
        ``json.dumps``. A non-JSON value written that way (``"0,0,10,10"``)
        made every later ``face.bbox`` read raise ``JSONDecodeError`` far from
        the write. SQLAlchemy calls this on every attribute set, including
        the constructor's and the setter's, and not on load from the database.

        Raises:
            ValueError: the text is not JSON, or not ``null`` or a list of
                exactly four finite numbers.
        """
        return check_bbox_text(f"Face.{key}", value)

    @property
    def bbox(self) -> Optional[List[int]]:
        """
        Return the bounding box as a list of integers, or None if not set.
        """
        if self.bbox_:
            return json.loads(self.bbox_)
        return None

    @bbox.setter
    def bbox(self, bbox: List[int]):
        """
        Set the bounding box from a list of integers.
        """
        self.bbox_ = json.dumps(bbox)

    @property
    def width(self) -> float:
        """Return the width of the stored face box, or 0.0 when there is none.

        A stored box that is not four finite numbers counts as none (logged
        once by :func:`read_stored_bbox`): the callers rank faces by area, and
        a bad row must sort last rather than fail the ranking.
        """
        box = self._stored_box()
        return box[2] - box[0] if box is not None else 0.0

    @property
    def height(self) -> float:
        """Return the height of the stored face box, or 0.0 when there is none.

        A stored box that is not four finite numbers counts as none, as in
        :attr:`width`.
        """
        box = self._stored_box()
        return box[3] - box[1] if box is not None else 0.0

    def _stored_box(self) -> Optional[list]:
        """The ``bbox_`` text read through :func:`read_stored_bbox`."""
        return read_stored_bbox(
            self.bbox_, owner="Face", row_id=self.id, picture_id=self.picture_id
        )

    def to_public_dict(self) -> dict:
        """Return the face fields that may be served over the API.

        This is an allowlist, not a dump: it is the projection behind
        ``GET /pictures/{id}/faces`` and ``GET /characters/{id}/faces``, which
        replaced serving the ``faces`` relationship through the generic by-name
        readers (issue #721). Two columns are deliberately **omitted**:

        * ``features`` - the ArcFace embedding. It is biometric data, and the
          generic reader used to hand it out base64-encoded to any token that
          could reach the picture. Nothing on the wire has ever read it: the
          SPA's face-box overlay uses ``frame_index``, ``bbox`` and
          ``character_id`` only.
        * ``model_pack`` - names the InsightFace pack that produced the
          embedding (``buffalo_l`` / ``auraface``). Not biometric in itself, but
          it tells a caller how embeddings obtained elsewhere could be compared
          against these, and no caller reads it (verified across
          ``frontend/src``, ``tests/``, ``pixlstash/`` and ``scripts/``: every
          occurrence is server-side model loading). Omitted by the same
          least-disclosure rule; add it back only with a named consumer.

        ``bbox`` is the ``bbox_`` text column parsed by
        :func:`read_stored_bbox`: the stored list, or ``null`` for a row with no
        box and for a stored box that is not four finite numbers (logged once),
        which would otherwise fail the whole response.

        Returns:
            A JSON-safe dict with ``id``, ``picture_id``, ``character_id``,
            ``frame_index``, ``face_index`` and ``bbox``.
        """
        return {
            "id": self.id,
            "picture_id": self.picture_id,
            "character_id": self.character_id,
            "frame_index": self.frame_index,
            "face_index": self.face_index,
            "bbox": read_stored_bbox(
                self.bbox_, owner="Face", row_id=self.id, picture_id=self.picture_id
            ),
        }

    @classmethod
    def find(cls, session, **filters) -> Optional["Face"]:
        """
        Find faces by picture_id, frame_index, and/or face_index.
        Supports passing a list for picture_id (uses IN_ if so).
        """
        query = select(cls).where(cls.face_index != -1)
        for attr, value in filters.items():
            if hasattr(cls, attr):
                col = getattr(cls, attr)
                if attr == "picture_id" and isinstance(value, list):
                    query = query.where(col.in_(value))
                else:
                    query = query.where(col == value)

        return session.exec(query).all()

    @staticmethod
    def expand_face_bbox(
        bbox: List[int],
        picture_width: int,
        picture_height: int,
        expansion_fraction: float,
    ) -> List[int]:
        """
        Expand the bounding box by a given expansion fraction and align to 64-pixel boundaries.
        Args:
            bbox: List or tuple of [x_min, y_min, x_max, y_max]
            expansion_fraction: Fraction to expand the bbox on each side
        Returns:
            Expanded bbox as [x_min, y_min, x_max, y_max]
        """
        if bbox is None or len(bbox) != 4:
            return bbox
        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min

        def round64(val):
            return int(math.ceil(val / 64.0) * 64)

        new_width = round64(width + width * expansion_fraction)
        new_height = round64(height + height * expansion_fraction)

        width_expansion = new_width - width
        height_expansion = new_height - height

        x_min = max(0, int(round(x_min - width_expansion / 2)))
        x_max = min(picture_width, int(round(x_min + new_width)))
        y_min = max(0, int(round(y_min - height_expansion / 2)))
        y_max = min(picture_height, int(round(y_min + new_height)))

        return [
            x_min,
            y_min,
            x_max,
            y_max,
        ]
