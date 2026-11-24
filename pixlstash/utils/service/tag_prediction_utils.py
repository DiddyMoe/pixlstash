from typing import Iterable

from sqlmodel import Session, select

from pixlstash.db_models.picture import Picture
from pixlstash.db_models.tag import (
    DEFAULT_SMART_SCORE_PENALIZED_TAGS,
    TAG_SENTINEL_LIKE_PATTERN,
    TAG_SENTINEL_ESCAPE_CHAR,
    Tag,
)
from pixlstash.db_models.tag_prediction import (
    PLUGIN_VERSION_SEPARATOR,
    TagPrediction,
)
from pixlstash.utils.quality.tag_exclusion import is_excluded, session_tag_exclusions
from pixlstash.utils.sql_chunking import chunked

PENALISED_TAG_SET = {t.strip().lower() for t in DEFAULT_SMART_SCORE_PENALIZED_TAGS}


def recompute_anomaly_tag_uncertainty(session: Session, picture_id: int) -> None:
    """Recompute and persist ``anomaly_tag_uncertainty`` on a Picture.

    The score is computed dynamically from the model's raw confidence scores and
    the *current* ``Tag`` rows - it does not rely on ``TagPrediction.status`` so
    it stays correct whether tags were changed via the prediction workflow or
    edited directly.

    For each anomaly tag the model has scored:

    - Tag currently applied: score = ``1 - confidence``
      (human says the defect is present; the model was less than fully confident)
    - Tag not currently applied: score = ``confidence``
      (model says the defect is probably there; human hasn't accepted it)

    A higher value means a stronger disagreement between the model score and the
    current human label on at least one anomaly tag.

    A tag the owner removed from tagging is left out, so rows still held on a
    locked picture do not keep it in the review queue.

    Call this **after** any ``session.flush()`` that modifies ``Tag`` rows, and
    **before** ``session.commit()``, so the read sees the latest state. Thin
    wrapper over :func:`recompute_anomaly_tag_uncertainties`, so the single and
    batched paths can never compute different values.
    """
    recompute_anomaly_tag_uncertainties(session, (picture_id,))


def recompute_anomaly_tag_uncertainties(
    session: Session, picture_ids: Iterable[int]
) -> None:
    """Batched :func:`recompute_anomaly_tag_uncertainty`: two reads per id chunk.

    A loop of single recomputes interleaves a read with a write per picture, so
    every read autoflushes the previous picture's write, and every flush pays for
    the flushed pictures' metadata hashes. Restoring a library-sized undo that way
    cost about a flush per picture. Here every read happens before the first
    write, so the writes reach the database together at the caller's next flush.

    Args:
        session: Active DB session; the caller flushes and commits.
        picture_ids: Pictures to recompute; duplicates and ``None`` are ignored,
            and an id with no picture row is skipped.
    """
    ids = sorted({int(pid) for pid in picture_ids if pid is not None})
    predictions: dict[int, list[tuple[str, float]]] = {}
    applied_tags: dict[int, set[str]] = {}
    for batch in chunked(ids):
        # Built-in tagger rows only (see
        # :func:`~pixlstash.db_models.tag_prediction.feeds_anomaly_score`): raw
        # confidences are not comparable across models, so letting a tagger
        # plugin's rows in here would move the smart score with no user action.
        for pid, tag, confidence in session.exec(
            select(
                TagPrediction.picture_id, TagPrediction.tag, TagPrediction.confidence
            ).where(
                TagPrediction.picture_id.in_(batch),
                ~TagPrediction.model_version.contains(PLUGIN_VERSION_SEPARATOR),
            )
        ).all():
            predictions.setdefault(int(pid), []).append((tag, confidence))
        for pid, tag in session.exec(
            select(Tag.picture_id, Tag.tag).where(
                Tag.picture_id.in_(batch),
                Tag.tag.is_not(None),
                ~Tag.tag.like(
                    TAG_SENTINEL_LIKE_PATTERN, escape=TAG_SENTINEL_ESCAPE_CHAR
                ),
            )
        ).all():
            if tag is not None:
                applied_tags.setdefault(int(pid), set()).add(tag)

    exclusions = session_tag_exclusions(session)
    # Every read is done, so nothing below needs pending writes flushed first: a
    # picture missing from the identity map has no pending change to see.
    with session.no_autoflush:
        for picture_id in ids:
            pic = session.get(Picture, picture_id)
            if pic is None:
                continue
            pic.anomaly_tag_uncertainty = _anomaly_uncertainty(
                predictions.get(picture_id, ()),
                applied_tags.get(picture_id, set()),
                exclusions,
            )


def _anomaly_uncertainty(
    predictions: Iterable[tuple[str, float]],
    applied_tags: set[str],
    exclusions: frozenset[str],
) -> float:
    """The strongest model-versus-label disagreement over one picture's anomaly tags.

    Args:
        predictions: The picture's built-in ``(tag, confidence)`` pairs.
        applied_tags: The real tags the picture carries.
        exclusions: Tags removed from tagging, which never count.

    Returns:
        The highest disagreement, or ``0.0`` when no anomaly tag is scored.
    """
    scores: list[float] = []
    for tag, confidence in predictions:
        if tag is None or tag.strip().lower() not in PENALISED_TAG_SET:
            continue
        if is_excluded(tag, exclusions):
            continue
        if tag in applied_tags:
            scores.append(1.0 - float(confidence))
        else:
            scores.append(float(confidence))
    return max(scores) if scores else 0.0
