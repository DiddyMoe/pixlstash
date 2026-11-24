"""Tags the owner has removed from tagging: one vocabulary for every producer.

Removing a penalised tag with its X does two things: the name stops counting
towards Smart Score, and nothing may add it to a picture again until the owner
puts it back. The second half has many writers (the tagger, imports, sidecars,
suggestions, undo, restores, stack merges, library moves), so the rule lives
here and each writer asks it.

**Where the rule is read.** The owner's choice is stored in the hub, in
``User.tagger_settings`` under :data:`EXCLUDED_TAGS_KEY`, and applies to every
library. The open vault keeps a snapshot of it (``VaultDatabase``), and every
session the vault opens carries that snapshot in ``session.info`` under
:data:`SESSION_INFO_KEY`. Producers read it from the session that persists the
row, via :func:`session_tag_exclusions`, so a task that computed its tags before
the owner removed one still drops it when it writes.

**What a name covers.** The exact normalised name, plus its
``DEFAULT_TAG_MERGES`` children when it is a canonical parent: a picture tagged
``extra digit`` is scored as ``malformed hand``, so removing the parent must
remove the child as well. Removing a child covers only the child. Anomaly
families are never expanded; their siblings are separate tags.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi import HTTPException
from sqlmodel import select

from pixlstash.db_models.tag import DEFAULT_TAG_MERGES, is_tag_sentinel

EXCLUDED_TAGS_KEY = "excluded_tags"
EXCLUDED_TAGS_GENERATION_KEY = "excluded_tags_generation"
SESSION_INFO_KEY = "tag_exclusions"
MAX_TAG_NAME_LENGTH = 128


class TagExcludedError(HTTPException):
    """A single interactive write named a tag the owner removed from tagging.

    A 409, because the request is well formed and allowed; it conflicts with
    the owner's setting. Bulk and import paths do not raise this: they drop the
    name and report it.
    """

    def __init__(self, tag: str):
        """Build the refusal for *tag*.

        Args:
            tag: The refused tag name, as normalised.
        """
        super().__init__(
            status_code=409,
            detail=(
                f"'{tag}' was removed from tagging. Add it back under "
                "Settings › Smart score & filters to use it again."
            ),
        )
        self.tag = tag


def normalise_tag(name: Any) -> str:
    """Return the comparison form of a tag name: stripped and lowercase."""
    if name is None:
        return ""
    return str(name).strip().lower()


def validate_tag_name(name: Any) -> str:
    """Normalise a tag name a request asked to exclude or re-add.

    Args:
        name: The raw name from the request path or body.

    Returns:
        The normalised name.

    Raises:
        ValueError: The name is empty, too long, or a reserved internal marker.
    """
    clean = normalise_tag(name)
    if not clean:
        raise ValueError("A tag name is required.")
    if len(clean) > MAX_TAG_NAME_LENGTH:
        raise ValueError(
            f"A tag name can be at most {MAX_TAG_NAME_LENGTH} characters long."
        )
    if is_tag_sentinel(clean):
        raise ValueError(f"'{clean}' is reserved and cannot be used as a tag.")
    return clean


def expand_aliases(names: Iterable[Any]) -> frozenset[str]:
    """Return *names* plus the merge children of any canonical parent among them.

    Args:
        names: Tag names, in any case.

    Returns:
        The normalised names and their ``DEFAULT_TAG_MERGES`` children.
    """
    clean = {normalise_tag(n) for n in names}
    clean.discard("")
    expanded = set(clean)
    for child, parent in DEFAULT_TAG_MERGES.items():
        if parent in clean:
            expanded.add(child)
    return frozenset(expanded)


def exclusions_from_tagger_settings(settings: Mapping | None) -> frozenset[str]:
    """Read the excluded names stored in a ``tagger_settings`` mapping.

    The stored list already holds the merge children, so no expansion happens
    here. Anything that is not a list of names reads as nothing excluded.

    Args:
        settings: A parsed ``User.tagger_settings`` value, or ``None``.

    Returns:
        The normalised excluded names.
    """
    if not isinstance(settings, Mapping):
        return frozenset()
    raw = settings.get(EXCLUDED_TAGS_KEY)
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    names = {normalise_tag(n) for n in raw if isinstance(n, str)}
    names.discard("")
    return frozenset(names)


def exclusion_generation(settings: Mapping | None) -> int:
    """Return the counter bumped on every change to the excluded names."""
    if not isinstance(settings, Mapping):
        return 0
    value = settings.get(EXCLUDED_TAGS_GENERATION_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def with_exclusions(settings: Mapping | None, names: Iterable[str]) -> dict:
    """Return a copy of *settings* with the excluded names replaced and the
    generation bumped. Every other key is kept as it was.

    Args:
        settings: The current parsed ``tagger_settings``, or ``None``.
        names: The full new set of excluded names, already expanded.

    Returns:
        The new ``tagger_settings`` mapping.
    """
    updated = dict(settings or {})
    updated[EXCLUDED_TAGS_KEY] = sorted({normalise_tag(n) for n in names} - {""})
    updated[EXCLUDED_TAGS_GENERATION_KEY] = exclusion_generation(settings) + 1
    return updated


def session_tag_exclusions(session) -> frozenset[str]:
    """Return the excluded names stamped on a vault session.

    Args:
        session: A session opened by ``VaultDatabase``.

    Returns:
        The snapshot current when the session was opened.
    """
    return session.info.get(SESSION_INFO_KEY, frozenset())


def is_excluded(name: Any, exclusions: frozenset[str]) -> bool:
    """Return whether *name* is one of *exclusions*."""
    return bool(exclusions) and normalise_tag(name) in exclusions


def excluded_spellings(
    session, column, exclusions: frozenset[str] | None = None
) -> list[str]:
    """Return the stored values of *column* that name an excluded tag.

    Matched in Python, because SQL cannot reproduce :func:`normalise_tag` (its
    ``lower`` and ``trim`` know only ASCII letters and spaces), so a query
    compares the stored column with ``column.in_(...)`` on the result, before
    any OFFSET or LIMIT.

    Args:
        session: A session opened by ``VaultDatabase``.
        column: A tag-name column, such as ``Tag.tag``.
        exclusions: The normalised names to match; the session's when ``None``.

    Returns:
        The distinct stored spellings; empty when nothing is excluded.
    """
    if exclusions is None:
        exclusions = session_tag_exclusions(session)
    if not exclusions:
        return []
    return [
        name
        for name in session.exec(select(column).distinct()).all()
        if is_excluded(name, exclusions)
    ]


def filter_tag_names(names: Iterable[Any], exclusions: frozenset[str]) -> list:
    """Return *names* without the excluded ones, order kept.

    Sentinel values are never excluded: they are internal markers, not tags.

    Args:
        names: Tag names as the producer holds them.
        exclusions: The session's excluded names.

    Returns:
        The permitted names, unchanged in spelling.
    """
    if not exclusions:
        return list(names)
    return [n for n in names if normalise_tag(n) not in exclusions]


def filter_scores(scores: Mapping, exclusions: frozenset[str]) -> dict:
    """Return a ``{tag: value}`` mapping without the excluded tags."""
    if not exclusions:
        return dict(scores)
    return {t: v for t, v in scores.items() if normalise_tag(t) not in exclusions}
