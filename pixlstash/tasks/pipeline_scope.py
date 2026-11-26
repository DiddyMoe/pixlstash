"""The pictures the background pipeline works on: live, and with a file.

Every per-picture stage (tagging, captioning, quality, faces, embeddings,
likeness, thumbnails, orientation, text score, ComfyUI extraction) holds the
Scrapheap out of its finder, and counts its progress over the same rows. The
two sides import this one predicate so that they cannot drift apart: a stage
whose finder skipped the Scrapheap while its progress total counted it read
"N / N" with N the whole library, and a stage whose finder kept the Scrapheap
spent the GPU on pictures the owner had deleted.

A picture restored from the Scrapheap still has its NULL columns, so each
finder picks it up on its next sweep: nothing is lost by holding it out while
it is deleted. See "Which pictures the pipeline works on" in
``docs/backend_architecture.md``.
"""

from pixlstash.db_models import Picture


def pipeline_picture_clauses(*, deleted_index: bool = False) -> tuple:
    """WHERE terms for the pictures a background stage may work on.

    Args:
        deleted_index: Spell the Scrapheap term ``deleted IS 0`` rather than
            ``deleted IS NOT 1``. Both select the same rows (``deleted`` is
            never NULL); the difference is which index the planner takes. A
            probe served by a partial index that carries ``deleted`` as its
            second column (``ix_picture_thumbnail_missing`` and its siblings in
            ``Picture.__table_args__``) needs the equality form to claim that
            column. A probe whose partial index has no ``deleted`` column needs
            the ``IS NOT`` form, because beside ``deleted IS 0`` SQLite serves
            the probe from ``ix_picture_deleted`` and walks every live row.
            A finder and its progress count pass the same value.

    Returns:
        ``(scrapheap_term, Picture.file_path IS NOT NULL)``.
    """
    scrapheap_term = (
        Picture.deleted.is_(False) if deleted_index else Picture.deleted.is_not(True)
    )
    return (scrapheap_term, Picture.file_path.is_not(None))
