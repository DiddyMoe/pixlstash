"""What an import does between placing files in the library and saving rows.

**Files go in before their rows.** Every import publishes a picture's file into
the image root (``ImageUtils.create_picture_from_file`` /
``create_picture_from_bytes``) before the insert of its row commits, so that a
committed row never names a file that is not there. The opposite gap is a file
with no row: the next root scan indexes it as a new picture, without the tags
and captions the import would have given it, and an import that is retried
publishes the file a second time. :class:`PublishedPictures` closes that gap
for a failure the process survives: it remembers each file one import
published and, when the import fails before its insert commits, removes them.
This is the compensating action of a write that spans the filesystem and the
database, which share no transaction: the database rolls back, and the import
undoes its own file writes. A crash between the publish and the commit still
leaves the file; the root scan then indexes it (see
``docs/backend_architecture.md``, Known limitations).

**Generated outputs and the Scrapheap.** :func:`plan_generated_outputs` is the
one lookup-restore-decide step for outputs PixlStash makes itself (plugin runs,
edit copies, character sheets, ComfyUI generations). An output whose content a
picture already holds maps to that picture, brought back out of the Scrapheap
when it is there; an output whose match was purged before the restore is a new
picture.
"""

from typing import Collection, Optional

from pixlstash.db_models import Picture
from pixlstash.pixl_logging import get_logger
from pixlstash.services import import_dedup_service, scrapheap_service
from pixlstash.utils.image_processing.image_utils import ImageUtils


logger = get_logger(__name__)


class PublishedPictures:
    """The files one import published whose rows have not committed yet.

    Use as a context manager around the publish and the insert::

        with PublishedPictures(image_root, "Watch-folder import") as published:
            pic = published.add(ImageUtils.create_picture_from_file(...))
            ...
            def insert(session):
                ...
                session.commit()
                published.mark_committed()

    Leaving the block with an exception before :meth:`mark_committed` removes
    every file :meth:`add` recorded, logs what it removed and what it could
    not, and lets the exception propagate. A picture that fails on its own,
    while the rest of the batch goes on, is taken back with :meth:`withdraw`.

    ``mark_committed`` is called inside the writer function, right after the
    commit, so a failure after the commit (a refresh, a later step) never
    removes the file of a saved row.
    """

    def __init__(self, image_root: str, label: str):
        self._image_root = image_root
        self._label = label
        self._pictures: list[Picture] = []
        self._committed = False

    def __enter__(self) -> "PublishedPictures":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and not self._committed and self._pictures:
            self._unpublish_all(exc)
        return False

    def add(self, picture: Picture) -> Picture:
        """Record *picture*, whose file was just published; return it."""
        self._pictures.append(picture)
        return picture

    def mark_committed(self) -> None:
        """The rows of every recorded picture are saved: keep their files."""
        self._committed = True

    def withdraw(self, picture: Picture, exc: BaseException) -> None:
        """Remove the file of one recorded picture that will not be inserted."""
        self._pictures = [pic for pic in self._pictures if pic is not picture]
        if ImageUtils.unpublish_picture(self._image_root, picture):
            logger.warning(
                "%s: removed %s, published for a picture that failed before "
                "it was saved: %s",
                self._label,
                picture.file_path,
                exc,
            )
        else:
            logger.error(
                "%s: could not remove %s, published for a picture that failed "
                "before it was saved: %s",
                self._label,
                picture.file_path,
                exc,
            )

    def withdraw_match(
        self, picture: Picture, match: import_dedup_service.ShaMatch
    ) -> None:
        """Take back one recorded picture whose content *match* already holds.

        The import looked for a match before it published the file, outside
        :func:`import_dedup_service.serialised_import`; the look under the
        lock found *match*, so the picture is not inserted. Its copy goes,
        unless *match* names that same file: publishing identical bytes to a
        path that already holds them keeps the file there, so it is the
        match's own file.
        """
        self._pictures = [pic for pic in self._pictures if pic is not picture]
        if match.file_path == picture.file_path:
            logger.info(
                "%s: not importing %s: picture %d already holds this file.",
                self._label,
                picture.file_path,
                match.id,
            )
            return
        if ImageUtils.unpublish_picture(self._image_root, picture):
            logger.info(
                "%s: not importing %s: picture %d holds the same content; "
                "removed the copy.",
                self._label,
                picture.file_path,
                match.id,
            )
        else:
            logger.error(
                "%s: could not remove %s, a copy of the content of picture %d "
                "that is not imported.",
                self._label,
                picture.file_path,
                match.id,
            )

    def _unpublish_all(self, exc: BaseException) -> None:
        kept = [
            pic.file_path
            for pic in self._pictures
            if not ImageUtils.unpublish_picture(self._image_root, pic)
        ]
        removed = len(self._pictures) - len(kept)
        log = logger.error if kept else logger.warning
        log(
            "%s: the insert of %d new picture(s) did not commit; removed %d "
            "published file(s), %d could not be removed %s: %s",
            self._label,
            len(self._pictures),
            removed,
            len(kept),
            kept,
            exc,
        )
        self._pictures = []


def withdraw_imported_meanwhile(
    db,
    published: PublishedPictures,
    built: list[tuple[Picture, Optional[str]]],
    compared: Collection[int] = (),
) -> list[Optional[import_dedup_service.ShaMatch]]:
    """Find the built pictures an import that held the lock meanwhile added.

    Call under :func:`import_dedup_service.serialised_import`, right before
    the insert. The import looked up its files, then read, hashed and
    published them, all outside the lock, so another import can have added
    the same content since. This is the look every holder of the lock makes,
    on the sampled hash and size each built picture stores, confirmed on the
    full SHA-256 for a picture with a candidate only: the common case under
    the lock is one indexed read. A picture that was matched is withdrawn
    (:meth:`PublishedPictures.withdraw_match`).

    Args:
        db: The vault database the pictures go into.
        published: Where the pictures' files were recorded.
        built: Each picture to insert with its full SHA-256, or ``None`` when
            the import did not compute it; then the published copy is hashed,
            when there is a candidate.
        compared: The ids of the candidates the import's own lookup already
            compared these files with. Their content is not these files', so
            they are no candidates here: a sampled-hash collision with a
            picture the library held before costs no read under the lock.

    Returns:
        Per entry of *built*, in order, the match it was withdrawn for, or
        ``None`` for a picture to insert.
    """
    matches: list[Optional[import_dedup_service.ShaMatch]] = [None] * len(built)
    keyed = [
        index
        for index, (pic, _full) in enumerate(built)
        if pic.pixel_sha and pic.size_bytes is not None
    ]
    if not keyed:
        return matches
    candidates = {
        key: fresh
        for key, rows in db.run_immediate_read_task(
            import_dedup_service.load_match_candidates_in_session,
            sorted({(built[i][0].pixel_sha, built[i][0].size_bytes) for i in keyed}),
        ).items()
        if (fresh := [row for row in rows if row.id not in compared])
    }
    fingerprints: dict[int, import_dedup_service.ContentFingerprint] = {}
    for index in keyed:
        pic, full_sha = built[index]
        if (pic.pixel_sha, pic.size_bytes) not in candidates:
            continue
        if full_sha is None:
            path = ImageUtils.resolve_picture_path(db.image_root, pic.file_path)
            try:
                full_sha = ImageUtils.calculate_full_hash_from_file_path(path)
            except OSError as exc:
                # Inserted: never dropped on evidence that could not be read,
                # as the look before the publish keeps a file it cannot hash.
                logger.warning(
                    "Import: could not hash %s to compare it with the library "
                    "again before saving it: %s",
                    path,
                    exc,
                )
                continue
        fingerprints[index] = (pic.pixel_sha, pic.size_bytes, full_sha)
    if not fingerprints:
        return matches
    live, scrapheaped = import_dedup_service.partition_confirmed_matches(
        candidates, fingerprints.values(), db.image_root
    )
    for index, fingerprint in fingerprints.items():
        match = live.get(fingerprint) or scrapheaped.get(fingerprint)
        if match is not None:
            published.withdraw_match(built[index][0], match)
            matches[index] = match
    return matches


def plan_generated_outputs(
    vault,
    fingerprints: list[import_dedup_service.ContentFingerprint],
    *,
    published: PublishedPictures,
    built: dict[int, Picture],
    label: Optional[str] = None,
) -> import_dedup_service.OutputImportPlan:
    """Look up, confirm and restore the outputs' matches, then decide.

    Call under :func:`import_dedup_service.serialised_import`. The Scrapheap
    does not take that lock, so a match can be binned, restored or purged
    between the lookup and the insert. The restore's own transaction decides:
    every confirmed match, live or scrapheaped, is passed to
    ``scrapheap_service.restore_pictures``, which brings back each one that is
    in the Scrapheap at that moment (a live match binned since the lookup
    included) and reports exactly those. A match it did not restore was live
    at that moment or gone: one that still has a row is the output's picture
    (binned after the restore, it was the owner's act after this import), and
    one with no row was purged, so its output is a new picture.

    The restore commits before the caller inserts the new outputs. When that
    insert fails, the restored pictures stay out of the Scrapheap, one
    undoable History entry, and are the matches of the retried import.

    The caller hashed and published the first output of each content before
    it took the lock, so that no bytes are read or written under it; each
    one the plan does not insert, because a picture holds its content, is
    withdrawn here (:meth:`PublishedPictures.withdraw_match`).

    Args:
        vault: The vault whose database and image root the outputs go into.
        fingerprints: ``(sampled hash, size, full hash)`` per output, in order.
        published: Where the caller recorded the outputs it published.
        built: Output index to its published picture, for the first output
            of each content in the batch.
        label: What is importing, for the log line.

    Returns:
        The plan, keyed by output index (``plan_output_imports``).
    """
    candidates = vault.db.run_immediate_read_task(
        import_dedup_service.load_match_candidates_in_session,
        [(sampled, size) for sampled, size, _full in fingerprints],
    )
    live, scrapheaped = import_dedup_service.partition_confirmed_matches(
        candidates, fingerprints, vault.image_root
    )
    matched_ids = sorted(
        {match.id for match in live.values()}
        | {match.id for match in scrapheaped.values()}
    )
    restored: set[int] = set()
    back: set[int] = set()
    if matched_ids:
        restored = set(
            scrapheap_service.restore_pictures(vault, matched_ids, source="ui")
        )
        not_restored = [pid for pid in matched_ids if pid not in restored]
        still_there = (
            vault.db.run_immediate_read_task(
                scrapheap_service.existing_picture_ids_in_session, not_restored
            )
            if not_restored
            else set()
        )
        back = restored | still_there
        gone = sorted(set(matched_ids) - back)
        if restored or gone:
            logger.info(
                "%s: outputs matched picture(s) %s: restored from the Scrapheap "
                "%s, purged meanwhile (imported as new) %s",
                label or "Generated outputs",
                matched_ids,
                sorted(restored),
                gone,
            )
    live_back = {fp: match for fp, match in live.items() if match.id in back}
    plan = import_dedup_service.plan_output_imports(
        fingerprints,
        live_back,
        scrapheaped,
        back_ids=back,
        restored_ids=restored,
    )
    inserted = set(plan.new_indexes)
    for index, picture in built.items():
        if index in inserted:
            continue
        fingerprint = fingerprints[index]
        published.withdraw_match(
            picture, live_back.get(fingerprint) or scrapheaped[fingerprint]
        )
    return plan
