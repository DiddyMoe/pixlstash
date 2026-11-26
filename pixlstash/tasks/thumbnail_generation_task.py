"""Background generation of picture thumbnails.

A thumbnail is ONE aspect-ratio-preserving bitmap of the whole frame (short edge
``THUMBNAIL_SHORT_EDGE`` px, long edge capped), plus the face-weighted square-crop
rectangle within it. Generation is MODE-AGNOSTIC - the per-user
``thumbnail_mode`` is a display-only preference the frontend owns and is never
read here.

This task is data-driven off ``Picture.thumbnail_width IS NULL``. New imports
populate the columns at import time, so the rows this task processes are the ones
something reset to NULL: an upgrade (a one-time regeneration of the whole-frame
bitmap) or an in-place rotate and its undo, which change the source file under an
existing thumbnail. Each processed picture is regenerated from its source: the
bitmap file is (re)written and every ``thumbnail_*`` / ``square_crop_*`` column is
set. Faces (when already present) weight the square-crop rectangle.

**Completion is announced.** While a picture sits at NULL it has no stored aspect
ratio, so its card lays out wrong, and its cache token is ``"0"``. Both are fixed
here - but a client cannot know that without being told, and this task used to
tell no one, so the repair reached the disk and the database while the open grid
kept showing the wrong picture until a full reload. See :meth:`_announce`.
"""

import os

from PIL import Image
from sqlmodel import Session

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.db_models.face import read_stored_bbox
from pixlstash.pixl_logging import get_logger
from pixlstash.services.scrapheap_service import file_location_is_unreachable
from pixlstash.tasks.base_task import BaseTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.image_processing.media_load import LoadFailure
from pixlstash.utils.unprocessable_image_registry import SignedLoad, load_signed

logger = get_logger(__name__)


class ThumbnailGenerationTask(BaseTask):
    """Generate thumbnails for one batch of pictures (CPU queue)."""

    BATCH_SIZE = 64
    #: The result key listing, as ``(picture_id, path, mtime_ns, size)``, each
    #: picture whose file decodes but whose EXIF turn raises (see
    #: :meth:`_note_orientation_failure`). Present only when there is one;
    #: ``MissingThumbnailFinder`` holds those until the file changes.
    ORIENTATION_FAILED = "orientation_failed"

    def __init__(self, database, pictures: list, notifier=None):
        picture_ids = [pic.id for pic in (pictures or []) if getattr(pic, "id", None)]
        super().__init__(
            task_type="ThumbnailGenerationTask",
            params={
                "picture_ids": picture_ids,
                "batch_size": len(picture_ids),
            },
        )
        self._db = database
        self._pictures = pictures or []
        self._notifier = notifier
        # (picture_id, path, mtime_ns, size) per ORIENTATION_FAILED picture.
        self._orientation_failures: list[tuple[int, str, int, int]] = []

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _face_bboxes(pic) -> list:
        """The picture's stored face boxes that weight its square crop.

        Read from each face row's ``bbox_`` text. A box that is not four
        finite numbers is logged once and left out, so the thumbnail is still
        made from the rest: a box that raised here left the picture without a
        thumbnail, and ``MissingThumbnailFinder`` (``thumbnail_width IS
        NULL``) selected it again on every sweep.
        """
        bboxes = []
        for face in getattr(pic, "faces", []) or []:
            bbox = read_stored_bbox(
                face.bbox_,
                owner="Face",
                row_id=face.id,
                picture_id=getattr(pic, "id", None),
            )
            if bbox is not None:
                bboxes.append([float(v) for v in bbox])
        return bboxes

    def _resolve_columns(self, pic) -> dict | None:
        """Regenerate the whole-frame bitmap for one picture; return column values.

        Writes the thumbnail file as a side effect. Returns ``None`` when the
        source is missing/unreadable (the missing-file purge finder owns cleanup).
        """
        file_path = getattr(pic, "file_path", None)
        if not file_path:
            return None
        resolved = ImageUtils.resolve_picture_path(self._db.image_root, file_path)
        if not resolved or not os.path.exists(resolved):
            return None

        img = ImageUtils.load_image_or_video(resolved)
        if img is None:
            self._hold_unloadable(getattr(pic, "id", None), str(resolved))
            return None
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        rendered = ImageUtils.render_thumbnail(img, face_bboxes=self._face_bboxes(pic))
        if rendered is None:
            return None
        thumbnail_bytes, bmp_w, bmp_h, crop = rendered

        saved = ImageUtils.write_thumbnail_bytes(
            self._db.image_root, file_path, thumbnail_bytes
        )
        if not saved:
            logger.warning(
                "ThumbnailGenerationTask: failed to persist thumbnail for picture %s",
                getattr(pic, "id", None),
            )
        return {
            "thumbnail_width": bmp_w,
            "thumbnail_height": bmp_h,
            "square_crop_x": crop["x"],
            "square_crop_y": crop["y"],
            "square_crop_side": crop["side"],
        }

    def _hold_unloadable(self, pic_id, resolved: str) -> None:
        """Keep a source that would not load from being re-selected, by cause.

        Returning ``None`` leaves ``thumbnail_width`` NULL, which is exactly
        what ``MissingThumbnailFinder`` selects on: whatever is not held is
        selected again on the next sweep (#585). ``load_image_or_video``
        answers ``None`` for every failure, so the file is loaded once more
        with the classifying loader (:func:`load_signed`) to learn why, and
        each cause gets the hold that ends with it:

        - bytes that do not decode: ``mark_unprocessable``, pinned to the
          file's signature, lifted when the file changes;
        - a location that is not mounted: ``mark_unreachable``, lifted when
          it is back;
        - any other read failure (no permission, a bad sector, a dropped
          share): ``hold_unreadable``, lifted the moment the file reads again;
        - bytes that decode but whose EXIF turn raises: reported to the
          finder, which holds the picture until the file changes
          (:meth:`_note_orientation_failure`).

        A load that runs out of memory, an unclassified failure, or a file that
        loads on the second try is only logged, and is retried.
        """
        registry = getattr(self._db, "unprocessable_images", None)
        loaded = load_signed(resolved, multiframe_count=1)
        failure = loaded.result
        if failure.failure is None:
            self._note_orientation_failure(pic_id, loaded)
            return
        if registry is None:
            logger.warning(
                "ThumbnailGenerationTask: failed to load source for picture %s "
                "(%s); %s: %s.",
                pic_id,
                resolved,
                failure.failure.value,
                failure.error,
            )
            return
        if failure.failure.permanent:
            # The load is the gate's evidence: the file is not decoded again.
            registry.mark_unprocessable(
                pic_id,
                resolved,
                reason=f"thumbnail source could not be decoded ({failure.failure.value})",
                load=loaded,
            )
        elif failure.failure is LoadFailure.IO_ERROR:
            if file_location_is_unreachable(resolved):
                registry.mark_unreachable(
                    pic_id, resolved, reason="thumbnail source location is not mounted"
                )
            else:
                registry.hold_unreadable(
                    pic_id, resolved, reason=f"thumbnail source: {failure.error}"
                )
        else:
            logger.warning(
                "ThumbnailGenerationTask: failed to load source for picture %s "
                "(%s): %s (%s); not held, it will be retried.",
                pic_id,
                resolved,
                failure.failure.value,
                failure.error,
            )

    def _note_orientation_failure(self, pic_id, loaded: SignedLoad) -> None:
        """Report a source that decodes but cannot be turned by its EXIF orientation.

        ``load_image_or_video`` orients through
        :meth:`ImageUtils.apply_exif_orientation`. When that raises and its
        video fallback cannot read the file either, the loader answers
        ``None`` while the bytes still decode. No known file does this any
        more: EXIF too damaged to read (a header that does not parse, one cut
        short, a PNG raw profile that is not hex) is shown as stored
        (``exif_orientation`` answers 1 for every error in
        ``EXIF_PARSE_ERRORS``), and EXIF that reads but cannot be written
        back is turned without writing it. A file that does not
        decode (a truncated one) is not this case either: the registry marks
        it. The hold stays for anything else that makes the turn raise on a
        decoded frame, which would otherwise be selected on every sweep.
        Nothing in the file changes between sweeps, so the picture is
        reported under :attr:`ORIENTATION_FAILED` with the signature taken
        before the decode, and ``MissingThumbnailFinder`` holds it until that
        signature moves.

        The probe is the loader's own orientation step, so the two agree.
        Every exception it raises on a decoded frame is this case, except
        running out of memory, which is the machine's and is retried. A frame
        that orients here loaded on a second attempt, and is retried.
        """
        try:
            ImageUtils.apply_exif_orientation(loaded.result.frames[0])
        except MemoryError as exc:
            logger.warning(
                "ThumbnailGenerationTask: picture %s (%s) ran out of memory "
                "applying its EXIF orientation (%s); not held, it will be retried.",
                pic_id,
                loaded.file_path,
                exc,
            )
            return
        except Exception as exc:
            if loaded.signature is None:
                logger.warning(
                    "ThumbnailGenerationTask: picture %s (%s) decodes but its EXIF "
                    "orientation cannot be read (%s: %s), and the file could "
                    "not be stat'd to hold it; it will be retried.",
                    pic_id,
                    loaded.file_path,
                    type(exc).__name__,
                    exc,
                )
                return
            self._orientation_failures.append(
                (int(pic_id), loaded.file_path, *loaded.signature)
            )
            logger.warning(
                "ThumbnailGenerationTask: picture %s (%s) decodes but its EXIF "
                "orientation cannot be read (%s: %s); it is held until the "
                "file changes.",
                pic_id,
                loaded.file_path,
                type(exc).__name__,
                exc,
            )
            return
        logger.warning(
            "ThumbnailGenerationTask: failed to load source for picture %s (%s); "
            "it loaded on a second attempt, so it will be retried.",
            pic_id,
            loaded.file_path,
        )

    def _run_task(self):
        self._orientation_failures = []
        updates: dict[int, dict] = {}
        for pic in self._pictures:
            pic_id = getattr(pic, "id", None)
            if pic_id is None:
                continue
            try:
                columns = self._resolve_columns(pic)
            except Exception as exc:
                logger.warning(
                    "ThumbnailGenerationTask: error processing picture %s: %s",
                    pic_id,
                    exc,
                )
                continue
            if columns:
                updates[pic_id] = columns

        if not updates:
            return self._result(0)

        def _persist(session: Session, updates: dict[int, dict]):
            changed = 0
            for pic_id, columns in updates.items():
                db_pic = session.get(Picture, pic_id)
                if db_pic is None:
                    continue
                for key, value in columns.items():
                    setattr(db_pic, key, value)
                session.add(db_pic)
                changed += 1
            session.commit()
            return changed

        changed = self._db.run_task(_persist, updates, priority=DBPriority.LOW)
        logger.debug("ThumbnailGenerationTask updated %s pictures.", changed)
        self._announce(list(updates.keys()))
        return self._result(changed or 0)

    def _result(self, changed: int) -> dict:
        """The task result: the changed count, and any :attr:`ORIENTATION_FAILED`."""
        result = {"changed_count": changed}
        if self._orientation_failures:
            result[self.ORIENTATION_FAILED] = list(self._orientation_failures)
        return result

    def _announce(self, picture_ids: list[int]) -> None:
        """Tell open clients their tiles have a new bitmap to fetch.

        Regeneration is the *completion* of a change something else started: an
        upgrade that reset these columns to NULL, a reference-folder source swap,
        or an in-place rotate and its undo. Until this ran, those pictures had no
        stored dimensions at all - so a card had no aspect ratio to lay out with
        and rendered stretched, and the cache token was ``"0"``. Both are only
        repaired by re-reading the thumbnail endpoint, which nothing prompted:
        the whole task was silent, so the fix landed on disk and in the database
        and the open grid went on showing the wrong picture until a full reload.

        ``pixels`` is the field name because the picture's own bytes are what the
        client must re-read - the same marker an in-place rotate raises, handled
        by the same branch. Best-effort: a broken notifier must never fail a
        thumbnail batch that has already been persisted.
        """
        if not self._notifier or not picture_ids:
            return
        try:
            from pixlstash.event_types import EventType

            self._notifier(
                EventType.CHANGED_PICTURES,
                {
                    "picture_ids": picture_ids,
                    "change_kind": "updated",
                    "fields": ["pixels"],
                },
            )
        except Exception as exc:
            logger.warning(
                "ThumbnailGenerationTask could not announce %d regenerated "
                "thumbnails (%s); their cards will look stale until the view is "
                "reloaded",
                len(picture_ids),
                exc,
            )
