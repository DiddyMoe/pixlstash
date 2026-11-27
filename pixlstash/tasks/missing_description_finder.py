from collections import Counter
from typing import Callable, Collection

from sqlmodel import Session, select
from sqlalchemy import func, literal, or_

from pixlstash.db_models import (
    Picture,
    DESCRIPTION_SENTINEL_LIKE_PATTERN,
    DESCRIPTION_SENTINEL_ESCAPE_CHAR,
    make_description_sentinel,
    parse_engine_from_description_sentinel,
    is_description_sentinel,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.services.set_lock_service import locked_picture_id_subquery
from pixlstash.task_runner import TaskCancelledError
from pixlstash.utils.sql_chunking import not_in_json

from .base_task import TaskStatus
from .description_task import DescriptionTask, describable_picture_clauses
from .task_outcome import Reason, RetryHold, id_sample, read_outcomes
from .task_type import TaskType
from .base_task_finder import BaseTaskFinder

logger = get_logger(__name__)


class MissingDescriptionFinder(BaseTaskFinder):
    """Find a batch of pictures missing descriptions and create a DescriptionTask.

    Reads every task's ``outcomes`` (see :mod:`pixlstash.tasks.task_outcome`)
    so that a batch that failed is not selected again on the next sweep:

    - ``FAILED_RETRYABLE`` holds the picture back on a capped, jittered
      exponential backoff (:class:`RetryHold`), so one that keeps failing is
      retried more and more rarely, and its description is never cleared.
    - ``FAILED_RETRYABLE`` because the batch captioned nobody
      (``BACKEND_UNAVAILABLE``) also holds the engine the batch captioned
      with (its sentinel's engine, else the active plugin): none of that
      engine's pictures are read until the hold expires, and the next batch
      for it is the probe; other engines' work goes on. The pictures are held
      one by one as well, because the same answer comes from a batch that one
      picture in it makes the model fail every time: their holds outgrow the
      backend hold (capped at 300 s against 60 s), so such a batch is passed
      over and the backlog behind it moves on.
    - ``FAILED_PERMANENT`` was written as ``""``, which the query never
      selects, or is held out by the unprocessable-image registry, whose
      pictures the query leaves out; ``DEFERRED`` was not attempted and stays
      eligible.
    - ``SUCCEEDED`` clears the picture's hold and its engine's hold.

    A task that failed outright holds all of its pictures; a cancelled one
    holds none. The holds live in memory: a restart retries everything.
    """

    #: Longest wait between two probes of a backend that captions nothing, seconds.
    BACKEND_HOLD_CAP_S = 60.0

    def __init__(
        self,
        database,
        engine_getter: Callable,
        *,
        retry_hold: RetryHold | None = None,
        backend_hold: RetryHold | None = None,
    ):
        """
        Args:
            database: The vault database.
            engine_getter: Returns the inference engine, or ``None`` while
                there is none.
            retry_hold: Per-picture backoff; a default one when omitted.
            backend_hold: Per-engine backoff for an engine that captions
                nothing; a default one capped at :attr:`BACKEND_HOLD_CAP_S`
                when omitted.
        """
        super().__init__()
        self._db = database
        self._engine_getter = engine_getter
        self._retry_hold = retry_hold if retry_hold is not None else RetryHold()
        self._backend_hold = (
            backend_hold
            if backend_hold is not None
            else RetryHold(cap_s=self.BACKEND_HOLD_CAP_S)
        )
        # Task id -> the engine it captions with, the key of its backend hold.
        self._task_engines: dict = {}

    def finder_name(self) -> str:
        return "MissingDescriptionFinder"

    def depends_on(self) -> list[TaskType]:
        return [TaskType.FACE_EXTRACTION, TaskType.TAGGER]

    def find_task(self):
        engine = self._engine_getter()
        if engine is None:
            return None

        # Only queue description work when an active description plugin is configured.
        tagger_settings = getattr(engine, "tagger_settings", None)
        if tagger_settings is not None:
            active_plugin = tagger_settings.get("active_description_plugin")
            if not active_plugin:
                return None
        else:
            # No tagger_settings at all: Florence-2 is always active.
            active_plugin = "florence2"

        batch_limit = max(
            1,
            int(engine.description_batch_size()),
        )

        # Held and registry-suppressed pictures are left out of the read
        # itself, so they cannot fill the candidate window and stall the rest.
        excluded_ids = self._retry_hold.active() | self._suppressed_picture_ids()
        pictures = self._db.run_immediate_read_task(
            self._fetch_missing_descriptions,
            batch_limit * 3,
            excluded_ids,
            held_engines=self._backend_hold.active(),
            active_engine=active_plugin,
        )
        if not pictures:
            return None

        # A sentinel is a user's reset (#1162): those pictures come first, as
        # an urgent task, ahead of the NULL backlog the import left behind.
        # Group them by the engine embedded in the sentinel (None = use
        # active_description_plugin) and process one group per cycle.
        requested = [
            pic for pic in pictures if is_description_sentinel(pic.description)
        ]
        if requested:
            groups: dict[str | None, list] = {}
            for pic in requested:
                engine_name = parse_engine_from_description_sentinel(pic.description)
                groups.setdefault(engine_name, []).append(pic)
            first_engine = next((k for k in groups if k is not None), None)
            first_pics = groups[first_engine]
        else:
            first_engine = None
            first_pics = pictures
        selected = self._drop_held(self._filter_and_claim(first_pics, batch_limit))
        if not selected:
            return None

        task = DescriptionTask(
            database=self._db,
            workflow=engine.description_workflow,
            pictures=selected,
            engine_override=first_engine,
            interactive=bool(requested),
        )
        self._task_engines[task.id] = first_engine or active_plugin
        return task

    def on_task_complete(self, task, error) -> None:
        """Hold back what failed and may be retried, then release the batch.

        The holds are recorded before the claims are released: the planner may
        run ``find_task`` the moment a claim is let go, and a picture released
        before its hold exists would be selected straight back. In a
        ``finally``, so a malformed result cannot strand the batch's claims.
        """
        try:
            self._settle(task, error)
        finally:
            super().on_task_complete(task, error)

    def _drop_held(self, selected: list) -> list:
        """*selected* without the pictures held since the candidates were read.

        A task completing between the read and the claim holds its failed
        pictures and then releases their claims, so the claim check alone lets
        them through. Their new claims are released here.
        """
        held = self._retry_hold.active()
        raced = {pic.id for pic in selected if pic.id in held}
        if not raced:
            return selected
        with self._claim_lock:
            self._claimed_picture_ids.difference_update(raced)
        return [pic for pic in selected if pic.id not in raced]

    def _settle(self, task, error) -> None:
        """Record what *task*'s outcomes say about its pictures' holds."""
        picture_ids = (getattr(task, "params", None) or {}).get("picture_ids") or []
        engine = self._task_engines.pop(getattr(task, "id", None), None)
        if isinstance(error, TaskCancelledError) or (
            error is None and getattr(task, "status", None) == TaskStatus.CANCELLED
        ):
            # It never ran, or was stopped: nothing was learnt about the
            # pictures, and they stay eligible. A task cancelled before it
            # started reports no error at all, only its status.
            return
        if error is not None:
            self._hold(picture_ids, f"the task failed: {error}")
            return

        try:
            summary = read_outcomes(getattr(task, "result", None))
        except ValueError as exc:
            summary = None
            logger.error(
                "MissingDescriptionFinder: task %s reported an outcome that is "
                "not one: %s",
                getattr(task, "id", None),
                exc,
            )
        if summary is None:
            # The task always reports outcomes, so this is a bug in it. Its
            # pictures are not known to have succeeded; holding them keeps the
            # bug from turning into a loop.
            logger.error(
                "MissingDescriptionFinder: task %s returned no outcomes for its "
                "%d pictures",
                getattr(task, "id", None),
                len(picture_ids),
            )
            self._hold(picture_ids, "the task reported no outcomes")
            return

        for pid in summary.succeeded | summary.failed_permanent:
            self._retry_hold.clear(pid)
        if summary.succeeded:
            self._backend_hold.clear(engine)

        backend_down = {
            pid
            for pid in summary.failed_retryable
            if summary.reasons.get(pid) is Reason.BACKEND_UNAVAILABLE
        }
        # One task in flight at a time (``max_inflight_tasks``), and none is
        # created for a held engine, so each failure lengthens its hold once.
        if backend_down:
            wait = self._backend_hold.hold(engine)
            logger.warning(
                "MissingDescriptionFinder: the last %r batch captioned nobody; "
                "no new %r batch for %.0f s",
                engine,
                engine,
                wait,
            )

        # The backend-down pictures are held one by one too (see the class
        # docstring): a batch that fails because of one of its own pictures
        # reads exactly like a backend that is down.
        if summary.failed_retryable:
            reasons = Counter(
                summary.reasons[pid].value if pid in summary.reasons else "unspecified"
                for pid in summary.failed_retryable
            )
            self._hold(
                summary.failed_retryable,
                ", ".join(f"{r}={n}" for r, n in sorted(reasons.items())),
            )

    def _hold(self, picture_ids, why: str) -> None:
        """Hold *picture_ids* back once more, and say so in one line."""
        if not picture_ids:
            return
        waits = [self._retry_hold.hold(pid) for pid in picture_ids]
        logger.warning(
            "MissingDescriptionFinder: %d pictures are held back for "
            "%.0f-%.0f s before their next attempt (%s): %r",
            len(waits),
            min(waits),
            max(waits),
            why,
            id_sample(picture_ids),
        )

    @staticmethod
    def _fetch_missing_descriptions(
        session: Session,
        limit: int,
        exclude_ids: Collection[int] = frozenset(),
        *,
        held_engines: Collection[str] = frozenset(),
        active_engine: str | None = None,
    ):
        # *exclude_ids* (the held and the suppressed pictures) are left out
        # here, ahead of the LIMIT, so they cannot fill the window. So are the
        # pictures a held engine would caption: a sentinel naming it, and, when
        # *active_engine* is held, NULL and the sentinel naming no engine.
        #
        # A picture frozen by a locked set has a read-only description (rule 3):
        # never re-queue it for machine (re)description. Parity with the tagger's
        # MissingTagFinder exclusion; the description_task write-side also skips
        # locked pics as defense in depth.
        #
        # Must be the shared set_lock_service predicate, not a local
        # PictureSetMember join. The local join had no stack arm, while
        # DescriptionTask's write guard (`locked_picture_ids`) does: a picture
        # merely *sharing a stack* with a locked-set member was selected here, ran
        # full captioning inference, had its write skipped, kept its NULL
        # description, and was selected again next sweep - an unbounded loop.
        #
        # Scrapheap pictures and pictures with no file are left out the same
        # way (describable_picture_clauses, shared with the awaiting count):
        # the write skips a Scrapheap picture, so selecting one would caption
        # it again on every sweep.
        not_locked = ~Picture.id.in_(locked_picture_id_subquery())
        stmt = select(Picture).where(
            or_(
                Picture.description.is_(None),
                # Rendered inline, not bound: ``ix_picture_description_pending``
                # is a partial index on exactly this OR, and SQLite only
                # matches a LIKE term to it when the pattern is the same
                # literal. Bound as ``?``, every planner sweep scans the
                # whole picture table, even with nothing pending.
                Picture.description.like(
                    literal(DESCRIPTION_SENTINEL_LIKE_PATTERN, literal_execute=True),
                    escape=DESCRIPTION_SENTINEL_ESCAPE_CHAR,
                ),
            ),
            not_locked,
            *describable_picture_clauses(),
            not_in_json(Picture.id, exclude_ids),
        )
        if held_engines:
            held = [make_description_sentinel(name) for name in held_engines]
            if active_engine in held_engines:
                held.append(make_description_sentinel(None))
            # NULL reads as the sentinel naming no engine: both caption with
            # the active one.
            stmt = stmt.where(
                func.coalesce(
                    Picture.description, make_description_sentinel(None)
                ).not_in(held)
            )
        return session.exec(
            # Sentinels (a reset) ahead of NULLs (never captioned), then by id,
            # so a request never waits behind the backlog.
            stmt.order_by(Picture.description.is_(None), Picture.id).limit(limit)
        ).all()
