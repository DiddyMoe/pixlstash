import threading
import time
from typing import Callable, Collection

from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

from pixlstash.db_models import (
    Picture,
    Tag,
    TAG_ENGINE_SENTINEL_PREFIX,
    is_tag_sentinel,
    make_tag_sentinel,
    parse_tag_engine_from_sentinel,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.tagger_plugins.pixlstash_tagger import TaggerUnavailableError
from pixlstash.task_runner import TaskCancelledError
from pixlstash.utils.query.predicate_filter import awaiting_tagger_clauses
from pixlstash.worker_config import IMAGE_TASK_PIXEL_BUDGET, TAGGER_MAX_INFLIGHT
from .base_task_finder import BaseTaskFinder
from .tag_task import TagTask
from .task_outcome import RetryHold

logger = get_logger(__name__)

# The sentinel set is the range `awaiting_tagger_clauses` selects, which
# ``ix_tag_tag`` serves. Within it, the engine-named sentinels,
# [``__tag:``, ``__tag;``), are exactly the values
# `parse_tag_engine_from_sentinel` reads an engine from. Every other sentinel
# value tags with the active plugin.
_ENGINE_SENTINEL_RANGE_END = TAG_ENGINE_SENTINEL_PREFIX[:-1] + chr(
    ord(TAG_ENGINE_SENTINEL_PREFIX[-1]) + 1
)


class MissingTagFinder(BaseTaskFinder):
    """Find a batch of pictures with a pending-retag sentinel and create a TagTask.

    A TagTask loads its tag plugin before it tags anything, and a load that
    fails fails the whole task (``TaggingWorkflow.ensure_active_plugin_ready``).
    Its pictures keep their sentinel, so without a hold the next sweep, a few
    milliseconds later, builds the same batch and loads the plugin again: a
    tagger that cannot load is retried for ever. A task that fails outright
    therefore holds that plugin back on the capped, jittered backoff
    ``MissingImageEmbeddingFinder`` applies to an unavailable CLIP backend
    (:class:`RetryHold`, capped at :attr:`BACKEND_HOLD_CAP_S`); the next batch
    after the hold is the probe, and a task that succeeds clears it. The hold
    is per plugin, so a retag requested with a plugin that fails does not stop
    the configured one. A cancelled task holds nothing. In memory: a restart
    retries at once.

    A task that failed with :class:`TaggerUnavailableError` carrying a
    :class:`TaggerLoadFailure` is held for the tagger's own
    ``failure.retry_in_s`` instead: the tagger turns every caller away until
    then, so a batch built sooner would only be refused, and one built at the
    end of it is the load that is tried next. The tagger's wait grows to
    ``LOAD_RETRY_MAX_S``, past this finder's cap.
    """

    #: Longest wait between two attempts to load a tag plugin that failed, seconds.
    BACKEND_HOLD_CAP_S = 60.0

    def __init__(
        self,
        database,
        engine_getter: Callable,
        *,
        backend_hold: RetryHold | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            database: The vault database.
            engine_getter: Returns the inference engine, or ``None`` while
                there is none.
            backend_hold: Per-plugin backoff after a task fails outright; a
                default one capped at :attr:`BACKEND_HOLD_CAP_S` when omitted.
            clock: Monotonic seconds, for the tagger's own load wait.
        """
        super().__init__()
        self._db = database
        self._engine_getter = engine_getter
        self._backend_hold = (
            backend_hold
            if backend_hold is not None
            else RetryHold(cap_s=self.BACKEND_HOLD_CAP_S)
        )
        # Task id -> the plugin it tags with, for ``on_task_complete``. Written
        # on the planner thread, popped on the runner's, hence the lock.
        self._task_plugins: dict[str, str] = {}
        self._task_plugins_lock = threading.Lock()
        self._clock = clock
        # Plugin -> when the tagger next tries its load (``clock`` seconds),
        # from a TaggerUnavailableError. Under ``_task_plugins_lock``.
        self._load_waits: dict[str, float] = {}

    def finder_name(self) -> str:
        return "MissingTagFinder"

    def max_inflight_tasks(self) -> int:
        return TAGGER_MAX_INFLIGHT

    # No `on_all_tasks_complete`: the drain used to tear the WD14 session down,
    # and with per-row dependencies a finder drains and refills many times per
    # pass, so that was a reload storm. The session is bounded by its own
    # `gpu_mem_limit` now, and `Vault._maybe_aggressive_unload` frees it once
    # every worker has been idle - only under `keep_models_in_memory=False`.

    def find_task(self):
        engine = self._engine_getter()
        if engine is None:
            return None
        # Only queue tagging when an active tag plugin is configured.
        tagger_settings = getattr(engine, "tagger_settings", None)
        active_tag_plugin = (tagger_settings or {}).get("active_tag_plugin")
        if not active_tag_plugin:
            return None

        batch_limit = max(
            1,
            int(engine.tagging_workflow.suggested_task_size()),
        )
        # Fetch enough candidates that _filter_and_claim can always fill one
        # additional task even when all max_inflight slots are already in-flight.
        max_inflight = max(1, self.max_inflight_tasks())
        # Exclude undecodable pictures at the query, as the face, thumbnail and
        # embedding finders already do. `_filter_and_claim` drops them anyway,
        # but only after they have taken up room in the candidate window - and
        # an undecodable picture keeps its pending-tag sentinel for ever
        # (TagTask marks it unprocessable and writes nothing, so nothing
        # clears the sentinel). They therefore accumulate permanently, and
        # this window is ordered by `Picture.id`: once as few as
        # `batch_limit * (max_inflight + 1)` of them sit below the pictures
        # that still need tagging, every sweep reads the same corrupt rows,
        # claims none of them and returns None - which the planner reads as
        # "no work" and answers with a growing backoff. That is tagging
        # starving library-wide on a handful of bad files, the same shape
        # `MissingFaceExtractionFinder` documents on its own query.
        suppressed_ids = self._suppressed_picture_ids()
        # Held plugins are left out at the query for the same reason: the
        # window is ordered by id, so a held plugin's pictures below everyone
        # else's would fill it, every group would be skipped below, and the
        # plugins that work would wait out the hold too.
        held = self._held_plugins()
        # Before the read: a reset landing between the two must make the task
        # older than the reset, never newer (#1361).
        reset_generation = self._db.tag_resets.current()
        pictures = self._db.run_immediate_read_task(
            lambda session: self._fetch_missing_tags(
                session,
                batch_limit * (max_inflight + 1),
                suppressed_ids,
                held_plugins=held,
                active_tag_plugin=active_tag_plugin,
            )
        )
        if not pictures:
            return None

        # Group pictures by their requested engine (None = use active_tag_plugin).
        groups: dict[str | None, list] = {}
        for pic in pictures:
            sentinel_tag = next(
                (t.tag for t in pic.tags if is_tag_sentinel(t.tag)),
                None,
            )
            engine_name = parse_tag_engine_from_sentinel(sentinel_tag)
            groups.setdefault(engine_name, []).append(pic)

        # Process the first group whose plugin is not held back (subsequent
        # calls will handle the rest).
        # The plugin a group loads is the one ``TaggingWorkflow.active_plugin_name``
        # picks: the sentinel's engine when it names one, else the active plugin.
        # The query already left out held plugins; this skip still matters for
        # a picture carrying two sentinels, one of them held.
        workflow = engine.tagging_workflow
        group = next(
            (
                (engine_name, pics, plugin)
                for engine_name, pics in groups.items()
                for plugin in [
                    engine_name if engine_name is not None else active_tag_plugin
                ]
                if plugin not in held
            ),
            None,
        )
        if group is None:
            return None
        first_engine, first_pics, plugin = group
        selected = self._filter_and_claim(
            first_pics, batch_limit, pixel_budget=IMAGE_TASK_PIXEL_BUDGET
        )
        if not selected:
            return None

        task = TagTask(
            database=self._db,
            tagging_workflow=workflow,
            pictures=selected,
            engine_override=first_engine,
            reset_generation=reset_generation,
        )
        with self._task_plugins_lock:
            self._task_plugins[task.id] = plugin
        return task

    def on_task_complete(self, task, error) -> None:
        """Hold the task's plugin back if it failed outright, then release the batch.

        The hold is recorded before the claims are released: the planner may
        run ``find_task`` the moment a claim is let go, and a batch built before
        the hold exists would load the failing plugin straight away. In a
        ``finally``, so a failure here cannot strand the batch's claims.
        """
        try:
            self._settle_backend(task, error)
        finally:
            super().on_task_complete(task, error)

    def _held_plugins(self) -> set:
        """Plugins no new batch may load yet: backed off, or in the tagger's
        own load wait."""
        now = self._clock()
        with self._task_plugins_lock:
            waiting = {
                plugin
                for plugin, ready_at in self._load_waits.items()
                if ready_at > now
            }
        return self._backend_hold.active() | waiting

    def _settle_backend(self, task, error) -> None:
        """Hold or clear the plugin *task* tagged with, from how it ended."""
        with self._task_plugins_lock:
            plugin = self._task_plugins.pop(getattr(task, "id", None), None)
        if plugin is None:
            # Not a task this finder built (an interactive retag).
            return
        if isinstance(error, TaskCancelledError):
            # Stopped or never started (the runner settles both as a
            # cancellation): nothing was learnt about the plugin.
            return
        if error is None:
            self._backend_hold.clear(plugin)
            with self._task_plugins_lock:
                self._load_waits.pop(plugin, None)
            return
        failure = getattr(error, "failure", None)
        if (
            isinstance(error, TaggerUnavailableError)
            and failure is not None
            and failure.retry_in_s > 0
        ):
            # The tagger says when it loads again; every task that fails
            # together carries the same time, so the latest one stands.
            with self._task_plugins_lock:
                self._load_waits[plugin] = self._clock() + failure.retry_in_s
            logger.warning(
                "MissingTagFinder: plugin %r is not loaded: %s No new %r batch "
                "for %.0f s; %d pictures keep their place in the queue.",
                plugin,
                error,
                plugin,
                failure.retry_in_s,
                len((getattr(task, "params", None) or {}).get("picture_ids") or []),
            )
            return
        # Up to TAGGER_MAX_INFLIGHT tasks fail together when the plugin cannot
        # load; only the first of them lengthens the hold.
        if plugin in self._backend_hold.active():
            return
        wait = self._backend_hold.hold(plugin)
        logger.warning(
            "MissingTagFinder: a tagging task with plugin %r failed: %s. No new "
            "%r batch for %.0f s; %d pictures keep their place in the queue.",
            plugin,
            error,
            plugin,
            wait,
            len((getattr(task, "params", None) or {}).get("picture_ids") or []),
        )

    @staticmethod
    def _fetch_missing_tags(
        session: Session,
        limit: int,
        suppressed_ids: Collection[int] = frozenset(),
        *,
        held_plugins: Collection[str] = frozenset(),
        active_tag_plugin: str | None = None,
    ):
        """The window of pictures waiting to be tagged, lowest id first.

        Args:
            session: The read session.
            limit: Window size.
            suppressed_ids: Pictures the unprocessable registry holds out.
            held_plugins: Plugins on a backend hold. A sentinel naming one
                (``__tag:<plugin>``) does not select its picture, and when
                ``active_tag_plugin`` is among them neither does a sentinel
                that names no engine, since that one loads the active plugin.
            active_tag_plugin: The plugin an engine-less sentinel tags with.
        """
        sentinel_terms = []
        if held_plugins:
            sentinel_terms.append(
                Tag.tag.not_in([make_tag_sentinel(p) for p in sorted(held_plugins)])
            )
            if active_tag_plugin in held_plugins:
                sentinel_terms.extend(
                    (
                        Tag.tag >= TAG_ENGINE_SENTINEL_PREFIX,
                        Tag.tag < _ENGINE_SENTINEL_RANGE_END,
                    )
                )
        # Per-row stage dependency: tag a picture once ITS faces are known, not
        # once the whole face stage has drained. Face extraction always writes
        # at least one Face row (a ``face_index=-1`` sentinel when there is no
        # face), so ``faces.any()`` is exactly "face extraction has run". The
        # quality crop in TagTask only *prefers* a face (it centre-crops
        # without one), but cropping on the face is the shipped behaviour, so
        # the picture waits for it.
        #
        # The wait is bounded because `MissingFaceExtractionFinder` selects on
        # the exact complement, `~Picture.faces.any()`: a picture with no row
        # is by definition still face-stage work, so a cancelled or failed
        # batch is re-offered rather than stranded. The one predicate the two
        # finders did not share was the suppressed set, which is why
        # `awaiting_tagger_clauses` applies it.
        faces_known = Picture.faces.any()
        # Everything else is `awaiting_tagger_clauses`, the definition
        # `TagTask.count_missing_tags` and the stats "Waiting for the tagger"
        # row apply, so neither can count a picture this query will not hand
        # out. The copies used to drift: the count held the Scrapheap out while
        # this query kept tagging it (#1206 item 3), and a local locked-set join
        # here had no stack arm while TagTask's write guard does, so a stack
        # co-member of a locked picture was tagged, its write skipped, and
        # selected again every sweep - an unbounded inference loop. The locked
        # term in that function is the shared set_lock_service predicate.
        stmt = select(Picture).where(
            *awaiting_tagger_clauses(suppressed_ids, sentinel_terms=sentinel_terms),
            faces_known,
        )
        return session.exec(
            stmt.options(
                selectinload(Picture.tags),
            )
            .order_by(Picture.id)
            .limit(limit)
        ).all()
