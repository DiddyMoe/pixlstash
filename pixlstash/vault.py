import concurrent
import ctypes
import platform

import datetime
import os
import time
import threading
import numpy as np

from pathlib import Path
from typing import Collection, Optional
from concurrent.futures import Future

from sqlmodel import Session, select
from sqlalchemy import func, or_, update


from .database import DBPriority, VaultDatabase
from .db_models import (
    MetaData,
    Character,
    Face,
    Picture,
    PicturePendingPerson,
    PictureSet,
    ReferenceFolder,
    Snapshot,
    DESCRIPTION_SENTINEL_LIKE_PATTERN,
    DESCRIPTION_SENTINEL_ESCAPE_CHAR,
    make_description_sentinel,
)
from .pixl_logging import get_logger
from .scoring import match_faces_to_characters
from pixlstash.startup_permissions import mkdir_private
from pixlstash.inference.engine import InferenceEngine
from .utils.image_processing.image_utils import ImageUtils
from .tasks.description_task import describable_picture_clauses
from .tasks.face_extraction_task import FaceExtractionTask
from .tasks.image_embedding_task import ImageEmbeddingTask
from .tasks.likeness_parameters_task import LikenessParametersTask
from .tasks.likeness_task import LikenessTask
from .tasks.missing_comfyui_extraction_finder import MissingComfyUIExtractionFinder
from .tasks.missing_face_extraction_finder import MissingFaceExtractionFinder
from .tasks.missing_tag_prediction_finder import MissingTagPredictionFinder
from .tasks.missing_text_embedding_finder import MissingTextEmbeddingFinder
from .tasks.missing_thumbnail_finder import MissingThumbnailFinder
from .tasks.ocr_task import OcrTask
from .tasks.orientation_task import OrientationTask
from .tasks.pipeline_scope import pipeline_picture_clauses
from .tasks.quality_task import QualityTask
from .tasks.smart_score_task import SmartScoreTask
from .tasks.base_task import TaskStatus
from .tasks.device_release_task import DeviceReleaseTask
from .task_runner import TaskRunner, TaskRunnerStopped
from .resources import cuda_checkpoint
from .resources.aitoolkit_tenant import AiToolkitProbe
from .resources.arbiter import Lease, LeaseKind, Tenant, arbiter
from .resources.comfyui_tenant import probe as comfyui_probe
from .resources.ledger import ledger
from .work_planner import WorkPlanner
from .tasks import TaskType
from .utils.reference_folder_watcher import ReferenceFolderWatcher
from .utils.vram_utils import empty_device_cache
from . import worker_config

from pixlstash.event_types import EventType
from pixlstash.utils.service.smart_score_invalidation import (
    InteractiveRescoreRegistry,
    SmartScoreAnchorWatch,
)
from pixlstash.tagger_plugins.registry import (
    get_tagger_plugin_manager,
    unload_loaded_tagger_plugins,
)
from pixlstash.services.set_lock_service import (
    enforce_pictures_not_locked,
    locked_picture_id_subquery,
)
from pixlstash.utils.sql_chunking import chunked, not_in_json
from pixlstash.utils.quality.tag_exclusion import (
    exclusion_generation,
    exclusions_from_tagger_settings,
)
from pixlstash.services.scrapheap_run_service import (
    ScrapheapRunRegistry,
    recover_runs,
)
from pixlstash.services.scrapheap_service import DEFAULT_RETENTION_DAYS
from pixlstash.services.import_publish_service import PublishedPictures
from pixlstash.services.workflow_ghost_service import DEFAULT_GHOST_RETENTION
from pixlstash.services.snapshot_service import SnapshotService
from pixlstash.utils.service.export_utils import ExportUtils
from pixlstash.services.training_service import ADOPT_QUIET_S, TrainingService
from pixlstash.services import comfyui_connections
from pixlstash.utils.comfyui_url import SavedComfyUIAddressInvalid
from pixlstash.services.comfyui_service import RUN_CLIENT_ID
from pixlstash.services.stack_detector import set_cover
from pixlstash.hub import training_runs
from pixlstash.services.run_importer import STATUS_IMPORTED, RunImporter
from pixlstash.services.model_mover import SHELF_IO_LOCK, publish_no_clobber
from pixlstash.utils.aitoolkit_run import read_run
from pixlstash.services.managed_model_store import find_download_folder
from pixlstash.services.model_shelf_service import (
    UnknownAttachmentEntityError,
    fetch_attachments,
    replace_attachments,
)
from pixlstash.db_models.adapter_attachment import ENTITY_CHARACTER, ENTITY_SET
from pixlstash.services.restore import RestoreService
from pixlstash.services.restore._models import TASK_LABELS, task_label
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    SwitchState,
)
from pixlstash.services.folder_structure_commit_service import (
    reconcile_pending_people_with_faces,
)
from pixlstash.services.dedup_auto_stack_service import (
    interrupt_orphaned_runs_in_session,
)
from pixlstash.hub.registry import VAULT_FILENAME
from pixlstash.trusted_sqlite import TrustedSQLiteLocation
from pixlstash.utils.path_utils import LibraryRootsUnavailable
from pixlstash.tasks.missing_saved_recipe_workflow_finder import (
    MissingSavedRecipeWorkflowFinder,
)
from pixlstash.tasks.downloads_intake_finder import DownloadsIntakeFinder
from pixlstash.tasks.model_folder_scan_finder import ModelFolderScanFinder
from pixlstash.services.model_folder_scanner import SHELF_MODEL_SUFFIXES


logger = get_logger(__name__)

# Each worker-progress row carries its task's label, and a refusal that names
# the work in the way reads the same table. A TaskType with no label is a
# programming error: checked here, where both modules are already imported
# (``pixlstash.tasks`` imports the snapshot service, which imports the restore
# package, so the restore models cannot import ``TaskType`` themselves).
_unlabelled_task_types = sorted(
    worker_type.value
    for worker_type in TaskType
    if worker_type.value not in TASK_LABELS
)
if _unlabelled_task_types:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "Every TaskType needs a label in TASK_LABELS (these have none): "
        f"{_unlabelled_task_types}"
    )
del _unlabelled_task_types

#: How long making room waits, on the arbiter's thread, for the GPU batch in
#: flight to end and the models to unload; past it they unload when the batch
#: ends, without the arbiter waiting.
YIELD_WAIT_S = 2.0

#: How long stopping the vault waits for face extraction's follow-ups: the face
#: writes still on the writer and the pending-person matches they queue when
#: they land. Past it the writer is closed anyway: a match that has not run is
#: dropped, and its pictures keep their pending people.
FACE_FOLLOWUP_STOP_WAIT_S = 10.0

#: The GPU worker task that unloads the models and suspends the CUDA context
#: while a training run has the card (#2557).
CUDA_SUSPEND_TASK = "SuspendCuda"

#: The worker types whose tasks queue on the task runner's GPU lane (their
#: ``BaseTask.queue_type``); every other worker type queues on the CPU lane.
#: tests/test_vault_worker_lanes.py checks it against the task classes.
GPU_LANE_WORKER_TYPES = frozenset(
    {
        TaskType.FACE_EXTRACTION,
        TaskType.FACE_MODEL_REFRESH,
        TaskType.DETECTION,
        TaskType.TAGGER,
        TaskType.TAG_PREDICTION_BACKFILL,
        TaskType.DESCRIPTION,
        TaskType.TEXT_EMBEDDING,
        TaskType.IMAGE_EMBEDDING,
        TaskType.PROMPT_MATCH,
        TaskType.OCR,
    }
)


def _move_trained_attachment(
    vault, run: dict, old_sha: Optional[str], new_sha: str
) -> None:
    """Attach *new_sha* to the run's set or character, and *old_sha* no more.

    Every other attachment either model carries is kept: each write is a full
    replacement, so it is built from what the model holds now. A run of neither
    a set nor a character attaches nothing.
    """
    if run.get("character_id") is not None:
        entity = (ENTITY_CHARACTER, run["character_id"])
    elif run.get("set_id") is not None:
        entity = (ENTITY_SET, run["set_id"])
    else:
        return
    if old_sha:
        held = fetch_attachments(vault, sha256=old_sha).get(old_sha, [])
        kept = [(a["entity_type"], a["entity_id"]) for a in held]
        if entity in kept:
            replace_attachments(vault, old_sha, [e for e in kept if e != entity])
    held = fetch_attachments(vault, sha256=new_sha).get(new_sha, [])
    kept = [(a["entity_type"], a["entity_id"]) for a in held]
    if entity not in kept:
        try:
            replace_attachments(vault, new_sha, kept + [entity])
        except UnknownAttachmentEntityError:
            # The run outlived its set or character: deleted while it trained,
            # or gone with a restore to a snapshot older than it. The LoRA is
            # on the shelf already, so the run is not failed for it.
            logger.warning(
                "Training run %s (%s): its %s %s no longer exists in this "
                "library, so LoRA %s stays on the shelf unattached.",
                run.get("id"),
                run.get("name"),
                entity[0],
                entity[1],
                new_sha,
                exc_info=True,
            )


class NotTheActiveLibraryError(LibraryNotReadyError):
    """The vault finalizing a training run is no longer the active library.

    A run imports only into the library that trained it, which is the vault
    whose training service watched it. Once a switch has published another
    vault, this one refuses, and the run's row stays for the reconcile the next
    time its library opens.
    """

    def __str__(self) -> str:
        return f"This vault is not the active library (admission {self.state.value})"


class Vault:
    AGGRESSIVE_UNLOAD_INTERVAL = 180

    def __enter__(self):
        # Allow use as a context manager for robust cleanup
        return self

    def __exit__(self, _, __, ___):
        self.close()

    """
    Represents a vault for storing images and metadata.

    The vault contains a database that manages a SQLite database and stores the vault description in the metadata table.
    """

    def __init__(
        self,
        image_root: str,
        description: Optional[str] = None,
        server_config_path: Optional[str] = None,
        path_mapper=None,
        disable_background_workers: bool = False,
        planner_finders: Optional[Collection[TaskType]] = None,
        force_cpu: bool = False,
        fast_captions: bool = False,
        daily_snapshots_enabled: bool = True,
        insightface_model_pack: str = "buffalo_l",
        scrapheap_retention_days: Optional[int] = DEFAULT_RETENTION_DAYS,
        scrapheap_retention_reduced_at: Optional[datetime.datetime] = None,
        db_filename: Optional[str] = None,
        ghost_retention: str = DEFAULT_GHOST_RETENTION,
    ):
        """
        Initialize a Vault instance.

        Args:
            db_path (str): Path to the SQLite database file.
            image_root (Optional[str]): Path to the image root directory.
            description (Optional[str]): Description of the vault.
            disable_background_workers (bool): When True, the TaskRunner,
                WorkPlanner and ReferenceFolderWatcher are not started and
                no models are ever loaded.  Intended for read-only deployments
                where all processing is already complete.
            planner_finders (Optional[Collection[TaskType]]): Register only
                these finders with the work planner; ``None`` registers all of
                them. A restricted vault also builds no inference engine, so a
                model finder kept in the set registers but never queues work:
                the import routes refuse unless FACE_EXTRACTION is registered,
                and an e2e run must not load models. Only the e2e backend sets
                it (``e2e_planner_finders``, under ``enable_test_hooks``).
            db_filename (Optional[str]): Name of the database file inside
                *image_root*. ``None`` takes it from the hub registration when
                the vault is opened through one, which is what puts a library
                still on its first import onto the temporary name without any
                caller having to know. Falls back to ``vault.db``.
        """
        registered_hub = getattr(image_root, "hub", None)
        registered_library = getattr(image_root, "library", None)
        self.image_root = str(image_root)
        logger.debug(f"Image root: {self.image_root}")
        assert self.image_root is not None, "image_root cannot be None"
        logger.debug(f"Using image_root: {self.image_root}")
        # Creation only: existing directories keep their modes, whatever they
        # are - the user may deliberately share their image folder with a media
        # server or sync agent. Not makedirs(mode=0o700): since Python 3.7 that
        # mode reaches only the LEAF, so intermediates of a deep new image_root
        # came out 0775 under Ubuntu's umask 002 and the guarded open then
        # refused the namespace the app itself had just created (W21).
        mkdir_private(Path(self.image_root))
        assert os.path.exists(self.image_root), (
            f"Image root path does not exist: {self.image_root}"
        )

        # The registration is the single source of truth for which file this
        # library's database is: a first import runs against the temporary
        # name and the promotion renames it. Reading it here means start-up,
        # the library switch and the switch's own recovery path all open the
        # right file without passing the name around.
        if db_filename is None:
            db_filename = (
                registered_library.vault_filename
                if registered_library is not None
                else VAULT_FILENAME
            )
        self._db_path = os.path.join(self.image_root, db_filename)
        if registered_hub is not None and registered_library is not None:
            from pixlstash.hub.bootstrap import (
                finalize_library_connection,
                prevalidate_opened_library,
            )

            # The engine's connect hook legitimately creates WAL/SHM before
            # verification. That is handled inside the guard now, for every
            # caller, rather than by this one site opting out of a directory
            # check the other three still paid for.
            location_guard = TrustedSQLiteLocation.open(
                self._db_path,
                create=not os.path.exists(self._db_path),
            )
            self.db = VaultDatabase(
                self._db_path,
                location_guard=location_guard,
                pre_migration_hook=lambda connection: prevalidate_opened_library(
                    connection, registered_library
                ),
                post_migration_hook=lambda connection: finalize_library_connection(
                    registered_hub, registered_library, connection
                ),
            )
        else:
            self.db = VaultDatabase(self._db_path)
        self.set_description(description or "")

        self.snapshot_service = SnapshotService(self)
        self.restore_service = RestoreService(self)
        # The server's AuthService, attached by ``Server.__init__`` once it has
        # been constructed (it takes ``self.db``, so it cannot exist yet here).
        # The restore path needs it: a full restore replaces the database file,
        # which invalidates every piece of authentication state held in memory
        # (see AuthService.reset_after_restore and §18.5). Stays ``None`` for a
        # Vault built without a Server - tests, the CLI tools - and the restore
        # path treats that as "nothing to reset".
        self.auth_service = None
        # The server's LibraryGenerationCoordinator, attached by whoever
        # publishes this vault as the active library (``Server.__init__`` and
        # the library switch). A vault write that runs outside a request holds
        # one of its read leases, so a restore or a switch waits for it. Stays
        # ``None`` for a vault nobody published: tests, the CLI tools, and the
        # library ``open_other_library`` opens to read.
        self.library_coordinator = None

        self._engine: InferenceEngine | None = None
        self._force_cpu = force_cpu
        self._fast_captions = fast_captions
        self._insightface_model_pack = insightface_model_pack
        self._last_aggressive_unload_at = 0.0
        # Two progress polls can reach the sweep at once; one queues it.
        self._aggressive_unload_lock = threading.Lock()
        self._keep_models_in_memory = True
        self._max_vram_gb = None
        self._wd14_tagger_enabled = False
        self._pixlstash_tagger_enabled = True
        self._wd14_threshold = None
        self._pixlstash_tagger_threshold_offset = None
        self._tagger_settings: dict | None = None
        # Single-entry memo for resolve_anomaly_apply_thresholds: (key, resolved_map),
        # keyed on (meta_path, offset, default_threshold) so it self-invalidates when any
        # of those move. Declared before the disable_background_workers early-return so
        # both modes carry it.
        self._anomaly_apply_thresholds_memo: tuple | None = None
        self._server_config_path = server_config_path
        self._disable_background_workers = disable_background_workers
        self._planner_finders = (
            frozenset(planner_finders) if planner_finders is not None else None
        )
        self._daily_snapshots_enabled: bool = daily_snapshots_enabled
        # Scrapheap auto-purge retention window (days) or None = "Never", plus
        # the instant it was last LOWERED (puts a +1 day floor under EVERY
        # picture's purge deadline, however old). See scrapheap_service.
        # Defaults to None: auto-purge stays off until the user turns it on.
        self._scrapheap_retention_days: Optional[int] = scrapheap_retention_days
        self._scrapheap_retention_reduced_at: Optional[datetime.datetime] = (
            scrapheap_retention_reduced_at
        )
        # The hub this vault was opened through, and which library it is, or
        # None for both (the CLI tools, most tests). The purge needs them:
        # ghosts live in the hub, ``deleted_file_log`` does not reach across,
        # and a ghost is scoped to the library whose vault holds its cover
        # (§B4).
        self._hub = registered_hub
        self._library_uuid: Optional[str] = getattr(registered_library, "uuid", None)
        # Built after the hub and library are known: training runs live in the
        # hub, scoped to this library.
        self.training_service = self._build_training_service(disable_background_workers)
        self._ghost_retention: str = ghost_retention

        self._planner_watchers = {}
        self._planner_watchers_lock = threading.Lock()
        # Face extractions whose follow-up (announcing the landed rows and
        # matching pending people to them) has not finished, each a future
        # mapped to its picture ids. stop() waits on them before closing the
        # writer; see _after_face_writes_land.
        self._face_followups: dict[Future, list[int]] = {}
        self._face_followups_lock = threading.Lock()
        self._changed_tags_notify_lock = threading.Lock()
        self._changed_tags_pending_ids: set[int] = set()
        self._changed_tags_flush_timer: threading.Timer | None = None
        self._changed_tags_flush_delay_s = 2.0
        # Set while quiesce_background_writers has the watcher stopped.
        self._ref_folder_watcher_quiesced = False
        self._event_listeners = []
        self._event_listeners_lock = threading.Lock()
        self._path_mapper = path_mapper
        # Bridges an interactive tag edit that NULLs a smart_score to the immediate,
        # origin-stamped grid refresh emitted when its background rescore lands - so the
        # edited card updates in place instead of waiting for the whole backfill to drain
        # (see _on_task_completed's SmartScoreTask branch). Created before the
        # disable_background_workers early-return because the mutation handlers that
        # record into it run in both modes.
        self.interactive_rescore_registry = InteractiveRescoreRegistry()
        # Clears every cached smart score when a rating, undo, soft delete or dedup
        # score lift actually changes the scorer's anchor set, and only then. Hooked
        # into every writer session; before the early-return for the same reason.
        self.smart_score_anchor_watch = SmartScoreAnchorWatch(self.db)
        self.db.add_session_hook(self.smart_score_anchor_watch.attach)
        # The background Scrapheap runs (empty, delete forever, restore all) and
        # the last few that ended. Before the early-return so a server without
        # workers can still answer "no run" rather than raise; it refuses to
        # start one (scrapheap_run_service.start_run).
        self.scrapheap_runs = ScrapheapRunRegistry()
        # Set by the first start(), which closes the runs a previous process
        # left active; start() runs more than once per vault (Server init and
        # lifespan), and a later call must not interrupt this process's runs.
        self._runs_recovered = False

        if disable_background_workers:
            logger.info(
                "disable_background_workers=True: TaskRunner, WorkPlanner and "
                "ReferenceFolderWatcher will not be started. No models will be loaded."
            )
            self._task_runner = None
            self._planner_work_finders = {}
            self._work_planner = None
            self._ref_folder_watcher = None
            self._model_folder_watcher = None
            self._closed = False
            self._started = False
            return

        # The machine's one VRAM arbiter. It outlives this vault: a library
        # switch builds a new vault and the leases ComfyUI holds stay put.
        self._arbiter = arbiter()
        self._arbiter.set_pixlstash_on_gpu(not force_cpu)
        # Lease ids PixlStash made room for, each holding the planner.
        self._yield_holds: set[str] = set()
        self._cuda_suspend_task: Optional[DeviceReleaseTask] = None
        # Suspends in a row a worker that could not start kept off the queue.
        self._cuda_suspend_queue_failures = 0
        self._task_runner = TaskRunner(
            name="vault-task-runner",
            num_workers=worker_config.NUM_WORKERS,
            # So a GPU out-of-memory retry can raise a toast instead of only a
            # log line. Bound method, like the finders' notifier below.
            notifier=self.notify,
            resource_arbiter=self._arbiter,
        )
        self._planner_work_finders = WorkPlanner.work_finders(
            database=self.db,
            engine_getter=lambda: self._engine,
            image_root=self.image_root,
            path_mapper=path_mapper,
            # So a regenerated thumbnail can announce itself. Passed as the bound
            # method rather than the vault, to keep the finders' surface to what
            # they actually use.
            notifier=self.notify,
        )
        from pixlstash.tasks import (
            TaskType,
            EnsureGfsSnapshotFinder,
            ScrapheapRetentionPurgeFinder,
            TagHealthAutoRebuildFinder,
        )

        self._planner_work_finders[TaskType.GFS_SNAPSHOT] = EnsureGfsSnapshotFinder(
            vault=self
        )
        # Needs a full Vault to read the retention settings (and to notify on
        # purge); same reason GFS_SNAPSHOT is registered here rather than in
        # WorkPlanner.work_finders().
        self._planner_work_finders[TaskType.SCRAPHEAP_RETENTION_PURGE] = (
            ScrapheapRetentionPurgeFinder(vault=self)
        )
        # Needs a full Vault (not just `database`) to reach its service
        # layer's vault-facing wrapper (start_rebuild), same reason
        # GFS_SNAPSHOT is registered here rather than in
        # WorkPlanner.work_finders().
        self._planner_work_finders[TaskType.TAG_HEALTH_AUTO_REBUILD] = (
            TagHealthAutoRebuildFinder(vault=self)
        )
        # Needs a full Vault to resolve the tagger's per-label acceptance thresholds
        # (meta.json path + user threshold offset) for the anomaly penalty; same reason
        # as the two finders above.
        from pixlstash.tasks.missing_smart_score_finder import MissingSmartScoreFinder

        self._planner_work_finders[TaskType.SMART_SCORE] = MissingSmartScoreFinder(
            vault=self
        )
        # Repairs the one window the hub/vault split opened: a penalised-tag
        # weight change commits its setting to the hub and its score
        # invalidation to the vault, and those cannot share a transaction. The
        # settings handler records the invalidation durably first and applies it
        # inline; this finder is what completes it if the process died in
        # between. Registered here, like the finders above, because it needs the
        # Vault rather than just the database.
        from pixlstash.tasks.pending_score_invalidation_finder import (
            PendingScoreInvalidationFinder,
        )

        self._planner_work_finders[TaskType.PENDING_SCORE_INVALIDATION] = (
            PendingScoreInvalidationFinder(vault=self)
        )
        # Works the background auto-stack runs the Duplicates screen requests.
        # Needs the Vault, not just the database: a run announces the pictures
        # it stacks and a revert restores through the undo registry.
        from pixlstash.tasks.dedup_auto_stack_finder import DedupAutoStackFinder

        self._planner_work_finders[TaskType.DEDUP_AUTO_STACK] = DedupAutoStackFinder(
            vault=self
        )
        self._model_folder_watcher = None
        # The one finder here that works on the HUB rather than on this vault:
        # a model folder is a fact about the machine, so the checkpoints it
        # registers are shared by every library. Registered only when the vault
        # was opened through a hub registration (a Vault built without one - the
        # CLI tools, most tests - has no hub to reach), and harmless in the
        # multi-library case because only one vault is live at a time.
        if registered_hub is not None:
            # Every permanent delete of a picture row, on whatever path, leaves
            # its instance hash in ``pending_ghost_cascade`` (a vault trigger,
            # migration 0117), and this finder is what runs the covered-ghost
            # cascade over them (§B4). Hub-only because ghosts live in the hub;
            # a hubless vault keeps the queue until a hub opens it.
            from pixlstash.tasks.ghost_cascade_finder import GhostCascadeFinder

            self._planner_work_finders[TaskType.GHOST_CASCADE] = GhostCascadeFinder(
                vault=self
            )
            from pixlstash.tasks.missing_checkpoint_hash_finder import (
                MissingCheckpointHashFinder,
            )

            self._planner_work_finders[TaskType.CHECKPOINT_HASH] = (
                MissingCheckpointHashFinder(hub=registered_hub)
            )
            # The watched Downloads folder, a hub fact for the same reason.
            self._planner_work_finders[TaskType.DOWNLOADS_INTAKE] = (
                DownloadsIntakeFinder(hub=registered_hub, notifier=self.notify)
            )
            # And the model folders themselves: rescanned when a model file in
            # one changes, and every ten minutes besides. There is no Rescan.
            self._model_folder_watcher = ReferenceFolderWatcher(
                on_folder_changed=self._on_model_folder_fs_changed,
                extensions=frozenset(SHELF_MODEL_SUFFIXES),
                # A download writes for minutes; one scan after it goes quiet.
                debounce_s=5.0,
            )
            self._planner_work_finders[TaskType.MODEL_FOLDER_SCAN] = (
                ModelFolderScanFinder(
                    hub=registered_hub,
                    watcher=self._model_folder_watcher,
                    notifier=self.notify,
                )
            )
            # The workflow-card backfill (§B2). Hub-only in the strongest sense:
            # it reads stored documents and never a picture, so it keys the
            # recipes of libraries that are not even attached, and the pictures
            # a trimmed library no longer has still have their workflow.
            from pixlstash.tasks.workflow_card_backfill_finder import (
                WorkflowCardBackfillFinder,
            )

            self._planner_work_finders[TaskType.WORKFLOW_CARD_BACKFILL] = (
                WorkflowCardBackfillFinder(hub=registered_hub, vault=self)
            )
            # The cut-over (#1623): saved recipes naming a card are put on the
            # workflow it became. Needs the hub's `workflow_key_successor`.
            self._planner_work_finders[TaskType.SAVED_RECIPE_CONVERT] = (
                MissingSavedRecipeWorkflowFinder(vault=self)
            )
            # And the workflow library's store is the hub too, so the ComfyUI
            # extraction can file a picture's graph where it outlives the
            # picture (§B3). Replaces the hubless finder work_finders() built:
            # that one still extracts ComfyUI metadata, on the pre-B3
            # `comfyui_models IS NULL` predicate -- it is only the WORKFLOW scan
            # it cannot do, because without a hub there is nowhere to file a
            # graph.
            self._planner_work_finders[TaskType.COMFYUI_EXTRACTION] = (
                MissingComfyUIExtractionFinder(
                    database=self.db,
                    image_root=self.image_root,
                    hub=registered_hub,
                    library_uuid=self._library_uuid,
                    vault=self,
                )
            )
            # A training run's previews are files in its folder, so nothing
            # else finds their faces; the live view scores them against the
            # run's character (#2522).
            if self._library_uuid:
                from pixlstash.tasks.training_preview_faces_finder import (
                    TrainingPreviewFacesFinder,
                )

                self._planner_work_finders[TaskType.TRAINING_PREVIEW_FACES] = (
                    TrainingPreviewFacesFinder(
                        hub=registered_hub,
                        library_uuid=self._library_uuid,
                        image_root=self.image_root,
                        engine_getter=lambda: self._engine,
                    )
                )
        if self._planner_finders is not None:
            self._planner_work_finders = {
                task_type: finder
                for task_type, finder in self._planner_work_finders.items()
                if task_type in self._planner_finders
            }
        self._work_planner = WorkPlanner(
            task_runner=self._task_runner,
            task_finders=self._planner_work_finders,
        )
        self._task_runner.add_task_complete_callback(self._on_task_completed)
        self._task_runner.add_task_complete_callback(
            self._work_planner.on_task_complete
        )
        # Last, so a task's library lease outlives the callbacks above.
        self._task_runner.add_task_complete_callback(self._release_task_library_lease)

        self._ref_folder_watcher = ReferenceFolderWatcher(
            on_folder_changed=self._on_reference_folder_fs_changed,
        )
        self._closed = False
        self._started = False

    def ensure_ready(self):
        """Initialise the picture tagger so the planner can process work immediately.

        Call this at server startup. Tests that do not need the tagger can skip it;
        tagger init is also triggered lazily by get_worker_future().

        On Metal this blocks for a few seconds while the CPU query encoders
        load; see :meth:`_create_engine`.
        """
        if self._disable_background_workers or self._planner_finders is not None:
            return
        if not self._engine:
            self._engine = self._create_engine()
            self._bind_engine_services()

    def start(self) -> None:
        """Start background workers.

        Must be called once after configuration is complete (e.g. from
        ``Server.lifespan``).  Idempotent: subsequent calls are no-ops.
        When ``disable_background_workers=True`` only the run records are
        reconciled; no worker starts.
        """
        if not self._runs_recovered:
            self._runs_recovered = True
            # An auto-stack or Scrapheap run the previous process left pending or
            # running has no task to own it any more. Close it as interrupted
            # before the planner can queue it: a run is never resumed, the user
            # decides whether to run again. Before the no-worker return, so a
            # server without workers still reports how such a run ended rather
            # than showing it running forever.
            self.db.run_task(
                interrupt_orphaned_runs_in_session, priority=DBPriority.IMMEDIATE
            )
            recover_runs(self)
        if self._disable_background_workers or self._started:
            return
        # A pending person on a picture that already has faces is never offered
        # again by the face finder; close what a crash or a timed-out stop left.
        reconcile_pending_people_with_faces(self)
        # A trainer that outlived the restart gets its card before PixlStash's
        # own GPU workers start, so none of them loads a model under it.
        self.training_service.adopt_live_runs()
        if self._task_runner.start():
            self._work_planner.start()
        else:
            # Refused beside a worker that outlived the last stop, or during a
            # stop (logged by the runner). The planner is not started: every
            # cycle would claim work, have the stopped runner refuse it and
            # release it again, warning each time. Background work is then
            # reported as unavailable with the planner not alive ("restart
            # the server", ``worker_unavailable_reason``), and every
            # submission is refused as stopped (``submit_task`` answers None).
            # The watchers below still start.
            logger.error(
                "The task runner did not start (a stop is still under way, or a "
                "worker from its last stop is still running); the work planner "
                "is not started either. Background work is off until a restart."
            )
        self._arbiter.add_yielder(self._yield_gpu)
        self._arbiter.add_listener(self._on_resource_lease)
        self._arbiter.set_suspender(self._queue_cuda_suspend)
        self._start_reference_folder_watches()
        if self._model_folder_watcher is not None:
            self._model_folder_watcher.start()
        # Reconcile before the training worker starts: a trainer that survived a
        # restart is re-attached (or finalized, if it died) before the worker can
        # claim a queued job and ask the arbiter for the card. The yielder is
        # registered above, so the arbiter can move PixlStash's GPU work off the
        # card for either.
        self.training_service.reconcile_on_startup()
        self.training_service.start()
        self._started = True

    def _create_engine(self) -> InferenceEngine:
        """Build the inference engine, with its CPU query encoders loaded.

        The caller publishes the result to ``self._engine`` only after this
        returns, and that order is the point. The planner's model finders queue
        nothing while ``self._engine`` is ``None``, so loading the copies here
        (Metal hosts only) runs no other model load beside them, even at boot,
        where ``Server.__init__`` has already started the planner. A load beside
        another one races transformers' and accelerate's imports. See
        ``inference/cpu_query_encoders.py``.

        Returns:
            The engine, ready to publish.
        """
        engine = InferenceEngine.create(
            image_root=self.image_root,
            force_cpu=self._force_cpu,
            fast_captions=self._fast_captions,
            max_vram_gb=self._max_vram_gb,
            wd14_enabled=self._wd14_tagger_enabled,
            pixlstash_tagger_enabled=self._pixlstash_tagger_enabled,
            wd14_threshold=self._wd14_threshold,
            pixlstash_tagger_threshold_offset=self._pixlstash_tagger_threshold_offset
            or 0.0,
            keep_models_in_memory=self._keep_models_in_memory,
            insightface_model_pack=self._insightface_model_pack,
            tagger_settings=self._tagger_settings,
        )
        if engine.query_encoders is not None:
            engine.query_encoders.load()
        return engine

    def _bind_engine_services(self) -> None:
        """Inject the engine's service instances into registry plugins.

        Built-in tagger plugins (WD14, PixlStash Tagger, Florence-2) have their
        own ``_service`` attribute that stays ``None`` on the registry instance
        because the engine creates its own dedicated service objects.  Binding
        those services here makes ``plugin.is_loaded()`` reflect reality so the
        Settings UI shows the correct loaded state.
        """
        if self._engine is None:
            return
        mgr = get_tagger_plugin_manager()
        bindings = [
            ("wd14", self._engine.wd14_service),
            ("pixlstash_tagger", self._engine.pixlstash_tagger_service),
            ("florence2", self._engine.florence_service),
        ]
        for plugin_name, service in bindings:
            plugin = mgr.get_plugin(plugin_name)
            if (
                plugin is not None
                and service is not None
                and hasattr(plugin, "bind_service")
            ):
                plugin.bind_service(service)

    def emit_event(self, event_type: EventType, data=None):
        """Emit an event to all registered listeners and wake the work planner.

        Alias for ``notify()`` used by the service layer.

        Args:
            event_type: The event type to emit.
            data: Optional data payload.
        """
        self.notify(event_type, data)

    def notify(self, event_type: EventType, data=None):
        """
        Notify all relevant workers for a given event type.

        Example:
            vault.notify(Vault.VaultEventType.NEW_PICTURE)
        """
        # Waking the planner means "there may be work to pick up". A VRAM_OOM is
        # the opposite statement - the GPU is full - and waking on it would send
        # the planner looking for more GPU work three times per failing task, at
        # the one moment there is no room for any. A RESOURCE_LEASE is the same
        # kind of statement, and arrives on every heavy lease's state change.
        if (
            event_type not in (EventType.VRAM_OOM, EventType.RESOURCE_LEASE)
            and self._work_planner
            and self._work_planner.is_running()
        ):
            self._work_planner.wake()
        with self._event_listeners_lock:
            listeners = list(self._event_listeners)
        for listener in listeners:
            try:
                listener(event_type, data)
            except Exception as exc:
                logger.warning("Event listener failed for %s: %s", event_type, exc)

    def wake(self) -> None:
        """Wake the work planner without emitting an event.

        This preserves compatibility with older call sites that used
        ``vault.wake()`` to resume background work after DB mutations.
        """
        if self._work_planner and self._work_planner.is_running():
            self._work_planner.wake()

    def add_event_listener(self, listener):
        """Register a callback to be invoked when vault events occur."""
        if not callable(listener):
            raise ValueError("listener must be callable")
        with self._event_listeners_lock:
            if listener not in self._event_listeners:
                self._event_listeners.append(listener)

    # -------------------------------------------------------------------------
    # Reference folder filesystem watching
    # -------------------------------------------------------------------------

    def _start_existing_folder_watches(self) -> None:
        """Start filesystem watches for all reference folders already in the DB."""
        if self._ref_folder_watcher is None:
            return
        from pixlstash.db_models.reference_folder import ReferenceFolder
        from pixlstash.utils.path_mapper import PathMapper

        effective_mapper = (
            self._path_mapper if self._path_mapper is not None else PathMapper()
        )

        def fetch(session: Session) -> list[ReferenceFolder]:
            return list(session.exec(select(ReferenceFolder)).all())

        try:
            folders = self.db.run_task(fetch, priority=DBPriority.IMMEDIATE)
        except Exception as exc:
            logger.warning("Could not fetch reference folders for FS watching: %s", exc)
            return

        # The library's own root is watched under the id None, the same key
        # layout_move_service.layout_roots gives it. A rename the owner makes
        # in the laid-out tree is otherwise a row the purge sweep deletes.
        try:
            self._ref_folder_watcher.watch_folder(None, self.image_root)
        except Exception as exc:
            logger.warning(
                "Could not start FS watch for the library root %s: %s",
                self.image_root,
                exc,
            )

        for rf in folders:
            try:
                resolved = effective_mapper.resolve(rf.folder)
                self._ref_folder_watcher.watch_folder(rf.id, resolved)
            except Exception as exc:
                logger.warning(
                    "Could not start FS watch for reference folder %d (%s): %s",
                    rf.id,
                    rf.folder,
                    exc,
                )

    def reference_folder_roots(self) -> tuple[str, ...]:
        """Return the roots pictures may legitimately live under besides image_root.

        Each configured reference folder contributes its stored (host-side)
        path and, when a path mapper is configured, its mapped
        (container-side) form: stored ``Picture.file_path`` values are written
        under the mapped form in Docker deployments, while the
        ``reference_folder`` row keeps the host form.

        **A read failure raises; it is never an empty answer.** The two things
        this list is used for want opposite behaviour from the same value:
        :func:`~pixlstash.services.scrapheap_service.remove_picture_files`
        treats it as an *allowlist* of places a purge may delete from, where
        empty is safe, while the folder-export destination check treats it as
        a *blocklist* of places nothing may be written to, where empty silently
        permits what it exists to refuse
        (#1177 item 59). Returning ``()`` on error therefore fails closed for
        one caller and open for the other. Only the caller knows which
        direction is safe, so the error is made distinguishable from "no
        reference folders are configured" and the allowlist caller is the one
        that catches it.

        Returns:
            The root directories; an empty tuple only when no reference folder
            is configured.

        Raises:
            LibraryRootsUnavailable: The folder list could not be read.
        """
        from pixlstash.db_models.reference_folder import ReferenceFolder

        def fetch(session: Session) -> list[str]:
            return [
                row.folder
                for row in session.exec(select(ReferenceFolder)).all()
                if row.folder
            ]

        try:
            folders = self.db.run_immediate_read_task(fetch)
        except Exception as exc:
            logger.warning(
                "Could not read the reference folders of %s; refusing to "
                "answer rather than reporting none, because an empty answer "
                "reads as 'nothing to protect' to some callers: %s",
                self.image_root,
                exc,
            )
            raise LibraryRootsUnavailable(
                f"Could not read the reference folders of {self.image_root}: {exc}"
            ) from exc

        roots: list[str] = []
        for folder in folders:
            roots.append(folder)
            if self._path_mapper is not None:
                mapped = self._path_mapper.resolve(folder)
                if mapped != folder:
                    roots.append(mapped)
        return tuple(roots)

    def watch_reference_folder(self, folder_id: int, folder_path: str) -> None:
        """Start watching *folder_path* for reference folder *folder_id*.

        Should be called after a new reference folder is created so that
        filesystem changes are picked up immediately.

        Args:
            folder_id: Primary key of the reference folder.
            folder_path: Stored (host) path of the folder.
        """
        from pixlstash.utils.path_mapper import PathMapper

        effective_mapper = (
            self._path_mapper if self._path_mapper is not None else PathMapper()
        )
        if self._ref_folder_watcher is None:
            return
        try:
            resolved = effective_mapper.resolve(folder_path)
            self._ref_folder_watcher.watch_folder(folder_id, resolved)
        except Exception as exc:
            logger.warning(
                "Could not start FS watch for new reference folder %d (%s): %s",
                folder_id,
                folder_path,
                exc,
            )

    def unwatch_reference_folder(self, folder_id: int) -> None:
        """Stop watching the directory for *folder_id*.

        Should be called after a reference folder is deleted.

        Args:
            folder_id: Primary key of the reference folder to stop watching.
        """
        if self._ref_folder_watcher is None:
            return
        self._ref_folder_watcher.unwatch_folder(folder_id)

    def rescan_library_root(self) -> None:
        """Ask for the library's own picture root to be rescanned now: the pass
        that reads existing caption files in and writes the missing ones out
        once caption sync is turned on."""
        self._on_reference_folder_fs_changed(None)

    def mark_reference_folder_changed(self, folder_id: int) -> bool:
        """Do what the filesystem watcher does when a file in a reference
        folder changes: mark the folder due, so the next scan is now.

        Only the e2e test hook calls this. The e2e specs call it after every
        change they make on disk rather than wait on the watcher, which needs
        an inotify instance a busy machine may not have.

        Args:
            folder_id: Primary key of the reference folder.

        Returns:
            bool: False when there is no such folder, and nothing was marked.
        """
        exists = self.db.run_immediate_read_task(
            lambda session: session.get(ReferenceFolder, folder_id) is not None
        )
        if exists:
            self._on_reference_folder_fs_changed(folder_id)
        return exists

    def _on_model_folder_fs_changed(self, folder_id: int) -> None:
        """A model file in a registered model folder changed: scan it now."""
        finder = self._planner_work_finders.get(TaskType.MODEL_FOLDER_SCAN)
        if finder is None:
            return
        finder.mark_due(folder_id)
        if self._work_planner is not None:
            self._work_planner.wake()

    def _on_reference_folder_fs_changed(self, folder_id: int | None) -> None:
        """Callback invoked by the filesystem watcher when a relevant file changes.

        Resets ``last_scanned`` to zero so the
        :class:`~pixlstash.tasks.reference_folder_scan_finder.ReferenceFolderScanFinder`
        schedules an immediate rescan, then wakes the work-planner thread.
        ``None`` is the library root, whose schedule the finder keeps itself.
        """
        from pixlstash.db_models.reference_folder import ReferenceFolder

        def reset_last_scanned(session: Session, fid: int) -> None:
            rf = session.get(ReferenceFolder, fid)
            if rf is not None:
                rf.last_scanned = 0.0
                session.commit()

        if folder_id is None:
            finder = self._planner_work_finders.get(TaskType.REFERENCE_FOLDER_SCAN)
            if finder is not None:
                finder.mark_root_due()
        else:
            try:
                self.db.run_task(
                    reset_last_scanned, folder_id, priority=DBPriority.IMMEDIATE
                )
            except Exception as exc:
                logger.warning(
                    "Failed to reset last_scanned for reference folder %d: %s",
                    folder_id,
                    exc,
                )
        if self._work_planner and self._work_planner.is_running():
            self._work_planner.wake()

    def __repr__(self):
        """
        Return a string representation of the Vault instance.

        Returns:
            str: String representation.
        """
        return f"Vault(db_path='{self._db_path}')"

    @property
    def is_open(self) -> bool:
        return not self._closed and self.db is not None and self.db.is_open

    def close(self):
        """
        Cleanly close the vault, including stopping background workers and closing DB connection.
        """
        if self._closed:
            return
        self._closed = True
        self.stop()

    def quiesce_background_writers(self, timeout_s: float) -> list[str]:
        """Stop the vault's own writers that no task or lease fences.

        For a full restore, after the task runner has drained: the changed-tags
        flush timer, the reference-folder watcher, and face extraction's
        follow-ups. The first two would act on ids from the database the
        restore replaces; a follow-up is a write still queued after its task
        ended, which the drain does not see. ``resume_background_writers``
        undoes it, on whichever database file is then live.

        Args:
            timeout_s: How long to wait for the face follow-ups.

        Returns:
            What is still running: ``["face extraction follow-up"]`` when a
            follow-up outlived the wait, else ``[]``. The caller refuses the
            restore rather than swap the file under it.
        """
        self._cancel_changed_tags_flush()
        if self._started and self._ref_folder_watcher is not None:
            # Marked first: a stop that raises partway has still marked the
            # watcher stopped, and the resume must start a new observer.
            self._ref_folder_watcher_quiesced = True
            self._ref_folder_watcher.stop()
        if self._wait_for_face_followups(timeout_s):
            return ["face extraction follow-up"]
        return []

    def resume_background_writers(self) -> None:
        """Restart what ``quiesce_background_writers`` stopped.

        The reference-folder watches are read again from the live database,
        because a restore may have replaced the folders they named. The
        changed-tags timer re-arms on the next change; follow-ups need nothing.
        Does nothing for a watcher the quiesce did not stop, so a restore may
        call it on every exit path, including after a quiesce that raised.
        """
        if self._started and self._ref_folder_watcher_quiesced:
            self._ref_folder_watcher_quiesced = False
            self._start_reference_folder_watches()

    def _start_reference_folder_watches(self) -> None:
        """Start the reference-folder watcher and watch every folder it has."""
        self._ref_folder_watcher.start()
        self._start_existing_folder_watches()

    def _cancel_changed_tags_flush(self) -> None:
        """Cancel the changed-tags flush timer and drop the ids it would send.

        The event names pictures in the current database; sent after a restore
        or a close it would name rows that are not those pictures, or none.
        """
        with self._changed_tags_notify_lock:
            timer = self._changed_tags_flush_timer
            self._changed_tags_flush_timer = None
            self._changed_tags_pending_ids.clear()
        if timer is not None:
            timer.cancel()

    def stop(self) -> None:
        """Stop background workers and release all resources.

        Safe to call multiple times; subsequent calls after the first are no-ops.
        Called automatically by ``close()``.

        Once the task runner has stopped, and before the writer is closed, it
        waits up to ``FACE_FOLLOWUP_STOP_WAIT_S`` for face extraction's
        follow-ups: the face writes still on the writer and the pending-person
        matches they queue when they land (see `_after_face_writes_land`).
        """
        self._cancel_changed_tags_flush()
        # A snapshot copies and compresses off the writer and can take minutes.
        # Cancel it first so it removes its files and ends before the workers
        # are joined and the database is closed under it.
        self.snapshot_service.cancel_inflight()
        # Its quiet-period timer would queue a check on the database closed below.
        self.smart_score_anchor_watch.close()
        if self._started:
            # Stop the training poll thread first.  This does NOT kill a running
            # external trainer (by policy): it survives the restart and is
            # re-attached by reconcile_on_startup on the next start().
            if self.training_service is not None:
                self.training_service.stop()
            if self._ref_folder_watcher is not None:
                self._ref_folder_watcher.stop()
            if self._model_folder_watcher is not None:
                self._model_folder_watcher.stop()
            if self._task_runner is not None:
                self._arbiter.remove_yielder(self._yield_gpu)
                self._arbiter.remove_listener(self._on_resource_lease)
                self._arbiter.set_suspender(None)
            if self._work_planner is not None:
                self._work_planner.stop()
            if self._task_runner is not None:
                self._task_runner.stop()
        # The task runner has stopped: every face extraction it completed is in
        # `_face_followups` by now.
        self._wait_for_face_followups()
        # stop() gives up on a worker after 60 s. Releasing ends in a device-
        # cache flush, and on Metal a flush beside a worker still running a
        # model kills the process, so the models are left for the process exit.
        worker_alive = (
            self._task_runner is not None and self._task_runner.gpu_worker_alive()
        )
        if worker_alive:
            logger.warning(
                "The GPU worker is still running after the task runner stopped; "
                "leaving the models loaded rather than flushing the device under "
                "it. They stay resident until the process exits."
            )
        # Suspended for a training run, releasing a model is a CUDA call that
        # blocks until the run ends; they were unloaded before the suspend.
        keep_off_device = worker_alive or cuda_checkpoint.suspended()
        if not self._disable_background_workers and not keep_off_device:
            FaceExtractionTask.release_detection_models()
            ImageEmbeddingTask.release_models()
        if self._engine:
            if not keep_off_device:
                self._engine.close()
            del self._engine
            self._engine = None
        if self.db:
            self.db.close()
            del self.db
            self.db = None
        self._started = False

    def set_daily_snapshots_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic (GFS) snapshots at runtime.

        This is the master switch for the whole DAILY/WEEKLY/MONTHLY schedule.
        Takes effect immediately; the next EnsureGfsSnapshotFinder cycle will
        skip snapshot creation when ``enabled`` is False.

        Args:
            enabled: True to allow automatic snapshots, False to suppress them.
        """
        self._daily_snapshots_enabled = bool(enabled)

    @property
    def daily_snapshots_enabled(self) -> bool:
        """Whether automatic (GFS) snapshots are enabled."""
        return self._daily_snapshots_enabled

    def set_scrapheap_retention(
        self,
        retention_days: Optional[int],
        reduced_at: Optional[datetime.datetime],
    ) -> None:
        """Set the scrapheap auto-purge window at runtime.

        Takes effect on the next ScrapheapRetentionPurgeFinder cycle. It never
        purges anything synchronously - a config save must not destroy files.

        Args:
            retention_days: Days an UNPROTECTED soft-deleted picture stays in
                the scrapheap, or None for "Never" (auto-purge disabled).
            reduced_at: When the window was last lowered, or None if it never
                was. Puts a floor of ``reduced_at + 1 day`` under every
                picture's deadline, so a lowering never purges anything within
                a day of the save, however old it is.
        """
        self._scrapheap_retention_days = (
            None if retention_days is None else int(retention_days)
        )
        self._scrapheap_retention_reduced_at = reduced_at

    @property
    def disable_background_workers(self) -> bool:
        """Whether this vault was opened with its background workers off."""
        return self._disable_background_workers

    @property
    def hub(self):
        """The hub database this vault was opened through, or None."""
        return self._hub

    @property
    def library_uuid(self) -> Optional[str]:
        """Which library this vault is, in the hub's registry, or None."""
        return self._library_uuid

    def set_ghost_retention(self, retention: str) -> None:
        """Set the picture-ghost retention position at runtime.

        Takes effect on the next purge. It never destroys anything
        synchronously: turning the setting down is a statement about what future
        purges may keep, and clearing what is ALREADY kept is its own request,
        ``DELETE /server-config/ghost-retention/ghosts``.
        """
        self._ghost_retention = retention

    @property
    def ghost_retention(self) -> str:
        """Which picture ghosts a purge may keep: off / covered / on."""
        return self._ghost_retention

    @property
    def scrapheap_retention_days(self) -> Optional[int]:
        """Scrapheap auto-purge window in days; None means "Never"."""
        return self._scrapheap_retention_days

    @property
    def scrapheap_retention_reduced_at(self) -> Optional[datetime.datetime]:
        """When the retention window was last lowered, or None."""
        return self._scrapheap_retention_reduced_at

    @property
    def unprocessable_images(self):
        """Registry of pictures whose image file cannot be decoded (issue #585).

        Lives on the database so task threads (which mark decode failures) and
        finder threads (which suppress them) reach the same instance via their
        ``self._db``; exposed here for discoverability.
        """
        return self.db.unprocessable_images

    def set_keep_models_in_memory(self, keep_models_in_memory: bool):
        previous = self._keep_models_in_memory
        self._keep_models_in_memory = bool(keep_models_in_memory)

        if self._engine:
            self._engine.set_keep_models_in_memory(self._keep_models_in_memory)

        if previous and not self._keep_models_in_memory:
            logger.info(
                "keep_models_in_memory disabled; attempting immediate model unload."
            )
            self._last_aggressive_unload_at = 0.0
            progress = self._build_worker_progress_snapshot()
            self._maybe_aggressive_unload(progress)

    # -------------------------------------------------------------------------
    # Training service
    # -------------------------------------------------------------------------

    def _build_training_service(
        self, disable_background_workers: bool
    ) -> TrainingService:
        """Construct the TrainingService with vault-bound dependencies injected.

        The service is decoupled from the Vault: it takes callables, not the
        Vault itself.  Here we bind those callables to this vault: the VRAM
        lease request, the event emitter, the owner-config reader, and the dataset
        builder (``ExportUtils.build_training_dataset``, which needs an object
        exposing ``.vault``; a tiny shim provides that).

        Args:
            disable_background_workers: Forwarded so the service's worker is a
                no-op for read-only deployments.

        Returns:
            A configured :class:`TrainingService` (not yet started).
        """
        vault = self

        class _DatasetServerShim:
            """Minimal ``server`` stand-in exposing only ``.vault``.

            ``ExportUtils.build_training_dataset`` reads only ``server.vault.*``
            (db + image_root); the training dataset path never touches the
            request/query branches, so this is sufficient and keeps the service
            free of any HTTP-server coupling.
            """

            vault = None

        shim = _DatasetServerShim()
        shim.vault = vault

        def dataset_builder(**kwargs):
            return ExportUtils.build_training_dataset(shim, **kwargs)

        # What the trainer holds is AI Toolkit's to the arbiter, not other
        # apps', so a lease shrunk to a run's peak is not counted twice.
        tenant = None
        if not disable_background_workers:
            tenant = AiToolkitProbe()
            # The process's arbiter: self._arbiter is bound after this runs.
            arbiter().register_probe(Tenant.AITOOLKIT, tenant)

        return TrainingService(
            self._hub,
            self._library_uuid,
            self.image_root,
            request_gpu=self._request_training_gpu,
            import_run=self._import_training_run_under_lease,
            notify=self.notify,
            config_provider=self._training_config_provider,
            dataset_builder=dataset_builder,
            disable_background_workers=disable_background_workers,
            adopt_gpu=self._adopt_training_gpu,
            tenant=tenant,
        )

    def _training_config_provider(self) -> dict:
        """Read the owner's ai-toolkit config from the hub-backed owner row.

        The ai-toolkit paths and pinned commit are owner-level machine settings,
        so they live on the hub's ``user`` row (the AuthService reads it), not
        on this vault's legacy ``user`` table.

        Returns:
            ``{"repo_path", "python"}`` (empty values if unset, or
            ``{}`` when no owner row or AuthService is attached yet).
        """
        auth = self.auth_service
        if auth is None:
            return {}
        try:
            user = auth.get_user()
        except Exception as exc:
            logger.warning("Could not read ai-toolkit owner config: %s", exc)
            return {}
        if user is None:
            return {}
        return {
            "repo_path": user.ai_toolkit_repo_path,
            "python": user.ai_toolkit_python,
        }

    def _import_training_run_under_lease(self, run_dir: str, info: dict) -> dict:
        """Import a finished run into the active library, leased only to attach it.

        The run starts only while this vault is the active library and no
        restore or switch holds admission. Its checkpoints then go to the shelf
        with no lease held: the shelf is the hub's, which a restore or a switch
        does not replace, and hashing and copying a large LoRA can outlast the
        time a switch waits for its readers. The attachment to the run's set or
        character writes this vault's database, so it alone is leased.

        Refused at the attachment, the LoRA is on the shelf and unattached. The
        caller asks again, and :meth:`_import_training_run` answers that retry
        from the shelf and attaches it then.

        Args:
            run_dir: The ai-toolkit run folder holding the checkpoints.
            info: The run's fields, as :meth:`_import_training_run`.

        Returns:
            What :meth:`_import_training_run` returns.

        Raises:
            LibraryNotReadyError: A restore or a switch holds admission (SWITCHING:
                the caller asks again), or no coherent library is open
                (UNAVAILABLE). Raised at the start nothing was imported; raised
                at the attachment the LoRA is on the shelf, unattached.
            NotTheActiveLibraryError: Another vault is the active library.
        """
        coordinator = self.library_coordinator
        if coordinator is None:
            logger.info(
                "Training run %s imports without a library lease: no coordinator "
                "is attached to the vault at %s.",
                info.get("id"),
                self.image_root,
            )
            return self._import_training_run(run_dir, info)
        # Admitted, then given back at once: the lease answers whether the run
        # may start, and is not held over the copy.
        # A lease is granted only while admission is READY: that is the state
        # the refusal below reports, not ``coordinator.state`` read afterwards
        # outside its lock, where a switch begun since would read as SWITCHING.
        lease = coordinator.acquire_read_or_raise()
        try:
            if lease.vault is not self:
                raise NotTheActiveLibraryError(SwitchState.READY)
        finally:
            coordinator.release_read(lease)
        result = self._import_training_run(run_dir, info, attach=False)
        cover_sha = result.get("cover_sha256")
        if cover_sha is None:
            return result
        lease = coordinator.acquire_read_or_raise()
        try:
            if lease.vault is not self:
                raise NotTheActiveLibraryError(SwitchState.READY)
            _move_trained_attachment(self, info, None, cover_sha)
        finally:
            coordinator.release_read(lease)
        return result

    def _import_training_run(
        self, run_dir: str, info: dict, *, attach: bool = True
    ) -> dict:
        """Bring a finished ai-toolkit run onto the model shelf as the job's LoRA.

        The one path a trained LoRA takes onto the shelf: the run importer, into
        the download folder's ``loras/`` subfolder when it has one (the rule a
        LoRA download follows), with provenance ``trained``. The cover (stack
        position 0) is then attached to the job's character or set.

        A first run whose bare final is on the shelf already (an earlier import
        of it was cut short before its end state or its attachment) answers from
        the shelf and imports nothing: its cover is the head of its stack.

        Args:
            run_dir: The ai-toolkit run folder holding the checkpoints.
            info: The run's fields (``id``, ``name``, ``set_id``,
                ``character_id``). ``id`` is the hub ``training_run`` row every
                imported model is stamped with.
            attach: Attach the cover to the run's set or character. False when
                the caller attaches it under a lease of its own.

        Returns:
            ``{"stack_id", "imported", "cover_sha256"}``. ``cover_sha256`` is
            ``None`` when the cover itself did not import.

        Raises:
            RuntimeError: No shelf or download folder to receive the run, or no
                checkpoint of it imported.
            MoveRefused: The importer refused the batch before writing anything.
        """
        hub = self._hub
        if hub is None:
            raise RuntimeError("No model shelf is open to receive the trained LoRA.")
        if info.get("continued_from_step") is None:
            with SHELF_IO_LOCK:
                final = hub.fetchone(
                    "SELECT sha256, stack_id FROM model "
                    "WHERE training_run_id = ? AND training_step IS NULL",
                    (info["id"],),
                )
                if final is not None:
                    imported = hub.fetchone(
                        "SELECT COUNT(*) AS n FROM model WHERE training_run_id = ?",
                        (info["id"],),
                    )["n"]
                    head = (
                        None
                        if final["stack_id"] is None
                        else hub.fetchone(
                            "SELECT sha256 FROM model "
                            "WHERE training_run_id = ? AND stack_id = ? "
                            "ORDER BY stack_position IS NULL, stack_position, id "
                            "LIMIT 1",
                            (info["id"], final["stack_id"]),
                        )
                    )
            if final is not None:
                cover_sha = final["sha256"] if head is None else head["sha256"]
                logger.info(
                    "Training run %s (%s) is on the shelf already (stack %s, %d "
                    "file(s)); nothing is imported again.",
                    info["id"],
                    info.get("name"),
                    final["stack_id"],
                    imported,
                )
                if attach:
                    _move_trained_attachment(self, info, None, cover_sha)
                return {
                    "stack_id": final["stack_id"],
                    "imported": imported,
                    "cover_sha256": cover_sha,
                }
        folder = find_download_folder(hub)
        if folder is None:
            raise RuntimeError("No model folder is set up to receive the trained LoRA.")
        subfolder = (
            "loras" if os.path.isdir(os.path.join(folder["path"], "loras")) else None
        )
        # The run sits in the shelf's ai-toolkit folder, so that folder's
        # `delete_after_import` decides whether its files stay, exactly as for
        # an import the owner starts from the shelf.
        run_real = os.path.realpath(run_dir)
        delete_source = False
        for row in hub.fetchall(
            "SELECT path, delete_after_import FROM model_folder WHERE kind = 'source'"
        ):
            root = os.path.realpath(row["path"])
            try:
                under_root = os.path.commonpath([root, run_real]) == root
            except ValueError as exc:
                # Windows: a source folder on another drive has no common path
                # with the run, so it cannot be the run's folder. Logged, since
                # it is the reason this source is skipped.
                logger.warning(
                    "Source folder %s and run folder %s share no common path (%s); "
                    "the source is not the run's folder.",
                    root,
                    run_real,
                    exc,
                )
                under_root = False
            if under_root:
                delete_source = bool(row["delete_after_import"])
                break
        if info.get("continued_from_step") is not None and hub.fetchone(
            "SELECT 1 FROM model WHERE training_run_id = ?", (info["id"],)
        ):
            return self._import_trained_longer(
                run_dir, info, folder, subfolder, delete_source
            )
        with SHELF_IO_LOCK:
            report = RunImporter(hub).import_run(
                run_dir,
                folder["id"],
                subfolder=subfolder,
                delete_source=delete_source,
                training_run_id=info["id"],
            )
        landed = [o for o in report.outcomes if o.status == STATUS_IMPORTED]
        if not landed:
            raise RuntimeError(f"No checkpoint of {report.run_name} could be imported.")

        # Nothing is cancelled here, so outcomes[0] is the cover (position 0),
        # unless the owner chose a step in the live view while it trained. The
        # row is read again: the one the worker claimed predates that choice.
        cover = report.outcomes[0]
        run = (
            training_runs.get_run_by_uuid(hub, info["uuid"])
            if info.get("uuid")
            else None
        )
        chosen = run.get("cover_step") if run else None
        if chosen is not None:
            final_step = run.get("total_steps")
            picked = next(
                (
                    o
                    for o in report.outcomes
                    if o.status == STATUS_IMPORTED
                    and o.model_id is not None
                    and (o.step == chosen or (o.step is None and chosen == final_step))
                ),
                None,
            )
            if picked is None:
                logger.warning(
                    "Step %s chosen as the cover of %s has no imported checkpoint "
                    "(pruned by the trainer?); the final stays the cover.",
                    chosen,
                    report.run_name,
                )
            elif picked is not cover:
                set_cover(hub, report.stack_id, picked.model_id)
                cover = picked
        cover_sha = None
        if cover.status == STATUS_IMPORTED and cover.model_id is not None:
            cover_row = hub.fetchone(
                "SELECT sha256 FROM model WHERE id = ?", (cover.model_id,)
            )
            if cover_row is None:
                # The files have landed, so do not fail the run for a row that
                # vanished after the import: the cover just has no hash to attach.
                logger.warning(
                    "Cover model %s of %s landed but its row is gone; its hash is "
                    "not attached to the job's character or set.",
                    cover.model_id,
                    report.run_name,
                )
            else:
                cover_sha = cover_row["sha256"]
        else:
            logger.warning(
                "The cover of %s (%s) did not import; no LoRA is attached to %s.",
                report.run_name,
                cover.filename,
                info.get("name"),
            )

        if cover_sha is not None and attach:
            _move_trained_attachment(self, info, None, cover_sha)

        return {
            "stack_id": report.stack_id,
            "imported": len(landed),
            "cover_sha256": cover_sha,
        }

    def _import_trained_longer(
        self,
        run_dir: str,
        info: dict,
        folder,
        subfolder: Optional[str],
        delete_source: bool,
    ) -> dict:
        """Add a run trained longer to the stack its first import made.

        The shelf's own copies give their names up first, as the run folder's
        did: the bare final becomes its stamped step, so the longer-trained
        final takes ``<stem>.safetensors`` and everything that loads that name
        gets it. After a rewind, the copies past the step it trained on from
        are renamed into the take that set them aside (``<stem>_take-1_<step>``),
        because the new take writes the same steps again. Only the steps past
        that one and the new final are imported, after the stack's members, so
        the cover stays where the owner put it.

        Returns:
            ``{"stack_id", "imported", "cover_sha256"}`` as
            :meth:`_import_training_run`; ``cover_sha256`` is None, since the
            cover and its attachment are left as they are.

        Raises:
            RuntimeError: No checkpoint imported, or a shelf copy could not be
                renamed out of the way.
            MoveRefused: The importer refused the batch.
        """
        hub = self._hub
        from_step = int(info["continued_from_step"])
        take = info.get("set_aside_take")
        stem = os.path.basename(os.path.normpath(run_dir))
        with SHELF_IO_LOCK:
            locations = hub.fetchall(
                "SELECT mf.relpath, m.id AS model_id, m.training_step "
                "FROM model_file mf JOIN model m ON m.id = mf.model_id "
                "WHERE mf.model_folder_id = ? AND m.training_run_id = ?",
                (folder["id"], info["id"]),
            )
            for location in locations:
                name = os.path.basename(location["relpath"])
                step = location["training_step"]
                if step is None:
                    continue
                if take and step > from_step:
                    renamed = f"{stem}_{take}_{step:09d}.safetensors"
                elif name == f"{stem}.safetensors":
                    renamed = f"{stem}_{step:09d}.safetensors"
                else:
                    continue
                if renamed == name:
                    continue
                source = os.path.join(folder["path"], location["relpath"])
                relpath = os.path.join(os.path.dirname(location["relpath"]), renamed)
                relpath = relpath.replace(os.sep, "/")
                try:
                    publish_no_clobber(source, os.path.join(folder["path"], relpath))
                except OSError as exc:
                    logger.error(
                        "Could not rename %s to %s on the shelf for run %s: %s",
                        source,
                        renamed,
                        info["uuid"],
                        exc,
                    )
                    raise RuntimeError(
                        f"The shelf's copy of {name} could not make room for the "
                        "longer-trained file."
                    ) from exc
                try:
                    with hub.transaction() as conn:
                        conn.execute(
                            "UPDATE model_file SET relpath = ? "
                            "WHERE model_folder_id = ? AND relpath = ?",
                            (relpath, folder["id"], location["relpath"]),
                        )
                        conn.execute(
                            "UPDATE model SET filename = ? WHERE id = ?",
                            (renamed, location["model_id"]),
                        )
                except Exception:
                    # The rows still name the old file: give it its name back.
                    try:
                        os.rename(os.path.join(folder["path"], relpath), source)
                    except OSError as undo_exc:
                        logger.error(
                            "Could not rename %s back to %s after the shelf rows "
                            "for run %s refused the new name (%s); the model's "
                            "file is missing under its recorded name.",
                            relpath,
                            source,
                            info["uuid"],
                            undo_exc,
                        )
                    raise
            stack = hub.fetchone(
                "SELECT stack_id, MAX(stack_position) AS last FROM model "
                "WHERE training_run_id = ? AND stack_id IS NOT NULL "
                "GROUP BY stack_id ORDER BY COUNT(*) DESC LIMIT 1",
                (info["id"],),
            )
            steps = [
                c.step
                for c in read_run(run_dir).checkpoints
                if c.step is None or c.step > from_step
            ]
            report = RunImporter(hub).import_run(
                run_dir,
                folder["id"],
                steps=steps,
                subfolder=subfolder,
                delete_source=delete_source,
                training_run_id=info["id"],
                stack_id=stack["stack_id"] if stack else None,
                first_position=(stack["last"] or 0) + 1 if stack else 0,
            )
        landed = [o for o in report.outcomes if o.status == STATUS_IMPORTED]
        if not landed:
            raise RuntimeError(f"No checkpoint of {report.run_name} could be imported.")
        return {
            "stack_id": report.stack_id,
            "imported": len(landed),
            "cover_sha256": None,
        }

    def choose_training_cover(self, run: dict, step: int) -> bool:
        """Make the checkpoint at *step* the cover of a trained run on the shelf.

        The stack's cover moves (``stack_detector.set_cover``) and the LoRA
        attached to the run's set or character moves with it, so the set uses
        the step the owner chose rather than the last one. A run not on the
        shelf yet is left to its import, which reads ``cover_step``.

        Args:
            run: The ``training_run`` row (``id``, ``total_steps``, ``set_id``,
                ``character_id``).
            step: The step chosen; the run's last step names its bare final.

        Returns:
            True when the shelf was changed, False when the run has no models
            on it yet.

        Raises:
            LookupError: The run is on the shelf but kept no checkpoint at
                *step*.
        """
        hub = self._hub
        if hub is None:
            return False
        rows = hub.fetchall(
            "SELECT id, sha256, stack_id, stack_position, training_step FROM model "
            "WHERE training_run_id = ? ORDER BY stack_position IS NULL, "
            "stack_position, id",
            (run["id"],),
        )
        if not rows:
            return False
        target = next(
            (
                row
                for row in rows
                if row["training_step"] == step
                or (row["training_step"] is None and step == run.get("total_steps"))
            ),
            None,
        )
        if target is None or target["stack_id"] is None:
            raise LookupError(f"No checkpoint of this run was kept at step {step}.")
        current = next(
            (
                row
                for row in rows
                if row["stack_id"] == target["stack_id"] and row["stack_position"] == 0
            ),
            None,
        )
        set_cover(hub, target["stack_id"], target["id"])
        old_sha = current["sha256"] if current is not None else None
        if target["sha256"] and target["sha256"] != old_sha:
            _move_trained_attachment(self, run, old_sha, target["sha256"])
        return True

    def _request_training_gpu(
        self, label: str, job_uuid: str, vram_mb: int, ram_mb: int
    ) -> Lease:
        """Ask the arbiter for a training run's card: the one PixlStash uses.

        Exclusive, so PixlStash yields its own GPU work first and an idle
        ComfyUI frees its cache; the run's own process then has the card. The
        needs are those of the run's resolved memory profile.
        """
        self._watch_comfyui()
        return self._arbiter.request(
            Tenant.AITOOLKIT,
            label,
            need_mb=vram_mb,
            ram_mb=ram_mb,
            kind=LeaseKind.EXCLUSIVE,
            device=ledger().pixlstash_device(),
            job_id=job_uuid,
        )

    # Whether the last _watch_comfyui found the saved address refused, so its
    # warning is said once per change rather than once per training run.
    _comfyui_address_refused = False

    def _watch_comfyui(self) -> None:
        """Point the arbiter's ComfyUI probe at the owner's ComfyUI.

        Unpointed, the arbiter sees no ComfyUI: a run queued by Train longer
        or resumed at start-up, before any ComfyUI run or the training dialog
        pointed it, then never has ComfyUI cleared or offers to quit it, and
        an adopted run counts ComfyUI's memory as other apps'.
        """
        auth = self.auth_service
        if auth is None:
            logger.warning(
                "No owner attached yet; the training run waits without "
                "ComfyUI's memory in view"
            )
            return
        try:
            user = auth.get_user()
        except Exception as exc:
            logger.warning(
                "Could not read the owner's ComfyUI address; the training run "
                "waits without ComfyUI's memory in view: %s",
                exc,
            )
            return
        try:
            url = comfyui_connections.current_base(user, None)
        except SavedComfyUIAddressInvalid:
            # Nothing is sent to an address the normaliser refuses, and the
            # run goes ahead without ComfyUI's memory in view, as with none
            # saved. request_base has logged the connection id.
            if not self._comfyui_address_refused:
                logger.warning(
                    "The saved ComfyUI address is not valid, so training runs "
                    "wait without ComfyUI's memory in view until it is set "
                    "again in Settings"
                )
            self._comfyui_address_refused = True
            return
        if self._comfyui_address_refused:
            logger.info(
                "The saved ComfyUI address was set again or cleared; it is "
                "no longer refused"
            )
        self._comfyui_address_refused = False
        if url:
            comfyui_probe(RUN_CLIENT_ID, url)

    def _adopt_training_gpu(
        self, label: str, job_uuid: str, vram_mb: int, ram_mb: int
    ) -> Lease:
        """Grant a trainer that outlived the restart its card at once.

        See :meth:`ResourceArbiter.adopt`: exclusive and unshrinkable for
        ``ADOPT_QUIET_S``, so PixlStash's own work keeps off the card.
        """
        self._watch_comfyui()
        return self._arbiter.adopt(
            Tenant.AITOOLKIT,
            label,
            vram_mb,
            ledger().pixlstash_device(),
            quiet_s=ADOPT_QUIET_S,
            ram_mb=ram_mb,
            job_id=job_uuid,
        )

    def training_room(self) -> dict:
        """What a training run could be given now: ``{vram_mb, ram_mb}``.

        On the card training leases (PixlStash's own); see
        :meth:`ResourceArbiter.training_room`.
        """
        return self._arbiter.training_room(ledger().pixlstash_device())

    def set_max_vram_usage_gb(self, max_vram_gb: Optional[float]):
        """The most PixlStash's own GPU work may hold, and its batch sizing.

        Caps the PIXLSTASH tenant's granted leases in the arbiter; ComfyUI and
        training are not limited by it. The engine sizes batches from it too.
        """
        self._max_vram_gb = max_vram_gb
        if self._task_runner is not None:
            self._arbiter.set_pixlstash_cap_gb(max_vram_gb)
        if self._engine:
            self._engine.set_max_vram_usage_gb(max_vram_gb)

    def set_tagger_settings(self, settings: dict) -> None:
        """Replace the full tagger plugin settings and propagate to the engine."""

        settings = get_tagger_plugin_manager().fill_defaults(settings)
        self._tagger_settings = settings
        # Keep legacy fields in sync so existing code paths work unchanged.
        plugins = settings.get("plugins", {})
        wd14_cfg = plugins.get("wd14", {})
        pixl_cfg = plugins.get("pixlstash_tagger", {})
        self._wd14_tagger_enabled = bool(wd14_cfg.get("enabled", False))
        self._pixlstash_tagger_enabled = bool(pixl_cfg.get("enabled", False))
        wd14_threshold = wd14_cfg.get("params", {}).get("threshold")
        if wd14_threshold is not None:
            self._wd14_threshold = float(wd14_threshold)
        threshold_offset = pixl_cfg.get("params", {}).get("threshold_offset")
        if threshold_offset is not None:
            self._pixlstash_tagger_threshold_offset = float(threshold_offset)
        self.db.set_tag_exclusions(
            exclusions_from_tagger_settings(settings),
            generation=exclusion_generation(settings),
        )
        if self._engine:
            self._engine.set_tagger_settings(settings)

    def set_wd14_tagger_enabled(self, enabled: bool):
        self._wd14_tagger_enabled = bool(enabled)
        if self._engine:
            self._engine.set_wd14_tagger_enabled(self._wd14_tagger_enabled)

    def set_pixlstash_tagger_enabled(self, enabled: bool):
        self._pixlstash_tagger_enabled = bool(enabled)
        if self._engine:
            self._engine.set_pixlstash_tagger_enabled(self._pixlstash_tagger_enabled)

    def set_wd14_threshold(self, threshold: Optional[float]):
        self._wd14_threshold = threshold
        if self._engine:
            value = threshold if threshold is not None else 0.85
            self._engine.set_wd14_threshold(value)

    def set_pixlstash_tagger_threshold_offset(self, offset: Optional[float]):
        self._pixlstash_tagger_threshold_offset = offset
        if self._engine:
            value = offset if offset is not None else 0.0
            self._engine.set_pixlstash_tagger_threshold_offset(value)

    def get_pixlstash_tagger_threshold_offset(self) -> float:
        return self._pixlstash_tagger_threshold_offset or 0.0

    def get_pixlstash_tagger_meta_path(self):
        if self._engine is None:
            return None
        tagger = self._engine.pixlstash_tagger_service
        return tagger.meta_path if tagger is not None else None

    def get_pixlstash_acceptance_threshold(self) -> float:
        from pixlstash.tagger_plugins.pixlstash_tagger import (
            PIXLSTASH_TAGGER_DEFAULT_THRESHOLD,
        )

        bias = self._pixlstash_tagger_threshold_offset or 0.0
        return max(0.01, float(PIXLSTASH_TAGGER_DEFAULT_THRESHOLD) + bias)

    def retag_picture_interactive(
        self, picture_id: int, engine_name: str | None = None
    ) -> None:
        if self._engine is None:
            return
        from pixlstash.tasks.tag_task import TagTask

        def _fetch_pic(session: Session):
            return session.get(Picture, picture_id)

        reset_generation = self.db.tag_resets.current()
        pic = self.db.run_immediate_read_task(_fetch_pic)
        if pic is not None:
            task = TagTask(
                database=self.db,
                tagging_workflow=self._engine.tagging_workflow,
                pictures=[pic],
                interactive=True,
                engine_override=engine_name,
                reset_generation=reset_generation,
            )
            self.submit_task(task)

    def redescribe_picture_interactive(
        self, picture_id: int, engine_name: str | None = None
    ) -> None:
        """Queue an immediate description-generation pass for a single picture.

        Args:
            picture_id: Primary key of the picture to describe.
            engine_name: Optional plugin name to use for this picture.  When
                ``None``, the current ``active_description_plugin`` setting is used.
        """
        if self._engine is None:
            return
        from pixlstash.tasks.description_task import DescriptionTask

        def _fetch_pic(session: Session):
            return session.get(Picture, picture_id)

        pic = self.db.run_immediate_read_task(_fetch_pic)
        if pic is not None:
            task = DescriptionTask(
                database=self.db,
                workflow=self._engine.description_workflow,
                pictures=[pic],
                engine_override=engine_name,
                interactive=True,
            )
            self.submit_task(task)

    def reset_description_interactive(
        self,
        picture_id: int,
        engine_name: str | None = None,
        origin_client_id: str | None = None,
    ) -> bool:
        """Request a fresh description pass for a picture by writing a sentinel.

        Sets ``description`` to a ``__description::<engine>`` sentinel value
        (or ``__description::`` for the default plugin).  The
        :class:`~pixlstash.tasks.missing_description_finder.MissingDescriptionFinder`
        picks up pictures with this sentinel and creates a
        :class:`~pixlstash.tasks.description_task.DescriptionTask` with the
        appropriate engine override.  Using a sentinel instead of ``NULL``
        prevents the background finder from racing with an interactive request
        and substituting the wrong plugin's output.

        Args:
            picture_id: Primary key of the picture to reset.
            engine_name: Optional plugin name to embed in the sentinel.
            origin_client_id: Opaque ``X-Client-Id`` of the originating tab,
                threaded from the route handler so the CHANGED_PICTURES echo is
                recognised as the user's own change (targeted reconcile, no pill)
                instead of being treated as external.

        Returns:
            ``True`` if the picture was found and updated, ``False`` otherwise.
        """
        sentinel = make_description_sentinel(engine_name)

        def _set_sentinel(session: Session) -> bool:
            pic = session.get(Picture, picture_id)
            if pic is None:
                return False
            # Resetting the description overwrites frozen label/curation data with
            # a redescribe sentinel - refuse on a picture in a locked set.
            enforce_pictures_not_locked(
                session, [picture_id], "reset the description of a locked picture"
            )
            pic.description = sentinel
            session.commit()
            return True

        found = self.db.run_task(_set_sentinel)
        if not found:
            return False
        self.notify(
            EventType.CHANGED_PICTURES,
            {
                "picture_ids": [picture_id],
                "origin_client_id": origin_client_id,
                "change_kind": "updated",
            },
        )
        self.redescribe_picture_interactive(picture_id, engine_name=engine_name)
        return True

    def reset_descriptions(
        self,
        picture_ids: list[int],
        engine_name: str | None = None,
        origin_client_id: str | None = None,
    ) -> int:
        """Queue a fresh description pass for many pictures in one write.

        Writes the ``__description::<engine>`` sentinel on every picture in one
        transaction and wakes the planner; ``MissingDescriptionFinder`` then
        captions them in properly sized batches. Unlike
        :meth:`reset_description_interactive` no per-picture task is submitted:
        one urgent single-image task per picture ran the captioner N times at
        batch size 1 while the finder, which cannot tell those sentinels from
        its own, captioned the same pictures a second time (#1162).

        Args:
            picture_ids: Primary keys of the pictures to reset.
            engine_name: Optional plugin name to embed in the sentinel.
            origin_client_id: Opaque ``X-Client-Id`` of the originating tab.

        Returns:
            How many pictures were found and reset.
        """
        ids = sorted({int(pid) for pid in picture_ids})
        if not ids:
            return 0
        sentinel = make_description_sentinel(engine_name)

        def _set_sentinels(session: Session) -> list[int]:
            found: list[int] = []
            # Chunked: "all pictures" can exceed SQLite's bound-parameter cap.
            for chunk in chunked(ids):
                enforce_pictures_not_locked(
                    session, chunk, "reset the description of a locked picture"
                )
                present = list(
                    session.exec(select(Picture.id).where(Picture.id.in_(chunk))).all()
                )
                if not present:
                    continue
                session.exec(
                    update(Picture)
                    .where(Picture.id.in_(present))
                    .values(description=sentinel, metadata_hash=None)
                )
                found.extend(present)
            session.commit()
            return found

        reset_ids = self.db.run_task(_set_sentinels)
        if not reset_ids:
            return 0
        self.notify(
            EventType.CHANGED_PICTURES,
            {
                "picture_ids": reset_ids,
                "origin_client_id": origin_client_id,
                "change_kind": "updated",
            },
        )
        return len(reset_ids)

    def generate_text_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate a text embedding using InferenceEngine.

        Args:
            text (str): Input text to generate embedding for.

        Returns:
            Optional[np.ndarray]: Generated text embedding or None if failed.
        """
        if self._engine is None:
            return None
        embedding = self._engine.text_embedding_workflow.encode_query(query)
        return embedding[0] if embedding is not None and len(embedding) > 0 else None

    def generate_clip_text_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate a CLIP text embedding for the provided query text.
        """
        if self._engine is None:
            return None
        return self._engine.text_embedding_workflow.encode_clip_query(query)

    def set_description(self, description: str):
        def op(session: Session):
            metadata = session.exec(
                select(MetaData).where(
                    MetaData.schema_version == MetaData.CURRENT_SCHEMA_VERSION
                )
            ).first()
            if metadata is None:
                metadata = MetaData(
                    schema_version=MetaData.CURRENT_SCHEMA_VERSION,
                    description=description,
                )
            else:
                metadata.description = description
            session.add(metadata)
            session.commit()

        self.db.submit_task(op, priority=DBPriority.IMMEDIATE)

    def runs_background_work(self) -> bool:
        """Whether this vault has a task runner to take background tasks.

        ``False`` when the server started with ``disable_background_workers``:
        :meth:`submit_task` then returns ``None`` and nothing runs. A caller
        that must refuse before it spends anything (a run's confirm token, a
        recorded run row) asks this first. A runner that has stopped still
        answers ``True``; its :meth:`submit_task` then returns ``None``.
        """
        return self._task_runner is not None

    def refuse_while_admission_paused(self) -> None:
        """Raise ``TaskAdmissionRefused`` now if a restore has paused admission.

        For a request that spends something before it submits its task (a
        purge's single-use confirm token): asked first, the refusal comes
        before the spend. A look only; see ``TaskRunner.refuse_while_paused``.
        Nothing to refuse without a runner.

        Raises:
            TaskAdmissionRefused: Admission is paused.
        """
        if self._task_runner is not None:
            self._task_runner.refuse_while_paused()

    def submit_task(self, task):
        """Submit an in-memory task to the shared task runner.

        Returns:
            The task id, or None when nothing will run it: there is no runner
            (``disable_background_workers``), or it has stopped (a shutdown, a
            library switch, or a ``start()`` refused beside a worker that
            outlived the last stop). Every caller takes None as "not taken";
            the stop is logged here, once, with the task.

        Raises:
            RuntimeError: A dead worker's replacement thread could not start
                (see ``TaskRunner.submit``). Not a stop, so not None.
            TaskAdmissionRefused: The task is a person's and a library restore
                has paused admission (see ``TaskRunner.submit``). Not None
                either: the caller undoes what it set up for the task, and the
                person is told to try again (the server answers 409).
        """
        if self._task_runner is None:
            return None
        try:
            return self._task_runner.submit(task)
        except TaskRunnerStopped as exc:
            logger.warning(
                "Task %s (%s) not queued: the task runner has stopped (%s).",
                task.id,
                task.type,
                exc,
            )
            return None

    @staticmethod
    def _release_task_library_lease(task, error):
        """Give back the library read lease a task holds, however it settled.

        A staged import holds one from commit until it ends, so a library
        switch waits for it (see ``PictureImportTask.release_library_lease``).
        """
        release = getattr(task, "release_library_lease", None)
        if callable(release):
            release()

    def _on_task_completed(self, task, error):
        if error is not None or task.status != TaskStatus.COMPLETED:
            return

        result = task.result if isinstance(task.result, dict) else {}

        if task.type == "ReferenceFolderScanTask":
            imported_ids = result.get("imported_picture_ids") or []
            if imported_ids:
                self.notify(EventType.CHANGED_PICTURES, imported_ids)
                self.notify(EventType.PICTURE_IMPORTED, imported_ids)
            # A followed move changed file_path on an existing row: the grid has
            # to refetch it, but nothing was imported, so no PICTURE_IMPORTED.
            moved_ids = result.get("moved_picture_ids") or []
            if moved_ids:
                self.notify(EventType.CHANGED_PICTURES, moved_ids)
            # v1.11 Phase 5: moves this scan could not attribute to PixlStash
            # itself, in a root laid out enough to have something to say about
            # them. The sidebar strip listens for this rather than polling
            # GET /moves/pending.
            queued_for_review = result.get("external_moves_queued_for_review") or []
            if queued_for_review:
                self.notify(EventType.EXTERNAL_MOVES_PENDING, queued_for_review)
            picture_ids = result.get("caption_updated_picture_ids") or []
            if picture_ids:
                self._queue_changed_tags_notification(picture_ids)
            return

        if task.type == "SmartScoreTask":
            changed_count = int(result.get("changed_count") or 0)
            picture_ids = list(task.params.get("picture_ids") or [])
            persisted_ids = [
                int(pid)
                for pid in (result.get("persisted_ids") or [])
                if pid is not None
            ]
            if changed_count > 0 and picture_ids:
                smart_score_changes = [
                    (Picture, int(pic_id), "smart_score", None)
                    for pic_id in picture_ids
                    if pic_id is not None
                ]
                if smart_score_changes:
                    self._notify_worker_ids_processed(
                        TaskType.SMART_SCORE,
                        smart_score_changes,
                    )
                # INTERACTIVE path: rescored ids that a user tag edit invalidated get an
                # immediate, origin-stamped CHANGED_PICTURES so the initiating tab
                # reconciles the card in place - independent of the global drain gate
                # below. Only *persisted* ids are consulted, so an id the CAS in
                # _persist_scores skipped (still NULL) is never announced as rescored.
                # Ids that were never registered (background-only rescores, or interactive
                # ids demoted when the registry was full) are left to the bulk drain emit.
                announced_ids: set[int] = set()
                for origin, ids in self.interactive_rescore_registry.consume(
                    persisted_ids
                ).items():
                    announced_ids.update(ids)
                    self.notify(
                        EventType.CHANGED_PICTURES,
                        {
                            "picture_ids": sorted(ids),
                            "fields": ["smart_score"],
                            "origin_client_id": origin,
                        },
                    )
                remaining = int(
                    self.db.run_immediate_read_task(SmartScoreTask.count_remaining) or 0
                )
                if remaining == 0:
                    # Exclude ids already announced origin-stamped above: re-announcing
                    # them origin-less here would raise the "view changed externally" pill
                    # on the very tab that made the edit (a self-pill). Unregistered
                    # ids - genuine background rescores, or interactive ids demoted when
                    # the registry was full - still ride this bulk emit so their cards
                    # refresh; over-suppressing them would strand real external changes.
                    drain_ids = [
                        pid for pid in picture_ids if int(pid) not in announced_ids
                    ]
                    if drain_ids:
                        # Tag the event with the changed field so the SPA only
                        # reloads the grid when it is actually sorting/filtering by
                        # smart_score; under any other sort this change is invisible.
                        self.notify(
                            EventType.CHANGED_PICTURES,
                            {"picture_ids": drain_ids, "fields": ["smart_score"]},
                        )
                    else:
                        logger.debug(
                            "SmartScoreTask drained with all %s rescored id(s) announced "
                            "origin-stamped; no origin-less drain event emitted.",
                            len(announced_ids),
                        )
                else:
                    logger.debug(
                        "SmartScoreTask updated %s pictures; deferring CHANGED_PICTURES event with %s smart scores remaining.",
                        changed_count,
                        remaining,
                    )
            return

        if task.type == "PictureImportTask":
            imported_ids = result.get("imported_picture_ids") or []
            origin_client_id = result.get("origin_client_id") or task.params.get(
                "origin_client_id"
            )
            # Mirror the streaming-upload import: a genuine PixlStash tab carries
            # an origin_client_id (slick in-place insert); an external caller does
            # not (raise the New-pictures pill instead).
            import_source = "ui" if origin_client_id else "external"
            self.notify(
                EventType.CHANGED_PICTURES,
                {
                    "picture_ids": imported_ids,
                    "source": import_source,
                    "origin_client_id": origin_client_id,
                    "change_kind": "added",
                },
            )
            if imported_ids:
                self.notify(
                    EventType.PICTURE_IMPORTED,
                    {
                        "ids": imported_ids,
                        "source": import_source,
                        "origin_client_id": origin_client_id,
                        "change_kind": "added",
                    },
                )
            return

        if task.type == "DetectionTask":
            picture_ids = result.get("picture_ids") or []
            if picture_ids:
                # Attribute to the originating tab and tag the change as the
                # (grid-invisible) "detections" field, so the SPA refreshes
                # detection overlays without raising the external-change pill or
                # reshuffling the grid.
                self.notify(
                    EventType.CHANGED_PICTURES,
                    {
                        "picture_ids": picture_ids,
                        "change_kind": "updated",
                        "fields": ["detections"],
                        "origin_client_id": task.params.get("origin_client_id"),
                    },
                )
            return

        if task.type == "FaceModelRefreshTask" and task.params.get("redetect"):
            picture_ids = result.get("picture_ids") or []
            if picture_ids:
                # "faces" is card content: the grid re-reads the boxes in place.
                self.notify(
                    EventType.CHANGED_PICTURES,
                    {
                        "picture_ids": picture_ids,
                        "change_kind": "updated",
                        "fields": ["faces"],
                        "origin_client_id": task.params.get("origin_client_id"),
                        "new_faces": result.get("new_faces", 0),
                    },
                )
                self.notify(EventType.CHANGED_FACES)
            return

        changed = result.get("changed") if isinstance(result, dict) else None
        if not changed:
            return

        if task.type == "TagTask":
            self._notify_worker_ids_processed(TaskType.TAGGER, changed)
            picture_ids = [pic_id for _, pic_id, _, _ in changed]
            if picture_ids:
                self._queue_changed_tags_notification(picture_ids)
            return

        if task.type == "QualityTask":
            self._notify_worker_ids_processed(TaskType.QUALITY, changed)
            self.notify(EventType.QUALITY_UPDATED)
            return

        if task.type == "LikenessParametersTask":
            self._notify_worker_ids_processed(TaskType.LIKENESS_PARAMETERS, changed)
            return

        if task.type == "LikenessTask":
            self._notify_worker_ids_processed(TaskType.LIKENESS, changed)
            return

        if task.type == "FaceExtractionTask":
            # Do NOT fire CHANGED_FACES here: newly-extracted faces always
            # have character_id=None so no character sidebar data changes.
            # _process_pending_character_assignments fires CHANGED_FACES
            # itself only when it actually assigns a face to a character.
            self._after_face_writes_land(task, changed, result.get("picture_ids") or [])
            return

        if task.type == "DescriptionTask":
            self._notify_worker_ids_processed(TaskType.DESCRIPTION, changed)
            picture_ids = [pic_id for _, pic_id, _, _ in changed]
            self.notify(
                EventType.CHANGED_DESCRIPTIONS, picture_ids if picture_ids else None
            )
            return

        if task.type == "OcrTask":
            self._notify_worker_ids_processed(TaskType.OCR, changed)
            picture_ids = [pic_id for _, pic_id, _, _ in changed]
            if picture_ids:
                # Not visible in the grid and no sort reads it: the open
                # overlay refetches, nothing else moves.
                self.notify(
                    EventType.CHANGED_PICTURES,
                    {
                        "picture_ids": picture_ids,
                        "change_kind": "updated",
                        "fields": ["ocr_text"],
                    },
                )
            return

        if task.type == "TextEmbeddingTask":
            self._notify_worker_ids_processed(TaskType.TEXT_EMBEDDING, changed)
            return

        if task.type == "ImageEmbeddingTask":
            self._notify_worker_ids_processed(TaskType.IMAGE_EMBEDDING, changed)
            return

        if task.type == "WatchFolderImportTask":
            picture_ids = result.get("imported_picture_ids") or []
            if picture_ids:
                self.notify(EventType.CHANGED_PICTURES, picture_ids)
                self.notify(EventType.PICTURE_IMPORTED, picture_ids)

    def _after_face_writes_land(
        self, task: FaceExtractionTask, changed, picture_ids: list[int]
    ) -> None:
        """Announce face extraction and resolve pending people once its rows land.

        The task completes with its face rows still queued on the writer
        (``task.pending_writes``; see `FaceExtractionTask._run_task`). Doing
        either at completion acted on faces that were not there yet: a waiter
        was told they were in, and a pending person was dropped as if the
        picture had none. The last write's callback runs on the writer
        thread, so nothing on this path may wait for the writer.

        A write that failed put no rows down: its pictures stay unannounced
        and their pending people keep waiting for the extraction that lands.

        Until the match it queued has run, or it has settled without one, the
        completion is held in ``_face_followups`` for `stop` to wait on, so
        closing the vault does not drop a match the writer has not reached.
        """
        tail: Future = Future()
        with self._face_followups_lock:
            self._face_followups[tail] = list(picture_ids)
        writes = list(task.pending_writes)
        remaining = {"count": len(writes)}
        lock = threading.Lock()

        def _settled() -> None:
            match: Future | None = None
            try:
                # `Future.exception()` raises CancelledError on a cancelled
                # future, so ask `cancelled()` first.
                cancelled = [f for f in writes if f.cancelled()]
                failures = [
                    f.exception()
                    for f in writes
                    if not f.cancelled() and f.exception() is not None
                ]
                if cancelled or failures:
                    logger.warning(
                        "Face rows for pictures %s did not land (%d cancelled, "
                        "failures %s); not announcing them and leaving their "
                        "pending people waiting.",
                        picture_ids,
                        len(cancelled),
                        failures,
                    )
                    return
                self._notify_worker_ids_processed(TaskType.FACE_EXTRACTION, changed)
                if picture_ids:
                    match = self._process_pending_character_assignments(
                        picture_ids, wait=False
                    )
            finally:
                if match is None:
                    self._release_face_followup(tail)
                else:
                    # Runs after the match's announcement, added to it first.
                    match.add_done_callback(
                        lambda _match: self._release_face_followup(tail)
                    )

        def _one_landed(_future: Future) -> None:
            with lock:
                remaining["count"] -= 1
                last = remaining["count"] == 0
            if last:
                _settled()

        if not writes:
            _settled()
            return
        for future in writes:
            future.add_done_callback(_one_landed)

    def _release_face_followup(self, tail: Future) -> None:
        """Mark one face extraction's follow-up finished; `stop` stops waiting."""
        with self._face_followups_lock:
            self._face_followups.pop(tail, None)
        tail.set_result(None)

    def _wait_for_face_followups(self, timeout_s: float | None = None) -> int:
        """Wait for face extraction's follow-ups to finish, so closing the
        writer (or a restore swapping its file) does not drop a queued
        pending-person match. Never called from the writer thread, which is
        what finishes them.

        Args:
            timeout_s: How long to wait; ``None`` is ``FACE_FOLLOWUP_STOP_WAIT_S``,
                read when called.

        Returns:
            How many follow-ups are still unfinished.
        """
        if timeout_s is None:
            timeout_s = FACE_FOLLOWUP_STOP_WAIT_S
        with self._face_followups_lock:
            waiting = list(self._face_followups)
        if not waiting:
            return 0
        _, unfinished = concurrent.futures.wait(waiting, timeout=timeout_s)
        if not unfinished:
            return 0
        with self._face_followups_lock:
            left = sorted(
                {
                    picture_id
                    for tail in unfinished
                    for picture_id in self._face_followups.get(tail, [])
                }
            )
        logger.warning(
            "%s face extraction follow-up(s) unfinished after %.1f s, for "
            "pictures %s. A close goes ahead without their pending-person "
            "match, and those pictures keep their pending people; a restore "
            "refuses rather than replace the database under them.",
            len(unfinished),
            timeout_s,
            left,
        )
        return len(unfinished)

    def _process_pending_character_assignments(
        self, picture_ids: list[int], *, wait: bool = True
    ) -> Future:
        """Honour deferred face-to-character assignments after face extraction runs.

        When POST /characters/{id}/faces is called before face extraction has run
        for a picture, the endpoint stores the target character id in
        Picture.pending_character_id.  This method is called once the face rows
        of a FaceExtractionTask have landed and assigns the largest detected
        face for each such picture to the stored character, then clears the
        field.

        A picture of several people (a run loading their LoRAs) waits in
        ``PicturePendingPerson`` rows instead; its faces are matched to them
        by likeness (``match_faces_to_characters``) and the rows deleted.

        With ``wait=False`` (face extraction's path, which may be on the writer
        thread) the assignment is queued at HIGH and announced from its
        future instead of waited for.

        Returns:
            Future: the queued assignment. With *wait* it has already run and
            been announced; without, its announcement is the first callback
            added to it.
        """

        def _assign_if_pending(session: Session) -> bool:
            # Bound directly: *picture_ids* is one face-extraction batch
            # (FACE_EXTRACTION_BATCH_LIMIT, 100) or one ``chunked`` piece from
            # the folder-structure commit, under SQLite's parameter limit.
            pending = session.exec(
                select(Picture).where(
                    Picture.id.in_(picture_ids),
                    Picture.pending_character_id.is_not(None),
                )
            ).all()
            assigned_any = False
            for pic in pending:
                character_id = pic.pending_character_id
                pic.pending_character_id = None
                session.add(pic)
                faces = session.exec(
                    select(Face).where(
                        Face.picture_id == pic.id,
                        Face.face_index != -1,
                    )
                ).all()
                if not faces:
                    # Extraction ran but found no real faces; discard pending.
                    continue
                best_face = max(faces, key=lambda f: (f.width or 0) * (f.height or 0))
                if best_face.character_id != character_id:
                    best_face.character_id = character_id
                    session.add(best_face)
                    assigned_any = True
            people_by_picture: dict[int, list[int]] = {}
            for row in session.exec(
                select(PicturePendingPerson).where(
                    PicturePendingPerson.picture_id.in_(picture_ids)
                )
            ).all():
                people_by_picture.setdefault(row.picture_id, []).append(
                    row.character_id
                )
                session.delete(row)
            session.flush()
            for picture_id, people in people_by_picture.items():
                faces = Face.find(session, picture_id=picture_id)
                by_id = {face.id: face for face in faces}
                pairs = match_faces_to_characters(session, faces, sorted(people))
                for face_id, character_id in pairs.items():
                    by_id[face_id].character_id = character_id
                    session.add(by_id[face_id])
                    assigned_any = True
                logger.info(
                    "Picture %s of people %s: faces %s matched to them by likeness.",
                    picture_id,
                    sorted(people),
                    pairs,
                )
            session.commit()
            return assigned_any

        def _announce(future: Future) -> None:
            try:
                assigned = future.result()
            except Exception:
                logger.exception(
                    "Failed to process pending character assignments for pictures %s",
                    picture_ids,
                )
                return
            if assigned:
                self.notify(EventType.CHANGED_CHARACTERS)
                self.notify(EventType.CHANGED_FACES)

        future = self.db.submit_task(
            _assign_if_pending,
            priority=DBPriority.IMMEDIATE if wait else DBPriority.HIGH,
        )
        if wait:
            _announce(future)
        else:
            future.add_done_callback(_announce)
        return future

    def _notify_worker_ids_processed(self, worker_type: TaskType, changed):
        self._notify_planner_ids_processed(worker_type, changed)

    def _queue_changed_tags_notification(self, picture_ids: list[int]) -> None:
        with self._changed_tags_notify_lock:
            self._changed_tags_pending_ids.update(
                int(pid) for pid in picture_ids if pid is not None
            )
            if self._changed_tags_flush_timer is not None:
                return
            self._changed_tags_flush_timer = threading.Timer(
                self._changed_tags_flush_delay_s,
                self._flush_changed_tags_notification,
            )
            self._changed_tags_flush_timer.daemon = True
            self._changed_tags_flush_timer.start()

    def _flush_changed_tags_notification(self) -> None:
        with self._changed_tags_notify_lock:
            pending_ids = list(self._changed_tags_pending_ids)
            self._changed_tags_pending_ids.clear()
            self._changed_tags_flush_timer = None
        if pending_ids:
            self.notify(EventType.CHANGED_TAGS, pending_ids)

    def _planner_attr_current_value(self, cls: type, object_id: int, attr: str):
        def fetch(session: Session):
            obj = session.get(cls, object_id)
            if obj is None:
                return False, None
            value = getattr(obj, attr, None)
            if attr in {"faces", "tags"}:
                try:
                    return len(value or []) > 0, value
                except Exception:
                    # Relationship-length guard: a non-lenable value is simply
                    # "not populated"; (False, value) IS the answer.
                    return False, value
            return value is not None, value

        return self.db.run_task(fetch, priority=DBPriority.IMMEDIATE)

    def _notify_planner_ids_processed(self, worker_type: TaskType, changed):
        with self._planner_watchers_lock:
            for cls, object_id, attr, payload in changed:
                future = self._planner_watchers.pop(
                    (worker_type, cls, object_id, attr),
                    None,
                )
                if future:
                    future.set_result((object_id, payload))

    def get_worker_future(
        self, worker_type: TaskType, cls: type, object_id: int, attr: str
    ) -> "concurrent.futures.Future":
        """
        Returns a Future that will be set when the specified worker has processed the given object ID.
        Args:
            worker_type (TaskType): The type of worker to wait for.
        Returns:
            concurrent.futures.Future: Future set to True when completed.
        """
        if not self._engine and self._planner_finders is None:
            self._engine = self._create_engine()
            self._bind_engine_services()

        # Register the watcher BEFORE checking the DB to avoid a TOCTOU race where
        # the task completes (and fires _notify_planner_ids_processed) in the gap
        # between the "already done?" check and registering the watcher, causing the
        # notification to be silently dropped and the future to never resolve.
        future = Future()
        with self._planner_watchers_lock:
            self._planner_watchers[(worker_type, cls, object_id, attr)] = future

        # Double-check: if the value is already in the DB (either it was ready all
        # along, or the task completed before we registered the watcher), resolve
        # the future ourselves.  If _notify_planner_ids_processed already fired and
        # resolved it, the pop will return None and we skip the duplicate set_result.
        is_ready, payload = self._planner_attr_current_value(cls, object_id, attr)
        if is_ready:
            with self._planner_watchers_lock:
                popped = self._planner_watchers.pop(
                    (worker_type, cls, object_id, attr), None
                )
            if popped is not None:
                future.set_result((object_id, payload))

        return future

    def is_worker_running(self, worker_type: TaskType) -> bool:
        """Whether *worker_type* will actually be scheduled right now.

        Both conditions matter and used to be conflated: the planner thread has
        to be alive, and a finder for this task type has to still be registered
        with it. Ignoring the argument meant a dead planner and a detached
        finder produced the same answer, and callers reported whichever symptom
        they happened to name. The lane its tasks queue on also needs a live
        worker; `worker_unavailable_reason` holds all three checks.
        """
        return self.worker_unavailable_reason(worker_type) is None

    def worker_unavailable_reason(self, worker_type: TaskType) -> str | None:
        """Why *worker_type* is not schedulable, or None when it is.

        Callers that refuse a request need to say which condition failed; "the
        worker is not running" reads as a configuration choice and has sent
        every investigation of a dead planner thread down the wrong path.

        A lane whose workers have all exited is replaced here
        (`_replace_dead_lane`) rather than reported: a refusal that waited for
        the next ``submit()`` to replace it would itself hold back the
        submission that does. A replacement that cannot start is logged, and
        the lane is reported dead, never raised to the caller.
        """
        if not self._work_planner:
            return "the background work planner was never started"
        if not self._work_planner.is_running():
            return (
                "the background work planner thread is not alive (it was "
                "stopped, or it died); restart the server"
            )
        if not self._work_planner.has_finder(worker_type):
            return f"no {worker_type} finder is registered with the work planner"
        try:
            lane = self._replace_dead_lane(worker_type)
        except RuntimeError as exc:
            # replace_dead_workers raises when a replacement thread cannot start
            # (Thread.start). The runner logged that failure with its traceback;
            # the lane is read again, since a repair can fail on one slot after
            # another came back.
            lane = self._dead_lane(worker_type)
            logger.error(
                "Checking %s, the task runner could not start a replacement "
                "worker: %s: %s. %s",
                worker_type.value,
                type(exc).__name__,
                exc,
                f"Its {lane.upper()} lane still has no live worker, so it is "
                "reported unavailable."
                if lane is not None
                else "Its lane has a live worker, so it is still available.",
            )
        if lane is not None:
            return (
                f"the task runner's {lane.upper()} worker has exited and could "
                "not be replaced; restart the server"
            )
        return None

    def _dead_lane(self, worker_type: TaskType) -> str | None:
        """The runner lane *worker_type* queues on, when it has no live worker.

        Read from ``TaskRunner.lane_health`` on every call, never cached: the
        next ``submit()`` or planner tick replaces a dead worker, and the answer
        clears with it.

        Returns:
            ``"gpu"`` or ``"cpu"`` while the runner is meant to be running and
            that lane's ``alive`` is 0; None otherwise, including when there is
            no runner or it is stopped (nothing is meant to run then).
        """
        runner = self._task_runner
        if runner is None:
            return None
        health = runner.lane_health()
        lane = "gpu" if worker_type in GPU_LANE_WORKER_TYPES else "cpu"
        if health["running"] and health[lane]["alive"] == 0:
            return lane
        return None

    def _replace_dead_lane(self, worker_type: TaskType) -> str | None:
        """Replace the dead workers of *worker_type*'s lane, if it has none alive.

        ``TaskRunner.replace_dead_workers`` is the runner's repair path: it
        logs each replacement with the lane's restart count (``lane_health``
        keeps the last failure), and it never reopens a runner that a stop
        landing meanwhile has closed.

        Returns:
            The lane, when it still has no live worker afterwards; else None.
        """
        lane = self._dead_lane(worker_type)
        if lane is None:
            return None
        logger.error(
            "The task runner's %s lane has no live worker while %s needs it; "
            "replacing it now.",
            lane,
            worker_type.value,
        )
        self._task_runner.replace_dead_workers()
        return self._dead_lane(worker_type)

    def _is_worker_active(self, worker_type: TaskType) -> bool:
        if not self._work_planner or not self._work_planner.is_running():
            return False
        # Work queued on a lane with no live worker is not being done, however
        # many tasks the planner counts in flight for it.
        if self._dead_lane(worker_type) is not None:
            return False
        finder = self._planner_work_finders.get(worker_type)
        if finder is None:
            return False
        finder_name = finder.finder_name()
        return self._work_planner.inflight_count(finder_name) > 0

    def _task_phase_progress(self, task_type: TaskType) -> tuple[int, int, bool]:
        """Progress of a task that counts its own work, read from the task.

        For tasks whose denominator is their own work (a duplicate scan's
        phases, an auto-stack run's groups) rather than the library's pictures.

        Returns:
            ``(total, missing, active)``; ``(0, 0, False)`` when none is running.
        """
        active_tasks = (
            self._task_runner.get_active_tasks_of_type(task_type.value)
            if self._task_runner is not None
            else []
        )
        if not active_tasks:
            return 0, 0, False
        snapshots = [task.task_progress() for task in active_tasks]
        total = max(1, sum(snapshot[1] for snapshot in snapshots))
        processed = sum(snapshot[0] for snapshot in snapshots)
        # The task remains active for a tiny interval after its final DB slice
        # returns. Never advertise terminal progress until TaskRunner has
        # actually removed it from the active set.
        processed = min(processed, total - 1)
        return total, total - processed, True

    def _build_worker_progress_snapshot(self) -> dict:
        progress = {}
        # The total of every per-picture row: the pictures the pipeline works
        # on, live and with a file (``pipeline_picture_clauses``), the set each
        # of those finders draws from. Counting the whole table put the
        # Scrapheap in every total, so a library of 500 live pictures read
        # "1,500 / 1,500". Counted once per snapshot, not once per worker type.
        pipeline_total = int(
            self.db.run_immediate_read_task(self._count_total_pictures) or 0
        )
        for worker_type in TaskType.all():
            # When set by a branch below, this overrides the finder-based
            # activity check (used by user-triggered tasks that have no finder).
            worker_active_override = None
            total = pipeline_total
            if worker_type == TaskType.DESCRIPTION:
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_descriptions)
                    or 0
                )
                label = "descriptions_generated"
            elif worker_type == TaskType.TAGGER:
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_tags) or 0
                )
                label = "pictures_tagged"
            elif worker_type == TaskType.TAG_PREDICTION_BACKFILL:
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_tag_predictions)
                    or 0
                )
                label = "tag_prediction_backfill"
            elif worker_type == TaskType.QUALITY:
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_quality) or 0
                )
                label = "quality_scored"
            elif worker_type == TaskType.FACE_EXTRACTION:
                missing = int(
                    self.db.run_immediate_read_task(
                        self._count_missing_feature_extractions
                    )
                    or 0
                )
                label = "faces_extracted"
            elif worker_type == TaskType.TEXT_EMBEDDING:
                described = int(
                    self.db.run_immediate_read_task(self._count_total_described) or 0
                )
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_text_embeddings)
                    or 0
                )
                total = max(described, 0)
                label = "text_embeddings"
            elif worker_type == TaskType.IMAGE_EMBEDDING:
                missing = int(
                    self.db.run_immediate_read_task(
                        self._count_missing_image_embeddings
                    )
                    or 0
                )
                label = "image_embeddings"
            elif worker_type == TaskType.LIKENESS_PARAMETERS:
                missing = int(
                    self.db.run_immediate_read_task(
                        self._count_pending_likeness_parameters
                    )
                    or 0
                )
                label = "likeness_parameters"
            elif worker_type == TaskType.LIKENESS:
                total = int(
                    self.db.run_immediate_read_task(
                        self._count_total_likeness_candidates
                    )
                    or 0
                )
                missing = int(
                    self.db.run_immediate_read_task(self._count_pending_likeness_queue)
                    or 0
                )
                label = "likeness_pairs"
            elif worker_type == TaskType.WATCH_FOLDERS:
                label = "watch_folder_import"
                active_wf_tasks = (
                    self._task_runner.get_active_tasks_of_type("WatchFolderImportTask")
                    if self._task_runner is not None
                    else []
                )
                if active_wf_tasks:
                    total = sum(
                        int(getattr(t, "_total_candidates", 0)) for t in active_wf_tasks
                    )
                    processed = sum(
                        int(getattr(t, "_processed_count", 0)) for t in active_wf_tasks
                    )
                    missing = max(0, total - processed)
                else:
                    total = 0
                    missing = 0
            elif worker_type == TaskType.PICTURE_IMPORT:
                # User-triggered (no finder): surface live progress straight from
                # the running task(s), mirroring the WATCH_FOLDERS / DETECTION
                # handling so an async streaming import (#459) renders as a task
                # row in the frontend task manager.
                label = "picture_import"
                active_import_tasks = (
                    self._task_runner.get_active_tasks_of_type("PictureImportTask")
                    if self._task_runner is not None
                    else []
                )
                if active_import_tasks:
                    total = sum(
                        int(getattr(t, "_total_count", 0)) for t in active_import_tasks
                    )
                    processed = sum(
                        int(getattr(t, "_processed_count", 0))
                        for t in active_import_tasks
                    )
                    missing = max(0, total - processed)
                    worker_active_override = True
                else:
                    total = 0
                    missing = 0
                    worker_active_override = False
            elif worker_type == TaskType.MODEL_FOLDER_SCAN:
                # Live counters straight off the running task(s), whether the
                # finder or a registration queued them, the PICTURE_IMPORT /
                # DETECTION shape. Counted
                # in model files, not pictures - the shelf lives in the hub and
                # the generic library total would be meaningless here.
                label = "model_folder_scan"
                active_scan_tasks = (
                    self._task_runner.get_active_tasks_of_type("ModelFolderScanTask")
                    if self._task_runner is not None
                    else []
                )
                if active_scan_tasks:
                    total = sum(
                        int(getattr(t, "_total_count", 0)) for t in active_scan_tasks
                    )
                    processed = sum(
                        int(getattr(t, "_processed_count", 0))
                        for t in active_scan_tasks
                    )
                    missing = max(0, total - processed)
                    worker_active_override = True
                else:
                    total = 0
                    missing = 0
                    worker_active_override = False
            elif worker_type == TaskType.COMFYUI_EXTRACTION:
                missing = int(
                    self.db.run_immediate_read_task(
                        self._count_missing_comfyui_extraction
                    )
                    or 0
                )
                label = "comfyui_extraction"
            elif worker_type == TaskType.MISSING_FILE_PURGE:
                total = 0
                missing = 0
                label = "missing_file_purge"
            elif worker_type == TaskType.SNAPSHOT_IDENTITY_SCRUB:
                # Counted in snapshot archives, not pictures. This is a one-time
                # migration with a real finish line, so the generic picture total
                # would claim "N / N, nothing remaining" while the scrub still
                # has archives to rewrite: the same misleading shape the dedup
                # scan avoids below, but worse here because the user is waiting
                # for it to end.
                total = int(
                    self.db.run_immediate_read_task(self._count_total_snapshots) or 0
                )
                missing = int(
                    self.db.run_immediate_read_task(self._count_unscrubbed_snapshots)
                    or 0
                )
                label = "snapshot_identity_scrub"
            elif worker_type == TaskType.TEXT_SCORE:
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_text_score) or 0
                )
                label = "text_score"
            elif worker_type == TaskType.OCR:
                # Counted over the pictures that qualify for reading, not the
                # library: most pictures are never read at all.
                total, missing = self.db.run_immediate_read_task(
                    OcrTask.count_progress
                ) or (0, 0)
                label = "text_read"
            elif worker_type == TaskType.DETECTION:
                # User-triggered (no finder): surface live progress straight from
                # the running task(s), mirroring the WATCH_FOLDERS handling.
                label = "object_detection"
                active_detection_tasks = (
                    self._task_runner.get_active_tasks_of_type("DetectionTask")
                    if self._task_runner is not None
                    else []
                )
                if active_detection_tasks:
                    total = sum(
                        int(getattr(t, "_total_count", 0))
                        for t in active_detection_tasks
                    )
                    processed = sum(
                        int(getattr(t, "_processed_count", 0))
                        for t in active_detection_tasks
                    )
                    missing = max(0, total - processed)
                    worker_active_override = True
                else:
                    total = 0
                    missing = 0
                    worker_active_override = False
            elif worker_type == TaskType.DEDUP_SCAN:
                # A duplicate scan keeps working after it has enumerated every
                # picture: near buckets and embedding components still remain.
                # Use the task's cooperative phase counters instead of the
                # generic library total, which otherwise renders a misleading
                # "N / N" while the active task is still running.
                label = "duplicate_scan"
                total, missing, worker_active_override = self._task_phase_progress(
                    TaskType.DEDUP_SCAN
                )
            elif worker_type == TaskType.DEDUP_AUTO_STACK:
                # Counts the run's planned groups (or, for a revert, its History
                # segments), not the library: same reason as the scan above.
                label = "duplicate_auto_stack"
                total, missing, worker_active_override = self._task_phase_progress(
                    TaskType.DEDUP_AUTO_STACK
                )
            elif worker_type == TaskType.SCRAPHEAP_RUN:
                # Counts the run's own pictures, not the library: a 50,000-picture
                # Scrapheap empties in slices, and the generic total would read
                # "N / N" for the whole of it.
                label = "scrapheap_run"
                total, missing, worker_active_override = self._task_phase_progress(
                    TaskType.SCRAPHEAP_RUN
                )
            elif worker_type == TaskType.CHECKPOINT_HASH:
                # The one worker here whose denominator is the model shelf and
                # not the picture library. Left in the `planner_managed` default
                # below it inherited the library's picture count as its own
                # total and a hardcoded `missing = 0`, so reading tens of
                # gigabytes off disk for minutes rendered as "N / N, nothing
                # remaining, 0/s" on a row that still said running - which is
                # what made a healthy long task look like a stuck one.
                finder = self._planner_work_finders.get(TaskType.CHECKPOINT_HASH)
                total, missing = finder.progress() if finder is not None else (0, 0)
                label = "checkpoints_hashed"
            elif worker_type == TaskType.WORKFLOW_CARD_BACKFILL:
                # Counted over the hub's recipes, not this library's pictures.
                # Left in `planner_managed` below it would inherit the picture
                # count and a hardcoded `missing = 0` - the "N / N, nothing
                # remaining" row the two branches above exist to correct.
                finder = self._planner_work_finders.get(TaskType.WORKFLOW_CARD_BACKFILL)
                total, missing = finder.progress() if finder is not None else (0, 0)
                label = "workflows_carded"
            elif worker_type == TaskType.TRAINING_PREVIEW_FACES:
                # Counted over the training previews, not the library's pictures.
                finder = self._planner_work_finders.get(TaskType.TRAINING_PREVIEW_FACES)
                total, missing = finder.progress() if finder is not None else (0, 0)
                label = "training_previews_scored"
            elif worker_type == TaskType.ORIENTATION:
                # Without its own count this row fell into `planner_managed`
                # below - `missing = 0`, so it read "12,094 / 12,094, 0.00/s"
                # while 6,462 pictures were still NULL and the finder was
                # working through them 128 at a time; every batch made the
                # row pop back in, apparently finished.
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_orientations)
                    or 0
                )
                label = "orientations_read"
            elif worker_type == TaskType.SMART_SCORE:
                # Counted over the pictures the scorer can work on. Left in
                # `planner_managed` below it read "N / N" from the first second
                # of a full re-score to the last, so the row said the library
                # was done while every picture in it was being scored again.
                total, missing = self.db.run_immediate_read_task(
                    SmartScoreTask.count_progress
                ) or (0, 0)
                label = "smart_scored"
            elif worker_type == TaskType.THUMBNAIL_GENERATION:
                # Whole-frame bitmap regeneration (MissingThumbnailFinder). After
                # the v1.8.0 upgrade this counted the entire library; a fresh
                # import does the same. The Tasks tab and Appearance read it.
                missing = int(
                    self.db.run_immediate_read_task(self._count_missing_thumbnails) or 0
                )
                label = "thumbnails_generated"
            else:
                # A worker with no count of its own reports none. Borrowing the
                # library's picture total with nothing remaining rendered every
                # such row as "N / N", finished, while it ran: the trap the
                # Orientation, Checkpoint Hash, Workflow cards and Smart score
                # rows each fell into before getting a count of their own.
                total = 0
                missing = 0
                label = "planner_managed"
            worker_active = (
                worker_active_override
                if worker_active_override is not None
                else self._is_worker_active(worker_type)
            )
            progress[worker_type.value] = {
                "label": label,
                # What a person calls the task; ``label`` is the progress
                # state, which the panel maps to its own words.
                "task_label": task_label(worker_type.value),
                "current": max(total - missing, 0),
                "total": total,
                "remaining": max(missing, 0),
                "updated_at": time.time(),
                "status": "running" if worker_active else "idle",
                "running": worker_active,
                "active": worker_active,
            }
        return progress

    @staticmethod
    def _count_total_pictures(session: Session) -> int:
        """The total of the per-picture progress rows: the pictures the
        pipeline works on (live, with a file), not the whole table."""
        result = session.exec(
            select(func.count()).select_from(Picture).where(*pipeline_picture_clauses())
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    @staticmethod
    def _count_total_snapshots(session: Session) -> int:
        """Every registered snapshot, scrubbed or not."""
        result = session.exec(select(func.count()).select_from(Snapshot)).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    @staticmethod
    def _count_unscrubbed_snapshots(session: Session) -> int:
        """Legacy archives the identity scrub still owes work to.

        Mirrors ``MissingSnapshotIdentityScrubFinder._fetch_next``: keyed on
        ``identity_scrubbed_at IS NULL``, which means a pre-hub archive and
        nothing else, because snapshots created since are stamped at creation.
        """
        result = session.exec(
            select(func.count())
            .select_from(Snapshot)
            .where(Snapshot.identity_scrubbed_at.is_(None))
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_orientations(self, session: Session) -> int:
        """The pictures ``OrientationTask.find_pictures_missing_orientation``
        will hand out: its own clauses, less the suppressed pictures it skips."""
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(*OrientationTask.missing_orientation_clauses())
            .where(not_in_json(Picture.id, suppressed_ids))
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_thumbnails(self, session: Session) -> int:
        """Pictures awaiting whole-frame thumbnail (re)generation.

        Mirrors ``MissingThumbnailFinder._fetch_missing``: keyed on
        ``thumbnail_width IS NULL`` for live, file-backed pictures. Excludes
        undecodable pictures (issue #585) so the thumbnail worker's
        `remaining` can reach 0 instead of stalling on files that can never
        produce a thumbnail.
        """
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        stmt = (
            select(func.count())
            .select_from(Picture)
            .where(*MissingThumbnailFinder.missing_thumbnail_clauses())
            .where(not_in_json(Picture.id, suppressed_ids))
        )
        result = session.exec(stmt).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_descriptions(self, session: Session) -> int:
        # The same predicate MissingDescriptionFinder selects on: NULL or a
        # ``__description::`` sentinel, never a picture frozen by a locked
        # set, never one in the Scrapheap or without a file, and never a
        # suppressed one (undecodable or unreachable), which the finder will
        # not hand out. A reset writes the sentinel, not NULL, so counting NULL
        # alone read "N / N" through an entire regeneration (#1162).
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                or_(
                    Picture.description.is_(None),
                    Picture.description.like(
                        DESCRIPTION_SENTINEL_LIKE_PATTERN,
                        escape=DESCRIPTION_SENTINEL_ESCAPE_CHAR,
                    ),
                ),
                ~Picture.id.in_(locked_picture_id_subquery()),
                *describable_picture_clauses(),
                not_in_json(Picture.id, suppressed_ids),
            )
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_tags(self, session: Session) -> int:
        # The finder's own count (`TagTask.count_missing_tags`): this used to
        # count every sentinel, scrapheaped, file-less, locked and unreadable
        # pictures included, none of which the tagger will ever take, so the
        # bar never reached done. Imported here as `retag_picture_interactive`
        # imports it: `tag_task` pulls in the inference stack.
        from pixlstash.tasks.tag_task import TagTask

        return TagTask.count_missing_tags(
            session, self.db.unprocessable_images.active_suppressed_ids()
        )

    def _count_missing_tag_predictions(self, session: Session) -> int:
        """Count pictures with real tags but no tag-prediction rows.

        These were tagged before predictions were written inline (or by a
        different engine then); the MissingTagPredictionFinder back-fills them.
        Its own clauses, less the suppressed pictures it skips.
        """
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                *MissingTagPredictionFinder.missing_prediction_clauses(session),
                not_in_json(Picture.id, suppressed_ids),
            )
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_feature_extractions(self, session: Session) -> int:
        """The pictures ``MissingFaceExtractionFinder`` will hand out: its own
        clauses, less the suppressed pictures it skips."""
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                *MissingFaceExtractionFinder.missing_face_clauses(),
                not_in_json(Picture.id, suppressed_ids),
            )
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_quality(self, session: Session) -> int:
        # Suppressed pictures (undecodable or unreachable) are left out, as
        # MissingQualityFinder leaves them out, so "remaining" can reach 0.
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        return QualityTask.count_missing_quality(session, suppressed_ids)

    @staticmethod
    def _count_total_described(session: Session) -> int:
        """The text-embedding row's total: the described pictures the pipeline
        works on (live, with a file)."""
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(Picture.description.is_not(None), *pipeline_picture_clauses())
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_text_embeddings(self, session: Session) -> int:
        """Described pictures ``MissingTextEmbeddingFinder`` has still to embed:
        its own clauses, less the suppressed pictures it skips. A picture whose
        description or tags are still pending is counted: it is waiting, and
        the finder takes it once they land."""
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                Picture.description.is_not(None),
                *MissingTextEmbeddingFinder.missing_text_embedding_clauses(),
                not_in_json(Picture.id, suppressed_ids),
            )
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_comfyui_extraction(self, session: Session) -> int:
        """The pictures the registered ComfyUI finder will hand out: its own
        clauses, on the predicate it is running now (the workflow scan once a
        hub is attached, until it stands down), less the suppressed pictures it
        skips. Counted on ``comfyui_models`` alone it read done while a workflow
        scan still had the library to go."""
        finder = self._planner_work_finders.get(TaskType.COMFYUI_EXTRACTION)
        scanning_workflows = finder.scanning_workflows if finder is not None else False
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        result = session.exec(
            select(func.count())
            .select_from(Picture)
            .where(
                *MissingComfyUIExtractionFinder.unscanned_clauses(scanning_workflows)
            )
            .where(not_in_json(Picture.id, suppressed_ids))
        ).one()
        if isinstance(result, (tuple, list)):
            return result[0]
        return result or 0

    def _count_missing_image_embeddings(self, session: Session) -> int:
        # Exclude undecodable pictures (issue #585) so "remaining" can reach 0
        # instead of stalling on files that can never produce an embedding.
        suppressed_ids = self.db.unprocessable_images.active_suppressed_ids()
        return ImageEmbeddingTask.count_remaining(
            session, suppressed_ids=suppressed_ids
        )

    def _count_pending_likeness_parameters(self, session: Session) -> int:
        """The pictures owed likeness parameters that the finder will offer:
        an unprocessable one still waiting on its quality score never gets
        it, so it is left out and "remaining" can reach 0."""
        return LikenessParametersTask.count_pending(
            session, self.db.unprocessable_images.active_suppressed_ids()
        )

    def _count_missing_text_score(self, session: Session) -> int:
        from pixlstash.tasks.text_score_task import TextScoreTask

        return TextScoreTask.count_missing_text_score(
            session, self.db.unprocessable_images.active_suppressed_ids()
        )

    @staticmethod
    def _count_pending_likeness_queue(session: Session) -> int:
        return LikenessTask.count_queue(session)

    @staticmethod
    def _count_total_likeness_candidates(session: Session) -> int:
        return LikenessTask.count_total_candidates(session)

    def get_worker_progress(self) -> dict:
        progress = self._build_worker_progress_snapshot()
        self._maybe_aggressive_unload(progress)
        return progress

    def _maybe_aggressive_unload(self, progress: dict):
        """Queue an unload of every model when all the workers are idle.

        Reached from the worker-progress poll (a request thread) and from
        switching "keep models in memory" off (the event loop), so it only
        queues the unload on the GPU worker (:meth:`_unload_models_on_worker`)
        and never waits for it.
        """
        if self._keep_models_in_memory:
            return
        if not self._engine:
            return
        with self._aggressive_unload_lock:
            now = time.time()
            if now - self._last_aggressive_unload_at < self.AGGRESSIVE_UNLOAD_INTERVAL:
                return
            for snapshot in progress.values():
                status = snapshot.get("status")
                running = bool(snapshot.get("running"))
                if not running:
                    continue
                if status in ("idle", "stopped", "uninitialized"):
                    continue

                current = int(snapshot.get("current") or 0)
                total = int(snapshot.get("total") or 0)
                remaining = snapshot.get("remaining")
                if remaining is None:
                    remaining = max(0, total - current)
                else:
                    remaining = max(0, int(remaining))

                if remaining > 0 or (total > 0 and current < total):
                    return
            # The snapshot only sees planner in-flight counts. A task submitted
            # straight to the runner (an interactive retag or redescribe) is
            # invisible to it. The unload could not land under it - it waits
            # for the running task on the worker - but it would cost the next
            # batch a reload. GPU-queue tasks only: a dedup scan or an import
            # holds no model.
            if (
                self._task_runner is not None
                and self._task_runner.has_active_gpu_tasks()
            ):
                return
            self._last_aggressive_unload_at = now

        logger.warning("All workers idle; aggressively unloading models.")
        # The setting can be switched back on while the unload waits.
        self._unload_models_on_worker(when=lambda: not self._keep_models_in_memory)

    def _unload_models_on_worker(self, when=None) -> Optional[DeviceReleaseTask]:
        """Queue :meth:`unload_models` on the GPU worker; inline if there is none.

        It ends in a device-cache flush, and on Metal a flush on any other
        thread while the worker runs a model kills the process
        (``docs/apple-metal-thread-safety.md``, cause 3). Without a running
        GPU worker nothing else is on the device, so it runs here.

        Args:
            when: Checked just before unloading; ``False`` skips it.

        Returns:
            The queued task, or ``None`` when it ran inline or could not be
            queued.
        """
        runner = self._task_runner
        if runner is None or not runner.gpu_worker_alive():
            if when is None or when():
                self.unload_models()
            return None
        task = DeviceReleaseTask("UnloadModels", self.unload_models, when=when)
        try:
            runner.submit(task)
        except TaskRunnerStopped as exc:
            # The runner is closed. Vault.stop() releases the models once the
            # GPU worker has exited; if the worker outlives the stop it skips
            # the release rather than flush under it, and nothing unloads them.
            logger.warning(
                "Model unload not queued, the task runner is stopped. Vault.stop() "
                "releases the models if the GPU worker exits; if it outlived the "
                "stop they stay resident until the process exits: %s",
                exc,
            )
            return None
        except RuntimeError as exc:
            # submit() raises this, before it touches the task, when a dead
            # worker's replacement thread cannot start (Thread.start). Not a
            # stop, and not run here instead: the GPU worker was alive a moment
            # ago, and a flush beside it kills a Metal process. Raised on, it
            # failed the request that asked for the idle unload (the worker
            # progress poll, or switching "keep models in memory" off after the
            # setting was saved), and lost the make-room yield the GPU work it
            # had just paused (_yield_gpu).
            logger.error(
                "Model unload %s not queued: the task runner could not start a "
                "replacement worker (%s: %s). It is not run beside the live GPU "
                "worker; the models stay loaded until an unload is queued again.",
                task.id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None
        return task

    def unload_models(self) -> None:
        """Unload every model PixlStash holds, whatever ``keep_models_in_memory`` says.

        The engine's services, tagger plugins, face detection and the embedding
        models, then a malloc_trim. The one exception is the CPU query encoders
        on Metal, kept while a search is encoding or used them in the last
        ``IDLE_UNLOAD_S`` (``InferenceEngine.close``). Runs on the GPU worker, queued by
        :meth:`_unload_models_on_worker`: unloading under a running caption
        batch makes it return nothing for every picture in it, and on Metal
        the flush at the end kills the process beside a running model.
        """
        if not self._engine or cuda_checkpoint.suspended():
            # Suspended (#2557), they were unloaded before it and a CUDA call
            # would block until the training run ends.
            return
        try:
            self._engine.aggressive_unload()
        except Exception as exc:
            logger.warning("Aggressive unload failed for InferenceEngine: %s", exc)
        # The engine unloads the services it owns by name; a tagger plugin's
        # model is reachable only through the registry, and until this it was
        # not reachable at all - a plugin captioner stayed resident for the
        # life of the process and this setting could not free it (#967).
        try:
            unload_loaded_tagger_plugins()
        except Exception as exc:
            logger.warning("Aggressive unload failed for tagger plugins: %s", exc)
        try:
            FaceExtractionTask.release_detection_models()
        except Exception as exc:
            logger.warning(
                "Aggressive unload failed for feature extraction models: %s", exc
            )
        try:
            ImageEmbeddingTask.release_models()
        except Exception as exc:
            logger.warning(
                "Aggressive unload failed for image embedding models: %s", exc
            )
        if platform.system().lower().startswith("linux"):
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception as exc:
                logger.debug("malloc_trim call failed: %s", exc)
        empty_device_cache()

    def _yield_gpu(self, lease) -> list[str]:
        """Make room on the GPU for *lease*: the first rung of the reclaim ladder.

        Holds the planner so nothing new is queued, stops the GPU work in
        flight at a task boundary (``TaskRunner.preempt_gpu``), waits for the
        GPU worker to finish its batch, and unloads every model. The hold ends
        once *lease* is granted or released (``_on_resource_lease``), and the
        finders find the pictures that were not processed again.

        Returns:
            The worker types stopped, for the snapshot's ``preempted`` entries.
        """
        if self._task_runner is None or self._closed:
            return []
        if self._work_planner is not None:
            self._yield_holds.add(lease.id)
            self._work_planner.hold(lease.id, None)
            if lease.state.value != "waiting":
                # Granted or withdrawn while the hold was being taken: nothing
                # needs the room any more, so no GPU work is stopped. Released
                # here even if the listener already ended it, since its
                # release can land before this hold does.
                self._yield_holds.discard(lease.id)
                self._work_planner.release(lease.id)
                return []
        stopped = self._task_runner.preempt_gpu(f"preempted by {lease.label}")
        # The unload runs on the GPU worker once the batch in flight ends; the
        # planner hold keeps new work off, so that batch is the last before
        # the room is made. A lease granted meanwhile has its room, and
        # unloading then would only cost the next batch a reload.
        task = self._unload_models_on_worker(
            when=lambda: lease.state.value == "waiting" and not self._closed
        )
        # Briefly: this runs on the arbiter's one thread, which every other
        # lease waits on.
        if task is not None and not task.wait(YIELD_WAIT_S):
            logger.info(
                "The GPU worker is still finishing its batch; the models are "
                "unloaded once it does, for %s (%s)",
                lease.id,
                lease.label,
            )
        return stopped

    def _queue_cuda_suspend(self) -> None:
        """Have the GPU worker unload the models and suspend the CUDA context.

        Called by the arbiter's poll thread while a training run has the card
        to itself (#2557). On the GPU worker, nothing of PixlStash's is using
        the device when the context goes; one request at a time.
        """
        runner = self._task_runner
        pending = self._cuda_suspend_task
        if (
            self._closed
            or runner is None
            or not runner.gpu_worker_alive()
            or (pending is not None and not pending.wait(0))
        ):
            return
        task = DeviceReleaseTask(
            CUDA_SUSPEND_TASK,
            self._unload_and_suspend,
            # Checked on the worker: no unload under PixlStash's own GPU work.
            when=lambda: self._arbiter.wants_suspend(CUDA_SUSPEND_TASK),
        )
        try:
            runner.submit(task)
        except TaskRunnerStopped as exc:
            logger.warning(
                "Could not queue the CUDA suspend, the task runner is stopped: %s",
                exc,
            )
            return
        except RuntimeError as exc:
            # A dead worker's replacement could not start (Thread.start); see
            # TaskRunner.submit. The arbiter asks again on its next pass, which
            # may get the worker back. Raised on, it failed every pass with a
            # traceback, every POLL_S for the length of the training run; like
            # the runner's own repair log, the first in a row is the error.
            self._cuda_suspend_queue_failures += 1
            failures = self._cuda_suspend_queue_failures
            if failures == 1:
                logger.error(
                    "CUDA suspend not queued: the task runner could not start a "
                    "replacement worker (%s: %s). PixlStash keeps its CUDA "
                    "context; the arbiter asks again on each pass, and only the "
                    "first failure in a row is logged here.",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "CUDA suspend not queued again (%d in a row): %s: %s",
                    failures,
                    type(exc).__name__,
                    exc,
                )
            return
        self._cuda_suspend_queue_failures = 0
        self._cuda_suspend_task = task

    def _unload_and_suspend(self) -> None:
        self.unload_models()
        self._arbiter.suspend_pixlstash(CUDA_SUSPEND_TASK)

    def _release_yield_hold(self, lease_id: str) -> None:
        if lease_id in self._yield_holds:
            self._yield_holds.discard(lease_id)
            if self._work_planner is not None:
                self._work_planner.release(lease_id)

    def _on_resource_lease(self, payload: dict) -> None:
        """Forward an arbiter event to the clients; end a make-room hold."""
        if payload.get("change") in ("granted", "released"):
            self._release_yield_hold(payload.get("lease_id"))
        self.notify(EventType.RESOURCE_LEASE, payload)

    def import_default_data(self, add_tagger_test_images: bool = False):
        """
        Import default data into the vault.
        Extend this method to add default pictures or metadata as needed.
        """
        # Add Logo.png to every vault

        logo_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logo.png")
        logo_dest_folder = self.image_root
        logger.debug(f"logo_dest_folder in _import_default_data: {logo_dest_folder}")

        characters = [
            "Esmeralda Vault",
            "Barbara Vault",
            "Barry Vault",
            "Cassandra Vault",
        ]

        def add_character(session: Session, character: Character):
            session.add(character)
            session.commit()
            session.refresh(character)
            char_id = character.id
            char_name = character.name
            # Create reference picture set for this character, using character name
            reference_set = PictureSet(
                name="reference_pictures", description=str(char_name)
            )
            session.add(reference_set)
            session.commit()
            session.refresh(reference_set)
            return char_id, char_name

        for character_name in characters:
            self.db.run_task(
                lambda session: add_character(
                    session,
                    Character(
                        name=character_name, description="Built-in vault character"
                    ),
                ),
                priority=DBPriority.IMMEDIATE,
            )

        def add_picture(
            session: Session, picture: Picture, published: PublishedPictures
        ):
            session.add(picture)
            session.commit()
            published.mark_committed()
            session.refresh(picture)
            return picture

        # A seed picture whose insert does not commit has its file removed
        # again: left behind, the next root scan would index it as a picture
        # without its description.
        with PublishedPictures(logo_dest_folder, "Default data") as published:
            picture = published.add(
                ImageUtils.create_picture_from_file(
                    image_root_path=logo_dest_folder,
                    source_file_path=logo_src,
                )
            )
            picture.description = "PixlStash Logo"
            picture.imported_at = datetime.datetime.now(datetime.timezone.utc)

            assert picture.file_path

            picture = self.db.run_task(
                add_picture,
                picture,
                published,
                priority=DBPriority.IMMEDIATE,
            )

        if add_tagger_test_images:
            # Add all pictures/TaggerTest*.png
            for file in os.listdir(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "pictures")
            ):
                if file.startswith("TaggerTest") and file.endswith(".png"):
                    src_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "pictures",
                        file,
                    )
                    with PublishedPictures(
                        logo_dest_folder, "Default data"
                    ) as published:
                        pic = published.add(
                            ImageUtils.create_picture_from_file(
                                image_root_path=logo_dest_folder,
                                source_file_path=src_path,
                            )
                        )
                        pic.description = os.path.basename(src_path)
                        pic.imported_at = datetime.datetime.now(datetime.timezone.utc)
                        assert pic.file_path
                        self.db.run_task(
                            add_picture,
                            pic,
                            published,
                            priority=DBPriority.IMMEDIATE,
                        )
                    logger.debug(f"Imported default picture: {pic.file_path}")
        logger.info("Imported default data into the vault.")
