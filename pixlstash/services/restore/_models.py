"""Shared data models, exceptions, and constants for the restore package.

Dependency-free leaf definitions imported by every restore submodule and
re-exported from :mod:`pixlstash.services.restore`.  Kept in one module so the
mixins can import them without any circular dependency.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)

#: What a person calls each kind of task, for a refusal that names the work in
#: the way. Keyed by ``BaseTask.type``: the class names are internal and never
#: reach a person (``tests/test_restore_drain.py`` checks that every task type
#: in the code has a label here). The three ``DeviceReleaseTask`` labels
#: (``UnloadModels``, ``SuspendCuda``, ``ReleaseFaceModels``) are listed too: a
#: drain can time out on one.
TASK_LABELS: dict[str, str] = {
    "AnomalyRegionTask": "Anomaly regions",
    "CheckpointHashTask": "Checkpoint hashing",
    "ComfyUIExtractionTask": "ComfyUI backfill",
    "DedupAutoStackTask": "Auto-stack",
    "DedupScanTask": "Duplicate scan",
    "DescriptionTask": "Descriptions",
    "DetectionTask": "Object detection",
    "DownloadsIntakeTask": "Downloads intake",
    "EnsureGfsSnapshotTask": "Scheduled snapshot",
    "FaceDetectionTask": "Face detection",
    "FaceExtractionTask": "Face extraction",
    "FaceModelRefreshTask": "Detect faces again",
    "FamilyReidentifyTask": "Family re-identification",
    "GhostCascadeTask": "Workflow ghost cleanup",
    "ImageEmbeddingTask": "Image embeddings",
    "LayoutMoveTask": "Folder layout move",
    "LikenessParametersTask": "Likeness parameters",
    "LikenessTask": "Likeness",
    "MissingFilePurgeTask": "File cleanup",
    "ModelFolderScanTask": "Model folder scan",
    "OcrTask": "Text in pictures",
    "OrientationTask": "Orientation",
    "PendingScoreInvalidationTask": "Score refresh",
    "PictureImportTask": "Import",
    "PixelShaTask": "Picture fingerprints",
    "PromptMatchTask": "Prompt match",
    "QualityTask": "Quality scoring",
    "ReferenceFolderScanTask": "Reference folder scan",
    "ReleaseFaceModels": "Face model release",
    "SavedRecipeConvertTask": "Saved recipe conversion",
    "ScrapheapRetentionPurgeTask": "Scrapheap cleanup",
    "ScrapheapRunTask": "Scrapheap",
    "SmartScoreTask": "Smart score",
    "SnapshotIdentityScrubTask": "Snapshot cleanup",
    "SourceFaceLikenessTask": "Source face likeness",
    "StackCohesionTask": "Stack check",
    "SuspendCuda": "GPU release",
    "TagExclusionCleanupTask": "Penalised tag cleanup",
    "TagHealthAutoRebuildTask": "Tag health check",
    "TagHealthRebuildTask": "Tag health rebuild",
    "TagPredictionBackfillTask": "Tag predictions",
    "TagTask": "Tagging",
    "TextEmbeddingTask": "Text embeddings",
    "TextScoreTask": "Text score",
    "ThumbnailGenerationTask": "Thumbnails",
    "TrainingPreviewFacesTask": "Training preview faces",
    "UnloadModels": "Model unload",
    "WatchFolderImportTask": "Folder import",
    "WorkflowCardBackfillTask": "Workflow cards",
}


def task_label(task_type: str) -> str:
    """What a person calls a task of *task_type*.

    Every :class:`TaskType` has a label, checked when ``pixlstash.vault`` is
    imported (this module cannot import the task types: see there), so a type
    added without one stops the server at start-up. A ``task_type``
    the code passes as a literal string is checked by
    ``tests/test_restore_drain.py``. A type with no label is a programming
    error and raises ``KeyError``: there is no name made up from the class
    name.
    """
    return TASK_LABELS[task_type]


class RestoreInProgressError(RuntimeError):
    """Raised when a restore is requested while another one is still running.

    Two concurrent restores would both stop the planner, take safety snapshots
    and enqueue swap + cleanup tasks on the writer queue - the cleanups would
    then run against each other's swapped DB file with stale ``missing_ids``
    sets, corrupting the live database. Restore is intentionally serialised
    instead of being made concurrent; callers should retry after the running
    job (visible at ``GET /snapshots/status``) completes.
    """


class SafetySnapshotFailedError(RuntimeError):
    """Raised when the pre-restore safety snapshot fails and the caller did
    not opt into ``allow_without_safety``.

    The safety snapshot is the only rollback path if the swap or the
    post-restore cleanup later corrupts the live DB; silently continuing
    without it can turn a recoverable problem into an unrecoverable one.
    The message is fixed text a client may be shown; the cause is logged.
    """


class SnapshotNotFoundError(ValueError):
    """The snapshot row, or its file on disk, does not exist.

    The message names only the snapshot id, so a route may show it; the path
    that was looked up is logged, never put in the message.
    """


class UnsupportedResourceTypeError(ValueError):
    """A per-resource restore or preview named a type it does not support.

    The message lists the supported types and never echoes the request.
    """


class RestoreRefusedError(RuntimeError):
    """A restore declined to start, and nothing in the library was changed.

    Raised before the database swap or any row is rewritten, with a message
    the restore composed itself from counts and fixed text, so a route can
    show it as it is (409). A full restore raises it before its safety
    snapshot and before ``RESTORE_STARTED``, so a refusal costs no snapshot
    and reloads no tab. The caller can retry once the condition has passed.
    """


class BackgroundWorkStillRunningError(RestoreRefusedError):
    """Background work was still running when the restore's drain timed out.

    Attributes:
        still_running: What a person calls each task still running
            (:func:`task_label`), sorted, or the vault writers
            ``Vault.quiesce_background_writers`` named.
        timeout_s: How long the restore waited for them, in seconds.
    """

    def __init__(self, still_running: list[str], timeout_s: float):
        self.still_running = still_running
        self.timeout_s = timeout_s
        super().__init__(
            "The restore did not start because background work was still "
            f"running after {timeout_s:.0f} seconds: "
            f"{', '.join(still_running)}. Nothing was changed. Try again once "
            "it has finished."
        )


class WorkInProgressError(RestoreRefusedError):
    """Work somebody started was queued or running when a restore was asked for.

    A restore drains background work, which would stop this work too: a
    request waiting on a task, a picture-menu action, an import, a Scrapheap
    run. So the restore refuses instead, before anything is paused or stopped.

    Attributes:
        task_types: The ``task_type`` of every such task, sorted, once each.
            For the log and tests; the message names the work by
            :func:`task_label`.
    """

    def __init__(self, task_types: list[str]):
        self.task_types = sorted(set(task_types))
        labels = sorted({task_label(task_type) for task_type in self.task_types})
        super().__init__(
            "The restore did not start because this work is still queued or "
            "running, and the restore would have stopped it: "
            f"{', '.join(labels)}. Nothing was changed. Try again once "
            "it has finished."
        )


class LibraryStillInUseError(RestoreRefusedError):
    """Work holding the active library did not end within the restore's wait.

    A full restore first closes library admission (``begin_switch``) and waits
    for every read lease but its own request's: an import thread, a folder
    mapping commit, a ComfyUI output import, any other request. Raised when
    one is still held at the deadline. Admission is back to READY and nothing
    was changed.

    Attributes:
        holder: What is holding the library, as
            ``LibrarySwitchService.describe_library_holders`` names it, or
            ``None`` when nothing recognisable is.
        timeout_s: How long the restore waited, in seconds.
    """

    def __init__(self, holder: Optional[str], timeout_s: float):
        self.holder = holder
        self.timeout_s = timeout_s
        what = holder if holder else "other work is still using the library"
        super().__init__(
            f"The restore did not start because, after {timeout_s:.0f} "
            f"seconds, {what}. Restoring now would leave that work writing into the "
            "restored library. Nothing was changed. Try again once it has "
            "finished."
        )


@dataclass(frozen=True)
class LibraryFence:
    """What a full restore needs to close library admission around the swap.

    Built by the HTTP route from the server; the CLI restore has no server
    and passes none.

    Attributes:
        coordinator: The server's ``LibraryGenerationCoordinator``.
        own_lease: The restore request's own read lease, which the switch must
            not wait for (``begin_switch(exclude=...)``), or ``None``.
        describe_holders: Names what holds the library when the wait times
            out (``LibrarySwitchService.describe_library_holders``), or
            ``None``.
    """

    coordinator: Any
    own_lease: Any = None
    describe_holders: Optional[Callable[[], Optional[str]]] = None


@dataclass
class LibrarySwitch:
    """A full restore's library switch, as the steps report it back.

    Attributes:
        swap_attempted: Set just before the database swap is queued. From
            then on the file may be the snapshot's, so the switch ends with a
            new generation (``publish_ready``): work that began against the
            old file cannot pass its generation check against the new one.
    """

    swap_attempted: bool = False


#: What a client is told when a full restore fails for a reason the restore
#: did not compose itself (an unexpected exception). The exception is logged.
#: Sign-in can stay closed after a failure past the swap, hence the restart.
RESTORE_FAILED_MESSAGE = (
    "The restore failed; the server log has the details. If signing in stays "
    "unavailable, restart PixlStash."
)


#: Added to a full restore's report when the runs the snapshot recorded as
#: active could not be closed after the swap. The exception is logged; the
#: start-up recovery closes them.
RECONCILE_RUNS_FAILED_MESSAGE = (
    "The restore finished, but runs the snapshot recorded as in progress could "
    "not be closed; restart PixlStash to close them."
)


#: What a client is told when a per-resource or batch restore fails for a
#: reason the restore did not compose itself. The exception is logged.
RESOURCE_RESTORE_FAILED_MESSAGE = "The restore failed; the server log has the details."


#: Warning added to a preview whose computation raised. The exception is logged.
PREVIEW_INCOMPLETE_WARNING = (
    "The preview could not be computed in full; the server log has the details."
)


class MissingDependenciesError(RuntimeError):
    """Raised by ``restore_resource`` / ``restore_batch`` when the snapshot
    rows reference parent resources (Character / PictureSet / Project) that
    have been deleted from the live DB since the snapshot was taken.

    Restoring the picture without the parent would trigger ``IntegrityError``
    on commit (FK violations) and roll back the whole batch.  The caller
    surfaces ``self.missing`` to the user, who can either re-issue the
    request with ``confirm_restore_dependencies=True`` (restores the parents
    from the snapshot first) or decline and leave the live DB untouched.

    Attributes:
        missing: dict mapping resource-type plural (``"characters"`` /
            ``"picture_sets"`` / ``"projects"``) → sorted list of IDs that
            need to be restored from the snapshot before the requested
            resource can be safely upserted.
    """

    def __init__(self, missing: dict[str, list[int]]):
        self.missing = missing
        super().__init__(
            "Restore would reference resources that no longer exist in the "
            f"live DB: {missing}. Retry with confirm_restore_dependencies=True "
            "to restore them from the snapshot first, or cancel to leave the "
            "live DB untouched."
        )


# Resource types currently supported by ``restore_resource`` / ``preview_resource``.
#
# ``"project"`` is intentionally excluded for this release:
# Project's graph spans ``ProjectAttachment`` (CASCADE FK), ``Character.project_id``,
# ``PictureSet.project_id``, and ``PictureProjectMember`` rows. The current
# per-resource path only touches Project + pictures-via-PPM, and the PPM
# bulk-delete is keyed by ``picture_id`` rather than by ``project_id`` - which
# would over-delete PPMs to *other* projects for any picture also in the
# restored project. Use the full restore for project-level recovery until the
# proper graph-replace is implemented + tested.
_SUPPORTED_RESOURCE_TYPES: tuple[str, ...] = ("picture", "picture_set", "character")

#: Message of :class:`UnsupportedResourceTypeError`; it lists the supported
#: types and does not echo the requested one.
UNSUPPORTED_RESOURCE_TYPE_MESSAGE = (
    f"Unsupported resource_type. Supported: {', '.join(_SUPPORTED_RESOURCE_TYPES)}. "
    "Use the full restore for project-level recovery."
)


# Refuse the post-restore cleanup if this fraction or more of the snapshot's
# pictures appear to be missing on disk. A partial network-mount failure
# would otherwise be silently treated as "the user deleted these files" and
# wipe their metadata. The check only kicks in once the snapshot contains
# more than ``_MIN_PICTURES_FOR_MISSING_RATIO_CHECK`` rows - at small scale
# "100% missing" is a legitimate one-picture deletion, not a mount blip.
_MAX_MISSING_RATIO_FOR_CLEANUP: float = 0.5
_MIN_PICTURES_FOR_MISSING_RATIO_CHECK: int = 10


@dataclass
class RestoreReport:
    """Summary of a completed restore operation.

    Attributes:
        snapshot_id: ID of the snapshot that was restored.
        resource_type: ``'full'``, ``'picture'``, ``'picture_set'``,
            ``'project'``, or ``'character'``.
        resource_id: Primary key of the specific resource (None for full
            restore).
        missing_files_count: Number of Picture rows skipped because their
            files were absent on disk.
        permanently_deleted_count: Number of Picture rows skipped because
            their file/content was recorded in ``deleted_file_log`` (a
            permanent deletion that restore must never resurrect).
        upserted_count: Number of rows upserted (per-resource restores only).
        skipped_excluded: Number of Tag rows left out because the owner removed
            the tag from tagging (per-resource restores only).
        errors: Non-fatal error messages accumulated during the restore.
    """

    snapshot_id: int
    resource_type: str
    resource_id: Optional[int] = None
    missing_files_count: int = 0
    permanently_deleted_count: int = 0
    upserted_count: int = 0
    skipped_excluded: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ResourcePreview:
    """Preview information for a single resource that would be affected by a restore.

    Attributes:
        type: Resource type string (``'picture'``, ``'picture_set'``, etc.).
        id: Primary key of the resource.
        exists_in_live: True if the resource exists in the live database.
        exists_in_snapshot: True if the resource exists in the snapshot.
        file_on_disk: True if the picture file exists on disk (always True
            for non-picture resources).
        changed_fields: List of column names that differ between live and
            snapshot (for picture resources).
        dependent_counts: Counts of dependent objects (e.g.
            ``{"faces": 2, "tags": 10}``).
    """

    type: str
    id: int
    exists_in_live: bool = True
    exists_in_snapshot: bool = True
    file_on_disk: bool = True
    changed_fields: list[str] = field(default_factory=list)
    dependent_counts: dict = field(default_factory=dict)


@dataclass
class RestorePreview:
    """Dry-run preview of a restore operation.

    Attributes:
        snapshot_id: ID of the snapshot to be restored.
        snapshot_kind: Kind of the snapshot.
        snapshot_label: Optional user label.
        snapshot_created_at: ISO timestamp string.
        resources: Per-resource preview entries (capped at 200).
        summary: High-level counts of what would change.
        warnings: Human-readable warning strings (e.g. missing files).
    """

    snapshot_id: int
    snapshot_kind: str
    snapshot_label: Optional[str]
    snapshot_created_at: str
    resources: list[ResourcePreview] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
