"""Durable single-worker orchestrator that turns queued jobs into LoRAs.

The :class:`TrainingService` is the layer between PixlStash's data model and the
trainer adapter (:class:`~pixlstash.training.local_ai_toolkit.LocalAiToolkit`).
It owns the job lifecycle: pick the oldest queued training run, snapshot
its dataset, wait for the VRAM arbiter's lease, drive the adapter (``prepare`` → ``submit`` →
poll → finalize), and hand the finished run to the model shelf when it completes.
Exactly one job runs at a time because a single GPU is the binding constraint.

It is built with **dependency injection** so the whole thing is testable without
a GPU (or even a full Vault): every external collaborator is a callable handed
to the constructor: the VRAM lease request, the run importer, the event emitter,
the owner-config reader, the dataset builder, and the trainer-backend factory. The default backend
factory builds a real :class:`LocalAiToolkit`; tests pass one pointed at a CPU
stub.

Shutdown policy (important): :meth:`stop` does **not** terminate a running
external trainer. The trainer runs in its own process group and is expected to
survive a PixlStash restart; :meth:`stop` only halts the poll thread and joins
it with a timeout. On the next startup, :meth:`reconcile_on_startup` re-attaches
to any still-alive run (asking for its lease again and resuming polling) so a
multi-hour run is never killed by a server bounce. The only thing that stops a
run is an explicit :meth:`cancel` (a user action).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from pixlstash.event_types import EventType
from pixlstash.hub import dataset_snapshots, training_runs
from pixlstash.hub.db import HubDatabase, now_iso
from pixlstash.pixl_logging import get_logger
from pixlstash.resources.aitoolkit_tenant import AiToolkitProbe
from pixlstash.resources.arbiter import Lease
from pixlstash.resources.ledger import ledger
from pixlstash.resources.ledger import ram as read_ram
from pixlstash.services.downloads_intake import safetensors_complete
from pixlstash.services.library_generation_coordinator import (
    LibraryNotReadyError,
    SwitchState,
)
from pixlstash.services.model_folder_scanner import SOURCE_FOLDER_KIND
from pixlstash.services.model_mover import file_digest
from pixlstash.training import ai_toolkit_install, live_view, measurements, presets
from pixlstash.training.backend import TrainerBackend, TrainingProgress, TrainingSpec
from pixlstash.training.config_render import deep_merge
from pixlstash.training.local_ai_toolkit import LocalAiToolkit
from pixlstash.utils.atomic_write import write_json_atomic
from pixlstash.utils.image_processing.video_utils import VideoUtils

logger = get_logger(__name__)

# Statuses that mean the job is finished and must not be re-polled / re-run.
_TERMINAL_JOB_STATUSES = frozenset({"completed", "stopped", "error"})
# Adapter progress statuses that mean the external process has reached an end
# state; the poll loop exits once the process is dead AND status is terminal.
_TERMINAL_PROGRESS_STATUSES = frozenset({"completed", "stopped", "error"})
# Cap on the trainer log tail we persist onto the job row.
_LOG_TAIL_BYTES = 8192
# A write of a terminal status that hits a busy or locked hub is tried this
# many times in all. Each try already waits out the hub's busy timeout, so a
# lost one is a sustained outage, which more tries would not cure.
_TERMINAL_WRITE_ATTEMPTS = 3
# The capped exponential backoff between those tries, drawn with full jitter.
_TERMINAL_WRITE_BACKOFF_S = 0.5
_TERMINAL_WRITE_BACKOFF_CAP_S = 4.0
# How often a finished run's import asks again while a restore or a library
# switch holds admission. It never gives up: the LoRA is the owner's work.
_LIBRARY_NOT_READY_RETRY_S = 5.0
# A write that moves a run on names the states it may move it from
# (``training_runs.update_run``'s compare-and-set), so it never lands on a
# newer state another thread wrote meanwhile.
_ONLY_QUEUED = {"status": ("queued",)}
_ONLY_UNFINISHED = {"status": training_runs.UNFINISHED_STATUSES}
_ONLY_UNENDED = {"status": ("queued", *training_runs.UNFINISHED_STATUSES)}
# Trained, and not yet on the shelf: its previews are drawn or it is imported.
_ONLY_COMPLETED_UNIMPORTED = {"status": ("completed",), "completed_at": (None,)}

#: How long a run adopted at start-up keeps its lease exclusive and unshrunk:
#: PixlStash has not watched it reach its peak since the restart.
ADOPT_QUIET_S = 120.0
#: Updates a probe trains: enough for a warm speed after the first.
PROBE_STEPS = 6
#: The most probes one search runs; a measured profile costs none.
MAX_PROBES = 3
#: A probe is stopped when the machine's available RAM falls below this.
RAM_GUARD_MB = 1024
#: How long a dead trainer's processes get to give their GPU memory back.
RELEASE_GRACE_S = 10.0
#: Why a profile is passed over when the run cannot be resolved on it again.
#: The cause names a key or a pinned value: the log's, not the run's error.
RESOLVE_AGAIN_FAILED = "the run could not be resolved again; the server log says why"
# How often a probe or an after-training preview pass is polled.
_SHORT_POLL_S = 1.0
# A checkpoint's step: ``<stem>_<step:09d>.safetensors``.
_CHECKPOINT_STEP_RE = re.compile(r"_(\d{4,})\.safetensors$")


class TrainingStoreError(RuntimeError):
    """A change to a training run could not be written to the hub."""


@dataclass
class _Watch:
    """What one poll loop has seen of a trainer.

    Attributes:
        peaks: ``{phase: {vram_mb, ram_mb}}``, the highest figures per phase.
        seconds: Seconds spent per phase, between polls.
        pids: Every process of the trainer's tree seen, so the ones left
            holding memory after it exits can be found.
        from_start: Whether the loop saw the trainer start (a run re-attached
            after a restart did not, so its setup time is unknown).
        shrunk: Whether the run's lease has been lowered to its peak.
        offsets: Where each of the trainer's allocator peak files was read to.
    """

    peaks: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    pids: set = field(default_factory=set)
    from_start: bool = True
    shrunk: bool = False
    offsets: dict = field(default_factory=dict)
    _last: tuple | None = None

    def see(self, phase: str | None, usage: dict, now: float) -> None:
        if self._last is not None and self._last[0] is not None:
            self.seconds[self._last[0]] = self.seconds.get(self._last[0], 0.0) + (
                now - self._last[1]
            )
        self._last = (phase, now)
        self.pids |= usage["pids"]
        measurements.note_peak(
            self.peaks, phase, usage.get("peak_vram_mb"), usage["ram_mb"]
        )

    @property
    def phase(self) -> str | None:
        """The phase the last poll saw."""
        return self._last[0] if self._last is not None else None

    def peak_vram_mb(self) -> int | None:
        figures = [e["vram_mb"] for e in self.peaks.values() if e["vram_mb"]]
        return max(figures) if figures else None


def handle_for_run(training_root: str, info: dict, backend=None):
    """Reconstruct a run's :class:`TrainingHandle` from its persisted fields.

    The work dir is ``<training_root>/<uuid>``. The persisted
    ``training_folder`` is the shelf source folder for a run started since the
    shelf took runs, or the legacy ``<work_dir>/output`` for an older one; only
    the former is passed on, so an older run keeps the save root it was started
    with. The adapter's ``handle_for`` is the source of truth for the paths.

    Args:
        training_root: ``<image_root>/training``.
        info: Persisted job fields (uuid, name, pid, training_folder).
        backend: A backend whose ``handle_for`` to use; None uses
            :class:`LocalAiToolkit`'s.

    Returns:
        A reconstructed :class:`TrainingHandle`.
    """
    work_dir = os.path.join(training_root, info["uuid"])
    stored = info.get("training_folder")
    shared = None
    if stored and os.path.normcase(os.path.abspath(stored)) != os.path.normcase(
        os.path.join(os.path.abspath(work_dir), "output")
    ):
        shared = stored
    handle_for = getattr(backend, "handle_for", None)
    if not callable(handle_for):
        handle_for = LocalAiToolkit.handle_for
    return handle_for(
        work_dir,
        info["uuid"],
        info["name"],
        info.get("pid"),
        training_folder=shared,
    )


class TrainingService:
    """Drive queued training runs to finished LoRAs on the shelf.

    One background thread runs at a time and processes queued jobs FIFO (oldest
    ``created_at`` first). See the module docstring for the dependency-injection
    rationale and the shutdown/reconcile policy.

    Attributes:
        poll_interval_s: Seconds between progress polls of a running job.
    """

    def __init__(
        self,
        hub: HubDatabase | None,
        library_uuid: str | None,
        image_root: str,
        *,
        request_gpu: Callable[[str, str, int, int], Lease],
        import_run: Callable[[str, dict], dict],
        notify: Callable[[EventType, dict], None],
        config_provider: Callable[[], dict],
        dataset_builder: Callable[..., dict],
        backend_factory: Callable[[str, str], TrainerBackend] | None = None,
        poll_interval_s: float = 5.0,
        disable_background_workers: bool = False,
        register_source_folder: Callable[[str], None] | None = None,
        adopt_gpu: Callable[[str, str, int, int], Lease] | None = None,
        tenant: AiToolkitProbe | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            hub: The open hub, which holds the ``training_run`` rows. None when
                no hub is open: the service then never starts and refuses new
                runs.
            library_uuid: The library this service trains for. Every run it
                reads or writes is scoped to it, so another library's queue is
                never claimed here.
            image_root: Vault root. Per-job work dirs live in
                ``<image_root>/training/<job_uuid>/``.
            request_gpu: Asks the VRAM arbiter for a run's memory, called as
                ``request_gpu(label, job_uuid, vram_mb, ram_mb)`` with the
                needs of the run's resolved memory profile; returns the
                arbiter's :class:`~pixlstash.resources.arbiter.Lease`, granted
                or waiting.
            import_run: Brings a completed run onto the model shelf, called as
                ``import_run(run_dir, job_info)`` where ``run_dir`` is the
                ai-toolkit run folder holding the checkpoints. Returns a dict
                describing what landed. Raising marks the job ``error``.
            notify: Emits an event (``event_type, data``) to the UI/listeners.
            config_provider: Returns the owner's ai-toolkit config as
                ``{"repo_path", "python"}``.
            dataset_builder: Builds the dataset snapshot; called as
                ``dataset_builder(dest_dir=..., set_id=..., character_id=...,
                trigger_word=...)`` and returns a manifest with ``image_count``
                and ``picture_ids``.
            backend_factory: Builds a :class:`TrainerBackend` from
                ``(repo_path, python)``. Defaults to constructing a real
                :class:`LocalAiToolkit`.
            poll_interval_s: Seconds between progress polls of a running job.
            disable_background_workers: When True, :meth:`start` is a no-op
                (mirrors the Vault flag for read-only deployments).
            register_source_folder: Adds a folder to the model shelf as its
                ``source`` (ai-toolkit output) folder, called as
                ``register_source_folder(path)``. Goes through the same
                registration the folder routes use. Called at most once, when a
                job starts and the shelf has no source folder yet. ``None``
                means a job that needs one fails with a clear error.
            adopt_gpu: Grants a live run found at start-up its memory at once,
                called like ``request_gpu``
                (:meth:`~pixlstash.resources.arbiter.ResourceArbiter.adopt`).
                None asks with ``request_gpu`` instead.
            tenant: The arbiter's AI-toolkit probe, told which processes are
                the trainer and which held memory after it exited. None tells
                nobody.
        """
        self._hub = hub
        self._library_uuid = library_uuid
        self._image_root = image_root
        self._request_gpu = request_gpu
        self._import_run = import_run
        self._notify = notify
        self._config_provider = config_provider
        self._dataset_builder = dataset_builder
        self._backend_factory = backend_factory or (
            lambda repo_path, python: LocalAiToolkit(repo_path, python)
        )
        self.poll_interval_s = poll_interval_s
        self._disable_background_workers = disable_background_workers
        self._register_source_folder = register_source_folder
        self._adopt_gpu = adopt_gpu
        self._tenant = tenant
        # Leases granted at start-up to runs that outlived the restart, by job
        # uuid, until reconcile_on_startup resumes watching them.
        self._adopted: dict[str, Lease] = {}
        # Runs drawing their after-training previews, and those of them whose
        # remaining previews were cancelled. Both under ``_active_lock``.
        self._drawing_previews: set[str] = set()
        self._skip_previews: set[str] = set()
        # One Train longer at a time: it renames files in the run's folder.
        self._train_longer_lock = threading.Lock()

        # Per-job work dirs (job.db, the rendered config, the pending sidecar).
        # The run itself lives in the shelf's source folder, the dataset under
        # the checkout's datasets/ folder.
        self._training_root = os.path.join(image_root, "training")

        # Per-job spec extras (template_path / config_overrides) that have no
        # dedicated columns. Keyed by job uuid; populated by enqueue, consumed
        # by the worker. Persisted to a per-job sidecar so a restart can recover
        # them when re-running a job that was still queued.
        self._pending_specs: dict[str, dict] = {}
        self._pending_specs_lock = threading.Lock()

        # Single worker thread + its lifecycle primitives.
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        # The adapter currently driving a job, so cancel() can reach it. Set
        # while a job is running, cleared in the worker's finally.
        self._active_backend: TrainerBackend | None = None
        self._active_job_uuid: str | None = None
        # The lease a claimed job is still waiting on, so cancel() can drop it.
        self._waiting: tuple[str, Lease] | None = None
        self._active_lock = threading.Lock()
        # Each run's phase log as last written, so a poll that stays in one
        # phase writes nothing; loaded from the row the first time it is needed.
        self._phase_logs: dict[str, list[dict]] = {}
        self._phase_lock = threading.Lock()
        # Each polled run's progress columns as last written, so a poll that
        # changed nothing writes nothing and one that did writes only what
        # changed. Read and set under ``_active_lock``; dropped when the run's
        # polling ends.
        self._persisted: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the single worker thread. No-op if workers are disabled."""
        if self._disable_background_workers:
            logger.info("TrainingService: background workers disabled; not starting.")
            return
        if self._hub is None or not self._library_uuid:
            logger.warning(
                "TrainingService: no hub or library open; training is unavailable."
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run, name="TrainingService", daemon=True
        )
        self._thread.start()
        logger.info("TrainingService worker started.")

    def stop(self) -> None:
        """Stop the poll thread and join it with a timeout.

        Does NOT terminate a running external trainer (see the module
        docstring): the process is left running and re-attached on the next
        :meth:`reconcile_on_startup`. Safe to call multiple times.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning(
                    "TrainingService worker did not stop within timeout; the poll "
                    "loop will exit at its next iteration. The external trainer "
                    "(if any) is intentionally left running for reconcile."
                )
        self._thread = None
        logger.info("TrainingService worker stopped (external trainer left running).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        set_id: int | None = None,
        character_id: int | None = None,
        name: str,
        base_model: str,
        trigger_word: str,
        template_path: str,
        config_overrides: dict | None = None,
        locked: dict | None = None,
        resolution: dict | None = None,
        pins: dict | None = None,
        caption_mode: str | None = None,
        quit_comfyui: bool = False,
    ) -> dict:
        """Create a queued training run and wake the worker.

        At least one of ``set_id`` / ``character_id`` must be given. Both
        mean a character's reference set: the pictures come from the set and
        the run is filed under the character (the route checks the pairing).
        The job is persisted with a fresh uuid and ``status="queued"``; the
        worker picks it up FIFO.

        Args:
            set_id: Picture-set to train on.
            character_id: Character to train on, or to file a reference-set
                run under.
            name: Human-readable job name.
            base_model: Base checkpoint the LoRA trains against.
            trigger_word: Caption trigger word baked into the dataset.
            template_path: Path to the golden ai-toolkit config template.
            config_overrides: The run's tunable values as a config patch
                (kept in the pending spec, not a column).
            locked: The recipe's locked keys as a config patch (pending spec).
            resolution: The resolver's result
                (:func:`pixlstash.training.presets.resolve`). Recorded on the
                row with its ``preset``; its ``vram_mb`` / ``ram_mb`` are what
                the run asks the arbiter for.
            pins: The owner's Advanced values, recorded on the row.
            caption_mode: How the dataset's captions are written
                (:func:`pixlstash.training.presets.caption_mode`); the
                builder's default (tags) when None.
            quit_comfyui: The owner said ComfyUI may be quit if only that
                makes the run fit (#2558).

        Returns:
            The persisted run, as the hub row (a dict of its columns).

        Raises:
            ValueError: If neither ``set_id`` nor ``character_id`` is given.
            RuntimeError: No hub or library is open to hold the run.
            TrainingStoreError: The hub was too busy to write the run, so
                nothing was queued.
        """
        if set_id is None and character_id is None:
            raise ValueError("A training run needs a set_id or a character_id.")
        hub, library_uuid = self._store()

        job_uuid = uuid.uuid4().hex
        try:
            run_id = training_runs.create_run(
                hub,
                library_uuid=library_uuid,
                uuid=job_uuid,
                name=name,
                set_id=set_id,
                character_id=character_id,
                base_model=base_model,
                trigger_word=trigger_word,
                preset=(resolution or {}).get("preset"),
                resolved_profile=json.dumps(resolution) if resolution else None,
                advanced_pins=json.dumps(pins or {}),
                # In the insert itself: the worker's lease reads the answer,
                # and a run it could claim without it would wait for the GPU
                # as though the owner had said no.
                comfyui_quit="allowed" if quit_comfyui else None,
            )
        except sqlite3.OperationalError as exc:
            # A busy or locked hub: the insert rolled back, so there is no run
            # and nothing is staged for one.
            raise TrainingStoreError(
                f"Training run {name!r} could not be written to the hub."
            ) from exc
        self._note_phase(job_uuid, "waiting_gpu")
        job = training_runs.get_run(hub, library_uuid, run_id)
        # The template path and the config patches have no column. Keep them
        # in memory for the common (no-restart) case and also persist a sidecar
        # so a restart can recover a still-queued job's spec.
        pending = {
            "template_path": template_path,
            "config_overrides": config_overrides,
            "locked": locked,
            "caption_mode": caption_mode,
        }
        with self._pending_specs_lock:
            self._pending_specs[job_uuid] = pending
        self._write_pending_sidecar(job_uuid, pending)
        logger.info(
            "Enqueued training job %s (name=%s, set_id=%s, character_id=%s).",
            job_uuid,
            name,
            set_id,
            character_id,
        )
        self._wake.set()
        return job

    def train_longer_options(self, run: dict) -> dict:
        """What Train longer offers *run*: the steps it kept, newest first.

        A checkpoint that does not read whole (cut off mid-save) is not
        offered. Only the newest kept checkpoint keeps the optimizer, which
        ai-toolkit saves beside every checkpoint and so belongs to the newest.

        Args:
            run: The ``training_run`` row.

        Returns:
            ``{"steps": [{"step", "final", "keeps_optimizer"}], "reason"}``.
            ``reason`` says why the run cannot be trained on, or is None.
        """
        handle = handle_for_run(self._training_root, run)
        kept = LocalAiToolkit.kept_checkpoints(handle, run.get("total_steps"))
        final = os.path.join(
            handle.save_root, f"{os.path.basename(handle.save_root)}.safetensors"
        )
        newest = max(kept, default=None)
        optimizer = os.path.isfile(os.path.join(handle.save_root, "optimizer.pt"))
        steps = []
        for step, path in sorted(kept.items(), reverse=True):
            if not safetensors_complete(path):
                logger.warning(
                    "Run %s: %s does not read whole; not offered to train on from.",
                    run["uuid"],
                    path,
                )
                continue
            steps.append(
                {
                    "step": step,
                    "final": path == final,
                    "keeps_optimizer": step == newest and optimizer,
                }
            )
        reason = None
        busy = run["status"] not in _TERMINAL_JOB_STATUSES or (
            run["status"] == "completed" and not run.get("completed_at")
        )
        if busy:
            reason = "It is still training."
        elif not steps:
            reason = "It kept no checkpoint to train on from."
        elif not self._read_json(run.get("rendered_config")):
            reason = "Its settings were not recorded."
        elif not run.get("dataset_snapshot_path") or not os.path.isdir(
            run["dataset_snapshot_path"]
        ):
            reason = "The pictures it trained on are gone from its dataset folder."
        return {"steps": steps, "reason": reason}

    def train_longer(self, run_id: int, *, from_step: int, more_steps: int) -> dict:
        """Queue *run_id* to train on from *from_step*, *more_steps* further.

        The same run, in the same folder, with the same pictures and settings:
        ai-toolkit resumes from the newest checkpoint it finds, so the run
        folder is left with *from_step* as that (see
        :meth:`LocalAiToolkit.carry_on_from`), and the config the run trained
        with is rendered again with only ``train.steps`` raised and the
        first-step preview skipped. The worker then skips the dataset build
        and the memory check: both are the run's own already.

        Args:
            run_id: The ``training_run`` id.
            from_step: A step :meth:`train_longer_options` offers.
            more_steps: How many steps to train past *from_step*.

        Returns:
            The run's row, queued.

        Raises:
            LookupError: No such run in this library.
            ValueError: The run cannot be trained on from *from_step*; the
                message says why.
            TrainingStoreError: The hub was too busy to write the requeue and
                the shelf's final, which are one transaction, so the run keeps
                the status it ended with and is not queued, and its folder is
                put back as it was.
        """
        hub, library_uuid = self._store()
        with self._train_longer_lock:
            run = training_runs.get_run(hub, library_uuid, run_id)
            if run is None:
                raise LookupError(f"No training run {run_id}.")
            options = self.train_longer_options(run)
            if options["reason"]:
                raise ValueError(
                    f"{run['name']} cannot be trained longer. {options['reason']}"
                )
            if from_step not in {o["step"] for o in options["steps"]}:
                raise ValueError(
                    f"{run['name']} kept no checkpoint at step {from_step}."
                )
            if more_steps < 1:
                raise ValueError("Train at least one more step.")
            config = json.loads(run["rendered_config"])
            work_dir = os.path.join(self._training_root, run["uuid"])
            handle = handle_for_run(self._training_root, run)
            final_step = next((o["step"] for o in options["steps"] if o["final"]), None)
            template_path = os.path.join(work_dir, "trained_on_template.json")
            # A press that is not queued takes away a template it was the first
            # to write. One an earlier press left is that press's, read by
            # nothing since its stretch launched; the next press rewrites it.
            template_is_new = not os.path.exists(template_path)
            write_json_atomic(
                template_path, LocalAiToolkit.template_from_rendered(config)
            )
            total = from_step + more_steps
            pending = {
                "template_path": template_path,
                "config_overrides": None,
                "locked": {
                    "config": {
                        "process": [
                            {"train": {"steps": total, "skip_first_sample": True}}
                        ]
                    }
                },
                "caption_mode": None,
                "continuation": True,
            }
            # Everything that can be written first is: the run's folder is
            # taken back only by undoing each rename, not by writing again.
            with self._pending_specs_lock:
                self._pending_specs[run["uuid"]] = pending
            self._write_pending_sidecar(run["uuid"], pending)
            requeued = False
            carried = None
            try:
                carried = LocalAiToolkit.carry_on_from(
                    handle, from_step, run.get("total_steps")
                )
                take = carried.take
                # The press is answered as queued only once the row says so, so
                # the write is tried as often as an end state is.
                for attempt in range(1, _TERMINAL_WRITE_ATTEMPTS + 1):
                    try:
                        # One transaction: the shelf's final renumbered without
                        # the requeue would map the cover to a step of a run
                        # that is not training on.
                        with hub.transaction() as conn:
                            if final_step is not None:
                                # The shelf's copy of the final is that step
                                # from now on; the bare name is the
                                # longer-trained file's once it is imported.
                                conn.execute(
                                    "UPDATE model SET training_step = ? "
                                    "WHERE training_run_id = ? "
                                    "AND training_step IS NULL",
                                    (final_step, run["id"]),
                                )
                            cursor = conn.execute(
                                "UPDATE training_run SET status = 'queued', "
                                "step = ?, total_steps = ?, error = NULL, "
                                "report = NULL, pid = NULL, completed_at = NULL, "
                                "finished_unwatched = 0, comfyui_quit = NULL, "
                                "continued_from_step = ?, continued_at = ?, "
                                "set_aside_take = ?, "
                                # This stretch's phases are timed from its own
                                # start.
                                "phase_log = NULL "
                                "WHERE uuid = ?",
                                (
                                    from_step,
                                    total,
                                    from_step,
                                    now_iso(),
                                    take,
                                    run["uuid"],
                                ),
                            )
                            if cursor.rowcount == 0:
                                # Deleted since it was read: raised inside the
                                # transaction, so the renumbering rolls back.
                                raise TrainingStoreError(
                                    f"Run {run['uuid']} could not be queued to "
                                    f"train on from step {from_step}: its row is "
                                    "gone."
                                )
                        requeued = True
                        break
                    except sqlite3.OperationalError as exc:
                        # A busy or locked hub: the transaction rolled back
                        # whole, the renumbering with it.
                        if attempt == _TERMINAL_WRITE_ATTEMPTS:
                            logger.error(
                                "Run %s: could not be queued to train on from "
                                "step %s after %d tries (%s).",
                                run["uuid"],
                                from_step,
                                attempt,
                                exc,
                                exc_info=True,
                            )
                            raise TrainingStoreError(
                                f"Run {run['uuid']} could not be queued to train "
                                f"on from step {from_step}: the hub was busy."
                            ) from exc
                        # Full jitter, as _update_job waits between tries.
                        delay = random.uniform(
                            0.0,
                            min(
                                _TERMINAL_WRITE_BACKOFF_CAP_S,
                                _TERMINAL_WRITE_BACKOFF_S * 2 ** (attempt - 1),
                            ),
                        )
                        logger.warning(
                            "Run %s: queueing it to train on failed (try %d of "
                            "%d: %s); trying again in %.2f s.",
                            run["uuid"],
                            attempt,
                            _TERMINAL_WRITE_ATTEMPTS,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
            finally:
                if not requeued:
                    # Every way out before the requeue lands. The row still
                    # says how the run ended, so the worker, which claims only
                    # queued runs, never reads this spec: dropping it strands
                    # nothing, and a second press stages a fresh one. The
                    # folder is put back as it was, so the row (which names no
                    # take) still describes it, and a second press is offered
                    # the same steps as the first.
                    with self._pending_specs_lock:
                        self._pending_specs.pop(run["uuid"], None)
                    self._remove_pending_sidecar(run["uuid"])
                    if carried is not None and not LocalAiToolkit.undo_carry_on(
                        handle, carried
                    ):
                        logger.error(
                            "Run %s: the carry-on from step %s could not be "
                            "undone whole after it failed to queue; the paths "
                            "left moved are logged above.",
                            run["uuid"],
                            from_step,
                        )
                    if template_is_new:
                        try:
                            os.remove(template_path)
                        except OSError as exc:
                            logger.warning(
                                "Run %s: could not remove %s after it failed to "
                                "queue: %s",
                                run["uuid"],
                                template_path,
                                exc,
                            )
            with self._phase_lock:
                self._phase_logs.pop(run["uuid"], None)
            self._note_phase(run["uuid"], "waiting_gpu")
            logger.info(
                "Run %s trains on from step %s to %s%s.",
                run["uuid"],
                from_step,
                total,
                f" (later steps set aside in {take})" if take else "",
            )
        self._wake.set()
        return training_runs.get_run(hub, library_uuid, run_id)

    def set_source_folder_registrar(self, register: Callable[[str], None]) -> None:
        """Bind the function that registers the shelf's ``source`` folder.

        The server calls this once it has its folder registration; see the
        ``register_source_folder`` argument of the constructor.

        Args:
            register: Called as ``register(path)``.
        """
        self._register_source_folder = register

    def cancel(self, job_id: int) -> bool:
        """Cooperatively cancel a job by primary-key id.

        Loads the job, and if it is the one currently running, asks the active
        backend to cancel it (cooperative stop, escalating to signals if
        needed). Marks the job ``stopping``; the poll loop observes the terminal
        process state and finalizes it. If the job is queued (not yet started),
        it is marked ``stopped`` directly. A trained run drawing its deferred
        previews skips the rest of them and is imported as it is.

        Every write is conditional on the state the cancel moves the run from,
        so a run that moved on since it was read (launched, or ended) is never
        written back.

        Args:
            job_id: Primary key of the ``training_run`` row to cancel.

        Returns:
            ``True`` if a job was found and a cancel was actioned, else
            ``False`` (no such job, or nothing left to stop: it has ended, or
            it has trained and is being imported).

        Raises:
            TrainingStoreError: A queued job's stop could not be written, so
                it is still queued and would still launch.
        """

        hub, library_uuid = self._store()
        info = training_runs.get_run(hub, library_uuid, job_id)
        if info is None:
            logger.warning("cancel: no training job with id=%s", job_id)
            return False

        if info["status"] == "queued":
            if self._stop_queued(info["uuid"]):
                return True
            # Launched or ended since it was read.
            info = training_runs.get_run(hub, library_uuid, job_id)
            if info is None:
                return False

        if info["status"] not in training_runs.UNFINISHED_STATUSES:
            return self._stop_previews(info)

        # Running/stopping: mark the job 'stopping' BEFORE signalling the
        # backend. backend.cancel() blocks until the process exits, by which
        # point the worker's poll loop may have already finalized the job to its
        # terminal status; writing 'stopping' after that would clobber the
        # terminal state. Marking it first lets the worker's _finalize win, and
        # the write lands only while the run is still running or stopping, so
        # a run that ended since it was read keeps its end state. Under the
        # lock _persist_progress reads and writes under, so a poll cannot read
        # "running" and then write it over this flag.
        with self._active_lock:
            marked = self._update_job(
                info["uuid"], status="stopping", only_if=_ONLY_UNFINISHED
            )
        if marked is False:
            status = self._job_status(info["uuid"])
            if status is None:
                return False
            if status not in training_runs.UNFINISHED_STATUSES:
                # Ended, or trained, since it was read.
                return self._stop_previews({**info, "status": status})
            # Still running, so the flag was lost to a busy hub: the poll
            # keeps the row "running" until the trainer stops, which it is
            # asked to below all the same.
            logger.warning(
                "cancel: job %s could not be marked stopping; asking its "
                "trainer to stop all the same.",
                info["uuid"],
            )
        logger.info("cancel: job %s asked to stop (status=stopping).", info["uuid"])

        with self._active_lock:
            backend = self._active_backend
            active_uuid = self._active_job_uuid
        if backend is not None and active_uuid == info["uuid"]:
            handle = self._handle_for_job(backend, info)
            backend.cancel(handle)
        else:
            # Not the active job (e.g. a re-attached run polled by another
            # cycle): build a transient backend to deliver the cooperative stop.
            transient = self._make_backend()
            handle = self._handle_for_job(transient, info)
            transient.cancel(handle)

        return True

    def _stop_queued(self, job_uuid: str) -> bool:
        """Mark a queued run ``stopped``, and drop the lease it waits on.

        The stop is written only while the run is still queued, and outside
        ``_active_lock``: a write that meets a busy hub is retried for
        seconds, which must not hold up the worker or a poll. The worker's
        wait is read after the write, under the lock the worker starts and
        clears its wait under: either the worker starts waiting after the
        stop, and reads the stop first (``_run_job``), or this sees its
        wait. The lease is released only while still waiting: one granted
        meanwhile belongs to the worker, which releases it when it sees the
        stop.

        Args:
            job_uuid: The run's uuid.

        Returns:
            True when the stop was written. False when the run is no longer
            queued: launched, ended or gone.

        Raises:
            TrainingStoreError: The stop could not be written and the run is
                still queued, so it would still launch.
        """
        if not self._update_job(
            job_uuid,
            status="stopped",
            error="Cancelled before start",
            only_if=_ONLY_QUEUED,
        ):
            if self._job_status(job_uuid) == "queued":
                # Unrecorded, the worker would still launch the run.
                raise TrainingStoreError(
                    f"The stop of queued job {job_uuid} could not be recorded."
                )
            return False
        with self._active_lock:
            waiting = self._waiting
        if waiting is not None and waiting[0] == job_uuid:
            waiting[1].release(only_waiting=True)
        self._emit_progress(
            job_uuid, status="stopped", message="Cancelled before start"
        )
        logger.info("cancel: queued job %s marked stopped.", job_uuid)
        return True

    def _stop_previews(self, info: dict) -> bool:
        """Skip the remaining previews of a run that has trained and draws them.

        Args:
            info: The run's fields, with the status it was last read with.

        Returns:
            True when the run is drawing its deferred previews: the rest are
            skipped and the run is imported as it is. False when there is
            nothing left to stop: the run has ended, or it is being imported.
        """
        with self._active_lock:
            drawing = info["uuid"] in self._drawing_previews
            if drawing:
                self._skip_previews.add(info["uuid"])
        if drawing:
            logger.info("cancel: job %s skips its remaining previews.", info["uuid"])
            return True
        logger.info(
            "cancel: job %s is %s%s; nothing to stop.",
            info["uuid"],
            info["status"],
            " and being imported"
            if info["status"] == "completed" and not info.get("completed_at")
            else "",
        )
        return False

    def adopt_live_runs(self) -> None:
        """Give every run whose trainer outlived the restart its lease at once.

        Called before PixlStash's own GPU workers start, so none of them loads
        a model onto a card a live trainer is using. The lease is granted
        without waiting and stays exclusive for :data:`ADOPT_QUIET_S`;
        :meth:`reconcile_on_startup` then resumes watching the run with it.
        """
        if self._disable_background_workers or self._hub is None:
            return
        try:
            unfinished = training_runs.unfinished_runs(self._hub, self._library_uuid)
        except Exception:
            logger.exception("adopt_live_runs: failed to load unfinished jobs")
            return
        for info in unfinished:
            if not info.get("training_folder"):
                continue
            try:
                backend = self._make_backend()
                if not backend.is_alive(self._handle_for_job(backend, info)):
                    continue
                vram_mb, ram_mb = self._lease_need(info) or (0, 0)
                adopt = self._adopt_gpu or self._request_gpu
                self._adopted[info["uuid"]] = adopt(
                    self._lease_label(info), info["uuid"], vram_mb, ram_mb
                )
                logger.info(
                    "Adopted the live training run %s (pid %s) at start-up.",
                    info["uuid"],
                    info.get("pid"),
                )
            except Exception:
                logger.exception(
                    "adopt_live_runs: could not adopt job %s; reconcile asks for "
                    "its lease instead",
                    info["uuid"],
                )

    def reconcile_on_startup(self) -> None:
        """Resolve jobs left ``running`` / ``stopping`` across a restart.

        For each such job: reconstruct its handle and check liveness. If the
        external trainer is still alive, re-attach: ask for its VRAM lease again and
        resume polling it on the worker thread (a restart must not kill an
        in-flight multi-hour run). If it is dead, finalize from its persisted
        status / on-disk checkpoints (completed → imported to the shelf, else
        ``error``).
        """
        if self._disable_background_workers or self._hub is None:
            return

        try:
            unfinished = training_runs.unfinished_runs(self._hub, self._library_uuid)
        except Exception:
            logger.exception("reconcile_on_startup: failed to load unfinished jobs")
            return

        for info in unfinished:
            try:
                self._reconcile_one(info)
            except Exception:
                logger.exception(
                    "reconcile_on_startup: failed to reconcile job %s", info["uuid"]
                )
        # A lease adopted for a run the reconcile could not take over would keep
        # the card exclusive until the next restart.
        for job_uuid, lease in list(self._adopted.items()):
            logger.warning(
                "reconcile_on_startup: releasing the lease adopted for job %s, "
                "which could not be re-attached.",
                job_uuid,
            )
            lease.release()
        self._adopted.clear()

    def _reconcile_one(self, info: dict) -> None:
        """Reconcile a single unfinished job (alive → re-attach, dead → finish).

        Args:
            info: A dict of persisted job fields (uuid, name, status, pid,
                training_folder, set_id, character_id).
        """
        if not info.get("training_folder"):
            # Never got far enough to launch; treat as interrupted.
            logger.warning(
                "reconcile: job %s has no training_folder; marking error.",
                info["uuid"],
            )
            self._update_job(
                info["uuid"],
                status="error",
                error="Interrupted before launch (no training folder recorded)",
                only_if={"status": (info["status"],)},
            )
            return

        backend = self._make_backend()
        handle = self._handle_for_job(backend, info)
        if backend.is_alive(handle):
            logger.info(
                "reconcile: job %s still alive (pid=%s); re-attaching.",
                info["uuid"],
                info["pid"],
            )
            # Adopted at start-up (adopt_live_runs) when it could be; else asked
            # for, granted once PixlStash has yielded. A live run without a
            # profile (started before recipes) still takes the exclusive lease
            # at 0 MB: it keeps other GPU work off the card under the running
            # trainer, and is released when the run ends.
            lease = self._adopted.pop(info["uuid"], None)
            if lease is None:
                vram_mb, ram_mb = self._lease_need(info) or (0, 0)
                lease = self._request_gpu(
                    self._lease_label(info), info["uuid"], vram_mb, ram_mb
                )
            # Resume polling on the worker thread, owning the lease until the
            # run ends. We deliberately hand off to a fresh thread so startup is
            # not blocked by a multi-hour run.
            self._wake.set()  # ensure the worker is awake to also pick queued work
            reattach = threading.Thread(
                target=self._poll_and_finalize_reattached,
                args=(backend, handle, info, lease),
                name=f"TrainingReattach-{info['uuid'][:8]}",
                daemon=True,
            )
            reattach.start()
            return

        # Dead process (it may have exited since it was adopted): trust the
        # adapter's reconcile to derive the terminal state, then finalize from
        # it. Nobody watched it end, which the live view says.
        adopted = self._adopted.pop(info["uuid"], None)
        if adopted is not None:
            adopted.release()
        logger.info("reconcile: job %s is dead; finalizing from disk.", info["uuid"])
        if info["status"] != "completed":
            # A completed run left unimported was watched to its end.
            self._update_job(info["uuid"], finished_unwatched=1)
        progress = backend.reconcile(handle)
        # Reconciled while the library starts: during a switch, admission opens
        # only once the start-up returns, so an import that must wait for it
        # waits on its own thread.
        self._finalize(backend, handle, info, progress, defer_wait=True)

    def _poll_and_finalize_reattached(
        self, backend: TrainerBackend, handle, info: dict, lease: Lease
    ) -> None:
        """Resume polling a re-attached live job, release its GPU, then finalize.

        Args:
            backend: The backend bound to the live run.
            handle: The reconstructed handle.
            info: Persisted job fields.
            lease: The run's VRAM lease, released once the trainer is done
                with the card: before the import, as :meth:`_run_job` does.
        """
        with self._active_lock:
            self._active_backend = backend
            self._active_job_uuid = info["uuid"]
        watch = _Watch(
            from_start=False,
            offsets=measurements.peaks_read_to_end(
                LocalAiToolkit.peaks_folder(handle.work_dir)
            ),
        )
        released = False
        try:
            progress = self._poll_until_done(
                backend, handle, info["uuid"], lease, watch
            )
            if self._after_training(backend, handle, info, progress, lease, watch):
                released = True
                self._release_gpu(info["uuid"], lease, watch)
                self._finalize(backend, handle, info, progress)
        except Exception:
            logger.exception("Re-attached job %s failed during polling", info["uuid"])
            self._write_failure(info["uuid"], "Polling failed after re-attach")
        finally:
            with self._active_lock:
                # Only the run that holds the slot clears it: a later run may
                # have taken it since this one's lease shrank.
                if self._active_backend is backend:
                    self._active_backend = None
                    self._active_job_uuid = None
            if not released:
                self._release_gpu(info["uuid"], lease, watch)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """The single worker thread: process queued jobs FIFO."""
        while not self._stop.is_set():
            ran = False
            try:
                ran = self._process_next_queued_job()
            except Exception:
                logger.exception("TrainingService worker iteration failed")
            if self._stop.is_set():
                break
            # If we ran a job, immediately look for the next one; otherwise wait
            # for a wake (enqueue/cancel) or a poll interval.
            if not ran:
                self._wake.wait(self.poll_interval_s)
            self._wake.clear()
        logger.info("TrainingService worker loop exited.")

    def _process_next_queued_job(self) -> bool:
        """Pick the oldest queued job and run it to completion.

        Returns:
            ``True`` if a job was processed (so the loop should re-check
            immediately), ``False`` if there was no queued work.
        """

        if self._hub is None:
            return False
        info = training_runs.next_queued_run(self._hub, self._library_uuid)
        if info is None:
            return False

        self._run_job(info)
        return True

    def _run_job(self, info: dict) -> None:
        """Run one job end to end: dataset → lease → train → finalize.

        Args:
            info: Persisted queued-job fields.
        """
        job_uuid = info["uuid"]
        work_dir = os.path.join(self._training_root, job_uuid)
        # Trained longer (train_longer): its folder, pictures and memory
        # profile are the ones it already trained with.
        continuing = bool(self._load_pending_spec(job_uuid).get("continuation"))

        need = self._lease_need(info)
        if need is None:
            self._fail(
                job_uuid,
                "Queued before training recipes existed; queue it again.",
                log=False,
            )
            return

        if continuing:
            training_folder = handle_for_run(self._training_root, info).training_folder
            if training_folder == os.path.join(os.path.abspath(work_dir), "output"):
                training_folder = None  # a run from before the shelf took runs
            manifest = {"dest_dir": info.get("dataset_snapshot_path")}
            if not manifest["dest_dir"] or not os.path.isdir(manifest["dest_dir"]):
                self._fail(
                    job_uuid,
                    "The pictures this run trained on are gone from its dataset "
                    "folder, so it cannot be trained longer.",
                )
                return
        else:
            # 0. Where the run lands and where its dataset is written.
            try:
                training_folder, dataset_dir = self._run_folders(job_uuid)
            except RuntimeError as exc:
                logger.error("Job %s: %s", job_uuid, exc)
                self._fail(job_uuid, str(exc), log=False)
                return

            # 1. Build the dataset snapshot.
            manifest = self._build_dataset(info, dataset_dir)
            if manifest is None:
                return  # dataset error already recorded
            if int(manifest.get("image_count") or 0) <= 0:
                self._fail(job_uuid, "Training dataset is empty (no eligible images).")
                return

            self._update_job(
                job_uuid,
                dataset_snapshot_path=manifest.get("dest_dir"),
                dataset_picture_ids=json.dumps(manifest.get("picture_ids") or []),
            )
            if not self._record_snapshot(info, manifest):
                return  # snapshot error already recorded

        # 2. Wait for the arbiter: PixlStash yields the card, an idle ComfyUI
        # frees it, and the Tasks tab shows the job waiting until then.
        lease = self._request_gpu(self._lease_label(info), job_uuid, *need)
        # Read now, not from the claim: Quit ComfyUI now on the run page may
        # have answered while the dataset was built. It writes the row first
        # and then looks for this lease, so one of the two sees the other.
        run = training_runs.get_run_by_uuid(self._hub, job_uuid) or {}
        if run.get("comfyui_quit") == "allowed":
            lease.allow_comfyui_quit()
        watch = _Watch()
        backend: TrainerBackend | None = None
        released = False
        try:
            # A cancel writes its stop and then looks for this wait under the
            # lock (_stop_queued): read under it, the stop is seen here or the
            # wait is seen there, so a cancelled run never waits on.
            with self._active_lock:
                cancelled = self._job_status(job_uuid) != "queued"
                if not cancelled:
                    self._waiting = (job_uuid, lease)
            granted = False if cancelled else lease.wait(self._stop)
            with self._active_lock:
                self._waiting = None
            if any(w.get("code") == "comfyui_quit" for w in lease.warnings):
                # The run page says, after the run, it can be started again,
                # cancelled meanwhile or not.
                self._update_job(job_uuid, comfyui_quit="quit")
            if self._job_status(job_uuid) != "queued":
                return  # cancelled while it waited
            if not granted:
                if not self._stop.is_set():
                    # Cancelled from the Tasks tab's waiting row.
                    msg = "Cancelled while waiting for the GPU"
                    if self._update_job(
                        job_uuid, status="stopped", error=msg, only_if=_ONLY_QUEUED
                    ):
                        self._emit_progress(job_uuid, status="stopped", message=msg)
                return  # on shutdown it stays queued for the next start
            backend = self._make_backend()
            if not continuing and not self._check_memory_and_speed(
                backend, info, work_dir, manifest, lease
            ):
                return  # stopped, failed or shut down; the row says which
            launched = self._train_under_lease(
                backend,
                info,
                work_dir,
                manifest,
                lease,
                gpu_ids=lease.device or "0",
                training_folder=training_folder,
                watch=watch,
            )
            if launched is None:
                return  # cancelled before launch, or its launch went unrecorded
            handle, progress = launched
            if self._after_training(backend, handle, info, progress, lease, watch):
                # The trainer is done with the card. Bringing the run to the
                # shelf needs none, and may wait out a restore or a library
                # switch, so the lease goes back first: other GPU work runs
                # meanwhile. The row is past queued, so the worker, which
                # claims only queued runs, never claims it again.
                released = True
                self._release_gpu(job_uuid, lease, watch)
                self._finalize(backend, handle, info, progress)
        except Exception:
            logger.exception("Job %s failed during training", job_uuid)
            self._write_failure(job_uuid, "Training run failed")
        finally:
            with self._active_lock:
                # Only the run that holds the slot clears it, as a re-attached
                # run's end does (_poll_and_finalize_reattached).
                if backend is not None and self._active_backend is backend:
                    self._active_backend = None
                    self._active_job_uuid = None
                self._waiting = None
            if not released:
                self._release_gpu(job_uuid, lease, watch)
            # A job still queued at shutdown runs on the next start, and needs
            # its spec then, so only drop it once the job has left the queue.
            if self._job_status(job_uuid) != "queued":
                with self._pending_specs_lock:
                    self._pending_specs.pop(job_uuid, None)
                self._remove_pending_sidecar(job_uuid)

    def _train_under_lease(
        self,
        backend: TrainerBackend,
        info: dict,
        work_dir: str,
        manifest: dict,
        lease: Lease,
        gpu_ids: str = "0",
        training_folder: str | None = None,
        watch: _Watch | None = None,
    ) -> tuple[object, TrainingProgress] | None:
        """Prepare + submit + poll a run while the GPU lease is held.

        The run is launched only while it is still queued, and marked
        ``running`` only from queued: a cancel during the memory check's last
        seconds, or between the claim and the launch, is never overwritten.
        The launch write is the row's only record of the trainer's pid and
        folder, which the start-up reconcile re-attaches by, so it is retried
        on a busy hub like an end state. A trainer whose launch could not be
        recorded is stopped at once rather than left running unrecorded: see
        :meth:`_abandon_launch`.

        Args:
            backend: The trainer backend.
            info: Queued-job fields.
            work_dir: The per-job work directory.
            manifest: The dataset manifest (provides ``dest_dir``).
            lease: The run's granted lease; its progress fields are kept live.
            gpu_ids: ``CUDA_VISIBLE_DEVICES`` for the trainer: the leased
                card's UUID, so it trains on the card the arbiter cleared.
            training_folder: The shelf source folder the run is written into.
            watch: Collects the run's phase peaks and times.

        Returns:
            A ``(handle, progress)`` tuple: the launched handle and the terminal
            :class:`TrainingProgress` snapshot. None when the run was cancelled
            before it could launch, or its launch could not be recorded.
        """
        job_uuid = info["uuid"]
        pending = self._load_pending_spec(job_uuid)
        spec = TrainingSpec(
            job_id=job_uuid,
            job_name=info["name"],
            template_path=pending.get("template_path"),
            dataset_dir=manifest["dest_dir"],
            trigger_word=info.get("trigger_word") or "",
            work_dir=work_dir,
            hf_token=os.environ.get("HF_TOKEN") or None,
            config_overrides=pending.get("config_overrides"),
            locked=pending.get("locked"),
            gpu_ids=gpu_ids,
            training_folder=training_folder,
        )
        # The commit is what the checkout holds when the job starts, read from
        # git and never from the owner's config (review F6).
        cfg = self._config_provider() or {}
        commit = (
            ai_toolkit_install.read_commit(cfg["repo_path"])
            if cfg.get("repo_path")
            else None
        )
        with self._active_lock:
            if self._job_status(job_uuid) != "queued":
                logger.info("Job %s was cancelled before launch.", job_uuid)
                return None
            handle = backend.prepare(spec)
            backend.submit(handle, spec)
            self._active_backend = backend
            self._active_job_uuid = job_uuid
        # Outside the lock: a busy hub's retries must not hold up a cancel. A
        # cancel meanwhile reads the run queued and writes its stop from
        # queued, so exactly one of the two writes lands: this one, and the
        # cancel stops the trainer as a running one; or the stop, and the
        # trainer is stopped here.
        config = self._read_rendered_config(handle.config_path)
        total_steps = self._total_steps_from_config(config)
        if not self._update_job(
            job_uuid,
            status="running",
            pid=handle.pid,
            training_folder=handle.training_folder,
            rendered_config=json.dumps(config) if config is not None else None,
            ai_toolkit_commit=commit,
            total_steps=total_steps,
            started_at=now_iso(),
            only_if=_ONLY_QUEUED,
            attempts=_TERMINAL_WRITE_ATTEMPTS,
        ):
            self._abandon_launch(backend, handle, job_uuid)
            return None
        self._note_phase(job_uuid, "loading")
        self._emit_progress(
            job_uuid, status="running", current=0, total=total_steps, message="Started"
        )

        progress = self._poll_until_done(backend, handle, job_uuid, lease, watch)
        return handle, progress

    def _abandon_launch(self, backend: TrainerBackend, handle, job_uuid: str) -> None:
        """Stop a trainer whose launch is not on its row.

        Without it the row names neither the trainer's pid nor its folder, so
        nothing re-attaches to it after a restart and it trains under a lease
        nobody holds. Stopped first, the row is then left saying where the run
        is: stopped by a cancel that came first, or, while still queued, an
        error. A row that cannot take even that write stays queued, and the
        worker tries the run again.

        Args:
            backend: The backend the trainer was submitted to.
            handle: The trainer's handle.
            job_uuid: The run's uuid.
        """
        backend.cancel(handle)
        status = self._job_status(job_uuid)
        if status != "queued":
            logger.info(
                "Job %s was %s before its launch was recorded; its trainer "
                "(pid %s) is stopped.",
                job_uuid,
                status,
                handle.pid,
            )
            return
        logger.error(
            "Job %s: its launch could not be recorded, so its trainer (pid %s) "
            "is stopped.",
            job_uuid,
            handle.pid,
        )
        self._fail(
            job_uuid,
            "The run could not be recorded as started, so it was stopped.",
            log=False,
        )

    def _poll_until_done(
        self,
        backend: TrainerBackend,
        handle,
        job_uuid: str,
        lease: Lease,
        watch: _Watch | None = None,
    ) -> TrainingProgress:
        """Poll progress until the process is dead AND status is terminal.

        Persists step / loss curve / samples / log tail and emits a
        PLUGIN_PROGRESS event each cycle. Each cycle also reads what the
        trainer holds into *watch* and onto the lease, and lowers the lease to
        the run's measured peak once its first checkpoint is saved.

        Args:
            backend: The trainer backend.
            handle: The job handle.
            job_uuid: The job uuid (used as the event ``run_id``).
            lease: The run's lease; the Tasks tab's AI Toolkit row reads its
                step, total, latest loss, phase and memory from it.
            watch: Collects the phase peaks and times; None keeps a private one.

        Returns:
            The final :class:`TrainingProgress` snapshot.
        """
        watch = watch if watch is not None else _Watch()
        deferred = bool(self._resolution(job_uuid).get("previews_deferred"))
        progress = backend.progress(handle)
        try:
            while not self._stop.is_set():
                self._persist_progress(job_uuid, progress, handle)
                phase = live_view.phase_of(
                    progress.status, progress.info, progress.step, None
                )
                self._observe(handle, progress, lease, watch, phase)
                self._maybe_shrink(job_uuid, lease, watch, progress, deferred)
                self._emit_progress(
                    job_uuid,
                    status=progress.status,
                    current=progress.step,
                    total=progress.total_steps,
                    message=progress.info,
                )
                alive = backend.is_alive(handle)
                terminal = progress.status in _TERMINAL_PROGRESS_STATUSES
                if not alive and terminal:
                    break
                if not alive and not terminal:
                    # Process gone but status not terminal; let the adapter
                    # reconcile the real end state (interrupted → error), then stop.
                    progress = backend.reconcile(handle)
                    self._persist_progress(job_uuid, progress, handle)
                    break
                self._stop.wait(self.poll_interval_s)
                progress = backend.progress(handle)
        finally:
            # The next loop over this run starts by writing every column again.
            with self._active_lock:
                self._persisted.pop(job_uuid, None)
        return progress

    # ------------------------------------------------------------------
    # Memory and speed (#2120)
    # ------------------------------------------------------------------

    def _resolution(self, job_uuid: str) -> dict:
        """The run's recorded resolution; {} when it has none or it is unreadable."""
        run = training_runs.get_run_by_uuid(self._hub, job_uuid)
        try:
            resolution = json.loads((run or {}).get("resolved_profile") or "null")
        except ValueError as exc:
            logger.warning("Job %s has an unreadable resolution: %s", job_uuid, exc)
            return {}
        return resolution if isinstance(resolution, dict) else {}

    def _observe(
        self,
        handle,
        progress: TrainingProgress,
        lease: Lease,
        watch: _Watch,
        phase: str | None,
        **extra,
    ) -> None:
        """Read what the trainer holds into *watch* and onto the lease.

        The lease's progress is the worker's phase telemetry: the Tasks tab
        and the live view read the phase and the trainer's memory from it.
        """
        usage = measurements.trainer_usage(
            handle.pid,
            lease.device,
            LocalAiToolkit.peaks_folder(handle.work_dir),
            watch.offsets,
        )
        watch.see(phase, usage, time.monotonic())
        if self._tenant is not None:
            self._tenant.track(usage["pids"])
        lease.set_progress(
            step=progress.step,
            total=progress.total_steps,
            loss=progress.loss_curve[-1].get("loss") if progress.loss_curve else None,
            phase=phase,
            vram_mb=usage["vram_mb"],
            ram_mb=usage["ram_mb"],
            **extra,
        )

    def _maybe_shrink(
        self,
        job_uuid: str,
        lease: Lease,
        watch: _Watch,
        progress: TrainingProgress,
        deferred: bool,
    ) -> None:
        """Lower the lease to the run's peak once its first checkpoint is saved.

        By then it has loaded, cached, trained, saved and (unless its previews
        are deferred) drawn a round of previews, so its peak has been seen. A
        round is drawn once the run has left the previews phase: the first
        poll inside it comes before the VAE decode each preview ends with.
        The lease stops being exclusive, and other GPU work may use what the
        run does not. An adopted lease refuses while it is quiet; the next
        poll tries again.
        """
        if watch.shrunk or not (progress.checkpoints or progress.final_checkpoint):
            return
        if "training" not in watch.peaks:
            return
        if not deferred and (
            "previews" not in watch.peaks or watch.phase == "previews"
        ):
            return
        peak = watch.peak_vram_mb()
        if peak is None:
            return
        need = measurements.with_margin(peak)
        if lease.shrink(need):
            watch.shrunk = True
            logger.info(
                "Job %s: first checkpoint saved; its lease now asks for %s MB "
                "(peak %s MB).",
                job_uuid,
                need,
                peak,
            )

    def _check_released(self, job_uuid: str, lease: Lease, watch: _Watch) -> None:
        """Warn when a trainer that exited left its GPU memory held.

        Its processes get :data:`RELEASE_GRACE_S` to let go. Whatever they
        still hold is announced on the lease and named "memory not released"
        in the Tasks tab rather than counted as other apps'.
        """
        if not watch.pids or lease.device is None or self._stop.is_set():
            return
        deadline = time.monotonic() + RELEASE_GRACE_S
        held = measurements.gpu_mb(watch.pids, lease.device)
        while held and time.monotonic() < deadline and not self._stop.is_set():
            self._stop.wait(1.0)
            held = measurements.gpu_mb(watch.pids, lease.device)
        if not held:
            if self._tenant is not None:
                self._tenant.ended(set())
            return
        leftover = {
            pid for pid in ledger().processes(lease.device) if pid in watch.pids
        }
        text = (
            f"AI-toolkit exited but {held / 1024:.1f} GB of GPU memory was not "
            f"released (process {', '.join(str(p) for p in sorted(leftover))}). "
            "Other GPU work counts it as used until those processes end."
        )
        logger.warning("Job %s: %s", job_uuid, text)
        lease.warn("aitoolkit_memory_not_released", text)
        if self._tenant is not None:
            self._tenant.ended(leftover)

    def _check_memory_and_speed(
        self,
        backend: TrainerBackend,
        info: dict,
        work_dir: str,
        manifest: dict,
        lease: Lease,
    ) -> bool:
        """Probe a run whose setup this machine has not measured yet.

        The probe trains a few updates at the largest bucket, draws one preview
        and saves once, under a RAM guard, and records what each
        phase peaked at. Running out of GPU memory steps to the next profile
        of the fit choice (more offloading); a loader error, a Triton error or
        the RAM guard ends the search. A profile that held but whose previews
        do not fit beside training steps on too, so previews are drawn while
        training wherever a profile allows it. The run then trains with the
        first profile that held with its previews, or else the first that held,
        re-resolved on its measurement, its previews deferred.

        Returns:
            True to train. False when the run was stopped, failed (the row says
            why) or PixlStash is shutting down (it stays queued).
        """
        job_uuid = info["uuid"]
        resolution = self._resolution(job_uuid)
        if "probe" not in resolution:
            return True  # resolved without measurements
        preset_id = str(resolution.get("preset") or "").split("@")[0]
        preset = presets.get_preset(preset_id)
        if preset is None:
            logger.warning(
                "Job %s: recipe %s is gone; training without checking memory.",
                job_uuid,
                resolution.get("preset"),
            )
            return True
        cfg = self._config_provider() or {}
        commit = (
            ai_toolkit_install.read_commit(cfg["repo_path"])
            if cfg.get("repo_path")
            else None
        )
        lookup = measurements.make_lookup(
            self._hub, preset, commit=commit, device=self._device_name(lease.device)
        )
        # Resolved again at launch even when it was measured at enqueue: the
        # checkout or the card may have changed since, which is a new setup.
        # The profiles passed over for their previews come first: measured,
        # they cost no probe and are kept to fall back on (#2411).
        names = [
            *(resolution.get("passed_over") or []),
            resolution["profile"],
            *(resolution.get("fallbacks") or []),
        ]
        reason = None
        # The first profile that held but whose previews do not fit: kept while
        # the next ones are tried for previews drawn while training (#2411).
        deferred = None
        probes = 0
        for name in names:
            try:
                settled = self._reresolve(info, resolution, preset, lookup, name)
                if settled["probe"]:
                    if probes == MAX_PROBES:
                        # Out of probes; a measured profile further on costs none.
                        continue
                    probes += 1
                    self._note_phase(job_uuid, "checking")
                    outcome, reason = self._probe_once(
                        backend, info, work_dir, manifest, lease, settled
                    )
                    if outcome == "oom":
                        logger.info(
                            "Job %s: %s ran out of GPU memory; trying the next "
                            "profile.",
                            job_uuid,
                            name,
                        )
                        continue
                    if outcome == "failed" and deferred is not None:
                        # A profile that trains is in hand: its previews wait.
                        logger.warning(
                            "Job %s: checking %s failed (%s); training with %s, "
                            "previews after training.",
                            job_uuid,
                            name,
                            reason,
                            deferred["profile"],
                        )
                        break
                    if outcome == "failed":
                        self._fail(job_uuid, reason)
                    if outcome != "ok":
                        return False
                    settled = self._reresolve(info, resolution, preset, lookup, name)
            except presets.NoProfileFits as exc:
                reason = str(exc)
                logger.info("Job %s: %s does not fit: %s", job_uuid, name, exc)
                continue
            sampling = settled["values"].get("train.disable_sampling") or {}
            if settled["previews_deferred"] and sampling.get("source") == "gpu":
                logger.info(
                    "Job %s: %s's previews do not fit while training; trying "
                    "the next profile.",
                    job_uuid,
                    name,
                )
                deferred = deferred or settled
                continue
            self._settle(job_uuid, settled)
            return True
        if deferred is not None:
            self._settle(job_uuid, deferred)
            return True
        self._fail(
            job_uuid,
            "Not enough GPU memory: none of this recipe's ways of fitting the GPU "
            "held while checking memory and speed"
            + (f" ({reason})" if reason else "")
            + ".",
        )
        return False

    def _reresolve(
        self, info: dict, resolution: dict, preset: dict, lookup, profile: str
    ) -> dict:
        """The run resolved again on *profile* alone, measurements included.

        The same choices, pictures, room, pins, prompts and preservation as at
        enqueue.
        """
        try:
            pins = json.loads(info.get("advanced_pins") or "{}")
        except ValueError as exc:
            logger.warning("Job %s has unreadable pins: %s", info["uuid"], exc)
            pins = {}
        written = {
            key: entry.get("value")
            for key, entry in (resolution.get("values") or {}).items()
            if isinstance(entry, dict) and entry.get("source") == "captions"
        }
        prompts = written.get(presets.PROMPTS_KEY)
        preservation = {
            key: written[key]
            for key in (presets.PRESERVATION_KEY, presets.PRESERVATION_CLASS_KEY)
            if key in written
        }
        room = resolution.get("room") or {}
        try:
            return presets.resolve(
                preset,
                subject=resolution["subject"],
                effort=resolution["effort"],
                fit=resolution["fit"],
                picture_count=int(resolution["picture_count"]),
                vram_mb=int(room["vram_mb"]),
                ram_mb=room.get("ram_mb"),
                pins=pins,
                prompts=prompts,
                preservation=preservation,
                lookup=lookup,
                profile=profile,
            )
        except presets.NoProfileFits:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Job %s: profile %s could not be resolved again.",
                info["uuid"],
                profile,
                exc_info=True,
            )
            raise presets.NoProfileFits(RESOLVE_AGAIN_FAILED, "profile", 0, 0) from exc

    def _settle(self, job_uuid: str, settled: dict) -> None:
        """Make *settled* the run's resolution and what it renders."""
        pending = dict(self._load_pending_spec(job_uuid))
        pending["config_overrides"] = presets.to_patch(presets.resolved_values(settled))
        with self._pending_specs_lock:
            self._pending_specs[job_uuid] = pending
        self._write_pending_sidecar(job_uuid, pending)
        # The recipe may be a newer version than at enqueue (#2122): the row
        # names the one it now trains with.
        self._update_job(
            job_uuid, resolved_profile=json.dumps(settled), preset=settled["preset"]
        )
        logger.info(
            "Job %s trains with %s (%s; previews %s).",
            job_uuid,
            settled["profile"],
            settled["evidence"],
            "deferred" if settled["previews_deferred"] else "while training",
        )

    def _stop_orphaned_pass(
        self, backend: TrainerBackend, job_uuid: str, name: str, work_dir: str
    ) -> None:
        """Stop the trainer a crashed probe or preview pass left running.

        A pass records its pid on the run, as the run does. Before a new pass
        clears its directory and launches, the trainer that pid names is
        cancelled if it is a pass of this run, so a second one never trains
        beside it on the card. The pid may belong to any earlier pass (the
        probe, or a preview pass at another step), so each pass directory the
        run has on disk is tried. Any other process is left alone.

        Args:
            backend: The trainer backend.
            job_uuid: The run's uuid.
            name: The run's display name, as the pass's paths derive from it.
            work_dir: The run's work directory, holding ``probe`` and
                ``after/<step>``.
        """
        run = training_runs.get_run_by_uuid(self._hub, job_uuid) or {}
        pid = run.get("pid")
        if not pid:
            return
        after = os.path.join(work_dir, "after")
        steps = sorted(os.listdir(after)) if os.path.isdir(after) else []
        pass_dirs = [os.path.join(work_dir, "probe")]
        pass_dirs += [os.path.join(after, step) for step in steps]
        for pass_dir in pass_dirs:
            handle = LocalAiToolkit.handle_for(pass_dir, job_uuid, name, pid)
            if backend.is_alive(handle):
                logger.warning(
                    "Job %s: stopping pid %s, a pass of it left running in %s.",
                    job_uuid,
                    pid,
                    pass_dir,
                )
                backend.cancel(handle)
                return

    def _probe_once(
        self,
        backend: TrainerBackend,
        info: dict,
        work_dir: str,
        manifest: dict,
        lease: Lease,
        settled: dict,
    ) -> tuple[str, str | None]:
        """Run one probe of *settled*'s profile and record what it measured.

        Returns:
            ``(outcome, reason)``: ``ok`` (measured and recorded), ``oom`` (try
            more offloading), ``failed`` (the RAM guard or a trainer error
            stopped it; *reason* is the run's error) or ``stopped`` (the run
            was cancelled, failed or PixlStash is shutting down; the row
            already says which).
        """
        job_uuid = info["uuid"]
        pending = self._load_pending_spec(job_uuid)
        values = presets.resolved_values(settled)
        # One preview peaks and takes as long as any other, and each one more
        # costs a whole preview (a minute on a large model) before the run.
        if values.get(presets.PROMPTS_KEY):
            values[presets.PROMPTS_KEY] = values[presets.PROMPTS_KEY][:1]
        process: dict = {
            "train": {
                "steps": PROBE_STEPS,
                "skip_first_sample": True,
                "disable_sampling": False,
            },
            "save": {"save_every": PROBE_STEPS},
            "sample": {"sample_every": PROBE_STEPS},
        }
        sizes = values.get("datasets[*].resolution") or []
        if sizes:
            # The worst bucket only, so the peak is the largest a step makes.
            process["datasets"] = [{"resolution": [max(sizes)]}]
        probe_dir = os.path.join(work_dir, "probe")
        self._stop_orphaned_pass(backend, job_uuid, info["name"], work_dir)
        self._remove_tree(probe_dir)
        spec = TrainingSpec(
            job_id=job_uuid,
            job_name=info["name"],
            template_path=pending.get("template_path"),
            dataset_dir=manifest["dest_dir"],
            trigger_word=info.get("trigger_word") or "",
            work_dir=probe_dir,
            hf_token=os.environ.get("HF_TOKEN") or None,
            config_overrides=presets.to_patch(values),
            locked=deep_merge(
                pending.get("locked") or {}, {"config": {"process": [process]}}
            ),
            gpu_ids=lease.device or "0",
        )
        handle = backend.prepare(spec)
        backend.submit(handle, spec)
        self._update_job(job_uuid, pid=handle.pid)
        started = time.time()
        watch = _Watch()

        def stop_probe() -> str | None:
            if self._job_status(job_uuid) != "queued":
                return "stopped"
            available = read_ram().get("available_mb")
            if available and available < RAM_GUARD_MB:
                return "ram"
            return "shutdown" if self._stop.is_set() else None

        progress, guard = self._watch_pass(
            backend, handle, lease, watch, "checking", stop_probe
        )
        self._check_released(job_uuid, lease, watch)
        if guard == "stopped":
            return "stopped", None
        if guard == "shutdown":
            logger.info("Job %s: probe stopped by shutdown; stays queued.", job_uuid)
            return "stopped", None
        if guard == "ram":
            return "failed", (
                "Not enough system RAM to start: less than "
                f"{RAM_GUARD_MB // 1024} GB was free while checking memory and "
                "speed, so the check was stopped. Close other programs or choose "
                "a fit that offloads less."
            )
        text = f"{progress.info or ''}\n{self._read_log_tail(handle.log_path) or ''}"
        if live_view.OOM_RE.search(text):
            return "oom", settled["profile_label"]
        if progress.status != "completed" or not progress.final_checkpoint:
            return "failed", (
                "Checking memory and speed failed: "
                + (
                    live_view.failure_summary(progress.info, progress.step)
                    if progress.info
                    else "the trainer stopped with an error"
                ).rstrip(".")
                + ". See the trainer log."
            )
        step_times = live_view.read_step_times(handle.loss_db_path)
        images = len(progress.samples)
        try:
            measurements.record(
                self._hub,
                settled["fingerprint"],
                preset=settled["preset"],
                profile=settled["profile"],
                device=self._device_name(lease.device),
                phase_peaks=watch.peaks,
                seconds_per_update=live_view.speed(step_times, None)[
                    "seconds_per_step"
                ],
                setup_seconds=(step_times[0][1] - started) if step_times else None,
                seconds_per_preview=(
                    watch.seconds["previews"] / images
                    if images and watch.seconds.get("previews")
                    else None
                ),
                source="probe",
            )
        except Exception:
            logger.exception("Job %s: could not record the probe", job_uuid)
            self._fail(job_uuid, "Checking memory and speed could not be recorded.")
            return "stopped", None
        self._remove_tree(handle.save_root)
        return "ok", None

    def _watch_pass(
        self,
        backend: TrainerBackend,
        handle,
        lease: Lease,
        watch: _Watch,
        shown_as: str,
        should_stop: Callable[[], str | None],
    ) -> tuple[TrainingProgress, str | None]:
        """Poll a short trainer pass (a probe, an after-training preview) to its end.

        The lease shows *shown_as* as the phase, since the trainer's own info
        line cannot tell these passes from training. *should_stop* is asked
        each poll; what it answers stops the pass and is returned with it.
        """
        while True:
            progress = backend.progress(handle)
            phase = live_view.phase_of(
                progress.status, progress.info, progress.step, None
            )
            self._observe(handle, progress, lease, watch, phase, shown_as=shown_as)
            stop = should_stop()
            if stop is not None:
                backend.cancel(handle)
                return backend.progress(handle), stop
            if not backend.is_alive(handle):
                if progress.status not in _TERMINAL_PROGRESS_STATUSES:
                    progress = backend.reconcile(handle)
                return progress, None
            self._stop.wait(_SHORT_POLL_S)

    def _after_training(
        self,
        backend: TrainerBackend,
        handle,
        info: dict,
        progress: TrainingProgress,
        lease: Lease,
        watch: _Watch,
    ) -> bool:
        """What a finished run leaves before it is imported.

        Its measurement is recorded for the next run of the same setup, and
        a run whose previews were deferred has each kept checkpoint sampled.

        Returns:
            False when a shutdown stopped the previews: the run is left
            completed and unimported, and the next start's reconcile imports
            it. True to import now.
        """
        if progress.status != "completed":
            return True
        resolution = self._resolution(info["uuid"])
        # Recorded even on shutdown: only the preview passes wait for a start.
        self._record_run(info["uuid"], handle, progress, lease, watch, resolution)
        if self._stop.is_set():
            return True
        if resolution.get("previews_deferred") and progress.final_checkpoint:
            # A cancel skips the remaining previews only while they are drawn
            # (_stop_previews), so no skip outlives them.
            with self._active_lock:
                self._drawing_previews.add(info["uuid"])
            try:
                return self._sample_after_training(
                    backend, handle, info, progress, lease
                )
            finally:
                with self._active_lock:
                    self._drawing_previews.discard(info["uuid"])
                    self._skip_previews.discard(info["uuid"])
        return True

    def _record_run(
        self,
        job_uuid: str,
        handle,
        progress: TrainingProgress,
        lease: Lease,
        watch: _Watch,
        resolution: dict,
    ) -> None:
        """Store what a finished run measured under its setup's fingerprint."""
        key = resolution.get("fingerprint")
        if not key or not watch.peaks:
            return
        step_times = live_view.read_step_times(handle.loss_db_path)
        setup = None
        if watch.from_start and "loading" in watch.seconds:
            setup = watch.seconds["loading"] + watch.seconds.get("caching", 0.0)
        previews = watch.seconds.get("previews")
        images = len(progress.samples)
        try:
            measurements.record(
                self._hub,
                key,
                preset=resolution["preset"],
                profile=resolution["profile"],
                device=self._device_name(lease.device),
                phase_peaks=watch.peaks,
                seconds_per_update=live_view.speed(step_times, None)[
                    "seconds_per_step"
                ],
                setup_seconds=setup,
                seconds_per_preview=previews / images
                if previews and images and watch.from_start
                else None,
                source="run",
            )
        except Exception:
            logger.exception("Job %s: could not record its measurement", job_uuid)

    def _sample_after_training(
        self,
        backend: TrainerBackend,
        handle,
        info: dict,
        progress: TrainingProgress,
        lease: Lease,
    ) -> bool:
        """Draw the previews of a run that trained without them.

        Each kept checkpoint is loaded on its own (ai-toolkit samples a
        pretrained LoRA at a given step without training), and the pictures
        are moved into the run's ``samples/`` under that step. A pass that
        fails is logged and skipped: the run itself is done. A cancel stops the
        passes and the run is imported as it is.

        Returns:
            False when a shutdown stopped them, else True.
        """
        job_uuid = info["uuid"]
        pending = self._load_pending_spec(job_uuid)
        run = training_runs.get_run_by_uuid(self._hub, job_uuid) or {}
        self._note_phase(job_uuid, "previews")

        def stop() -> str | None:
            if self._stop.is_set():
                return "shutdown"
            return "cancelled" if job_uuid in self._skip_previews else None

        checkpoints = [*progress.checkpoints, progress.final_checkpoint]
        drawn: set[int] = set()
        # A run trained longer drew the steps it had before then.
        done = info.get("continued_from_step")
        for path in checkpoints:
            # The final is ``<job_name>.safetensors``, with no step to read: a
            # job name ending in digits would otherwise read as one.
            match = (
                None
                if path == progress.final_checkpoint
                else _CHECKPOINT_STEP_RE.search(os.path.basename(path))
            )
            step = int(match.group(1)) if match else int(progress.step)
            if step in drawn or (done is not None and step <= done):
                continue  # the final can share its step with the last save
            drawn.add(step)
            if stop() is not None:
                break
            pass_dir = os.path.join(handle.work_dir, "after", str(step))
            self._stop_orphaned_pass(backend, job_uuid, info["name"], handle.work_dir)
            self._remove_tree(pass_dir)
            process = {
                "train": {
                    "steps": step,
                    "start_step": step,
                    "skip_first_sample": True,
                    "disable_sampling": False,
                },
                "network": {"pretrained_lora_path": path},
            }
            spec = TrainingSpec(
                job_id=job_uuid,
                job_name=info["name"],
                template_path=pending.get("template_path"),
                dataset_dir=run.get("dataset_snapshot_path") or "",
                trigger_word=info.get("trigger_word") or "",
                work_dir=pass_dir,
                hf_token=os.environ.get("HF_TOKEN") or None,
                config_overrides=pending.get("config_overrides"),
                locked=deep_merge(
                    pending.get("locked") or {}, {"config": {"process": [process]}}
                ),
                gpu_ids=lease.device or "0",
            )
            try:
                pass_handle = backend.prepare(spec)
                backend.submit(pass_handle, spec)
                self._update_job(job_uuid, pid=pass_handle.pid)
                ended, stopped = self._watch_pass(
                    backend, pass_handle, lease, _Watch(), "previews", stop
                )
                if stopped is not None:
                    break
                if ended.status != "completed":
                    logger.warning(
                        "Job %s: previews of step %s failed: %s",
                        job_uuid,
                        step,
                        ended.info,
                    )
                    continue
                self._move_samples(pass_handle.save_root, handle.save_root)
            except Exception:
                logger.exception(
                    "Job %s: could not draw the previews of step %s", job_uuid, step
                )
            finally:
                self._remove_tree(pass_dir)
        if stop() == "shutdown":
            logger.info(
                "Job %s: previews stopped by shutdown; it is imported on the next "
                "start.",
                job_uuid,
            )
            return False
        self._note_phase(job_uuid, "importing")
        return True

    @staticmethod
    def _move_samples(from_root: str, to_root: str) -> None:
        """Move the pictures in ``<from_root>/samples`` into ``<to_root>/samples``."""
        source = os.path.join(from_root, "samples")
        if not os.path.isdir(source):
            return
        target = os.path.join(to_root, "samples")
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(source):
            shutil.move(os.path.join(source, name), os.path.join(target, name))

    @staticmethod
    def _remove_tree(path: str) -> None:
        """Remove a folder the service made for a run, if it is there."""
        if not os.path.isdir(path):
            return
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path, exc)

    @staticmethod
    def _device_name(device: str | None) -> str | None:
        """The card's name, which a measurement is tied to; None when unknown."""
        reading = ledger().device(device) if device else None
        return reading.name if reading is not None else None

    def _fail(self, job_uuid: str, message: str, *, log: bool = True) -> None:
        """End a run that never trained as ``error`` with *message*.

        Written only while the run is still queued: a cancel that stopped it
        meanwhile keeps its stop.

        Args:
            job_uuid: The run's uuid.
            message: The error the run's card shows.
            log: False when the caller has logged the cause already.
        """
        if log:
            logger.warning("Job %s: %s", job_uuid, message)
        if self._update_job(
            job_uuid, status="error", error=message, only_if=_ONLY_QUEUED
        ):
            self._emit_progress(job_uuid, status="error", message=message)

    def _write_failure(self, job_uuid: str, message: str) -> None:
        """End a run an unexpected error broke off as ``error`` with *message*.

        It may have broken off anywhere from its claim to its import, so it is
        written over any state short of the run's end, and never over an end
        already written.
        """
        if self._update_job(
            job_uuid, status="error", error=message, only_if=_ONLY_UNENDED
        ) or self._update_job(
            job_uuid, status="error", error=message, only_if=_ONLY_COMPLETED_UNIMPORTED
        ):
            self._emit_progress(job_uuid, status="error", message=message)

    def _release_gpu(self, job_uuid: str, lease: Lease, watch: _Watch) -> None:
        """Give the run's GPU lease back once its trainer has let go of the card.

        The check before it only warns about memory the trainer kept. It runs
        before the import, so its failure is logged and costs that warning, not
        the trained run: the lease goes back and the import goes on, as when
        the check ran after the import and its error ended in the worker loop's
        log.

        Args:
            job_uuid: The run's uuid.
            lease: The run's lease.
            watch: The run's watch, which names the trainer's processes.
        """
        try:
            self._check_released(job_uuid, lease, watch)
        except Exception:
            logger.exception(
                "Job %s: could not check what GPU memory its trainer (pids %s) "
                "let go of on device %s; its lease is released all the same.",
                job_uuid,
                sorted(watch.pids),
                lease.device,
            )
        finally:
            lease.release()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(
        self,
        backend: TrainerBackend,
        handle,
        info: dict,
        progress: TrainingProgress,
        *,
        defer_wait: bool = False,
    ) -> None:
        """Write the terminal job state and, on success, bring the run to the shelf.

        Called once the run's GPU lease is released: the import needs no GPU,
        and may wait out a restore or a library switch.

        Args:
            backend: The trainer backend (unused beyond context, kept for
                symmetry / future per-backend finalization).
            handle: The job handle.
            info: Persisted job fields (needs uuid, name, set_id, character_id).
            progress: The terminal progress snapshot.
            defer_wait: An import that must wait for a library switch waits on
                a thread of its own, and this returns at once. For the
                start-up reconcile, which a switch's admission waits on.
        """
        job_uuid = info["uuid"]
        status = progress.status

        if status in _TERMINAL_PROGRESS_STATUSES:
            with self._phase_lock:
                self._phase_logs.pop(job_uuid, None)

        if self._stop.is_set() and status not in _TERMINAL_PROGRESS_STATUSES:
            # Shutdown interrupted the poll while the trainer is still running.
            # Leave the row as it is: reconcile_on_startup re-attaches to the
            # live process, or finalizes it from disk once it has exited.
            logger.info(
                "Job %s is still %s at shutdown; leaving it for reconcile_on_startup.",
                job_uuid,
                status,
            )
            return

        if status == "completed":
            try:
                # A run finalized again (its end state was lost to a busy hub,
                # or a restart or its library's refusal came during its
                # import) may be on the shelf already. Its source folder may
                # then have taken the run's files, so there is no final to read.
                landed = self._landed_on_shelf(info, progress.final_checkpoint)
                if landed is None and not progress.final_checkpoint:
                    msg = "Run reported completed but no final checkpoint was found."
                    logger.warning("Job %s: %s", job_uuid, msg)
                    self._update_job(
                        job_uuid,
                        status="error",
                        error=msg,
                        completed_at=now_iso(),
                    )
                    self._emit_progress(job_uuid, status="error", message=msg)
                    return
                if (
                    landed is None
                    and training_runs.get_run_by_uuid(self._hub, job_uuid) is None
                ):
                    # Deleted while it trained, or gone with the database:
                    # nothing owns the LoRA now, so its files stay put.
                    logger.warning(
                        "Training run %s (id %s) finished, but its row is gone; "
                        "%s is left in place and not imported.",
                        job_uuid,
                        info.get("id"),
                        progress.final_checkpoint,
                    )
                    return
                if landed is None or info.get("continued_from_step") is None:
                    # A first run on the shelf goes through the import all the
                    # same: it answers from the shelf, and attaches the LoRA to
                    # the run's set or character, which a refusal or a restart
                    # may have come before. A run trained longer attaches
                    # nothing, so it has nothing left to do.
                    # The final checkpoint sits in the run folder, which is the
                    # ai-toolkit run the importer reads (``save_root``).
                    run_dir = (
                        os.path.dirname(progress.final_checkpoint)
                        if progress.final_checkpoint
                        else handle.save_root
                    )
                    landed = self._import_when_admitted(
                        backend, handle, info, progress, run_dir, defer_wait
                    )
                    if landed is None:
                        return
                self._write_completed(job_uuid, info, progress, landed)
                return
            except Exception:
                logger.exception(
                    "Job %s reached 'completed' but importing the run to the shelf failed",
                    job_uuid,
                )
                self._update_job(
                    job_uuid,
                    status="error",
                    error="Run completed but the LoRA artifact could not be imported",
                    completed_at=now_iso(),
                )
                self._emit_progress(
                    job_uuid, status="error", message="LoRA import failed"
                )
                return

        # stopped / error / anything else terminal.
        final_status = "stopped" if status == "stopped" else "error"
        # The card shows the error; the trainer's whole text stays in the report.
        error_msg = (
            None
            if final_status == "stopped"
            else live_view.failure_summary(progress.info, progress.step)
        )
        self._update_job(
            job_uuid,
            status=final_status,
            step=progress.step,
            error=error_msg,
            report=progress.info or None,
            completed_at=now_iso(),
        )
        self._emit_progress(
            job_uuid,
            status=final_status,
            current=progress.step,
            total=progress.total_steps,
            message=error_msg or progress.info or final_status,
        )
        logger.info("Job %s finalized as %s.", job_uuid, final_status)

    def _import_when_admitted(
        self,
        backend: TrainerBackend,
        handle,
        info: dict,
        progress: TrainingProgress,
        run_dir: str,
        defer_wait: bool,
    ) -> dict | None:
        """Import the run once its library admits it.

        Args:
            backend: As :meth:`_finalize`.
            handle: As :meth:`_finalize`.
            info: As :meth:`_finalize`.
            progress: As :meth:`_finalize`.
            run_dir: The ai-toolkit run folder the importer reads.
            defer_wait: As :meth:`_finalize`.

        Returns:
            What the importer answers (``stack_id``, ``imported``). None when
            the import is left to a later finalize: refused until the library
            opens again, stopped while it waited, or handed to a thread of its
            own (*defer_wait*). The row is left as it is for that finalize.
        """
        job_uuid = info["uuid"]
        while True:
            try:
                return self._import_run(run_dir, info)
            except LibraryNotReadyError as exc:
                if exc.state is not SwitchState.SWITCHING:
                    # UNAVAILABLE stays so until a restart, and READY here
                    # means another vault is the active library. Either way
                    # the row stays, and the reconcile the next time this
                    # library opens imports the run.
                    logger.warning(
                        "Job %s: the import of %s is refused (%s); "
                        "leaving it for reconcile_on_startup.",
                        job_uuid,
                        run_dir,
                        exc,
                    )
                    return None
                if defer_wait:
                    logger.info(
                        "Job %s: the import of %s waits for the library (%s) on "
                        "a thread of its own, so the library's start goes on.",
                        job_uuid,
                        run_dir,
                        exc,
                    )
                    threading.Thread(
                        target=self._finalize,
                        args=(backend, handle, info, progress),
                        name=f"TrainingImport-{job_uuid[:8]}",
                        daemon=True,
                    ).start()
                    return None
                # A restore or a library switch holds the library. The import
                # waits for it rather than fail the run.
                logger.info(
                    "Job %s: the import of %s waits for the library (%s); "
                    "asking again in %.0f s.",
                    job_uuid,
                    run_dir,
                    exc,
                    _LIBRARY_NOT_READY_RETRY_S,
                )
                if self._stop.wait(_LIBRARY_NOT_READY_RETRY_S):
                    # As at shutdown: the row stays, and the next
                    # reconcile_on_startup imports the run.
                    logger.info(
                        "Job %s: stopping before its import was admitted; "
                        "leaving it for reconcile_on_startup.",
                        job_uuid,
                    )
                    return None

    def _write_completed(
        self, job_uuid: str, info: dict, progress: TrainingProgress, landed: dict
    ) -> None:
        """Write and announce ``completed`` for a run whose LoRA is on the shelf.

        Args:
            job_uuid: The run's uuid.
            info: Persisted run fields (needs ``name``).
            progress: The terminal progress snapshot.
            landed: What is on the shelf (``imported``, ``stack_id``).
        """
        report = (
            f"Trained {info['name']} to step {progress.step}; "
            f"{landed['imported']} file(s) on the shelf "
            f"(stack {landed['stack_id']})."
        )
        self._update_job(
            job_uuid,
            status="completed",
            step=progress.step,
            report=report,
            completed_at=now_iso(),
        )
        self._emit_progress(
            job_uuid,
            status="completed",
            current=progress.step,
            total=progress.total_steps,
            message="Completed",
        )
        logger.info("Job %s completed: %s", job_uuid, report)

    def _landed_on_shelf(self, info: dict, final_checkpoint: str | None) -> dict | None:
        """What an earlier import of this run's final put on the shelf, or None.

        The import stamps every model it registers with the run's id, and the
        shelf keys a model by ``sha256`` (UNIQUE), so the final is on the shelf
        when a model of the run has its hash. When the run folder no longer
        holds the final (its source folder deletes after import), the shelf's
        bare final says so instead: the importer registers it with no
        ``training_step``, and :meth:`train_longer` stamps it with its step
        before a run trains on, so a run trained longer is matched by hash
        only. A first finalize finds no model of the run and hashes nothing.

        Args:
            info: Persisted run fields (needs ``id``, ``uuid`` and
                ``continued_from_step``).
            final_checkpoint: The final checkpoint in the run folder, or None
                when the run folder has none.

        Returns:
            ``{"stack_id", "imported"}``, as the importer answers them, with
            ``imported`` counting every model of the run on the shelf; None
            when the final has not landed.
        """
        rows = self._hub.fetchall(
            "SELECT sha256, stack_id, training_step FROM model "
            "WHERE training_run_id = ?",
            (info["id"],),
        )
        if not rows:
            return None
        if final_checkpoint is not None:
            digest = file_digest(final_checkpoint)
            final = next((r for r in rows if r["sha256"] == digest), None)
        elif info.get("continued_from_step") is None:
            final = next((r for r in rows if r["training_step"] is None), None)
        else:
            final = None
        if final is None:
            return None
        logger.info(
            "Job %s: its final is already on the shelf (stack %s); not imported again.",
            info["uuid"],
            final["stack_id"],
        )
        return {"stack_id": final["stack_id"], "imported": len(rows)}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lease_need(info: dict) -> tuple[int, int] | None:
        """``(vram_mb, ram_mb)`` of the run's resolved profile, or None.

        None for a row without one (queued before recipes existed).
        """
        try:
            resolution = json.loads(info.get("resolved_profile") or "null")
            return int(resolution["vram_mb"]), int(resolution["ram_mb"])
        except (TypeError, KeyError, ValueError) as exc:
            logger.warning(
                "Job %s has no readable memory profile: %s", info.get("uuid"), exc
            )
            return None

    @staticmethod
    def _lease_label(info: dict) -> str:
        """The name a job's lease carries in the Tasks tab."""
        return f"LoRA training '{info.get('name') or info['uuid'][:8]}'"

    def _store(self) -> tuple[HubDatabase, str]:
        """The hub and library the runs live in, or raise if none is open.

        Raises:
            RuntimeError: The service was built without a hub or library.
        """
        if self._hub is None or not self._library_uuid:
            raise RuntimeError(
                "No hub or library is open, so training runs cannot be stored."
            )
        return self._hub, self._library_uuid

    def _job_status(self, job_uuid: str) -> str | None:
        """A job's persisted status, or None when it is gone."""
        return training_runs.get_run_status(self._hub, job_uuid)

    def _update_job(
        self,
        job_uuid: str,
        *,
        only_if: Mapping[str, Sequence[object]] | None = None,
        attempts: int | None = None,
        **fields,
    ) -> bool:
        """Update columns on a training run by uuid.

        A write that ends the run (``status`` completed, stopped or error) is
        retried when the hub is busy or locked: lost, it would leave the run
        ``running`` or ``queued`` in the list until the next start-up's
        :meth:`reconcile_on_startup`. Any other write is tried once unless
        *attempts* says otherwise. Only the progress columns
        :meth:`_persist_progress` writes are written again by the next poll;
        any other write tried once stays lost when it fails, so a write that
        must land (the launch's) passes *attempts*.

        Args:
            job_uuid: The job uuid.
            only_if: Write only while each named column holds one of its
                values (compare-and-set: see ``training_runs.update_run``);
                None writes whatever the row holds.
            attempts: Tries in all on a busy or locked hub. None:
                :data:`_TERMINAL_WRITE_ATTEMPTS` for an end state, else 1.
            **fields: Column names → values to set.

        Returns:
            True when the row was written. False, after logging why, when no
            run has that uuid, *only_if* did not hold, or the write failed.
        """
        terminal = fields.get("status") in _TERMINAL_JOB_STATUSES
        if attempts is None:
            attempts = _TERMINAL_WRITE_ATTEMPTS if terminal else 1
        # Passed only when given, so an unconditional write calls the store as
        # it always has.
        condition = {} if only_if is None else {"only_if": only_if}
        for attempt in range(1, attempts + 1):
            try:
                if training_runs.update_run(self._hub, job_uuid, **condition, **fields):
                    return True
                if only_if is None:
                    logger.warning("_update_job: no job with uuid=%s", job_uuid)
                else:
                    logger.info(
                        "Job %s: %s not written, as the run no longer holds %s "
                        "(or is gone).",
                        job_uuid,
                        list(fields),
                        dict(only_if),
                    )
                return False
            except sqlite3.OperationalError as exc:
                if attempt == attempts:
                    if terminal:
                        logger.error(
                            "Job %s: could not write its end state %r after %d "
                            "tries (%s); it stays at its last status until the "
                            "next start-up reconciles it.",
                            job_uuid,
                            fields["status"],
                            attempts,
                            exc,
                            exc_info=True,
                        )
                    elif attempts > 1:
                        logger.error(
                            "Job %s: could not write %s after %d tries (%s).",
                            job_uuid,
                            list(fields),
                            attempts,
                            exc,
                            exc_info=True,
                        )
                    else:
                        logger.exception(
                            "Failed to update job %s with %s", job_uuid, list(fields)
                        )
                    return False
                # Full jitter: a random wait up to the capped exponential
                # step, so writers that failed together do not retry together.
                delay = random.uniform(
                    0.0,
                    min(
                        _TERMINAL_WRITE_BACKOFF_CAP_S,
                        _TERMINAL_WRITE_BACKOFF_S * 2 ** (attempt - 1),
                    ),
                )
                logger.warning(
                    "Job %s: writing %s failed (try %d of %d: %s); "
                    "trying again in %.2f s.",
                    job_uuid,
                    f"its end state {fields['status']!r}" if terminal else list(fields),
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except Exception:
                # Not a busy hub (an unknown column, a constraint): another try
                # would fail the same way.
                logger.exception(
                    "Failed to update job %s with %s", job_uuid, list(fields)
                )
                return False
        return False

    def _persist_progress(
        self, job_uuid: str, progress: TrainingProgress, handle
    ) -> None:
        """Persist the live progress fields onto the job row.

        Only the columns that changed since this run's last write are written,
        and nothing when none did and the row already holds the status. A loss
        curve the backend could not read leaves the stored one as it is.

        Args:
            job_uuid: The job uuid.
            progress: The latest progress snapshot.
            handle: The job handle (for reading the log tail).
        """
        status = progress.status if progress.status != "unknown" else "running"
        values = {
            "step": progress.step,
            "total_steps": progress.total_steps,
            "sample_image_paths": progress.samples or None,
            "log_tail": self._read_log_tail(handle.log_path),
        }
        # A curve of None is one the backend could not read this poll, so the
        # stored curve stays; an empty one was read and has no points. Every
        # loss in it is finite: the backend leaves out a step that is not.
        if progress.loss_curve is not None:
            values["loss_curve"] = progress.loss_curve or None
        # The trainer reports running for a while after a cancel; the row's
        # "stopping" stands until the trainer reaches an end state. The read and
        # the write sit under the lock cancel writes "stopping" under.
        with self._active_lock:
            row_status = self._job_status(job_uuid)
            if status == "running" and row_status == "stopping":
                status = "stopping"
            # Lists compare with ``==``, which is cheaper than encoding them.
            # A backend never changes a list it has returned, so what the
            # snapshot holds is what was written.
            written = self._persisted.get(job_uuid, {})
            changed = {
                column: value
                for column, value in values.items()
                if column not in written or written[column] != value
            }
            if changed or status != row_status:
                fields = dict(changed)
                for column in ("loss_curve", "sample_image_paths"):
                    if fields.get(column) is not None:
                        fields[column] = json.dumps(fields[column])
                # A failed write leaves the snapshot as it was, so the next
                # poll writes these columns again. Tried once even for an end
                # state: a retry would hold the lock a cancel waits on, and
                # _finalize writes the end state again, with its retries,
                # outside it.
                if self._update_job(job_uuid, status=status, attempts=1, **fields):
                    self._persisted[job_uuid] = {**written, **changed}
        # A completed trainer is imported next: `_finalize` writes completed_at.
        self._note_phase(
            job_uuid, live_view.phase_of(status, progress.info, progress.step, None)
        )

    def _note_phase(self, job_uuid: str, phase: str | None) -> None:
        """Append *phase* to the run's phase log when it is a new one.

        Args:
            job_uuid: The job uuid.
            phase: One of :data:`pixlstash.training.live_view.PHASES`, or None
                (nothing is written).
        """
        if phase is None:
            return
        with self._phase_lock:
            log = self._phase_logs.get(job_uuid)
            if log is None:
                run = training_runs.get_run_by_uuid(self._hub, job_uuid)
                log = live_view.parse_phase_log(run.get("phase_log") if run else None)
                self._phase_logs[job_uuid] = log
            if not live_view.append_phase(log, phase, now_iso()):
                return
            # Written under the lock, so two threads cannot store their
            # snapshots out of order.
            self._update_job(job_uuid, phase_log=json.dumps(log))

    def _emit_progress(
        self,
        run_id: str,
        *,
        status: str,
        current: int = 0,
        total: int | None = None,
        message: str = "",
    ) -> None:
        """Emit a PLUGIN_PROGRESS event in the ComfyUI-compatible shape.

        Args:
            run_id: The job uuid (the event ``run_id``).
            status: Lifecycle status string.
            current: Current step.
            total: Total steps (``None`` → treated as 0 for progress).
            message: Human-readable info string.
        """
        total_val = int(total or 0)
        progress_frac = (current / total_val) if total_val > 0 else 0.0
        try:
            self._notify(
                EventType.PLUGIN_PROGRESS,
                {
                    "plugin": "ai-toolkit",
                    "status": status,
                    "run_id": run_id,
                    "message": str(message or ""),
                    "current": int(current or 0),
                    "total": total_val,
                    "progress": progress_frac,
                },
            )
        except Exception as exc:
            logger.debug("Failed to emit training progress event: %s", exc)

    # ------------------------------------------------------------------
    # Dataset / backend / handle helpers
    # ------------------------------------------------------------------

    def _run_folders(self, job_uuid: str) -> tuple[str, str]:
        """Resolve where a job's run and its dataset go.

        The run goes in the model shelf's ``source`` folder. When the shelf has
        none, ``<repo>/output`` is registered as one, the first time a job
        needs it. The dataset goes in ``<repo>/datasets/pixlstash-<uuid>``.

        Args:
            job_uuid: The job's uuid.

        Returns:
            ``(training_folder, dataset_dir)``.

        Raises:
            RuntimeError: ai-toolkit is not configured, or no shelf folder could
                be found or registered. The message names no host path; the
                detail is logged.
        """
        cfg = self._config_provider() or {}
        repo_path = cfg.get("repo_path")
        if not repo_path:
            raise RuntimeError("ai-toolkit is not configured; cannot run training.")

        training_folder = self.ensure_source_folder(repo_path)
        dataset_dir = os.path.join(repo_path, "datasets", f"pixlstash-{job_uuid}")
        return training_folder, dataset_dir

    def ensure_source_folder(self, repo_path: str) -> str:
        """The shelf's ``source`` folder, registering ``<repo>/output`` if none.

        A fresh checkout may not have ``output`` yet, and the registration
        refuses a folder that does not exist, so it is created first.

        Args:
            repo_path: The ai-toolkit checkout.

        Returns:
            The stored path of the shelf's first ``source`` folder.

        Raises:
            RuntimeError: No shelf folder could be found or registered. The
                message names no host path; the detail is logged.
        """
        training_folder = self._shelf_source_folder()
        if training_folder is not None:
            return training_folder
        if self._register_source_folder is None:
            raise RuntimeError("No model shelf folder is set up for ai-toolkit runs.")
        output = os.path.join(repo_path, "output")
        try:
            os.makedirs(output, exist_ok=True)
            self._register_source_folder(output)
        except Exception as exc:
            # A registration that lost a race to another one is fine: the
            # folder is on the shelf either way, and is read back below.
            training_folder = self._shelf_source_folder()
            if training_folder is not None:
                return training_folder
            logger.exception(
                "Could not register %s as the shelf's ai-toolkit folder: %s",
                output,
                exc,
            )
            raise RuntimeError(
                "Could not add the ai-toolkit output folder to the model shelf."
            ) from exc
        training_folder = self._shelf_source_folder()
        if training_folder is None:
            raise RuntimeError("No model shelf folder is set up for ai-toolkit runs.")
        return training_folder

    def _shelf_source_folder(self) -> str | None:
        """The path of the first registered shelf ``source`` folder, or None.

        Returns:
            The folder's stored path, or None when none is registered.
        """
        row = self._hub.fetchone(
            "SELECT path FROM model_folder WHERE kind = ? ORDER BY id LIMIT 1",
            (SOURCE_FOLDER_KIND,),
        )
        return row["path"] if row is not None else None

    def _build_dataset(self, info: dict, dataset_dir: str) -> dict | None:
        """Build the dataset snapshot for a job into *dataset_dir*.

        Args:
            info: Queued-job fields (set_id / character_id / trigger_word).
            dataset_dir: The directory the image/caption pairs are written to.

        Returns:
            The dataset manifest, or ``None`` if the build raised (the job is
            marked ``error`` in that case).
        """
        # A run rebuilt after a restart starts from an empty folder: the
        # trainer reads every file in it, so a leftover from the first build
        # would train without being in the snapshot. The folder is this run's
        # own (``pixlstash-<uuid>``, see _run_folders).
        self._remove_tree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)
        # The clips go in when the job's own locked config reads video: the
        # config it renders, not a recipe looked up again, which may be gone.
        pending = self._load_pending_spec(info["uuid"])
        clips = presets.reads_video(pending.get("locked") or {})
        try:
            manifest = self._dataset_builder(
                dest_dir=dataset_dir,
                set_id=info.get("set_id"),
                character_id=info.get("character_id"),
                trigger_word=info.get("trigger_word"),
                clips=clips,
                caption_mode=pending.get("caption_mode"),
            )
        except Exception:
            logger.exception("Job %s: dataset build failed", info["uuid"])
            self._fail(info["uuid"], "Dataset build failed", log=False)
            return None
        # Enqueue refused a motion run with no clips, but they can be removed
        # while it waits; it must not then train its motion on stills alone.
        items = manifest.get("items") or []
        if clips and not any(VideoUtils.is_video_file(i["file_name"]) for i in items):
            msg = "The video clips this run learns motion from are gone."
            logger.warning(
                "Job %s: %s (%d stills built)", info["uuid"], msg, len(items)
            )
            self._fail(info["uuid"], msg, log=False)
            return None
        return manifest

    def _record_snapshot(self, info: dict, manifest: dict) -> bool:
        """Freeze what the run is about to train on (#2123).

        A run whose lineage cannot be recorded does not train: the LoRA it
        made would name a dataset nobody can show.

        Args:
            info: Queued-job fields.
            manifest: The dataset builder's manifest, with its ``items``.

        Returns:
            True when the snapshot is stored; False when it failed and the job
            was marked ``error``.
        """
        hub, library_uuid = self._store()
        try:
            dataset_snapshots.record_snapshot(
                hub,
                library_uuid=library_uuid,
                run_uuid=info["uuid"],
                set_id=info.get("set_id"),
                character_id=info.get("character_id"),
                trigger_word=manifest.get("trigger_word"),
                caption_mode=manifest.get("caption_mode"),
                items=manifest.get("items") or [],
            )
            return True
        except Exception:
            logger.exception(
                "Job %s: the dataset snapshot of %s could not be recorded",
                info["uuid"],
                manifest.get("dest_dir"),
            )
            self._fail(
                info["uuid"], "Dataset snapshot could not be recorded", log=False
            )
            return False

    def _make_backend(self) -> TrainerBackend:
        """Construct a trainer backend from the owner's ai-toolkit config.

        Returns:
            A :class:`TrainerBackend` from the injected factory.

        Raises:
            ValueError: If the owner config is missing the repo path / python.
        """
        cfg = self._config_provider() or {}
        repo_path = cfg.get("repo_path")
        python = cfg.get("python")
        if not repo_path or not python:
            raise ValueError(
                "ai-toolkit is not configured (missing repo_path/python in owner "
                "config); cannot run training."
            )
        return self._backend_factory(repo_path, python)

    def _handle_for_job(self, backend: TrainerBackend, info: dict):
        """Reconstruct a :class:`TrainingHandle` from persisted job fields.

        See :func:`handle_for_run`; *backend*'s ``handle_for`` is used when it
        has one.
        """
        return handle_for_run(self._training_root, info, backend)

    # ------------------------------------------------------------------
    # Pending-spec sidecar (template_path / config_overrides without columns)
    # ------------------------------------------------------------------

    def _pending_sidecar_path(self, job_uuid: str) -> str:
        """Path to the per-job pending-spec sidecar JSON."""
        return os.path.join(self._training_root, job_uuid, "pending_spec.json")

    def _write_pending_sidecar(self, job_uuid: str, pending: dict) -> None:
        """Persist the pending spec so a restart can recover a queued job.

        Args:
            job_uuid: The job uuid.
            pending: ``{"template_path", "config_overrides"}``.
        """
        path = self._pending_sidecar_path(job_uuid)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(pending, fh)
        except OSError as exc:
            logger.warning(
                "Could not write pending-spec sidecar for job %s: %s", job_uuid, exc
            )

    def _load_pending_spec(self, job_uuid: str) -> dict:
        """Load the pending spec from memory, falling back to the sidecar.

        Args:
            job_uuid: The job uuid.

        Returns:
            ``{"template_path", "config_overrides"}`` (possibly empty values).
        """
        with self._pending_specs_lock:
            pending = self._pending_specs.get(job_uuid)
        if pending is not None:
            return pending
        path = self._pending_sidecar_path(job_uuid)
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "No pending spec for job %s (memory or sidecar %s): %s",
                job_uuid,
                path,
                exc,
            )
            return {}

    def _remove_pending_sidecar(self, job_uuid: str) -> None:
        """Delete the pending-spec sidecar once the job is done with it."""
        path = self._pending_sidecar_path(job_uuid)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.debug(
                "Could not remove pending-spec sidecar for job %s: %s", job_uuid, exc
            )

    # ------------------------------------------------------------------
    # Small static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(raw: str | None) -> dict | None:
        """A JSON column as a dict, or None when it does not parse."""
        try:
            value = json.loads(raw) if raw else None
        except ValueError as exc:
            logger.warning("Unreadable JSON column: %s", exc)
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _read_rendered_config(config_path: str) -> dict | None:
        """Read the rendered job config JSON, or ``None`` if unreadable."""
        try:
            with open(config_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read rendered config %s: %s", config_path, exc)
            return None

    @staticmethod
    def _total_steps_from_config(config: dict | None) -> int | None:
        """Pull ``config.process[0].train.steps`` from a rendered config."""
        if not config:
            return None
        try:
            return int(config["config"]["process"][0]["train"]["steps"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _read_log_tail(log_path: str) -> str | None:
        """Read the last ``_LOG_TAIL_BYTES`` of the trainer log, or ``None``."""
        if not log_path or not os.path.exists(log_path):
            return None
        try:
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as fh:
                if size > _LOG_TAIL_BYTES:
                    fh.seek(size - _LOG_TAIL_BYTES)
                data = fh.read()
            return data.decode("utf-8", errors="replace")
        except OSError as exc:
            logger.debug("Could not read log tail %s: %s", log_path, exc)
            return None
