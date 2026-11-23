"""Service layer for vault snapshot creation, listing, and GFS-style retention.

The SnapshotService copies the live database with ``VACUUM INTO`` on a
read-only connection of its own, off the writer, prepares and compresses the
copy, publishes it with JSON manifest and hash sidecars, records a
``Snapshot`` row in the live DB, and prunes old snapshots according to the
GFS retention constants.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import func, text
from sqlmodel import Session, select

from pixlstash.database import (
    SQLITE_BUSY_TIMEOUT_S,
    DBPriority,
    _compute_picture_metadata_hashes,
)
from pixlstash.db_models import Character, Picture, PictureSet, Project
from pixlstash.db_models.snapshot import Snapshot
from pixlstash.pixl_logging import get_logger
from pixlstash.services.portable_identity import sanitize_vault_file
from pixlstash.services.restore.schema_upgrade import snapshot_engine
from pixlstash.utils.path_utils import resolve_path_within
from pixlstash.utils.snapshot_compression import (
    COMPRESSED_SUFFIX,
    compress_snapshot,
    snapshot_scratch_dir,
    write_file_atomically,
)
from pixlstash.utils.sqlite_uri import sqlite_file_uri

if TYPE_CHECKING:
    from pixlstash.vault import Vault

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GFS retention constants (v1 - not user-configurable yet)
# ---------------------------------------------------------------------------
GFS_KEEP_DAILY: int = 7
GFS_KEEP_WEEKLY: int = 4  # most-recent Sunday of each of the last 4 weeks
GFS_KEEP_MONTHLY: int = 12  # first-of-month snapshot for the last 12 months
# OPPORTUNISTIC snapshots accumulate from safety-snapshot-before-restore and
# from snapshot_if_due() - without a cap they grow without bound. MANUAL
# snapshots are intentionally not capped: they are user-curated archives.
GFS_KEEP_OPPORTUNISTIC: int = 5

# Minimum hours between opportunistic snapshots.
OPPORTUNISTIC_MIN_HOURS: float = 1.0

# SQLite virtual-machine instructions the capture runs between two looks at its
# cancellation. The probe on a 20 000-row database saw about five per row, so
# a cancel is seen within a few hundred rows' worth of copying.
_CAPTURE_PROGRESS_OPS = 1000

# How long ``cancel_inflight`` waits for cancelled snapshots to clean up.
_CANCEL_WAIT_S = 10.0

# How often a snapshot waiting to register behind a restore checks whether it
# has been cancelled.
_REGISTRATION_POLL_S = 0.1

# Pictures per call when the manifest hashes rows whose ``metadata_hash`` is
# NULL. The batched helper holds every picture it was asked for (embedding
# blobs included) until it returns, so the batch bounds that memory.
_NULL_HASH_BATCH = 500


class SnapshotCancelledError(RuntimeError):
    """Raised by ``create_snapshot`` when it was cancelled before it finished.

    Nothing of the cancelled snapshot is left: no archive, no sidecars, no
    scratch copy and no ``Snapshot`` row.
    """


class SnapshotCancellation:
    """The cancel flag one ``create_snapshot`` call checks while it works.

    ``cancel()`` may be called from any thread. The capture asks
    ``should_stop`` from SQLite's progress handler while it copies the
    database; the later steps call ``raise_if_cancelled`` between steps and
    between compression chunks.

    Attributes:
        _event: Set once the snapshot is to stop.
    """

    def __init__(self, event: Optional[threading.Event] = None) -> None:
        """Initialise the flag.

        Args:
            event: The event that is the flag. A task passes its own cancel
                event, so cancelling the task cancels its snapshot, even a
                cancel that arrives before the snapshot starts. A fresh, unset
                one when omitted.
        """
        self._event = event if event is not None else threading.Event()

    @property
    def cancelled(self) -> bool:
        """Whether ``cancel`` has been called or the event is otherwise set."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Ask the snapshot to stop at its next check."""
        self._event.set()

    def should_stop(self) -> bool:
        """Return True once the capture should abandon its copy.

        Called from SQLite's progress handler during ``VACUUM INTO``.

        Returns:
            Whether the snapshot has been cancelled.
        """
        return self.cancelled

    def raise_if_cancelled(self) -> None:
        """Raise if the snapshot has been cancelled.

        Raises:
            SnapshotCancelledError: If ``cancel`` has been called.
        """
        if self.cancelled:
            raise SnapshotCancelledError(
                "The snapshot was cancelled before it finished."
            )


class SnapshotService:
    """Creates and manages vault-wide SQLite snapshots (snapshots).

    Attributes:
        _vault: Back-reference to the owning Vault.
        _registration_lock: Held while a snapshot registers its row, and by a
            full restore from its safety snapshot until it has reinserted the
            snapshot index (``hold_registrations``). Reentrant, because the
            restore's own safety snapshot registers on the same thread.
        _inflight_cond: Guards ``_inflight`` and is notified when it shrinks.
        _inflight: The cancellation of every ``create_snapshot`` call still
            running, so ``cancel_inflight`` can stop them.
    """

    def __init__(self, vault: "Vault") -> None:
        """Initialise the service.

        Args:
            vault: The owning Vault instance.
        """
        self._vault = vault
        self._registration_lock = threading.RLock()
        self._inflight_cond = threading.Condition()
        self._inflight: set[SnapshotCancellation] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_snapshot(
        self,
        kind: str,
        label: Optional[str] = None,
        cancellation: Optional[SnapshotCancellation] = None,
    ) -> Snapshot:
        """Create a full SQLite snapshot of the live vault database.

        Only the ``Snapshot`` row insert runs on the writer; everything else
        runs on the calling thread while imports and edits carry on:

        1. ``_capture`` copies the live database into a scratch file with
           ``VACUUM INTO``, a single read transaction, so the copy is the
           database exactly as it was committed when the copy began.
        2. The likeness pipeline tables are emptied in the copy, the manifest
           and per-picture hash map are read from it, and
           ``sanitize_vault_file`` scrubs its identity data.
        3. The copy is compressed to
           ``<vault_root>/snapshots/YYYY/MM/DD/<uuid>.sqlite.zst`` and the
           archive and both JSON sidecars are each published by rename, so
           none is ever seen half-written.
        4. The row is inserted under the registration lock (see
           ``hold_registrations``), GFS retention is applied, and a
           ``SNAPSHOT_CREATED`` event is emitted.

        Nothing is left behind by a failure or a cancel: the scratch copy is
        always removed, and the archive and sidecars are removed again unless
        their row was committed.

        Args:
            kind: One of ``'DAILY'``, ``'WEEKLY'``, ``'MONTHLY'``,
                ``'MANUAL'``, or ``'OPPORTUNISTIC'``.
            label: Optional user label (only meaningful for MANUAL
                snapshots).
            cancellation: The flag that stops this snapshot; a task that runs
                it passes one so it can cancel it. ``cancel_inflight`` sets it
                too. A fresh one is used when omitted.

        Returns:
            The newly created ``Snapshot`` row.

        Raises:
            SnapshotCancelledError: If the snapshot was cancelled before its
                row was registered.
            DatabaseUnavailableError: If the vault database is closing or
                closed.
        """
        if cancellation is None:
            cancellation = SnapshotCancellation()
        vault_root = self._vault.image_root
        db = self._vault.db

        snapshot_uuid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        date_dir = now.strftime("%Y/%m/%d")
        rel_dir = os.path.join("snapshots", date_dir)
        abs_dir = os.path.join(vault_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        rel_snapshot = os.path.join(rel_dir, f"{snapshot_uuid}{COMPRESSED_SUFFIX}")
        abs_snapshot = os.path.join(vault_root, rel_snapshot)
        rel_manifest = os.path.join(rel_dir, f"{snapshot_uuid}.manifest.json")
        abs_manifest = os.path.join(vault_root, rel_manifest)
        # The per-picture hash map lives in its own sidecar (not the manifest)
        # so the snapshot-list endpoint - which parses every manifest for its
        # small resource counts - never pays to read a multi-MB hash blob.
        rel_hashes = os.path.join(rel_dir, f"{snapshot_uuid}.hashes.json")
        abs_hashes = os.path.join(vault_root, rel_hashes)

        logger.info(
            "SnapshotService: creating %s snapshot → %s",
            kind,
            rel_snapshot,
        )
        # Set on the writer the moment the row is committed: from then on the
        # files belong to the row and a later failure must not remove them.
        committed = threading.Event()

        # The scratch copy is the full database. The expensive GPU-regenerated
        # blobs (CLIP image/text embeddings, InsightFace face features) are
        # deliberately KEPT - zstd makes carrying them affordable (~3x
        # smaller) so a restore does not trigger a full re-embedding /
        # re-detection pass. Only the .zst is retained.
        tmp_dir = tempfile.mkdtemp(
            prefix="pixlstash_snapshot_", dir=snapshot_scratch_dir(vault_root)
        )
        try:
            with self._inflight_cond:
                self._inflight.add(cancellation)
            tmp_sqlite = os.path.join(tmp_dir, "snapshot.sqlite")
            self._capture(tmp_sqlite, cancellation)
            self._checkpoint_after_capture()
            cancellation.raise_if_cancelled()
            self._strip_pipeline_tables(tmp_sqlite)
            # Read before the identity scrub, as the live database was before:
            # a hash computed here digests what the live row held.
            manifest, picture_hashes = self._build_manifest(tmp_sqlite)
            manifest["snapshot_uuid"] = snapshot_uuid
            manifest["kind"] = kind
            manifest["created_at"] = now.isoformat()
            cancellation.raise_if_cancelled()
            sanitize_vault_file(tmp_sqlite)
            cancellation.raise_if_cancelled()
            byte_size = compress_snapshot(
                tmp_sqlite,
                abs_snapshot,
                check_cancelled=cancellation.raise_if_cancelled,
            )
            manifest_json = json.dumps(manifest, indent=2).encode("utf-8")
            write_file_atomically(abs_manifest, lambda fh: fh.write(manifest_json))
            # Hash-map sidecar (separate so the list endpoint's manifest reads
            # stay lean). compact JSON - no indent - to keep it small.
            hashes_json = json.dumps(picture_hashes, separators=(",", ":")).encode(
                "utf-8"
            )
            write_file_atomically(abs_hashes, lambda fh: fh.write(hashes_json))

            def _record(session):
                # Checked again here, on the writer: a cancel that lands while
                # this insert waits behind other writer work must still win.
                cancellation.raise_if_cancelled()
                cp = Snapshot(
                    kind=kind,
                    created_at=now,
                    relative_path=rel_snapshot,
                    manifest_relative_path=rel_manifest,
                    byte_size=byte_size,
                    picture_count=manifest.get("picture_count", 0),
                    schema_version=manifest.get("schema_version", ""),
                    label=label,
                    # Stamped because ``sanitize_vault_file`` already scrubbed
                    # the scratch copy this archive was compressed from. Marking
                    # it here is what makes NULL mean exactly one thing: a
                    # legacy archive written before the hub, which the
                    # background scrub (MissingSnapshotIdentityScrubFinder)
                    # still owes work to.
                    identity_scrubbed_at=now,
                )
                session.add(cp)
                session.commit()
                committed.set()
                session.refresh(cp)
                return cp

            self._acquire_registration(cancellation)
            try:
                cancellation.raise_if_cancelled()
                snapshot = db.run_task(_record, priority=DBPriority.IMMEDIATE)
            finally:
                self._registration_lock.release()
        except Exception as exc:
            if committed.is_set():
                logger.error(
                    "SnapshotService: %s snapshot %s failed after its row was "
                    "committed; its files are kept: %s",
                    kind,
                    rel_snapshot,
                    exc,
                )
            else:
                logger.warning(
                    "SnapshotService: %s snapshot %s not created; removing what "
                    "it wrote: %s",
                    kind,
                    rel_snapshot,
                    exc,
                )
                self._remove_unregistered_files(
                    (abs_snapshot, abs_manifest, abs_hashes)
                )
            raise
        finally:
            self._remove_scratch_dir(tmp_dir)
            with self._inflight_cond:
                self._inflight.discard(cancellation)
                self._inflight_cond.notify_all()

        # --- GFS retention ------------------------------------------------
        self._apply_gfs_retention(now)

        # --- Emit event ---------------------------------------------------
        try:
            from pixlstash.event_types import EventType

            self._vault.emit_event(
                EventType.SNAPSHOT_CREATED, {"id": snapshot.id, "kind": kind}
            )
        except Exception as exc:
            logger.warning("SnapshotService: failed to emit SNAPSHOT_CREATED: %s", exc)

        logger.info(
            "SnapshotService: snapshot %d created (%d bytes, %d pictures)",
            snapshot.id,
            snapshot.byte_size,
            snapshot.picture_count,
        )
        return snapshot

    def cancel_inflight(self, timeout: float = _CANCEL_WAIT_S) -> bool:
        """Cancel every snapshot being created and wait for them to end.

        Each stops at its next check (SQLite's progress handler while it
        copies, between steps and compression chunks after), removes what it
        wrote and raises ``SnapshotCancelledError``. ``Vault.stop`` calls this
        before it stops the workers and closes the database. A snapshot
        started after this call is not affected.

        Args:
            timeout: Seconds to wait for the cancelled snapshots to end.

        Returns:
            True if none is still running, False if the wait timed out.
        """
        with self._inflight_cond:
            for cancellation in self._inflight:
                cancellation.cancel()
            finished = self._inflight_cond.wait_for(
                lambda: not self._inflight, timeout=timeout
            )
            still_running = len(self._inflight)
        if not finished:
            logger.warning(
                "SnapshotService: %d cancelled snapshot(s) still running after "
                "%.1f s; each removes its files when it reaches its next check",
                still_running,
                timeout,
            )
        return finished

    @contextmanager
    def hold_registrations(self):
        """Hold off every snapshot's row registration for the block.

        A full restore holds this from its safety snapshot until it has put
        the snapshot index back into the restored database, so a snapshot
        that finishes meanwhile registers its row in the restored database
        rather than in the one the swap discards. Reentrant: the restore's own
        safety snapshot registers on the same thread. Capture never takes it,
        so the lock order stays registration lock, then engine lock.
        """
        with self._registration_lock:
            yield

    def list_snapshots(self) -> list[Snapshot]:
        """Return all snapshot rows ordered by creation time (newest first).

        Returns:
            List of Snapshot rows.
        """
        return self._vault.db.run_immediate_read_task(
            lambda session: session.exec(
                select(Snapshot).order_by(Snapshot.created_at.desc())
            ).all()
        )

    def get_snapshot(self, snapshot_id: int) -> Optional[Snapshot]:
        """Return the snapshot with the given ID, or None if not found.

        Args:
            snapshot_id: Primary key of the snapshot.

        Returns:
            The Snapshot row or None.
        """
        return self._vault.db.run_immediate_read_task(
            lambda session: session.get(Snapshot, snapshot_id)
        )

    def snapshots_containing(self, picture_ids) -> list[dict]:
        """Return snapshots whose manifest still references any of *picture_ids*.

        Used after a permanent purge to tell the user which snapshots continue
        to hold metadata (tags, description, captions, …) for the pictures they
        just deleted. The snapshot archives are not scrubbed, so deleting those
        snapshots is currently the way to erase that retained metadata.

        Discovery reads only the JSON manifests (which list each snapshot's
        ``picture_ids``) - no snapshot database is opened or decompressed.

        Args:
            picture_ids: Live picture IDs that were just purged.

        Returns:
            List of ``{"id", "kind", "label", "created_at", "matched_count"}``
            dicts, one per snapshot containing at least one of the IDs, ordered
            newest first.
        """
        wanted = {int(p) for p in picture_ids if p is not None}
        if not wanted:
            return []
        result: list[dict] = []
        for cp in self.list_snapshots():
            manifest = self.load_manifest(cp.id)
            manifest_ids = manifest.get("picture_ids") or []
            matched = wanted.intersection(manifest_ids)
            if matched:
                result.append(
                    {
                        "id": cp.id,
                        "kind": cp.kind,
                        "label": cp.label,
                        "created_at": cp.created_at.isoformat(),
                        "matched_count": len(matched),
                    }
                )
        return result

    def delete_snapshot(self, snapshot_id: int) -> bool:
        """Delete a snapshot row and its snapshot file from disk.

        Args:
            snapshot_id: Primary key of the snapshot to delete.

        Returns:
            True if the snapshot was found and deleted, False otherwise.
        """
        vault_root = self._vault.image_root

        def _delete(session):
            cp = session.get(Snapshot, snapshot_id)
            if cp is None:
                return False
            rel_paths = (
                cp.relative_path,
                cp.manifest_relative_path,
                self._hashes_relative_path(cp.manifest_relative_path),
            )
            session.delete(cp)
            session.commit()
            for rel_path in rel_paths:
                try:
                    path = resolve_path_within(vault_root, rel_path)
                except ValueError as exc:
                    logger.warning(
                        "SnapshotService: refusing to remove snapshot file "
                        "outside the vault root (snapshot id=%d, rel_path=%s): %s",
                        snapshot_id,
                        rel_path,
                        exc,
                    )
                    continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as exc:
                    logger.warning(
                        "SnapshotService: could not remove %s: %s", path, exc
                    )
            return True

        deleted = self._vault.db.run_task(_delete, priority=DBPriority.IMMEDIATE)
        if deleted:
            try:
                from pixlstash.event_types import EventType

                self._vault.emit_event(EventType.SNAPSHOT_DELETED, {"id": snapshot_id})
            except Exception as exc:
                logger.warning(
                    "SnapshotService: failed to emit SNAPSHOT_DELETED: %s", exc
                )
        return deleted

    def rename_snapshot(
        self, snapshot_id: int, label: Optional[str]
    ) -> Optional[Snapshot]:
        """Update the label of an existing snapshot.

        Args:
            snapshot_id: Primary key of the snapshot.
            label: New label string, or None to clear it.

        Returns:
            The updated Snapshot row, or None if not found.
        """

        def _rename(session):
            cp = session.get(Snapshot, snapshot_id)
            if cp is None:
                return None
            cp.label = label
            session.add(cp)
            session.commit()
            session.refresh(cp)
            return cp

        return self._vault.db.run_task(_rename, priority=DBPriority.IMMEDIATE)

    def load_manifest(self, snapshot_id: int) -> dict:
        """Load the JSON manifest sidecar for a snapshot.

        Args:
            snapshot_id: Primary key of the snapshot.

        Returns:
            Parsed manifest dict, or empty dict if not found / unreadable.
        """
        cp = self.get_snapshot(snapshot_id)
        if cp is None:
            return {}
        abs_manifest = os.path.join(self._vault.image_root, cp.manifest_relative_path)
        try:
            with open(abs_manifest, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning(
                "SnapshotService: could not read manifest for snapshot %d: %s",
                snapshot_id,
                exc,
            )
            return {}

    @staticmethod
    def _hashes_relative_path(manifest_relative_path: str) -> str:
        """Derive the hash-sidecar path from a snapshot's manifest path.

        ``<uuid>.manifest.json`` → ``<uuid>.hashes.json`` in the same dir.

        Args:
            manifest_relative_path: The snapshot's ``manifest_relative_path``.

        Returns:
            The relative path of the per-picture hash sidecar.
        """
        if manifest_relative_path.endswith(".manifest.json"):
            return manifest_relative_path[: -len(".manifest.json")] + ".hashes.json"
        return manifest_relative_path + ".hashes.json"

    def load_picture_hashes(self, snapshot_id: int) -> dict:
        """Load the per-picture ``{str(id): metadata_hash}`` sidecar.

        Returns an empty dict for legacy snapshots that predate the sidecar
        (the caller then falls back to reading hashes from the snapshot file).

        Args:
            snapshot_id: Primary key of the snapshot.

        Returns:
            Parsed hash map, or empty dict if not found / unreadable.
        """
        cp = self.get_snapshot(snapshot_id)
        if cp is None:
            return {}
        abs_hashes = os.path.join(
            self._vault.image_root,
            self._hashes_relative_path(cp.manifest_relative_path),
        )
        if not os.path.exists(abs_hashes):
            return {}
        try:
            with open(abs_hashes, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning(
                "SnapshotService: could not read hash sidecar for snapshot %d: %s",
                snapshot_id,
                exc,
            )
            return {}

    def get_live_schema_version(self) -> str:
        """Return the current alembic head revision of the live database.

        Returns:
            Schema version string, or empty string on failure.
        """
        try:
            return self._vault.db.run_immediate_read_task(
                lambda session: (
                    session.exec(
                        text("SELECT version_num FROM alembic_version LIMIT 1")
                    ).scalar()
                    or ""
                )
            )
        except Exception as exc:
            logger.warning(
                "SnapshotService: could not read live schema version: %s", exc
            )
            return ""

    def snapshot_if_due(self, reason: str = "opportunistic") -> Optional[Snapshot]:
        """Take an opportunistic snapshot if more than OPPORTUNISTIC_MIN_HOURS have passed.

        Only automatic snapshots (DAILY, WEEKLY, MONTHLY, OPPORTUNISTIC) are
        considered when checking timing - MANUAL snapshots are user-curated
        archives and should not suppress the opportunistic schedule.

        Args:
            reason: Short label for logging.

        Returns:
            A new Snapshot if one was created, or None if skipped.
        """
        auto_kinds = {"DAILY", "WEEKLY", "MONTHLY", "OPPORTUNISTIC"}
        auto_snapshots = [s for s in self.list_snapshots() if s.kind in auto_kinds]
        if auto_snapshots:
            last = auto_snapshots[0]
            last_dt = last.created_at
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if age_hours < OPPORTUNISTIC_MIN_HOURS:
                logger.debug(
                    "SnapshotService: opportunistic snapshot skipped "
                    "(last was %.1fh ago, minimum %.1fh)",
                    age_hours,
                    OPPORTUNISTIC_MIN_HOURS,
                )
                return None
        logger.info("SnapshotService: creating opportunistic snapshot (%s)", reason)
        return self.create_snapshot("OPPORTUNISTIC")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture(self, dst_sqlite: str, cancellation: SnapshotCancellation) -> None:
        """Copy the live database to *dst_sqlite* with ``VACUUM INTO``, off the writer.

        The copy is consistent although writes carry on: ``VACUUM INTO`` reads
        the source in one read transaction, and in WAL mode a reader sees the
        database as committed when that transaction began, whatever the writer
        commits meanwhile (https://sqlite.org/lang_vacuum.html,
        https://sqlite.org/wal.html). The online backup API is not used: it
        restarts whenever another connection writes the source
        (https://sqlite.org/backup.html), and the writer always is one.

        The read side of the engine lock is held throughout, so the restore
        file swap and ``VaultDatabase.close`` wait for the copy to end, and a
        closing or closed database refuses it. The connection is read-only,
        its own, and closed before the lock is released: SQLite keeps a
        handle on the ``VACUUM INTO`` destination until its connection
        closes, and Windows will not delete a file that is still open.

        A progress handler polls *cancellation*; once it says stop, SQLite
        aborts the statement (https://sqlite.org/c3ref/progress_handler.html).

        Args:
            dst_sqlite: Scratch path for the copy; must not exist yet.
            cancellation: The flag that abandons the copy.

        Raises:
            SnapshotCancelledError: If the copy was abandoned on a cancel.
            DatabaseUnavailableError: If the database no longer serves reads.
            sqlite3.Error: If the copy fails.
        """
        db = self._vault.db
        with db.engine_read_access():
            # isolation_level=None keeps the connection in autocommit: SQLite
            # refuses VACUUM inside an open transaction.
            conn = sqlite3.connect(
                sqlite_file_uri(db._db_path, read_only=True),
                uri=True,
                timeout=SQLITE_BUSY_TIMEOUT_S,
                isolation_level=None,
            )
            try:
                conn.set_progress_handler(
                    lambda: 1 if cancellation.should_stop() else 0,
                    _CAPTURE_PROGRESS_OPS,
                )
                conn.execute("VACUUM INTO ?", (dst_sqlite,))
            except sqlite3.OperationalError as exc:
                if cancellation.cancelled:
                    raise SnapshotCancelledError(
                        "The snapshot was cancelled while it copied the database."
                    ) from exc
                raise
            finally:
                conn.close()

    def _checkpoint_after_capture(self) -> None:
        """Let the WAL be reused from the start after the capture's long read.

        While the capture's read transaction was open, no checkpoint could
        move the WAL back to its start past that reader, so the WAL grew with
        every write made meanwhile (https://sqlite.org/wal.html). One PASSIVE
        checkpoint through the writer copies those frames into the database so
        the next writes reuse the file. PASSIVE never waits on readers.
        """
        busy, log_frames, checkpointed = self._vault.db.run_task(
            lambda session: session.exec(text("PRAGMA wal_checkpoint(PASSIVE)")).one(),
            priority=DBPriority.IMMEDIATE,
        )
        logger.debug(
            "SnapshotService: checkpoint after capture: busy=%s, %s of %s WAL "
            "frames checkpointed",
            busy,
            checkpointed,
            log_frames,
        )

    def _acquire_registration(self, cancellation: SnapshotCancellation) -> None:
        """Take the registration lock, giving up if the snapshot is cancelled.

        A restore holds the lock until it has reinserted the snapshot index,
        which can take a while; checking the cancellation while waiting lets
        ``cancel_inflight`` end a snapshot queued behind it.

        Raises:
            SnapshotCancelledError: If cancelled before the lock was taken.
        """
        while not self._registration_lock.acquire(timeout=_REGISTRATION_POLL_S):
            cancellation.raise_if_cancelled()

    @staticmethod
    def _remove_unregistered_files(paths) -> None:
        """Remove the archive and sidecars of a snapshot whose row never committed.

        Args:
            paths: Absolute paths; those not written yet are skipped.
        """
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "SnapshotService: could not remove %s, written for a "
                    "snapshot that was not registered; it stays on disk until "
                    "removed by hand: %s",
                    path,
                    exc,
                )

    @staticmethod
    def _remove_scratch_dir(tmp_dir: str) -> None:
        """Remove a snapshot's scratch directory and the database copy in it.

        Args:
            tmp_dir: The ``mkdtemp`` directory under the snapshot scratch dir.
        """
        try:
            shutil.rmtree(tmp_dir)
        except OSError as exc:
            logger.warning(
                "SnapshotService: could not remove the scratch dir %s; the "
                "database copy in it uses disk until removed by hand: %s",
                tmp_dir,
                exc,
            )

    # Tables whose entire contents are dropped from a snapshot. These are
    # all regenerable / pipeline-state tables - keeping the snapshot's
    # rows would either waste disk on values the workers will recompute
    # (``picturelikeness``) or stomp on the live pipeline's progress
    # tracking when the restore swaps the file in (``picturelikenessqueue``
    # / ``picturelikenessfrontier``).  The full-restore path captures the
    # live queue + frontier before the swap and reinserts a reconciled
    # set after.
    _STRIP_TABLES_DELETE_ALL: tuple[str, ...] = (
        "picturelikeness",
        "picturelikenessqueue",
        "picturelikenessfrontier",
    )

    @classmethod
    def _strip_pipeline_tables(cls, snapshot_sqlite: str) -> None:
        """Empty the likeness pipeline tables in the captured copy.

        Empties the ``picturelikeness`` / ``picturelikenessqueue`` /
        ``picturelikenessfrontier`` tables (live similarity progress that the
        restore path reconstructs from the live DB). The expensive per-picture
        blobs (embeddings, face features, scores) are intentionally retained.
        The freed pages are reclaimed by the ``VACUUM`` in
        ``sanitize_vault_file``, which runs next.

        The copy is of the live database at the head schema, so every table
        listed exists; a failure here is a real one and propagates, and the
        snapshot publishes nothing.

        Args:
            snapshot_sqlite: Path to the scratch copy (before compression).

        Raises:
            sqlite3.Error: If a table cannot be emptied.
        """
        conn = sqlite3.connect(snapshot_sqlite)
        try:
            with conn:
                for table_name in cls._STRIP_TABLES_DELETE_ALL:
                    conn.execute(f"DELETE FROM {table_name}")
        finally:
            conn.close()

    @staticmethod
    def _build_manifest(snapshot_sqlite: str) -> tuple[dict, dict]:
        """Build the manifest dict and the per-picture hash map from the copy.

        Read from the captured copy, not the live database, so the manifest
        describes exactly the rows in the archive.

        Args:
            snapshot_sqlite: Path to the scratch copy.

        Returns:
            ``(manifest, picture_hashes)``. The manifest holds resource counts
            and the id list (written to the ``.manifest.json``); the hash map
            is ``{str(picture_id): metadata_hash}`` (written to the separate
            ``.hashes.json`` sidecar so the list endpoint's manifest reads stay
            lean).
        """
        engine = snapshot_engine(snapshot_sqlite)
        try:
            with Session(engine) as session:
                return SnapshotService._read_manifest(session)
        finally:
            engine.dispose()

    @staticmethod
    def _read_manifest(session) -> tuple[dict, dict]:
        """Read the manifest and hash map through *session*; see ``_build_manifest``."""
        # Pull id + metadata_hash together. The hash map lets the interactive
        # restore preview and hash-compare read hashes from the sidecar and
        # never decompress the snapshot archive. NULL hashes (legacy rows the
        # after_flush hook hasn't stamped yet) are computed on the spot, in
        # batches, so the map is complete.
        hash_rows = session.exec(select(Picture.id, Picture.metadata_hash)).all()
        picture_ids: list[int] = [row[0] for row in hash_rows]
        null_ids = [pid for pid, h in hash_rows if h is None]
        computed: dict[int, str] = {}
        for start in range(0, len(null_ids), _NULL_HASH_BATCH):
            computed.update(
                _compute_picture_metadata_hashes(
                    session, null_ids[start : start + _NULL_HASH_BATCH]
                )
            )
        picture_hashes: dict[str, str] = {}
        for pid, h in hash_rows:
            if h is None:
                h = computed.get(pid)
            if h is not None:
                picture_hashes[str(pid)] = h
        picture_set_count = session.exec(select(func.count(PictureSet.id))).one()
        project_count = session.exec(select(func.count(Project.id))).one()
        character_count = session.exec(select(func.count(Character.id))).one()

        # schema_version from alembic_version table
        try:
            result = session.exec(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            )
            schema_version = result.scalar() or ""
        except Exception as exc:
            logger.warning(
                "SnapshotService: failed to read schema_version for manifest: %s",
                exc,
            )
            schema_version = ""

        manifest = {
            "picture_count": len(picture_ids),
            "picture_ids": picture_ids,
            "picture_set_count": picture_set_count,
            "project_count": project_count,
            "character_count": character_count,
            "schema_version": schema_version,
        }
        return manifest, picture_hashes

    def _apply_gfs_retention(self, now: datetime) -> None:
        """Prune snapshots beyond GFS retention limits.

        Keeps:
        - ``GFS_KEEP_DAILY`` most-recent DAILY snapshots.
        - ``GFS_KEEP_WEEKLY`` most-recent WEEKLY snapshots.
        - ``GFS_KEEP_MONTHLY`` most-recent MONTHLY snapshots.
        - ``GFS_KEEP_OPPORTUNISTIC`` most-recent OPPORTUNISTIC snapshots.
        - All MANUAL snapshots (user-curated; user must delete them manually).

        Rows are deleted and committed first and their files removed only
        after, so a failed commit leaves every row pointing at files that are
        still there. A file that then cannot be removed is an orphan the next
        prune does not see; that is logged.

        Args:
            now: Current UTC datetime (used only for logging).
        """

        def _prune(session):
            pruned = []
            for kind, keep in (
                ("DAILY", GFS_KEEP_DAILY),
                ("WEEKLY", GFS_KEEP_WEEKLY),
                ("MONTHLY", GFS_KEEP_MONTHLY),
                ("OPPORTUNISTIC", GFS_KEEP_OPPORTUNISTIC),
            ):
                rows = session.exec(
                    select(Snapshot)
                    .where(Snapshot.kind == kind)
                    .order_by(Snapshot.created_at.desc())
                ).all()
                for cp in rows[keep:]:
                    pruned.append(
                        (
                            cp.id,
                            kind,
                            (
                                cp.relative_path,
                                cp.manifest_relative_path,
                                self._hashes_relative_path(cp.manifest_relative_path),
                            ),
                        )
                    )
                    session.delete(cp)
            session.commit()
            for snapshot_id, _kind, rel_paths in pruned:
                for rel_path in rel_paths:
                    try:
                        abs_path = resolve_path_within(vault_root, rel_path)
                    except ValueError as exc:
                        logger.warning(
                            "SnapshotService: GFS prune refusing to remove "
                            "a file outside the vault root (snapshot id=%d, "
                            "rel_path=%s): %s",
                            snapshot_id,
                            rel_path,
                            exc,
                        )
                        continue
                    try:
                        if os.path.exists(abs_path):
                            os.remove(abs_path)
                    except OSError as exc:
                        logger.warning(
                            "SnapshotService: GFS prune could not remove %s of "
                            "snapshot id=%d, whose row is already deleted; it "
                            "stays on disk until removed by hand: %s",
                            abs_path,
                            snapshot_id,
                            exc,
                        )
            return pruned

        vault_root = self._vault.image_root
        pruned = self._vault.db.run_task(_prune, priority=DBPriority.IMMEDIATE)
        for snapshot_id, kind, _rel_paths in pruned:
            logger.info(
                "SnapshotService: GFS pruned %s snapshot id=%d",
                kind,
                snapshot_id,
            )
