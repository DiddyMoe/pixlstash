"""Reads and writes for the ``training_run`` table.

A run is the hub's record of one ai-toolkit LoRA training: what it was trained
on, with what config, and how it ended. ``model.training_run_id`` points here.
Every read and write is scoped by ``library_uuid``, because the set and character
ids a run carries are vault ids and mean nothing in another library.

Rows come back as plain dicts with the column names, timestamps as the ISO
strings the hub stores. Callers that need JSON columns parse them.
"""

from __future__ import annotations

from typing import Optional

from pixlstash.hub.db import HubDatabase, now_iso

# The columns the worker may change after a run is created. Identity and
# ownership (``id``, ``uuid``, ``library_uuid``, ``created_at``) never change.
# ``resolved_profile`` changes once, when checking memory and speed settles
# the profile the run trains with (#2120).
_MUTABLE_COLUMNS = frozenset(
    {
        "status",
        "base_model",
        "trigger_word",
        "preset",
        "resolved_profile",
        "rendered_config",
        "dataset_snapshot_path",
        "dataset_picture_ids",
        "ai_toolkit_commit",
        "training_folder",
        "pid",
        "step",
        "total_steps",
        "loss_curve",
        "sample_image_paths",
        "log_tail",
        "report",
        "error",
        "started_at",
        "completed_at",
        "phase_log",
        "cover_step",
        "finished_unwatched",
        "comfyui_quit",
        "continued_from_step",
        "continued_at",
        "set_aside_take",
    }
)

# Statuses a run holds while the trainer may still be alive.
UNFINISHED_STATUSES = ("running", "stopping")

# Statuses a run ends in: its trainer is gone and nothing more will write to it.
FINISHED_STATUSES = ("completed", "stopped", "error")

# A completed run the worker has not finished with: it is still drawing its
# deferred previews or being imported, and ``_finalize`` writes completed_at.
_COMPLETED_BUT_BUSY = "(status = 'completed' AND completed_at IS NULL)"

# The columns a run listing carries: what the job list shows, plus what it
# derives a run's folder name and phase from. The large text columns (loss
# curve, log tail, rendered config, picture ids, report) stay out, so listing
# long runs does not read curves nobody shows. A field the list starts to show
# is added here. The list reads ``phase_log`` with ``.get``, so leaving it out
# would quietly drop a running run's phase instead of failing.
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "id",
    "uuid",
    "name",
    "status",
    "phase_log",
    "set_id",
    "character_id",
    "base_model",
    "trigger_word",
    "step",
    "total_steps",
    "created_at",
    "started_at",
    "completed_at",
    "training_folder",
    "error",
    "dataset_snapshot_id",
)


def create_run(
    hub: HubDatabase,
    *,
    library_uuid: str,
    uuid: str,
    name: str,
    set_id: Optional[int],
    character_id: Optional[int],
    base_model: Optional[str],
    trigger_word: Optional[str],
    preset: Optional[str] = None,
    resolved_profile: Optional[str] = None,
    advanced_pins: Optional[str] = None,
) -> int:
    """Insert a queued run and return its id.

    Args:
        hub: The open hub.
        library_uuid: The library the set or character belongs to.
        uuid: The run's stable external identifier.
        name: Human-readable name.
        set_id: Picture set trained on (a vault id), or None.
        character_id: Character trained on (a vault id), or None.
        base_model: Base checkpoint the LoRA trains against.
        trigger_word: Caption trigger word.
        preset: The recipe as ``id@version``.
        resolved_profile: The resolver's result, as JSON.
        advanced_pins: The owner's Advanced values, as JSON.

    Returns:
        The new ``training_run.id``.
    """
    with hub.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO training_run (uuid, library_uuid, name, status, set_id, "
            "character_id, base_model, trigger_word, preset, resolved_profile, "
            "advanced_pins, created_at) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uuid,
                library_uuid,
                name,
                set_id,
                character_id,
                base_model,
                trigger_word,
                preset,
                resolved_profile,
                advanced_pins,
                now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def get_run(hub: HubDatabase, library_uuid: str, run_id: int) -> Optional[dict]:
    """One run of the library, or None when it is absent or in another library."""
    row = hub.fetchone(
        "SELECT * FROM training_run WHERE id = ? AND library_uuid = ?",
        (run_id, library_uuid),
    )
    return dict(row) if row is not None else None


def get_run_by_uuid(hub: HubDatabase, run_uuid: str) -> Optional[dict]:
    """One run by its uuid, whichever library holds it, or None.

    For the worker, which already holds the uuid of a run it owns. Routes take
    the integer id through :func:`get_run`, which is library-scoped.
    """
    row = hub.fetchone("SELECT * FROM training_run WHERE uuid = ?", (run_uuid,))
    return dict(row) if row is not None else None


def get_run_status(hub: HubDatabase, run_uuid: str) -> Optional[str]:
    """One run's status by its uuid, whichever library holds it, or None.

    For the worker, which checks the status on every poll of a run and needs
    none of the run's other columns.
    """
    row = hub.fetchone("SELECT status FROM training_run WHERE uuid = ?", (run_uuid,))
    return str(row[0]) if row is not None else None


# The loss of a stored curve's last point, read in SQL so the status poll never
# loads the curve. NULL when there is no curve, it has no points, or it is not
# valid JSON. The last index is computed rather than written ``$[#-1]``: that
# form needs SQLite 3.31, and on a malformed curve it raises instead of
# returning NULL.
_LATEST_LOSS_SQL = (
    "CASE WHEN json_valid(loss_curve) AND json_array_length(loss_curve) > 0 "
    "THEN json_extract(loss_curve, "
    "'$[' || (json_array_length(loss_curve) - 1) || '].loss') END"
)


def get_run_status_summary(
    hub: HubDatabase, library_uuid: str, run_id: int
) -> Optional[dict]:
    """One run's status fields and its latest loss, without its curve.

    For the status poll. The row holds ``id``, ``uuid``, ``status``, ``step``,
    ``total_steps``, ``report`` and ``error``, plus ``latest_loss`` (the last
    curve point's loss, or None) and ``loss_curve_malformed`` (true when a
    stored curve is not valid JSON). None when the run is absent or in another
    library.
    """
    row = hub.fetchone(
        "SELECT id, uuid, status, step, total_steps, report, error, "
        f"{_LATEST_LOSS_SQL} AS latest_loss, "
        "(loss_curve <> '' AND NOT json_valid(loss_curve)) AS loss_curve_malformed "
        "FROM training_run WHERE id = ? AND library_uuid = ?",
        (run_id, library_uuid),
    )
    return dict(row) if row is not None else None


def list_runs(
    hub: HubDatabase,
    library_uuid: str,
    *,
    set_id: Optional[int] = None,
    character_id: Optional[int] = None,
) -> list[dict]:
    """The library's runs as summaries, newest first.

    Optionally only one set's or one character's runs. Each row holds only
    :data:`_SUMMARY_COLUMNS`, not the whole run: read one run with
    :func:`get_run` for its curve, log, config and report.
    """
    clauses = ["library_uuid = ?"]
    params: list = [library_uuid]
    if set_id is not None:
        clauses.append("set_id = ?")
        params.append(set_id)
    if character_id is not None:
        clauses.append("character_id = ?")
        params.append(character_id)
    rows = hub.fetchall(
        f"SELECT {', '.join(_SUMMARY_COLUMNS)} FROM training_run WHERE "
        + " AND ".join(clauses)
        + " ORDER BY created_at DESC, id DESC",
        tuple(params),
    )
    return [dict(row) for row in rows]


def next_queued_run(hub: HubDatabase, library_uuid: str) -> Optional[dict]:
    """The run queued longest ago (FIFO), or None.

    A run trained longer is queued when it was asked to train on, not when it
    was first made, so it waits behind the runs queued before it.
    """
    row = hub.fetchone(
        "SELECT * FROM training_run WHERE library_uuid = ? AND status = 'queued' "
        "ORDER BY COALESCE(continued_at, created_at), id LIMIT 1",
        (library_uuid,),
    )
    return dict(row) if row is not None else None


def unfinished_runs(hub: HubDatabase, library_uuid: str) -> list[dict]:
    """The library's runs left running or stopping, for the startup reconcile.

    A run that completed but was never imported (``completed_at`` unset: the
    restart came while its previews were drawn or it was imported) is one too,
    so the reconcile imports it.
    """
    marks = ", ".join("?" for _ in UNFINISHED_STATUSES)
    rows = hub.fetchall(
        "SELECT * FROM training_run WHERE library_uuid = ? "
        f"AND (status IN ({marks}) OR {_COMPLETED_BUT_BUSY}) "
        "ORDER BY created_at, id",
        (library_uuid, *UNFINISHED_STATUSES),
    )
    return [dict(row) for row in rows]


def busy_runs(hub: HubDatabase) -> list[dict]:
    """Every library's runs that may still write into their folder.

    Queued, running or stopping, or completed but still drawing its previews or
    being imported: the runs :func:`delete_run` with *only_finished* refuses.
    Not scoped by library, because the shelf's ai-toolkit folder is shared by
    every library, so a run another library is training writes into it too.
    """
    marks = ", ".join("?" for _ in FINISHED_STATUSES)
    rows = hub.fetchall(
        "SELECT * FROM training_run "
        f"WHERE status NOT IN ({marks}) OR {_COMPLETED_BUT_BUSY}",
        FINISHED_STATUSES,
    )
    return [dict(row) for row in rows]


def update_run(hub: HubDatabase, run_uuid: str, **fields) -> bool:
    """Set mutable columns on one run, by uuid.

    Args:
        hub: The open hub.
        run_uuid: The run's uuid.
        **fields: Column names to set. Names outside the mutable set raise.

    Returns:
        True when a row was updated, False when no run has that uuid.

    Raises:
        ValueError: A field is not a mutable column.
    """
    unknown = set(fields) - _MUTABLE_COLUMNS
    if unknown:
        raise ValueError(f"Not a mutable training_run column: {sorted(unknown)}")
    if not fields:
        return False
    assignments = ", ".join(f"{column} = ?" for column in fields)
    with hub.transaction() as conn:
        cursor = conn.execute(
            f"UPDATE training_run SET {assignments} WHERE uuid = ?",
            (*fields.values(), run_uuid),
        )
        return cursor.rowcount > 0


def delete_run(
    hub: HubDatabase, library_uuid: str, run_id: int, *, only_finished: bool = False
) -> Optional[str]:
    """Delete one run of the library. Returns its uuid, or None when absent.

    With *only_finished*, the status test is part of the DELETE itself, so a
    run that is queued, running or stopping (its trainer may still write into
    its folder), or completed but still drawing its previews or being imported,
    is left alone and None is returned. Checking the status in a
    separate read would leave a window in which the worker can start the run.

    Models the run put on the shelf are not touched: they keep their
    ``training_run_id``, which then names nothing, the same as after any other
    deleted run. The run's dataset snapshot goes with it.
    """
    with hub.transaction() as conn:
        row = conn.execute(
            "SELECT uuid, dataset_snapshot_id FROM training_run "
            "WHERE id = ? AND library_uuid = ?",
            (run_id, library_uuid),
        ).fetchone()
        if row is None:
            return None
        if only_finished:
            marks = ", ".join("?" for _ in FINISHED_STATUSES)
            deleted = conn.execute(
                f"DELETE FROM training_run WHERE id = ? AND status IN ({marks}) "
                f"AND NOT {_COMPLETED_BUT_BUSY}",
                (run_id, *FINISHED_STATUSES),
            )
            if not deleted.rowcount:
                return None
        else:
            conn.execute("DELETE FROM training_run WHERE id = ?", (run_id,))
        if row[1] is not None:
            conn.execute("DELETE FROM dataset_snapshot WHERE id = ?", (row[1],))
        return str(row[0])


__all__ = [
    "FINISHED_STATUSES",
    "UNFINISHED_STATUSES",
    "busy_runs",
    "create_run",
    "delete_run",
    "get_run",
    "get_run_by_uuid",
    "get_run_status",
    "get_run_status_summary",
    "list_runs",
    "next_queued_run",
    "unfinished_runs",
    "update_run",
]
