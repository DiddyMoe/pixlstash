"""Measure a background Scrapheap run: slice sizes, writer hold, probe waits, Stop.

Stands up a throwaway server (a temporary directory holds its library, so no
real library is opened), seeds N Scrapheap pictures with realistic child rows
(tags, tag predictions, a quality row, a face, likeness pairs) and a file each,
then works a restore run, an undo of that restore and a purge run through the
real :class:`~pixlstash.tasks.scrapheap_run_task.ScrapheapRunTask`, for each
slice maximum asked for. While a run works, a probe thread submits an
IMMEDIATE writer task every 50 ms and records how long it waited, which is
what an interactive write sees. Each purge is stopped once ``--stop-at`` of it
is done, to measure what Stop still does after it is pressed.

Per run it prints wall time and ms per picture, every slice's size, writer time
and wall time, the probe's p50/p99/max wait, and for the purge the Stop
overshoot in pictures and seconds. An undo slice is one History segment, so
its slices are the segments, timed one undo each; ``--segment-cap`` sets
``SEGMENT_PICTURE_CAP`` to compare segment sizes.

Usage:

    python scripts/bench_scrapheap_run.py --pictures 20000 --max 500,250
    python scripts/bench_scrapheap_run.py --pictures 50000 --segment-cap 1000
"""

import argparse
import json
import os
import random
import statistics
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlmodel import update

from pixlstash.database import DBPriority
from pixlstash.db_models import Picture
from pixlstash.server import Server
from pixlstash.services import scrapheap_run_service
from pixlstash.tasks import scrapheap_run_task

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)
TAGS_PER_PICTURE = 15
PREDICTIONS_PER_PICTURE = 20
LIKENESS_PER_PICTURE = 5
PROBE_INTERVAL_S = 0.05


def _fill_for(col_type: str):
    kind = (col_type or "").upper()
    if "INT" in kind or "BOOL" in kind:
        return 0
    if "REAL" in kind or "FLOA" in kind or "NUM" in kind:
        return 0.0
    if "DATE" in kind or "TIME" in kind:
        return datetime.now(timezone.utc).isoformat(" ")
    return "x"


def _required_columns(conn, table: str) -> dict:
    """A filler value for every NOT NULL column without a default."""
    rows = conn.execute(text(f"PRAGMA table_info({table})")).all()
    return {
        row[1]: _fill_for(row[2])
        for row in rows
        if row[3] and row[4] is None and not row[5]
    }


def _insert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    conn.execute(
        text(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES "
            f"({','.join(':' + column for column in columns)})"
        ),
        rows,
    )


def seed(vault, count: int, tag: str) -> list[int]:
    """Put *count* pictures, with child rows and a file each, in the Scrapheap."""
    folder = os.path.join(vault.image_root, tag)
    os.makedirs(folder, exist_ok=True)
    for index in range(count):
        with open(os.path.join(folder, f"p{index:06d}.png"), "wb") as fh:
            fh.write(PNG)

    def insert(session):
        conn = session.connection()
        deleted_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(" ")
        base = _required_columns(conn, "picture")
        _insert(
            conn,
            "picture",
            [
                {
                    **base,
                    "file_path": f"{tag}/p{index:06d}.png",
                    "pixel_sha": f"{tag}-{index:058x}",
                    "format": "PNG",
                    "width": 1,
                    "height": 1,
                    "deleted": 1,
                    "deleted_at": deleted_at,
                }
                for index in range(count)
            ],
        )
        ids = [
            row[0]
            for row in conn.execute(
                text("SELECT id FROM picture WHERE file_path LIKE :prefix ORDER BY id"),
                {"prefix": f"{tag}/%"},
            )
        ]
        _insert(
            conn,
            "tag",
            [
                {"picture_id": pid, "tag": f"tag{j}"}
                for pid in ids
                for j in range(TAGS_PER_PICTURE)
            ],
        )
        _insert(conn, "quality", [{"picture_id": pid, "sharpness": 0.5} for pid in ids])
        face = _required_columns(conn, "face")
        _insert(
            conn,
            "face",
            [
                {**face, "picture_id": pid, "frame_index": 0, "face_index": 0}
                for pid in ids
            ],
        )
        prediction = _required_columns(conn, "tag_prediction")
        _insert(
            conn,
            "tag_prediction",
            [
                {
                    **prediction,
                    "picture_id": pid,
                    **({"tag": f"pred{j}"} if "tag" in prediction else {}),
                }
                for pid in ids
                for j in range(PREDICTIONS_PER_PICTURE)
            ],
        )
        likeness = _required_columns(conn, "picturelikeness")
        rng = random.Random(7)
        pairs = set()
        for pid in ids:
            for _ in range(LIKENESS_PER_PICTURE):
                low, high = sorted((pid, rng.choice(ids)))
                if low != high:
                    pairs.add((low, high))
        _insert(
            conn,
            "picturelikeness",
            [
                {**likeness, "picture_id_a": low, "picture_id_b": high}
                for low, high in sorted(pairs)
            ],
        )
        session.commit()
        return ids

    return vault.db.run_task(insert)


def _rebin(vault, ids: list[int]) -> None:
    """Put the restored pictures back in the Scrapheap for the purge run."""

    def rebin(session):
        for start in range(0, len(ids), 500):
            session.exec(
                update(Picture)
                .where(Picture.id.in_(ids[start : start + 500]))
                .values(deleted=True, deleted_at=datetime.now(timezone.utc))
            )
        session.commit()

    vault.db.run_task(rebin)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return (
        ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else 0.0
    )


def _probe(vault, stop: threading.Event, waits: list[float]) -> None:
    """An interactive write every 50 ms; record how long each waited."""
    while not stop.is_set():
        started = time.perf_counter()
        vault.db.run_task(
            lambda session: session.execute(text("SELECT 1")).all(),
            priority=DBPriority.IMMEDIATE,
        )
        waits.append(time.perf_counter() - started)
        stop.wait(PROBE_INTERVAL_S)


class _SliceLog:
    """Every slice's size, writer time and wall time, via the run's own sizer."""

    def __init__(self) -> None:
        self.slices: list[tuple[int, float, float]] = []
        self._original = scrapheap_run_task._SliceSizer.record
        self._last = time.perf_counter()

    def __enter__(self):
        log = self
        original = self._original

        def record(sizer, pictures, writer_s):
            now = time.perf_counter()
            log.slices.append((pictures, writer_s, now - log._last))
            log._last = now
            original(sizer, pictures, writer_s)

        self._last = time.perf_counter()
        scrapheap_run_task._SliceSizer.record = record
        return self

    def __exit__(self, *exc_info):
        scrapheap_run_task._SliceSizer.record = self._original
        return False


def _stopper(
    vault, run_id: str, total: int, fraction: float, out: dict, done: threading.Event
) -> None:
    """Press Stop once *fraction* of the run is processed; note when and where."""
    target = int(total * fraction)
    while not done.is_set():
        view = scrapheap_run_service.get_run(vault, run_id)
        if view["processed"] >= target:
            out["processed_at_stop"] = view["processed"]
            out["stopped_at"] = time.perf_counter()
            scrapheap_run_service.cancel_run(vault, run_id)
            return
        done.wait(0.005)


class _UndoLog:
    """Every undo segment's size and wall time, via the run's own undo_slice."""

    def __init__(self, slices: list[tuple[int, float, float]]) -> None:
        self.slices = slices
        self._original = scrapheap_run_service.undo_slice

    def __enter__(self):
        log = self
        original = self._original

        def undo_slice(vault, run_id, batch_id):
            started = time.perf_counter()
            outcome = original(vault, run_id, batch_id)
            elapsed = time.perf_counter() - started
            # One writer task per segment: its wall time is its writer hold
            # plus the queue wait, which the probe shows is small.
            log.slices.append((len(outcome.get("removed", ())), elapsed, elapsed))
            return outcome

        scrapheap_run_service.undo_slice = undo_slice
        return self

    def __exit__(self, *exc_info):
        scrapheap_run_service.undo_slice = self._original
        return False


def work_run(
    vault, kind: str, ids: list[int], stop_at=None, target_run_id=None
) -> dict:
    """Start one run, work it on this thread and return the measurements."""
    captured = []
    vault.submit_task = lambda task: captured.append(task) or task.id
    if kind == scrapheap_run_service.KIND_UNDO:
        view, _ = scrapheap_run_service.start_undo_run(
            vault, target_run_id, actor="bench"
        )
    else:
        deleted_before = (
            datetime.now(timezone.utc)
            if kind == scrapheap_run_service.KIND_PURGE
            else None
        )
        view, _ = scrapheap_run_service.start_run(
            vault, kind, ids, actor="bench", deleted_before=deleted_before
        )
    task = captured[0]
    stop_probe, waits = threading.Event(), []
    probe = threading.Thread(
        target=_probe, args=(vault, stop_probe, waits), daemon=True
    )
    stop, done = {}, threading.Event()
    stopper = None
    if stop_at is not None:
        stopper = threading.Thread(
            target=_stopper,
            args=(vault, view["id"], len(ids), stop_at, stop, done),
            daemon=True,
        )
    probe.start()
    with _SliceLog() as log, _UndoLog(log.slices):
        if stopper is not None:
            stopper.start()
        started = time.perf_counter()
        task._run_task()
        ended = time.perf_counter()
    done.set()
    stop_probe.set()
    probe.join()
    if stopper is not None:
        stopper.join()
    final = scrapheap_run_service.get_run(vault, view["id"])
    result = {
        "id": final["id"],
        "kind": kind,
        "status": final["status"],
        "processed": final["processed"],
        "wall_s": ended - started,
        "ms_per_picture": (ended - started) / max(final["processed"], 1) * 1000,
        "segments": len(final["segment_batch_ids"]),
        "slices": [
            {"pictures": pictures, "writer_s": writer_s, "wall_s": wall_s}
            for pictures, writer_s, wall_s in log.slices
        ],
        "probe_ms": {
            "n": len(waits),
            "p50": _percentile(waits, 0.5) * 1000,
            "p99": _percentile(waits, 0.99) * 1000,
            "max": max(waits, default=0.0) * 1000,
        },
    }
    if "stopped_at" in stop:
        result["stop"] = {
            "processed_at_stop": stop["processed_at_stop"],
            "overshoot_pictures": final["processed"] - stop["processed_at_stop"],
            "overshoot_s": ended - stop["stopped_at"],
        }
    return result


def _print(label: str, result: dict) -> None:
    slices = result["slices"]
    writer = [entry["writer_s"] for entry in slices]
    walls = [entry["wall_s"] for entry in slices]
    sizes = [entry["pictures"] for entry in slices]
    probe = result["probe_ms"]
    print(
        f"{label} {result['kind']:7s} {result['status']:9s} processed={result['processed']} "
        f"wall={result['wall_s']:.1f}s ({result['ms_per_picture']:.2f} ms/picture) "
        f"segments={result['segments']} slices={len(slices)}",
        flush=True,
    )
    if slices:
        print(
            f"    slice size min/median/max={min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)} "
            f"first sizes={sizes[:6]}",
            flush=True,
        )
        print(
            f"    slice writer s mean={statistics.mean(writer):.3f} max={max(writer):.3f}; "
            f"slice wall s mean={statistics.mean(walls):.3f} max={max(walls):.3f}",
            flush=True,
        )
    print(
        f"    IMMEDIATE probe wait p50={probe['p50']:.0f} ms p99={probe['p99']:.0f} ms "
        f"max={probe['max']:.0f} ms (n={probe['n']})",
        flush=True,
    )
    if "stop" in result:
        stop = result["stop"]
        print(
            f"    Stop at {stop['processed_at_stop']}: {stop['overshoot_pictures']} more picture(s), "
            f"{stop['overshoot_s']:.2f} s until the run ended",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pictures", type=int, default=5000, help="Pictures per run.")
    parser.add_argument(
        "--max",
        default="500",
        help="Comma-separated SLICE_MAX_PICTURES values to compare.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=scrapheap_run_service.SLICE_BUDGET_S,
        help="SLICE_BUDGET_S, seconds of writer time a slice aims for.",
    )
    parser.add_argument(
        "--stop-at",
        type=float,
        default=0.5,
        help="Fraction of the purge after which to press Stop.",
    )
    parser.add_argument(
        "--segment-cap",
        type=int,
        default=scrapheap_run_service.SEGMENT_PICTURE_CAP,
        help="SEGMENT_PICTURE_CAP, the most pictures one restore segment holds.",
    )
    parser.add_argument("--json", help="Also write every measurement to this file.")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="pixlstash-bench-") as temp_dir:
        config = os.path.join(temp_dir, "server-config.json")
        with open(config, "w") as fh:
            json.dump(
                {
                    "host": "localhost",
                    "port": 9598,
                    "image_root": os.path.join(temp_dir, "images"),
                    "disable_background_workers": True,
                },
                fh,
            )
        with Server(config) as server:
            vault = server.vault
            # The run routes refuse without a task runner; this bench is one.
            vault.runs_background_work = lambda: True
            scrapheap_run_service.SLICE_BUDGET_S = args.budget
            scrapheap_run_service.SEGMENT_PICTURE_CAP = args.segment_cap
            for index, maximum in enumerate(
                int(value) for value in args.max.split(",")
            ):
                scrapheap_run_service.SLICE_MAX_PICTURES = maximum
                label = (
                    f"[max={maximum} budget={args.budget}s segment={args.segment_cap}]"
                )
                started = time.perf_counter()
                ids = seed(vault, args.pictures, f"bench{index}")
                print(
                    f"{label} seeded {len(ids)} in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                restored = work_run(vault, scrapheap_run_service.KIND_RESTORE, ids)
                _print(label, restored)
                undone = work_run(
                    vault,
                    scrapheap_run_service.KIND_UNDO,
                    ids,
                    target_run_id=restored["id"],
                )
                _print(label, undone)
                # The undo already put them back; this also covers a run that
                # did not finish, so the purge always sees the whole selection.
                _rebin(vault, ids)
                purged = work_run(
                    vault, scrapheap_run_service.KIND_PURGE, ids, stop_at=args.stop_at
                )
                _print(label, purged)
                results.append(
                    {
                        "max": maximum,
                        "budget_s": args.budget,
                        "segment_cap": args.segment_cap,
                        "runs": [restored, undone, purged],
                    }
                )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
