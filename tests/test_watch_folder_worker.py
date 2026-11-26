import errno
import io
import itertools
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from pixlstash.server import Server
from pixlstash.db_models.import_folder import ImportFolder
from pixlstash.db_models.picture import Picture
from pixlstash.db_models.tag import Tag
from pixlstash.services import import_dedup_service
from pixlstash.tasks import missing_watch_folder_import_finder
from pixlstash.tasks import watch_folder_import_task
from pixlstash.tasks.watch_folder_import_task import WatchFolderImportTask
from pixlstash.utils.image_processing.image_utils import (
    ImageUtils,
    is_in_progress_file_name,
)
from pixlstash.utils import reference_folder_validator

API_PREFIX = "/api/v1"

# A folder is locked with `chmod 000`. Windows does not honour it and root reads
# through it, so the tests that need a folder the server cannot read skip there.
_CANNOT_LOCK_A_FOLDER = os.name == "nt" or os.geteuid() == 0
_LOCK_SKIP_REASON = "chmod 000 does not lock a folder on Windows or for root"

# Upper bound for "the watcher eventually imports the file" polls below. These
# loops break as soon as the expected pictures appear, so a generous bound costs
# a passing run nothing and only bounds the failing case. It has to be generous:
# CI runs the backend gate as 8 concurrent shards (plus 4 Windows), so a runner
# is far more contended than when these waits were written against a single
# sequential job, and a scan cycle can take much longer in wall-clock terms.
_IMPORT_WAIT_SECONDS = 60


def test_watch_folder():
    """Test watching a folder for changes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = f"{temp_dir}/server-config.json"
        pictures_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pictures")
        )
        assert os.path.isdir(pictures_dir), "Pictures directory not found"
        image_files = [
            f
            for dirpath, _, filenames in os.walk(pictures_dir)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert image_files, "No images found in pictures directory"

        with Server(server_config_path) as server:
            with TestClient(server.api) as client:
                # First login to set the password
                response = client.post(
                    f"{API_PREFIX}/login",
                    json={"username": "testuser", "password": "testpassword"},
                )
                assert response.status_code == 200
                assert (
                    response.json()["message"]
                    == "Username and password set successfully."
                )

                create_folder = client.post(
                    f"{API_PREFIX}/import-folders",
                    json={
                        "folder": pictures_dir,
                        "delete_after_import": False,
                    },
                )
                assert create_folder.status_code == 200

                start = time.monotonic()
                pictures = []
                expected_count = len(image_files)
                while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )
                    if len(pictures) >= expected_count:
                        break
                    time.sleep(0.25)

                assert len(pictures) == expected_count, (
                    f"Expected {expected_count} pictures, got {len(pictures)}"
                )
                assert all(pic.import_source_folder == pictures_dir for pic in pictures)

                filtered = client.get(
                    f"{API_PREFIX}/pictures",
                    params={
                        "fields": "grid",
                        "import_source_folder": pictures_dir,
                    },
                )
                assert filtered.status_code == 200
                assert len(filtered.json()) == expected_count

                empty_filtered = client.get(
                    f"{API_PREFIX}/pictures",
                    params={
                        "fields": "grid",
                        "import_source_folder": os.path.join(pictures_dir, "_missing"),
                    },
                )
                assert empty_filtered.status_code == 200
                assert empty_filtered.json() == []

                _assert_inbox_round_trip(client, server, pictures)


def _assert_inbox_round_trip(client, server, pictures):
    """The Inbox is what was added within its period and has no person or set."""
    ids = sorted(pic.id for pic in pictures)

    def inbox_ids(period="14d", **extra):
        listed = client.get(
            f"{API_PREFIX}/pictures",
            params={"fields": "grid", "inbox": period, **extra},
        )
        assert listed.status_code == 200, listed.text
        return sorted(row["id"] for row in listed.json())

    def inbox_count(period="14d"):
        counted = client.get(f"{API_PREFIX}/pictures/count", params={"inbox": period})
        assert counted.status_code == 200, counted.text
        return counted.json()["count"]

    # Just imported and filed nowhere: every period holds all of them.
    assert inbox_ids() == ids
    assert inbox_ids("24h") == ids
    assert inbox_ids("all") == ids
    assert inbox_count() == len(ids)
    (row,) = client.get(f"{API_PREFIX}/import-folders").json()["folders"]
    assert "unseen_count" not in row
    assert row["last_imported_at"] is not None
    assert row["missing"] is False
    # The Inbox is both halves of "unassigned", whatever a filter asks for.
    assert inbox_ids(character_id="UNASSIGNED", unassigned_by="character") == ids
    # An unknown period is refused rather than read as no Inbox at all.
    refused = client.get(f"{API_PREFIX}/pictures", params={"inbox": "2w"})
    assert refused.status_code == 422, refused.text

    # Rating is not filing: the picture stays.
    rated = client.post(
        f"{API_PREFIX}/pictures/apply-scores",
        json={"scores": {str(ids[0]): 4}, "only_unscored": False},
    )
    assert rated.status_code == 200, rated.text
    assert ids[0] in inbox_ids()

    # Filing in a set takes it out of the Inbox, not out of the library.
    created = client.post(f"{API_PREFIX}/picture_sets", json={"name": "Inbox test"})
    assert created.status_code == 200, created.text
    set_id = created.json()["picture_set"]["id"]
    filed = client.post(f"{API_PREFIX}/picture_sets/{set_id}/members/{ids[0]}")
    assert filed.status_code == 200, filed.text
    left = inbox_ids()
    assert ids[0] not in left and left
    assert inbox_count() == len(left)
    everything = client.get(f"{API_PREFIX}/pictures", params={"fields": "grid"})
    assert sorted(row["id"] for row in everything.json()) == ids

    # Added three weeks ago: outside the default fortnight, inside 30 days.
    aged = left[0]

    def backdate(session):
        picture = session.get(Picture, aged)
        picture.imported_at = datetime.now(timezone.utc) - timedelta(days=21)
        session.add(picture)
        session.commit()

    server.vault.db.run_task(backdate)
    assert aged not in inbox_ids()
    assert aged in inbox_ids("30d")
    assert inbox_count("30d") == inbox_count() + 1

    # The stats sidebar reads the same Inbox.
    stats = client.get(f"{API_PREFIX}/pictures/stats", params={"inbox": "14d"})
    assert stats.status_code == 200, stats.text
    assert stats.json()["total"] == inbox_count()


def test_watch_folder_delete_after_import():
    """Test watch folder delete_after_import removes source files after import."""
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = f"{temp_dir}/server-config.json"
        source_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pictures")
        )
        assert os.path.isdir(source_dir), "Pictures directory not found"
        image_files = [
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(source_dir)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert image_files, "No images found in pictures directory"

        watch_dir = os.path.join(temp_dir, "watch")
        os.makedirs(watch_dir, exist_ok=True)
        # One file named as a manual workflow's run saves it (#1688).
        manual = "manual:" + "a" * 32
        for index, src_path in enumerate(image_files):
            rel = os.path.relpath(src_path, source_dir)
            if index == 0:
                stem, ext = os.path.splitext(rel)
                rel = f"{stem}__wf_{'a' * 32}_00001_{ext}"
            dst = os.path.join(watch_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_path, dst)

        with Server(server_config_path) as server:
            with TestClient(server.api) as client:
                response = client.post(
                    f"{API_PREFIX}/login",
                    json={"username": "testuser", "password": "testpassword"},
                )
                assert response.status_code == 200

                create_folder = client.post(
                    f"{API_PREFIX}/import-folders",
                    json={
                        "folder": watch_dir,
                        "delete_after_import": True,
                    },
                )
                assert create_folder.status_code == 200

                expected_count = len(image_files)
                start = time.monotonic()
                pictures = []
                while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )
                    remaining_files = [
                        os.path.join(dp, f)
                        for dp, _, fnames in os.walk(watch_dir)
                        for f in fnames
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ]
                    if len(pictures) >= expected_count and not remaining_files:
                        break
                    time.sleep(0.25)

                assert len(pictures) == expected_count, (
                    f"Expected {expected_count} pictures, got {len(pictures)}"
                )
                assert all(pic.import_source_folder == watch_dir for pic in pictures)
                assert sorted(pic.run_workflow_id or "" for pic in pictures) == (
                    [""] * (expected_count - 1) + [manual]
                )
                remaining_files = [
                    os.path.join(dp, f)
                    for dp, _, fnames in os.walk(watch_dir)
                    for f in fnames
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ]
                assert not remaining_files, "Watch folder did not delete source files"


def test_watch_folder_imports_copied_files_with_old_mtime_after_initial_scan():
    """Copied files with preserved old mtime should still be imported.

    Regression test: after the first scan advances import_folder.last_checked,
    copying in files whose mtime is older than last_checked must still be
    discovered by the import finder.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = f"{temp_dir}/server-config.json"
        source_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pictures")
        )
        assert os.path.isdir(source_dir), "Pictures directory not found"
        image_files = [
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(source_dir)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert len(image_files) >= 2, "Need at least two images for this test"

        watch_dir = os.path.join(temp_dir, "watch")
        os.makedirs(watch_dir, exist_ok=True)

        first_src = image_files[0]
        second_src = image_files[1]
        first_dst = os.path.join(watch_dir, os.path.basename(first_src))
        second_dst = os.path.join(watch_dir, os.path.basename(second_src))

        shutil.copy2(first_src, first_dst)

        with Server(server_config_path) as server:
            with TestClient(server.api) as client:
                response = client.post(
                    f"{API_PREFIX}/login",
                    json={"username": "testuser", "password": "testpassword"},
                )
                assert response.status_code == 200

                create_folder = client.post(
                    f"{API_PREFIX}/import-folders",
                    json={
                        "folder": watch_dir,
                        "delete_after_import": False,
                    },
                )
                assert create_folder.status_code == 200

                # Wait for initial import to complete.
                start = time.monotonic()
                while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )
                    if len(pictures) >= 1:
                        break
                    time.sleep(0.25)

                initial_pictures = server.vault.db.run_task(
                    lambda session: Picture.find(session)
                )
                assert len(initial_pictures) == 1

                # Copy another file and force its mtime far in the past to
                # mimic file-manager copy behavior that preserves source mtime.
                shutil.copy2(second_src, second_dst)
                old_ts = time.time() - (7 * 24 * 60 * 60)
                os.utime(second_dst, (old_ts, old_ts))

                # The second file should still be discovered and imported.
                start = time.monotonic()
                while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )
                    if len(pictures) >= 2:
                        break
                    time.sleep(0.25)

                final_pictures = server.vault.db.run_task(
                    lambda session: Picture.find(session)
                )
                assert len(final_pictures) == 2, (
                    "Expected second copied file to be imported even with old mtime"
                )


def test_watch_folder_retries_after_transient_hash_failure(monkeypatch):
    """A transient hash failure must not drop a watched file permanently.

    Regression test (platform-independent): the finder marks a candidate as
    "seen" at discovery time. If hashing then fails once and the file carries a
    copy-preserved old mtime, the old code left it in the seen-set forever, so it
    could never become a candidate again. The fix has the task report the failed
    path back to the finder, which forgets it and re-discovers it on the next
    scan. We simulate the transient failure by making
    ``ImageUtils.calculate_hash_from_open_file`` raise once for the target file,
    then succeed, and assert the file is eventually imported.
    """
    from pixlstash.utils.image_processing.image_utils import ImageUtils

    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = f"{temp_dir}/server-config.json"
        source_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pictures")
        )
        assert os.path.isdir(source_dir), "Pictures directory not found"
        image_files = [
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(source_dir)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert image_files, "No images found in pictures directory"

        watch_dir = os.path.join(temp_dir, "watch")
        os.makedirs(watch_dir, exist_ok=True)

        target_src = image_files[0]
        target_dst = os.path.join(watch_dir, os.path.basename(target_src))
        shutil.copy2(target_src, target_dst)
        # Preserve an old mtime, mimicking file-manager copies. This is what
        # makes the failure permanent without the fix: seen_ts <= last_checked,
        # so the only path back to candidacy is is_new_path being True again.
        old_ts = time.time() - (7 * 24 * 60 * 60)
        os.utime(target_dst, (old_ts, old_ts))

        target_normcase = os.path.normcase(os.path.abspath(target_dst))
        real_calculate_hash = ImageUtils.calculate_hash_from_open_file
        failed_once = {"done": False}

        def flaky_calculate_hash(handle, source_label=None):
            if (
                not failed_once["done"]
                and source_label is not None
                and os.path.normcase(os.path.abspath(source_label)) == target_normcase
            ):
                failed_once["done"] = True
                raise PermissionError(
                    f"[Errno 13] Permission denied (simulated): {source_label}"
                )
            return real_calculate_hash(handle, source_label=source_label)

        monkeypatch.setattr(
            ImageUtils,
            "calculate_hash_from_open_file",
            staticmethod(flaky_calculate_hash),
        )

        with Server(server_config_path) as server:
            with TestClient(server.api) as client:
                response = client.post(
                    f"{API_PREFIX}/login",
                    json={"username": "testuser", "password": "testpassword"},
                )
                assert response.status_code == 200

                create_folder = client.post(
                    f"{API_PREFIX}/import-folders",
                    json={
                        "folder": watch_dir,
                        "delete_after_import": False,
                    },
                )
                assert create_folder.status_code == 200

                # The first hash attempt fails; the file must still be imported
                # on a later scan after the finder re-discovers it.
                start = time.monotonic()
                pictures = []
                while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )
                    if len(pictures) >= 1:
                        break
                    time.sleep(0.25)

                assert failed_once["done"], (
                    "Expected the simulated transient hash failure to fire"
                )
                assert len(pictures) == 1, (
                    "Expected the watched file to be imported after a transient "
                    "hash failure (retry on a later scan)"
                )


def test_watch_folder_waits_for_a_file_to_finish_being_written():
    """A file still being written must not be imported until it settles.

    Regression test: the finder scans every 50ms, so it can see a file that is
    still being copied into the watch folder. ``pixel_sha`` is a digest of the
    raw file bytes, so hashing a half-written file yields a hash that does not
    describe the finished file. The duplicate check therefore missed it and the
    settled file was imported a *second* time, leaving one truncated picture
    plus a duplicate. This reproduced as an intermittent CI failure in
    ``test_watch_folder_imports_copied_files_with_old_mtime_after_initial_scan``
    (3 pictures for 2 files).

    The file is left incomplete *before* the folder is registered, so the very
    first scan sees it unfinished, and a background thread keeps bumping its
    mtime until the test is ready to finish the copy. That churn is what makes
    this deterministic: the finder sees a stat signature that never repeats, so
    it cannot claim the file however often it scans. An earlier version left the
    file short but *static* and relied on writing the last bytes before the next
    scan, a race a contended runner loses (it failed on a Windows shard with 2
    pictures).

    The file is short by only its last few bytes, which keeps it decodable.
    That is precisely the case the duplicate check cannot catch, because a
    truncated file that fails to decode is already handled as a transient
    failure and retried.

    The finder's own quiet-window rule is pinned separately, without a server,
    by ``test_claim_if_settled_requires_a_wall_clock_quiet_window``.
    """
    from pixlstash.db_models.import_folder import ImportFolder
    from pixlstash.utils.image_processing.image_utils import ImageUtils
    from sqlmodel import select

    with tempfile.TemporaryDirectory() as temp_dir:
        server_config_path = f"{temp_dir}/server-config.json"
        source_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pictures")
        )
        assert os.path.isdir(source_dir), "Pictures directory not found"
        image_files = [
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(source_dir)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert image_files, "No images found in pictures directory"

        watch_dir = os.path.join(temp_dir, "watch")
        os.makedirs(watch_dir, exist_ok=True)

        src = image_files[0]
        dst = os.path.join(watch_dir, os.path.basename(src))

        with open(src, "rb") as source_handle:
            payload = source_handle.read()
        assert len(payload) > 4, "Need a non-trivial image for this test"

        # The copy is "in flight": everything but the last few bytes is on disk.
        with open(dst, "wb") as out:
            out.write(payload[:-4])
            out.flush()
            os.fsync(out.fileno())

        stop_touching = threading.Event()
        touch_errors: list[OSError] = []
        stamp_base = time.time() - (24 * 60 * 60)

        def keep_the_copy_in_flight():
            """Bump the file's mtime until the test finishes the copy.

            The finder claims a path only once its ``(size, mtime, ctime)``
            signature has held still, so a moving mtime stands in for a copy
            that is still dribbling bytes in. Each bump is a distinct integer
            timestamp, so this works regardless of the filesystem's timestamp
            granularity.
            """
            bump = 0
            while not stop_touching.is_set():
                bump += 1
                try:
                    os.utime(dst, (stamp_base + bump, stamp_base + bump))
                except OSError as exc:
                    touch_errors.append(exc)
                    return
                stop_touching.wait(0.05)

        toucher = threading.Thread(target=keep_the_copy_in_flight, daemon=True)
        toucher.start()
        try:
            with Server(server_config_path) as server:
                with TestClient(server.api) as client:
                    response = client.post(
                        f"{API_PREFIX}/login",
                        json={"username": "testuser", "password": "testpassword"},
                    )
                    assert response.status_code == 200

                    create_folder = client.post(
                        f"{API_PREFIX}/import-folders",
                        json={
                            "folder": watch_dir,
                            "delete_after_import": False,
                        },
                    )
                    assert create_folder.status_code == 200

                    # ``last_checked`` only advances once a scan has completed,
                    # so this proves the finder has already looked at the
                    # partial file.
                    def read_folder(session):
                        return session.exec(select(ImportFolder)).first()

                    scanned = False
                    start = time.monotonic()
                    while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                        folder_row = server.vault.db.run_task(read_folder)
                        if (
                            folder_row is not None
                            and (folder_row.last_checked or 0) > 0
                        ):
                            scanned = True
                            break
                        time.sleep(0.1)
                    assert scanned, "Finder never completed a scan of the partial file"
                    assert not touch_errors, (
                        f"Could not keep the copy in flight: {touch_errors[0]!r}"
                    )

                    # Finish the copy, then mimic a file manager restoring the
                    # source mtime (which bumps ctime and re-exposes the file to
                    # the finder).
                    stop_touching.set()
                    toucher.join(timeout=10)
                    assert not toucher.is_alive(), "Touch thread failed to stop"
                    with open(dst, "ab") as out:
                        out.write(payload[-4:])
                        out.flush()
                        os.fsync(out.fileno())
                    old_ts = time.time() - (7 * 24 * 60 * 60)
                    os.utime(dst, (old_ts, old_ts))

                    start = time.monotonic()
                    pictures = []
                    while time.monotonic() - start < _IMPORT_WAIT_SECONDS:
                        pictures = server.vault.db.run_task(
                            lambda session: Picture.find(session)
                        )
                        if len(pictures) >= 1:
                            break
                        time.sleep(0.25)

                    # Give a second, erroneous import a chance to land before
                    # asserting there is only one picture.
                    time.sleep(3.0)
                    pictures = server.vault.db.run_task(
                        lambda session: Picture.find(session)
                    )

                    assert len(pictures) == 1, (
                        "Expected exactly one picture for one watched file; a "
                        "half-written import would add a duplicate (got "
                        f"{len(pictures)})"
                    )
                    settled_sha = ImageUtils.calculate_hash_from_file_path(dst)
                    assert pictures[0].pixel_sha == settled_sha, (
                        "Imported picture must carry the hash of the settled "
                        "file, not of the bytes written so far"
                    )
                    assert pictures[0].size_bytes == os.path.getsize(dst), (
                        "Imported picture must record the settled file size"
                    )
        finally:
            stop_touching.set()
            toucher.join(timeout=10)


def test_claim_if_settled_requires_a_wall_clock_quiet_window():
    """A signature that merely repeats across two scans is not settled yet.

    The planner polls as often as ``WorkPlanner.MIN_INTERVAL_S`` (50ms), so
    "unchanged since the previous scan" can mean "unchanged for 50ms", which a
    copy that pauses mid-write clears easily. The file is then hashed
    half-written, and imported a *second* time once it really finishes, because
    the settled bytes hash differently and the duplicate check misses them. The
    claim therefore has to be gated on wall-clock quiet, not on scan count.
    """
    from pixlstash.tasks.missing_watch_folder_import_finder import (
        MissingWatchFolderImportFinder,
    )

    finder = MissingWatchFolderImportFinder(database=None)
    settle = MissingWatchFolderImportFinder.SETTLE_SECONDS
    path = os.path.normcase(os.path.abspath("watch/in-flight.png"))
    partial = (1024, 100.0, 100.0)
    complete = (2048, 101.0, 100.0)

    # First sighting never claims, and neither does a fast follow-up scan.
    assert finder._claim_if_settled(path, partial, 0.0) is False
    assert finder._claim_if_settled(path, partial, 0.05) is False
    assert finder._claim_if_settled(path, partial, settle - 0.01) is False

    # The write resumes: the quiet window restarts from this observation.
    assert finder._claim_if_settled(path, complete, settle) is False
    assert finder._claim_if_settled(path, complete, 2 * settle - 0.01) is False

    # Quiet for the full window: now it is safe to hand over.
    assert finder._claim_if_settled(path, complete, 2 * settle) is True

    # Claimed once. Later scans of the same bytes must not offer it again.
    assert finder._claim_if_settled(path, complete, 2 * settle + 60.0) is False


@pytest.mark.skipif(_CANNOT_LOCK_A_FOLDER, reason=_LOCK_SKIP_REASON)
def test_watch_folder_scan_warns_once_per_folder_it_cannot_read(tmp_path, caplog):
    """A directory the scan cannot list is reported, not skipped in silence.

    ``os.walk`` drops a directory it cannot list unless it is given an
    ``onerror``, so a locked folder imported nothing and the log said nothing.
    The scan now logs one warning per watch folder - not one per directory -
    naming the folder and the error, and only when the set of directories it
    cannot read changes: the planner scans every cycle, many times a second
    while it is busy, so a warning per scan repeats the same line without end.
    One info line says when the folder can be read again.

    A folder behind a parent the server may not enter, and one it may list but
    not enter, are reported the same way rather than taken for missing or
    passed over with a debug line per file.
    """
    root_locked = tmp_path / "root-locked"
    part_locked = tmp_path / "part-locked"
    behind_locked_parent = tmp_path / "gated" / "watch"
    list_only = tmp_path / "list-only"
    readable = tmp_path / "readable"
    unreadable_folders = [root_locked, part_locked, behind_locked_parent, list_only]
    # Path -> the mode that keeps the scan out of it.
    locks = {
        root_locked: 0,
        part_locked / "locked-a": 0,
        part_locked / "locked-b": 0,
        behind_locked_parent.parent: 0,
        list_only: 0o444,
    }
    for path in [*locks, behind_locked_parent, readable / "nested"]:
        path.mkdir(parents=True, exist_ok=True)

    rows = [
        ImportFolder(id=index, folder=str(path), label=path.name)
        for index, path in enumerate([*unreadable_folders, readable], start=1)
    ]
    finder = missing_watch_folder_import_finder.MissingWatchFolderImportFinder(
        database=mock.Mock()
    )
    logger_name = missing_watch_folder_import_finder.logger.name

    def ours():
        return [record for record in caplog.records if record.name == logger_name]

    def logged(level):
        return [record.getMessage() for record in ours() if record.levelno == level]

    def naming(messages, folder):
        return [message for message in messages if str(folder) in message]

    def scan(scan_rows=rows):
        assert finder._find_task_from_db(scan_rows) is None

    denied = os.strerror(errno.EACCES)
    try:
        for path, mode in locks.items():
            path.chmod(mode)
        assert not os.access(root_locked, os.R_OK), "chmod 000 did not lock it"
        with caplog.at_level(logging.INFO, logger=logger_name):
            scan()
            warnings = logged(logging.WARNING)
            # One line for each folder it cannot read, none for the readable one.
            assert {
                path.name: len(naming(warnings, path))
                for path in [*unreadable_folders, readable]
            } == {
                root_locked.name: 1,
                part_locked.name: 1,
                behind_locked_parent.name: 1,
                list_only.name: 1,
                readable.name: 0,
            }
            for folder in unreadable_folders:
                (line,) = naming(warnings, folder)
                assert denied in line
            # Two locked subdirectories, one line.
            (part_warning,) = naming(warnings, part_locked)
            assert "2 directories" in part_warning

            # Nothing changed, so nothing new to say.
            scan()
            scan()
            assert logged(logging.WARNING) == warnings
            assert logged(logging.INFO) == []

            # One of the two locked subdirectories opens up: that is a change.
            (part_locked / "locked-a").chmod(0o700)
            scan()
            (part_changed,) = logged(logging.WARNING)[len(warnings) :]
            assert str(part_locked) in part_changed
            assert "1 directory" in part_changed
            warnings = logged(logging.WARNING)

            # Everything opens up: one line per folder that was locked, then quiet.
            for path in locks:
                path.chmod(0o700)
            scan()
            recovered = logged(logging.INFO)
            for folder in unreadable_folders:
                (line,) = naming(recovered, folder)
                assert "can be read again" in line
            assert naming(recovered, readable) == []
            assert logged(logging.WARNING) == warnings
            settled = ours()
            scan()
            assert ours() == settled

            # A folder dropped from the list is forgotten, so it warns afresh
            # when it comes back still locked.
            root_locked.chmod(0)
            scan()
            (relocked,) = logged(logging.WARNING)[len(warnings) :]
            assert str(root_locked) in relocked
            settled = ours()
            scan([row for row in rows if row.folder != str(root_locked)])
            assert ours() == settled
            scan()
            (returned,) = logged(logging.WARNING)[len(warnings) + 1 :]
            assert str(root_locked) in returned
            warnings = logged(logging.WARNING)

            # So it does when no folder at all was registered for a while.
            with mock.patch.object(finder, "_load_db_watch_folders", return_value=[]):
                assert finder.find_task() is None
            scan()
            (after_none,) = logged(logging.WARNING)[len(warnings) :]
            assert str(root_locked) in after_none
    finally:
        for path in locks:
            path.chmod(0o700)


def test_watch_folder_scan_skips_a_folder_the_path_rules_refuse(tmp_path, caplog):
    """The scan applies the add route's path rule on every pass.

    ``POST /import-folders`` refuses a folder in a restricted system directory,
    but the scan used to take the stored row as given, so a folder that came to
    resolve into one later - a link retargeted, a mount moved, a row written
    before the rule - was walked and imported from. The scan now checks
    ``validate_reference_folder_path`` itself: a refused folder offers no file,
    is not marked checked, and is reported once while the refusal holds.
    """
    allowed = tmp_path / "allowed"
    target = tmp_path / "target"
    for folder in (allowed, target):
        folder.mkdir()
        Image.new("RGB", (1, 1)).save(folder / "picture.png")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    rows = [
        ImportFolder(id=1, folder=str(link), label="link"),
        ImportFolder(id=2, folder=str(allowed), label="allowed"),
    ]
    database = mock.Mock()
    finder = missing_watch_folder_import_finder.MissingWatchFolderImportFinder(
        database=database
    )
    logger_name = missing_watch_folder_import_finder.logger.name

    def logged(level, folder):
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == logger_name
            and record.levelno == level
            and str(folder) in record.getMessage()
        ]

    def scan():
        database.run_task.reset_mock()
        assert finder._find_task_from_db(rows) is None
        ((_, checked), _) = database.run_task.call_args
        return set(checked)

    def offered(folder):
        return os.path.normcase(os.path.abspath(folder / "picture.png"))

    refuse_target = mock.patch.object(
        reference_folder_validator,
        "_get_blocklist",
        return_value=frozenset({os.path.realpath(target)}),
    )
    with caplog.at_level(logging.INFO, logger=logger_name):
        with refuse_target:
            assert scan() == {2}
            assert scan() == {2}
        # The allowed folder was walked; the link into the refused one was not.
        assert offered(allowed) in finder._pending_stats
        assert offered(link) not in finder._pending_stats
        (warning,) = logged(logging.WARNING, link)
        assert "restricted system directory" in warning
        assert logged(logging.WARNING, allowed) == []

        # The rule stops refusing it: one info line, and the scan walks it.
        assert scan() == {1, 2}
        assert offered(link) in finder._pending_stats
        (allowed_again,) = logged(logging.INFO, link)
        assert "allowed by the path rules again" in allowed_again
        assert len(logged(logging.WARNING, link)) == 1


def test_watch_folder_listing_refuses_rather_than_reporting_none():
    """#1177 item 59's import-folder sibling, from the review of #1201.

    `get_import_folder_paths` used to answer a failed read with `[]`, which is
    exactly what "no watch folders are configured" looks like. In the export
    route that turned a destination check off silently; here it is a plainer
    lie - the owner is told they watch nothing, which is how someone re-adds a
    folder they already have or concludes their folders were lost. It now
    raises, and this route turns that into a 503 rather than an unhandled 500.
    """
    from unittest import mock

    from pixlstash.utils.path_utils import LibraryRootsUnavailable

    with tempfile.TemporaryDirectory() as temp_dir:
        with Server(f"{temp_dir}/server-config.json") as server:
            with TestClient(server.api) as client:
                response = client.post(
                    f"{API_PREFIX}/login",
                    json={"username": "testuser", "password": "testpassword"},
                )
                assert response.status_code == 200, response.text

                # The positive control first, so a 503 below cannot be the
                # route being broken outright: a readable table still lists.
                ok = client.get(f"{API_PREFIX}/server-config/watch-folders")
                assert ok.status_code == 200, ok.text
                assert isinstance(ok.json()["watch_folders"], list)

                def _explode(_vault):
                    raise LibraryRootsUnavailable("test-induced read failure")

                with mock.patch(
                    "pixlstash.routes.config.config_service.get_import_folder_paths",
                    _explode,
                ):
                    refused = client.get(f"{API_PREFIX}/server-config/watch-folders")
                assert refused.status_code == 503, refused.text
                assert "watched folders" in refused.json()["detail"]

                # And it recovers: the refusal is about this read, not a latch.
                again = client.get(f"{API_PREFIX}/server-config/watch-folders")
                assert again.status_code == 200, again.text


@pytest.mark.skipif(_CANNOT_LOCK_A_FOLDER, reason=_LOCK_SKIP_REASON)
def test_import_folder_listing_tells_an_unreadable_folder_from_a_missing_one():
    """A folder on disk that the server cannot list is listed as unreadable.

    It used to read as healthy: the scan skipped it without a word and the
    settings row said "Nothing yet" for ever. Adding one stays allowed, as does
    adding a folder that is not there yet (a drive or mount that comes later),
    because both can become importable without the owner touching PixlStash.

    Its own server rather than a shared one: locking a folder needs the skip on
    Windows and for root, which would take the neighbouring test's coverage
    with it.

    The answer comes from the filesystem alone. A folder whose parent the
    server may not enter is there but unreadable, not missing; a link into a
    directory the path rules refuse is still readable, so calling it unreadable
    would be false. The scan does not walk it: that refusal is the path rules',
    checked separately (``test_watch_folder_scan_skips_a_folder_the_path_rules_refuse``).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        locked = os.path.join(temp_dir, "locked")
        readable = os.path.join(temp_dir, "readable")
        not_there_yet = os.path.join(temp_dir, "not-mounted-yet")
        gate = os.path.join(temp_dir, "gated")
        behind_locked_parent = os.path.join(gate, "watch")
        link_target = os.path.join(temp_dir, "link-target")
        link = os.path.join(temp_dir, "link")
        for folder in (locked, readable, behind_locked_parent, link_target):
            os.makedirs(folder)
        Image.new("RGB", (1, 1)).save(os.path.join(link_target, "picture.png"))
        os.symlink(link_target, link)
        try:
            os.chmod(locked, 0)
            assert not os.access(locked, os.R_OK), "chmod 000 did not lock it"
            with Server(f"{temp_dir}/server-config.json") as server:
                with TestClient(server.api) as client:
                    response = client.post(
                        f"{API_PREFIX}/login",
                        json={"username": "testuser", "password": "testpassword"},
                    )
                    assert response.status_code == 200, response.text

                    for folder in (
                        locked,
                        readable,
                        not_there_yet,
                        behind_locked_parent,
                        link,
                    ):
                        added = client.post(
                            f"{API_PREFIX}/import-folders",
                            json={"folder": folder},
                        )
                        assert added.status_code == 200, (folder, added.text)

                    # After registration, as when a parent's permissions change
                    # or the rules come to refuse where a link points.
                    os.chmod(gate, 0)
                    with mock.patch.object(
                        reference_folder_validator,
                        "_get_blocklist",
                        return_value=frozenset({os.path.realpath(link_target)}),
                    ):
                        assert (
                            reference_folder_validator.validate_reference_folder_path(
                                link
                            )
                            is not None
                        ), "the path rules do not refuse the link's target"
                        listed = client.get(f"{API_PREFIX}/import-folders")
                    assert listed.status_code == 200, listed.text
                    state = {
                        row["folder"]: (
                            row["missing"],
                            row["unreadable"],
                            row["refused"] is not None,
                        )
                        for row in listed.json()["folders"]
                    }
                    # The link is readable, but listed as refused: the watcher
                    # will not touch it, so it must not look healthy.
                    assert state == {
                        locked: (False, True, False),
                        readable: (False, False, False),
                        not_there_yet: (True, False, False),
                        behind_locked_parent: (False, True, False),
                        link: (False, False, True),
                    }
        finally:
            os.chmod(locked, 0o700)
            os.chmod(gate, 0o700)


# ---------------------------------------------------------------------------
# Delete-after-import removes only the exact file it imported
# ---------------------------------------------------------------------------

# Every picture these tests write has its own colour, so none of them is a
# duplicate of another on the shared server.
_watch_colours = itertools.count(1)


@pytest.fixture(scope="module")
def watch_server():
    """One server for the task-boundary delete tests below.

    They run the task directly on candidates they build, so no watch folder is
    registered and the finder offers nothing of its own.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with Server(os.path.join(temp_dir, "server-config.json")) as server:
            yield server


def _write_picture(path) -> bytes:
    colour = next(_watch_colours)
    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), (colour % 256, colour // 256, 77)).save(
        buffer, format="PNG"
    )
    data = buffer.getvalue()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return data


def _watch_pass(server, file_path, root) -> dict:
    task = WatchFolderImportTask(
        server.vault.db,
        [
            {
                "file_path": str(file_path),
                "delete_after_import": True,
                "import_source_folder": str(root),
            }
        ],
        1,
    )
    return task._run_task()


def _after_insert(monkeypatch, server, action):
    """Run *action* once the task's insert has committed, before it deletes."""
    database = server.vault.db
    real_run_task = database.run_task

    def run_task(func, *args, **kwargs):
        result = real_run_task(func, *args, **kwargs)
        if getattr(func, "__name__", "") == "insert_pictures":
            action()
        return result

    monkeypatch.setattr(database, "run_task", run_task)


def _picture_ids_with_bytes(server, data: bytes) -> list[int]:
    sha = ImageUtils.calculate_full_hash_from_bytes(data)
    pictures = server.vault.db.run_task(lambda session: Picture.find(session))
    root = server.vault.image_root
    return [
        pic.id
        for pic in pictures
        if pic.file_path
        and ImageUtils.calculate_full_hash_from_file_path(
            ImageUtils.resolve_picture_path(root, pic.file_path)
        )
        == sha
    ]


def test_delete_after_import_removes_the_file_it_imported(watch_server, tmp_path):
    """The positive control: nothing changed, so the imported file goes."""
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    data = _write_picture(file_path)

    result = _watch_pass(watch_server, file_path, root)

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data) == [picture_id]
    assert not os.path.lexists(file_path)


def _replace_file(ctx):
    _write_picture(ctx.file_path.with_suffix(".tmp"))
    os.replace(ctx.file_path.with_suffix(".tmp"), ctx.file_path)


def _rewrite_in_place(ctx):
    with open(ctx.file_path, "ab") as handle:
        handle.write(b"appended after the read")


def _refuse_by_the_rules(ctx):
    refused = frozenset({os.path.realpath(ctx.root)})
    ctx.monkeypatch.setattr(
        reference_folder_validator, "_get_blocklist", lambda: refused
    )


@pytest.mark.parametrize(
    "change",
    [_replace_file, _rewrite_in_place, _refuse_by_the_rules],
    ids=["replaced", "rewritten", "rules-refuse"],
)
def test_delete_after_import_keeps_a_file_that_changed_before_the_delete(
    watch_server, tmp_path, monkeypatch, change
):
    """Each check runs again at delete time, after the rows are committed.

    The name now leads to a different file, the same file holds different
    bytes, or the path rules refuse the folder: the picture is imported and
    the file on disk stays.
    """
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    data = _write_picture(file_path)
    ctx = SimpleNamespace(file_path=file_path, root=root, monkeypatch=monkeypatch)
    _after_insert(monkeypatch, watch_server, lambda: change(ctx))

    result = _watch_pass(watch_server, file_path, root)

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data) == [picture_id]
    assert os.path.isfile(file_path)


def _link_to_real(ctx):
    os.symlink(ctx.real_root, ctx.root, target_is_directory=True)


def _link_above(ctx):
    # The watch folder itself is a plain directory; one above it is a link.
    os.symlink(ctx.real_root.parent, ctx.tmp_path / "linked-parent")
    ctx.root = ctx.tmp_path / "linked-parent" / ctx.real_root.name


def _link_re_pointed_before_the_scan(ctx):
    # Registered as a link to an allowed folder, re-pointed before the scan at
    # another allowed folder the owner never registered.
    os.symlink(ctx.real_root, ctx.root, target_is_directory=True)
    os.unlink(ctx.root)
    os.symlink(ctx.other_dir, ctx.root, target_is_directory=True)
    ctx.read = ctx.other_dir


def _link_re_pointed_after_the_insert(ctx):
    os.symlink(ctx.real_root, ctx.root, target_is_directory=True)

    def _re_point():
        os.unlink(ctx.root)
        os.symlink(ctx.other_dir, ctx.root, target_is_directory=True)

    ctx.after_insert = _re_point


@pytest.mark.parametrize(
    "layout",
    [
        _link_to_real,
        _link_above,
        _link_re_pointed_before_the_scan,
        _link_re_pointed_after_the_insert,
    ],
    ids=["link", "link-above", "re-pointed-before-scan", "re-pointed-after-insert"],
)
def test_a_watch_folder_reached_through_a_link_is_imported_but_never_deleted_from(
    watch_server, tmp_path, monkeypatch, layout
):
    """Where a link leads can change at any time, and was checked only once,
    when the folder was registered: delete-after-import never follows one."""
    real_root = tmp_path / "parent" / "real"
    other_dir = tmp_path / "elsewhere"
    ctx = SimpleNamespace(
        tmp_path=tmp_path,
        root=tmp_path / "watch",
        real_root=real_root,
        other_dir=other_dir,
        read=real_root,
        after_insert=None,
    )
    data = {
        real_root: _write_picture(real_root / "picture.png"),
        other_dir: _write_picture(other_dir / "picture.png"),
    }
    layout(ctx)
    if ctx.after_insert is not None:
        _after_insert(monkeypatch, watch_server, ctx.after_insert)

    result = _watch_pass(watch_server, ctx.root / "picture.png", ctx.root)

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data[ctx.read]) == [picture_id]
    for folder, content in data.items():
        with open(folder / "picture.png", "rb") as handle:
            assert handle.read() == content, f"{folder} lost its file"


def _file_link_out(root, outside):
    os.makedirs(root)
    os.symlink(outside / "picture.png", root / "picture.png")
    return root / "picture.png"


def _parent_link_out(root, outside):
    os.makedirs(root)
    os.symlink(outside, root / "sub", target_is_directory=True)
    return root / "sub" / "picture.png"


@pytest.mark.parametrize(
    "layout", [_file_link_out, _parent_link_out], ids=["file-link", "parent-link"]
)
def test_a_link_out_of_the_watch_folder_is_neither_read_nor_deleted(
    watch_server, tmp_path, layout
):
    """A file reached through a link that leaves the folder is not touched."""
    outside = tmp_path / "outside"
    data = _write_picture(outside / "picture.png")
    file_path = layout(tmp_path / "watch", outside)

    result = _watch_pass(watch_server, file_path, tmp_path / "watch")

    assert result["imported_picture_ids"] == []
    assert _picture_ids_with_bytes(watch_server, data) == []
    with open(outside / "picture.png", "rb") as handle:
        assert handle.read() == data


def test_a_watch_pass_looks_up_matches_off_the_writer_and_sees_earlier_batches(
    watch_server, tmp_path, monkeypatch
):
    """Each file is looked up twice: before the lock, while the pass reads it,
    and again under the import lock, right before the insert. Neither lookup
    may queue behind background writes, and both see the pictures the batch
    before added."""
    monkeypatch.setattr(watch_folder_import_task, "_IMPORT_BATCH", 1)
    real_lookup = import_dedup_service.load_match_candidates_in_session
    lookups = []

    def _spy(session, keys, *args, **kwargs):
        lookups.append(
            {
                "keys": tuple(sorted(keys)),
                "on_writer": threading.current_thread()
                is watch_server.vault.db._task_worker,
                "rows_visible": len(session.exec(select(Picture.id)).all()),
            }
        )
        return real_lookup(session, keys, *args, **kwargs)

    monkeypatch.setattr(import_dedup_service, "load_match_candidates_in_session", _spy)
    root = tmp_path / "watch"
    paths = [root / "first.png", root / "second.png"]
    for path in paths:
        _write_picture(path)
    task = WatchFolderImportTask(
        watch_server.vault.db,
        [{"file_path": str(path), "import_source_folder": str(root)} for path in paths],
        len(paths),
    )

    result = task._run_task()

    assert len(result["imported_picture_ids"]) == 2
    assert lookups, "the pass looked nothing up"
    assert [lookup for lookup in lookups if lookup["on_writer"]] == [], lookups
    rows_seen: dict[tuple, list[int]] = {}
    for lookup in lookups:
        rows_seen.setdefault(lookup["keys"], []).append(lookup["rows_visible"])
    first, second = rows_seen.values()
    assert set(second) == {max(first) + 1}, lookups


def _caption_and_tags(server, picture_id) -> tuple[str | None, list[str]]:
    def read(session):
        picture = session.get(Picture, picture_id)
        tags = session.exec(select(Tag.tag).where(Tag.picture_id == picture_id))
        return picture.description, sorted(tags.all())

    return server.vault.db.run_task(read)


@pytest.mark.parametrize("where, read", [("outside", False), ("watch/captions", True)])
def test_a_caption_link_is_read_only_when_it_stays_in_the_watch_folder(
    watch_server, tmp_path, where, read
):
    """A caption file named after the picture is read by path, so a link
    leading out of the folder would bring that file's text into the library.
    The picture is imported either way; a link that stays inside is read."""
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    data = _write_picture(file_path)
    caption = tmp_path / where / "caption_description.txt"
    os.makedirs(caption.parent, exist_ok=True)
    caption.write_text(f"example-caption from {where}", encoding="utf-8")
    os.symlink(caption, root / "picture_description.txt")

    result = _watch_pass(watch_server, file_path, root)

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data) == [picture_id]
    description, tags = _caption_and_tags(watch_server, picture_id)
    assert description == (f"example-caption from {where}" if read else None)
    assert not any("example-caption" in tag for tag in tags)


def _swap_after_check(monkeypatch, checked_path, swap):
    """Run *swap* once, right after the task has resolved and checked
    *checked_path*: the window a link swapped in after the check would use."""
    real_resolved = watch_folder_import_task._resolved_source
    done = []

    def _resolved_source(file_path, root, image_root):
        result = real_resolved(file_path, root, image_root)
        if not done and os.fspath(file_path) == os.fspath(checked_path):
            done.append(file_path)
            swap()
        return result

    monkeypatch.setattr(watch_folder_import_task, "_resolved_source", _resolved_source)
    return done


def _no_swap(ctx):
    return ctx.root / "picture.png"


def _file_swapped_for_a_link(ctx):
    def swap():
        os.symlink(ctx.outside / "picture.png", ctx.root / "link.tmp")
        os.replace(ctx.root / "link.tmp", ctx.root / "picture.png")

    ctx.swap = swap
    return ctx.root / "picture.png"


def _parent_swapped_for_a_link(ctx):
    os.makedirs(ctx.root / "sub")
    os.replace(ctx.root / "picture.png", ctx.root / "sub" / "picture.png")

    def swap():
        os.rename(ctx.root / "sub", ctx.root / "sub-moved")
        os.symlink(ctx.outside, ctx.root / "sub", target_is_directory=True)

    ctx.swap = swap
    return ctx.root / "sub" / "picture.png"


@pytest.mark.parametrize(
    "layout, imported",
    [
        (_no_swap, True),
        (_file_swapped_for_a_link, False),
        (_parent_swapped_for_a_link, False),
    ],
    ids=["control", "file-swapped", "parent-swapped"],
)
def test_a_link_swapped_in_after_the_check_is_never_read(
    watch_server, tmp_path, monkeypatch, layout, imported
):
    """The bytes hashed and copied come from the file the check passed.

    A file or a directory above it swapped for a link out of the watch folder
    between the check and the read would otherwise bring the outside file into
    the library (and its name would be a delete candidate). The control: with
    nothing swapped the same file is imported.
    """
    ctx = SimpleNamespace(root=tmp_path / "watch", outside=tmp_path / "outside")
    inside = _write_picture(ctx.root / "picture.png")
    outside = _write_picture(ctx.outside / "picture.png")
    ctx.swap = None
    file_path = layout(ctx)
    swapped = _swap_after_check(monkeypatch, file_path, ctx.swap) if ctx.swap else None

    result = _watch_pass(watch_server, file_path, ctx.root)

    assert swapped is None or swapped, "the swap never ran: the race was not staged"
    assert _picture_ids_with_bytes(watch_server, outside) == []
    if imported:
        (picture_id,) = result["imported_picture_ids"]
        assert _picture_ids_with_bytes(watch_server, inside) == [picture_id]
    else:
        assert result["imported_picture_ids"] == []
    with open(ctx.outside / "picture.png", "rb") as handle:
        assert handle.read() == outside, "the outside file was touched"


@pytest.mark.parametrize("replacement", ["link", "file"])
def test_a_file_swapped_after_it_was_opened_is_imported_as_opened(
    watch_server, tmp_path, monkeypatch, replacement
):
    """Once open, the name is not read again: the sampled hash, the copy and
    the identity delete-after-import looks for come from the open file. Wrong
    if the library holds the other bytes, if the recorded hash is the other
    file's, or if the name, now a link or another file, is deleted."""
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    inside = _write_picture(file_path)
    other_path = tmp_path / "outside" / "picture.png"
    other = _write_picture(other_path)
    real_open = watch_folder_import_task._open_checked
    swapped = []

    def _open_checked(real_path):
        handle = real_open(real_path)
        if not swapped and real_path == os.path.realpath(file_path):
            swapped.append(real_path)
            if replacement == "link":
                os.symlink(other_path, root / "swap.tmp")
            else:
                shutil.copyfile(other_path, root / "swap.tmp")
            os.replace(root / "swap.tmp", file_path)
        return handle

    monkeypatch.setattr(watch_folder_import_task, "_open_checked", _open_checked)

    result = _watch_pass(watch_server, file_path, root)

    assert swapped, "the swap never ran: the race was not staged"
    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, inside) == [picture_id]
    assert _picture_ids_with_bytes(watch_server, other) == []
    picture = watch_server.vault.db.run_task(
        lambda session: session.get(Picture, picture_id)
    )
    assert picture.pixel_sha == ImageUtils.calculate_hash_from_bytes(inside)
    assert os.path.lexists(file_path), "the swapped-in name was deleted"
    with open(file_path, "rb") as handle:
        assert handle.read() == other
    with open(other_path, "rb") as handle:
        assert handle.read() == other


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_in_place_of_a_checked_file_is_refused_without_waiting(tmp_path):
    """Opened O_NONBLOCK, so the open returns at once; then refused, as it is
    not a regular file. Wrong if it opens (or if the test hangs)."""
    fifo = tmp_path / "picture.png"
    os.mkfifo(fifo)

    with pytest.raises(OSError, match="not a regular file"):
        watch_folder_import_task._open_checked(str(fifo))


def test_the_windows_checked_open_refuses_a_path_that_became_a_link(tmp_path):
    """Windows has no dir_fd, so the checked path is opened by name and then
    confirmed to be the file that name holds, with no link on the way. Run
    here directly: the fallback is plain path code. The control opens."""
    plain = tmp_path / "watch" / "picture.png"
    data = _write_picture(plain)
    outside = tmp_path / "outside" / "picture.png"
    _write_picture(outside)
    with watch_folder_import_task._open_checked_by_path(str(plain)) as handle:
        assert handle.read() == data

    os.symlink(outside, tmp_path / "watch" / "link.tmp")
    os.replace(tmp_path / "watch" / "link.tmp", plain)
    with pytest.raises(OSError, match="changed while it was opened"):
        watch_folder_import_task._open_checked_by_path(str(plain))

    os.unlink(plain)
    os.rename(tmp_path / "watch", tmp_path / "watch-moved")
    os.symlink(tmp_path / "outside", tmp_path / "watch", target_is_directory=True)
    with pytest.raises(OSError, match="changed while it was opened"):
        watch_folder_import_task._open_checked_by_path(str(plain))


def test_the_windows_checked_open_refuses_a_link_gone_before_the_check(
    tmp_path, monkeypatch
):
    """A link in place for the open and swapped back before the check: the
    name is a regular file again and resolves to itself, so only the open
    file's index against the name's tells them apart. Staged by opening the
    outside file where the name is opened. Wrong if it opens."""
    plain = tmp_path / "watch" / "picture.png"
    _write_picture(plain)
    outside = tmp_path / "outside" / "picture.png"
    _write_picture(outside)
    opened = []

    def _open_through_the_link(path, mode):
        assert path == str(plain)
        opened.append(open(outside, mode))
        return opened[-1]

    monkeypatch.setattr(
        watch_folder_import_task, "open", _open_through_the_link, raising=False
    )

    with pytest.raises(OSError, match="changed while it was opened"):
        watch_folder_import_task._open_checked_by_path(str(plain))
    assert opened and opened[0].closed, "the refused handle was left open"


@pytest.mark.skipif(os.name == "nt", reason="uses the POSIX null device")
def test_the_windows_checked_open_refuses_a_file_that_is_not_regular():
    """The null device is its own name, resolves to itself and is not a link,
    so only the regular-file requirement refuses it. Wrong if it opens."""
    with pytest.raises(OSError, match="changed while it was opened"):
        watch_folder_import_task._open_checked_by_path(os.devnull)


def _caption_swapped_for(target_text):
    def swap(ctx):
        if target_text is None:
            os.symlink(ctx.outside_caption, ctx.root / "caption.tmp")
        else:
            (ctx.root / "caption.tmp").write_text(target_text, encoding="utf-8")
        os.replace(ctx.root / "caption.tmp", ctx.caption)

    return swap


@pytest.mark.parametrize(
    "swap, description",
    [
        (
            _caption_swapped_for("example-caption rewritten"),
            "example-caption rewritten",
        ),
        (_caption_swapped_for(None), None),
    ],
    ids=["control-rewritten", "link-out"],
)
def test_a_caption_swapped_for_a_link_after_its_check_is_not_read(
    watch_server, tmp_path, monkeypatch, swap, description
):
    """A caption file is read from the file its own check passed.

    Swapped for a link out of the watch folder after the check, the outside
    text must not become the picture's description or tags. The control: a
    caption rewritten in place is still read, so captions are not simply off.
    """
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    data = _write_picture(file_path)
    ctx = SimpleNamespace(
        root=root,
        caption=root / "picture_description.txt",
        outside_caption=tmp_path / "outside" / "caption.txt",
    )
    ctx.caption.write_text("example-caption inside", encoding="utf-8")
    os.makedirs(ctx.outside_caption.parent)
    ctx.outside_caption.write_text("example-caption from outside", encoding="utf-8")
    swapped = _swap_after_check(monkeypatch, ctx.caption, lambda: swap(ctx))

    result = _watch_pass(watch_server, file_path, root)

    assert swapped, "the caption was never checked: the race was not staged"
    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data) == [picture_id]
    got_description, tags = _caption_and_tags(watch_server, picture_id)
    assert got_description == description
    assert not any("outside" in tag for tag in tags)


def test_a_junction_on_the_way_to_the_watch_folder_counts_as_a_link(
    tmp_path, monkeypatch
):
    """``os.path.islink`` is False for a Windows junction, which needs only
    write access to make. No junction exists on Linux, so the reparse tag is
    simulated, as tests/test_hub_engine.py does for the hub path."""
    root = tmp_path / "junction" / "watch"
    root.mkdir(parents=True)
    junction = str(tmp_path / "junction")
    mount_point_tag = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT
    real_lstat = os.lstat

    class _JunctionStat:
        def __init__(self, info):
            self._info = info
            self.st_reparse_tag = mount_point_tag

        def __getattr__(self, name):
            return getattr(self._info, name)

    assert watch_folder_import_task._link_on_the_way_to(str(root)) is None
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(
        stat, "IO_REPARSE_TAG_MOUNT_POINT", mount_point_tag, raising=False
    )
    monkeypatch.setattr(
        os,
        "lstat",
        lambda p, *a, **k: (
            _JunctionStat(real_lstat(p, *a, **k))
            if str(p) == junction
            else real_lstat(p, *a, **k)
        ),
    )

    assert watch_folder_import_task._link_on_the_way_to(str(root)) == junction


# ---------------------------------------------------------------------------
# Publishing a watched file into the library
# ---------------------------------------------------------------------------


def _write_video(path, fourcc: str) -> None:
    colour = next(_watch_colours)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 5.0, (32, 24))
    if not writer.isOpened():
        pytest.skip(f"no OpenCV {fourcc} encoder in this environment")
    for index in range(3):
        writer.write(np.full((24, 32, 3), (colour * 7 + index * 40) % 256, np.uint8))
    writer.release()


@pytest.mark.parametrize(
    "ext, fourcc", [(".webm", "VP80"), (".mov", "mp4v"), (".mkv", "XVID")]
)
def test_a_watched_video_keeps_its_own_container(watch_server, tmp_path, ext, fourcc):
    root = tmp_path / "watch"
    root.mkdir()
    file_path = root / f"clip{ext}"
    _write_video(file_path, fourcc)
    data = file_path.read_bytes()

    result = _watch_pass(watch_server, file_path, root)

    (picture_id,) = result["imported_picture_ids"]
    picture = watch_server.vault.db.run_task(
        lambda session: session.get(Picture, picture_id)
    )
    assert picture.is_video
    assert picture.format == ext[1:].upper()
    assert picture.file_path.endswith(ext)
    published = ImageUtils.resolve_picture_path(
        watch_server.vault.image_root, picture.file_path
    )
    with open(published, "rb") as handle:
        assert handle.read() == data


def _rival_arrives_as_the_name_is_taken(monkeypatch, rival: bytes) -> list[str]:
    """Write *rival* at a picture's final name just before the import takes it."""
    arrived: list[str] = []

    def _wrap(real):
        def publish(src, dst, *args, **kwargs):
            if is_in_progress_file_name(
                os.path.basename(str(src))
            ) and not is_in_progress_file_name(os.path.basename(str(dst))):
                with open(dst, "xb") as handle:
                    handle.write(rival)
                arrived.append(str(dst))
            return real(src, dst, *args, **kwargs)

        return publish

    monkeypatch.setattr(os, "link", _wrap(os.link))
    monkeypatch.setattr(os, "replace", _wrap(os.replace))
    return arrived


def test_a_file_that_takes_the_name_while_publishing_is_never_replaced(
    watch_server, tmp_path, monkeypatch
):
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    data = _write_picture(file_path)
    rival = b"test-rival-bytes"
    arrived = _rival_arrives_as_the_name_is_taken(monkeypatch, rival)

    result = _watch_pass(watch_server, file_path, root)

    assert len(arrived) == 1, arrived
    with open(arrived[0], "rb") as handle:
        assert handle.read() == rival
    assert result["imported_picture_ids"] == []
    assert _picture_ids_with_bytes(watch_server, data) == []
    assert os.path.exists(file_path), "a file that did not import is kept"
    assert not [
        name
        for name in os.listdir(os.path.dirname(arrived[0]))
        if is_in_progress_file_name(name)
    ]


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot fsync a directory")
def test_a_published_picture_is_on_disk_before_and_after_it_takes_its_name(
    watch_server, tmp_path, monkeypatch
):
    """The bytes are flushed before the name points at them, the name after."""
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    _write_picture(file_path)
    events: list[tuple[str, tuple[int, int]]] = []
    real_fsync, real_link = os.fsync, os.link

    def fsync(fd):
        info = os.fstat(fd)
        events.append(("fsync", (info.st_dev, info.st_ino)))
        return real_fsync(fd)

    def link(src, dst, *args, **kwargs):
        result = real_link(src, dst, *args, **kwargs)
        events.append(("publish", (0, 0)))
        return result

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", link)

    result = _watch_pass(watch_server, file_path, root)

    (picture_id,) = result["imported_picture_ids"]
    picture = watch_server.vault.db.run_task(
        lambda session: session.get(Picture, picture_id)
    )
    published = ImageUtils.resolve_picture_path(
        watch_server.vault.image_root, picture.file_path
    )
    file_info, dir_info = os.stat(published), os.stat(os.path.dirname(published))
    file_key = (file_info.st_dev, file_info.st_ino)
    dir_key = (dir_info.st_dev, dir_info.st_ino)
    order = [event for event in events if event[1] in (file_key, dir_key, (0, 0))]
    publish_at = order.index(("publish", (0, 0)))
    assert ("fsync", file_key) in order[:publish_at], order
    assert ("fsync", dir_key) in order[publish_at:], order


@pytest.mark.skipif(os.name == "nt", reason="Windows enforces the rule itself")
def test_a_published_picture_is_fsynced_through_a_writable_handle(
    watch_server, tmp_path, monkeypatch
):
    """Windows refuses to fsync a read-only handle with EBADF, since
    FlushFileBuffers needs write access. Linux flushes one fine, so a read-only
    open of the in-progress file passed here and failed every import on
    Windows CI. A directory's handle is exempt: that flush is POSIX-only."""
    fcntl = pytest.importorskip("fcntl")
    root = tmp_path / "watch"
    file_path = root / "picture.png"
    _write_picture(file_path)
    file_modes: list[int] = []
    real_fsync = os.fsync

    def fsync(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            file_modes.append(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    result = _watch_pass(watch_server, file_path, root)

    assert len(result["imported_picture_ids"]) == 1, result
    assert file_modes, "the in-progress file was never flushed"
    assert os.O_RDONLY not in file_modes, file_modes


def test_an_import_folder_id_sqlite_cannot_store_is_a_422(watch_server):
    """``2**63`` overflowed the driver while binding (a 500); the largest
    storable id still reaches the handler, which answers 404."""
    with TestClient(watch_server.api) as client:
        response = client.post(
            f"{API_PREFIX}/login",
            json={"username": "testuser", "password": "testpassword"},
        )
        assert response.status_code == 200, response.text
        for folder_id, status in ((2**63, 422), (2**63 - 1, 404)):
            patched = client.patch(
                f"{API_PREFIX}/import-folders/{folder_id}", json={"label": "x"}
            )
            assert patched.status_code == status, patched.text
            deleted = client.delete(f"{API_PREFIX}/import-folders/{folder_id}")
            assert deleted.status_code == status, deleted.text


# ── A file published for a picture that is not saved, and the seen paths ──


def _library_files_with_bytes(server, data: bytes) -> list[str]:
    """Every file under the image root with the content *data*, row or not."""
    sha = ImageUtils.calculate_full_hash_from_bytes(data)
    found = []
    for dirpath, _dirs, names in os.walk(server.vault.image_root):
        for name in names:
            path = os.path.join(dirpath, name)
            if ImageUtils.calculate_full_hash_from_file_path(path) == sha:
                found.append(path)
    return found


class _RecordingFinder:
    """Stands in for the finder: records what the task hands back."""

    def __init__(self):
        self.handed_back: list[str] = []

    def discard_seen_paths(self, paths):
        self.handed_back.extend(paths)


def _watch_task(server, paths, root, finder, delete_after_import=False):
    return WatchFolderImportTask(
        server.vault.db,
        [
            {
                "file_path": str(path),
                "delete_after_import": delete_after_import,
                "import_source_folder": str(root),
            }
            for path in paths
        ],
        len(paths),
        finder=finder,
    )


def _fail_the_insert(monkeypatch, server):
    """Make the watch task's insert fail at its commit, as a full disk would."""
    database = server.vault.db
    real_run_task = database.run_task

    def run_task(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "insert_pictures":
            real_insert = func

            def func(session, *inner_args):
                def commit():
                    raise RuntimeError("test-insert-failure")

                session.commit = commit
                return real_insert(session, *inner_args)

        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(database, "run_task", run_task)


def test_a_watched_file_that_fails_after_its_copy_leaves_no_file_and_is_offered_again(
    watch_server, tmp_path, monkeypatch
):
    """The copy in the library goes with the failure, and the finder forgets
    the file so a later scan retries it; the rest of the pass imports."""
    root = tmp_path / "watch"
    failing, fine = root / "failing.png", root / "fine.png"
    failing_data = _write_picture(failing)
    fine_data = _write_picture(fine)
    real_attach = watch_folder_import_task.attach_sidecars

    def attach_sidecars(pic, file_path, **kwargs):
        if file_path == str(failing):
            raise OSError("test-sidecar-failure")
        return real_attach(pic, file_path, **kwargs)

    monkeypatch.setattr(watch_folder_import_task, "attach_sidecars", attach_sidecars)
    finder = _RecordingFinder()

    result = _watch_task(watch_server, [failing, fine], root, finder)._run_task()

    assert _library_files_with_bytes(watch_server, failing_data) == []
    assert finder.handed_back == [str(failing)]
    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, fine_data) == [picture_id]


def test_a_failed_watch_insert_removes_the_batch_files_and_offers_every_file_again(
    watch_server, tmp_path, monkeypatch
):
    root = tmp_path / "watch"
    paths = [root / "first.png", root / "second.png"]
    datas = [_write_picture(path) for path in paths]
    _fail_the_insert(monkeypatch, watch_server)
    finder = _RecordingFinder()
    task = _watch_task(watch_server, paths, root, finder, delete_after_import=True)

    with pytest.raises(RuntimeError, match="test-insert-failure"):
        task._run_task()

    for data in datas:
        assert _library_files_with_bytes(watch_server, data) == []
        assert _picture_ids_with_bytes(watch_server, data) == []
    assert sorted(finder.handed_back) == sorted(str(path) for path in paths)
    assert all(os.path.exists(path) for path in paths), "nothing imported, kept"


def test_a_watch_task_cancelled_before_it_ran_offers_its_files_again(
    watch_server, tmp_path
):
    root = tmp_path / "watch"
    paths = [root / "first.png", root / "second.png"]
    for path in paths:
        _write_picture(path)
    finder = _RecordingFinder()

    _watch_task(watch_server, paths, root, finder).on_cancel()

    assert sorted(finder.handed_back) == sorted(str(path) for path in paths)


def test_a_watch_pass_cancelled_midway_offers_the_files_it_did_not_reach_again(
    watch_server, tmp_path, monkeypatch
):
    """Cancelled once the first batch is saved: that file is imported and not
    handed back, the two the pass never reached are."""
    monkeypatch.setattr(watch_folder_import_task, "_IMPORT_BATCH", 1)
    root = tmp_path / "watch"
    paths = [root / "first.png", root / "second.png", root / "third.png"]
    first_data = _write_picture(paths[0])
    for path in paths[1:]:
        _write_picture(path)
    finder = _RecordingFinder()
    task = _watch_task(watch_server, paths, root, finder)
    _after_insert(monkeypatch, watch_server, task.on_cancel)

    result = task._run_task()

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, first_data) == [picture_id]
    assert sorted(finder.handed_back) == sorted(str(path) for path in paths[1:])


def test_a_watch_pass_cancelled_inside_a_batch_offers_the_rest_of_it_again(
    watch_server, tmp_path, monkeypatch
):
    root = tmp_path / "watch"
    paths = [root / "first.png", root / "second.png", root / "third.png"]
    first_data = _write_picture(paths[0])
    for path in paths[1:]:
        _write_picture(path)
    finder = _RecordingFinder()
    task = _watch_task(watch_server, paths, root, finder)
    real_create = ImageUtils.create_picture_from_file

    def create_then_cancel(*args, **kwargs):
        pic = real_create(*args, **kwargs)
        task.on_cancel()
        return pic

    monkeypatch.setattr(ImageUtils, "create_picture_from_file", create_then_cancel)

    result = task._run_task()

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, first_data) == [picture_id]
    assert sorted(finder.handed_back) == sorted(str(path) for path in paths[1:])


def test_identical_watched_files_in_one_pass_become_one_picture(watch_server, tmp_path):
    """The second file's content is the first's copy, which the library lookup
    cannot see before the batch's insert; it is skipped and kept on disk."""
    root = tmp_path / "watch"
    first, second = root / "first.png", root / "nested" / "second.png"
    data = _write_picture(first)
    os.makedirs(second.parent, exist_ok=True)
    shutil.copyfile(first, second)
    finder = _RecordingFinder()

    result = _watch_task(
        watch_server, [first, second], root, finder, delete_after_import=True
    )._run_task()

    (picture_id,) = result["imported_picture_ids"]
    assert _picture_ids_with_bytes(watch_server, data) == [picture_id]
    assert len(_library_files_with_bytes(watch_server, data)) == 1
    assert not os.path.exists(first)
    assert os.path.exists(second), "a skipped file is kept"
    assert finder.handed_back == []
