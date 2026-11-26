"""One unreadable file no longer sinks a one-shot ``POST /pictures/import``.

The one-shot route published each file before inserting any rows. A file that
could not be read as an image or a video raised out of the batch: the import
reported ``failed``, the files published before it stayed in the picture root
with no row, and the next root scan indexed them as new pictures without the
caption tags their upload carried. Now the bad file is counted as ``failed``
(as the staging path does), the rest of the batch imports, and a batch that
still stops before its insert commits removes the files it published.

The staging import (``PictureImportTask``) had the same orphan for a batch
whose insert failed; its failure contract is unchanged, and the batch's
published files are now removed too.
"""

import io
import itertools
import os
import shutil
import struct
import tempfile
import time
import uuid
import zipfile
import zlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session, select

from pixlstash.db_models import Picture, Tag
from pixlstash.db_models.folder_mapping_commit import (
    STATE_DEFERRED,
    FolderMappingCommit,
)
from pixlstash.routes.pictures._helpers import (
    _create_picture_imports,
    _parse_sidecar_tags,
)
from pixlstash.routes.pictures._import import _COPY_CHUNK_BYTES
from pixlstash.server import Server
from pixlstash.tasks import TaskType
from pixlstash.tasks import picture_import_task
from pixlstash.tasks.base_task import TASK_FAILURE_MESSAGE
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.media_files import SUPPORTED_MEDIA_EXTS
from tests.authz_guard import no_spa_fallback  # noqa: F401

API = "/api/v1"

pytestmark = pytest.mark.usefixtures("no_spa_fallback")

# A root scan run by hand would race these (tests/test_library_root_scan.py).
_CONFLICTING_FINDERS = (
    TaskType.THUMBNAIL_GENERATION,
    TaskType.REFERENCE_FOLDER_SCAN,
    TaskType.MISSING_FILE_PURGE,
    TaskType.TAGGER,
)

_SIDECAR_TEXT = "red apple, blue sky, wooden table"

# Every image in this module gets its own colour, so no two tests' uploads
# ever deduplicate against each other on the shared server.
_colours = itertools.count(1)


@pytest.fixture(scope="module")
def server():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "server-config.json")
        with Server(config_path) as srv:
            for task_type in _CONFLICTING_FINDERS:
                srv.vault._planner_work_finders.pop(task_type)
            srv.vault._work_planner.detach_finders(_CONFLICTING_FINDERS)

            # The root scan waits for the first import offer to be answered.
            def answer(session: Session):
                session.add(
                    FolderMappingCommit(
                        task_id="answered-by-test",
                        root_path=srv.vault.image_root,
                        mode="local_import",
                        state=STATE_DEFERRED,
                    )
                )
                session.commit()

            srv.vault.db.run_task(answer)
            yield srv


@pytest.fixture(scope="module")
def client(server):
    client = TestClient(server.api, raise_server_exceptions=True)
    r = client.post(
        f"{API}/login",
        json={"username": "owner", "password": "example-owner-password"},
    )
    assert r.status_code == 200, r.text
    return client


def _good_png() -> bytes:
    n = next(_colours)
    img = Image.new("RGB", (48, 48), color=(n % 256, (n * 7) % 256, (n * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bad_bytes(kind: str) -> bytes:
    if kind == "garbage":
        return b"not an image at all " * 200
    n = next(_colours)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(n % 256, 40, 90)).save(
        buf, format="JPEG", quality=95
    )
    data = buf.getvalue()
    return data[: len(data) // 2]


def _one_shot_import(client, files, timeout_s=30):
    r = client.post(
        f"{API}/pictures/import",
        files=[
            ("file", (name, data, "application/octet-stream")) for name, data in files
        ],
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/pictures/import/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.1)
    raise AssertionError(f"Import task {task_id} did not finish: {last}")


def _media_on_disk(server) -> set[str]:
    """Every media file and thumbnail under the picture root, root-relative."""
    root = server.vault.image_root
    found = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_MEDIA_EXTS | {".webp"}:
                found.add(os.path.relpath(os.path.join(dirpath, name), root))
    return found


def _files_of(server, pictures) -> set[str]:
    """The media file and thumbnail each picture owns, root-relative."""
    root = server.vault.image_root
    owned = set()
    for pic in pictures:
        owned.add(os.path.normpath(pic.file_path))
        owned.add(
            os.path.relpath(ImageUtils.get_thumbnail_path(root, pic.file_path), root)
        )
    return owned


def _pictures_named(server, names) -> list[Picture]:
    return server.vault.db.run_task(
        lambda s: list(
            s.exec(
                select(Picture)
                .where(Picture.original_file_name.in_(names))
                .order_by(Picture.id)
            ).all()
        )
    )


def _all_picture_ids(server) -> set[int]:
    return set(server.vault.db.run_task(lambda s: s.exec(select(Picture.id)).all()))


def _tags(server, picture_id) -> set[str]:
    rows = server.vault.db.run_task(
        lambda s: s.exec(select(Tag).where(Tag.picture_id == picture_id)).all()
    )
    return {t.tag for t in rows if not t.tag.startswith("__")}


def _run_root_scan_on_settled_files(server):
    """Backdate everything under the root past the scan's settle window, then
    scan it the way the planner would."""
    root = server.vault.image_root
    old = time.time() - 600
    for rel in _media_on_disk(server):
        os.utime(os.path.join(root, rel), (old, old))
    ReferenceFolderScanTask(
        database=server.vault.db,
        folder_id=None,
        folder_path=root,
        resolved_path=root,
    )._run_task()


def _staged_import(client, files, timeout_s=30):
    """Open a staging session, stream ``files`` in one request, commit, and
    wait for the background import to settle."""
    r = client.post(f"{API}/pictures/import/staging", json={})
    assert r.status_code == 200, r.text
    staging_id = r.json()["staging_id"]
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[("file", (name, data, "image/png")) for name, data in files],
    )
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/pictures/import/staging/{staging_id}/status")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["stage"] in ("completed", "failed"):
            return last
        time.sleep(0.1)
    raise AssertionError(f"Staging {staging_id} did not finish: {last}")


def _fail_nth_insert(server, monkeypatch, n, after_commit=False):
    """Make the staging import's ``n``-th batch insert (1-based) fail inside
    its own session: at its commit, so the rows are flushed and then rolled
    back, or just after the commit has gone through."""
    real_run_task = server.vault.db.run_task
    calls = {"insert_pictures": 0}

    def fail(*_args, **_kwargs):
        raise RuntimeError("example-injected failure in insert_pictures")

    def run_task(func, *args, **kwargs):
        if getattr(func, "__name__", "") != "insert_pictures":
            return real_run_task(func, *args, **kwargs)
        calls["insert_pictures"] += 1
        if calls["insert_pictures"] != n:
            return real_run_task(func, *args, **kwargs)
        if after_commit:
            real_run_task(func, *args, **kwargs)
            fail()

        def insert_pictures(session, *inner_args, **inner_kwargs):
            session.commit = fail
            return func(session, *inner_args, **inner_kwargs)

        return real_run_task(insert_pictures, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", run_task)


def _fail_db_task(server, monkeypatch, task_name):
    """Make the vault's DB raise when the import runs ``task_name``."""
    real_run_task = server.vault.db.run_task

    def run_task(func, *args, **kwargs):
        if getattr(func, "__name__", "") == task_name:
            raise RuntimeError(f"example-injected failure in {task_name}")
        return real_run_task(func, *args, **kwargs)

    monkeypatch.setattr(server.vault.db, "run_task", run_task)


@pytest.mark.parametrize("bad_kind", ["garbage", "truncated"])
def test_one_bad_file_is_counted_and_the_rest_of_the_batch_imports(
    server, client, bad_kind
):
    good_a = f"partial_{bad_kind}_a.png"
    bad = f"partial_{bad_kind}_bad.jpg"
    good_b = f"partial_{bad_kind}_b.png"
    on_disk_before = _media_on_disk(server)

    status = _one_shot_import(
        client,
        [
            (good_a, _good_png()),
            (f"partial_{bad_kind}_a.txt", _SIDECAR_TEXT.encode()),
            (bad, _bad_bytes(bad_kind)),
            (good_b, _good_png()),
        ],
    )

    assert status["status"] == "completed", status
    assert status["imported_count"] == 2, status
    assert status["failed_count"] == 1, status
    assert status["duplicate_count"] == 0, status
    assert status["scrapheaped_count"] == 0, status
    assert status["total"] == 3, status
    (failed,) = [r for r in status["results"] if r["status"] == "failed"]
    assert failed["file"] == bad, failed
    assert failed["picture_id"] is None, failed
    # A fixed sentence for the kind of failure; the decoder's own text stays
    # in the log.
    assert failed["error"] == "Not a picture or video this library can read.", failed

    new_pictures = _pictures_named(server, [good_a, bad, good_b])
    assert [p.original_file_name for p in new_pictures] == [good_a, good_b]
    assert {r["picture_id"] for r in status["results"] if r["status"] == "success"} == {
        p.id for p in new_pictures
    }
    # The upload's caption landed on the picture it came with.
    assert set(_parse_sidecar_tags(_SIDECAR_TEXT)) <= _tags(server, new_pictures[0].id)

    # The only files this import left behind belong to its two pictures.
    assert _media_on_disk(server) - on_disk_before == _files_of(server, new_pictures)

    # ...so a later root scan finds nothing new to index.
    ids_before_scan = _all_picture_ids(server)
    _run_root_scan_on_settled_files(server)
    assert _all_picture_ids(server) == ids_before_scan


def test_an_import_whose_every_file_fails_reports_failed(server, client):
    bad = "only_file_bad.jpg"
    on_disk_before = _media_on_disk(server)
    ids_before = _all_picture_ids(server)

    status = _one_shot_import(client, [(bad, _bad_bytes("garbage"))])

    # One file per request is how the ComfyUI node uploads; it raises on
    # ``failed`` with this error, so it must still see one, naming the file.
    assert status["status"] == "failed", status
    assert status["error"] == (
        f"None of the 1 file(s) could be imported: {bad}: "
        "Not a picture or video this library can read."
    ), status
    assert _all_picture_ids(server) == ids_before
    assert _media_on_disk(server) == on_disk_before


def _import_failures_logged(caplog) -> list[str]:
    """The exception text of every one-shot import failure logged with its
    traceback, which the import status no longer carries."""
    return [
        str(r.exc_info[1])
        for r in caplog.records
        if r.name == "pixlstash.routes.pictures._import" and r.exc_info
    ]


@pytest.mark.parametrize(
    ("raised", "reason"),
    [
        (
            PermissionError(13, "test-denied writing /home/me/library/x.png"),
            "The file could not be written to the library.",
        ),
        (
            RuntimeError("test-reader-bug at /home/me/library/x.png"),
            "The file could not be imported; the server log has the details.",
        ),
    ],
    ids=["oserror", "other"],
)
def test_a_file_that_fails_is_given_the_sentence_for_its_kind(
    server, client, monkeypatch, caplog, raised, reason
):
    """The reason is chosen by the exception's class, never built from its
    text, and the exception is logged with the file's name and traceback.

    Wrong if the status carries the exception's text (a path here), or if the
    log lacks the traceback.
    """
    name = f"kind_{type(raised).__name__}.png"

    def create_picture_from_file(**_kwargs):
        raise raised

    monkeypatch.setattr(
        ImageUtils, "create_picture_from_file", staticmethod(create_picture_from_file)
    )

    with caplog.at_level("WARNING", logger="pixlstash"):
        status = _one_shot_import(client, [(name, _good_png())])

    assert status["status"] == "failed", status
    assert status["error"] == (
        f"None of the 1 file(s) could be imported: {name}: {reason}"
    ), status
    assert [
        r.exc_info[1]
        for r in caplog.records
        if r.name == "pixlstash.routes.pictures._helpers"
        and r.exc_info
        and name in r.getMessage()
    ] == [raised]


def test_an_insert_that_fails_removes_the_files_the_batch_published(
    server, client, monkeypatch, caplog
):
    on_disk_before = _media_on_disk(server)
    ids_before = _all_picture_ids(server)
    _fail_db_task(server, monkeypatch, "import_task")

    with caplog.at_level("WARNING", logger="pixlstash"):
        status = _one_shot_import(
            client,
            [("insert_fails_a.png", _good_png()), ("insert_fails_b.png", _good_png())],
        )

    # The status says the import failed; what failed is in the log.
    assert status["status"] == "failed", status
    assert status["error"] == TASK_FAILURE_MESSAGE, status
    assert _import_failures_logged(caplog) == [
        "example-injected failure in import_task"
    ]
    assert _all_picture_ids(server) == ids_before
    # Neither file nor either thumbnail is left for a root scan to pick up.
    assert _media_on_disk(server) == on_disk_before


def test_a_failure_after_the_insert_keeps_the_pictures_files(
    server, client, monkeypatch
):
    names = ["after_insert_a.png", "after_insert_b.png"]
    on_disk_before = _media_on_disk(server)
    _fail_db_task(server, monkeypatch, "apply_import_context")

    status = _one_shot_import(client, [(name, _good_png()) for name in names])

    assert status["status"] == "failed", status
    saved = _pictures_named(server, names)
    assert [p.original_file_name for p in saved] == names
    # The rows committed, so their files are theirs: removing them would leave
    # two pictures pointing at nothing.
    assert _media_on_disk(server) - on_disk_before == _files_of(server, saved)


def test_an_engine_that_cannot_start_fails_no_import_and_is_retried(
    server, client, monkeypatch
):
    """The pictures are saved either way; the next import builds the engine again."""
    attempts: list[int] = []

    def ensure_ready():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("test-engine-build-failure")

    monkeypatch.setattr(server.vault, "ensure_ready", ensure_ready)

    first = _one_shot_import(client, [("engine_down.png", _good_png())])
    second = _one_shot_import(client, [("engine_back.png", _good_png())])

    assert first["status"] == "completed", first
    assert first["imported_count"] == 1, first
    assert first["engine_error"] == (
        "The model engine could not start, so tagging and other model work "
        "wait until it can; the server log has the details."
    ), first
    (saved,) = _pictures_named(server, ["engine_down.png"])
    assert saved.imported_at is not None, "the import finished its pictures"
    assert second["status"] == "completed", second
    assert second["imported_count"] == 1, second
    assert second["engine_error"] is None, second
    assert len(attempts) == 2, "the second import tried the engine again"


def test_an_import_whose_free_space_cannot_be_read_is_refused(
    server, client, monkeypatch, caplog
):
    """An unreadable library volume is a 503, not a 500. Its detail is the
    server's sentence; the OSError, which names the path, is logged with its
    traceback."""
    real_disk_usage = shutil.disk_usage
    image_root = os.path.realpath(server.vault.image_root)

    def disk_usage(path):
        if os.path.realpath(path) == image_root:
            raise PermissionError(13, "Permission denied", path)
        return real_disk_usage(path)

    before = _all_picture_ids(server)
    with monkeypatch.context() as patch, caplog.at_level("ERROR"):
        patch.setattr(shutil, "disk_usage", disk_usage)
        r = client.post(
            f"{API}/pictures/import",
            files=[("file", ("no_space_check.png", _good_png(), "image/png"))],
        )

    assert r.status_code == 503, r.text
    assert r.json()["detail"] == (
        "Cannot import: could not check free space on the library volume."
    )
    assert any(
        record.exc_info and "cannot check free space" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]
    assert _all_picture_ids(server) == before
    # With the volume readable again the same upload imports.
    result = _one_shot_import(client, [("no_space_check.png", _good_png())])
    assert result["status"] == "completed", result
    assert result["imported_count"] == 1, result


def _zip_with_a_corrupt_entry(
    prefix: str, kind: str = "deflate", bad_ext: str = ".png"
) -> bytes:
    """A zip of ``<prefix>_a.png``, ``<prefix>_bad<bad_ext>`` and
    ``<prefix>_b.png`` whose middle entry cannot be read.

    ``deflate``: the entry's deflate stream is overwritten with garbage, so
    reading it raises ``zlib.error``. ``crc``: the zip is stored, not
    compressed, and 16 of the entry's bytes are flipped, so it reads to the end
    and then raises ``BadZipFile`` on its CRC. A ``.txt`` middle entry is a
    caption, which the import reads by another path than a picture.
    """
    compression = zipfile.ZIP_DEFLATED if kind == "deflate" else zipfile.ZIP_STORED
    bad_name = f"{prefix}_bad{bad_ext}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as archive:
        archive.writestr(f"{prefix}_a.png", _good_png())
        archive.writestr(
            bad_name, _good_png() if bad_ext == ".png" else b"test-caption " * 40
        )
        archive.writestr(f"{prefix}_b.png", _good_png())
    data = bytearray(buf.getvalue())
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as archive:
        info = archive.getinfo(bad_name)
    name_len, extra_len = struct.unpack_from("<HH", data, info.header_offset + 26)
    start = info.header_offset + 30 + name_len + extra_len
    if kind == "deflate":
        data[start : start + info.compress_size] = b"\xff" * info.compress_size
        expected = zlib.error
    else:
        middle = start + info.compress_size // 2
        data[middle : middle + 16] = bytes(b ^ 0xFF for b in data[middle : middle + 16])
        expected = zipfile.BadZipFile
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as archive:
        with pytest.raises(expected):
            archive.read(bad_name)
    return bytes(data)


@pytest.mark.parametrize("kind", ["deflate", "crc"])
def test_a_zip_entry_that_cannot_be_decompressed_fails_alone(server, client, kind):
    """One corrupt entry is a failed file; the zip's other entries import."""
    status = _one_shot_import(
        client,
        [
            (
                "test-corrupt-oneshot.zip",
                _zip_with_a_corrupt_entry(f"zipone{kind}", kind),
            )
        ],
    )

    assert status["status"] == "completed", status
    assert status["imported_count"] == 2, status
    assert status["failed_count"] == 1, status
    assert status["total"] == 3, status
    (failed,) = [r for r in status["results"] if r["status"] == "failed"]
    assert failed["file"] == f"test-corrupt-oneshot.zip/zipone{kind}_bad.png", failed
    assert failed["error"] == (
        "This zip entry is damaged and cannot be decompressed."
    ), failed
    saved = _pictures_named(
        server, [f"zipone{kind}_{suffix}.png" for suffix in ("a", "bad", "b")]
    )
    assert [p.original_file_name for p in saved] == [
        f"zipone{kind}_a.png",
        f"zipone{kind}_b.png",
    ]


@pytest.mark.parametrize("bad_ext", [".png", ".txt"], ids=["picture", "caption"])
@pytest.mark.parametrize("kind", ["deflate", "crc"])
def test_a_staged_zip_entry_that_cannot_be_decompressed_is_skipped_alone(
    server, client, kind, bad_ext
):
    """The entries on both sides of the corrupt one are staged and import, and
    only the corrupt entry is reported skipped, by name: a picture or a
    caption."""
    prefix = f"zipstaged{kind}{bad_ext[1:]}"
    r = client.post(f"{API}/pictures/import/staging", json={})
    assert r.status_code == 200, r.text
    staging_id = r.json()["staging_id"]
    r = client.post(
        f"{API}/pictures/import/staging/{staging_id}/files",
        files=[
            (
                "file",
                (
                    "test-corrupt-staged.zip",
                    _zip_with_a_corrupt_entry(prefix, kind, bad_ext),
                    "application/zip",
                ),
            )
        ],
    )

    assert r.status_code == 200, r.text
    assert r.json()["skipped"] == [f"test-corrupt-staged.zip/{prefix}_bad{bad_ext}"]
    assert r.json()["staged"] == 2, r.text
    r = client.post(f"{API}/pictures/import/staging/{staging_id}/commit")
    assert r.status_code == 200, r.text
    deadline = time.time() + 30
    while True:
        status = client.get(f"{API}/pictures/import/staging/{staging_id}/status").json()
        if status["stage"] in ("completed", "failed") or time.time() > deadline:
            break
        time.sleep(0.1)
    assert status["stage"] == "completed", status
    saved = _pictures_named(
        server, [f"{prefix}_{suffix}.png" for suffix in ("a", "bad", "b")]
    )
    assert [p.original_file_name for p in saved] == [
        f"{prefix}_a.png",
        f"{prefix}_b.png",
    ]


def test_the_helper_without_a_failure_callback_removes_what_it_published(server):
    """Library move and the e2e seed call the helper without ``on_file_failed``
    and expect it to raise; it must not leave the files it wrote first."""
    on_disk_before = _media_on_disk(server)

    with pytest.raises(ValueError):
        _create_picture_imports(
            SimpleNamespace(vault=server.vault),
            [
                (_good_png(), ".png", "helper_good.png"),
                (_bad_bytes("garbage"), ".jpg", "helper_bad.jpg"),
            ],
            server.vault.image_root,
        )

    assert _media_on_disk(server) == on_disk_before


def _task_failures(caplog) -> list[str]:
    """The exception text of every task failure logged, which the import
    status no longer carries (it answers ``TASK_FAILURE_MESSAGE``)."""
    return [
        str(r.exc_info[1])
        for r in caplog.records
        if r.name == "pixlstash.tasks.base_task" and r.exc_info
    ]


def test_a_staged_batch_whose_insert_fails_leaves_no_files_behind(
    server, client, monkeypatch, caplog
):
    names = ["staged_ok_a.png", "staged_ok_b.png", "staged_lost_c.png"]
    on_disk_before = _media_on_disk(server)
    # Two files a batch: the first batch commits, the second's insert fails.
    monkeypatch.setattr(picture_import_task, "_DEDUP_BATCH", 2)
    _fail_nth_insert(server, monkeypatch, 2)

    with caplog.at_level("WARNING", logger="pixlstash"):
        status = _staged_import(client, [(name, _good_png()) for name in names])

    # The task's own failure contract: it stops and says how many pictures it
    # added before the failure, never the cause's text, which is logged.
    assert status["stage"] == "failed", status
    assert status["error"] == "The import stopped after 2 new picture(s) were added."
    assert any(
        "example-injected failure in insert_pictures" in r.getMessage() and r.exc_info
        for r in caplog.records
        if r.name == "pixlstash.tasks.picture_import_task"
    )
    saved = _pictures_named(server, names)
    assert [p.original_file_name for p in saved] == names[:2]
    # Only the committed batch's files remain.
    assert _media_on_disk(server) - on_disk_before == _files_of(server, saved)

    # ...so a later root scan finds nothing new to index.
    monkeypatch.undo()
    ids_before_scan = _all_picture_ids(server)
    _run_root_scan_on_settled_files(server)
    assert _all_picture_ids(server) == ids_before_scan


def test_a_staged_insert_that_committed_before_failing_keeps_its_files(
    server, client, monkeypatch, caplog
):
    names = ["staged_committed_a.png", "staged_committed_b.png"]
    on_disk_before = _media_on_disk(server)
    _fail_nth_insert(server, monkeypatch, 1, after_commit=True)

    with caplog.at_level("WARNING", logger="pixlstash"):
        status = _staged_import(client, [(name, _good_png()) for name in names])

    # The failure logged is the one that happened, not one raised while
    # handling it.
    assert status["stage"] == "failed", status
    assert status["error"] == TASK_FAILURE_MESSAGE, status
    assert _task_failures(caplog) == ["example-injected failure in insert_pictures"]
    saved = _pictures_named(server, names)
    assert [p.original_file_name for p in saved] == names
    # The rows are in the library, so their files must stay.
    assert _media_on_disk(server) - on_disk_before == _files_of(server, saved)


def test_a_root_scan_sweeps_only_the_in_progress_files_a_dead_import_left(server):
    """A process that died mid-import leaves its hidden in-progress file."""
    root = server.vault.image_root
    sub = os.path.join(root, "test-sweep")
    os.makedirs(sub, exist_ok=True)
    day_and_more = time.time() - 25 * 3600

    def _write(path, mtime):
        with open(path, "wb") as handle:
            handle.write(b"test-in-progress-bytes")
        os.utime(path, (mtime, mtime))
        return path

    stale = [
        _write(os.path.join(root, f".{uuid.uuid4().hex}.part.png"), day_and_more),
        _write(os.path.join(sub, f".{uuid.uuid4().hex}.part.webm"), day_and_more),
    ]
    running = _write(os.path.join(root, f".{uuid.uuid4().hex}.part.png"), time.time())
    lookalike = _write(os.path.join(root, ".test-notes.part"), day_and_more)

    # Scanned as is: the settle-window helper would backdate these files too.
    ReferenceFolderScanTask(
        database=server.vault.db,
        folder_id=None,
        folder_path=root,
        resolved_path=root,
    )._run_task()

    assert [path for path in stale if os.path.lexists(path)] == []
    assert os.path.exists(running), "an import still running keeps its file"
    assert os.path.exists(lookalike), "only an import's own name is swept"


def test_a_root_scan_leaves_a_copy_of_a_library_picture_untracked(
    server, client, caplog
):
    """An untracked file in the image root that a picture already holds is a
    copy: the scan logs it with the picture's id, leaves it, imports nothing."""
    status = _one_shot_import(client, [("root_copy_src.png", _good_png())])
    assert status["imported_count"] == 1, status
    (source,) = _pictures_named(server, ["root_copy_src.png"])
    root = server.vault.image_root
    with open(ImageUtils.resolve_picture_path(root, source.file_path), "rb") as fh:
        data = fh.read()
    copy_path = os.path.join(root, "test-root-copy.png")
    new_path = os.path.join(root, "test-root-new.png")
    with open(copy_path, "wb") as fh:
        fh.write(data)
    with open(new_path, "wb") as fh:
        fh.write(_good_png())
    ids_before = _all_picture_ids(server)

    with caplog.at_level("INFO", logger="pixlstash"):
        _run_root_scan_on_settled_files(server)

    added = _all_picture_ids(server) - ids_before
    (new_picture,) = _pictures_named(server, ["test-root-new.png"])
    assert added == {new_picture.id}, "only the new file is imported"
    with open(copy_path, "rb") as fh:
        assert fh.read() == data, "the copy is left as it was"
    skips = [
        r.getMessage()
        for r in caplog.records
        if "Root scan: not importing" in r.getMessage()
    ]
    assert skips == [
        f"Root scan: not importing {copy_path}: its content is already picture "
        f"{source.id}. The file is left where it is."
    ]
    os.remove(copy_path)


def test_identical_new_files_in_one_root_scan_become_one_picture(server, caplog):
    """Two untracked files with the same new content: the lookup cannot see
    the first one's row before the scan's insert, so the second is matched
    against the first, logged and left; the next scan finds the picture."""
    root = server.vault.image_root
    data = _good_png()
    paths = [
        os.path.join(root, "test-root-twin-a.png"),
        os.path.join(root, "test-root-twin-dir", "test-root-twin-b.png"),
    ]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
    ids_before = _all_picture_ids(server)

    with caplog.at_level("INFO", logger="pixlstash"):
        _run_root_scan_on_settled_files(server)

    added = _all_picture_ids(server) - ids_before
    (picture,) = _pictures_named(server, [os.path.basename(path) for path in paths])
    assert added == {picture.id}
    imported = ImageUtils.resolve_picture_path(root, picture.file_path)
    (left,) = [path for path in paths if os.path.abspath(path) != imported]
    assert all(os.path.exists(path) for path in paths), "both files stay"
    skips = [
        r.getMessage()
        for r in caplog.records
        if "Root scan: not importing" in r.getMessage()
    ]
    assert skips == [
        f"Root scan: not importing {left}: it has the content of {imported}, "
        "imported in the same scan. The file is left where it is."
    ]

    _run_root_scan_on_settled_files(server)

    assert _all_picture_ids(server) - ids_before == {picture.id}
    os.remove(left)


def test_a_root_scan_reads_an_unchanged_copy_once(server, client, monkeypatch):
    """A copy stays in the root, so every scan finds it again. Its full hash
    is taken once while nothing writes it; a change that leaves the size and
    mtime alone still shows in the ctime, and the file is read again. A
    fingerprint taken while the ctime is too recent to trust is not kept."""
    status = _one_shot_import(client, [("root_memo_src.png", _good_png())])
    assert status["imported_count"] == 1, status
    (source,) = _pictures_named(server, ["root_memo_src.png"])
    root = server.vault.image_root
    with open(ImageUtils.resolve_picture_path(root, source.file_path), "rb") as fh:
        data = fh.read()
    copy_name = "test-root-memo-copy.png"
    copy_path = os.path.join(root, copy_name)
    with open(copy_path, "wb") as fh:
        fh.write(data)
    old_ns = time.time_ns() - 600 * 10**9
    os.utime(copy_path, ns=(old_ns, old_ns))
    real_full_hash = ImageUtils.calculate_full_hash_from_file_path
    hashed: list[str] = []

    def _counting_full_hash(path):
        hashed.append(path)
        return real_full_hash(path)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_file_path", _counting_full_hash
    )

    def _scan():
        ReferenceFolderScanTask(
            database=server.vault.db,
            folder_id=None,
            folder_path=root,
            resolved_path=root,
        )._run_task()
        assert _pictures_named(server, [copy_name]) == [], "the copy was imported"

    racy = "pixlstash.tasks.reference_folder_scan_task._RACY_STAT_S"
    try:
        # The utime above just set the ctime: with the window wider than any
        # scan, that ctime is always too recent to keep.
        monkeypatch.setattr(racy, 3600.0)
        _scan()
        _scan()
        assert hashed.count(copy_path) == 2, (
            "a fingerprint taken just after a write was kept"
        )

        monkeypatch.setattr(racy, 0.0)
        _scan()
        _scan()
        assert hashed.count(copy_path) == 3, (
            "the second scan read the unchanged copy again"
        )

        # A write that keeps the size and puts the mtime back leaves only the
        # ctime moved; a mode change moves just that, and the file stays a copy.
        mode = os.stat(copy_path).st_mode
        os.chmod(copy_path, mode | 0o100)
        os.chmod(copy_path, mode)
        _scan()
        assert hashed.count(copy_path) == 4, (
            "a file whose ctime moved was not read again"
        )
    finally:
        os.remove(copy_path)


def _noise_png(side: int) -> bytes:
    """A PNG that does not compress, so it spans several copy chunks."""
    img = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    return buf.getvalue()


def test_a_one_shot_upload_is_spooled_in_chunks_never_read_whole(
    server, client, monkeypatch
):
    """Neither a plain upload nor a zip entry is read into memory whole: both
    are copied to disk a chunk at a time, and the copies are gone once the
    import has finished."""
    plain = _noise_png(1100)
    zipped = _noise_png(1100)
    assert min(len(plain), len(zipped)) > 2 * _COPY_CHUNK_BYTES
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("spool_zipped.png", zipped)
    whole_reads: list[tuple[str, int]] = []

    def _recording(cls):
        real = cls.read

        def read(self, *args):
            data = real(self, *args)
            if len(data) > _COPY_CHUNK_BYTES:
                whole_reads.append((cls.__name__, len(data)))
            return data

        monkeypatch.setattr(cls, "read", read)

    _recording(tempfile.SpooledTemporaryFile)
    _recording(zipfile.ZipExtFile)

    status = _one_shot_import(
        client,
        [("spool_plain.png", plain), ("test-spool.zip", archive.getvalue())],
    )
    monkeypatch.undo()

    assert whole_reads == []
    assert status["imported_count"] == 2, status
    root = server.vault.image_root
    for picture, data in zip(
        _pictures_named(server, ["spool_plain.png", "spool_zipped.png"]),
        [plain, zipped],
    ):
        with open(ImageUtils.resolve_picture_path(root, picture.file_path), "rb") as fh:
            assert fh.read() == data
    staging = os.path.join(root, ".staging")
    leftovers = os.listdir(staging) if os.path.isdir(staging) else []
    assert [name for name in leftovers if name.startswith("import-")] == []
