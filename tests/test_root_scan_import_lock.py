"""A root scan does not hold the import lock while it hashes and builds a chunk.

Every import holds ``import_dedup_service.serialised_import()`` from its match
lookup to the insert of the pictures that lookup decided are new, so two
overlapping imports of one new file add it once. The root scan held it across
the whole chunk: the copy check, the hashing and the thumbnail rendering, which
take seconds, while every upload and watch-folder import waited. Now the copy
check runs before the build and outside the lock, and the scan makes the lookup
again under the lock, right before its insert, so content an import inserted
while the scan read the file is still not indexed twice.
"""

import io
import itertools
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session, select

from pixlstash.db_models import Picture
from pixlstash.db_models.folder_mapping_commit import (
    STATE_DEFERRED,
    FolderMappingCommit,
)
from pixlstash.server import Server
from pixlstash.services import import_dedup_service
from pixlstash.tasks import TaskType, reference_folder_scan_task
from pixlstash.tasks.reference_folder_scan_task import ReferenceFolderScanTask
from pixlstash.utils.image_processing.image_utils import ImageUtils
from pixlstash.utils.media_files import SUPPORTED_MEDIA_EXTS
from tests.authz_guard import no_spa_fallback  # noqa: F401

API = "/api/v1"

pytestmark = pytest.mark.usefixtures("no_spa_fallback")

# A root scan run by the planner would race the one each test runs by hand.
_CONFLICTING_FINDERS = (
    TaskType.THUMBNAIL_GENERATION,
    TaskType.REFERENCE_FOLDER_SCAN,
    TaskType.MISSING_FILE_PURGE,
    TaskType.TAGGER,
)

# Every image gets its own colour, so no two tests' files share content.
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


def _png() -> bytes:
    n = next(_colours)
    img = Image.new("RGB", (48, 48), color=(n % 256, (n * 11) % 256, (n * 5) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bmp() -> bytes:
    """An uncompressed image: any two are the same size, byte for byte."""
    n = next(_colours)
    img = Image.new("RGB", (48, 48), color=(n % 256, (n * 11) % 256, (n * 5) % 256))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _write(path: str, data: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _one_shot_import(client, name: str, data: bytes, timeout_s: float = 30) -> dict:
    r = client.post(
        f"{API}/pictures/import",
        files=[("file", (name, data, "application/octet-stream"))],
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"{API}/pictures/import/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"Import task {task_id} did not finish")


def _scan_settled_root(server) -> None:
    """Backdate every media file past the settle window, then scan the root."""
    root = server.vault.image_root
    old = time.time() - 600
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_MEDIA_EXTS:
                os.utime(os.path.join(dirpath, name), (old, old))
    ReferenceFolderScanTask(
        database=server.vault.db,
        folder_id=None,
        folder_path=root,
        resolved_path=root,
    )._run_task()


def _library_picture(server, data: bytes, pixel_sha: str) -> tuple[int, str]:
    """Add a picture with *data* to the library: its id and file's real path."""
    root = server.vault.image_root
    picture = ImageUtils.create_picture_from_bytes(root, data, pixel_sha=pixel_sha)

    def insert(session: Session) -> int:
        session.add(picture)
        session.commit()
        return int(picture.id)

    picture_id = server.vault.db.run_task(insert)
    real = os.path.realpath(ImageUtils.resolve_picture_path(root, picture.file_path))
    return picture_id, real


def _colliding(monkeypatch, sampled: str) -> None:
    """Give every file the scan hashes the sampled hash *sampled*.

    The files the test writes were just backdated, which sets their ctime to
    now, so the stat-key rule for a racy stat would trust none of them yet;
    the tests below are about what happens when the stat key holds.
    """
    monkeypatch.setattr(
        ImageUtils,
        "calculate_hash_from_file_path",
        staticmethod(lambda *_args, **_kwargs: sampled),
    )
    monkeypatch.setattr(reference_folder_scan_task, "_RACY_STAT_S", 0.0)


def _full_hash_reads(monkeypatch, fail_once: str | None = None) -> dict[str, list]:
    """Per real path, whether the import lock was held at each full read.

    *fail_once*: a real path whose first read raises ``OSError``.
    """
    real_full_hash = ImageUtils.calculate_full_hash_from_file_path
    reads: dict[str, list] = {}

    def spy(path):
        real = os.path.realpath(path)
        reads.setdefault(real, []).append(import_dedup_service._IMPORT_LOCK.locked())
        if real == fail_once and len(reads[real]) == 1:
            raise OSError("test-unreadable-once")
        return real_full_hash(path)

    monkeypatch.setattr(
        ImageUtils, "calculate_full_hash_from_file_path", staticmethod(spy)
    )
    return reads


def _pictures_named(server, name: str) -> list[Picture]:
    return server.vault.db.run_task(
        lambda s: list(
            s.exec(select(Picture).where(Picture.original_file_name == name)).all()
        )
    )


def _ids_at(server, stored_path: str) -> list[int]:
    return server.vault.db.run_task(
        lambda s: list(
            s.exec(select(Picture.id).where(Picture.file_path == stored_path)).all()
        )
    )


def _skips(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "Root scan: not importing" in r.getMessage()
    ]


def test_a_root_scan_hashes_and_builds_without_holding_the_import_lock(
    server, monkeypatch
):
    """The copy check and the build run with the lock free; the lookup made
    again and the insert run holding it, as every import's do."""
    path = _write(os.path.join(server.vault.image_root, "test-unlocked.png"), _png())
    held: dict[str, set[bool]] = {}

    def _record(name: str) -> None:
        real = getattr(ReferenceFolderScanTask, name)

        def wrapper(self, *args, **kwargs):
            locked = import_dedup_service._IMPORT_LOCK.locked()
            held.setdefault(name, set()).add(locked)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(ReferenceFolderScanTask, name, wrapper)

    for name in (
        "_without_library_copies",
        "_build_picture_chunk",
        "_without_imported_meanwhile",
        "_insert_pictures",
    ):
        _record(name)

    try:
        _scan_settled_root(server)

        assert len(_ids_at(server, "test-unlocked.png")) == 1
        assert held == {
            "_without_library_copies": {False},
            "_build_picture_chunk": {False},
            "_without_imported_meanwhile": {True},
            "_insert_pictures": {True},
        }
    finally:
        os.remove(path)


def test_content_imported_while_a_root_scan_builds_it_is_indexed_once(
    server, client, monkeypatch, caplog
):
    """An upload of the same content lands after the copy check and before
    the insert: the scan finds it under the lock, leaves the file where it is,
    and removes the thumbnail it built for a path nothing indexes."""
    root = server.vault.image_root
    data = _png()
    path = _write(os.path.join(root, "test-raced.png"), data)
    thumb = ImageUtils.get_thumbnail_path(root, "test-raced.png")
    real_build = ReferenceFolderScanTask._build_picture_chunk
    uploads: list[dict] = []

    def _build_then_upload(self, file_paths, folder_id):
        built = real_build(self, file_paths, folder_id)
        if path in file_paths:
            assert os.path.exists(thumb), "the build wrote no thumbnail to remove"
            uploads.append(_one_shot_import(client, "test-raced-upload.png", data))
        return built

    monkeypatch.setattr(
        ReferenceFolderScanTask, "_build_picture_chunk", _build_then_upload
    )

    try:
        with caplog.at_level("INFO", logger="pixlstash"):
            _scan_settled_root(server)

        assert [u["imported_count"] for u in uploads] == [1], uploads
        (upload,) = _pictures_named(server, "test-raced-upload.png")
        assert _ids_at(server, "test-raced.png") == [], (
            "the file was indexed beside the picture uploaded while the scan built it"
        )
        with open(path, "rb") as fh:
            assert fh.read() == data, "the file is left as it was"
        assert not os.path.lexists(thumb), "the thumbnail of an unindexed path was left"
        assert _skips(caplog) == [
            f"Root scan: not importing {path}: picture {upload.id} already holds its "
            "content. The file is left where it is."
        ]
    finally:
        os.remove(path)


def test_a_picture_indexed_at_the_same_path_meanwhile_keeps_its_thumbnail(
    server, monkeypatch, caplog
):
    """Another writer indexed this very file while the scan built it: the
    scan drops its build, and the thumbnail at that path is the other row's."""
    root = server.vault.image_root
    path = _write(os.path.join(root, "test-same-path.png"), _png())
    thumb = ImageUtils.get_thumbnail_path(root, "test-same-path.png")
    real_build = ReferenceFolderScanTask._build_picture_chunk
    rows: list[int] = []

    def _build_then_index(self, file_paths, folder_id):
        built = real_build(self, file_paths, folder_id)
        for pic in built:
            if pic.file_path != "test-same-path.png":
                continue
            other = Picture(
                file_path=pic.file_path,
                pixel_sha=pic.pixel_sha,
                size_bytes=pic.size_bytes,
                original_file_name="test-same-path-other.png",
            )

            def insert(session: Session) -> int:
                session.add(other)
                session.commit()
                return int(other.id)

            rows.append(server.vault.db.run_task(insert))
        return built

    monkeypatch.setattr(
        ReferenceFolderScanTask, "_build_picture_chunk", _build_then_index
    )

    try:
        with caplog.at_level("INFO", logger="pixlstash"):
            _scan_settled_root(server)

        assert len(rows) == 1
        assert _ids_at(server, "test-same-path.png") == rows, (
            "the scan indexed the path a second time"
        )
        assert os.path.exists(thumb), "the other row's thumbnail was removed"
        assert _skips(caplog) == [
            f"Root scan: not importing {path}: picture {rows[0]} already holds its "
            "content. The file is left where it is."
        ]
    finally:
        os.remove(path)


def test_a_collision_compared_before_the_lock_costs_no_read_under_it(
    server, monkeypatch
):
    """A picture the library held before, sharing a root file's sampled hash
    and size, is compared with the file once, before the import lock. The
    look again under the lock knows it was, and reads neither file again."""
    root = server.vault.image_root
    there, new = _bmp(), _bmp()
    assert len(there) == len(new)
    _id, candidate = _library_picture(server, there, "forced-root-compared-collision")
    path = _write(os.path.join(root, "test-compared-collision.bmp"), new)
    _colliding(monkeypatch, "forced-root-compared-collision")
    reads = _full_hash_reads(monkeypatch)

    try:
        _scan_settled_root(server)

        assert len(_ids_at(server, "test-compared-collision.bmp")) == 1
        # Compared before the lock, and not again under it.
        assert reads == {candidate: [False], os.path.realpath(path): [False]}
    finally:
        os.remove(path)


def test_a_file_the_look_before_the_lock_could_not_hash_is_compared_under_it(
    server, monkeypatch, caplog
):
    """What the look before the lock compared is per file: another file of
    the same sampled hash and size, compared with a picture there, does not
    excuse a file that could not be read then from that picture."""
    root = server.vault.image_root
    there, other = _bmp(), _bmp()
    candidate_id, _path = _library_picture(
        server, there, "forced-root-unhashed-collision"
    )
    copy = _write(os.path.join(root, "test-compared-unhashed.bmp"), there)
    second = _write(os.path.join(root, "test-compared-unhashed-other.bmp"), other)
    _colliding(monkeypatch, "forced-root-unhashed-collision")
    _full_hash_reads(monkeypatch, fail_once=os.path.realpath(copy))

    try:
        with caplog.at_level("INFO", logger="pixlstash"):
            _scan_settled_root(server)

        assert _ids_at(server, "test-compared-unhashed.bmp") == [], (
            "a copy of a picture was indexed because another file was compared"
        )
        assert len(_ids_at(server, "test-compared-unhashed-other.bmp")) == 1
        assert _skips(caplog) == [
            f"Root scan: not importing {copy}: picture {candidate_id} already holds "
            "its content. The file is left where it is."
        ]
    finally:
        os.remove(copy)
        os.remove(second)


def test_a_file_written_after_the_look_before_the_lock_is_compared_under_it(
    server, monkeypatch, caplog
):
    """A file rewritten between the look before the lock and the look under
    it holds content that look never saw, so the pictures it was compared
    with are compared again."""
    root = server.vault.image_root
    there, new = _bmp(), _bmp()
    candidate_id, _path = _library_picture(
        server, there, "forced-root-rewritten-collision"
    )
    path = _write(os.path.join(root, "test-compared-rewritten.bmp"), new)
    _colliding(monkeypatch, "forced-root-rewritten-collision")
    real_build = ReferenceFolderScanTask._build_picture_chunk

    def _build_then_rewrite(self, file_paths, folder_id):
        built = real_build(self, file_paths, folder_id)
        if path in file_paths:
            _write(path, there)
        return built

    monkeypatch.setattr(
        ReferenceFolderScanTask, "_build_picture_chunk", _build_then_rewrite
    )

    try:
        with caplog.at_level("INFO", logger="pixlstash"):
            _scan_settled_root(server)

        assert _ids_at(server, "test-compared-rewritten.bmp") == [], (
            "a file rewritten into a copy of a picture was indexed"
        )
        assert _skips(caplog) == [
            f"Root scan: not importing {path}: picture {candidate_id} already holds "
            "its content. The file is left where it is."
        ]
    finally:
        os.remove(path)
