"""Model-folder edges: which rows a directory listing counts, how it shows a
linked directory, a first scan whose submission raised, and a folder registered
during a restore.

The listing reads the folder's `present` copies below the listed directory
with :class:`~pixlstash.utils.query.folder_prefix.FolderPrefix`, so a prefix
sibling, a LIKE wildcard in a name and (where paths are case-sensitive) a case
variant stay out without a Python filter after the query.

A symlink or junction to a directory is a read-only ``link`` node with nothing
to expand, carrying the models the scanner registered under its name; a broken
link and a link to a file are not listed.

A first scan is claimed (``claim_scan``) before it is submitted. A submission
that raises must give the claim back, or the folder reads as being scanned
for good and the background scan never claims it again.

Registering a catalogued folder writes its row, then submits its first scan.
While a library restore has paused task admission, the request is refused
(409) before the row is written. A submission that raises undoes the row
before the error answers, so the client's retry is not told "already
registered": a ``RuntimeError`` (a dead worker's replacement thread could not
start) answers 503 with Retry-After, the restore pause 409. A ComfyUI
registration, which nobody waits on, keeps the row for the background scan
and goes on to its next root.
"""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from pixlstash.routes import model_folders as model_folders_routes
from pixlstash.server import Server
from pixlstash.task_runner import TaskAdmissionRefused
from pixlstash.tasks import model_folder_scan_task
from pixlstash.utils.query.folder_prefix import PATHS_FOLD_CASE
from tests.utils import reset_owner_auth

API = "/api/v1"


@pytest.fixture(scope="module")
def server():
    """One Server, no background work: no scan rewrites the rows written here."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = f"{temp_dir}/server-config.json"
        with open(config_path, "w") as handle:
            json.dump({"disable_background_workers": True}, handle)
        with Server(config_path) as srv:
            yield srv


@pytest.fixture
def owner(server) -> TestClient:
    reset_owner_auth(server)
    client = TestClient(server.api)
    response = client.post(
        "/login",
        json={"username": "folderowner", "password": "example-folderowner-password"},
    )
    assert response.status_code == 200, response.text
    return client


def _folder_with_copies(server, root: str, copies) -> int:
    """Register *root* as a catalogued folder holding exactly these copies.

    Each copy is ``(posix relpath, size)``; stored with ``os.sep`` as the
    scanner stores it.
    """
    with server.hub.transaction() as conn:
        folder_id = conn.execute(
            "INSERT INTO model_folder (path, kind, movable, created_at) "
            "VALUES (?, 'user', 'per_item', '2026-08-09T00:00:00Z')",
            (root,),
        ).lastrowid
        for number, (relpath, size) in enumerate(copies):
            model_id = conn.execute(
                "INSERT INTO model (file_kind, kind, sha256, filename, provenance, "
                "file_size, created_at) VALUES ('adapter', 'lora', ?, ?, "
                "'external', ?, '2026-08-08T00:00:00Z')",
                (
                    (f"listing-{folder_id}-{number}" + "0" * 64)[:64],
                    relpath.rsplit("/", 1)[-1],
                    size,
                ),
            ).lastrowid
            conn.execute(
                "INSERT INTO model_file (model_id, model_folder_id, relpath, state, "
                "seen_at, file_mtime) VALUES (?, ?, ?, 'present', "
                "'2026-08-09T00:00:00Z', 0)",
                (model_id, folder_id, relpath.replace("/", os.sep)),
            )
    return int(folder_id)


def _listing(client, folder_id: int, path: str) -> dict:
    response = client.get(
        f"{API}/model-folders/{folder_id}/dirs", params={"path": path}
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestTheListingCountsOnlyWhatIsBelowIt:
    def test_a_sibling_a_wildcard_name_and_a_case_variant_stay_out(
        self, server, owner, tmp_path
    ):
        root = tmp_path / "loras"
        (root / "people" / "alice").mkdir(parents=True)
        (root / "peo_le").mkdir()
        folder_id = _folder_with_copies(
            server,
            str(root),
            [
                ("people/a.safetensors", 1),
                ("people/alice/b.safetensors", 2),
                # A prefix sibling: `people` followed by no separator.
                ("people-old/c.safetensors", 4),
                # `_` is LIKE's one-character wildcard: `peo_le/%` matches
                # every path under `people/` unless it is escaped.
                ("peo_le/d.safetensors", 8),
                # Another directory where paths are case-sensitive, the same
                # one where they are not (Windows).
                ("People/e.safetensors", 16),
            ],
        )

        body = _listing(owner, folder_id, "people")
        case_variant = 16 if PATHS_FOLD_CASE else 0
        assert body["bytes"] == 1 + 2 + case_variant
        assert body["model_count"] == (3 if PATHS_FOLD_CASE else 2)
        assert body["direct_model_count"] == (2 if PATHS_FOLD_CASE else 1)
        dirs = {entry["name"]: entry for entry in body["dirs"]}
        assert list(dirs) == ["alice"]
        assert (dirs["alice"]["model_count"], dirs["alice"]["bytes"]) == (1, 2)

        body = _listing(owner, folder_id, "peo_le")
        assert (body["model_count"], body["bytes"], body["direct_model_count"]) == (
            1,
            8,
            1,
        )

    def test_the_root_listing_counts_every_copy(self, server, owner, tmp_path):
        """The positive control: no directory, no range, every row."""
        root = tmp_path / "loras"
        (root / "people").mkdir(parents=True)
        folder_id = _folder_with_copies(
            server,
            str(root),
            [("top.safetensors", 1), ("people/a.safetensors", 2)],
        )

        body = _listing(owner, folder_id, "")

        assert (body["model_count"], body["bytes"], body["direct_model_count"]) == (
            2,
            3,
            1,
        )


def _symlink(link, target, *, target_is_directory=True) -> None:
    """Make *link* point at *target*, or skip where links need a privilege."""
    try:
        os.symlink(target, link, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")


class _ReparsePointEntry:
    """A ``DirEntry`` for a Windows reparse point to a directory, as Windows
    lists one: ``is_symlink`` is False, and its own ``lstat`` carries *tag*."""

    def __init__(self, path, tag):
        self.path = os.fspath(path)
        self.name = os.path.basename(self.path)
        self._info = _ReparseStat(tag)

    def is_symlink(self):
        return False

    def is_dir(self, follow_symlinks=True):
        return True

    def stat(self, follow_symlinks=True):
        assert follow_symlinks is False, "a link test must not follow the entry"
        return self._info


class _ReparseStat:
    """A directory's ``lstat`` result carrying a reparse tag."""

    def __init__(self, tag):
        self.st_mode = 0o040755
        self.st_reparse_tag = tag


class TestALinkedDirectoryIsAReadOnlyNode:
    """The scanner follows a link to a directory and registers the models behind
    it under the link's name, so the tree shows the link as a node carrying
    them: ``link: true``, nothing to expand, and read-only in the client."""

    def test_a_link_node_carries_its_models_and_has_nothing_to_expand(
        self, server, owner, tmp_path
    ):
        """Wrong if the link is missing (its models would have no node), if it
        offers an expand arrow, or if a real directory is marked as a link."""
        root = tmp_path / "loras"
        (root / "real").mkdir(parents=True)
        elsewhere = tmp_path / "test-elsewhere"
        (elsewhere / "deeper").mkdir(parents=True)
        _symlink(root / "linked", elsewhere)
        folder_id = _folder_with_copies(
            server,
            str(root),
            [
                ("real/a.safetensors", 1),
                ("linked/b.safetensors", 2),
                ("linked/deeper/c.safetensors", 4),
            ],
        )

        body = _listing(owner, folder_id, "")

        dirs = {entry["name"]: entry for entry in body["dirs"]}
        assert list(dirs) == ["linked", "real"]
        linked = dirs["linked"]
        assert (linked["link"], linked["has_dirs"]) == (True, False)
        assert (
            linked["path"],
            linked["model_count"],
            linked["direct_model_count"],
            linked["bytes"],
        ) == ("linked", 2, 1, 6)
        assert dirs["real"]["link"] is False

    def test_a_link_the_scanner_skipped_is_listed_with_no_models(
        self, server, owner, tmp_path
    ):
        """A link inside the folder itself is one the scanner skips (it would
        walk those models twice), so it has no rows under its name. It is still
        a node, with zero counts."""
        root = tmp_path / "loras"
        (root / "style").mkdir(parents=True)
        _symlink(root / "alias", root / "style")
        folder_id = _folder_with_copies(server, str(root), [("style/a.safetensors", 8)])

        dirs = {
            entry["name"]: entry for entry in _listing(owner, folder_id, "")["dirs"]
        }

        assert list(dirs) == ["alias", "style"]
        alias = dirs["alias"]
        assert (alias["link"], alias["model_count"], alias["bytes"]) == (True, 0, 0)
        assert (dirs["style"]["link"], dirs["style"]["bytes"]) == (False, 8)

    def test_a_broken_link_and_a_link_to_a_file_are_not_listed(
        self, server, owner, tmp_path
    ):
        """Neither holds models the scanner could have registered, so a node
        for either would be a place with nothing behind it."""
        root = tmp_path / "loras"
        (root / "real").mkdir(parents=True)
        model = tmp_path / "test-model.safetensors"
        model.write_bytes(b"test-")
        _symlink(root / "dangling", tmp_path / "test-gone")
        _symlink(root / "to-a-file", model, target_is_directory=False)
        folder_id = _folder_with_copies(server, str(root), [])

        names = [entry["name"] for entry in _listing(owner, folder_id, "")["dirs"]]

        assert names == ["real"]

    def test_a_directory_holding_only_a_link_can_be_expanded(
        self, server, owner, tmp_path
    ):
        """Wrong if ``has_dirs`` ignores a link node: the tree would draw no
        expand arrow, and the link and its models would be unreachable."""
        root = tmp_path / "loras"
        (root / "people").mkdir(parents=True)
        elsewhere = tmp_path / "test-elsewhere"
        elsewhere.mkdir()
        _symlink(root / "people" / "linked", elsewhere)
        folder_id = _folder_with_copies(server, str(root), [])

        dirs = {
            entry["name"]: entry for entry in _listing(owner, folder_id, "")["dirs"]
        }

        assert (dirs["people"]["has_dirs"], dirs["people"]["link"]) == (True, False)
        inside = _listing(owner, folder_id, "people")["dirs"]
        assert [(entry["name"], entry["link"]) for entry in inside] == [
            ("linked", True)
        ]

    def test_a_junction_to_a_directory_is_a_link_node_and_not_a_place(self, tmp_path):
        """A junction is not a symlink to ``is_symlink``; its own ``lstat``
        carries ``IO_REPARSE_TAG_MOUNT_POINT``. Wrong if it is shown as a place
        to move into, or not shown at all. A cloud-storage placeholder is a
        reparse point with another tag, and is a real directory."""
        junction = _ReparsePointEntry(tmp_path / "Lora", 0xA0000003)
        placeholder = _ReparsePointEntry(tmp_path / "Synced", 0x9000001A)

        assert model_folders_routes._offered_link(junction, set()) is True
        assert model_folders_routes._offered_dir(junction, set()) is False
        assert model_folders_routes._offered_link(placeholder, set()) is False
        assert model_folders_routes._offered_dir(placeholder, set()) is True


class _Interrupted(BaseException):
    """Not an ``Exception``: stands in for ``KeyboardInterrupt`` and kin."""


FOLDER_ID = 9_000_001


@pytest.fixture
def no_claim():
    """Start and end each test with no scan recorded for :data:`FOLDER_ID`."""
    model_folder_scan_task.forget_scan(FOLDER_ID)
    yield
    model_folder_scan_task.forget_scan(FOLDER_ID)


def _claimable(server) -> bool:
    """Whether a fresh claim on the folder succeeds (and give it back)."""
    task = model_folder_scan_task.claim_scan(
        server.hub, FOLDER_ID, "/example-models", "user"
    )
    if task is None:
        return False
    model_folder_scan_task.release_scan(task)
    return True


class TestAFirstScanWhoseSubmissionRaised:
    @pytest.mark.parametrize(
        "raised",
        [
            TaskAdmissionRefused("library restore"),
            RuntimeError("example-replacement-thread-could-not-start"),
            ValueError("example-unexpected-error"),
            _Interrupted("example-interrupt"),
        ],
        ids=[
            "admission-refused",
            "worker-cannot-start",
            "other-exception",
            "base-exception",
        ],
    )
    def test_the_claim_is_released_and_the_error_propagates(
        self, server, monkeypatch, caplog, no_claim, raised
    ):
        def refuse(task):
            raise raised

        monkeypatch.setattr(server.vault, "submit_task", refuse)

        with caplog.at_level("ERROR"):
            with pytest.raises(type(raised)) as caught:
                model_folders_routes.queue_model_folder_scan(
                    server, FOLDER_ID, "/example-models", "user"
                )

        assert caught.value is raised
        assert model_folder_scan_task.scan_state(FOLDER_ID) == (None, None)
        assert _claimable(server)
        logged = [
            record.getMessage()
            for record in caplog.records
            if record.name.endswith("model_folders")
        ]
        assert any(
            f"(id={FOLDER_ID})" in message
            and "/example-models" in message
            and str(raised) in message
            for message in logged
        ), logged

    def test_a_runner_that_takes_none_releases_the_claim_and_answers_false(
        self, server, monkeypatch, no_claim
    ):
        monkeypatch.setattr(server.vault, "submit_task", lambda task: None)

        assert not model_folders_routes.queue_model_folder_scan(
            server, FOLDER_ID, "/example-models", "user"
        )
        assert _claimable(server)

    def test_a_queued_scan_keeps_its_claim(self, server, monkeypatch, no_claim):
        """The positive control: only a scan that did not queue gives it back."""
        monkeypatch.setattr(server.vault, "submit_task", lambda task: task.id)

        assert model_folders_routes.queue_model_folder_scan(
            server, FOLDER_ID, "/example-models", "user"
        )
        assert model_folder_scan_task.scan_state(FOLDER_ID)[0] == "pending"
        assert not _claimable(server)


RESTORE_PAUSE = TaskAdmissionRefused("library restore")


def _folder_row(server, path) -> dict | None:
    """The registered row for *path*, stored resolved, or None."""
    return server.hub.fetchone(
        "SELECT id, kind FROM model_folder WHERE path = ?",
        (os.path.realpath(str(path)),),
    )


@pytest.fixture
def submitted(server, monkeypatch) -> list:
    """Record every task submitted; the runner takes none of them."""
    tasks = []

    def record(task):
        tasks.append(task)
        return None

    monkeypatch.setattr(server.vault, "submit_task", record)
    return tasks


@pytest.fixture
def paused(server, monkeypatch):
    """A library restore has paused task admission (the module's server has
    no runner, whose own look is then a no-op)."""

    def refuse():
        raise RESTORE_PAUSE

    monkeypatch.setattr(server.vault, "refuse_while_admission_paused", refuse)


class TestRegisteringAFolderDuringARestore:
    def test_a_catalogued_folder_is_refused_before_its_row_is_written(
        self, server, owner, tmp_path, submitted, paused
    ):
        folder = tmp_path / "models"
        folder.mkdir()

        response = owner.post(
            f"{API}/model-folders", json={"path": str(folder), "kind": "user"}
        )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == str(RESTORE_PAUSE)
        assert _folder_row(server, folder) is None
        assert submitted == []

    def test_a_source_folder_queues_no_scan_and_is_still_registered(
        self, server, owner, tmp_path, submitted, paused
    ):
        """Over-blocking control: a source folder queues no scan, so the pause
        has nothing to refuse."""
        folder = tmp_path / "runs"
        folder.mkdir()

        response = owner.post(
            f"{API}/model-folders", json={"path": str(folder), "kind": "source"}
        )

        assert response.status_code == 200, response.text
        assert _folder_row(server, folder)["kind"] == "source"
        assert submitted == []

    def test_without_a_pause_the_first_scan_is_submitted(
        self, server, owner, tmp_path, submitted
    ):
        """The positive control: the same request outside a restore."""
        folder = tmp_path / "models"
        folder.mkdir()

        response = owner.post(
            f"{API}/model-folders", json={"path": str(folder), "kind": "user"}
        )

        assert response.status_code == 200, response.text
        folder_id = response.json()["id"]
        model_folder_scan_task.forget_scan(folder_id)
        assert [task.folder_id for task in submitted] == [folder_id]

    def test_a_pause_after_the_look_answers_409_and_gives_the_claim_back(
        self, server, owner, tmp_path, monkeypatch
    ):
        """The pause lands between the look and the submit: the submit refuses
        it. The error leaves nothing behind: no row, no claim."""
        tasks = []

        def refuse(task):
            tasks.append(task)
            raise RESTORE_PAUSE

        monkeypatch.setattr(server.vault, "submit_task", refuse)
        folder = tmp_path / "models"
        folder.mkdir()

        response = owner.post(
            f"{API}/model-folders", json={"path": str(folder), "kind": "user"}
        )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == str(RESTORE_PAUSE)
        assert _folder_row(server, folder) is None
        assert len(tasks) == 1
        assert model_folder_scan_task.scan_state(tasks[0].folder_id) == (None, None)


@pytest.fixture
def no_worker(server, monkeypatch) -> list:
    """Every submit raises as a dead worker's replacement that cannot start
    does (a process at its thread limit fails every one); records each task."""
    tasks = []

    def refuse(task):
        tasks.append(task)
        raise RuntimeError("example-replacement-thread-could-not-start")

    monkeypatch.setattr(server.vault, "submit_task", refuse)
    return tasks


class TestARegistrationWhoseFirstScanCannotQueue:
    """The row is written before the first scan is submitted. A submit that
    raises ``RuntimeError`` undoes it and answers 503 with Retry-After, so the
    retry registers the folder rather than hitting "already registered"."""

    def test_the_route_answers_503_and_leaves_no_row_and_no_claim(
        self, server, owner, tmp_path, no_worker
    ):
        folder = tmp_path / "models"
        folder.mkdir()

        response = owner.post(
            f"{API}/model-folders", json={"path": str(folder), "kind": "user"}
        )

        assert response.status_code == 503, response.text
        assert response.json()["detail"] == (
            "The folder was not added: PixlStash could not start its first "
            "scan. Try again in a moment."
        )
        assert response.headers["Retry-After"] == "2"
        assert "example-replacement" not in response.text
        assert _folder_row(server, folder) is None
        assert len(no_worker) == 1
        assert model_folder_scan_task.scan_state(no_worker[0].folder_id) == (
            None,
            None,
        )

    def test_the_retry_registers_the_folder(
        self, server, owner, tmp_path, monkeypatch, no_worker
    ):
        """The positive control: once the worker can start, the same request
        succeeds rather than answering "already registered"."""
        folder = tmp_path / "models"
        folder.mkdir()
        payload = {"path": str(folder), "kind": "user"}
        assert owner.post(f"{API}/model-folders", json=payload).status_code == 503

        monkeypatch.setattr(server.vault, "submit_task", lambda task: None)
        response = owner.post(f"{API}/model-folders", json=payload)

        assert response.status_code == 200, response.text
        assert _folder_row(server, folder)["id"] == response.json()["id"]

    def test_a_comfyui_root_that_cannot_queue_does_not_drop_the_next(
        self, server, tmp_path, monkeypatch, no_worker
    ):
        first = tmp_path / "comfy-a" / "models"
        second = tmp_path / "comfy-b" / "models"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        monkeypatch.setattr(
            model_folders_routes.comfyui_link_service,
            "locate",
            lambda url: ("this_computer", url),
        )
        monkeypatch.setattr(
            model_folders_routes,
            "comfyui_model_roots",
            lambda url: [str(first), str(second)],
        )

        registered = model_folders_routes.register_comfyui_model_folders(
            server, "http://127.0.0.1:18188"
        )

        rows = [_folder_row(server, first), _folder_row(server, second)]
        try:
            assert all(row is not None for row in rows), rows
            assert registered == [row["id"] for row in rows]
            assert [task.folder_id for task in no_worker] == registered
            for folder_id in registered:
                assert model_folder_scan_task.scan_state(folder_id) == (None, None)
        finally:
            for row in rows:
                if row is not None:
                    model_folder_scan_task.forget_scan(row["id"])


class TestComfyuiRootsAreCheckedBeforeTheyAreResolved:
    def test_a_nul_a_relative_and_a_missing_root_are_skipped_and_logged(
        self, server, tmp_path, monkeypatch, caplog, submitted
    ):
        """ComfyUI's answer is checked before realpath sees it: a NUL byte
        made realpath raise and lost every root after it; a relative path was
        anchored at this server's working directory, so here it names a real
        directory under it and only the absolute-path check refuses it."""
        good = tmp_path / "comfy" / "models"
        good.mkdir(parents=True)
        (tmp_path / "example-relative" / "models").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        invalid = [
            str(tmp_path / "with\0nul"),
            "example-relative/models",
            str(tmp_path / "example-missing"),
        ]
        monkeypatch.setattr(
            model_folders_routes.comfyui_link_service,
            "locate",
            lambda url: ("this_computer", url),
        )
        monkeypatch.setattr(
            model_folders_routes,
            "comfyui_model_roots",
            lambda url: [*invalid, str(good)],
        )

        with caplog.at_level("WARNING"):
            registered = model_folders_routes.register_comfyui_model_folders(
                server, "http://127.0.0.1:18188"
            )

        row = _folder_row(server, good)
        try:
            assert registered == [row["id"]]
            warned = [
                record.getMessage()
                for record in caplog.records
                if record.name.endswith("model_folders")
                and record.levelname == "WARNING"
            ]
            for root in invalid:
                assert (
                    f"ComfyUI reported {root!r} as a model folder; it is not an "
                    "absolute path to a directory on this computer, so it is "
                    "skipped."
                ) in warned, warned
        finally:
            if row is not None:
                model_folder_scan_task.forget_scan(row["id"])


class TestAnOverlappingFolderIsRefusedWithoutItsPath:
    """The 409 says which way the folders overlap in a fixed sentence; the
    registered path it collided with goes to the log, not to the caller."""

    @pytest.mark.parametrize(
        ("registered", "requested", "detail"),
        [
            ("drive", "drive/loras", "This path is inside a registered model folder."),
            ("drive/loras", "drive", "A registered model folder is inside this path."),
        ],
        ids=["inside-registered", "holds-registered"],
    )
    def test_the_detail_is_the_fixed_sentence(
        self, server, owner, tmp_path, caplog, submitted, registered, requested, detail
    ):
        (tmp_path / "drive" / "loras").mkdir(parents=True)
        first = owner.post(
            f"{API}/model-folders",
            json={"path": str(tmp_path / registered), "kind": "user"},
        )
        assert first.status_code == 200, first.text

        with caplog.at_level("INFO"):
            response = owner.post(
                f"{API}/model-folders",
                json={"path": str(tmp_path / requested), "kind": "user"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == detail
        assert str(tmp_path) not in response.text
        logged = [
            record.getMessage()
            for record in caplog.records
            if record.name.endswith("model_folders")
        ]
        assert any(
            f"id={first.json()['id']}" in message
            and os.path.realpath(str(tmp_path / registered)) in message
            for message in logged
        ), logged
