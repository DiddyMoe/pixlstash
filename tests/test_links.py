"""A Windows directory junction is a link to every directory guard.

``os.path.islink`` and ``DirEntry.is_symlink`` say False for a junction, and
``os.walk`` descends one, so a guard that asks only those lets a walk, move or
delete follow the redirect that needs no privilege (``mklink /J``).

No junction can exist on Linux, so one is simulated: a real directory whose
``lstat`` carries the ``IO_REPARSE_TAG_MOUNT_POINT`` tag. ``os.walk`` descends
it, as Windows descends a real junction.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from pixlstash.database import VaultDatabase
from pixlstash.db_models.picture import Picture
from pixlstash.hub import registry
from pixlstash.routes import model_files, model_folders, model_moves
from pixlstash.services import (
    edit_copies,
    folder_structure_commit_service,
    library_backup_service,
    library_restore_service,
    model_dir_mover,
    portable_identity,
)
from pixlstash.services.folder_structure_service import FolderStructureRead
from pixlstash.services.library_backup_service import BackupError
from pixlstash.services.library_restore_service import RestoreError
from pixlstash.services.model_mover import (
    MoveRefused,
    resolve_subfolder,
    samples_relpath,
)
from pixlstash.services.portable_identity import PortableIdentityScrubError
from pixlstash.server import Server
from pixlstash.utils import links
from pixlstash.utils.media_files import count_media_files
from tests.utils import API_PREFIX

JUNCTION_TAG = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT
# IO_REPARSE_TAG_CLOUD_6: a OneDrive placeholder directory. A reparse point,
# not a link: its files are really there.
CLOUD_PLACEHOLDER_TAG = 0x9000601A


class _ReparseStat:
    """An ``lstat`` result carrying a reparse tag, as Windows reports one."""

    def __init__(self, info, tag):
        self._info = info
        self.st_reparse_tag = tag

    def __getattr__(self, name):
        return getattr(self._info, name)


def _simulate_junction(monkeypatch, path, tag=JUNCTION_TAG) -> None:
    """Make *path*'s ``lstat`` carry *tag*."""
    real_lstat = os.lstat
    target = os.fspath(path)

    def lstat(candidate, *args, **kwargs):
        info = real_lstat(candidate, *args, **kwargs)
        if not isinstance(candidate, int) and os.fspath(candidate) == target:
            return _ReparseStat(info, tag)
        return info

    monkeypatch.setattr(os, "lstat", lstat)


class _Entry:
    """A ``DirEntry`` stand-in whose own ``lstat`` the test decides."""

    def __init__(self, path, info=None, error=None):
        self.path = os.fspath(path)
        self.name = os.path.basename(self.path)
        self._info = info
        self._error = error

    def is_symlink(self):
        return False

    def is_dir(self, follow_symlinks=True):
        return True

    def stat(self, follow_symlinks=True):
        assert follow_symlinks is False, "a link test must not follow the entry"
        if self._error is not None:
            raise self._error
        return self._info


class _LstatEntry:
    """A real ``DirEntry`` whose own ``lstat`` is read through ``os.lstat``.

    Windows caches the entry's ``lstat``, reparse tag included, when it lists
    the directory; reading it through ``os.lstat`` lets the simulated tag reach
    the entry the same way.
    """

    def __init__(self, entry):
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry, name)

    def stat(self, *, follow_symlinks=True):
        if follow_symlinks:
            return self._entry.stat()
        return os.lstat(self._entry.path)


def _simulate_junction_entries(monkeypatch, root) -> None:
    """Make the entries ``os.scandir`` lists for *root* read their ``lstat``
    through ``os.lstat``. Every other listing is untouched."""
    real_scandir = os.scandir
    listed = os.fspath(root)

    def scandir(path="."):
        if isinstance(path, int) or os.fspath(path) != listed:
            return real_scandir(path)
        with real_scandir(path) as entries:
            return iter([_LstatEntry(entry) for entry in entries])

    monkeypatch.setattr(os, "scandir", scandir)


def _record_unlinks(monkeypatch, path) -> list[str]:
    """Record ``os.unlink`` of *path* instead of running it.

    ``os.unlink`` removes a junction on Windows; Linux cannot unlink a
    directory. Every other path is unlinked for real.
    """
    real_unlink = os.unlink
    target = os.fspath(path)
    unlinked: list[str] = []

    def unlink(candidate, *args, **kwargs):
        if os.fspath(candidate) == target:
            unlinked.append(target)
            return
        real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    return unlinked


def _picture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (32, 64, 96)).save(path)


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


def test_a_junction_is_a_link_and_a_plain_or_placeholder_directory_is_not(
    tmp_path, monkeypatch
):
    """Only the mount-point tag counts: calling a cloud placeholder directory a
    link would hide files that are really there."""
    junction = tmp_path / "junction"
    placeholder = tmp_path / "placeholder"
    plain = tmp_path / "plain"
    for folder in (junction, placeholder, plain):
        folder.mkdir()
    _simulate_junction(monkeypatch, junction)
    real_lstat = os.lstat

    def lstat(candidate, *args, **kwargs):
        info = real_lstat(candidate, *args, **kwargs)
        if os.fspath(candidate) == str(placeholder):
            return _ReparseStat(info, CLOUD_PLACEHOLDER_TAG)
        return info

    monkeypatch.setattr(os, "lstat", lstat)

    assert links.is_link(junction) is True
    assert links.is_link_or_missing(junction) is True
    assert links.is_link_entry(_Entry(junction, info=os.lstat(junction))) is True
    assert links.is_link(placeholder) is False
    assert links.is_link(plain) is False
    assert links.is_link_entry(_Entry(plain, info=os.lstat(plain))) is False


def test_a_path_that_cannot_be_read_is_the_callers_decision(tmp_path):
    """``is_link`` raises; ``is_link_or_missing`` answers as ``islink`` does."""
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        links.is_link(missing)
    assert links.is_link_or_missing(missing) is False
    assert links.is_link_or_missing(f"{tmp_path}/nul\0byte") is False


# --------------------------------------------------------------------------
# The guards, one junction each, with a plain folder as the control
# --------------------------------------------------------------------------


def test_the_model_folder_tree_does_not_offer_a_junction(tmp_path, caplog):
    """Wrong if the tree offers a junction as a place to move into, or if an
    entry removed since the listing fails the whole listing."""
    junction_info = _ReparseStat(os.lstat(tmp_path), JUNCTION_TAG)
    gone = FileNotFoundError(2, "No such file or directory")

    offered = model_folders._offered_dir
    assert offered(_Entry(tmp_path / "plain", info=os.lstat(tmp_path)), set()) is True
    assert offered(_Entry(tmp_path / "junction", info=junction_info), set()) is False
    with caplog.at_level("WARNING", logger=model_folders.logger.name):
        assert offered(_Entry(tmp_path / "test-gone", error=gone), set()) is False
    assert "test-gone" in caplog.text, "the left-out entry did not reach the log"


def test_a_move_into_a_junction_is_refused(tmp_path, monkeypatch):
    """Wrong if a subpath through a junction resolves: the move would write
    into wherever the link points."""
    (tmp_path / "Lora").mkdir()
    _simulate_junction(monkeypatch, tmp_path / "Lora")

    with pytest.raises(MoveRefused, match="goes through a link"):
        resolve_subfolder(str(tmp_path), "Lora")
    assert resolve_subfolder(str(tmp_path), "New")[0] == "New", (
        "a folder that does not exist yet is not a link"
    )


def test_a_backup_refuses_a_junction_inside_the_library(tmp_path, monkeypatch):
    """``os.walk`` descends a junction, so an archive walk that only refused
    symlinks would pack whatever the junction points at."""
    junction = tmp_path / "junction"
    junction.mkdir()
    _simulate_junction(monkeypatch, junction)

    with pytest.raises(BackupError, match="^Refusing symlinked "):
        library_backup_service._validate_regular_directory(
            str(junction), label="library payload directory"
        )
    library_backup_service._validate_regular_directory(
        str(tmp_path), label="library payload directory"
    )


def test_the_edit_copy_sweep_leaves_a_junction_named_like_a_batch_alone(
    tmp_path, monkeypatch
):
    """A batch folder is the sweep's own; a junction with a batch's name is
    not, and emptying it empties the folder it points at."""
    root = tmp_path / "root"
    junction = root / ("a" * 32)
    stale = root / ("b" * 32)
    for batch in (junction, stale):
        batch.mkdir(parents=True)
        (batch / "test-picture.png").write_bytes(b"test-bytes")
        os.utime(batch / "test-picture.png", (0, 0))
        os.utime(batch, (0, 0))
    _simulate_junction(monkeypatch, junction)
    _simulate_junction_entries(monkeypatch, root)

    removed = edit_copies.sweep_edit_sessions(str(root), max_age_seconds=60)

    assert (junction / "test-picture.png").read_bytes() == b"test-bytes"
    assert removed == 1, "the stale batch beside it is the control"
    assert not stale.exists()


def test_start_up_never_imports_an_edit_through_a_junction(tmp_path, monkeypatch):
    """A junction named like a session would import whatever folder it points
    at. Wrong if the junction is processed, or the plain session is not."""
    root = tmp_path / "root"
    junction = root / ("a" * 32)
    session = root / ("b" * 32)
    junction.mkdir(parents=True)
    session.mkdir()
    processed = []

    def process_session(_server, path):
        processed.append(path)
        return []

    monkeypatch.setattr(edit_copies, "process_session", process_session)
    _simulate_junction(monkeypatch, junction)
    _simulate_junction_entries(monkeypatch, root)

    assert edit_copies.reconcile_all(object(), str(root)) == 0
    assert processed == [str(session)]


def test_a_restore_refuses_a_junction_as_its_destination(tmp_path, monkeypatch):
    """An empty junction passes every other test the destination gets, and the
    restore would then write the library into the folder it points at."""
    junction = tmp_path / "junction"
    junction.mkdir()
    plain = tmp_path / "plain"
    plain.mkdir()
    _simulate_junction(monkeypatch, junction)

    with pytest.raises(RestoreError, match="^Refusing to restore through symlink "):
        library_restore_service._require_empty_destination(str(junction))
    library_restore_service._require_empty_destination(str(plain))


def test_withdrawing_a_restore_unlinks_a_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """A failed restore takes back what it published. A junction among it is
    removed as a link, never emptied with ``rmtree``."""
    folder = tmp_path / "library"
    junction = folder / "test-junction"
    junction.mkdir(parents=True)
    (junction / "test-owner.png").write_bytes(b"test-bytes")
    restored = folder / "test-restored"
    restored.mkdir()
    (restored / "test-picture.png").write_bytes(b"test-bytes")
    _simulate_junction(monkeypatch, junction)
    unlinked = _record_unlinks(monkeypatch, junction)

    library_restore_service._withdraw_library_folder(str(folder), remove_folder=False)

    assert unlinked == [str(junction)]
    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not restored.exists(), "the plain restored folder is the control"


def test_an_abandoned_import_leaves_a_junction_alone(tmp_path, monkeypatch):
    """An abandoned first import removes the folders it made, by name. A
    junction with one of those names is the owner's, and emptying it empties
    the folder it points at."""
    library = registry.Library(
        id=1,
        uuid="test-uuid",
        name="test-library",
        path=str(tmp_path),
        created_at="",
        attached_at="",
        is_active=False,
        pending_import_kept=(),
    )
    junction = tmp_path / "snapshots"
    junction.mkdir()
    (junction / "test-owner.png").write_bytes(b"test-bytes")
    made = tmp_path / ".staging"
    made.mkdir()
    _simulate_junction(monkeypatch, junction)

    registry.remove_library_made_entries(library)

    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not made.exists(), "the folder the import made is the control"


def test_a_junction_as_the_staging_folder_is_neither_swept_nor_staged_into(
    tmp_path, monkeypatch
):
    """Opening an import session sweeps old entries under ``.staging`` and then
    stages into it. Through a junction, the sweep empties the folder it points
    at and the upload lands there."""
    with Server(str(tmp_path / "server-config.json")) as server:
        client = TestClient(server.api)
        login = client.post(
            f"{API_PREFIX}/login",
            json={"username": "owner", "password": "example-owner-password"},
        )
        assert login.status_code == 200, login.text
        staging = os.path.join(server.vault.image_root, ".staging")

        def open_beside_an_old_entry():
            old = Path(staging) / "test-old-entry"
            old.mkdir(parents=True)
            os.utime(old, (0, 0))
            return old, client.post(f"{API_PREFIX}/pictures/import/staging", json={})

        old, opened = open_beside_an_old_entry()
        assert opened.status_code == 200, "the plain folder is the control"
        assert not old.exists(), "the plain folder is the control"

        _simulate_junction(monkeypatch, staging)
        old, opened = open_beside_an_old_entry()

        assert old.exists()
        assert opened.status_code == 500
        assert opened.json()["detail"] == (
            "The library staging path is not a trusted directory."
        )


def test_a_junction_as_the_snapshots_folder_is_not_trusted(tmp_path, monkeypatch):
    """The identity scrub rewrites and removes files below ``snapshots``. A
    junction there leads out of the library."""
    vault = tmp_path / "vault"
    (vault / "snapshots").mkdir(parents=True)
    snapshots = os.path.join(os.path.realpath(vault), "snapshots")
    assert portable_identity._snapshots_root(str(vault)) == snapshots, (
        "the plain folder is the control"
    )
    _simulate_junction(monkeypatch, snapshots)

    with pytest.raises(PortableIdentityScrubError, match="not a trusted directory"):
        portable_identity._snapshots_root(str(vault))


def test_a_snapshot_path_through_a_junction_is_refused(tmp_path, monkeypatch):
    """A registered snapshot path is untrusted and its containment check is
    lexical, so the walk that refuses a link in it is all that stops one
    through a junction."""
    vault = tmp_path / "vault"
    for name in ("test-junction", "test-plain"):
        (vault / "snapshots" / name).mkdir(parents=True)
        (vault / "snapshots" / name / "snapshot.sqlite").write_bytes(b"test-bytes")
    root = portable_identity._snapshots_root(str(vault))
    _simulate_junction(monkeypatch, os.path.join(root, "test-junction"))

    with pytest.raises(PortableIdentityScrubError, match="contains a symlink: "):
        portable_identity._registered_snapshot_path(
            str(vault),
            root,
            os.path.join("snapshots", "test-junction", "snapshot.sqlite"),
        )
    assert portable_identity._registered_snapshot_path(
        str(vault), root, os.path.join("snapshots", "test-plain", "snapshot.sqlite")
    ) == os.path.join(root, "test-plain", "snapshot.sqlite")


def test_the_stale_scrub_sweep_never_walks_a_junction(tmp_path, monkeypatch):
    """The sweep removes every stale scrub folder it finds below ``snapshots``.
    One reached through a junction is outside the library."""
    snapshots = tmp_path / "vault" / "snapshots"
    owner = snapshots / "test-junction" / f"{portable_identity._SCRUB_PREFIX}test-a"
    owner.mkdir(parents=True)
    (owner / "test-owner.txt").write_text("test-owner-file")
    stale = snapshots / f"{portable_identity._SCRUB_PREFIX}test-b"
    stale.mkdir()
    _simulate_junction(
        monkeypatch, os.path.join(os.path.realpath(snapshots), "test-junction")
    )

    portable_identity._cleanup_stale_snapshot_scrubs(str(tmp_path / "vault"))

    assert (owner / "test-owner.txt").read_text() == "test-owner-file"
    assert not stale.exists(), "the plain stale folder is the control"


def test_a_folder_move_keeps_a_source_holding_a_junction(tmp_path, monkeypatch):
    """A cross-drive folder move deletes the source only when it lists the same
    as the copy. ``copytree`` copies a junction as a plain folder of the files
    it points at; a source listing that walked the junction the same way would
    match, and deleting the source would delete those files."""
    source = tmp_path / "source"
    junction = source / "test-junction"
    junction.mkdir(parents=True)
    (junction / "test-model.safetensors").write_bytes(b"test-bytes")
    # Copied before the simulation, so the copy holds a plain folder, as
    # ``copytree`` makes of a junction.
    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    assert model_dir_mover._tree_listing(str(source)) == (
        model_dir_mover._tree_listing(str(copy))
    ), "the control: as a plain folder, the two listings match"
    _simulate_junction(monkeypatch, junction)

    listed = model_dir_mover._tree_listing(str(source))

    assert "test-junction" in listed
    assert listed != model_dir_mover._tree_listing(str(copy))


def test_a_merge_does_not_claim_previews_behind_a_junction(
    tmp_path, monkeypatch, caplog
):
    """A merge logs that the removed copy's previews folder is still the
    model's. A junction there leads elsewhere and is not named as the model's."""
    junction = samples_relpath(str(tmp_path / "test-junction.safetensors"))
    plain = samples_relpath(str(tmp_path / "test-plain.safetensors"))
    os.mkdir(junction)
    os.mkdir(plain)
    _simulate_junction(monkeypatch, junction)

    with caplog.at_level("INFO", logger=model_files.logger.name):
        model_files._warn_about_samples(str(tmp_path / "test-junction.safetensors"))
        model_files._warn_about_samples(str(tmp_path / "test-plain.safetensors"))

    assert junction not in caplog.text
    assert plain in caplog.text, "the plain folder is the control"


def test_deleting_a_model_never_removes_previews_behind_a_junction(
    tmp_path, monkeypatch
):
    """A deleted model takes its previews folder when it holds only previews.
    What a junction there points at is not the model's to delete."""
    preview = "1700000000000__000000100_0.jpg"
    junction = samples_relpath(str(tmp_path / "test-junction.safetensors"))
    plain = samples_relpath(str(tmp_path / "test-plain.safetensors"))
    for folder in (junction, plain):
        os.mkdir(folder)
        Path(folder, preview).write_bytes(b"test-bytes")
    _simulate_junction(monkeypatch, junction)

    assert not model_files._remove_samples(
        str(tmp_path / "test-junction.safetensors"), permanent=True
    )
    assert Path(junction, preview).read_bytes() == b"test-bytes"
    assert model_files._remove_samples(
        str(tmp_path / "test-plain.safetensors"), permanent=True
    ), "the plain folder is the control"
    assert not os.path.exists(plain)


def test_a_relocation_leaves_a_junction_in_the_vacated_folder_alone(
    tmp_path, monkeypatch
):
    """A relocation carries every file in the download folder across, then
    removes the empty folders it left. A junction there is the owner's and
    leads out of it."""
    source = tmp_path / "test-source"
    (source / "test-plain").mkdir(parents=True)
    (source / "test-plain" / "test-tagger.onnx").write_bytes(b"test-bytes")
    junction = source / "test-elsewhere"
    (junction / "test-empty").mkdir(parents=True)
    (junction / "test-owner.onnx").write_bytes(b"test-bytes")
    destination = tmp_path / "test-destination"
    _simulate_junction(monkeypatch, junction)

    model_moves._move_leftovers(str(source), str(destination))
    model_moves._remove_if_empty(str(source))

    assert (junction / "test-owner.onnx").read_bytes() == b"test-bytes"
    assert (junction / "test-empty").is_dir()
    assert (destination / "test-plain" / "test-tagger.onnx").read_bytes() == (
        b"test-bytes"
    ), "the plain file is the control"
    assert not (source / "test-plain").exists(), "the vacated folder is the control"


# --------------------------------------------------------------------------
# The walks that skip a directory symlink by leaving it to ``os.walk``
# --------------------------------------------------------------------------


def test_a_picture_count_does_not_follow_a_junction(tmp_path, monkeypatch):
    """The folder pickers' count promises what the import will index. Wrong if
    the junction's picture counts."""
    _picture(tmp_path / "test-plain" / "test-a.png")
    junction = tmp_path / "test-elsewhere"
    _picture(junction / "test-b.png")
    assert count_media_files(str(tmp_path)) == (2, False), (
        "the control: as a plain directory, its picture counts"
    )
    _simulate_junction(monkeypatch, junction)

    assert count_media_files(str(tmp_path)) == (1, False)


def test_the_folder_structure_read_lists_a_junction_but_never_walks_it(
    tmp_path, monkeypatch
):
    """The read walks a caller-supplied root, so a junction loop would walk
    until the folder cap. Wrong if the junction is mapped or its picture
    counted, or if the root stops counting it as a child."""
    root = tmp_path / "test-root"
    _picture(root / "test-plain" / "test-a.png")
    junction = root / "test-elsewhere"
    _picture(junction / "test-b.png")
    _simulate_junction(monkeypatch, junction)

    result = FolderStructureRead(str(root)).run()

    assert result["folder_count"] == 2, "the root and its plain folder"
    assert result["picture_count"] == 1
    (root_level,) = (level for level in result["levels"] if level["depth"] == 1)
    assert root_level["folders"][0]["child_count"] == 2


def test_a_folder_structure_commit_never_imports_a_file_through_a_junction(
    tmp_path, monkeypatch
):
    """A local-import commit makes every picture under the chosen root a
    managed picture of this library, and a junction under that root leads out
    of it."""
    library = tmp_path / "test-library"
    library.mkdir()
    vault_db = VaultDatabase(str(library / "vault.db"))
    try:
        _picture(library / "test-plain" / "test-a.png")
        junction = library / "test-elsewhere"
        _picture(junction / "test-b.png")
        _simulate_junction(monkeypatch, junction)
        server = SimpleNamespace(
            vault=SimpleNamespace(
                db=vault_db, image_root=str(library), wake=lambda: None
            )
        )

        ids = folder_structure_commit_service.local_import_pictures(
            server, str(library), expected_pictures=0
        )

        imported = vault_db.run_immediate_read_task(
            lambda session: sorted(
                session.exec(select(Picture.file_path).where(Picture.id.in_(ids)))
            )
        )
    finally:
        vault_db.close()

    assert imported == ["test-plain/test-a.png"]
