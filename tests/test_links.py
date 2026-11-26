"""A Windows directory junction is a link to every directory guard.

``os.path.islink`` and ``DirEntry.is_symlink`` say False for a junction, and
``os.walk`` descends one, so a guard that asks only those refuses the redirect
that needs a privilege and accepts the one that needs none (``mklink /J``).
``pixlstash.utils.links`` answers it on Python 3.11 too; these tests hold the
helper and the directory guards that use it to that.

No junction can exist on Linux, so one is simulated as tests/test_hub_engine.py
does: the platform reads as ``nt`` and the directory's ``lstat`` carries the
``IO_REPARSE_TAG_MOUNT_POINT`` tag.
"""

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from pixlstash import plugin_install
from pixlstash.database import VaultDatabase
from pixlstash.db_models.import_folder import ImportFolder
from pixlstash.db_models.picture import Picture
from pixlstash.hub import registry
from pixlstash.plugin_install import IMAGE, PluginError
from pixlstash.routes import model_files, model_folders, model_moves
from pixlstash.routes.pictures import _export as export_routes
from pixlstash.services import (
    comfyui_link_service,
    comfyui_pack_service,
    edit_copies,
    folder_structure_commit_service,
    layout_move_service,
    library_backup_service,
    library_restore_service,
    model_dir_mover,
    portable_identity,
)
from pixlstash.services.folder_structure_service import FolderStructureRead
from pixlstash.services.library_backup_service import BackupError
from pixlstash.services.library_restore_service import RestoreError
from pixlstash.services.library_switch_service import LibrarySwitchService
from pixlstash.services.model_mover import (
    MoveRefused,
    resolve_subfolder,
    samples_relpath,
)
from pixlstash.server import Server
from pixlstash.services.portable_identity import PortableIdentityScrubError
from pixlstash.tasks import missing_watch_folder_import_finder
from pixlstash.utils import links
from pixlstash.utils.caption_file_utils import detect_folder_suffixes
from pixlstash.utils.media_files import count_media_files
from pixlstash.utils.service.share_copies import sweep_share_copies

JUNCTION_TAG = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT
# IO_REPARSE_TAG_CLOUD_6: a OneDrive placeholder directory. A reparse point,
# not a link: its files are really there.
CLOUD_PLACEHOLDER_TAG = 0x9000601A

needs_symlinks = pytest.mark.skipif(
    sys.platform == "win32", reason="creating a symlink needs privileges on Windows"
)


class _ReparseStat:
    """An ``lstat`` result carrying a reparse tag, as Windows reports one."""

    def __init__(self, info, tag):
        self._info = info
        self.st_reparse_tag = tag

    def __getattr__(self, name):
        return getattr(self._info, name)


def _simulate_reparse_point(monkeypatch, path, tag=JUNCTION_TAG) -> None:
    """Make *path*'s ``lstat`` carry *tag* on a platform that reads as ``nt``."""
    real_lstat = os.lstat
    target = os.fspath(path)

    def lstat(candidate, *args, **kwargs):
        info = real_lstat(candidate, *args, **kwargs)
        if os.fspath(candidate) == target:
            return _ReparseStat(info, tag)
        return info

    monkeypatch.setattr(os, "name", "nt")
    # raising=False: the constant is Windows-only, so it does not exist to be
    # replaced on the machine running this test.
    monkeypatch.setattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", JUNCTION_TAG, raising=False)
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


class _ListedEntries:
    """``os.scandir``'s iterator, yielding :class:`_LstatEntry` stand-ins."""

    def __init__(self, entries):
        self._entries = entries

    def __iter__(self):
        return (_LstatEntry(entry) for entry in self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._entries.close()

    def close(self):
        self._entries.close()


def _simulate_junction_entries(monkeypatch, root) -> None:
    """Make the entries ``os.scandir`` lists for *root* carry their ``lstat``
    tag. Every other listing (``rmtree`` lists by descriptor) is untouched."""
    real_scandir = os.scandir
    listed = os.fspath(root)

    def scandir(path="."):
        entries = real_scandir(path)
        if isinstance(path, int) or os.fspath(path) != listed:
            return entries
        return _ListedEntries(entries)

    monkeypatch.setattr(os, "scandir", scandir)


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


def test_a_junction_is_a_link_and_a_plain_directory_is_not(tmp_path, monkeypatch):
    junction = tmp_path / "junction"
    junction.mkdir()
    plain = tmp_path / "plain"
    plain.mkdir()
    _simulate_reparse_point(monkeypatch, junction)

    assert links.is_link(junction) is True
    assert links.is_link_or_missing(junction) is True
    assert links.is_link_stat(os.lstat(junction)) is True
    assert links.is_link(plain) is False
    assert links.is_link_or_missing(plain) is False


def test_a_reparse_point_that_is_not_a_junction_is_not_a_link(tmp_path, monkeypatch):
    """Only the mount-point tag counts: calling a cloud placeholder directory a
    link would hide files that are really there."""
    placeholder = tmp_path / "placeholder"
    placeholder.mkdir()
    _simulate_reparse_point(monkeypatch, placeholder, tag=CLOUD_PLACEHOLDER_TAG)

    assert links.is_link(placeholder) is False


@needs_symlinks
def test_a_symlink_is_a_link_and_is_never_followed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    dangling = tmp_path / "dangling"
    os.symlink(tmp_path / "missing", dangling)

    assert links.is_link(link) is True
    assert links.is_link(dangling) is True, "a dangling link is still a link"
    with os.scandir(tmp_path) as entries:
        by_name = {entry.name: links.is_link_entry(entry) for entry in entries}
    assert by_name == {"target": False, "link": True, "dangling": True}


def test_a_path_that_cannot_be_read_is_the_callers_decision(tmp_path):
    """``is_link`` raises; ``is_link_or_missing`` answers as ``islink`` does."""
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        links.is_link(missing)
    assert links.is_link_or_missing(missing) is False
    assert links.is_link_or_missing(f"{tmp_path}/nul\0byte") is False


def test_an_entry_is_a_junction_by_its_own_lstat(tmp_path):
    junction_info = _ReparseStat(os.lstat(tmp_path), JUNCTION_TAG)
    gone = FileNotFoundError(2, "No such file or directory")

    assert links.is_link_entry(_Entry(tmp_path / "j", info=junction_info)) is True
    assert links.is_link_entry(_Entry(tmp_path, info=os.lstat(tmp_path))) is False
    with pytest.raises(FileNotFoundError):
        links.is_link_entry(_Entry(tmp_path / "gone", error=gone))


# --------------------------------------------------------------------------
# The directory guards
# --------------------------------------------------------------------------


def test_the_model_folder_tree_does_not_offer_a_junction(tmp_path, caplog):
    """Wrong if the tree offers a junction as a place, or if an entry removed
    since the listing fails the whole listing instead of being left out.

    The tree is where models are moved, renamed and removed, and a move through
    a link is refused (``resolve_subfolder``), so a link is never an offered
    directory. A link to a directory is still listed, as a read-only node with
    ``link: true`` and ``has_dirs: false`` made through ``_offered_link``,
    carrying the models registered under its name, so the counts under its
    parent add up (tests/test_model_folders_listing_and_scan_claim.py)."""
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
    into wherever the link points, which can be outside the folder or inside
    another registered folder whose rows the move does not touch."""
    (tmp_path / "Lora").mkdir()
    _simulate_reparse_point(monkeypatch, tmp_path / "Lora")

    with pytest.raises(MoveRefused) as refused:
        resolve_subfolder(str(tmp_path), "Lora")
    assert str(refused.value) == (
        "That path goes through a link or a hidden folder, which the shelf "
        "does not move into."
    )
    assert resolve_subfolder(str(tmp_path), "New")[0] == "New", (
        "a folder that does not exist yet is not a link"
    )


def test_a_backup_refuses_a_junction_inside_the_library(tmp_path, monkeypatch):
    """``os.walk`` descends a junction, so an archive walk that only refused
    symlinks would pack whatever the junction points at."""
    junction = tmp_path / "junction"
    junction.mkdir()
    _simulate_reparse_point(monkeypatch, junction)

    with pytest.raises(BackupError, match=r"it is a link \(a symlink or junction\)\.$"):
        library_backup_service._validate_regular_directory(
            str(junction), label="library payload directory"
        )
    library_backup_service._validate_regular_directory(
        str(tmp_path), label="library payload directory"
    )


def test_sidecar_detection_does_not_learn_a_convention_through_a_junction(
    tmp_path, monkeypatch
):
    """The reference scan prunes a junction, so the convention of the files it
    leads to is not this folder's. Wrong if the suffix is detected."""
    junction = tmp_path / "elsewhere"
    junction.mkdir()
    (junction / "cat.png").write_bytes(b"not read")
    (junction / "cat.tags.txt").write_text("test-tag, test-other")
    assert detect_folder_suffixes(str(tmp_path))["found_tags"] is True, (
        "the control: as a plain directory, its convention is found"
    )
    _simulate_reparse_point(monkeypatch, junction)

    detected = detect_folder_suffixes(str(tmp_path))

    assert detected["found_tags"] is False
    assert detected["tags_suffix"] is None


@pytest.mark.parametrize(
    "sweep",
    [sweep_share_copies, edit_copies.sweep_edit_sessions],
    ids=["share copies", "edit copies"],
)
def test_a_sweep_leaves_a_junction_named_like_a_batch_alone(
    tmp_path, monkeypatch, caplog, sweep
):
    """A batch folder is the sweep's own; a junction with a batch's name is
    not, and emptying it empties the folder it points at. Wrong if the
    junction's file is gone, if the sweep counts it, or if it logged a failure
    over it (on Windows ``rmtree`` refuses a junction, so a sweep that took one
    for a batch would complain at every start)."""
    root = tmp_path / "root"
    junction = root / ("a" * 32)
    stale = root / ("b" * 32)
    for batch in (junction, stale):
        batch.mkdir(parents=True)
        (batch / "test-picture.png").write_bytes(b"test-bytes")
        os.utime(batch / "test-picture.png", (0, 0))
        os.utime(batch, (0, 0))
    _simulate_reparse_point(monkeypatch, junction)
    _simulate_junction_entries(monkeypatch, root)

    with caplog.at_level("WARNING"):
        removed = sweep(str(root), max_age_seconds=60)

    assert (junction / "test-picture.png").read_bytes() == b"test-bytes"
    assert removed == 1, "the stale batch beside it is the control"
    assert not stale.exists()
    assert str(junction) not in caplog.text


def test_start_up_never_imports_an_edit_through_a_junction(tmp_path, monkeypatch):
    """At start-up every session folder is read for an edit to import. A
    junction named like a session would import whatever folder it points at.
    Wrong if the junction is processed, or the plain session is not."""
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
    _simulate_reparse_point(monkeypatch, junction)
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
    _simulate_reparse_point(monkeypatch, junction)

    with pytest.raises(
        RestoreError,
        match=r"^Refusing to restore through a link \(a symlink or junction\) at ",
    ):
        library_restore_service._require_empty_destination(str(junction))
    library_restore_service._require_empty_destination(str(plain))


def test_plugin_removal_refuses_a_junction(tmp_path, monkeypatch):
    """Removing a plugin folder that is a junction would delete the folder it
    points at, which the person did not name. Wrong if it resolves as a plugin
    to remove; a plain plugin folder is the control."""
    monkeypatch.setattr(
        plugin_install, "user_data_dir", lambda _app: str(tmp_path / "userdata")
    )
    plugins = Path(os.path.realpath(plugin_install.user_dir(IMAGE)))
    junction = plugins / "test-junction"
    junction.mkdir(parents=True)
    plain = plugins / "test-plain"
    plain.mkdir()
    real_os_name = os.name
    _simulate_reparse_point(monkeypatch, junction)
    # pathlib picks WindowsPath from ``os.name`` at every construction, and
    # none can be made on this machine. The guard reads only the lstat tag.
    monkeypatch.setattr(os, "name", real_os_name)

    with pytest.raises(PluginError, match=r"is a link \(a symlink or junction\)\."):
        plugin_install.resolve_removal("test-junction", IMAGE)
    assert plugin_install.resolve_removal("test-plain", IMAGE) == (IMAGE, plain)


def test_withdrawing_a_restore_unlinks_a_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """A failed restore takes back what it published. A junction among it is
    removed as a link (``os.unlink`` removes a junction on Windows), never
    emptied with ``rmtree``. Wrong if the junction's file is gone."""
    folder = tmp_path / "library"
    junction = folder / "test-junction"
    junction.mkdir(parents=True)
    (junction / "test-owner.png").write_bytes(b"test-bytes")
    restored = folder / "test-restored"
    restored.mkdir()
    (restored / "test-picture.png").write_bytes(b"test-bytes")
    _simulate_reparse_point(monkeypatch, junction)
    real_unlink = os.unlink
    unlinked = []

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == str(junction):
            # Recorded, not removed: Linux cannot unlink a directory.
            unlinked.append(os.fspath(path))
            return
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)

    library_restore_service._withdraw_library_folder(str(folder), remove_folder=False)

    assert unlinked == [str(junction)]
    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not restored.exists(), "the plain restored folder is the control"


def test_an_abandoned_import_leaves_a_junction_alone(tmp_path, monkeypatch, caplog):
    """An abandoned first import removes the folders it made, by name. A
    junction with one of those names is the owner's, and emptying it empties
    the folder it points at. Wrong if its file is gone or a failure was logged
    over it."""
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
    _simulate_reparse_point(monkeypatch, junction)

    with caplog.at_level("WARNING", logger=registry.logger.name):
        registry.remove_library_made_entries(library)

    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not made.exists(), "the folder the import made is the control"
    assert str(junction) not in caplog.text


def test_a_snapshot_path_through_a_junction_is_refused(tmp_path, monkeypatch):
    """A registered snapshot path is untrusted, and its containment check is
    lexical, so the walk that refuses a link in it is all that stops one
    through a junction. Wrong if the path resolves."""
    vault = tmp_path / "vault"
    for name in ("test-junction", "test-plain"):
        (vault / "snapshots" / name).mkdir(parents=True)
        (vault / "snapshots" / name / "snapshot.sqlite").write_bytes(b"test-bytes")
    root = portable_identity._snapshots_root(str(vault))
    _simulate_reparse_point(monkeypatch, os.path.join(root, "test-junction"))

    with pytest.raises(
        PortableIdentityScrubError,
        match=r"contains a link \(a symlink or junction\): ",
    ):
        portable_identity._registered_snapshot_path(
            str(vault),
            root,
            os.path.join("snapshots", "test-junction", "snapshot.sqlite"),
        )
    assert portable_identity._registered_snapshot_path(
        str(vault), root, os.path.join("snapshots", "test-plain", "snapshot.sqlite")
    ) == os.path.join(root, "test-plain", "snapshot.sqlite")


def test_a_folder_move_keeps_a_source_holding_a_junction(tmp_path, monkeypatch):
    """A cross-drive folder move deletes the source only when it lists the same
    as the copy. ``copytree`` copies a junction as a plain folder of the files
    it points at; a source listing that walked the junction the same way would
    match, and deleting the source would delete those files. Wrong if the two
    listings are equal."""
    source = tmp_path / "source"
    junction = source / "test-junction"
    junction.mkdir(parents=True)
    (junction / "test-model.safetensors").write_bytes(b"test-bytes")
    # Copied before the simulation, so the copy holds a plain folder, as
    # ``copytree`` makes of a junction.
    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    _simulate_reparse_point(monkeypatch, junction)

    listed = model_dir_mover._tree_listing(str(source))

    assert "test-junction" in listed
    assert listed != model_dir_mover._tree_listing(str(copy))


# --------------------------------------------------------------------------
# The walks that skip a directory symlink by leaving it to ``os.walk``
# --------------------------------------------------------------------------


def _picture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (32, 64, 96)).save(path)


def test_a_layout_rename_never_looks_inside_a_junction(tmp_path, monkeypatch):
    """A rename looks for the entity's folder in every directory at its depth,
    and renames what it finds. A folder found inside a junction is outside the
    library. Wrong if the junction is one of the directories searched."""
    root = tmp_path / "library"
    (root / "test-project" / "test-person").mkdir(parents=True)
    junction = root / "test-elsewhere"
    (junction / "test-person").mkdir(parents=True)
    _simulate_reparse_point(monkeypatch, junction)

    searched = layout_move_service._directories_at_depth(str(root), 1)

    assert searched == [str(root / "test-project")]


def test_a_picture_count_does_not_follow_a_junction(tmp_path, monkeypatch):
    """The folder pickers' count promises what the reference scan will index,
    and the scan prunes a junction. Wrong if the junction's picture counts."""
    _picture(tmp_path / "test-plain" / "test-a.png")
    junction = tmp_path / "test-elsewhere"
    _picture(junction / "test-b.png")
    assert count_media_files(str(tmp_path)) == (2, False), (
        "the control: as a plain directory, its picture counts"
    )
    _simulate_reparse_point(monkeypatch, junction)

    assert count_media_files(str(tmp_path)) == (1, False)


def test_the_folder_structure_read_lists_a_junction_but_never_walks_it(
    tmp_path, monkeypatch
):
    """The read walks a caller-supplied root, so a junction loop would walk
    until the folder cap. Wrong if the junction is mapped or its picture
    counted, or if the root stops counting it as a child, as it counts a
    directory symlink."""
    root = tmp_path / "test-root"
    _picture(root / "test-plain" / "test-a.png")
    junction = root / "test-elsewhere"
    _picture(junction / "test-b.png")
    _simulate_reparse_point(monkeypatch, junction)

    result = FolderStructureRead(str(root)).run()

    assert result["folder_count"] == 2, "the root and its plain folder"
    assert result["picture_count"] == 1
    (root_level,) = (level for level in result["levels"] if level["depth"] == 1)
    assert root_level["folders"][0]["child_count"] == 2


def test_a_watch_folder_scan_never_offers_a_file_through_a_junction(
    tmp_path, monkeypatch
):
    """A watched file can be imported and then deleted, and a junction inside
    the watch folder leads out of it. Wrong if the file behind the junction is
    offered, or the plain one beside it is not."""
    watch = tmp_path / "test-watch"
    _picture(watch / "test-plain" / "test-a.png")
    junction = watch / "test-elsewhere"
    _picture(junction / "test-b.png")
    _simulate_reparse_point(monkeypatch, junction)
    finder = missing_watch_folder_import_finder.MissingWatchFolderImportFinder(
        database=mock.Mock()
    )

    finder._find_task_from_db([ImportFolder(id=1, folder=str(watch), label="test")])

    def offered(path):
        return os.path.normcase(os.path.abspath(path)) in finder._pending_stats

    assert offered(watch / "test-plain" / "test-a.png")
    assert not offered(junction / "test-b.png")


def test_a_folder_structure_commit_never_imports_a_file_through_a_junction(
    tmp_path, monkeypatch
):
    """A local-import commit makes every picture under the chosen root a
    managed picture of this library, and a junction under that root leads out
    of it. Wrong if the file behind the junction is imported, or the plain one
    beside it is not."""
    library = tmp_path / "test-library"
    library.mkdir()
    vault_db = VaultDatabase(str(library / "vault.db"))
    try:
        _picture(library / "test-plain" / "test-a.png")
        junction = library / "test-elsewhere"
        _picture(junction / "test-b.png")
        real_os_name = os.name
        _simulate_reparse_point(monkeypatch, junction)
        # pathlib picks WindowsPath from ``os.name`` at every construction, and
        # the import's thumbnail and database steps build paths. The walk's
        # guard reads only the lstat tag.
        monkeypatch.setattr(os, "name", real_os_name)
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


def test_a_relocation_leaves_a_junction_in_the_vacated_folder_alone(
    tmp_path, monkeypatch
):
    """A relocation carries every file in PixlStash's download folder across,
    then removes the empty folders it left. A junction in that folder is the
    owner's and leads out of it, and a directory symlink there is already left
    alone. Wrong if the file behind the junction moved or the empty folder
    behind it is gone."""
    source = tmp_path / "test-source"
    (source / "test-plain").mkdir(parents=True)
    (source / "test-plain" / "test-tagger.onnx").write_bytes(b"test-bytes")
    junction = source / "test-elsewhere"
    (junction / "test-empty").mkdir(parents=True)
    (junction / "test-owner.onnx").write_bytes(b"test-bytes")
    destination = tmp_path / "test-destination"
    real_os_name = os.name
    _simulate_reparse_point(monkeypatch, junction)
    # pathlib picks WindowsPath from ``os.name`` at every construction, and
    # none can be made on this machine. The walks' guards read only the tag.
    monkeypatch.setattr(os, "name", real_os_name)

    model_moves._move_leftovers(str(source), str(destination))
    model_moves._remove_if_empty(str(source))

    assert (junction / "test-owner.onnx").read_bytes() == b"test-bytes"
    assert (junction / "test-empty").is_dir()
    assert (destination / "test-plain" / "test-tagger.onnx").read_bytes() == (
        b"test-bytes"
    ), "the plain file is the control"
    assert not (source / "test-plain").exists(), "the vacated folder is the control"


# --------------------------------------------------------------------------
# The SPA catch-all, which serves files from the built frontend
# --------------------------------------------------------------------------

SPA_DOCUMENT = "<!doctype html><title>test-spa</title>"


@pytest.fixture
def spa(tmp_path):
    """A Server with a stand-in built frontend, as
    tests/test_server_http_contract.py gives one, and a client without a
    session (the catch-all needs none)."""
    dist = tmp_path / "test-dist"
    dist.mkdir()
    (dist / "index.html").write_text(SPA_DOCUMENT)
    (dist / "test-asset.txt").write_text("test-asset-file")
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"port": 8000}))
    server = Server(str(config))
    server._get_frontend_dist_dir = lambda: str(dist)
    try:
        yield server, dist, TestClient(server.api, raise_server_exceptions=False)
    finally:
        server.close()


def _served(client, path):
    # ``request``, not ``get``: the suite's conftest prefixes a ``get`` path
    # with the API prefix, which would never reach the catch-all.
    response = client.request("GET", path)
    return response.status_code, response.text


def test_the_spa_never_serves_a_file_through_a_junction_in_the_build(spa, monkeypatch):
    """A junction planted inside the built frontend leads out of it, and the
    ``/assets`` mount already refuses one (it resolves the path first). Wrong
    if the file behind the junction is served; the plain file is the control."""
    _server, dist, client = spa
    junction = dist / "test-elsewhere"
    junction.mkdir()
    (junction / "test-outside.txt").write_text("test-outside-file")
    real_os_name = os.name
    _simulate_reparse_point(monkeypatch, junction)
    # pathlib picks WindowsPath from ``os.name`` at every construction, and
    # none can be made on this machine. The guard reads only the lstat tag.
    monkeypatch.setattr(os, "name", real_os_name)

    assert _served(client, "/test-elsewhere/test-outside.txt") == (200, SPA_DOCUMENT)
    assert _served(client, "/test-asset.txt") == (200, "test-asset-file")


@needs_symlinks
def test_the_spa_never_serves_a_file_through_a_symlink_in_the_build(spa, tmp_path):
    """A folder or file symlink planted inside the built frontend is refused
    the same way. Wrong if the file it leads to is served, or if a build that
    is itself reached through a link stops serving its own files."""
    server, dist, client = spa
    outside = tmp_path / "test-outside"
    outside.mkdir()
    (outside / "test-secret.txt").write_text("test-outside-file")
    (dist / "test-folder-link").symlink_to(outside, target_is_directory=True)
    (dist / "test-file-link.txt").symlink_to(outside / "test-secret.txt")

    assert _served(client, "/test-folder-link/test-secret.txt") == (200, SPA_DOCUMENT)
    assert _served(client, "/test-file-link.txt") == (200, SPA_DOCUMENT)
    linked_build = tmp_path / "test-dist-link"
    linked_build.symlink_to(dist, target_is_directory=True)
    server._get_frontend_dist_dir = lambda: str(linked_build)
    assert _served(client, "/test-asset.txt") == (200, "test-asset-file")


# --------------------------------------------------------------------------
# The guards that already ask utils/links.py, one junction each
# --------------------------------------------------------------------------


def _simulate_junction(monkeypatch, path) -> None:
    """:func:`_simulate_reparse_point`, with the platform read back as itself.

    pathlib picks WindowsPath from ``os.name`` at every construction, and none
    can be made on this machine. The guards read only the lstat tag.
    """
    real_os_name = os.name
    _simulate_reparse_point(monkeypatch, path)
    monkeypatch.setattr(os, "name", real_os_name)


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


def test_a_junction_as_the_snapshots_folder_is_not_trusted(tmp_path, monkeypatch):
    """The identity scrub rewrites and removes files below ``snapshots``. A
    junction there leads out of the library. Wrong if the scrub accepts it;
    the plain folder is the control."""
    vault = tmp_path / "vault"
    (vault / "snapshots").mkdir(parents=True)
    snapshots = os.path.join(os.path.realpath(vault), "snapshots")
    assert portable_identity._snapshots_root(str(vault)) == snapshots
    _simulate_junction(monkeypatch, snapshots)

    with pytest.raises(PortableIdentityScrubError, match="not a trusted directory"):
        portable_identity._snapshots_root(str(vault))


def test_the_stale_scrub_sweep_never_walks_a_junction(tmp_path, monkeypatch):
    """An interrupted scrub leaves plaintext in scrub folders below
    ``snapshots``, and the sweep removes every one it finds. One reached
    through a junction is outside the library. Wrong if the folder behind the
    junction is gone; the plain stale folder is the control."""
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


def _switching_server(*, export_tasks=None, staging_sessions=None):
    """What a library switch clears off the server it is switching."""
    return SimpleNamespace(
        export_tasks=export_tasks or {},
        import_tasks={},
        staging_sessions=staging_sessions or {},
        auth=SimpleNamespace(clear_guest_session_tracking=lambda: None),
    )


def test_a_library_switch_unlinks_an_export_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """A switch discards the old library's exports. A junction named as an
    export's private folder is removed as a link, never emptied with
    ``rmtree``. Wrong if the junction's file is gone; the plain export folder
    is the control."""
    previous = tmp_path / "test-library"
    previous.mkdir()
    junction = tmp_path / "pixlstash_export_test-junction"
    plain = tmp_path / "pixlstash_export_test-plain"
    for folder in (junction, plain):
        folder.mkdir()
        (folder / "test-owner.png").write_bytes(b"test-bytes")
    server = _switching_server(
        export_tasks={
            folder.name: {"private_dir": str(folder)} for folder in (junction, plain)
        }
    )
    _simulate_junction(monkeypatch, junction)
    unlinked = _record_unlinks(monkeypatch, junction)

    LibrarySwitchService(server)._clear_generation_retained_state(str(previous))

    assert unlinked == [str(junction)]
    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not plain.exists(), "the plain export folder is the control"


def test_a_library_switch_never_empties_a_session_below_a_staging_junction(
    tmp_path, monkeypatch
):
    """A switch discards upload sessions only below the old library's own
    ``.staging``. A ``.staging`` that is a junction leads out of the library,
    so no session below it is removed. Wrong if the session's file is gone."""
    previous = tmp_path / "test-library"
    session = previous / ".staging" / "test-session"
    session.mkdir(parents=True)
    (session / "test-owner.png").write_bytes(b"test-bytes")
    server = _switching_server(
        staging_sessions={"test-session": {"staging_dir": str(session)}}
    )
    _simulate_junction(monkeypatch, previous / ".staging")

    LibrarySwitchService(server)._clear_generation_retained_state(str(previous))

    assert (session / "test-owner.png").read_bytes() == b"test-bytes"
    assert server.staging_sessions == {}


def test_a_library_switch_unlinks_a_session_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """A session's staging folder that is a junction is removed as a link,
    never emptied with ``rmtree``. Wrong if the junction's file is gone; the
    plain session folder is the control."""
    previous = tmp_path / "test-library"
    junction = previous / ".staging" / "test-junction"
    plain = previous / ".staging" / "test-plain"
    for folder in (junction, plain):
        folder.mkdir(parents=True)
        (folder / "test-owner.png").write_bytes(b"test-bytes")
    server = _switching_server(
        staging_sessions={
            folder.name: {"staging_dir": str(folder)} for folder in (junction, plain)
        }
    )
    _simulate_junction(monkeypatch, junction)
    unlinked = _record_unlinks(monkeypatch, junction)

    LibrarySwitchService(server)._clear_generation_retained_state(str(previous))

    assert unlinked == [str(junction)]
    assert (junction / "test-owner.png").read_bytes() == b"test-bytes"
    assert not plain.exists(), "the plain session folder is the control"


def test_a_merge_does_not_claim_previews_behind_a_junction(
    tmp_path, monkeypatch, caplog
):
    """A merge that removes a duplicate copy logs that the copy's previews
    folder is still the model's. A junction there leads elsewhere, and is not
    named as the model's. Wrong if it is; the plain folder is the control."""
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
    A junction there leads elsewhere, and what it points at is not the model's
    to delete. Wrong if the file behind it is gone; the plain folder is the
    control."""
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


def test_a_downloaded_export_unlinks_a_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """A finished export's private folder goes once its download is sent. A
    junction named as that folder is removed as a link, never emptied with
    ``rmtree``. Wrong if the junction's file is gone; the plain export folder
    is the control."""
    lease = SimpleNamespace(library_uuid="test-uuid", generation=1)
    request = SimpleNamespace(state=SimpleNamespace(library_lease=lease))
    server = SimpleNamespace(export_tasks={})
    router = APIRouter()
    export_routes.register_routes(router, server)
    (download,) = [
        route.endpoint
        for route in router.routes
        if route.path == "/pictures/export/download/{task_id}"
    ]
    junction = tmp_path / "pixlstash_export_test-junction"
    plain = tmp_path / "pixlstash_export_test-plain"
    for folder in (junction, plain):
        folder.mkdir()
        (folder / "test-export.zip").write_bytes(b"test-bytes")
        server.export_tasks[folder.name] = {
            "status": "completed",
            "library_uuid": "test-uuid",
            "generation": 1,
            "file_path": str(folder / "test-export.zip"),
            "private_dir": str(folder),
        }
    _simulate_junction(monkeypatch, junction)
    unlinked = _record_unlinks(monkeypatch, junction)

    for folder in (junction, plain):
        sent = download(request, folder.name).background
        sent.func(*sent.args, **sent.kwargs)

    assert unlinked == [str(junction)]
    assert (junction / "test-export.zip").read_bytes() == b"test-bytes"
    assert not plain.exists(), "the plain export folder is the control"
    assert server.export_tasks == {}


def test_installing_the_nodes_refuses_a_junction_in_their_place(tmp_path, monkeypatch):
    """The install trashes the older copies of the nodes, then moves the new
    one into ``custom_nodes/ComfyUI-PixlStash``. A junction there leads out of
    ComfyUI's folder and is refused before anything is trashed, as a symlink
    is. Wrong if the install goes ahead or trashes anything."""
    source = tmp_path / "test-bundled" / comfyui_pack_service.PACK_FOLDER
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# test-bundled\n")
    custom_nodes = tmp_path / "custom_nodes"
    junction = custom_nodes / comfyui_pack_service.PACK_FOLDER
    junction.mkdir(parents=True)
    (junction / "test-owner.txt").write_text("test-owner-file")
    trashed: list[str] = []
    monkeypatch.setattr(
        comfyui_link_service, "locate", lambda url: ("this_computer", url)
    )
    monkeypatch.setattr(comfyui_pack_service, "bundled_pack", lambda: source)
    monkeypatch.setattr(
        comfyui_pack_service, "_custom_nodes_folder", lambda pinned: custom_nodes
    )
    monkeypatch.setattr(comfyui_pack_service, "send2trash", trashed.append)
    monkeypatch.setattr(comfyui_pack_service, "restart_comfyui", lambda pinned: None)
    _simulate_junction(monkeypatch, junction)

    with pytest.raises(
        comfyui_pack_service.PackInstallRefused, match="is a file or a link"
    ):
        comfyui_pack_service.install("http://127.0.0.1:8188/")

    assert trashed == []
    assert (junction / "test-owner.txt").read_text() == "test-owner-file"
    assert os.listdir(custom_nodes) == [comfyui_pack_service.PACK_FOLDER]


def _installable_pack(tmp_path, monkeypatch):
    """A bundled copy of the nodes and an empty ``custom_nodes`` the install
    writes into, with ComfyUI, the trash and the restart stood in for. Returns
    ``(custom_nodes, trashed)``; ``trashed`` lists what went to the trash."""
    source = tmp_path / "test-bundled" / comfyui_pack_service.PACK_FOLDER
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# test-bundled\n")
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    trashed: list[str] = []
    monkeypatch.setattr(
        comfyui_link_service, "locate", lambda url: ("this_computer", url)
    )
    monkeypatch.setattr(comfyui_pack_service, "bundled_pack", lambda: source)
    monkeypatch.setattr(
        comfyui_pack_service, "_custom_nodes_folder", lambda pinned: custom_nodes
    )
    monkeypatch.setattr(comfyui_pack_service, "send2trash", trashed.append)
    monkeypatch.setattr(comfyui_pack_service, "restart_comfyui", lambda pinned: None)
    return custom_nodes, trashed


def test_installing_the_nodes_unlinks_a_staging_junction_and_never_empties_it(
    tmp_path, monkeypatch
):
    """The install clears ``.ComfyUI-PixlStash.installing`` before it stages a
    copy there. A junction at that name is unlinked as a link; emptying it
    would delete what it points at. Wrong if the folder behind it loses a file
    or the junction is never unlinked."""
    custom_nodes, _trashed = _installable_pack(tmp_path, monkeypatch)
    junction = custom_nodes / f".{comfyui_pack_service.PACK_FOLDER}.installing"
    junction.mkdir()
    (junction / "test-owner.txt").write_text("test-owner-file")
    behind = tmp_path / "test-behind-the-junction"
    real_unlink = os.unlink
    unlinked: list[str] = []

    def unlink(candidate, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(junction):
            # Unlinking a junction frees its name and leaves the folder it
            # points at. Linux cannot unlink a directory, so the folder is
            # moved aside to stand for that.
            unlinked.append(os.fspath(candidate))
            os.rename(junction, behind)
            return
        real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    _simulate_junction(monkeypatch, junction)

    result = comfyui_pack_service.install("http://127.0.0.1:8188/")

    assert unlinked == [os.fspath(junction)]
    assert (behind / "test-owner.txt").read_text() == "test-owner-file"
    installed = custom_nodes / comfyui_pack_service.PACK_FOLDER
    assert result["installed_to"] == str(installed)
    assert result["kept_links"] == []
    assert (installed / "__init__.py").is_file()


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_installing_the_nodes_keeps_a_link_to_another_copy(
    tmp_path, monkeypatch, caplog, kind
):
    """The install trashes every older copy of the nodes in ``custom_nodes``,
    under any name. A link there, such as a development checkout linked in as
    ``ComfyUI-PixlStash-dev``, is kept and logged: trashing it is not
    PixlStash's call, and the copy behind it is outside ComfyUI's folder. An
    older copy that is a real folder still goes to the trash. Wrong if the
    link is trashed or the older folder is not."""
    custom_nodes, trashed = _installable_pack(tmp_path, monkeypatch)
    older = custom_nodes / "comfyui-pixlstash"
    older.mkdir()
    (older / "__init__.py").write_text("# test-older\n")
    linked = custom_nodes / f"{comfyui_pack_service.PACK_FOLDER}-dev"
    if kind == "symlink":
        checkout = tmp_path / "test-dev-checkout"
        checkout.mkdir()
        (checkout / "test-owner.txt").write_text("test-owner-file")
        linked.symlink_to(checkout, target_is_directory=True)
    else:
        linked.mkdir()
        (linked / "test-owner.txt").write_text("test-owner-file")
        _simulate_junction(monkeypatch, linked)

    with caplog.at_level("WARNING", logger=comfyui_pack_service.logger.name):
        result = comfyui_pack_service.install("http://127.0.0.1:8188/")

    assert trashed == [str(older)]
    assert result["replaced"] == [str(older)]
    assert result["kept_links"] == [str(linked)]
    assert (linked / "test-owner.txt").read_text() == "test-owner-file"
    assert str(linked) in caplog.text, "the kept link did not reach the log"
    assert (custom_nodes / comfyui_pack_service.PACK_FOLDER / "__init__.py").is_file()
