"""A move's copy whose row does not commit leaves no file behind, and the
failure that stopped it is the one raised.

The copy is published into the other library before its row is inserted, and
recorded in an :class:`~pixlstash.services.import_publish_service.PublishedPictures`,
which removes the file and its thumbnail when the insert fails. Removing them
can fail too; that is logged, and the insert's own error still propagates.
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pixlstash.routes.pictures import _library_move
from pixlstash.utils.image_processing.image_utils import ImageUtils

FINGERPRINT = ("example-sampled-sha", 14, "example-full-sha")


def _failing_copy(tmp_path, monkeypatch):
    """A copy into a target whose insert fails, with its file and thumbnail
    already published. Returns ``(target, source_root, row, copy, thumbnail,
    persist_error)``."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "source.png").write_bytes(b"source picture")
    target_root = tmp_path / "target"
    target_root.mkdir()
    copy = target_root / "copy.png"
    copy.write_bytes(b"copied picture")
    thumbnail = ImageUtils.get_thumbnail_path(str(target_root), "copy.png")
    os.makedirs(os.path.dirname(thumbnail), exist_ok=True)
    with open(thumbnail, "wb") as handle:
        handle.write(b"thumbnail")
    picture = SimpleNamespace(
        file_path="copy.png",
        pixel_sha=FINGERPRINT[0],
        size_bytes=FINGERPRINT[1],
        created_at=None,
    )
    row = SimpleNamespace(
        id=7,
        file_path="source.png",
        original_file_name="original.png",
        created_at=None,
        score=3,
        description="A picture",
        tags=[],
    )
    persist_error = RuntimeError("example-target-persistence-failed")
    target = SimpleNamespace(
        image_root=str(target_root),
        db=SimpleNamespace(
            image_root=str(target_root),
            # The look again under the lock: no candidate in the target.
            run_immediate_read_task=Mock(return_value={}),
            run_task=Mock(side_effect=persist_error),
        ),
    )
    monkeypatch.setattr(
        _library_move,
        "_create_picture_imports",
        Mock(return_value=([FINGERPRINT], {}, {}, {FINGERPRINT: picture})),
    )
    return target, str(source_root), row, copy, thumbnail, persist_error


def test_a_failed_insert_removes_the_copy_and_its_thumbnail(tmp_path, monkeypatch):
    target, source_root, row, copy, thumbnail, persist_error = _failing_copy(
        tmp_path, monkeypatch
    )

    with pytest.raises(RuntimeError) as caught:
        _library_move._copy_into(target, source_root, row)

    assert caught.value is persist_error
    target.db.run_task.assert_called_once()
    assert not copy.exists()
    assert not os.path.exists(thumbnail)


def test_a_copy_that_cannot_be_removed_is_logged_and_the_insert_error_kept(
    tmp_path, monkeypatch, caplog
):
    target, source_root, row, copy, thumbnail, persist_error = _failing_copy(
        tmp_path, monkeypatch
    )
    real_remove = os.remove

    def refuse_the_copy(path, *args, **kwargs):
        if os.fspath(path) == str(copy):
            raise PermissionError(13, "example-permission-denied", str(copy))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", refuse_the_copy)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as caught:
            _library_move._copy_into(target, source_root, row)

    assert caught.value is persist_error
    assert copy.exists()
    # The thumbnail is removed on its own; only the copy stayed.
    assert not os.path.exists(thumbnail)
    kept = [
        record.getMessage()
        for record in caplog.records
        if record.name.endswith("import_publish_service")
        and record.levelname == "ERROR"
    ]
    assert any(
        "copy.png" in message and "example-target-persistence-failed" in message
        for message in kept
    ), kept
