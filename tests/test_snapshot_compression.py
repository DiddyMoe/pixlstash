"""Unit tests for the snapshot zstd compression helpers."""

import os

import pytest

from pixlstash.utils.snapshot_compression import (
    COMPRESSED_SUFFIX,
    compress_snapshot,
    decompress_snapshot,
    is_compressed,
    materialize_snapshot,
)


def _sample_bytes() -> bytes:
    # A header of random bytes plus a long compressible run - exercises both
    # that compression runs and that the round-trip is byte-exact.
    return os.urandom(2048) + (b"pixlstash" * 20000)


def test_is_compressed_suffix():
    assert is_compressed("a/b/c.sqlite.zst")
    assert is_compressed(f"x{COMPRESSED_SUFFIX}")
    assert not is_compressed("a/b/c.sqlite")


def test_compress_decompress_roundtrip(tmp_path):
    data = _sample_bytes()
    src = tmp_path / "snap.sqlite"
    src.write_bytes(data)

    archive = tmp_path / f"snap{COMPRESSED_SUFFIX}"
    size = compress_snapshot(str(src), str(archive))
    assert size == os.path.getsize(archive) > 0
    # The compressible payload must actually shrink.
    assert size < len(data), "Archive should be smaller than the source"

    out = tmp_path / "out.sqlite"
    decompress_snapshot(str(archive), str(out))
    assert out.read_bytes() == data, "Round-trip must be byte-exact"


def test_materialize_handles_compressed_and_plain(tmp_path):
    data = _sample_bytes()
    src = tmp_path / "snap.sqlite"
    src.write_bytes(data)
    archive = tmp_path / f"snap{COMPRESSED_SUFFIX}"
    compress_snapshot(str(src), str(archive))

    # Compressed → decompresses.
    from_zst = tmp_path / "from_zst.sqlite"
    materialize_snapshot(str(archive), str(from_zst))
    assert from_zst.read_bytes() == data

    # Plain → copies byte-for-byte.
    from_plain = tmp_path / "from_plain.sqlite"
    materialize_snapshot(str(src), str(from_plain))
    assert from_plain.read_bytes() == data


class _TestCancelled(Exception):
    """Raised by the test's cancellation probe."""


def test_a_cancel_between_chunks_leaves_no_archive_and_no_temp_file(tmp_path):
    src = tmp_path / "snap.sqlite"
    src.write_bytes(os.urandom(3 << 20))  # three 1 MiB chunks
    archive = tmp_path / f"snap{COMPRESSED_SUFFIX}"
    polls = []

    def _cancel_on_second_poll():
        polls.append(len(polls))
        if len(polls) == 2:
            raise _TestCancelled("test-cancelled")

    with pytest.raises(_TestCancelled):
        compress_snapshot(
            str(src), str(archive), check_cancelled=_cancel_on_second_poll
        )
    assert polls == [0, 1], "the cancel must be polled before each chunk"
    assert sorted(os.listdir(tmp_path)) == ["snap.sqlite"]


def test_a_finished_archive_is_published_without_a_temp_file(tmp_path):
    src = tmp_path / "snap.sqlite"
    src.write_bytes(_sample_bytes())
    archive = tmp_path / f"snap{COMPRESSED_SUFFIX}"
    archive.write_bytes(b"test-previous-archive")

    compress_snapshot(str(src), str(archive))

    assert sorted(os.listdir(tmp_path)) == ["snap.sqlite", archive.name]
    if os.name != "nt":
        assert os.stat(archive).st_mode & 0o777 == 0o600
    out = tmp_path / "out.sqlite"
    decompress_snapshot(str(archive), str(out))
    assert out.read_bytes() == src.read_bytes()
