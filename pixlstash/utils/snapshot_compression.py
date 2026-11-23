"""Whole-file zstd compression for vault snapshot archives.

Snapshots are full SQLite copies of the vault DB. To keep the expensive,
GPU-regenerated blobs (CLIP image/text embeddings, InsightFace face
features) inside the snapshot *without* the on-disk cost, the ``.sqlite``
file is compressed with zstd at rest and decompressed to a temp file only
when it is actually read (restore / preview).

SQLite cannot query a compressed file in place - its pager needs random,
seekable page access - so a snapshot is treated as an archive: compress on
creation, decompress to a scratch ``.sqlite`` before any DB work. This module
is the single place that knows the on-disk format.

Streaming (chunked) compress/decompress is used so a multi-GB snapshot never
has to be held in memory.

An archive is published all or nothing: it is written to a private temp file
beside its final name, flushed to disk, and renamed into place, so a crash or a
cancel never leaves a truncated archive under the name a snapshot row points at.
"""

import os
import shutil
import tempfile
from typing import BinaryIO, Callable, Optional

import zstandard

from pixlstash.pixl_logging import get_logger

logger = get_logger(__name__)

# On-disk suffixes. New snapshots are written compressed; legacy snapshots
# created before compression remain plain ``.sqlite`` and are detected by the
# absence of the ``.zst`` suffix so they still restore.
COMPRESSED_SUFFIX = ".sqlite.zst"
LEGACY_SUFFIX = ".sqlite"

# Compression level. Level 3 is zstd's default: it gave a ~3x ratio on real
# embedding-heavy snapshots at ~0.03s for a 14 MB file (≈8x faster than gzip
# for an essentially identical ratio). Higher levels buy little here because
# float32 embeddings are close to incompressible - the win is structural.
_COMPRESSION_LEVEL = 3

# Chunk size for streaming copy_stream (1 MiB).
_CHUNK_SIZE = 1 << 20


# Scratch subdirectory (under the vault's snapshots dir) for the full,
# uncompressed DB written during snapshot creation and decompressed during
# restore/preview.
_SCRATCH_DIRNAME = ".tmp"


def snapshot_scratch_dir(vault_root: str) -> str:
    """Return (creating if needed) the on-vault scratch dir for snapshot work.

    The scratch ``.sqlite`` is the **full** vault DB - now that embeddings are
    retained it can be hundreds of MB to several GB. Keeping it on the vault
    filesystem rather than the system ``/tmp`` avoids ENOSPC / RAM blow-ups when
    ``/tmp`` is a small tmpfs or root partition, which is exactly the failure
    mode on the large, embedding-heavy vaults this feature targets. The
    snapshots dir is already excluded from the orphan-file scanner, so the
    scratch dir under it is ignored too.

    Args:
        vault_root: The vault image root.

    Returns:
        Absolute path to ``<vault_root>/snapshots/.tmp``.
    """
    path = os.path.join(vault_root, "snapshots", _SCRATCH_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def is_compressed(path: str) -> bool:
    """Return True if *path* names a zstd-compressed snapshot archive.

    Args:
        path: Snapshot file path (absolute or relative).

    Returns:
        True when the path ends with ``.sqlite.zst``.
    """
    return path.endswith(".zst")


def write_file_atomically(dst_path: str, write: Callable[[BinaryIO], object]) -> None:
    """Write *dst_path* through a private temp file and publish it with a rename.

    *write* fills a temp file created (mode 0600) in the destination's
    directory. The temp file is flushed and fsynced, renamed over *dst_path*,
    and the directory is fsynced so the rename survives a power loss. If
    *write* raises, or anything before the rename fails, the temp file is
    removed and *dst_path* is left exactly as it was.

    Args:
        dst_path: Final path of the file.
        write: Callable that writes the full contents to the binary handle it
            is given; what it returns is ignored. It may raise to abandon the
            write.

    Raises:
        OSError: If the temp file cannot be written, renamed or synced.
        Exception: Whatever *write* raises.
    """
    directory = os.path.dirname(dst_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst_path)}.", suffix=".partial", dir=directory
    )
    published = False
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dst_path)
        published = True
        _fsync_directory(directory)
    finally:
        if not published:
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning(
                    "snapshot_compression: could not remove the unpublished temp "
                    "file %s for %s; it stays on disk until removed by hand: %s",
                    tmp_path,
                    dst_path,
                    exc,
                )


def compress_snapshot(
    src_sqlite_path: str,
    dst_zst_path: str,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> int:
    """Compress a plain SQLite file to a zstd archive.

    Streams the source through zstd so the whole DB is never buffered in
    memory, and publishes the archive with :func:`write_file_atomically`, so
    *dst_zst_path* either holds the complete archive or is left untouched.

    Args:
        src_sqlite_path: Path to the source ``.sqlite`` file.
        dst_zst_path: Destination path for the ``.sqlite.zst`` archive.
        check_cancelled: Optional callable polled before each chunk is read.
            It raises to abandon the compression; nothing is published then.

    Returns:
        Size of the written archive in bytes.

    Raises:
        OSError: If the source cannot be read or the destination written.
        zstandard.ZstdError: If compression fails.
    """
    cctx = zstandard.ZstdCompressor(level=_COMPRESSION_LEVEL)

    def _write_archive(dst: BinaryIO) -> None:
        with (
            open(src_sqlite_path, "rb") as src,
            cctx.stream_writer(dst, write_size=_CHUNK_SIZE, closefd=False) as writer,
        ):
            while True:
                if check_cancelled is not None:
                    check_cancelled()
                chunk = src.read(_CHUNK_SIZE)
                if not chunk:
                    break
                writer.write(chunk)

    write_file_atomically(dst_zst_path, _write_archive)
    size = os.path.getsize(dst_zst_path)
    logger.info(
        "snapshot_compression: compressed %s (%d bytes) → %s (%d bytes)",
        src_sqlite_path,
        os.path.getsize(src_sqlite_path),
        dst_zst_path,
        size,
    )
    return size


def decompress_snapshot(src_zst_path: str, dst_sqlite_path: str) -> None:
    """Decompress a zstd snapshot archive to a plain SQLite file.

    Streams so a multi-GB snapshot is never fully buffered.

    Args:
        src_zst_path: Path to the ``.sqlite.zst`` archive.
        dst_sqlite_path: Destination path for the decompressed ``.sqlite``.

    Raises:
        OSError: If the source cannot be read or the destination written.
        zstandard.ZstdError: If the archive is corrupt.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(src_zst_path, "rb") as src, open(dst_sqlite_path, "wb") as dst:
        dctx.copy_stream(src, dst, read_size=_CHUNK_SIZE, write_size=_CHUNK_SIZE)
    logger.debug(
        "snapshot_compression: decompressed %s → %s (%d bytes)",
        src_zst_path,
        dst_sqlite_path,
        os.path.getsize(dst_sqlite_path),
    )


def materialize_snapshot(src_path: str, dst_sqlite_path: str) -> None:
    """Produce a usable plain ``.sqlite`` at *dst_sqlite_path* from *src_path*.

    Decompresses when *src_path* is a zstd archive; otherwise copies the
    legacy plain ``.sqlite`` byte-for-byte. Either way the caller is left with
    an independent, writable SQLite file it can alembic-upgrade and read.

    Args:
        src_path: On-disk snapshot path (``.sqlite`` or ``.sqlite.zst``).
        dst_sqlite_path: Destination for the plain ``.sqlite`` copy.

    Raises:
        OSError / zstandard.ZstdError: On read/write/decompression failure.
    """
    if is_compressed(src_path):
        decompress_snapshot(src_path, dst_sqlite_path)
    else:
        shutil.copy2(src_path, dst_sqlite_path)


def _fsync_directory(path: str) -> None:
    """Flush a directory entry change (a rename) to disk.

    ``os.open`` cannot open a directory on Windows, so this is a no-op there,
    as in the other publish helpers (``library_backup_service``,
    ``portable_identity``).

    Args:
        path: Directory whose entries changed.
    """
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
