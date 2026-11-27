"""Whether a path is a link: a symbolic link, or a Windows directory junction.

``os.path.islink`` and ``DirEntry.is_symlink`` are False for a junction, yet
``realpath`` resolves one and ``os.walk`` descends one, and making one needs no
privilege. ``os.path.isjunction`` is 3.12+ and the floor is 3.11, so the test is
spelled as 3.12 spells it: the ``lstat`` reparse tag is
``IO_REPARSE_TAG_MOUNT_POINT``. Only that tag counts: a cloud placeholder
directory is a reparse point too, and calling it a link would hide real files.
``os.lstat`` is looked up at call time so tests can give a directory the tag.
"""

import os
import stat

# ``IO_REPARSE_TAG_MOUNT_POINT``; the ``stat`` constant exists only on Windows.
JUNCTION_REPARSE_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _is_junction_stat(info: os.stat_result) -> bool:
    """Whether an ``lstat`` result is a Windows directory junction."""
    return getattr(info, "st_reparse_tag", 0) == JUNCTION_REPARSE_TAG


def is_link_stat(info: os.stat_result) -> bool:
    """Whether an ``lstat`` result is a symbolic link or a directory junction."""
    return stat.S_ISLNK(info.st_mode) or _is_junction_stat(info)


def is_link(path: str | os.PathLike) -> bool:
    """Whether *path* itself is a symbolic link or a directory junction.

    Raises:
        OSError: *path* cannot be ``lstat``-ed, including when it is missing.
    """
    return is_link_stat(os.lstat(path))


def is_link_or_missing(path: str | os.PathLike) -> bool:
    """:func:`is_link`, but False when *path* cannot be ``lstat``-ed.

    What ``os.path.islink`` answers for a missing path (or one ``lstat``
    refuses with ``ValueError``, such as one holding a NUL byte), for callers
    to which a missing component is not a redirect.
    """
    try:
        return is_link(path)
    except (OSError, ValueError):
        return False


def is_link_entry(entry: os.DirEntry) -> bool:
    """Whether a directory entry is a symbolic link or a directory junction.

    Raises:
        OSError: the entry cannot be ``lstat``-ed (removed since the listing).
    """
    if entry.is_symlink():
        return True
    return _is_junction_stat(entry.stat(follow_symlinks=False))
