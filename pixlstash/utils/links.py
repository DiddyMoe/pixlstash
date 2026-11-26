"""Whether a path is a link: a symbolic link, or a Windows directory junction.

**Why this module exists.** ``os.path.islink`` and ``DirEntry.is_symlink`` are
False for a Windows directory junction, yet ``os.path.realpath`` resolves one
and ``os.walk`` walks into one. A check that asks only ``islink`` therefore
refuses the redirect that needs a privilege (a symlink needs
``SeCreateSymbolicLinkPrivilege``: an administrator or Developer Mode) and
accepts the one that needs nothing but write access to the directory
(``mklink /J``). ``os.path.isjunction`` and ``DirEntry.is_junction`` answer it,
but only from Python 3.12, and the floor is 3.11 (``pyproject.toml``).

So the test is spelled out here as 3.12 spells it
(``ntpath.isjunction``): the ``lstat`` of the path carries the reparse tag
``IO_REPARSE_TAG_MOUNT_POINT``. Only that tag counts, not every reparse point:
a cloud-storage placeholder directory is a reparse point too, and calling it a
link would hide files that are really there. ``st_reparse_tag`` exists only on
Windows, so on any other platform these functions answer exactly as
``islink`` / ``is_symlink`` do. The tests simulate ``nt`` on Linux by giving a
directory's ``lstat`` the tag, so ``os.lstat`` is looked up at call time.

A junction can only point at a directory, so a check on a *file* needs only
the symlink test; these helpers are for the places a directory is checked.

What a caller does when the path cannot be read is the caller's decision:
:func:`is_link` raises, :func:`is_link_or_missing` does not.
"""

import os
import stat

# ``IO_REPARSE_TAG_MOUNT_POINT`` from winnt.h. The ``stat`` constant exists only
# on Windows.
JUNCTION_REPARSE_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def is_junction_stat(info: os.stat_result) -> bool:
    """Whether an ``lstat`` result is a Windows directory junction."""
    return getattr(info, "st_reparse_tag", 0) == JUNCTION_REPARSE_TAG


def is_link_stat(info: os.stat_result) -> bool:
    """Whether an ``lstat`` result is a symbolic link or a directory junction."""
    return stat.S_ISLNK(info.st_mode) or is_junction_stat(info)


def is_link(path: str | os.PathLike) -> bool:
    """Whether *path* itself is a symbolic link or a directory junction.

    Never follows *path*.

    Raises:
        OSError: *path* cannot be ``lstat``-ed, including when it does not
            exist. The caller decides what that means.
    """
    return is_link_stat(os.lstat(path))


def is_link_or_missing(path: str | os.PathLike) -> bool:
    """:func:`is_link`, but ``False`` when *path* cannot be ``lstat``-ed.

    The answer ``os.path.islink`` gives for a path that does not exist (or
    cannot, such as one holding a NUL byte, which ``lstat`` refuses with
    ``ValueError``), for a caller to which a missing component is not a
    redirect: a path about to be created.
    """
    try:
        return is_link(path)
    except (OSError, ValueError):
        return False


def is_link_entry(entry: os.DirEntry) -> bool:
    """Whether a directory entry is a symbolic link or a directory junction.

    ``DirEntry.is_symlink`` is False for a junction; the entry's own ``lstat``
    (which ``os.scandir`` caches on Windows) carries the reparse tag.

    Raises:
        OSError: the entry cannot be ``lstat``-ed (it was removed since the
            listing). The caller decides what that means.
    """
    if entry.is_symlink():
        return True
    return is_junction_stat(entry.stat(follow_symlinks=False))
