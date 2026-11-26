"""Path safety utilities for server-side file I/O."""

import ntpath
import os
import re

# Windows' extended-length prefixes, and a drive path after the first one.
_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"
_DRIVE_PATH = re.compile(r"[A-Za-z]:\\")


class LibraryRootsUnavailable(RuntimeError):
    """A list of the library's own roots could not be read.

    Raised by the helpers that answer "which directories belong to this
    library" - :meth:`pixlstash.vault.Vault.reference_folder_roots` and
    :func:`pixlstash.services.config_service.get_import_folder_paths` - instead
    of returning an empty collection, because "there are none" and "we do not
    know" are the same value with opposite safety (#1177 items 59, and the
    import-folder sibling found in its review).

    Each list is read as an **allowlist** in one place and a **blocklist** in
    another. The scrapheap purge asks which roots its unattended ``os.remove``
    may follow a stored path into, where empty means "only ``image_root``" and
    is safe. The folder-export destination check asks which roots nothing may
    be written into, where empty means "nothing to avoid" and permits exactly
    what the check exists to refuse. No single value
    is safe in both directions, so the failure is made distinguishable and the
    caller picks the direction.
    """


def _without_added_prefix(path: str, resolved: str) -> str:
    """*resolved* without a ``\\\\?\\`` prefix ``realpath`` added to a missing path.

    CPython's Windows ``realpath`` resolves a missing path to its ``\\\\?\\``
    form, then strips the prefix if the plain path fails to resolve with the
    same error as the first try. A parent directory another thread creates in
    between changes that error (3, path not found, becomes 2, file not found),
    so the prefix stays: ``commonpath`` then refuses to compare it with a plain
    root, and a string compare misses the same file spelled plainly. The plain
    spelling is resolved again: with nothing racing it, ``realpath`` strips the
    prefix as designed, and a folder or file that appeared in between (a
    junction included) resolves to where it now points. A path raced a second
    time keeps the prefix, and containment refuses it. A volume GUID path
    (``\\\\?\\Volume{...}``) has no plain spelling and is kept.

    Args:
        path: The path as given to ``realpath``.
        resolved: What ``realpath`` returned for it.

    Returns:
        *resolved*, or its plain spelling resolved again when the prefix was
        only added.
    """
    # Normalised first, as realpath does, so ``//?/C:/`` is a prefix as given.
    given_prefix = ntpath.normpath(path).startswith(_EXTENDED_PREFIX)
    if given_prefix or not resolved.startswith(_EXTENDED_PREFIX):
        return resolved
    if resolved.startswith(_EXTENDED_UNC_PREFIX):
        plain = "\\\\" + resolved[len(_EXTENDED_UNC_PREFIX) :]
    elif _DRIVE_PATH.match(resolved, len(_EXTENDED_PREFIX)):
        plain = resolved[len(_EXTENDED_PREFIX) :]
    else:
        return resolved
    return os.path.realpath(plain)


def real_path(path: str | os.PathLike[str]) -> str:
    """``os.path.realpath``, spelled plainly on Windows whatever raced it.

    Use it wherever a resolved path is compared with another (containment, or
    the same file named twice) and may not exist yet.
    """
    path = os.fspath(path)
    resolved = os.path.realpath(path)
    if os.name != "nt":
        return resolved
    return _without_added_prefix(path, resolved)


def resolve_path_within(base_dir: str, *segments: str) -> str:
    """Resolve a path and confirm it remains strictly within *base_dir*.

    Args:
        base_dir: The permitted root directory.
        *segments: Path segments to join under *base_dir*. These may contain
            user-supplied values (e.g. filenames from HTTP requests or DB
            rows) and must not escape the root even through ``..`` components
            or symbolic links.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the resolved path would escape *base_dir*.

    Note:
        Some call sites pass values that are structurally incapable of path
        traversal - for example, integer IDs formatted into a fixed filename
        template such as ``f"character_{id}.png"`` where FastAPI has already
        validated the ``int`` type.  Those uses are redundant from a security
        standpoint but are kept intentionally so that CodeQL's taint-tracking
        analysis sees a recognised sanitizer at every path-construction site
        and does not emit false-positive findings that would need to be
        manually dismissed.
    """
    joined = os.path.join(base_dir, *segments)
    resolved = real_path(joined)
    safe_base = real_path(base_dir)
    try:
        common = os.path.commonpath([resolved, safe_base])
    except ValueError as exc:
        # Different drives / roots (not within base).
        raise ValueError(
            f"Path would escape allowed directory: {segments!r} is not within {base_dir!r}"
        ) from exc

    if common != safe_base:
        raise ValueError(
            f"Path would escape allowed directory: {segments!r} is not within {base_dir!r}"
        )
    return resolved


def path_is_within(path: str, base: str) -> bool:
    """Whether *path* lies within *base*, lexically or after symlink resolution.

    Unlike :func:`resolve_path_within` this answers a question instead of
    raising, and it does not rewrite the path. Use it where a caller holds an
    already-absolute path and must decide whether to act on it.

    The lexical check (``normpath``) neutralises ``..`` components without
    resolving symlinks, so a library whose *content* is reached through a
    symlink is not refused. The ``realpath`` check then additionally accepts a
    path spelled through a different alias of the same directory (e.g. a
    symlinked root). A symlink planted *inside* an allowed root that points
    outside it is therefore accepted: planting one requires filesystem write
    access, which is a bigger problem than this check is for.

    Args:
        path: The path to test. An empty value is never within anything.
        base: The directory *path* must be under.

    Returns:
        True when *path* is contained in *base*.
    """
    if not path or not base:
        return False
    try:
        norm_path = os.path.normcase(os.path.normpath(path))
        norm_base = os.path.normcase(os.path.normpath(base))
        if os.path.commonpath([norm_path, norm_base]) == norm_base:
            return True
        resolved_path = os.path.normcase(real_path(path))
        resolved_base = os.path.normcase(real_path(base))
        return os.path.commonpath([resolved_path, resolved_base]) == resolved_base
    except ValueError:
        # Mixed absolute/relative paths or different drives: not within.
        return False
