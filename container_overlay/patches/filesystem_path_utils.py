"""Shared path utilities for secure file path resolution.

This module provides utilities for safely resolving file paths within the
sandboxed root directory, with protection against path traversal attacks.

PATCHED: uses session_context for per-session root directories.
"""

import os


def get_fs_root() -> str:
    """Return the filesystem root directory, session-aware."""
    try:
        from session_context import get_session_fs_root
        return get_session_fs_root()
    except ImportError:
        return os.environ.get("APP_FS_ROOT", "/filesystem")


# Legacy constant for backward compatibility - reads at import time
FS_ROOT = os.environ.get("APP_FS_ROOT", "/filesystem")


class PathTraversalError(ValueError):
    """Raised when a path traversal attack is detected."""

    pass


def resolve_under_root(
    path: str,
    *,
    root: str | None = None,
    check_exists: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> str:
    """Safely resolve a path under the sandbox root directory."""
    if root is None:
        # Read from session context at call time
        root = get_fs_root()

    # Normalize the root path
    root = os.path.realpath(root)

    # Strip leading slashes to make path relative
    path = path.lstrip("/")

    # Combine and normalize
    full_path = os.path.normpath(os.path.join(root, path))

    # Always resolve symlinks to prevent path traversal via symlinks
    resolved_path = os.path.realpath(full_path)

    # Verify the resolved path is still under root
    if not resolved_path.startswith(root + os.sep) and resolved_path != root:
        raise PathTraversalError(
            f"Path '{path}' resolves outside the sandbox directory"
        )

    # Optional existence checks
    if check_exists and not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if must_be_file and not os.path.isfile(resolved_path):
        raise ValueError(f"Path is not a file: {path}")

    if must_be_dir and not os.path.isdir(resolved_path):
        raise ValueError(f"Path is not a directory: {path}")

    return resolved_path


def resolve_file_under_root(
    path: str,
    *,
    root: str | None = None,
    check_exists: bool = False,
) -> str:
    """Resolve a file path under the sandbox root."""
    return resolve_under_root(
        path,
        root=root,
        check_exists=check_exists,
        must_be_file=check_exists,
    )


def resolve_dir_under_root(
    path: str,
    *,
    root: str | None = None,
    check_exists: bool = False,
) -> str:
    """Resolve a directory path under the sandbox root."""
    return resolve_under_root(
        path,
        root=root,
        check_exists=check_exists,
        must_be_dir=check_exists,
    )


def resolve_new_file_path(
    directory: str,
    filename: str,
    *,
    root: str | None = None,
) -> str:
    """Resolve a path for a new file to be created within the sandbox."""
    if os.sep in filename or (os.altsep and os.altsep in filename):
        raise ValueError(f"Filename cannot contain path separators: {filename}")

    directory = directory.strip("/")
    if directory:
        path = f"{directory}/{filename}"
    else:
        path = filename

    return resolve_under_root(path, root=root)


def is_path_within_sandbox(path: str, root: str | None = None) -> bool:
    """Check if a path is within the sandbox without raising exceptions."""
    try:
        resolve_under_root(path, root=root)
        return True
    except (PathTraversalError, ValueError):
        return False
