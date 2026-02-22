import os

from utils.config import get_chat_data_root

# =============================================================================
# NOTE: TEMPORARY MEASURE - Google Chat Subdirectory Fallback
# =============================================================================
_ENABLE_GOOGLE_CHAT_SUBDIR_FALLBACK = True
_GOOGLE_CHAT_SUBDIR = "Google Chat"


def _try_google_chat_subdir_fallback(path: str) -> str | None:
    """Try to resolve path with 'Google Chat' subdirectory fallback."""
    if not _ENABLE_GOOGLE_CHAT_SUBDIR_FALLBACK:
        return None

    google_chat_path = os.path.normpath(
        os.path.join(get_chat_data_root(), _GOOGLE_CHAT_SUBDIR, path)
    )

    if os.path.exists(google_chat_path):
        return google_chat_path

    return None


def resolve_chat_path(path: str) -> str:
    """Map path to the chat data root (session-aware).

    Args:
        path: The relative path to resolve under the chat data root.

    Returns:
        The normalized absolute path under the chat data root.
    """
    path = path.lstrip("/")

    # Try standard path first
    standard_path = os.path.normpath(os.path.join(get_chat_data_root(), path))

    if os.path.exists(standard_path):
        return standard_path

    # TEMPORARY fallback for malformed data with "Google Chat" subdirectory
    fallback_path = _try_google_chat_subdir_fallback(path)
    if fallback_path:
        return fallback_path

    return standard_path
