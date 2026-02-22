import os

# ============================================================================
# Chat Storage Configuration
# PATCHED: uses session_context for per-session root directories.
# ============================================================================


def get_chat_data_root() -> str:
    """Return chat data root, session-aware."""
    try:
        from session_context import get_session_chat_root
        return get_session_chat_root()
    except ImportError:
        _apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
        return os.getenv("APP_CHAT_DATA_ROOT") or os.path.join(
            _apps_data_root, "chat"
        )


# Legacy constant
_apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
CHAT_DATA_ROOT = os.getenv("APP_CHAT_DATA_ROOT") or os.path.join(
    _apps_data_root, "chat"
)

CURRENT_USER_ID = os.getenv("CHAT_CURRENT_USER_ID", "User 000000000000000000000")
CURRENT_USER_EMAIL = os.getenv("CHAT_CURRENT_USER_EMAIL", "user@example.com")


# ============================================================================
# Pagination Configuration
# ============================================================================

DEFAULT_GROUPS_LIMIT = int(os.getenv("CHAT_DEFAULT_GROUPS_LIMIT", "100"))
MAX_GROUPS_LIMIT = int(os.getenv("CHAT_MAX_GROUPS_LIMIT", "200"))
DEFAULT_MESSAGES_LIMIT = int(os.getenv("CHAT_DEFAULT_MESSAGES_LIMIT", "30"))
MAX_MESSAGES_LIMIT = int(os.getenv("CHAT_MAX_MESSAGES_LIMIT", "200"))
DEFAULT_USERS_LIMIT = int(os.getenv("CHAT_DEFAULT_USERS_LIMIT", "100"))
MAX_USERS_LIMIT = int(os.getenv("CHAT_MAX_USERS_LIMIT", "200"))
