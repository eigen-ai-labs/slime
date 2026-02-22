import os

# ============================================================================
# Mail Storage Configuration
# PATCHED: uses session_context for per-session root directories.
# ============================================================================


def get_mail_data_root() -> str:
    """Return mail data root, session-aware."""
    try:
        from session_context import get_session_mail_root
        return get_session_mail_root()
    except ImportError:
        _apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
        return os.getenv("APP_MAIL_DATA_ROOT") or os.path.join(
            _apps_data_root, "mail"
        )


# Legacy constant
_apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
MAIL_DATA_ROOT = os.getenv("APP_MAIL_DATA_ROOT") or os.path.join(
    _apps_data_root, "mail"
)

MBOX_FILENAME = os.getenv("APP_MAIL_MBOX_FILENAME", "sent.mbox")


# ============================================================================
# Email Validation Configuration
# ============================================================================

MAX_SUBJECT_LENGTH = 998


# ============================================================================
# List Pagination Configuration
# ============================================================================

DEFAULT_LIST_LIMIT = int(os.getenv("APP_MAIL_LIST_DEFAULT_LIMIT", "50"))
MAX_LIST_LIMIT = int(os.getenv("APP_MAIL_LIST_MAX_LIMIT", "100"))
