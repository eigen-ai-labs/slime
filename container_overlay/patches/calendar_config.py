import os

# ============================================================================
# Calendar Storage Configuration
# PATCHED: uses session_context for per-session root directories.
# ============================================================================


def get_calendar_data_root() -> str:
    """Return calendar data root, session-aware."""
    try:
        from session_context import get_session_calendar_root
        return get_session_calendar_root()
    except ImportError:
        _apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
        return os.getenv("APP_CALENDAR_DATA_ROOT") or os.path.join(
            _apps_data_root, "calendar"
        )


# Legacy constant — kept for any code that imports it at module level
_apps_data_root = os.getenv("APP_APPS_DATA_ROOT", "/.apps_data")
CALENDAR_DATA_ROOT = os.getenv("APP_CALENDAR_DATA_ROOT") or os.path.join(
    _apps_data_root, "calendar"
)


# ============================================================================
# Event Validation Configuration
# ============================================================================

MAX_SUMMARY_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 8000
MAX_LOCATION_LENGTH = 500


# ============================================================================
# List Pagination Configuration
# ============================================================================

DEFAULT_LIST_LIMIT = int(os.getenv("APP_CALENDAR_LIST_DEFAULT_LIMIT", "50"))
MAX_LIST_LIMIT = int(os.getenv("APP_CALENDAR_LIST_MAX_LIMIT", "100"))
