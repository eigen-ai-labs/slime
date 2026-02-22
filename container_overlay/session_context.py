"""
Session context for multi-session MCP servers.

Provides a ContextVar-based mechanism for MCP servers to determine which
session's data directory to use. The session_id is extracted from the
X-Session-Id HTTP header by SessionMiddleware and stored in a ContextVar.

Path helpers return session-scoped directories when a session is active,
or the default directories when no session is set (backward compatible).

Layout when session is active:
    /filesystem/sessions/{sid}/...
    /.apps_data/sessions/{sid}/calendar/...
    /.apps_data/sessions/{sid}/chat/...
    /.apps_data/sessions/{sid}/mail/...
"""

import contextvars
import os

# ContextVar holding the current session ID (None = no session / legacy mode)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)


def get_session_id() -> str | None:
    """Return the current session ID, or None if not in a session."""
    return session_id_var.get()


# ── Filesystem root ──────────────────────────────────────────────────────────

def get_session_fs_root() -> str:
    """Return APP_FS_ROOT scoped to the current session.

    If a session is active:  /filesystem/sessions/{sid}
    Otherwise:               /filesystem  (default)
    """
    base = os.environ.get("APP_FS_ROOT", "/filesystem")
    sid = session_id_var.get()
    if sid:
        return os.path.join(base, "sessions", sid)
    return base


# ── Apps data root ───────────────────────────────────────────────────────────

def get_session_apps_data_root() -> str:
    """Return APP_APPS_DATA_ROOT scoped to the current session.

    If a session is active:  /.apps_data/sessions/{sid}
    Otherwise:               /.apps_data  (default)
    """
    base = os.environ.get("APP_APPS_DATA_ROOT", "/.apps_data")
    sid = session_id_var.get()
    if sid:
        return os.path.join(base, "sessions", sid)
    return base


# ── Per-service helpers ──────────────────────────────────────────────────────

def get_session_calendar_root() -> str:
    specific = os.getenv("APP_CALENDAR_DATA_ROOT")
    if specific:
        sid = session_id_var.get()
        if sid:
            return os.path.join(specific, "sessions", sid)
        return specific
    return os.path.join(get_session_apps_data_root(), "calendar")


def get_session_chat_root() -> str:
    specific = os.getenv("APP_CHAT_DATA_ROOT")
    if specific:
        sid = session_id_var.get()
        if sid:
            return os.path.join(specific, "sessions", sid)
        return specific
    return os.path.join(get_session_apps_data_root(), "chat")


def get_session_mail_root() -> str:
    specific = os.getenv("APP_MAIL_DATA_ROOT")
    if specific:
        sid = session_id_var.get()
        if sid:
            return os.path.join(specific, "sessions", sid)
        return specific
    return os.path.join(get_session_apps_data_root(), "mail")
