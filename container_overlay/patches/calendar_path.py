import os

from utils.config import get_calendar_data_root


def resolve_calendar_path(path: str) -> str:
    """Map path to the calendar data root (session-aware).

    Args:
        path: The relative path to resolve under the calendar data root.

    Returns:
        The normalized absolute path under the calendar data root.
    """
    path = path.lstrip("/")
    full_path = os.path.join(get_calendar_data_root(), path)
    return os.path.normpath(full_path)
