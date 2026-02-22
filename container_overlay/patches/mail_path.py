import os

from utils.config import get_mail_data_root, MBOX_FILENAME


def resolve_mail_path(path: str) -> str:
    """Map path to the mail data root (session-aware).

    Args:
        path: The relative path to resolve under the mail data root.

    Returns:
        The normalized absolute path under the mail data root.
    """
    path = path.lstrip("/")
    full_path = os.path.join(get_mail_data_root(), path)
    return os.path.normpath(full_path)


def get_mbox_path() -> str:
    """Get the path to the mbox file for storing emails.

    Looks for any existing .mbox file in the mail directory (recursively). If none exists,
    uses the configured MBOX_FILENAME (default: sent.mbox).

    Returns:
        The absolute path to the mbox file.
    """
    mail_dir = resolve_mail_path("")
    if os.path.exists(mail_dir):
        for root, _, files in os.walk(mail_dir):
            mbox_files = [f for f in files if f.endswith(".mbox")]
            if mbox_files:
                return os.path.join(root, mbox_files[0])

    return os.path.join(mail_dir, MBOX_FILENAME)
