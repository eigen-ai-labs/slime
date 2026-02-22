"""FastAPI router for session management endpoints.

Provides HTTP endpoints for creating and destroying sessions within a
running container. Each session gets isolated /filesystem and /.apps_data
directories populated from the world baseline + task overlay.

Endpoints:
    POST /sessions/create   — create a new session
    POST /sessions/destroy  — destroy an existing session
    POST /sessions/list     — list active sessions
"""

import os
import shutil
import subprocess
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions")


# ── Models ───────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    task_slug: Optional[str] = Field(None, description="Task slug for file overlay")


class CreateSessionResult(BaseModel):
    session_id: str
    fs_root: str
    apps_data_root: str


class DestroySessionRequest(BaseModel):
    session_id: str


class DestroySessionResult(BaseModel):
    session_id: str
    cleaned: bool


class ListSessionsResult(BaseModel):
    sessions: list[str]


# ── Paths ────────────────────────────────────────────────────────────────────

FS_BASE = os.environ.get("APP_FS_ROOT", "/filesystem")
APPS_BASE = os.environ.get("APP_APPS_DATA_ROOT", "/.apps_data")
SESSIONS_FS = os.path.join(FS_BASE, "sessions")
SESSIONS_APPS = os.path.join(APPS_BASE, "sessions")

# Baselines created by start.sh at container boot
WORLD_FILES_BASE = "/app/_world_files_base"
WORLD_APPS_BASE = "/app/_world_apps_data_base"
TASK_FILES_DIR = "/app/tools/files"
TASK_APPS_DIR = "/app/tools/.apps_data"

SERVICES = [
    "calendar", "chat", "code_execution", "excel",
    "filesystem", "mail", "pdfs", "powerpoint", "word",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _copytree_contents(src: str, dst: str) -> None:
    """Copy contents of src into dst (like cp -R src/. dst/)."""
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    # Use cp -R for speed and to preserve permissions
    subprocess.run(
        ["cp", "-R", f"{src}/.", f"{dst}/"],
        capture_output=True, timeout=60,
    )


def _fix_permissions(session_fs: str, session_apps: str) -> None:
    """Fix ownership/permissions on session directories.

    Mirrors start.sh Phase 4 logic.
    """
    script = f"""
set -e
chown root:workspace {session_fs} 2>/dev/null || true
chmod 2770 {session_fs} 2>/dev/null || true
chown runner:runner {session_apps} 2>/dev/null || true
chmod 0711 {session_apps} 2>/dev/null || true

for svc in calendar chat code_execution excel filesystem mail pdfs powerpoint word; do
    d="{session_apps}/$svc"
    if [ -d "$d" ]; then
        chown -R "svc_$svc:appsdata_$svc" "$d" 2>/dev/null || true
        chmod -R u+rwX,g+rwX,o-rwx "$d" 2>/dev/null || true
        find "$d" -type d -exec chmod 2770 {{}} + 2>/dev/null || true
    fi
done
"""
    subprocess.run(["bash", "-c", script], capture_output=True, timeout=30)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/create", response_model=CreateSessionResult)
async def create_session(request: CreateSessionRequest) -> CreateSessionResult:
    """Create a new session with isolated filesystem and apps_data."""
    sid = request.session_id
    task_slug = request.task_slug

    session_fs = os.path.join(SESSIONS_FS, sid)
    session_apps = os.path.join(SESSIONS_APPS, sid)

    logger.info(f"Creating session {sid} (task={task_slug})")

    try:
        # Clean up if session already exists (idempotent)
        for d in (session_fs, session_apps):
            if os.path.exists(d):
                shutil.rmtree(d)

        # Create session directories
        os.makedirs(session_fs, exist_ok=True)
        os.makedirs(session_apps, exist_ok=True)

        # Phase 2: Copy world baseline
        _copytree_contents(WORLD_FILES_BASE, session_fs)
        _copytree_contents(WORLD_APPS_BASE, session_apps)

        # Phase 3: Overlay task-specific files
        if task_slug:
            task_apps_src = os.path.join(TASK_APPS_DIR, task_slug)
            task_files_src = os.path.join(TASK_FILES_DIR, task_slug)
            _copytree_contents(task_apps_src, session_apps)
            _copytree_contents(task_files_src, session_fs)

        # Phase 4: Fix permissions
        _fix_permissions(session_fs, session_apps)

        logger.info(f"Session {sid} created at {session_fs}")

        return CreateSessionResult(
            session_id=sid,
            fs_root=session_fs,
            apps_data_root=session_apps,
        )

    except Exception as e:
        logger.error(f"Failed to create session {sid}: {e}")
        # Clean up on failure
        for d in (session_fs, session_apps):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/destroy", response_model=DestroySessionResult)
async def destroy_session(request: DestroySessionRequest) -> DestroySessionResult:
    """Destroy a session and clean up its directories."""
    sid = request.session_id

    session_fs = os.path.join(SESSIONS_FS, sid)
    session_apps = os.path.join(SESSIONS_APPS, sid)

    logger.info(f"Destroying session {sid}")

    cleaned = False
    for d in (session_fs, session_apps):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            cleaned = True

    return DestroySessionResult(session_id=sid, cleaned=cleaned)


@router.post("/list", response_model=ListSessionsResult)
async def list_sessions() -> ListSessionsResult:
    """List all active sessions."""
    sessions = set()
    for base in (SESSIONS_FS, SESSIONS_APPS):
        if os.path.isdir(base):
            sessions.update(os.listdir(base))
    return ListSessionsResult(sessions=sorted(sessions))
