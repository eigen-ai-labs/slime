#!/usr/bin/env python3
"""
Multi-session MCP container manager for RL training.

Key difference from the original: supports CONCURRENT sessions within a
single container via per-session data directories + X-Session-Id header.

Architecture:
    1 container per world (runs ~10 processes: runner + 9 MCP servers)
    N concurrent sessions per container (each has isolated /filesystem/sessions/{sid})
    M containers in a pool for M worlds

Session lifecycle:
    sid = container.create_session(task_slug)  # HTTP POST /sessions/create
    ... agent calls container.mcp_url with X-Session-Id header ...
    container.destroy_session(sid)             # HTTP POST /sessions/destroy

Container lifecycle:
    pool = ContainerPool(image, env, task_slug, base_port, size=N_worlds)
    pool.start()
    container = pool.get(i)
    ... create/use/destroy sessions on container ...
    pool.stop()
"""

import asyncio
import io
import os
import subprocess
import tarfile
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests


# ── Constants ────────────────────────────────────────────────────────────────

CONTAINER_STARTUP_TIMEOUT = 600
HEALTH_CHECK_INTERVAL = 2
SNAPSHOT_TIMEOUT = 300
SESSION_CREATE_TIMEOUT = 120


# ── Session ──────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """Represents an isolated rollout session on a container."""
    session_id: str
    container_id: str
    port: int
    task_slug: str
    created_at: float = field(default_factory=time.time)
    active: bool = True


# ── SessionContainer ─────────────────────────────────────────────────────────

class SessionContainer:
    """Manages a single Docker container with multi-session support.

    Unlike the original, sessions are created via HTTP endpoint inside the
    container (not via docker exec). Multiple sessions can be active at once;
    each gets its own directory tree under /filesystem/sessions/{sid}/ and
    /.apps_data/sessions/{sid}/.

    MCP tool calls must include the X-Session-Id header so that the MCP
    servers route to the correct session's data directory.
    """

    def __init__(self, image_tag, env_file, port, docker_cmd="docker",
                 host="localhost"):
        self.image_tag = image_tag
        self.env_file = env_file
        self.port = port
        self.docker_cmd = docker_cmd
        self.host = host
        self.container_id = None
        self.task_slug = None
        self._base_mcp_url = f"http://{host}:{port}/mcp/"
        self._tools_cache = None
        self._active_sessions: dict[str, Session] = {}

    @property
    def mcp_url(self):
        """Base MCP URL (without session). Use mcp_url_for(sid) for session-scoped calls."""
        return self._base_mcp_url

    def mcp_url_for(self, session_id: str) -> str:
        """Return the MCP URL. The session_id must be passed as X-Session-Id header."""
        return self._base_mcp_url

    def mcp_headers_for(self, session_id: str) -> dict:
        """Return HTTP headers that scope MCP calls to a session."""
        return {"X-Session-Id": session_id}

    # ── Container lifecycle ──────────────────────────────────────────────

    def start(self, task_slug):
        """Start container for a task. Blocks until healthy."""
        self._cleanup_port()
        self.task_slug = task_slug

        cmd = [
            self.docker_cmd, "run", "-d",
            "--env-file", self.env_file,
            "-p", f"{self.port}:8000",
            self.image_tag,
            "/app/tools/start.sh", task_slug,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")
        self.container_id = result.stdout.strip()
        self._wait_for_health()
        self._tools_cache = None

    def stop(self):
        """Stop and remove the container."""
        if self.container_id:
            try:
                subprocess.run(
                    [self.docker_cmd, "stop", self.container_id],
                    capture_output=True, text=True, timeout=30,
                )
            except Exception:
                try:
                    subprocess.run(
                        [self.docker_cmd, "kill", self.container_id],
                        capture_output=True, text=True, timeout=10,
                    )
                except Exception:
                    pass
            try:
                subprocess.run(
                    [self.docker_cmd, "rm", "-f", self.container_id],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:
                pass
            self.container_id = None
            self._active_sessions.clear()

    @property
    def is_running(self):
        if not self.container_id:
            return False
        try:
            result = subprocess.run(
                [self.docker_cmd, "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def ensure_running(self, task_slug=None):
        """Ensure the container is running. Restart if it died."""
        if self.is_running:
            return
        slug = task_slug or self.task_slug
        if not slug:
            raise RuntimeError("Cannot restart container: no task_slug")
        print(f"      [port {self.port}] Container died, restarting...")
        if self.container_id:
            try:
                subprocess.run(
                    [self.docker_cmd, "rm", "-f", self.container_id],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:
                pass
            self.container_id = None
        self._active_sessions.clear()
        self.start(slug)

    # ── Session lifecycle (HTTP-based, concurrent) ────────────────────────

    def create_session(self, task_slug=None, session_id=None):
        """Create a new session via HTTP endpoint.

        Creates isolated directories /filesystem/sessions/{sid}/ and
        /.apps_data/sessions/{sid}/ inside the container, populated
        from the world baseline + task overlay.

        Multiple sessions can be active concurrently.

        Returns a Session object.
        """
        if not self.container_id:
            raise RuntimeError("Container not running")

        effective_slug = task_slug or self.task_slug
        sid = session_id or f"s_{uuid.uuid4().hex[:12]}"

        url = f"http://{self.host}:{self.port}/sessions/create"
        payload = {"session_id": sid, "task_slug": effective_slug}

        resp = requests.post(url, json=payload, timeout=SESSION_CREATE_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Session create failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )

        session = Session(
            session_id=sid,
            container_id=self.container_id,
            port=self.port,
            task_slug=effective_slug,
        )
        self._active_sessions[sid] = session
        return session

    def destroy_session(self, session_id=None):
        """Destroy a session via HTTP endpoint."""
        if session_id is None:
            # Backward compat: destroy most recent
            if self._active_sessions:
                session_id = next(reversed(self._active_sessions))
            else:
                return

        url = f"http://{self.host}:{self.port}/sessions/destroy"
        try:
            requests.post(
                url, json={"session_id": session_id}, timeout=30,
            )
        except Exception:
            pass  # Best effort cleanup

        self._active_sessions.pop(session_id, None)

    # ── Snapshots ────────────────────────────────────────────────────────

    def snapshot_to_zip(self, output_path, session_id=None):
        """Take a snapshot via POST /data/snapshot, convert tar.gz to ZIP.

        If session_id is provided, passes X-Session-Id header so the
        snapshot is scoped to the session's directories.
        """
        url = f"http://{self.host}:{self.port}/data/snapshot"
        headers = {}
        if session_id:
            headers["X-Session-Id"] = session_id

        resp = requests.post(url, stream=True, timeout=SNAPSHOT_TIMEOUT, headers=headers)
        resp.raise_for_status()

        tar_bytes = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=65536):
            tar_bytes.write(chunk)
        tar_bytes.seek(0)

        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tf:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    arcname = member.name
                    if not arcname.startswith("filesystem/"):
                        if arcname.startswith(".apps_data/"):
                            continue
                        arcname = "filesystem/" + arcname
                    f = tf.extractfile(member)
                    if f is not None:
                        zf.writestr(arcname, f.read())

    def snapshot_to_targz(self, output_path, session_id=None):
        """Take a snapshot via POST /data/snapshot, save raw tar.gz."""
        url = f"http://{self.host}:{self.port}/data/snapshot"
        headers = {}
        if session_id:
            headers["X-Session-Id"] = session_id

        resp = requests.post(url, stream=True, timeout=SNAPSHOT_TIMEOUT, headers=headers)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

    # ── Tools ────────────────────────────────────────────────────────────

    def get_tools(self, retries=3, delay=5):
        """List tools from the MCP endpoint. Cached after first call."""
        if self._tools_cache is not None:
            return self._tools_cache

        from fastmcp import Client as FastMCPClient

        for attempt in range(retries):
            try:
                config = {"mcpServers": {"gateway": {"transport": "http", "url": self.mcp_url}}}
                client = FastMCPClient(config, timeout=60)

                async def _list():
                    async with client:
                        result = await client.session.list_tools()
                        tools = []
                        for tool in result.tools:
                            schema = tool.inputSchema or {"type": "object", "properties": {}}
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "parameters": schema,
                                },
                            })
                        return tools

                self._tools_cache = asyncio.run(_list())
                return self._tools_cache
            except Exception as e:
                if attempt < retries - 1:
                    print(f"      [port {self.port}] Tool discovery failed ({e}), "
                          f"retrying in {delay}s ({attempt + 1}/{retries})...")
                    time.sleep(delay)
                else:
                    raise

    # ── Internal ─────────────────────────────────────────────────────────

    def _cleanup_port(self):
        """Stop any container bound to our port."""
        try:
            result = subprocess.run(
                [self.docker_cmd, "ps", "--filter", f"publish={self.port}", "-q"],
                capture_output=True, text=True, timeout=10,
            )
            for cid in result.stdout.strip().split("\n"):
                cid = cid.strip()
                if cid:
                    subprocess.run(
                        [self.docker_cmd, "stop", cid],
                        capture_output=True, text=True, timeout=30,
                    )
        except Exception:
            pass

    def _wait_for_health(self, timeout=CONTAINER_STARTUP_TIMEOUT):
        """Wait for /health to respond."""
        url = f"http://{self.host}:{self.port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        raise TimeoutError(f"Container health check timed out after {timeout}s")


# ── ContainerPool ────────────────────────────────────────────────────────────

class ContainerPool:
    """Pool of containers for parallel rollouts within a world.

    With multi-session support, a single container can handle multiple
    concurrent sessions. The pool size can be 1 (one container per world)
    while still supporting concurrent rollouts.

    Usage:
        pool = ContainerPool(image, env, "any-task-slug", base_port=8000, size=1)
        pool.start()

        container = pool.get(0)
        s1 = container.create_session(task_slug="task-a")
        s2 = container.create_session(task_slug="task-b")  # concurrent!
        # ... use s1.session_id and s2.session_id in X-Session-Id header ...
        container.destroy_session(s1.session_id)
        container.destroy_session(s2.session_id)

        pool.stop()
    """

    def __init__(self, image_tag, env_file, initial_task_slug, base_port, size,
                 docker_cmd="docker", host="localhost"):
        self.initial_task_slug = initial_task_slug
        self.size = size
        self.containers = [
            SessionContainer(image_tag, env_file, base_port + i, docker_cmd,
                             host=host)
            for i in range(size)
        ]
        self._started = False

    def start(self, parallel=True):
        """Start all containers. Optionally in parallel."""
        if parallel and self.size > 1:
            with ThreadPoolExecutor(max_workers=self.size) as executor:
                futures = {
                    executor.submit(c.start, self.initial_task_slug): i
                    for i, c in enumerate(self.containers)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                        print(f"  [pool] Container {idx} ready "
                              f"(port {self.containers[idx].port}, "
                              f"id={self.containers[idx].container_id[:12]})")
                    except Exception as e:
                        print(f"  [pool] Container {idx} FAILED: {e}")
                        raise
        else:
            for i, c in enumerate(self.containers):
                c.start(self.initial_task_slug)
                print(f"  [pool] Container {i} ready "
                      f"(port {c.port}, id={c.container_id[:12]})")
        self._started = True

    def stop(self):
        """Stop all containers."""
        for c in self.containers:
            try:
                c.stop()
            except Exception:
                pass
        self._started = False

    def get(self, index):
        """Get container by index."""
        return self.containers[index % self.size]

    def __len__(self):
        return self.size

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


# ── Convenience functions ────────────────────────────────────────────────────

def run_rollout_on_session(container, run_agent_fn, task):
    """Run a single rollout using session create/snapshot/destroy pattern.

    Args:
        container: SessionContainer with a running container
        run_agent_fn: callable(task, tools, mcp_url, headers) -> agent_result
        task: task dict with prompt, task_id, etc.

    Returns:
        (session, agent_result, elapsed, initial_snap_path, final_snap_path)
    """
    session = container.create_session()
    sid = session.session_id
    headers = container.mcp_headers_for(sid)

    initial_snap = tempfile.mktemp(suffix=".zip", prefix=f"snap_init_{sid}_")
    container.snapshot_to_zip(initial_snap, session_id=sid)

    tools = container.get_tools()
    t0 = time.time()
    agent_result = run_agent_fn(task, tools, container.mcp_url, headers)
    elapsed = time.time() - t0

    final_snap = tempfile.mktemp(suffix=".zip", prefix=f"snap_final_{sid}_")
    container.snapshot_to_zip(final_snap, session_id=sid)

    container.destroy_session(sid)

    return session, agent_result, elapsed, initial_snap, final_snap


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Session MCP Manager demo")
    parser.add_argument("--image", required=True, help="Docker image tag")
    parser.add_argument("--env-file", required=True, help="Path to .env file")
    parser.add_argument("--task-slug", required=True, help="Task slug")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sessions", type=int, default=2,
                        help="Number of concurrent sessions to test")
    args = parser.parse_args()

    print(f"Starting container for {args.task_slug}...")
    sc = SessionContainer(args.image, args.env_file, args.port)
    try:
        sc.start(args.task_slug)
        print(f"Container {sc.container_id[:12]} ready on port {args.port}")

        tools = sc.get_tools()
        print(f"Tools: {len(tools)}")

        # Test concurrent sessions
        sessions = []
        for i in range(args.sessions):
            t0 = time.time()
            session = sc.create_session()
            elapsed = time.time() - t0
            sessions.append(session)
            print(f"  Session {session.session_id} created ({elapsed:.1f}s)")

        print(f"\n{len(sessions)} sessions active concurrently")

        # Snapshot each
        for session in sessions:
            t0 = time.time()
            snap_path = tempfile.mktemp(suffix=".zip")
            sc.snapshot_to_zip(snap_path, session_id=session.session_id)
            snap_time = time.time() - t0
            snap_size = os.path.getsize(snap_path) / 1024 / 1024
            print(f"  Snapshot {session.session_id}: {snap_size:.1f} MB ({snap_time:.1f}s)")
            os.unlink(snap_path)

        # Destroy all
        for session in sessions:
            sc.destroy_session(session.session_id)
            print(f"  Destroyed {session.session_id}")

    finally:
        print("\nStopping container...")
        sc.stop()
        print("Done.")
