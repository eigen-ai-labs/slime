"""
APEX Docker Rollout — run slime RL training against Docker-based APEX task environments.

This module provides a custom generate function that:
1. Manages Docker container lifecycle (one container per world, session-based isolation)
2. Connects to the container's MCP server for tool execution
3. Runs mcp_agent_loop() for multi-step agent rollouts
4. Captures before/after filesystem snapshots for Archipelago grading

Architecture:
    - One Docker container per world (all tasks in a world share the same MCP server)
    - Different tasks/rollouts get different session IDs via create_session()
    - Multiple worlds' containers run in parallel
    - Container pools persist across training iterations (cleanup via atexit)

Environment variables:
    APEX_POOL_SIZE     - Containers per world (default: 1)
    APEX_BASE_PORT     - Starting port for containers (default: 9000)
    APEX_DOCKER_CMD    - Docker command (default: "docker")
    APEX_DOCKER_ROOT   - Root dir for Docker exports (used by load_docker_image)
    APEX_TRAINING_ROOT - Root dir for training tasks (used by load_training_verifiers)

Usage:
    --custom-generate-function-path slime.rollout.apex_docker_rollout.generate_with_apex_docker
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys
import tempfile
import time
from argparse import Namespace
from typing import Any

from slime.rollout.mcp.client import MCPClient
from slime.rollout.mcp.protocols import MCPClientConfig, MCPTool, MCPTransport, ToolCall, ToolResult
from slime.rollout.mcp_agent_rollout import (
    _finalize_sample_tokens,
    mcp_agent_loop,
)
from slime.rollout.tool_parser import get_parser
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ── PerContainerMCPState ─────────────────────────────────────────────────────

class PerContainerMCPState:
    """Non-singleton MCP state adapter wrapping a single MCPClient.

    Provides the same get_all_tools() / execute_tool_calls() interface that
    mcp_agent_loop() expects, but without the SingletonMeta constraint of
    MCPState. Each instance connects to a single container's MCP URL.
    """

    def __init__(self, mcp_url: str, name: str = "apex-container",
                 headers: dict[str, str] | None = None):
        self._mcp_url = mcp_url
        self._name = name
        self._headers = headers or {}
        self._client: MCPClient | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Connect to the container's MCP server."""
        async with self._lock:
            if self._initialized:
                return
            config = MCPClientConfig(
                name=self._name,
                transport=MCPTransport.STREAMABLE_HTTP,
                url=self._mcp_url,
                headers=self._headers,
                concurrency_limit=16,
                timeout=120.0,
            )
            self._client = MCPClient(config)
            await self._client.connect()
            self._initialized = True
            logger.info("PerContainerMCPState connected to %s", self._mcp_url)

    async def get_all_tools(self, force_refresh: bool = False) -> list[MCPTool]:
        """Get tools from the container's MCP server."""
        if not self._initialized:
            await self.initialize()
        assert self._client is not None
        return await self._client.list_tools(force_refresh=force_refresh)

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        if not self._initialized:
            await self.initialize()
        assert self._client is not None
        return await self._client.call_tool(tool_call)

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        parallel: bool = True,
    ) -> list[ToolResult]:
        """Execute multiple tool calls."""
        if not tool_calls:
            return []
        if parallel:
            tasks = [self.execute_tool_call(tc) for tc in tool_calls]
            return await asyncio.gather(*tasks)
        else:
            return [await self.execute_tool_call(tc) for tc in tool_calls]

    async def shutdown(self) -> None:
        """Disconnect from MCP server."""
        async with self._lock:
            if self._client:
                await self._client.disconnect()
                self._client = None
            self._initialized = False


# ── ManagedContainerPool ─────────────────────────────────────────────────────

class ManagedContainerPool:
    """Wraps ContainerPool with async container allocation.

    Each world gets one ManagedContainerPool. Multiple sessions can run
    concurrently on each container (controlled by APEX_SESSION_CONCURRENCY).
    """

    def __init__(self, world_id: str, docker_dir: str, pool_size: int,
                 base_port: int, docker_cmd: str = "docker",
                 session_concurrency: int = 4):
        self.world_id = world_id
        self.docker_dir = docker_dir
        self.pool_size = pool_size
        self.base_port = base_port
        self.docker_cmd = docker_cmd
        self.session_concurrency = session_concurrency

        self._pool = None  # ContainerPool, created in start()
        self._semas: list[asyncio.Semaphore] = []
        self._started = False
        self._start_lock = asyncio.Lock()  # Prevents concurrent start() calls
        self._next_idx = 0  # round-robin counter

    async def start(self) -> None:
        """Load Docker image, extract env, and start the container pool."""
        async with self._start_lock:
            if self._started:
                return
            await self._start_unlocked()

    async def _start_unlocked(self) -> None:
        """Internal start logic, must be called under self._start_lock."""
        # Import from mcp_session_manager (must be on PYTHONPATH)
        from mcp_session_manager import ContainerPool

        # Import helpers from distill_apex_docker_v4 (must be on PYTHONPATH)
        from distill_apex_docker_v4 import extract_env_file, load_docker_image, load_docker_tasks

        logger.info("Starting container pool for world %s (size=%d, base_port=%d)",
                     self.world_id, self.pool_size, self.base_port)

        # Load Docker image
        image_tag = await asyncio.to_thread(load_docker_image, self.docker_dir)
        # Use multi-session overlay image if available (canonical-ms:xxx)
        ms_tag = image_tag.replace("canonical-delivery:", "canonical-ms:", 1)
        if ms_tag != image_tag:
            import subprocess as _sp
            check = _sp.run(["docker", "image", "inspect", ms_tag],
                            capture_output=True, timeout=10)
            if check.returncode == 0:
                logger.info("Using multi-session image: %s (base: %s)", ms_tag, image_tag)
                image_tag = ms_tag
            else:
                logger.info("Multi-session image not found, using base: %s", image_tag)
        else:
            logger.info("Loaded Docker image: %s", image_tag)

        # Extract .env file
        env_file = await asyncio.to_thread(extract_env_file, self.docker_dir)

        # Get any task slug for initial container startup (world baseline is the same)
        tasks = await asyncio.to_thread(load_docker_tasks, self.docker_dir, self.world_id)
        if not tasks:
            raise RuntimeError(f"No tasks found in {self.docker_dir}")
        initial_slug = tasks[0]["task_slug"]

        # Create and start pool
        self._pool = ContainerPool(
            image_tag=image_tag,
            env_file=env_file,
            initial_task_slug=initial_slug,
            base_port=self.base_port,
            size=self.pool_size,
            docker_cmd=self.docker_cmd,
        )
        await asyncio.to_thread(self._pool.start, True)

        # Create per-container semaphores (allow N concurrent sessions each)
        self._semas = [asyncio.Semaphore(self.session_concurrency)
                       for _ in range(self.pool_size)]
        self._started = True

        logger.info("Container pool for world %s ready (%d containers)",
                     self.world_id, self.pool_size)

    def stop(self) -> None:
        """Stop all containers in the pool."""
        if self._pool and self._started:
            try:
                self._pool.stop()
            except Exception as e:
                logger.error("Error stopping pool for world %s: %s", self.world_id, e)
            self._started = False
            logger.info("Container pool for world %s stopped", self.world_id)

    async def acquire(self) -> tuple[int, Any]:
        """Acquire a slot on a container from the pool.

        Uses round-robin across containers. Each container allows up to
        session_concurrency concurrent sessions via Semaphore.

        Returns (container_index, container) tuple.
        The caller MUST call release(idx) when done.
        """
        if not self._started:
            await self.start()

        # Round-robin: pick next container and wait for a semaphore slot
        idx = self._next_idx % self.pool_size
        self._next_idx += 1
        await self._semas[idx].acquire()
        container = self._pool.get(idx)
        return idx, container

    def release(self, idx: int) -> None:
        """Release a session slot back to the container."""
        if idx < len(self._semas):
            self._semas[idx].release()


# ── RemoteManagedPool (connect to pre-started containers on remote host) ─────

class RemoteManagedPool:
    """Connects to pre-started containers on a remote MCP host.

    Unlike ManagedContainerPool, this does NOT start/stop containers.
    It only wraps SessionContainer objects in connect-only mode for
    session creation, tool calls, and snapshots.
    """

    def __init__(self, world_id: str, host: str, port: int,
                 session_concurrency: int = 4):
        self.world_id = world_id
        self.host = host
        self.port = port
        self.session_concurrency = session_concurrency

        self._container = None
        self._sema = asyncio.Semaphore(session_concurrency)
        self._started = False
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        """Create a SessionContainer wrapper (no docker run, just connect)."""
        async with self._start_lock:
            if self._started:
                return
            from mcp_session_manager import SessionContainer

            # Create container wrapper in connect-only mode:
            # no image_tag/env_file needed, container already running
            self._container = SessionContainer(
                image_tag="remote",
                env_file="/dev/null",
                port=self.port,
                host=self.host,
            )
            # Set container_id to a placeholder so is_running checks pass
            self._container.container_id = f"remote-{self.host}-{self.port}"

            # Wait for health
            logger.info("Waiting for remote MCP at %s:%d ...", self.host, self.port)
            await asyncio.to_thread(self._container._wait_for_health, 120)
            self._started = True
            logger.info("Remote MCP pool for world %s ready at %s:%d",
                        self.world_id, self.host, self.port)

    def stop(self) -> None:
        """No-op: remote containers are managed externally."""
        pass

    async def acquire(self) -> tuple[int, any]:
        """Acquire a session slot. Returns (0, container)."""
        if not self._started:
            await self.start()
        await self._sema.acquire()
        return 0, self._container

    def release(self, idx: int) -> None:
        """Release a session slot."""
        self._sema.release()


# ── Port Map for Remote Mode ─────────────────────────────────────────────────

_port_map_cache: dict[str, int] | None = None


def _load_port_map() -> dict[str, int]:
    """Load world_id → port mapping from JSON file."""
    global _port_map_cache
    if _port_map_cache is not None:
        return _port_map_cache

    import json
    port_map_path = os.environ.get("APEX_MCP_PORT_MAP", "")
    if not port_map_path:
        raise RuntimeError(
            "APEX_MCP_HOST is set but APEX_MCP_PORT_MAP is not. "
            "Set APEX_MCP_PORT_MAP to a JSON file mapping world_id → port."
        )
    with open(port_map_path) as f:
        _port_map_cache = json.load(f)
    logger.info("Loaded port map with %d worlds from %s", len(_port_map_cache), port_map_path)
    return _port_map_cache


# ── Pool Registry (module-level, per world) ──────────────────────────────────

_pool_registry: dict[str, ManagedContainerPool | RemoteManagedPool] = {}
_registry_lock = asyncio.Lock()
_port_counter = int(os.environ.get("APEX_BASE_PORT", "9000"))


def _cleanup_all_pools():
    """atexit handler: stop all container pools."""
    for world_id, pool in _pool_registry.items():
        try:
            pool.stop()
        except Exception as e:
            print(f"Error cleaning up pool for {world_id}: {e}", file=sys.stderr)
    _pool_registry.clear()


atexit.register(_cleanup_all_pools)


async def _get_or_create_pool(world_id: str, docker_dir: str) -> ManagedContainerPool | RemoteManagedPool:
    """Get or create a pool for a world.

    If APEX_MCP_HOST is set, creates a RemoteManagedPool that connects to
    pre-started containers on the remote host. Otherwise, creates a
    ManagedContainerPool that starts containers locally.
    """
    global _port_counter

    async with _registry_lock:
        if world_id in _pool_registry:
            return _pool_registry[world_id]

        mcp_host = os.environ.get("APEX_MCP_HOST", "")
        session_concurrency = int(os.environ.get("APEX_SESSION_CONCURRENCY", "4"))

        if mcp_host:
            # Remote mode: connect to pre-started container
            port_map = _load_port_map()
            if world_id not in port_map:
                raise RuntimeError(
                    f"World {world_id} not found in port map. "
                    f"Available worlds: {list(port_map.keys())[:5]}..."
                )
            port = port_map[world_id]
            pool = RemoteManagedPool(
                world_id=world_id,
                host=mcp_host,
                port=port,
                session_concurrency=session_concurrency,
            )
            logger.info("Created remote pool for world %s → %s:%d", world_id, mcp_host, port)
        else:
            # Local mode: start containers on this node
            pool_size = int(os.environ.get("APEX_POOL_SIZE", "1"))
            docker_cmd = os.environ.get("APEX_DOCKER_CMD", "docker")

            pool = ManagedContainerPool(
                world_id=world_id,
                docker_dir=docker_dir,
                pool_size=pool_size,
                base_port=_port_counter,
                docker_cmd=docker_cmd,
                session_concurrency=session_concurrency,
            )
            # Reserve ports for this pool
            _port_counter += pool_size

        _pool_registry[world_id] = pool
        return pool


# ── Custom Generate Function ─────────────────────────────────────────────────

async def generate_with_apex_docker(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Custom generate function for APEX Docker agent rollout.

    Used as: --custom-generate-function-path slime.rollout.apex_docker_rollout.generate_with_apex_docker

    Flow per sample:
    1. Read task_slug, world_id, docker_dir from sample.metadata
    2. Get or create ManagedContainerPool for this world
    3. Acquire container from pool (async lock)
    4. create_session(task_slug) — resets filesystem (~5s)
    5. snapshot_to_zip() → initial snapshot
    6. Create PerContainerMCPState with container's MCP URL
    7. Call mcp_agent_loop() (reused from mcp_agent_rollout.py)
    8. snapshot_to_zip() → final snapshot
    9. destroy_session(), release container
    10. Store snapshot paths in metadata, finalize tokens, return samples
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    task_slug = metadata.get("task_slug")
    world_id = metadata.get("world_id")
    docker_dir = metadata.get("docker_dir")
    task_id = metadata.get("task_id", "unknown")

    if not all([task_slug, world_id, docker_dir]):
        logger.error("Missing required metadata (task_slug, world_id, docker_dir) for sample")
        sample.status = Sample.Status.FAILED
        sample.response = "Error: Missing required metadata fields"
        sample.reward = 0.0
        _finalize_sample_tokens(args, sample)
        return [sample]

    # Get or create container pool for this world
    pool = await _get_or_create_pool(world_id, docker_dir)

    # Ensure pool is started (start() is idempotent and properly locked)
    await pool.start()

    # Acquire a container
    container_idx, container = await pool.acquire()
    mcp_state = None
    initial_snap = None
    final_snap = None

    is_remote = isinstance(pool, RemoteManagedPool)

    try:
        # Ensure container is alive (skip for remote — managed externally)
        if not is_remote:
            await asyncio.to_thread(container.ensure_running, task_slug)

        # Create session — resets filesystem to clean task state (~5s)
        logger.info("Creating session for %s on container %d (port %d)",
                     task_slug, container_idx, container.port)
        try:
            session = await asyncio.to_thread(container.create_session, task_slug)
        except RuntimeError:
            if is_remote:
                raise  # Can't restart remote containers
            # Retry: restart container, then retry create_session
            logger.warning("Session reset failed for %s, restarting container...", task_slug)
            await asyncio.to_thread(container.ensure_running, task_slug)
            session = await asyncio.to_thread(container.create_session, task_slug)

        # Take initial snapshot (scoped to this session's directories)
        initial_snap = tempfile.mktemp(suffix=".zip", prefix=f"snap_init_{task_id}_")
        await asyncio.to_thread(container.snapshot_to_zip, initial_snap,
                                session_id=session.session_id)

        # Create per-container MCP state (non-singleton)
        # Pass X-Session-Id header so MCP servers route to the session's directories
        session_headers = getattr(container, 'mcp_headers_for', lambda s: {})(session.session_id)
        mcp_state = PerContainerMCPState(
            mcp_url=container.mcp_url,
            name=f"apex-{world_id}-{container_idx}",
            headers=session_headers,
        )
        await mcp_state.initialize()

        # Get parser
        parser_type = getattr(args, "mcp_tool_parser", "qwen")
        parser = get_parser(parser_type)

        # Get max steps (APEX tasks need more steps)
        max_steps = getattr(args, "mcp_max_steps", 25)

        # Get custom system prompt template
        system_template = getattr(args, "mcp_system_prompt_template", None)

        # Prepare sampling params
        chat_sampling_params = {
            "temperature": sampling_params.get("temperature", args.rollout_temperature),
            "top_p": sampling_params.get("top_p", args.rollout_top_p),
            "max_tokens": sampling_params.get("max_new_tokens", args.rollout_max_response_len),
        }

        stop_seqs = parser.get_stop_sequences()
        if stop_seqs:
            chat_sampling_params["stop"] = stop_seqs

        # Run agent loop (reused from mcp_agent_rollout.py)
        t0 = time.time()
        samples = await mcp_agent_loop(
            args=args,
            initial_sample=sample,
            mcp_state=mcp_state,
            parser=parser,
            sampling_params=chat_sampling_params,
            max_steps=max_steps,
            system_prompt_template=system_template,
        )
        elapsed = time.time() - t0

        # Take final snapshot (scoped to this session's directories)
        final_snap = tempfile.mktemp(suffix=".zip", prefix=f"snap_final_{task_id}_")
        await asyncio.to_thread(container.snapshot_to_zip, final_snap,
                                session_id=session.session_id)

        # Destroy session (container stays alive for next rollout)
        await asyncio.to_thread(container.destroy_session, session.session_id)

        if not samples:
            sample.status = Sample.Status.FAILED
            sample.response = "Error: Agent loop produced no samples"
            sample.reward = 0.0
            sample.metadata["initial_snapshot"] = initial_snap
            sample.metadata["final_snapshot"] = final_snap
            sample.metadata["elapsed_seconds"] = elapsed
            _finalize_sample_tokens(args, sample)
            return [sample]

        # Store snapshot paths and elapsed time in the final sample's metadata
        # so the reward model (archipelago.py) can find them
        final_sample = samples[-1]
        final_sample.metadata["initial_snapshot"] = initial_snap
        final_sample.metadata["final_snapshot"] = final_snap
        final_sample.metadata["elapsed_seconds"] = elapsed

        # Finalize all samples with proper token information
        for s in samples:
            _finalize_sample_tokens(args, s)

        if evaluation:
            return final_sample

        # Leave reward=None for the RM dispatcher to compute
        # (archipelago.py will be called via --custom-rm-path)

        # Log completion
        logger.info(
            "APEX Docker agent completed: task=%s world=%s steps=%d elapsed=%.1fs",
            task_id, world_id, len(samples), elapsed,
        )

        return samples

    except Exception as e:
        logger.error("APEX Docker rollout failed for task %s: %s", task_id, e, exc_info=True)
        # Clean up snapshots on error
        for snap in (initial_snap, final_snap):
            if snap and os.path.exists(snap):
                try:
                    os.unlink(snap)
                except OSError:
                    pass
        sample.status = Sample.Status.FAILED
        sample.response = f"Error: {type(e).__name__}: {e}"
        sample.reward = 0.0
        _finalize_sample_tokens(args, sample)
        return [sample]

    finally:
        # Shut down per-container MCP state
        if mcp_state:
            try:
                await mcp_state.shutdown()
            except Exception:
                pass
        # Release container back to pool
        pool.release(container_idx)
