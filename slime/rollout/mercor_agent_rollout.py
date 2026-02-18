"""
Mercor Agent Rollout — RL training on Mercor APEX-Agents Docker environments.

Each rollout gets an isolated stateful container with 9 MCP tools
(Calendar, Chat, Code, Excel, Filesystem, Mail, PDFs, Powerpoint, Word).

Reuses:
- slime's mcp_agent_loop() for the multi-step agent loop
- Mercor's container API (/data/populate, /apps, /data/snapshot, /mcp/)
- Mercor's mcp_config_all_oss_servers.json for MCP server configuration

Usage:
    uv run python -m slime.train \
        --rollout-function-path slime.rollout.mercor_agent_rollout.generate_mercor_rollout \
        --custom-rm-path slime.rollout.mercor_grading.grade \
        --prompt-data mercor_train.jsonl \
        --apply-chat-template --input-key messages --metadata-key metadata \
        --mcp-max-steps 15 --mcp-tool-parser qwen
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import tarfile
import time
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import httpx

from slime.rollout.mcp import MCPClientConfig, MCPState, MCPTransport
from slime.rollout.mcp_agent_rollout import (
    _finalize_sample_tokens,
    mcp_agent_loop,
)
from slime.rollout.tool_parser import get_parser
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Path to the MCP config bundled with slime
_MCP_CONFIG_PATH = Path(__file__).parent / "mercor_mcp_config.json"


# ---------------------------------------------------------------------------
# Container Pool
# ---------------------------------------------------------------------------


class ContainerHandle:
    """A handle to a running Mercor environment container."""

    __slots__ = ("container_id", "port", "env_url", "mcp_url")

    def __init__(self, container_id: str, port: int):
        self.container_id = container_id
        self.port = port
        self.env_url = f"http://localhost:{port}"
        self.mcp_url = f"http://localhost:{port}/mcp/"


class MercorContainerPool:
    """Pool of pre-started Mercor Docker containers.

    Containers are started once and reused across rollouts.
    Between rollouts, the container is reset via POST /data/populate.
    """

    def __init__(
        self,
        docker_image: str,
        pool_size: int = 8,
        base_port: int = 9080,
        health_timeout: int = 120,
    ):
        self.docker_image = docker_image
        self.pool_size = pool_size
        self.base_port = base_port
        self.health_timeout = health_timeout
        self._queue: asyncio.Queue[ContainerHandle] = asyncio.Queue()
        self._all_handles: list[ContainerHandle] = []
        self._started = False

    async def start(self) -> None:
        """Start all containers and wait for them to be healthy."""
        if self._started:
            return

        logger.info("Starting %d Mercor containers from image %s", self.pool_size, self.docker_image)

        for i in range(self.pool_size):
            port = self.base_port + i
            container_name = f"mercor-env-{port}"

            # Stop existing container with the same name (if any)
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
            )

            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-p",
                    f"{port}:8080",
                    self.docker_image,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container on port {port}: {result.stderr}")

            container_id = result.stdout.strip()[:12]
            handle = ContainerHandle(container_id, port)
            self._all_handles.append(handle)
            logger.info("Started container %s on port %d", container_id, port)

        # Wait for all containers to be healthy (parallel)
        await asyncio.gather(*[self._wait_healthy(h) for h in self._all_handles])

        for h in self._all_handles:
            await self._queue.put(h)

        self._started = True
        logger.info("All %d Mercor containers are healthy", self.pool_size)

    async def _wait_healthy(self, handle: ContainerHandle) -> None:
        """Wait until container responds to GET /health."""
        start = time.monotonic()
        async with httpx.AsyncClient() as client:
            while time.monotonic() - start < self.health_timeout:
                try:
                    resp = await client.get(f"{handle.env_url}/health", timeout=5)
                    if resp.status_code == 200:
                        return
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError(f"Container {handle.container_id} on port {handle.port} not healthy after {self.health_timeout}s")

    async def acquire(self) -> ContainerHandle:
        """Get a container from the pool. Blocks if none available."""
        return await self._queue.get()

    async def release(self, handle: ContainerHandle) -> None:
        """Return a container to the pool."""
        await self._queue.put(handle)

    async def shutdown(self) -> None:
        """Stop and remove all containers."""
        logger.info("Shutting down %d Mercor containers", len(self._all_handles))
        for h in self._all_handles:
            subprocess.run(["docker", "rm", "-f", h.container_id], capture_output=True)
        self._all_handles.clear()
        self._started = False


# ---------------------------------------------------------------------------
# Container lifecycle helpers (ported from archipelago examples/hugging_face_task/main.py)
# ---------------------------------------------------------------------------


async def populate_container(
    env_url: str,
    world_snapshot_path: str,
    timeout: float = 300.0,
) -> None:
    """Populate a container with world snapshot data.

    Supports both .zip (HuggingFace format) and .tar.gz files.
    Follows the same logic as archipelago's examples/hugging_face_task/main.py.
    """
    snapshot_path = Path(world_snapshot_path)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if snapshot_path.suffix == ".zip":
            # HuggingFace world_files format: zip containing filesystem/ and .apps_data/
            with zipfile.ZipFile(snapshot_path, "r") as zf:
                names = zf.namelist()

                for subsystem in ("filesystem", ".apps_data"):
                    subsystem_files = [n for n in names if n.startswith(f"{subsystem}/")]
                    if not subsystem_files:
                        continue

                    # Convert zip entries to tar.gz (what the populate API expects)
                    tar_buf = io.BytesIO()
                    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
                        for name in subsystem_files:
                            new_name = name[len(f"{subsystem}/") :]
                            if not new_name:
                                continue
                            info = tarfile.TarInfo(name=new_name)
                            if name.endswith("/"):
                                info.type = tarfile.DIRTYPE
                                info.mode = 0o755
                                tar.addfile(info)
                            else:
                                data = zf.read(name)
                                info.size = len(data)
                                info.mode = 0o644
                                tar.addfile(info, io.BytesIO(data))
                    tar_buf.seek(0)

                    resp = await client.post(
                        f"{env_url}/data/populate",
                        files={"archive": (f"{subsystem}.tar.gz", tar_buf, "application/gzip")},
                        params={"subsystem": subsystem},
                    )
                    resp.raise_for_status()
                    logger.debug("Populated %s: %s", subsystem, resp.json())

        elif snapshot_path.suffix == ".gz" or snapshot_path.name.endswith(".tar.gz"):
            # Direct tar.gz — populate filesystem subsystem
            with open(snapshot_path, "rb") as f:
                resp = await client.post(
                    f"{env_url}/data/populate",
                    files={"archive": (snapshot_path.name, f.read(), "application/gzip")},
                    params={"subsystem": "filesystem"},
                )
                resp.raise_for_status()
        else:
            raise ValueError(f"Unsupported snapshot format: {snapshot_path}")


async def configure_mcp(env_url: str, timeout: float = 600.0) -> None:
    """Configure MCP servers on the container using the bundled config."""
    with open(_MCP_CONFIG_PATH) as f:
        mcp_config = json.load(f)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(f"{env_url}/apps", json=mcp_config)
        resp.raise_for_status()
        result = resp.json()
        logger.info("MCP servers configured: %s (%.1fms)", result.get("servers"), result.get("duration_ms"))


async def take_snapshot(env_url: str, output_path: str, timeout: float = 300.0) -> str:
    """Take a snapshot of the container's current state.

    Returns the path to the saved snapshot tar.gz file.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async with client.stream("POST", f"{env_url}/data/snapshot") as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
    return output_path


# ---------------------------------------------------------------------------
# Global pool (lazy init)
# ---------------------------------------------------------------------------

_pool: MercorContainerPool | None = None
_pool_lock = asyncio.Lock()


async def _get_pool(args: Namespace) -> MercorContainerPool:
    """Get or create the global container pool."""
    global _pool
    if _pool is not None and _pool._started:
        return _pool

    async with _pool_lock:
        if _pool is not None and _pool._started:
            return _pool

        docker_image = getattr(args, "mercor_docker_image", None)
        if not docker_image:
            raise ValueError("--mercor-docker-image is required for Mercor agent rollout")

        pool_size = getattr(args, "mercor_pool_size", 8)
        base_port = getattr(args, "mercor_base_port", 9080)

        _pool = MercorContainerPool(
            docker_image=docker_image,
            pool_size=pool_size,
            base_port=base_port,
        )
        await _pool.start()
        return _pool


# ---------------------------------------------------------------------------
# Custom generate function (per-rollout container lifecycle)
# ---------------------------------------------------------------------------


async def generate_with_mercor(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Custom generate function for Mercor agent rollout.

    Each call acquires a container, populates it, runs the agent loop,
    takes a snapshot, and releases the container.
    """
    pool = await _get_pool(args)
    handle = await pool.acquire()

    try:
        # 1. Populate container with world snapshot
        world_snapshot_path = sample.metadata.get("world_snapshot_path")
        if world_snapshot_path:
            await populate_container(handle.env_url, world_snapshot_path)

        # 2. Configure MCP servers
        await configure_mcp(handle.env_url)

        # 3. Create per-rollout MCPState (NOT singleton — each rollout gets its own)
        mcp_config = MCPClientConfig(
            name="mercor-env",
            transport=MCPTransport.STREAMABLE_HTTP,
            url=handle.mcp_url,
            concurrency_limit=16,
            timeout=60.0,
        )
        # Bypass singleton by calling __new__ + __init__ directly
        mcp_state = MCPState.__new__(MCPState)
        MCPState.__init__(mcp_state, configs=[mcp_config])
        await mcp_state.initialize()

        try:
            # 4. Get parser and sampling params
            parser = get_parser(getattr(args, "mcp_tool_parser", "qwen"))
            max_steps = getattr(args, "mcp_max_steps", 15)
            system_template = getattr(args, "mcp_system_prompt_template", None)

            chat_sampling_params = {
                "temperature": sampling_params.get("temperature", args.rollout_temperature),
                "top_p": sampling_params.get("top_p", args.rollout_top_p),
                "max_tokens": sampling_params.get("max_new_tokens", args.rollout_max_response_len),
            }
            stop_seqs = parser.get_stop_sequences()
            if stop_seqs:
                chat_sampling_params["stop"] = stop_seqs

            # 5. Run agent loop (reuse from mcp_agent_rollout)
            samples = await mcp_agent_loop(
                args=args,
                initial_sample=sample,
                mcp_state=mcp_state,
                parser=parser,
                sampling_params=chat_sampling_params,
                max_steps=max_steps,
                system_prompt_template=system_template,
            )
        finally:
            await mcp_state.shutdown()

        if not samples:
            sample.status = Sample.Status.FAILED
            sample.response = "Error: Agent loop produced no samples"
            sample.reward = 0.0
            _finalize_sample_tokens(args, sample)
            return [sample]

        # 6. Take snapshot for grading
        snapshot_dir = getattr(args, "mercor_snapshot_dir", "/tmp/mercor_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(
            snapshot_dir,
            f"snap_{sample.group_index}_{sample.index}_{int(time.time())}.tar.gz",
        )
        await take_snapshot(handle.env_url, snapshot_path)

        # Store snapshot path in all samples' metadata for grading
        for s in samples:
            s.metadata["final_snapshot_path"] = snapshot_path

        # 7. Finalize tokens
        for s in samples:
            _finalize_sample_tokens(args, s)

        if evaluation:
            return samples[-1]

        return samples

    finally:
        await pool.release(handle)


# ---------------------------------------------------------------------------
# Rollout function entry point
# ---------------------------------------------------------------------------


def generate_mercor_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> Any:
    """Main rollout function for Mercor agent.

    This function serves as the entry point for the rollout workflow.
    Use with: --rollout-function-path slime.rollout.mercor_agent_rollout.generate_mercor_rollout
    """
    from slime.rollout.sglang_rollout import generate_rollout

    original_custom_path = args.custom_generate_function_path
    args.custom_generate_function_path = "slime.rollout.mercor_agent_rollout.generate_with_mercor"

    try:
        return generate_rollout(args, rollout_id, data_source, evaluation)
    finally:
        args.custom_generate_function_path = original_custom_path


async def cleanup_mercor():
    """Clean up all Mercor containers."""
    global _pool
    if _pool is not None:
        await _pool.shutdown()
        _pool = None
