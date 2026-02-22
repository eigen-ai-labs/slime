#!/usr/bin/env python3
"""
End-to-end MCP test: pick a real task from training data, run the full
session lifecycle with actual MCP tool calls against a remote container.

Usage:
    python3 scripts/test_e2e_mcp.py <MCP_HOST>
"""
import asyncio
import json
import os
import sys
import tempfile
import time

# Import from slime repo (not the stale /data/mingye_b200-1/container_overlay copy)
SLIME_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SLIME_ROOT)

MCP_HOST = sys.argv[1] if len(sys.argv) > 1 else "85.234.91.166"
PORT_MAP = "/data/mingye_b200-1/world_port_map.json"
TASKS_FILE = "/data/mingye_b200-1/slime/examples/mcp_agent/apex_docker_tasks.jsonl"


def main():
    # Load port map
    with open(PORT_MAP) as f:
        port_map = json.load(f)

    # Pick first task
    with open(TASKS_FILE) as f:
        task_data = json.loads(f.readline())

    meta = task_data["metadata"]
    world_id = meta["world_id"]
    task_slug = meta["task_slug"]
    task_id = meta["task_id"]
    prompt = task_data["messages"][0]["content"]

    port = port_map[world_id]
    print(f"=== E2E MCP Test ===")
    print(f"  world_id:  {world_id}")
    print(f"  task_slug: {task_slug}")
    print(f"  task_id:   {task_id}")
    print(f"  host:port: {MCP_HOST}:{port}")
    print(f"  prompt:    {prompt[:100]}...")
    print()

    # Import SessionContainer
    from container_overlay.mcp_session_manager import SessionContainer

    # Create a remote SessionContainer (no docker, connect-only)
    sc = SessionContainer(
        image_tag="remote",
        env_file="/dev/null",
        port=port,
        host=MCP_HOST,
    )
    sc.container_id = f"remote-{MCP_HOST}-{port}"

    # Step 1: Health check
    print("[1/7] Health check...")
    sc._wait_for_health(timeout=30)
    print("  OK")

    # Step 2: Create session
    print(f"[2/7] Creating session (task_slug={task_slug})...")
    session = sc.create_session(task_slug)
    sid = session.session_id
    print(f"  OK session_id={sid}")

    try:
        # Step 3: Initial snapshot
        print("[3/7] Taking initial snapshot...")
        t0 = time.time()
        snap_init = tempfile.mktemp(suffix=".zip", prefix="e2e_init_")
        sc.snapshot_to_zip(snap_init, session_id=sid)
        snap_size = os.path.getsize(snap_init) / 1024 / 1024
        print(f"  OK {snap_size:.1f} MB ({time.time()-t0:.1f}s) → {snap_init}")

        # Step 4: List MCP tools
        print("[4/7] Listing MCP tools...")
        tools = sc.get_tools()
        print(f"  OK {len(tools)} tools available:")
        for t in tools[:10]:
            name = t.name if hasattr(t, 'name') else t.get('name', t)
            print(f"    - {name}")
        if len(tools) > 10:
            print(f"    ... and {len(tools)-10} more")

        # Step 5: Call a tool via MCP with session header
        print("[5/7] Calling MCP tool (list_directory '/')...")
        result = asyncio.run(_call_tool(sc, sid, tools))
        print(f"  OK result: {str(result)[:300]}")

        # Step 6: Final snapshot
        print("[6/7] Taking final snapshot...")
        t0 = time.time()
        snap_final = tempfile.mktemp(suffix=".zip", prefix="e2e_final_")
        sc.snapshot_to_zip(snap_final, session_id=sid)
        snap_size = os.path.getsize(snap_final) / 1024 / 1024
        print(f"  OK {snap_size:.1f} MB ({time.time()-t0:.1f}s) → {snap_final}")

    finally:
        # Step 7: Destroy session
        print(f"[7/7] Destroying session {sid}...")
        sc.destroy_session(sid)
        print("  OK")

    # Cleanup temp files
    for f in (snap_init, snap_final):
        if f and os.path.exists(f):
            os.unlink(f)

    print()
    print("=== All E2E tests passed ===")


async def _call_tool(sc, session_id, tools):
    """Call a filesystem tool via MCP with session header."""
    from fastmcp import Client as FastMCPClient

    # Find a read-only tool to test (tools may be dicts or objects)
    def _get_name(t):
        if isinstance(t, dict):
            return t.get("function", {}).get("name", "") or t.get("name", "")
        return getattr(t, "name", "")

    tool_name = None
    tool_args = {}
    for t in tools:
        name = _get_name(t)
        if name == "filesystem_list_files":
            tool_name = "filesystem_list_files"
            tool_args = {"path": "/"}
            break
        if name == "filesystem_get_directory_tree":
            tool_name = "filesystem_get_directory_tree"
            tool_args = {"path": "/"}
            break

    if not tool_name:
        print("  WARN: no filesystem tool found, skipping tool call")
        return "skipped"

    headers = sc.mcp_headers_for(session_id)

    # Use streamablehttp transport with session header
    from fastmcp.client.transports import StreamableHttpTransport
    transport = StreamableHttpTransport(sc.mcp_url, headers=headers)
    client = FastMCPClient(transport=transport, timeout=30)

    async with client:
        result = await client.call_tool(tool_name, tool_args)
    return result


if __name__ == "__main__":
    main()
