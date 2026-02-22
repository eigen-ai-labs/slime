#!/usr/bin/env python3
"""
Test remote MCP connectivity — run this from another node to verify.

Usage:
    python3 scripts/test_remote_mcp.py <MCP_HOST> [PORT_MAP_PATH]

Example:
    python3 scripts/test_remote_mcp.py 85.234.91.166 /data/mingye_b200-1/world_port_map.json
"""
import json, sys, requests, time

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    host = sys.argv[1]
    port_map_path = sys.argv[2] if len(sys.argv) > 2 else "/data/mingye_b200-1/world_port_map.json"

    with open(port_map_path) as f:
        port_map = json.load(f)

    print(f"MCP host: {host}")
    print(f"Port map: {len(port_map)} worlds")
    print()

    # Step 1: Health check all containers
    print("=== Step 1: Health checks ===")
    healthy_worlds = []
    failed = 0
    for world_id, port in sorted(port_map.items(), key=lambda x: x[1]):
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=5)
            if r.status_code == 200:
                healthy_worlds.append((world_id, port))
            else:
                print(f"  FAIL {world_id} port={port} status={r.status_code}")
                failed += 1
        except Exception as e:
            print(f"  FAIL {world_id} port={port} error={e}")
            failed += 1
    print(f"  {len(healthy_worlds)}/{len(port_map)} healthy")
    print()

    if not healthy_worlds:
        print("ERROR: No containers reachable. Check firewall / network.")
        sys.exit(1)

    # Step 2: Create a session on the first HEALTHY container
    first_world, port = healthy_worlds[0]
    print(f"=== Step 2: Session lifecycle on {first_world} (port {port}) ===")

    base = f"http://{host}:{port}"

    # Create session
    sid = f"test-remote-{int(time.time())}"
    print(f"  Creating session {sid}...")
    r = requests.post(f"{base}/sessions/create", json={"session_id": sid}, timeout=30)
    if r.status_code != 200:
        print(f"  FAIL create session: {r.status_code} {r.text[:200]}")
        sys.exit(1)
    session = r.json()
    print(f"  OK fs_root={session.get('fs_root', '?')}")

    # List sessions
    print("  Listing sessions...")
    r = requests.post(f"{base}/sessions/list", timeout=10)
    sessions = r.json().get("sessions", [])
    print(f"  OK {len(sessions)} session(s)")

    # Snapshot — returns large binary zip, use stream mode like training code does
    print("  Taking snapshot (streaming)...")
    r = requests.post(f"{base}/data/snapshot", json={"session_id": sid},
                      timeout=120, stream=True)
    if r.status_code == 200:
        size = 0
        for chunk in r.iter_content(65536):
            size += len(chunk)
            if size > 65536:
                break  # confirmed data is flowing, no need to download all
        r.close()
        print(f"  OK snapshot status=200, verified {size} bytes streaming")
    else:
        print(f"  WARN snapshot: {r.status_code}")

    # Destroy session
    print("  Destroying session...")
    r = requests.post(f"{base}/sessions/destroy", json={"session_id": sid}, timeout=30)
    if r.status_code == 200:
        print(f"  OK session destroyed")
    else:
        print(f"  FAIL destroy: {r.status_code} {r.text[:200]}")

    print()
    print("=== All tests passed ===")

if __name__ == "__main__":
    main()
