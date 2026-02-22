#!/bin/bash
#
# start_mcp_containers.sh — Start all canonical-ms containers on a dedicated MCP node.
#
# This script:
# 1. Starts one container per canonical-ms image, binding to 0.0.0.0:{port}
# 2. Waits for all containers to pass health checks
# 3. Generates a world_port_map.json mapping world_id → port
#
# Usage:
#   bash scripts/start_mcp_containers.sh [--base-port 9000] [--output /path/to/map.json]
#
# The generated map file should be placed on shared storage accessible by rollout nodes.
# Rollout nodes use APEX_MCP_HOST=<this-node-ip> APEX_MCP_PORT_MAP=<map-path>

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────

BASE_PORT="${1:-9000}"
OUTPUT_MAP="${2:-/data/mingye_b200-1/world_port_map.json}"
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=3

echo "=== MCP Container Launcher ==="
echo "BASE_PORT: ${BASE_PORT}"
echo "OUTPUT_MAP: ${OUTPUT_MAP}"

# ── Collect images ──────────────────────────────────────────────────────────

mapfile -t IMAGES < <(sudo docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep '^canonical-ms:' \
    | grep -v ':test$' \
    | sort)

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "ERROR: No canonical-ms images found. Run build_all.sh first."
    exit 1
fi

echo "Found ${#IMAGES[@]} canonical-ms images"
echo ""

# ── Stop any existing mcp-* containers ──────────────────────────────────────

echo "Cleaning up existing mcp-* containers..."
existing=$(sudo docker ps -a --filter "name=^mcp-" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$existing" ]; then
    echo "$existing" | xargs -r sudo docker rm -f 2>/dev/null || true
    echo "Cleaned up $(echo "$existing" | wc -l) containers"
fi
echo ""

# ── Start containers ────────────────────────────────────────────────────────

declare -A WORLD_PORT_MAP
idx=0

for img in "${IMAGES[@]}"; do
    port=$((BASE_PORT + idx))
    tag="${img#canonical-ms:}"

    # Extract world_id from tag (after ---)
    world_id=""
    if [[ "$tag" =~ ---(.+)$ ]]; then
        world_id="${BASH_REMATCH[1]}"
    else
        world_id="$tag"
    fi

    name="mcp-${idx}"

    echo "[${idx}] ${img}"
    echo "      port=${port}  world_id=${world_id}  name=${name}"

    sudo docker run -d \
        --name "$name" \
        --restart unless-stopped \
        -p "0.0.0.0:${port}:8000" \
        "$img" \
        /app/tools/start.sh "default" >/dev/null

    WORLD_PORT_MAP["$world_id"]=$port
    idx=$((idx + 1))
done

echo ""
echo "Started ${idx} containers (ports ${BASE_PORT}-$((BASE_PORT + idx - 1)))"

# ── Wait for health ─────────────────────────────────────────────────────────

echo ""
echo "Waiting for health checks (timeout: ${HEALTH_TIMEOUT}s)..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
healthy=0

while [ $healthy -lt $idx ] && [ $SECONDS -lt $deadline ]; do
    healthy=0
    for i in $(seq 0 $((idx - 1))); do
        port=$((BASE_PORT + i))
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            healthy=$((healthy + 1))
        fi
    done
    echo "  ${healthy}/${idx} healthy ($(( deadline - SECONDS ))s remaining)"
    if [ $healthy -lt $idx ]; then
        sleep $HEALTH_INTERVAL
    fi
done

if [ $healthy -lt $idx ]; then
    echo "WARNING: Only ${healthy}/${idx} containers are healthy after ${HEALTH_TIMEOUT}s"
    echo "Unhealthy containers:"
    for i in $(seq 0 $((idx - 1))); do
        port=$((BASE_PORT + i))
        if ! curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  mcp-${i} (port ${port})"
        fi
    done
else
    echo "All ${idx} containers healthy!"
fi

# ── Generate port map JSON ──────────────────────────────────────────────────

echo ""
echo "Generating port map → ${OUTPUT_MAP}"

python3 -c "
import json, sys
m = {}
$(for world_id in "${!WORLD_PORT_MAP[@]}"; do
    echo "m['${world_id}'] = ${WORLD_PORT_MAP[$world_id]}"
done)
with open('${OUTPUT_MAP}', 'w') as f:
    json.dump(m, f, indent=2, sort_keys=True)
print(f'Wrote {len(m)} entries')
"

echo ""
echo "=== Done ==="
echo ""
echo "To use from rollout nodes:"
echo "  export APEX_MCP_HOST=$(hostname -I | awk '{print $1}')"
echo "  export APEX_MCP_PORT_MAP=${OUTPUT_MAP}"
