#!/bin/bash
# Stop all mcp-* containers started by start_mcp_containers.sh
set -euo pipefail

echo "Stopping all mcp-* containers..."
containers=$(sudo docker ps -a --filter "name=^mcp-" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$containers" ]; then
    echo "$containers" | xargs -r sudo docker rm -f
    echo "Stopped $(echo "$containers" | wc -l) containers"
else
    echo "No mcp-* containers found"
fi
