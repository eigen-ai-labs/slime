#!/bin/bash
#
# launch-apex-docker.sh — Run on BARE METAL to launch the slime training container.
#
# This script:
# 1. Sets environment variables for APEX Docker training
# 2. Launches the slimerl/slime:latest container with all required mounts
# 3. Executes the inner training script (run-qwen3-8B-apex-docker.sh) inside the container
#
# Usage:
#   sudo bash scripts/launch-apex-docker.sh
#
# Required mounts:
#   /var/run/docker.sock  — Docker-in-Docker: lets training process manage APEX containers
#   /usr/bin/docker       — Provides the docker CLI binary inside the container
#   /data:/data           — Access to APEX Docker exports, archipelago grading, mcp_session_manager
#   slime:/root/slime     — Overlays stale container slime with our updated code
#   --network host        — Container shares host network, can reach APEX containers on localhost

set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "=== APEX Docker Training Launcher ==="
echo "SLIME_ROOT: ${SLIME_ROOT}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

# ── APEX Docker Configuration (override via environment) ─────────────────────

export APEX_POOL_SIZE="${APEX_POOL_SIZE:-1}"
export APEX_BASE_PORT="${APEX_BASE_PORT:-9000}"
export APEX_DOCKER_CMD="${APEX_DOCKER_CMD:-docker}"
export APEX_SESSION_CONCURRENCY="${APEX_SESSION_CONCURRENCY:-4}"

# Archipelago grading
export APEX_JUDGE_MODEL="${APEX_JUDGE_MODEL:-openai/google/gemini-3-flash-preview}"
export APEX_JUDGE_API_BASE="${APEX_JUDGE_API_BASE:-https://api.gmi-serving.com/v1}"
export APEX_JUDGE_API_KEY="${APEX_JUDGE_API_KEY:-YOUR_JUDGE_API_KEY}"
export APEX_GRADING_DIR="${APEX_GRADING_DIR:-/data/mingye_b200-1/archipelago/grading}"

# W&B
export WANDB_KEY="${WANDB_KEY:-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}"

# Training data (prepared by prepare_apex_docker_data.py)
export APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-/data/mingye_b200-1/slime/examples/mcp_agent/apex_docker_tasks.jsonl}"

# ── Optional overrides ───────────────────────────────────────────────────────

APEX_ROLLOUT_BATCH_SIZE="${APEX_ROLLOUT_BATCH_SIZE:-32}"
APEX_N_SAMPLES="${APEX_N_SAMPLES:-2}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:latest}"

# ── Verify prerequisites ─────────────────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found on host"
    exit 1
fi

if [ ! -S /var/run/docker.sock ]; then
    echo "ERROR: /var/run/docker.sock not found"
    exit 1
fi

if [ ! -f "${SLIME_ROOT}/scripts/run-qwen3-8B-apex-docker.sh" ]; then
    echo "ERROR: Training script not found: ${SLIME_ROOT}/scripts/run-qwen3-8B-apex-docker.sh"
    exit 1
fi

# ── Launch container ─────────────────────────────────────────────────────────

echo ""
echo "Launching container: ${SLIME_IMAGE}"
echo "  APEX_POOL_SIZE=${APEX_POOL_SIZE}"
echo "  APEX_BASE_PORT=${APEX_BASE_PORT}"
echo "  APEX_JUDGE_MODEL=${APEX_JUDGE_MODEL}"
echo "  APEX_TRAIN_DATA=${APEX_TRAIN_DATA}"
echo ""

exec sudo docker run --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /data:/data \
    -v "${SLIME_ROOT}":/root/slime \
    -e APEX_POOL_SIZE="${APEX_POOL_SIZE}" \
    -e APEX_BASE_PORT="${APEX_BASE_PORT}" \
    -e APEX_DOCKER_CMD="${APEX_DOCKER_CMD}" \
    -e APEX_JUDGE_MODEL="${APEX_JUDGE_MODEL}" \
    -e APEX_JUDGE_API_BASE="${APEX_JUDGE_API_BASE}" \
    -e APEX_JUDGE_API_KEY="${APEX_JUDGE_API_KEY}" \
    -e APEX_GRADING_DIR="${APEX_GRADING_DIR}" \
    -e APEX_TRAIN_DATA="${APEX_TRAIN_DATA}" \
    -e APEX_SESSION_CONCURRENCY="${APEX_SESSION_CONCURRENCY}" \
    -e APEX_ROLLOUT_BATCH_SIZE="${APEX_ROLLOUT_BATCH_SIZE}" \
    -e APEX_N_SAMPLES="${APEX_N_SAMPLES}" \
    -e WANDB_KEY="${WANDB_KEY}" \
    "${SLIME_IMAGE}" \
    bash /root/slime/scripts/run-qwen3-8B-apex-docker.sh
