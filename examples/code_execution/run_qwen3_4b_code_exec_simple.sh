#!/bin/bash
#
# Simple Code Execution RL Training Script (2 GPUs)
# A minimal setup for testing with Qwen3-4B
#
# Usage:
#   # Set your model path
#   export HF_CHECKPOINT=/path/to/Qwen3-4B
#   bash examples/code_execution/run_qwen3_4b_code_exec_simple.sh
#

set -ex

# Cleanup
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Source model config
source "${PROJECT_ROOT}/scripts/models/qwen3-4B.sh"

# Paths - modify as needed
HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/examples/code_execution/sample_data.jsonl}"
SAVE_PATH="${SAVE_PATH:-/tmp/qwen3-4b-code-exec-test/}"

mkdir -p "${SAVE_PATH}"

# Start Ray (2 GPUs)
NUM_GPUS="${NUM_GPUS:-2}"
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   \
   ${MODEL_ARGS[@]} \
   \
   --hf-checkpoint "${HF_CHECKPOINT}" \
   --save "${SAVE_PATH}" \
   --save-interval 10 \
   \
   --prompt-data "${TRAIN_DATA}" \
   --input-key messages \
   --label-key label \
   --metadata-key metadata \
   --apply-chat-template \
   --rollout-shuffle \
   --rm-type code_execution \
   \
   --num-rollout 100 \
   --rollout-batch-size 8 \
   --n-samples-per-prompt 2 \
   --rollout-max-response-len 2048 \
   --rollout-temperature 0.7 \
   --global-batch-size 16 \
   \
   --tensor-model-parallel-size ${NUM_GPUS} \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 4096 \
   \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.01 \
   --eps-clip 0.2 \
   \
   --optimizer adam \
   --lr 5e-7 \
   --lr-decay-style constant \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --attention-backend flash \
   \
   --rollout-num-gpus-per-engine ${NUM_GPUS} \
   --sglang-mem-fraction-static 0.6
