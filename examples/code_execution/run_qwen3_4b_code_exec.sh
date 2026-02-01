#!/bin/bash
#
# Code Execution RL Training Script
# Uses Qwen3-4B model with code execution reward model
#
# Prerequisites:
#   - Download Qwen3-4B model to /root/Qwen3-4B (or change HF_CHECKPOINT below)
#   - Prepare training data in JSONL format with code execution metadata
#
# Usage:
#   bash examples/code_execution/run_qwen3_4b_code_exec.sh
#

# Cleanup any existing processes
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

set -ex

export PYTHONBUFFERED=16

# Check for NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Source model config
source "${PROJECT_ROOT}/scripts/models/qwen3-4B.sh"

# ============================================================
# Configuration - Modify these paths for your environment
# ============================================================

# Model paths
HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-/root/Qwen3-4B_torch_dist}"
SAVE_PATH="${SAVE_PATH:-/root/Qwen3-4B_code_exec/}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/examples/code_execution/sample_data.jsonl}"

# ============================================================

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_PATH}"
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key messages
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle

   # Code Execution RM settings
   --rm-type code_execution

   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 0.7

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   # Use same data for eval (in production, use a separate eval set)
   --eval-prompt-data code_exec "${TRAIN_DATA}"
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 4096
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # Uncomment to enable wandb logging
   # --use-wandb
   # --wandb-project slime-code-exec
   # --wandb-group qwen3-4B-code-exec
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build runtime environment
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
