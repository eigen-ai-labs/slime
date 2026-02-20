#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# ── Install missing deps (fastmcp not in base slime image) ──────────────────
pip install fastmcp requests 2>/dev/null || true

export PYTHONBUFFERED=16
export WANDB_KEY=${WANDB_KEY:-"YOUR_WANDB_KEY"}

# ── APEX Docker Configuration ────────────────────────────────────────────────

# Container management
export APEX_POOL_SIZE=${APEX_POOL_SIZE:-1}          # Containers per world (1 = one container per world)
export APEX_BASE_PORT=${APEX_BASE_PORT:-9000}        # Starting port for Docker containers
export APEX_DOCKER_CMD=${APEX_DOCKER_CMD:-"docker"}  # Docker command

# Archipelago grading
export APEX_JUDGE_MODEL=${APEX_JUDGE_MODEL:-"openai/google/gemini-3-flash-preview"}
export APEX_JUDGE_API_BASE=${APEX_JUDGE_API_BASE:-"https://api.gmi-serving.com/v1"}
export APEX_JUDGE_API_KEY=${APEX_JUDGE_API_KEY:-"YOUR_JUDGE_API_KEY"}
export APEX_GRADING_DIR=${APEX_GRADING_DIR:-"/data/mingye_b200-1/archipelago/grading"}

# ── NVLink detection ─────────────────────────────────────────────────────────

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="${APEX_DATA_DIR:-/data/mingye_b200-1/slime/examples/mcp_agent}"
MODEL_DIR="${MODEL_DIR:-/data/models}"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

# Training data (prepared by prepare_apex_docker_data.py)
APEX_TRAIN_DATA="${APEX_TRAIN_DATA:-${DATA_DIR}/apex_docker_tasks.jsonl}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-8B
   --ref-load ${MODEL_DIR}/Qwen3-8B_torch_dist
   --load ${MODEL_DIR}/Qwen3-8B_slime/
   --save ${MODEL_DIR}/Qwen3-8B_slime/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${APEX_TRAIN_DATA}
   --input-key messages
   --rollout-shuffle

   # APEX Docker custom generate function
   --custom-generate-function-path slime.rollout.apex_docker_rollout.generate_with_apex_docker

   # Archipelago grading reward model
   --custom-rm-path slime.rollout.rm_hub.archipelago.compute_archipelago_reward

   # APEX tasks are complex, need more agent steps than SerpAPI
   --mcp-max-steps 25

   --num-rollout 100
   # rollout-batch-size should be <= total containers across all worlds
   # Each sample needs a container for the duration of its rollout
   --rollout-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-4}
   # Fewer samples per prompt due to slow Docker rollouts
   --n-samples-per-prompt ${APEX_N_SAMPLES:-2}
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --over-sampling-batch-size ${APEX_ROLLOUT_BATCH_SIZE:-4}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 32
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-8B-apex-docker
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 64
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ── Launch ───────────────────────────────────────────────────────────────────

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --port 6381 --dashboard-port 8267 --dashboard-agent-listen-port 52400

sleep 5

# Build the runtime environment JSON
# Include mcp_session_manager.py and distill_apex_docker_v4.py on PYTHONPATH
EXTRA_PYTHONPATH="/data/mingye_b200-1"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXTRA_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"APEX_POOL_SIZE\": \"${APEX_POOL_SIZE}\",
    \"APEX_BASE_PORT\": \"${APEX_BASE_PORT}\",
    \"APEX_DOCKER_CMD\": \"${APEX_DOCKER_CMD}\",
    \"APEX_JUDGE_MODEL\": \"${APEX_JUDGE_MODEL}\",
    \"APEX_JUDGE_API_BASE\": \"${APEX_JUDGE_API_BASE}\",
    \"APEX_JUDGE_API_KEY\": \"${APEX_JUDGE_API_KEY}\",
    \"APEX_GRADING_DIR\": \"${APEX_GRADING_DIR}\"
  }
}"

RAY_ADDRESS="http://127.0.0.1:8267"

# Submit job (--no-wait so we can capture the job ID)
SUBMIT_OUTPUT=$(ray job submit --address="${RAY_ADDRESS}" --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} 2>&1)

echo "${SUBMIT_OUTPUT}"
JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'raysubmit_\w+' | head -1)
echo "Submitted job: ${JOB_ID}"

# Follow logs and block until job completes
ray job logs "${JOB_ID}" --address="${RAY_ADDRESS}" --follow
