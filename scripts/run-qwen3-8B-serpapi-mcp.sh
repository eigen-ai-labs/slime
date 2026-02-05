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

export PYTHONBUFFERED=16
export WANDB_KEY=${WANDB_KEY:-"YOUR_WANDB_KEY"}

# LLM Judge configuration (OpenRouter)
export OPENAI_API_BASE=${OPENAI_API_BASE:-https://openrouter.ai/api/v1}
export JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-4o-mini}
# export OPENAI_API_KEY=your_openrouter_key_here

# SerpAPI MCP URL - supports two ways:
# 1. Set SERPAPI_MCP_URL directly (full URL with token)
# 2. Set SERPAPI_KEY (token only, URL will be constructed)
if [ -z "${SERPAPI_MCP_URL}" ]; then
    SERPAPI_KEY="${SERPAPI_KEY:-YOUR_SERPAPI_KEY}"
    SERPAPI_MCP_URL="https://mcp.serpapi.com/${SERPAPI_KEY}/mcp"
fi

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="/root/slime/examples/mcp_agent"
MODEL_DIR="/root"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-8B
   --ref-load ${MODEL_DIR}/Qwen3-8B_torch_dist
   --load ${MODEL_DIR}/Qwen3-8B_slime/
   --save ${MODEL_DIR}/Qwen3-8B_slime/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/sample_data.jsonl
   --input-key messages
   --rollout-shuffle

   # MCP Agent configuration
   --custom-generate-function-path slime.rollout.mcp_agent_rollout.generate_with_mcp
   --mcp-server-url "${SERPAPI_MCP_URL}"

   # LLM Judge reward model
   --rm-type llm_judge

   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --over-sampling-batch-size 8
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
   --wandb-group qwen3-8B-serpapi-mcp
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

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

sleep 5

# Build the runtime environment JSON
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"OPENAI_API_BASE\": \"${OPENAI_API_BASE}\",
    \"JUDGE_MODEL\": \"${JUDGE_MODEL}\"
  }
}"

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
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
