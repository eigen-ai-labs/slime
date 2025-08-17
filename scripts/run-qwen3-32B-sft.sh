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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-32B.sh"
MODEL_DIR="/workspace/models"
DATA_DIR="/workspace/data"
DATA_NAME="merged_file_distractor_toolace_ticketapi_tradingbot_filemt_v1_no_think"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-32B
   --ref-load ${MODEL_DIR}/Qwen3-32B_torch_dist
   --load ${MODEL_DIR}/Qwen3-32B_bfcl_${DATA_NAME}_16k/
   --save ${MODEL_DIR}/Qwen3-32B_bfcl_${DATA_NAME}_16k/
   --save-interval 100
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_example.generate_rollout
   --prompt-data ${DATA_DIR}/function_calling/${DATA_NAME}.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 5
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# tp:pp: 1,8 / 2,4
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-warmup-iters 100
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.9
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-bfcl
   --wandb-group qwen3-32B-sft-BFCL-${DATA_NAME}-16k
   --wandb-key 44eba75898c6a091a64aada6ea2ae738efe1f4fb
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
