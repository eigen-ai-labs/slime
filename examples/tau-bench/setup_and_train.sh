#!/bin/bash
# Tau-bench Setup and Training Script
# For single H200 GPU setup with OpenRouter user simulation

set -ex

# =============================================================================
# Configuration - MODIFY THESE BEFORE RUNNING
# =============================================================================
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_OPENROUTER_API_KEY_HERE}"

# Paths
WORK_DIR="/root"
SLIME_DIR="${WORK_DIR}/slime"
TAU_BENCH_DIR="${WORK_DIR}/tau-bench"
MODEL_NAME="Qwen3-4B-Instruct-2507"
MODEL_DIR="${WORK_DIR}/${MODEL_NAME}"
MODEL_TORCH_DIST_DIR="${WORK_DIR}/${MODEL_NAME}_torch_dist"
MODEL_SLIME_DIR="${WORK_DIR}/${MODEL_NAME}_slime"

# =============================================================================
# Step 1: Install slime
# =============================================================================
echo ">>> Step 1: Installing slime..."
cd ${WORK_DIR}
if [ ! -d "${SLIME_DIR}" ]; then
    git clone https://github.com/eigen-ai-labs/slime.git
fi
cd ${SLIME_DIR}
pip install -e . --no-deps

# =============================================================================
# Step 2: Install tau-bench
# =============================================================================
echo ">>> Step 2: Installing tau-bench..."
cd ${WORK_DIR}
if [ ! -d "${TAU_BENCH_DIR}" ]; then
    git clone https://github.com/JD-ETH/tau-bench.git
fi
cd ${TAU_BENCH_DIR}
git checkout feature/litellm-retry
pip install -e . --no-deps

# =============================================================================
# Step 3: Generate mock data for training
# =============================================================================
echo ">>> Step 3: Generating mock data..."
cd ${SLIME_DIR}/examples/tau-bench
python tau1_mock.py --local_dir ${TAU_BENCH_DIR}/

# =============================================================================
# Step 4: Download model from HuggingFace
# =============================================================================
echo ">>> Step 4: Downloading model..."
if [ ! -d "${MODEL_DIR}" ]; then
    huggingface-cli download Qwen/${MODEL_NAME} --local-dir ${MODEL_DIR}
else
    echo "Model already exists at ${MODEL_DIR}, skipping download."
fi

# =============================================================================
# Step 5: Convert HF checkpoint to torch_dist format
# =============================================================================
echo ">>> Step 5: Converting model to torch_dist format..."
if [ ! -d "${MODEL_TORCH_DIST_DIR}" ]; then
    cd ${SLIME_DIR}
    source scripts/models/qwen3-4B-Instruct-2507.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${MODEL_DIR} \
        --save ${MODEL_TORCH_DIST_DIR}
else
    echo "Torch dist checkpoint already exists at ${MODEL_TORCH_DIST_DIR}, skipping conversion."
fi

# =============================================================================
# Step 6: Create initial slime checkpoint directory (if needed)
# =============================================================================
echo ">>> Step 6: Preparing slime checkpoint directory..."
if [ ! -d "${MODEL_SLIME_DIR}" ]; then
    mkdir -p ${MODEL_SLIME_DIR}
fi

# =============================================================================
# Step 7: Run training
# =============================================================================
echo ">>> Step 7: Starting tau-bench training..."
echo "Using OpenRouter API with model: google/gemini-3-flash-preview"
echo "GPU config: 1x H200, tensor-parallel-size=1"

cd ${SLIME_DIR}
bash examples/tau-bench/run_qwen3_4B.sh

echo ">>> Training complete!"
