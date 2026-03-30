#!/bin/bash

# Scientific Judge training with GRPO (DAPO-style clipping)
# Reproduces SciJudge-Qwen3-4B from "AI Can Learn Scientific Taste" (arXiv:2603.14473)
#
# Base model: Qwen3-4B-Instruct-2507 (or Qwen/Qwen3-4B)
# Dataset:    OpenMOSS-Team/SciJudgeBench (720K citation-pair preference data)
# Method:     GRPO with DAPO-style asymmetric clipping
#
# Usage:
#   # Step 1: Prepare data
#   python examples/scijudge/prepare_data.py --output-dir /root/datasets/scijudge
#
#   # Step 2: Train
#   NUM_GPUS=8 ./examples/scijudge/run_scijudge_4b.sh
#
#   # Step 3: Evaluate
#   python examples/scijudge/evaluate.py \
#       --model-path /root/checkpoints/scijudge-4b/latest \
#       --data-dir /root/datasets/scijudge --split test

set -ex

# Configuration
TRAIN_BACKEND="megatron"
MODEL_NAME="Qwen3-4B-Instruct-2507"
NUM_GPUS=${NUM_GPUS:-8}
DATA_DIR=${DATA_DIR:-"/root/datasets/scijudge"}
MODEL_DIR=${MODEL_DIR:-"/root/models/Qwen3-4B-Instruct-2507"}
CKPT_DIR=${CKPT_DIR:-"/root/checkpoints/scijudge-4b"}

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# Cleanup stale processes
pkill -9 sglang 2>/dev/null || true
sleep 2
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
fi
pkill -9 slime 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16

# Download model if needed
mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$DATA_DIR")"
if [ ! -d "$MODEL_DIR" ]; then
    hf download Qwen/${MODEL_NAME} --local-dir "$MODEL_DIR"
fi

# Prepare data if needed
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "Preparing SciJudgeBench data..."
    python "${SLIME_DIR}/examples/scijudge/prepare_data.py" --output-dir "$DATA_DIR"
fi

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

# ── Model architecture args (Qwen3-4B with rotary-base 5000000 for Instruct-2507) ──
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_DIR"
    --rotary-base 5000000
)

# ── Rollout / data args ──
# Paper: 720K pairs, n_samples_per_prompt=8 (GRPO group size)
# The prompt is already in chat format (list of message dicts), so we use --apply-chat-template
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/train.jsonl"
    --input-key input
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle
    --custom-rm-path "${SLIME_DIR}/examples/scijudge/reward.py"
    --num-rollout 3000
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 4096
    --rollout-temperature 0.7
    --rollout-top-p 0.8
    --rollout-top-k 20
    --rollout-stop "<|end|>"
    --global-batch-size 512
)

# ── Evaluation args ──
EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data scijudge_test "${DATA_DIR}/test.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 2048
    --eval-temperature 0.7
    --eval-top-p 0.8
    --eval-top-k 20
)

# ── GRPO with DAPO-style asymmetric clipping ──
# Paper: KL coefficient beta=0.03
GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.03
    --kl-loss-type low_var_kl
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# ── Optimizer args ──
# Paper: LR 8e-7, cosine schedule, 5% warmup, AdamW (beta1=0.9, beta2=0.95, wd=0.1)
# Effective batch = per_device(8) * GPUs(64) * grad_accum(2) = 1024
# For 8 GPUs: per_device(8) * 8 GPUs * grad_accum = 512 with global-batch-size 512
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 8e-7
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# ── SGLang inference engine args ──
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6
    --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
)

# ── Wandb logging ──
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-scijudge
        --wandb-group scijudge-4b-grpo
        --wandb-key "${WANDB_API_KEY}"
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

# ── Checkpoint saving ──
SAVE_ARGS=(
    --save "${CKPT_DIR}"
    --save-interval 50
)

# ── Megatron backend args ──
BACKEND_ARGS=(
    --train-backend megatron
    --load "$MODEL_DIR"
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
)

MISC_ARGS=(
    --colocate
)

# ── Start Ray ──
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# ── Build runtime env ──
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# ── Launch training ──
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${MISC_ARGS[@]}"
