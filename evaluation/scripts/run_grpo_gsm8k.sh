#!/usr/bin/env bash
# Strong GRPO run tuned for GSM8K.
# Dataset: allenai/RLVR-GSM (easy) or allenai/RLVR-GSM-MATH-IF-Mixed-Constraints
# (harder, better transfer). Switch via RLVR_DATASET env var.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${RLVR_DATASET:-allenai/RLVR-GSM}"

python rl_grpo_strong.py \
    --task gsm8k \
    --base_model "${BASE_MODEL:-Qwen/Qwen3-8B}" \
    --dataset_name "$DATASET" \
    --dataset_split train \
    --num_iterations 600 \
    --prompts_per_batch 32 \
    --group_size 8 \
    --learning_rate 5e-7 \
    --ppo_epochs 2 \
    --mini_batches_per_rollout 2 \
    --sampling_temperature 0.9 \
    --sampling_top_p 1.0 \
    --max_prompt_length 768 \
    --max_response_length 1024 \
    --loss_type dapo \
    --clip_eps_low 0.2 \
    --clip_eps_high 0.28 \
    --loss_normalizer token \
    --beta_kl 0.005 \
    --kl_estimator k3 \
    --ref_policy_update_freq 0 \
    --use_tis --tis_cap 2.0 \
    --drop_zero_variance_groups \
    --log_dir rl_grpo_runs \
    --save_every 50 \
    "$@"
