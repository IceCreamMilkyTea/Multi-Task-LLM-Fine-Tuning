#!/usr/bin/env bash
# Strong GRPO run tuned for HumanEval-style code.
# Dataset: PrimeIntellect/verifiable-coding-problems (has unit tests the verifier
# can execute). You CANNOT train on HumanEval itself — it's eval-only (164 items).
set -euo pipefail

cd "$(dirname "$0")/.."

python rl_grpo_strong.py \
    --task humaneval \
    --base_model "${BASE_MODEL:-Qwen/Qwen3-8B}" \
    --dataset_name "${RLVR_DATASET:-PrimeIntellect/verifiable-coding-problems}" \
    --dataset_split train \
    --num_iterations 500 \
    --prompts_per_batch 16 \
    --group_size 8 \
    --learning_rate 3e-7 \
    --ppo_epochs 2 \
    --mini_batches_per_rollout 2 \
    --sampling_temperature 0.8 \
    --sampling_top_p 0.95 \
    --max_prompt_length 1024 \
    --max_response_length 1536 \
    --loss_type dapo \
    --clip_eps_low 0.2 \
    --clip_eps_high 0.28 \
    --loss_normalizer token \
    --beta_kl 0.01 \
    --kl_estimator k3 \
    --ref_policy_update_freq 0 \
    --use_tis --tis_cap 2.0 \
    --drop_zero_variance_groups \
    --log_dir rl_grpo_runs \
    --save_every 50 \
    "$@"
