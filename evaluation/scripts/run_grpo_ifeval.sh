#!/usr/bin/env bash
# Strong GRPO run tuned for IFEval.
# Dataset: allenai/RLVR-IFeval — each row carries instruction_id_list + kwargs,
# scored by the exact IFEval checker. This is the same data Tulu-3 used.
set -euo pipefail

cd "$(dirname "$0")/.."

python rl_grpo_strong.py \
    --task ifeval \
    --base_model "${BASE_MODEL:-Qwen/Qwen3-8B}" \
    --dataset_name allenai/RLVR-IFeval \
    --dataset_split train \
    --num_iterations 400 \
    --prompts_per_batch 32 \
    --group_size 8 \
    --learning_rate 5e-7 \
    --ppo_epochs 2 \
    --mini_batches_per_rollout 2 \
    --sampling_temperature 1.0 \
    --sampling_top_p 1.0 \
    --max_prompt_length 1024 \
    --max_response_length 1024 \
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
