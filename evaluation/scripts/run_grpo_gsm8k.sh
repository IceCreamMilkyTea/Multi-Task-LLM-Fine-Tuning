#!/usr/bin/env bash
# Strong GRPO on RLVR-GSM (verifiable GSM8K-style math).
set -euo pipefail
cd "$(dirname "$0")/../.."

RESUME_FROM="${RESUME_FROM:-tinker://e37ebd44-7f95-5f13-a9c5-902885b445a0:train:0/weights/exp_0422_stage2_rank128_ifdata_state}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
NAME="${NAME:-grpo_strong_gsm8k}"
DATASET="${DATASET:-allenai/RLVR-GSM}"

python -u evaluation/rl_grpo_strong.py \
    --task gsm8k \
    --dataset_name "$DATASET" \
    --model "$MODEL" \
    --resume_from "$RESUME_FROM" \
    --num_iterations "${ITERS:-80}" \
    --prompts_per_iter "${PROMPTS:-16}" \
    --group_size "${GROUP:-8}" \
    --ppo_epochs "${PPO_EPOCHS:-2}" \
    --mini_batch_size "${MINI_BATCH:-4}" \
    --lr "${LR:-1e-6}" \
    --max_new_tokens "${MAX_NEW:-512}" \
    --temperature "${TEMP:-1.0}" \
    --loss_fn "${LOSS_FN:-ppo}" \
    --kl_coef "${KL_COEF:-0.01}" \
    --drop_zero_variance \
    --save_every 20 \
    --checkpoint_name "$NAME"
