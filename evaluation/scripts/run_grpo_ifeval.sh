#!/usr/bin/env bash
# Strong GRPO on RLVR-IFeval (aligned with IFEval prompt_strict_acc).
set -euo pipefail
cd "$(dirname "$0")/../.."

RESUME_FROM="${RESUME_FROM:-tinker://e37ebd44-7f95-5f13-a9c5-902885b445a0:train:0/weights/exp_0422_stage2_rank128_ifdata_state}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
NAME="${NAME:-grpo_strong_ifeval}"

python -u evaluation/rl_grpo_strong.py \
    --task ifeval \
    --ifeval_synthesize \
    --n_synth "${N_SYNTH:-4000}" \
    --model "$MODEL" \
    --resume_from "$RESUME_FROM" \
    --num_iterations "${ITERS:-40}" \
    --prompts_per_iter "${PROMPTS:-8}" \
    --group_size "${GROUP:-8}" \
    --ppo_epochs "${PPO_EPOCHS:-2}" \
    --mini_batch_size "${MINI_BATCH:-4}" \
    --lr "${LR:-1e-6}" \
    --max_new_tokens "${MAX_NEW:-384}" \
    --temperature "${TEMP:-1.0}" \
    --loss_fn "${LOSS_FN:-ppo}" \
    --kl_coef "${KL_COEF:-0.01}" \
    --save_every 20 \
    --checkpoint_name "$NAME"
