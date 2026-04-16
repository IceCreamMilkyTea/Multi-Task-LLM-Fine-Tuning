# Experiment Log

## Baselines (meta-llama/Llama-3.2-3B)

| Task | Baseline |
|------|----------|
| IFEval | 45.0% |
| GSM8K | 50.0% |
| HumanEval | 30.0% |

## Experiments

| exp_id | hypothesis | IFEval | GSM8K | HumanEval | Avg | Keep? | Notes |
|--------|-----------|--------|-------|-----------|-----|-------|-------|
| exp_01 | Baseline SFT: real task data (7473 GSM8K, 5000 Tulu-3, 5000 Code) with equal mix, 200 steps, lr=1e-4, rank=32 will outperform baselines | 18.5% | 24.0% | 40.0% | 27.5% | Discard | HumanEval strong (40%>30% target), but IFEval and GSM8K severely underperform. Only 200 steps = 800 examples seen out of 17k. Need more steps. Loss went 0.67→0.26. |
| exp_02 | Increasing steps to 500 and Tulu-3 data to 10k will improve IFEval and GSM8K because more training = more data coverage | 26.4% | 30.0% | 44.0% | 33.5% | Keep | All 3 tasks improved: IFEval +7.9pp, GSM8K +6pp, HumanEval +4pp. Total 22k samples, 500 steps. Loss fluctuated 1.38→1.26 (high variance from diverse data). Direction is strongly positive — more steps and more instruction data helps. Next: try even more steps or higher LR. checkpoint: `tinker://e2cf73b2-066a-500a-bf73-6583069a26f0:train:0/sampler_weights/exp_02_gsm7k_tulu10k_code5k_lr1e4_steps500_rank32` |
| exp_0416_0651 | Increasing steps from 500 to 1000 (same data: 7473 GSM8K, 10k Tulu, 5k Code, lr=1e-4, rank=32) will continue improving all metrics | 17.0% | 29.0% | 47.0% | 31.0% | No | IFEval catastrophically regressed (-9.4pp), GSM8K flat (-1pp), HumanEval improved (+3pp). 1000 steps causes overtraining — the model forgets instruction following. Loss oscillated 0.15-2.07 (very high variance). The sweet spot is around 500 steps for this data mix. checkpoint: `tinker://666c2755-35d3-5890-afe3-2269b543c4b5:train:0/sampler_weights/exp_0416_0651_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` save_state: `tinker://666c2755-35d3-5890-afe3-2269b543c4b5:train:0/weights/exp_0416_0651_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` |
| exp_0416_0755 | Increasing Tulu data from 10k to 20k (500 steps, same lr/rank) will boost IFEval because more instruction data helps | 20.0% | 30.0% | 45.0% | 31.7% | No | IFEval still regressed vs exp_02 (-6.4pp), GSM8K flat, HumanEval +1pp. More Tulu data did NOT help IFEval — the additional streaming samples may be lower quality or less relevant. The data mix change shifted proportions (62% Tulu vs 45% before) which may have diluted math/code signal. checkpoint: `tinker://670aed52-a66d-5cc8-b981-d91529de9c36:train:0/sampler_weights/exp_0416_0755_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32` save_state: `tinker://670aed52-a66d-5cc8-b981-d91529de9c36:train:0/weights/exp_0416_0755_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32` |

## Analysis

### Key findings so far:
1. **200→500 steps improved all metrics**, but **500→1000 steps caused overtraining**: IFEval crashed from 26.4% to 17%
2. **HumanEval is easiest to improve**: 30% → 40% → 44% → 47% across experiments — keeps going up with more training
3. **IFEval is fragile**: It's the most sensitive to overtraining AND data ratio changes. exp_02 at 500 steps with 10k Tulu remains the best
4. **GSM8K plateaued**: 30% across all experiments with 500+ steps — more training or data doesn't help math
5. **Overtraining pattern**: After ~500 steps, the model starts losing general instruction-following ability while continuing to improve on code
6. **More Tulu data doesn't help**: 10k→20k Tulu actually hurt IFEval (-6.4pp). The streaming data may have quality issues or the ratio shift hurt.

### Best checkpoint: exp_02
- IFEval: 26.4%, GSM8K: 30.0%, HumanEval: 44.0%, Avg: 33.5%
- Config: 500 steps, 7473 GSM8K, 10k Tulu, 5k Code, lr=1e-4, rank=32

### Suggested next experiments:
- Try higher LR (2e-4 or 3e-4) with 500 steps and original data mix (10k Tulu) to learn faster
- Try larger batch size (8 or 16) with 500 steps for more stable gradients
- Try reducing code data (3k) and increasing GSM8K exposure to boost math
- Try LoRA rank 64 for more model capacity
- Once a good 3B config is found, scale to 8B model
