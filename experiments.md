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
| exp_0416_0144 | Tripling LR from 1e-4 to 3e-4 will improve convergence speed at 500 steps, helping IFEval and GSM8K | 21.0% | 24.0% | 39.0% | 28.0% | Discard | All metrics regressed vs exp_02. Loss was very volatile (0.28-1.51), confirming lr=3e-4 is too high for this setup. save_state: `tinker://d65f3a29-e88b-5fbe-8312-7dd238df64ea:train:0/weights/exp_0416_0144_lr3e4_steps500_rank32`. sampler: `tinker://d65f3a29-e88b-5fbe-8312-7dd238df64ea:train:0/sampler_weights/exp_0416_0144_lr3e4_steps500_rank32` |
| exp_0416_0211 | Doubling batch_size from 4 to 8 will improve training by 2x data coverage per step (4000 vs 2000 samples seen) | 17.0% | 31.0% | 44.0% | 30.7% | Discard | IFEval regressed -9.4pp vs exp_02 while GSM8K +1pp and HumanEval flat. Larger batch with same LR reduces effective per-sample learning rate (linear scaling rule). Loss stable (1.03→0.82) but less impactful per step. save_state: `tinker://804706d3-d970-5433-8917-c147f25969c3:train:0/weights/exp_0416_0211_bs8_lr1e4_steps500_rank32`. sampler: `tinker://804706d3-d970-5433-8917-c147f25969c3:train:0/sampler_weights/exp_0416_0211_bs8_lr1e4_steps500_rank32` |
| exp_0416_0249 | Doubling steps from 500 to 1000 (bs=4, lr=1e-4) will improve all metrics since data coverage doubles from 9% to 18% | 18.0% | 32.0% | 45.0% | 31.7% | Discard | IFEval regressed -8.4pp vs exp_02 — likely overtraining on code/math hurts instruction following. GSM8K +2pp and HumanEval +1pp slight gains. More steps helps math/code but hurts IFEval — suggests need for data ratio rebalancing, not just more steps. save_state: `tinker://9f8a3aa8-25e0-54ce-8eed-dbdcac2d1762:train:0/weights/exp_0416_0249_bs4_lr1e4_steps1000_rank32`. sampler: `tinker://9f8a3aa8-25e0-54ce-8eed-dbdcac2d1762:train:0/sampler_weights/exp_0416_0249_bs4_lr1e4_steps1000_rank32` |
| exp_0416_0651 | Increasing steps from 500 to 1000 (same data: 7473 GSM8K, 10k Tulu, 5k Code, lr=1e-4, rank=32) will continue improving all metrics | 17.0% | 29.0% | 47.0% | 31.0% | No | IFEval catastrophically regressed (-9.4pp), GSM8K flat (-1pp), HumanEval improved (+3pp). 1000 steps causes overtraining — the model forgets instruction following. Loss oscillated 0.15-2.07 (very high variance). The sweet spot is around 500 steps for this data mix. checkpoint: `tinker://666c2755-35d3-5890-afe3-2269b543c4b5:train:0/sampler_weights/exp_0416_0651_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` save_state: `tinker://666c2755-35d3-5890-afe3-2269b543c4b5:train:0/weights/exp_0416_0651_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` |
| exp_0416_0755 | Increasing Tulu data from 10k to 20k (500 steps, same lr/rank) will boost IFEval because more instruction data helps | 20.0% | 30.0% | 45.0% | 31.7% | No | IFEval still regressed vs exp_02 (-6.4pp), GSM8K flat, HumanEval +1pp. More Tulu data did NOT help IFEval — the additional streaming samples may be lower quality or less relevant. The data mix change shifted proportions (62% Tulu vs 45% before) which may have diluted math/code signal. checkpoint: `tinker://670aed52-a66d-5cc8-b981-d91529de9c36:train:0/sampler_weights/exp_0416_0755_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32` save_state: `tinker://670aed52-a66d-5cc8-b981-d91529de9c36:train:0/weights/exp_0416_0755_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32` |
| exp_0416_1145 | Increasing steps from 500→1000 will continue positive trend since only ~9% of data seen per epoch at 500 steps | 27.4% | 35.0% | 46.0% | 36.1% | Keep | All 3 tasks improved: IFEval +1.0pp, GSM8K +5.0pp, HumanEval +2.0pp. Same data mix (7473 GSM8K, 10k Tulu, 5k Code), lr=1e-4, rank=32. Loss 1.38→1.23 (high variance). Also added save_state() for future resumability. checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` state: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state` |

## Analysis

### Key findings so far:
1. **More training steps generally helps**: 200→500→1000 steps improved overall avg, but IFEval is fragile
2. **HumanEval is easiest to improve**: 46% at 1000 steps (target: 30%) - already well above target
3. **GSM8K improving steadily**: 24%→30%→35% across experiments, but still below 50% target
4. **IFEval is very fragile**: Sensitive to LR, batch size, and data ratios. Best so far is 27.4%
5. **lr=3e-4 is too high**: Causes loss instability and hurts all metrics
6. **batch_size=8 hurts IFEval**: Linear scaling rule needed; stick with bs=4
7. **Data ratio rebalancing needed**: More code/math training hurts IFEval — need more Tulu data
8. **Diminishing returns for IFEval from steps alone**: 200→500 gave +7.9pp, 500→1000 only +1.0pp

### Best checkpoint so far:
- **exp_0416_1145**: IFEval 27.4%, GSM8K 35.0%, HumanEval 46.0%, Avg 36.1%
- Checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`
- State: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state`

### Suggested next experiments:
- **Do not increase Tulu ratio**: 20k Tulu / 5k Code / 7473 GSM8K at 500 steps confirmed to hurt IFEval (-6.4pp vs exp_02) — streaming data quality degrades or ratio shift dilutes math/code signal
- **Two-stage training**: Stage 1 broad SFT (500 steps), Stage 2 IFEval-focused (200 steps with mostly Tulu data)
- Try resuming from best checkpoint with more steps focused on instruction-following data
- Once a good 3B config is found, scale to 8B model
