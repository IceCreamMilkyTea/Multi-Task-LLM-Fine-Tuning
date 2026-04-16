# Experiment Log

## Baselines (meta-llama/Llama-3.2-3B)

| Task | Baseline |
|------|----------|
| IFEval | 45.0% |
| GSM8K | 50.0% |
| HumanEval | 30.0% |

## Experiments

All experiments use base model `meta-llama/Llama-3.2-3B`, data mix of GSM8K + Tulu-3 SFT + OpenCodeInstruct, LoRA fine-tuning. Sorted by timestamp. Source branch noted for traceability.

| # | exp_id | Steps | BS | LR | Rank | GSM8K | Tulu | Code | IFEval | GSM8K | HumanEval | Avg | Keep? | Source |
|---|--------|-------|----|----|------|-------|------|------|--------|-------|-----------|-----|-------|--------|
| 1 | exp_01 | 200 | 4 | 1e-4 | 32 | 7473 | 5000 | 5000 | 18.5% | 24.0% | 40.0% | 27.5% | Discard | multiple |
| 2 | exp_02 | 500 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 26.4% | 30.0% | 44.0% | 33.5% | Keep | multiple |
| 3 | exp_0416_0049 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 26.8% | 30.0% | 43.0% | 33.3% | Discard | cleanup |
| 4 | exp_0416_0144 | 500 | 4 | 3e-4 | 32 | 7473 | 10000 | 5000 | 21.0% | 24.0% | 39.0% | 28.0% | Discard | gJhPD |
| 5 | exp_0416_0146 | 500 | 4 | 3e-4 | 32 | 7473 | 10000 | 5000 | 28.3% | 24.0% | 43.0% | 31.8% | Discard | cleanup |
| 6 | exp_0416_0211 | 500 | 8 | 1e-4 | 32 | 7473 | 10000 | 5000 | 17.0% | 31.0% | 44.0% | 30.7% | Discard | gJhPD |
| 7 | exp_0416_0249 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 18.0% | 32.0% | 45.0% | 31.7% | Discard | gJhPD |
| 8 | exp_0416_0250 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 30.1% | 31.0% | 45.0% | 35.4% | Keep | bkJB0 |
| 9 | exp_0416_0453 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 27.1% | 36.0% | 42.0% | 35.0% | Keep | M3Tyj |
| 10 | exp_0416_0608 | 500 (resume from #9) | 4 | 3e-4 | 32 | 7473 | 10000 | 5000 | 27.9% | 30.0% | 39.0% | 32.3% | Discard | M3Tyj |
| 11 | exp_0416_0651 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 17.0% | 29.0% | 47.0% | 31.0% | Discard | gJhPD |
| 12 | exp_0416_0755 | 500 | 4 | 1e-4 | 32 | 7473 | 20000 | 5000 | 20.0% | 30.0% | 45.0% | 31.7% | Discard | gJhPD |
| 13 | exp_0416_0850 | 1000 | 4 | 1e-4 | 32 | 7473 | 10000 | 5000 | 19.0% | 38.0% | 44.0% | 33.7% | Discard | 7eIgD |
| 14 | exp_0416_0948 | 500 | 4 | 1e-4 | 32 | 7473 | 20000 | 5000 | 19.0% | 31.0% | 34.0% | 28.0% | Discard | 7eIgD |
| 15 | **exp_0416_1145** | **1000** | **4** | **1e-4** | **32** | **7473** | **10000** | **5000** | **27.4%** | **35.0%** | **46.0%** | **36.1%** | **BEST** | gJhPD |

### Experiment Details

**exp_01** — Baseline SFT with real task data. HumanEval strong (40%>30% target), but IFEval and GSM8K severely underperform. Only 200 steps = 800 examples seen out of 17k. Loss went 0.67→0.26.

**exp_02** — All 3 tasks improved: IFEval +7.9pp, GSM8K +6pp, HumanEval +4pp. Total 22k samples, 500 steps. Loss fluctuated 1.38→1.26 (high variance from diverse data). checkpoint: `tinker://e2cf73b2-066a-500a-bf73-6583069a26f0:train:0/sampler_weights/exp_02_gsm7k_tulu10k_code5k_lr1e4_steps500_rank32`

**exp_0416_0049** — No meaningful improvement from 2x steps. IFEval +0.4pp, GSM8K flat, HumanEval -1pp. Model appears plateaued at this LR/data mix. Loss was highly variable (0.15-2.06).

**exp_0416_0144** — All metrics regressed vs exp_02. Loss was very volatile (0.28-1.51), confirming lr=3e-4 is too high for this setup. save_state: `tinker://d65f3a29-e88b-5fbe-8312-7dd238df64ea:train:0/weights/exp_0416_0144_lr3e4_steps500_rank32`

**exp_0416_0146** — Higher LR significantly hurt GSM8K (-6pp vs exp_02) while only marginally helping IFEval (+1.9pp). HumanEval unchanged. Model loses math reasoning at higher LR.

**exp_0416_0211** — IFEval regressed -9.4pp vs exp_02 while GSM8K +1pp and HumanEval flat. Larger batch with same LR reduces effective per-sample learning rate (linear scaling rule). save_state: `tinker://804706d3-d970-5433-8917-c147f25969c3:train:0/weights/exp_0416_0211_bs8_lr1e4_steps500_rank32`

**exp_0416_0249** — IFEval regressed -8.4pp vs exp_02. More steps helps math/code but hurts IFEval — suggests need for data ratio rebalancing. save_state: `tinker://9f8a3aa8-25e0-54ce-8eed-dbdcac2d1762:train:0/weights/exp_0416_0249_bs4_lr1e4_steps1000_rank32`

**exp_0416_0250** — All 3 tasks improved: IFEval +3.7pp, GSM8K +1pp, HumanEval +1pp. Diminishing returns on more steps. save_state: `tinker://0d297950-1e79-5ddb-ac98-7d8c89544baf:train:0/weights/exp_0416_0250_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`

**exp_0416_0453** — GSM8K improved significantly (+6pp). IFEval +0.7pp. HumanEval -2pp (within noise). Avg +1.5pp. save_state: `tinker://2e5660f8-a11a-52bf-ae48-2d3c013de40b:train:0/weights/exp_0416_0453_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`

**exp_0416_0608** — Resume from exp_0416_0453 + higher LR (3e-4). Higher LR destabilized training. GSM8K regressed -6pp, HumanEval -3pp. Conclusion: 3e-4 LR is too aggressive when resuming — use ≤1e-4. save_state: `tinker://61c68df4-68b5-5b28-bd5a-18baf1645a4c:train:0/weights/exp_0416_0608_resume1000_lr3e4_steps500`

**exp_0416_0651** — IFEval catastrophically regressed (-9.4pp). 1000 steps causes overtraining — the model forgets instruction following. Loss oscillated 0.15-2.07. checkpoint: `tinker://666c2755-35d3-5890-afe3-2269b543c4b5:train:0/sampler_weights/exp_0416_0651_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`

**exp_0416_0755** — 20k Tulu did NOT help IFEval (-6.4pp vs exp_02). More Tulu data shifted proportions (62% Tulu vs 45% before) which diluted math/code signal. checkpoint: `tinker://670aed52-a66d-5cc8-b981-d91529de9c36:train:0/sampler_weights/exp_0416_0755_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32`

**exp_0416_0850** — GSM8K improved +8pp but IFEval degraded -7.4pp. More training helps math but hurts instruction following. checkpoint: `tinker://2f4a877c-570b-5ebb-91b7-dc16c8382081:train:0/sampler_weights/exp_0416_0850_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`

**exp_0416_0948** — CATASTROPHIC: HumanEval -10pp (44→34%), IFEval unchanged at 19%, GSM8K +1pp. 20k Tulu diluted code/math signal too much. checkpoint: `tinker://f86cf27a-f894-585a-b885-1ce24ed68480:train:0/sampler_weights/exp_0416_0948_gsm7k_tulu20k_code5k_lr1e4_steps500_rank32`

**exp_0416_1145** ★ BEST — All 3 tasks improved: IFEval +1.0pp, GSM8K +5.0pp, HumanEval +2.0pp. Same data mix (7473 GSM8K, 10k Tulu, 5k Code), lr=1e-4, rank=32. Loss 1.38→1.23. checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` state: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state`

## Analysis

### Key findings (consolidated from all 15 experiments):

1. **Best config: 1000 steps, bs=4, lr=1e-4, rank=32, 7473 GSM8K / 10k Tulu / 5k Code** → avg 36.1% (exp_0416_1145)
2. **1000 steps has HIGH VARIANCE across runs**: Same config yielded avg from 31.0% to 36.1% across 6 different runs (#3,7,8,9,11,15). Training randomness matters significantly.
3. **HumanEval is easiest to improve**: 40-47% across runs (target: 30%) — already well above target
4. **GSM8K responds well to more training**: 24%→30%→35% trend, but high variance (29-38% at 1000 steps)
5. **IFEval is the bottleneck and very fragile**: Best 27.4%, target 45%. Sensitive to LR, batch size, data ratios, and even random seed.
6. **lr=3e-4 is too high**: Confirmed in 3 separate experiments (#4,5,10). Causes instability and hurts GSM8K.
7. **batch_size=8 hurts IFEval**: -9.4pp vs bs=4 (#6). Linear scaling rule needed.
8. **20k Tulu data is harmful**: Confirmed in 2 experiments (#12,14). Dilutes math/code signal without helping IFEval. 10k is the sweet spot.
9. **Resuming with high LR is destructive**: Exp #10 showed 3e-4 LR when resuming causes GSM8K regression. Use ≤1e-4.
10. **Diminishing returns from steps alone**: 200→500 gave big gains; 500→1000 helps but with high variance and risk of IFEval regression.

### Failed approaches (DO NOT REPEAT):
- lr=3e-4 (too high, confirmed 3x)
- batch_size=8 without LR scaling (IFEval drops ~9pp)
- 20k Tulu data (dilutes code/math, confirmed 2x)
- Resuming from checkpoint with lr=3e-4 (destabilizes)

### Best checkpoint:
- **exp_0416_1145**: IFEval 27.4%, GSM8K 35.0%, HumanEval 46.0%, Avg 36.1%
- Checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`
- State: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state`

### Suggested next experiments:
- **Two-stage training**: Stage 1 broad SFT (500 steps), Stage 2 IFEval-focused (200 steps with Tulu-only)
- **Resume from best with lower LR (5e-5)**: Conservative fine-tuning for 200-500 more steps
- **LoRA rank 64**: Current rank 32 may be capacity bottleneck
- **Filtered Tulu data**: Use high-quality instruction-following subset instead of random streaming
- **Scale to 8B model**: Once 3B config optimized, try Llama-3.1-8B
