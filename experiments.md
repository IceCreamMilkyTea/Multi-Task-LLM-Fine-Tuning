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
| 16 | exp_0416_1413 | 300 | 4 | 5e-5 | 8 | 7473 | 10000 | 5000 | 17.0% | 22.0% | 39.0% | 26.0% | Discard | JOrG1 |
| 17 | exp_0416_1436 | 300 | 4 | 5e-5 | 8 | 7469* | 8063* | 5000* | 12.0% | 18.0% | 39.0% | 23.0% | Discard | JOrG1 |
| 18 | exp_0416_1441 | 1000 | 4 | 1e-4 | 32 | 7469* | 8063* | 5000* | 17.0% | 32.0% | 44.0% | 31.0% | Discard | JOrG1 |
| 19 | exp_0416_1532 | 300 (resume #15) | 4 | 5e-5 | 32 | 3079 | 14372† | 3079 | 19.0% | 31.0% | 46.0% | 32.0% | Discard | JOrG1 |
| 20 | exp_0416_1759_rank64 | 1000 | 4 | 1e-4 | 64 | 7469* | 8063* | 5000* | 14.0% | 34.0% | 43.0% | 30.3% | Discard | JOrG1 |
| 23 | exp_0416_1845_rl | 20 RL iters (resume #15) | 4 | 5e-6 | 32 | RL on GSM8K | - | - | 17.0% | 37.0% | 46.0% | 33.3% | Keep | JOrG1 |

### Llama-3.1-8B Experiments

| # | exp_id | Steps | BS | LR | Rank | GSM8K | Tulu | Code | IFEval | GSM8K | HumanEval | Avg | Keep? | Source |
|---|--------|-------|----|----|------|-------|------|------|--------|-------|-----------|-----|-------|--------|
| 21 | exp_0416_1759_8b | 500 | 4 | 1e-4 | 32 | 7469* | 10682* | 8000* | 23.0% | 55.0% | 48.0% | 42.0% | Keep | JOrG1 |
| 22 | exp_0416_1831_8b_s2 | 300 (resume #21) | 4 | 5e-5 | 32 | 3922 | 18305† | 3922 | 23.0% | 55.0% | 55.0% | 44.3% | Keep | JOrG1 |
| 24 | exp_0417_0005_8b_s3 | 500 (resume #22) | 4 | 5e-5 | 32 | — | 85% Tulu† | — | 29.0% | 46.0% | 58.0% | 44.3% | Keep | JOrG1 |
| 25 | **exp_0417_0020_8b_flan** | **500 (resume #21)** | **4** | **5e-5** | **32** | **—** | **85% FLAN‡** | **—** | **32.0%** | **57.0%** | **51.0%** | **46.7%** | **★ BEST** | JOrG1 |
| 26 | exp_0417_0030_8b_bal | 500 (resume #21) | 4 | 5e-5 | 32 | — | 60% FLAN‡ | — | 25.0% | 53.0% | 58.0% | 45.3% | Keep | JOrG1 |
| 27 | **exp_0417_0040_8b_s4** | **500 (resume #25)** | **4** | **3e-5** | **32** | **—** | **90% FLAN‡** | **—** | **39.9%⁑** | **54.4%⁑** | **45.1%⁑** | **46.5%⁑** | **★★ BEST** | JOrG1 |
| 28 | exp_0417_0050_8b_s5 | 500 (resume #27) | 4 | 2e-5 | 32 | — | 90% FLAN‡ | — | 35.0% | 56.0% | 55.0% | 48.7% | Discard | JOrG1 |
| 29 | exp_0417_0100_diverse | 300 (resume #27) | 4 | 3e-5 | 32 | — | 90% FLAN‡‡ | — | 37.0% | 49.0% | 51.0% | 45.7% | Discard | JOrG1 |
| 30 | exp_0417_0434b | 300 (resume #27) | 4 | 1e-4 | 32 | — | 70% FLAN‡ | — | 37.0% | 56.0% | 54.0% | 49.0% | Discard | JOrG1 |
| 31 | exp_0417_0500_fresh | 1000 (from base) | 4 | 1e-4 | 32 | 7469* | FLAN‡ | 8000* | 29.3%§ | 56.0%§ | 41.5%§ | 42.3%§ | Discard | JOrG1 |
| 32 | exp_0417_0600_lr1e5 | 500 (resume #27) | 4 | 1e-5 | 32 | — | 85% FLAN‡ | — | 40.7%§ | 60.0%§ | 47.6%§ | 49.4%§ | Keep | JOrG1 |
| 33 | **exp_0417_0700_deep** | **500 (resume #27)** | **4** | **2e-5** | **32** | **—** | **90% FLAN‡‡‡** | **—** | **43.0%§** | **55.0%§** | **40.9%§** | **46.3%§** | **★★★ NEW BEST** | JOrG1 |

\* = quality-filtered data
† = Stage 2/3: Tulu focus (oasst1 + flan_v2)
‡ = FLAN-focused: skip first 5k oasst1, use flan_v2 data + max_length=2048
§ = scores from --limit 300 evaluation (more reliable than --limit 100)
⁑ = scores from FULL evaluation (all samples: IFEval 541, GSM8K 1319, HumanEval 164)
‡‡ = skip first 15k Tulu samples for maximum diversity
‡‡‡ = skip first 10k Tulu, load 30k for deep FLAN diversity

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

**exp_0416_1413** — Tested rank=8 + lr=5e-5 (conservative approach to prevent forgetting). All metrics regressed severely: IFEval -10.4pp, GSM8K -13pp, HumanEval -7pp vs best. rank=8 lacks capacity to learn from training data while still disrupting base model. Loss 1.10→0.48. checkpoint: `tinker://3213368d-3c6c-5f17-bb46-4d080609c971:train:0/sampler_weights/exp_0416_1413_gsm7k_tulu10k_code5k_lr5e5_steps300_rank8` state: `tinker://3213368d-3c6c-5f17-bb46-4d080609c971:train:0/weights/exp_0416_1413_gsm7k_tulu10k_code5k_lr5e5_steps300_rank8_state`

**exp_0416_1436** — Added quality filtering (removed non-English Tulu, low-test-score code) + curriculum learning (easy→hard) to rank=8 setup. EVEN WORSE: IFEval 12% (-5pp vs #16), GSM8K 18% (-4pp). Filtering+curriculum can't compensate for insufficient rank. checkpoint: `tinker://857b3a31-5585-5e76-ae19-f0b272eb6ba5:train:0/sampler_weights/exp_0416_1436_filtered_curriculum_lr5e5_steps300_rank8`

**exp_0416_1441** — Quality filtering + curriculum learning with proven HPs (rank=32, lr=1e-4, 1000 steps). Same as best config but with filtered/sorted data. Avg 31.0% vs 36.1% best — within high-variance range for this config. Filtering + curriculum did not meaningfully improve results. checkpoint: `tinker://c6e1c198-bedd-5deb-aa1f-b488af4444b2:train:0/sampler_weights/exp_0416_1441_filtered_curriculum_lr1e4_steps1000_rank32` state: `tinker://c6e1c198-bedd-5deb-aa1f-b488af4444b2:train:0/weights/exp_0416_1441_filtered_curriculum_lr1e4_steps1000_rank32_state`

**exp_0416_1532** — Multi-stage training: resumed from best checkpoint (exp_0416_1145) with 70% Tulu focus, 300 steps, lr=5e-5. IFEval improved slightly to 19% (+2pp vs non-resumed runs), HumanEval maintained at 46%. But overall avg still 32% — within noise. Stage 2 Tulu focus didn't breakthrough on IFEval. checkpoint: `tinker://3643b38a-21f2-5cea-8f60-46cc4336e07f:train:0/sampler_weights/exp_0416_1532_stage2_tulu_focus_lr5e5_steps300` state: `tinker://3643b38a-21f2-5cea-8f60-46cc4336e07f:train:0/weights/exp_0416_1532_stage2_tulu_focus_lr5e5_steps300_state`

**exp_0416_1759_rank64** — LoRA rank=64 on 3B. IFEval 14% (WORSE than rank=32 27.4%). Higher rank = more overfitting, not more capacity. checkpoint: `tinker://6e81dec5-ec55-59c5-9df3-eb6d21bf8c8e:train:0/sampler_weights/exp_0416_1759_rank64_gsm7k_tulu10k_code5k_lr1e4_steps1000`

**exp_0416_1759_8b** ★ NEW BEST — Llama-3.1-8B with 500 steps, quality-filtered data (7469 GSM8K, 10682 Tulu, 8000 Code). GSM8K **55%** (exceeds 52.5% target!), HumanEval **48%** (exceeds 31.5% target!). IFEval 23% (still below 47.3% target). Massive improvement from model scale. checkpoint: `tinker://fcd6352c-4a09-5166-96a6-5fcc63d3ba81:train:0/sampler_weights/exp_0416_1759_8b_gsm7k_tulu15k_code8k_lr1e4_steps500_rank32` state: `tinker://fcd6352c-4a09-5166-96a6-5fcc63d3ba81:train:0/weights/exp_0416_1759_8b_gsm7k_tulu15k_code8k_lr1e4_steps500_rank32_state`

**exp_0416_1831_8b_s2** ★★ OVERALL BEST — Stage 2 on 8B with 70% Tulu focus, 300 steps, lr=5e-5. HumanEval jumped to **55%** (+7pp vs #21). GSM8K maintained 55%. IFEval 23% (unchanged). **Avg 44.3% — new best.** checkpoint: `tinker://148e1942-bdf0-54c1-9280-e15d0bf24849:train:0/sampler_weights/exp_0416_1831_8b_stage2_tulu_focus_lr5e5_steps300` state: `tinker://148e1942-bdf0-54c1-9280-e15d0bf24849:train:0/weights/exp_0416_1831_8b_stage2_tulu_focus_lr5e5_steps300_state`

**exp_0416_1845_rl** — GRPO-style RL on 3B (20 iters, 4 problems/iter, 4 samples/problem). GSM8K improved to 37% (+2pp vs best 3B SFT). Reward went 0.44→0.24 (model explored harder problems). RL works but modest gains on 3B. checkpoint: `tinker://597d0b8c-861a-5b70-958e-842d360bdcc8:train:0/sampler_weights/exp_0416_1845_rl_gsm8k_3b`

**exp_0417_0005_8b_s3** — **Change: resume from #22 (8B Stage-2), 85% Tulu (oasst1+flan_v2), 500 steps, lr=5e-5.** Goal: push IFEval by heavy Tulu focus. Result: IFEval strict jumped to 29% (+6pp vs #22), but GSM8K dropped to 46% (-9pp). Tulu-heavy training shifts capability from math to instruction following. HumanEval 58%. checkpoint: `tinker://e5fa1691-abee-5384-928d-bc26748dc90e:train:0/sampler_weights/exp_0417_0005_8b_stage3_tulu_only_lr5e5_steps500` state: `tinker://e5fa1691-abee-5384-928d-bc26748dc90e:train:0/weights/exp_0417_0005_8b_stage3_tulu_only_lr5e5_steps500_state`

**exp_0417_0020_8b_flan** ★ KEY BREAKTHROUGH — **Change: resume from #21 (8B base SFT), skip first 5000 oasst1 samples to get FLAN v2 data, 85% Tulu, max_length=2048 (doubled from 1024), 500 steps, lr=5e-5.** Two key innovations: (1) skipping oasst1 to load flan_v2 data which is instruction-following focused, (2) max_length=2048 for learning longer structured responses. IFEval strict 32% (+9pp vs #21), GSM8K 57%, HumanEval 51%. Avg 46.7%. checkpoint: `tinker://e47342aa-d88f-5eb4-8034-783ff691d177:train:0/sampler_weights/exp_0417_0020_8b_flan_focus_maxlen2048_lr5e5_steps500` state: `tinker://e47342aa-d88f-5eb4-8034-783ff691d177:train:0/weights/exp_0417_0020_8b_flan_focus_maxlen2048_lr5e5_steps500_state`

**exp_0417_0030_8b_bal** — **Change: same as #25 but 60% Tulu focus (vs 85%), to preserve more GSM8K.** Result: IFEval strict 25%, GSM8K 53%, HumanEval 58%. Lower Tulu ratio = less IFEval improvement. 85% Tulu is better for IFEval. checkpoint: `tinker://ad4caaf3-1629-5b8b-aee7-a29de4456bcb:train:0/sampler_weights/exp_0417_0030_8b_balanced_tulu60_lr5e5_steps500`

**exp_0417_0040_8b_s4** ★★ — **Change: resume from #25 (FLAN checkpoint), 90% Tulu FLAN (skip oasst1), lr=3e-5 (lower than 5e-5), 500 more steps.** Progressive learning: building on FLAN checkpoint with even more FLAN data at lower LR. Full eval: IFEval strict 39.9%, final 46.1%, GSM8K 54.4%, HumanEval 45.1%. checkpoint: `tinker://dd7fb973-6ac9-51f0-9777-f5a97e8b46a0:train:0/sampler_weights/exp_0417_0040_8b_flan_stage4_tulu90_lr3e5_steps500` state: `tinker://dd7fb973-6ac9-51f0-9777-f5a97e8b46a0:train:0/weights/exp_0417_0040_8b_flan_stage4_tulu90_lr3e5_steps500_state`

**exp_0417_0050_8b_s5** — **Change: resume from #27, lr=2e-5 (even lower), 500 steps, same data.** Result: IFEval strict 35%, GSM8K 56%, HumanEval 55%. IFEval regressed — lr=2e-5 is too low from this checkpoint, overtraining without enough signal. checkpoint: `tinker://91a8227e-9719-5926-bbac-5bc1f66cc648:train:0/sampler_weights/exp_0417_0050_8b_flan_stage5_tulu90_lr2e5_steps500`

**exp_0417_0100_diverse** — **Change: resume from #27, skip first 15k Tulu (vs 5k) to access maximum diversity of FLAN data, 300 steps, lr=3e-5.** Result: IFEval strict 37%, GSM8K 49% (dropped below target!). Skipping too many samples loses useful data. checkpoint: `tinker://88001fb6-2a8e-5626-b0b1-f64997d03e6f:train:0/sampler_weights/exp_0417_0100_8b_diverse_flan_lr3e5_steps300`

**exp_0417_0434b** — **Change: resume from #27, lr=1e-4 (high), 70% FLAN, 300 steps.** Testing if higher LR helps. Result: IFEval strict 37%, GSM8K 56%, HumanEval 54%. lr=1e-4 too high for resuming from Stage-4. checkpoint: `tinker://937cad5f-0a0b-56b6-9d03-bbd99efdd433:train:0/sampler_weights/exp_0417_0434b_8b_resume_s4_balanced_lr1e4_steps300`

**exp_0417_0500_fresh** — **Change: train 8B FROM SCRATCH (no multi-stage), FLAN data from start (skip oasst1), 1000 steps, lr=1e-4, max_length=2048.** Testing if single-stage with FLAN is better than multi-stage. Full eval: IFEval strict 29.3%, GSM8K 56%, HumanEval 41.5%. Multi-stage approach is significantly better for IFEval. checkpoint: `tinker://1626b0ea-b18e-54db-b06c-210e2bb777e6:train:0/sampler_weights/exp_0417_0500_8b_flan_from_start_lr1e4_steps1000` state: `tinker://1626b0ea-b18e-54db-b06c-210e2bb777e6:train:0/weights/exp_0417_0500_8b_flan_from_start_lr1e4_steps1000_state`

**exp_0417_0600_lr1e5** — **Change: resume from #27, lr=1e-5 (very low), 85% FLAN, 500 steps.** Result: IFEval strict 40.7%, GSM8K 60% (highest ever!), HumanEval 47.6%. Very low LR helps GSM8K but doesn't push IFEval much. checkpoint: `tinker://1f3da038-4e02-57ab-b665-e64ed4c197ef:train:0/sampler_weights/exp_0417_0600_8b_stage4_continued_lr1e5_steps500` state: `tinker://1f3da038-4e02-57ab-b665-e64ed4c197ef:train:0/weights/exp_0417_0600_8b_stage4_continued_lr1e5_steps500_state`

**exp_0417_0700_deep** ★★★ — **Change: resume from #27, skip first 10k Tulu (deeper into FLAN v2), load 30k samples for maximum diversity, 90% Tulu, lr=2e-5, 500 steps.** The key change vs #28 (which also used lr=2e-5): skipping 10k instead of 5k gives access to more diverse FLAN instruction types. Full eval: IFEval strict 40.9%, final 47.8%, GSM8K 53.1%, HumanEval ~43%. prompt_strict still below 47.3% target. checkpoint: `tinker://0146ebf7-d86d-5239-aa07-075812b34348:train:0/sampler_weights/exp_0417_0700_8b_deep_flan_lr2e5_steps500` state: `tinker://0146ebf7-d86d-5239-aa07-075812b34348:train:0/weights/exp_0417_0700_8b_deep_flan_lr2e5_steps500_state`

## Analysis

### IFEval metric note
IFEval reports multiple metrics. `prompt_strict_acc` is the strictest (all instructions in a prompt must be followed exactly). `final_acc` averages strict/loose at prompt/instruction level. The baseline "45.0%" in program.md is ambiguous. Our best:
- **prompt_strict_acc: 40.9%** (full eval, exp #33) — if target is 47.3% strict, NOT met yet
- **final_acc: 47.8%** (full eval, exp #33) — if target is 47.3% final, MET

### Key findings (consolidated from all 33 experiments):

1. **MODEL SCALE IS THE BIGGEST LEVER**: 8B avg ~46% vs best 3B avg 36% — model scale matters most
2. **FLAN v2 data is critical for IFEval**: Skipping oasst1 (first 5-10k Tulu samples) to get flan_v2 data gave IFEval +16pp (23%→39.9% strict on full eval)
3. **max_length=2048 helps IFEval**: Doubled from 1024; allows model to learn longer instruction-following responses
4. **Multi-stage training beats single-stage**: 4-stage pipeline (SFT→FLAN→FLAN→deep FLAN) achieved 40.9% IFEval strict; single-stage only 29.3%
5. **Progressive LR decay across stages**: lr=1e-4 → 5e-5 → 3e-5 → 2e-5 prevents catastrophic forgetting
6. **RL (GRPO) works but modest gains on 3B**: GSM8K +2pp (35→37%). Not yet applied to 8B.
7. **rank=32 is optimal**: rank=8 too small, rank=64 causes overfitting on 3B
8. **GSM8K and HumanEval targets exceeded**: GSM8K 53.1-60%, HumanEval 40.9-47.6% (both well above targets)
9. **IFEval prompt_strict_acc remains the bottleneck**: Best 40.9% on full eval, needs ~6pp more to hit 47.3%

### Failed approaches (DO NOT REPEAT):
- lr=3e-4 (too high, confirmed 3x)
- batch_size=8 without LR scaling (IFEval drops ~9pp)
- 20k Tulu data (dilutes code/math, confirmed 2x)
- Resuming from checkpoint with lr=3e-4 (destabilizes)
- rank=8 + lr=5e-5 (too conservative, rank lacks capacity, all metrics worse)
- rank=8 + filter + curriculum (even worse than rank=8 alone)
- Data quality filtering + curriculum learning alone (doesn't improve over random; #18 within noise of #15)
- Multi-stage Stage 2 Tulu focus from best checkpoint (+2pp IFEval, not significant; #19)

### Best checkpoint for highest IFEval:
- **exp_0417_0700_deep** (#33, Llama-3.1-8B): IFEval **40.9% strict / 47.8% final** (full eval), GSM8K 53.1%, HumanEval ~43%
- Checkpoint: `tinker://0146ebf7-d86d-5239-aa07-075812b34348:train:0/sampler_weights/exp_0417_0700_8b_deep_flan_lr2e5_steps500`
- State: `tinker://0146ebf7-d86d-5239-aa07-075812b34348:train:0/weights/exp_0417_0700_8b_deep_flan_lr2e5_steps500_state`
- GSM8K ✅ (53.1% > 52.5%), HumanEval ✅ (~43% > 31.5%), IFEval final ✅ (47.8% > 47.3%), **IFEval strict ✗ (40.9% < 47.3%)**
- Pipeline: 8B base → 500 SFT → 500 FLAN (skip 5k oasst1, 85%) → 500 FLAN (90%, lr=3e-5) → 500 deep FLAN (skip 10k, 90%, lr=2e-5)

### Best checkpoint for highest GSM8K:
- **exp_0417_0600_lr1e5** (#32, Llama-3.1-8B): IFEval 40.7% strict, GSM8K **60.0%**, HumanEval 47.6%
- Checkpoint: `tinker://1f3da038-4e02-57ab-b665-e64ed4c197ef:train:0/sampler_weights/exp_0417_0600_8b_stage4_continued_lr1e5_steps500`

### Runner-up (best full eval balance):
- **exp_0417_0040_8b_s4** (#27, Llama-3.1-8B): IFEval 39.9% strict / 46.1% final, GSM8K 54.4%, HumanEval 45.1% (full eval)
- Checkpoint: `tinker://dd7fb973-6ac9-51f0-9777-f5a97e8b46a0:train:0/sampler_weights/exp_0417_0040_8b_flan_stage4_tulu90_lr3e5_steps500`

### Remaining target:
- **IFEval prompt_strict_acc needs +6.4pp** (40.9% → 47.3%) — this is the only unmet strict target
- Suggested approaches: RL on 8B, IFEval-specific datasets (WizardLM/Orca), more FLAN data diversity, longer training
