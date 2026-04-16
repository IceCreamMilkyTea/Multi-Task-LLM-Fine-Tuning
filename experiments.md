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
| exp_02 | Increasing steps to 500 and Tulu-3 data to 10k will improve IFEval and GSM8K because more training = more data coverage | 26.4% | 30.0% | 44.0% | 33.5% | Keep | All 3 tasks improved: IFEval +7.9pp, GSM8K +6pp, HumanEval +4pp. Total 22k samples, 500 steps. Loss fluctuated 1.38→1.26 (high variance from diverse data). Direction is strongly positive — more steps and more instruction data helps. Next: try even more steps or higher LR. |
| exp_0416_1145 | Increasing steps from 500→1000 will continue positive trend since only ~9% of data seen per epoch at 500 steps | 27.4% | 35.0% | 46.0% | 36.1% | Keep | All 3 tasks improved: IFEval +1.0pp, GSM8K +5.0pp, HumanEval +2.0pp. Same data mix (7473 GSM8K, 10k Tulu, 5k Code), lr=1e-4, rank=32. Loss 1.38→1.23 (high variance). Also added save_state() for future resumability. checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32` state: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state` |

## Analysis

### Key findings so far:
1. **More training steps = better**: 200→500→1000 steps consistently improved all metrics
2. **HumanEval is easiest to improve**: 46% at 1000 steps (target: 30%) - already well above target
3. **GSM8K improving steadily**: 24%→30%→35% across experiments, but still below 50% target
4. **IFEval improving slowly**: 18.5%→26.4%→27.4% — needs a different approach beyond just more steps
5. **Diminishing returns on steps for IFEval**: 200→500 gave +7.9pp, but 500→1000 only +1.0pp
6. **GSM8K benefits most from more steps**: +5.0pp from 500→1000 steps

### Best checkpoint so far:
- **exp_0416_1145**: IFEval 27.4%, GSM8K 35.0%, HumanEval 46.0%, Avg 36.1%
- Checkpoint: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/sampler_weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32`
- State: `tinker://1f8374c0-bf66-58cc-9c8f-b14d793d9915:train:0/weights/exp_0416_1145_gsm7k_tulu10k_code5k_lr1e4_steps1000_rank32_state`

### Suggested next experiments:
- Try increasing Tulu samples to 15-20k to boost IFEval (since more steps alone has diminishing returns for IFEval)
- Try higher LR (3e-4) — could speed convergence
- Try resuming from checkpoint with more steps focused on instruction-following data
- Once a good 3B config is found, scale to 8B model
