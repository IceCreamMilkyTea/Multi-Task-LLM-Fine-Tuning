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

## Analysis

### Key findings so far:
1. **More training steps = better**: 200→500 steps improved all metrics significantly
2. **HumanEval is easiest to improve**: Already exceeds 8B baseline target (44% vs 30% target) on the 3B model
3. **IFEval needs more work**: 26.4% vs 45% target — still a large gap. More Tulu data helped but not enough
4. **GSM8K also lags**: 30% vs 50% target — need more training or better data strategy

### Suggested next experiments:
- exp_03: Increase steps to 1000 with same data mix to test if more training continues to help
- exp_04: Try higher LR (3e-4) with 500 steps to converge faster
- exp_05: Increase Tulu samples to 15-20k to boost IFEval further
- exp_06: Once a good 3B config is found, scale to 8B model
