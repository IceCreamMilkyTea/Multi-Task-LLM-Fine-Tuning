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
| exp_0416_0049 | Doubling steps from 500 to 1000 (same data: 7473 GSM8K + 10k Tulu + 5k Code, lr=1e-4, rank=32) will continue improving all three tasks since model only saw ~9% of data at 500 steps | 26.8% | 30.0% | 43.0% | 33.3% | Discard | No meaningful improvement from 2x steps. IFEval +0.4pp, GSM8K flat, HumanEval -1pp. Model appears plateaued at this LR/data mix. Loss was highly variable (0.15-2.06) throughout 1000 steps. Need a different lever — LR, data mix, or rank. |

## Analysis

### Key findings so far:
1. **More training steps = better up to a point**: 200→500 steps improved all metrics, but 500→1000 steps gave zero improvement
2. **HumanEval is easiest to improve**: Already exceeds 8B baseline target (43-44% vs 30% target) on the 3B model
3. **IFEval needs more work**: 26.8% vs 45% target — still a large gap. More steps alone don't help
4. **GSM8K also lags**: 30% vs 50% target — plateaued at 30% despite 2x training
5. **Plateau at 500 steps**: Model has converged for current LR=1e-4 config, need to change something else

### Suggested next experiments:
- Try higher LR (3e-4) with 500 steps to converge faster — may break through plateau
- Increase Tulu samples to 15-20k to boost IFEval further
- Try LoRA rank 64 for more model capacity
- Once a good 3B config is found, scale to 8B model
