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
| exp_0416_0144 | Tripling LR from 1e-4 to 3e-4 will improve convergence speed at 500 steps, helping IFEval and GSM8K | 21.0% | 24.0% | 39.0% | 28.0% | Discard | All metrics regressed vs exp_02. Loss was very volatile (0.28-1.51), confirming lr=3e-4 is too high for this setup. save_state: `tinker://d65f3a29-e88b-5fbe-8312-7dd238df64ea:train:0/weights/exp_0416_0144_lr3e4_steps500_rank32`. sampler: `tinker://d65f3a29-e88b-5fbe-8312-7dd238df64ea:train:0/sampler_weights/exp_0416_0144_lr3e4_steps500_rank32` |
| exp_0416_0211 | Doubling batch_size from 4 to 8 will improve training by 2x data coverage per step (4000 vs 2000 samples seen) | 17.0% | 31.0% | 44.0% | 30.7% | Discard | IFEval regressed -9.4pp vs exp_02 while GSM8K +1pp and HumanEval flat. Larger batch with same LR reduces effective per-sample learning rate (linear scaling rule). Loss stable (1.03→0.82) but less impactful per step. save_state: `tinker://804706d3-d970-5433-8917-c147f25969c3:train:0/weights/exp_0416_0211_bs8_lr1e4_steps500_rank32`. sampler: `tinker://804706d3-d970-5433-8917-c147f25969c3:train:0/sampler_weights/exp_0416_0211_bs8_lr1e4_steps500_rank32` |
| exp_0416_0249 | Doubling steps from 500 to 1000 (bs=4, lr=1e-4) will improve all metrics since data coverage doubles from 9% to 18% | 18.0% | 32.0% | 45.0% | 31.7% | Discard | IFEval regressed -8.4pp vs exp_02 — likely overtraining on code/math hurts instruction following. GSM8K +2pp and HumanEval +1pp slight gains. More steps helps math/code but hurts IFEval — suggests need for data ratio rebalancing, not just more steps. save_state: `tinker://9f8a3aa8-25e0-54ce-8eed-dbdcac2d1762:train:0/weights/exp_0416_0249_bs4_lr1e4_steps1000_rank32`. sampler: `tinker://9f8a3aa8-25e0-54ce-8eed-dbdcac2d1762:train:0/sampler_weights/exp_0416_0249_bs4_lr1e4_steps1000_rank32` |

## Analysis

### Key findings so far:
1. **200→500 steps improved all metrics**, but 500→1000 steps hurts IFEval while helping GSM8K/HumanEval
2. **HumanEval is easiest to improve**: 44-45% already well above 30% target
3. **IFEval is very fragile**: Every experiment except exp_02 scored worse than exp_02's 26.4%. More training, higher LR, larger batch all hurt IFEval.
4. **lr=3e-4 is too high**: Causes loss instability and hurts all metrics
5. **batch_size=8 hurts IFEval**: Linear scaling rule needed
6. **Overtraining on code/math degrades IFEval**: 1000 steps improved GSM8K/HumanEval but hurt IFEval — data ratio rebalancing is key
7. **Best config remains exp_02**: batch_size=4, lr=1e-4, 500 steps, avg 33.5%

### Suggested next experiments:
- **Increase Tulu ratio**: Try 20k Tulu / 5k GSM8K / 3k Code to boost IFEval while maintaining other metrics
- **Two-stage training**: Stage 1 broad SFT (500 steps), Stage 2 IFEval-focused (200 steps with mostly Tulu data)
- Try higher LoRA rank (64) for more model capacity
- Once a good 3B config is found, scale to 8B model
