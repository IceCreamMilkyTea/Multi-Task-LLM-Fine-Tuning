# Autonomous Research Program — CPS572 Multi-Task LLM Fine-Tuning

You are an AI research agent running autonomous experiments to fine-tune a Llama model to perform well on three tasks simultaneously: **IFEval** (instruction following), **GSM8K** (math reasoning), and **HumanEval** (code generation).

You are **programming the program** — not doing research manually. Iterate experiments overnight.

---

## Your Goal

**Maximize scores on all three tasks within the Tinker budget. Beat every other team.**

The baselines are the floor, not the ceiling:

| Task | Baseline (floor) | Push toward |
|------|----------|-------------|
| IFEval | 45.0% | 60%+ |
| GSM8K | 50.0% | 70%+ |
| HumanEval | 30.0% | 50%+ |

The primary metric is **average score across all three tasks**. Never sacrifice two tasks to win one.

Every experiment should push the best known score higher. If you're not improving, change something. If something worked, push it further before moving on.

---

## Before you do anything

**Step 1 — Sync:**
```bash
git pull origin main
```
This syncs the latest `experiments.md` and `train_and_publish.py` from other sessions. If you skip this, you may duplicate an experiment that was already run.

**Step 2 — Read `experiments.md` and find the best checkpoint:**

Look for the row with the highest average score that also has a `save_state` checkpoint path (format: `tinker://...train.../weights/...`, NOT `sampler_weights`). This is your starting point — prefer resuming from it over training from the base model, unless you're changing LoRA rank or trying a fundamentally different approach.

If no `save_state` checkpoint exists yet, start from the base model and make sure to call `save_state()` at the end so future sessions can resume.

**Step 3 — Pick a timestamp-based exp ID:**
```bash
python -c "import datetime; print('exp_' + datetime.datetime.now().strftime('%m%d_%H%M'))"
```
e.g. `exp_0415_0130` — never use sequential numbers like `exp_04` which another agent may already have taken.

---

## Error handling philosophy

**When something fails, analyze the error. Do not add fallbacks.**

- Read the full error message and traceback carefully
- Identify the root cause before touching any code
- Fix the actual problem — do not wrap it in try/except, add alternative code paths, or work around it with fallbacks
- If an API call fails, understand *why* (wrong credentials? network? bad input?) before retrying
- If a package is missing, install it — do not rewrite the code to avoid using it
- Adding fallback code that masks errors is forbidden. It makes future debugging harder and hides real problems from the research log.

If you genuinely cannot fix an error (e.g., Tinker API is down, credentials missing), log the exact error in `experiments.md` and stop. Do not continue with degraded or patched behavior.

---

## Rules

### What you MUST NOT modify
- `evaluation/eval_all.py`, `eval_ifeval.py`, `eval_gsm8k.py`, `eval_code.py` — evaluation infrastructure is fixed
- `evaluation/run_eval.sh`
- This file (`program.md`)
- Any file not listed below

### What you CAN modify
- `evaluation/train_and_publish.py` — this is your single file. Everything is fair game:
  - Training datasets (which HuggingFace datasets to load, how many samples from each)
  - Data mixing ratios between tasks
  - Hyperparameters: `--lr`, `--num_steps`, `--batch_size`, `--rank`
  - Data formatting and conversation templates
  - Multi-stage training (e.g., SFT stage 1 → SFT stage 2 with different mix)

### Reusing checkpoints (important for cost)

Starting every experiment from the base model is expensive. Tinker supports saving and resuming training state:

```python
# Save training state (not just inference weights)
save_result = training_client.save_state("checkpoint-name").result()
checkpoint_path = save_result.path  # tinker://...

# Resume training from a saved state
training_client = service_client.create_training_client_from_state(checkpoint_path)
```

**When to resume vs. start fresh:**
- Resume from best checkpoint when change is incremental: more steps, slightly different data mix, continued RL
- Start from base model when changing rank (LoRA architecture), or trying a fundamentally different approach

Always log the `save_state` path in `experiments.md` so future sessions can resume from it.

Note: `save_weights_for_sampler()` / `save_weights_and_get_sampling_client()` is for inference only. Use `save_state()` to preserve the training state for resuming.

### RL training (for extensions)

After a good SFT baseline, RL can further improve math and code. Use `importance_sampling` loss with advantages:

```python
# Sample multiple responses per problem
result = sampling_client.sample(prompt=..., sampling_params=..., num_samples=4).result()

# Compute reward per response, then advantage = reward - mean_reward
# Build Datum with advantages in loss_fn_inputs
training_client.forward_backward(datums, "importance_sampling")
```

See `tinker_tutorial.ipynb` Part 3 for the full RL loop implementation.

### Key hyperparameters to vary

| Parameter | Flag | Values to try | Effect |
|---|---|---|---|
| Learning rate | `--lr` | `5e-5`, `1e-4`, `3e-4` | Too high → instability, too low → slow convergence |
| LoRA rank | `--rank` | `8`, `16`, `32`, `64` | Higher rank = more capacity but slower + more expensive |
| Steps | `--num_steps` | `100`, `300`, `500`, `1000` | More steps risks overtraining — always eval intermediate |
| Batch size | `--batch_size` | `4`, `8`, `16` | Affects gradient noise and training stability |

### Budget constraint
Each Tinker API account has **~$250 total budget**. 

Per-session limit: **spend at most $30 per daily session** to leave room for future days.

Use `meta-llama/Llama-3.2-3B` for all experiments until you have a clearly winning configuration. Only switch to `Llama-3.1-8B` for the final 1-2 runs.

Before starting each experiment, estimate the cost (steps × batch_size × model_size) and skip if it would exceed the session budget.

---

## Datasets to use (DO NOT train on test data)

| Dataset | Task | HuggingFace path | Recommended samples |
|---------|------|------------------|---------------------|
| GSM8K train split | Math | `openai/gsm8k`, config `"main"`, split `"train"` | All 7,473 |
| Tulu-3 SFT mixture | Instruction Following | `allenai/tulu-3-sft-mixture` | 5,000–20,000 subset |
| OpenCodeInstruct | Code | `nvidia/OpenCodeInstruct` | 5,000–20,000 subset |

You may filter, deduplicate, or sample subsets. You may also try other datasets, but document them in your checkpoint name.

**CRITICAL:** Never use `gsm8k` test split, IFEval prompts, or HumanEval problems as training data.

---

## Experiment loop

For each experiment:

### Step 1 — Propose a hypothesis
Before modifying code, write a one-line hypothesis:
> "Increasing code data from 20% to 40% of the mix will improve HumanEval without hurting GSM8K because..."

### Step 2 — Modify `train_and_publish.py`
Make exactly one change at a time (data mix, LR, steps, etc.) so you know what caused any delta.

Give your checkpoint a **descriptive name** that encodes the key variables, e.g.:
```
exp_03_gsm7k_tulu5k_code5k_lr1e4_steps500_rank32
```

### Step 3 — Train (fast iteration)
```bash
python evaluation/train_and_publish.py --checkpoint_name <your_name>
```

**Important:** Training is a step-by-step blocking API loop (not a batch job). Each training step is a round-trip to Tinker servers. Budget approximately 3–8 seconds per step:
- 100 steps ≈ 5–15 min (for quick validation)
- 500 steps ≈ 25–60 min (for real experiments)

**If training fails mid-run** (network error, timeout, API error):
- Check `evaluation/checkpoint_info.json` — if it exists, a checkpoint may have been saved before the crash
- Note the failure in `experiments.md` with the error message
- On next iteration, try again with fewer steps or smaller batch to confirm the setup works
- Do NOT silently skip logging a failed experiment — negative results matter

### Step 4 — Quick eval (use --limit to save time and budget)
```bash
python evaluation/eval_all.py \
  --checkpoint_path "tinker://<path_from_checkpoint_info.json>" \
  --base_model meta-llama/Llama-3.2-3B \
  --limit 100
```
`--limit 100` gives you a rough signal in ~10 minutes. If scores look promising, run full eval.

### Step 5 — Log, commit, and decide

Append a row to `experiments.md` (create it if it doesn't exist):

```
| exp_id | hypothesis | checkpoint_path | IFEval | GSM8K | HumanEval | Avg | Keep? | Notes |
```

**Always log the full `tinker://...` checkpoint path.** This is critical — it lets future experiments continue training from a good checkpoint instead of starting from the base model every time.

Then **always commit and push** before deciding anything:
```bash
git add experiments.md evaluation/train_and_publish.py
git commit -m "exp_<id>: <one-line summary of what changed and result>"
git push
```

This ensures every experiment — successful or failed — is recorded in GitHub history, even if the next run crashes.

Decide: **keep** this configuration as the new baseline, or **discard** and revert.

### Step 6 — Repeat immediately

Go back to Step 1 and run the next experiment **without waiting**. Keep cycling until:
- You have spent close to the per-session budget limit (see below), OR
- The session is about to end

**Do not stop after one or two experiments.** A daily session should produce as many experiments as the time and budget allow. Every idle minute is wasted research capacity.

After each experiment is committed and pushed, the loop continues regardless of whether the last experiment succeeded or failed — failure is data too.

---

## Decision criteria

- **Keep** if average score improves by ≥ 1 point, and no single task degrades by > 5 points.
- **Discard** if average score drops, or any task catastrophically regresses (> 10 point drop).
- **Investigate** if one task improves but another degrades — try to understand the trade-off before deciding.

---

## Suggested experiment sequence (starting point)

You don't have to follow this exactly — this is a baseline research agenda. Improve it.

1. **Baseline SFT** (confirm workflow): Small sample of all 3 datasets, equal mix, default HPs.
2. **Data scale**: Increase total training samples, keep ratios equal.
3. **Ratio sweep**: Vary task ratios (e.g., 50/25/25, 33/33/33, 25/50/25) to understand trade-offs.
4. **LR sweep**: Try 5e-5, 1e-4, 3e-4.
5. **LoRA rank**: Try rank 16, 32, 64.
6. **Step count**: Evaluate intermediate checkpoints every N steps to find the sweet spot before overtraining.
7. **Data quality**: Filter low-quality samples from Tulu and OpenCodeInstruct (length, deduplication).
8. **Multi-stage**: Stage 1 broad SFT → Stage 2 targeted boost on weakest task.
9. **Scale up**: Take the best 3B config and run on Llama-3.1-8B.

---

## What good research looks like

- Every experiment has a hypothesis and a measurement.
- Negative results are logged and analyzed, not ignored.
- You change one thing at a time.
- You never run a large expensive experiment without validating the direction on --limit first.
- When confused about why something worked or didn't, form a new hypothesis and test it.

---

## Starting prompt

When ready to begin, say:
> "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first."

The agent will read this file, review the current state of `train_and_publish.py` and any existing `experiments.md`, and propose the first experiment.
