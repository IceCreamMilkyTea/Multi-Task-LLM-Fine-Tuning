"""Strong GRPO trainer for Tinker — IFEval / GSM8K / HumanEval.

Improvements over evaluation/rl_ifeval.py and evaluation/rl_train.py:

  1. Tinker built-in PPO/CISPO loss with DAPO-style asymmetric clipping
     (loss_fn="ppo" instead of "importance_sampling"; gives a real trust
     region rather than an unclipped IS estimator).
  2. Reference-policy KL regularization, injected into per-token advantages
     using the cookbook's `incorporate_kl_penalty` recipe (no separate
     forward pass needed at training time).
  3. Multiple PPO epochs over each rollout batch (forward_backward + step
     replayed several times on the same data, the way GRPO/DAPO does).
  4. DAPO group filtering — drop prompts where every sample got the same
     reward (zero-variance group → zero gradient anyway, just wastes compute).
  5. Token-level loss normalization (built into Tinker's "ppo" loss).
  6. Three task verifiers in one file (IFEval / GSM8K / HumanEval) wired
     to the dataset that gives the largest transfer to each eval.

Datasets (override with --dataset_name):
    ifeval     -> allenai/RLVR-IFeval                       (Tulu-3 RLVR)
    gsm8k      -> allenai/RLVR-GSM                          (numeric verifier)
    humaneval  -> PrimeIntellect/verifiable-coding-problems (unit tests)
"""

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from datasets import load_dataset

import tinker
from tinker import types
from tinker.types import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.metrics import incorporate_kl_penalty, compute_post_kl

from instruction_following_eval import instructions_registry
from instruction_following_eval.evaluation import (
    InputExample,
    ensure_nltk_resource,
    test_instruction_following,
)


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = {
    "ifeval":    "allenai/RLVR-IFeval",
    "gsm8k":     "allenai/RLVR-GSM",
    "humaneval": "PrimeIntellect/verifiable-coding-problems",
}


# ---------------------------------------------------------------------------
# Reward functions — each returns a float in [0, 1].
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.search(r"####\s*([-+]?[0-9][0-9,\.]*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def reward_gsm8k(response: str, gt: Dict[str, Any]) -> float:
    gold = str(gt.get("answer", "")).split("####")[-1].strip().replace(",", "")
    pred = _extract_number(response)
    if pred is None or gold == "":
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if pred == gold else 0.0


def reward_ifeval(response: str, gt: Dict[str, Any]) -> float:
    """Run the official IFEval checkers; returns fraction-satisfied + a
    +0.5 bonus when ALL constraints are met (matches rl_ifeval.py's
    prompt_strict-aligned reward shape)."""
    inst_ids: List[str] = gt.get("instruction_id_list", []) or []
    kwargs_list: List[Dict[str, Any]] = gt.get("kwargs") or [{} for _ in inst_ids]
    if not inst_ids:
        return 0.0
    try:
        ex = InputExample(key=0, instruction_id_list=inst_ids,
                          prompt="", kwargs=kwargs_list)
        out = test_instruction_following(ex, response, strict=True)
        flags = list(out.follow_instruction_list)
        n_total = len(flags)
        n_sat = sum(1 for f in flags if f)
        partial = n_sat / max(n_total, 1)
        strict = 1.0 if out.follow_all_instructions else 0.0
        return partial + 0.5 * strict
    except Exception:
        return 0.0


def reward_humaneval(response: str, gt: Dict[str, Any]) -> float:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = m.group(1) if m else response
    tests = gt.get("test", "") or gt.get("tests", "") or ""
    entry_point = gt.get("entry_point", "") or "solution"
    if not tests:
        return 0.0
    script = f"{code}\n\n{tests}\n\ncheck({entry_point})\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        r = subprocess.run(["python", path], capture_output=True, timeout=8)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        if path:
            try: os.unlink(path)
            except OSError: pass


REWARDS: Dict[str, Callable[[str, Dict[str, Any]], float]] = {
    "ifeval":    reward_ifeval,
    "gsm8k":     reward_gsm8k,
    "humaneval": reward_humaneval,
}

SYSTEM_PROMPTS = {
    "ifeval":    "Follow the user's instructions exactly. Output only the requested content.",
    "gsm8k":     "Solve the problem. Think step by step. Put the final numeric answer inside \\boxed{...}.",
    "humaneval": "Write a correct Python function. Return only one ```python ... ``` code block.",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _row_to_example(row: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    """Normalize a row into {prompt, ground_truth}."""
    if "messages" in row and row["messages"]:
        prompt = row["messages"][0]["content"]
    else:
        prompt = (row.get("prompt") or row.get("question")
                  or row.get("problem") or row.get("instruction"))
    if not prompt:
        return None

    gt = row.get("ground_truth")
    if isinstance(gt, str):
        try: gt = json.loads(gt)
        except Exception: gt = None
    if gt is None:
        if task == "gsm8k":
            gt = {"answer": row.get("answer", "")}
        elif task == "ifeval":
            gt = {"instruction_id_list": row.get("instruction_id_list", []),
                  "kwargs": row.get("kwargs", [])}
        elif task == "humaneval":
            gt = {"test": row.get("test", ""), "entry_point": row.get("entry_point", "")}
    return {"prompt": prompt, "ground_truth": gt}


def load_rl_examples(task: str, dataset_name: Optional[str], split: str) -> List[Dict]:
    name = dataset_name or DEFAULT_DATASETS[task]
    print(f"Loading {name} split={split}")
    ds = load_dataset(name, split=split)
    out = []
    for row in ds:
        ex = _row_to_example(row, task)
        if ex is not None:
            out.append(ex)
    print(f"  -> {len(out)} usable examples")
    return out


# ---------------------------------------------------------------------------
# Datum builder (matches loss_fn_inputs schema for "ppo" / "cispo" /
# "importance_sampling" — see tinker_cookbook/rl/data_processing.py).
# ---------------------------------------------------------------------------

def build_datum(prompt_tokens: List[int],
                response_tokens: List[int],
                response_logprobs: List[float],
                advantage: float) -> Optional[types.Datum]:
    full = list(prompt_tokens) + list(response_tokens)
    if len(full) < 2:
        return None
    input_tokens = full[:-1]
    target_tokens = full[1:]

    n_prompt = len(prompt_tokens)
    n_resp = len(response_tokens)
    full_lp = [0.0] * n_prompt + list(response_logprobs)
    full_adv = [0.0] * n_prompt + [float(advantage)] * n_resp
    full_mask = [0.0] * n_prompt + [1.0] * n_resp

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs":      TensorData.from_torch(torch.tensor(full_lp[1:], dtype=torch.float32)),
            "advantages":    TensorData.from_torch(torch.tensor(full_adv[1:], dtype=torch.float32)),
            "mask":          TensorData.from_torch(torch.tensor(full_mask[1:], dtype=torch.float32)),
        },
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Strong GRPO trainer (Tinker).")
    p.add_argument("--task", choices=list(REWARDS), required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")

    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--resume_from", type=str, required=True,
                   help="Tinker state path, e.g. tinker://...")

    p.add_argument("--num_iterations", type=int, default=60)
    p.add_argument("--prompts_per_iter", type=int, default=16)
    p.add_argument("--group_size", type=int, default=8,
                   help="GRPO samples per prompt (G).")
    p.add_argument("--ppo_epochs", type=int, default=2,
                   help="Replay each rollout batch this many times.")
    p.add_argument("--mini_batch_size", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--loss_fn", choices=["ppo", "cispo", "importance_sampling"],
                   default="ppo")
    p.add_argument("--kl_coef", type=float, default=0.01,
                   help="KL(policy || ref) penalty injected into advantages. 0 disables.")
    p.add_argument("--kl_discount", type=float, default=0.0)
    p.add_argument("--ref_refresh_every", type=int, default=0,
                   help="If >0, refresh the reference policy every N iters.")

    p.add_argument("--drop_zero_variance", action="store_true",
                   help="DAPO: drop prompts where all G samples got the same reward.")

    p.add_argument("--checkpoint_name", type=str, default="grpo_strong")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_publish", action="store_true")
    p.add_argument("--max_examples", type=int, default=0,
                   help="If >0, subsample dataset (useful for smoke tests).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.task == "ifeval":
        ensure_nltk_resource()

    examples = load_rl_examples(args.task, args.dataset_name, args.dataset_split)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    if not examples:
        raise SystemExit("No usable examples in dataset")

    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.model), tokenizer)
    print(f"Renderer: {model_info.get_recommended_renderer_name(args.model)}")

    sc = tinker.ServiceClient()
    print(f"Resuming from {args.resume_from}")
    tc = sc.create_training_client_from_state(args.resume_from)

    ref_client = None
    if args.kl_coef > 0:
        ref_path = tc.save_weights_for_sampler(name=f"{args.checkpoint_name}_ref0").result().path
        ref_client = sc.create_sampling_client(model_path=ref_path)
        print(f"Reference policy fixed at: {ref_path}")

    adam = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sampling_params = types.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    log_path = os.path.join(EVAL_DIR, f"{args.checkpoint_name}_log.jsonl")
    log_f = open(log_path, "w")

    history = {"reward": [], "strict": [], "kl": []}

    for it in range(args.num_iterations):
        t0 = time.time()
        # Snapshot sampler weights for this iteration's rollouts
        ckpt = tc.save_weights_for_sampler(
            name=f"{args.checkpoint_name}_iter{it}"
        ).result()
        sampler = sc.create_sampling_client(model_path=ckpt.path)

        rollouts = []  # tuples (prompt_tokens, [(resp_tokens, resp_lp, reward)])
        all_rewards = []

        for _ in range(args.prompts_per_iter):
            ex = rng.choice(examples)
            convo = [
                {"role": "system", "content": SYSTEM_PROMPTS[args.task]},
                {"role": "user",   "content": ex["prompt"]},
            ]
            prompt_input = renderer.build_generation_prompt(convo)
            prompt_tokens = list(prompt_input.to_ints())

            res = sampler.sample(
                prompt=prompt_input,
                sampling_params=sampling_params,
                num_samples=args.group_size,
            ).result()

            samples = []
            rs = []
            reward_fn = REWARDS[args.task]
            for seq in res.sequences:
                resp_tokens = list(seq.tokens)
                resp_lp = list(seq.logprobs) if seq.logprobs else [0.0] * len(resp_tokens)
                text = tokenizer.decode(resp_tokens)
                r = reward_fn(text, ex["ground_truth"])
                samples.append((resp_tokens, resp_lp, r))
                rs.append(r)
                all_rewards.append(r)

            # DAPO group filtering
            if args.drop_zero_variance and float(np.std(rs)) < 1e-6:
                continue

            rollouts.append((prompt_tokens, samples))

        if not rollouts:
            print(f"  Iter {it+1}: all groups filtered, skipping update")
            continue

        # GRPO group-relative advantages → datums
        datums = []
        for prompt_tokens, samples in rollouts:
            rs = np.array([s[2] for s in samples], dtype=np.float32)
            std = float(rs.std())
            if std > 1e-6:
                advs = (rs - rs.mean()) / std
            else:
                advs = rs - 0.5  # fallback so a uniformly-good group still pushes up
            for (resp_tokens, resp_lp, _r), adv in zip(samples, advs):
                d = build_datum(prompt_tokens, resp_tokens, resp_lp, float(adv))
                if d is not None:
                    datums.append(d)

        # KL injection: modifies advantages in-place using ref logprobs
        kl_metrics = {}
        if ref_client is not None and args.kl_coef > 0:
            try:
                kl_metrics = asyncio.run(incorporate_kl_penalty(
                    datums, ref_client,
                    kl_penalty_coef=args.kl_coef,
                    kl_discount_factor=args.kl_discount,
                ))
            except Exception as e:
                print(f"  WARN: KL injection failed: {e}")

        # The built-in losses don't accept the "mask" field — it's metadata
        # used only by the KL/metric helpers above. Strip it before sending.
        def _strip_mask(d: types.Datum) -> types.Datum:
            return types.Datum(
                model_input=d.model_input,
                loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items()
                                if k != "mask"},
            )
        train_datums = [_strip_mask(d) for d in datums]

        # PPO multi-epoch update over the rollout batch
        n_steps = 0
        for epoch in range(args.ppo_epochs):
            order = list(range(len(train_datums)))
            rng.shuffle(order)
            shuffled = [train_datums[i] for i in order]
            for i in range(0, len(shuffled), args.mini_batch_size):
                batch = shuffled[i:i + args.mini_batch_size]
                tc.forward_backward(batch, loss_fn=args.loss_fn).result()
            tc.optim_step(adam).result()
            n_steps += 1

        rew_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
        rew_std = float(np.std(all_rewards)) if all_rewards else 0.0
        strict = float(np.mean([1.0 if r >= 1.0 else 0.0 for r in all_rewards])) if all_rewards else 0.0
        kl_val = float(kl_metrics.get("kl_policy_base", 0.0))
        history["reward"].append(rew_mean)
        history["strict"].append(strict)
        history["kl"].append(kl_val)

        elapsed = time.time() - t0
        row = {
            "iter": it + 1,
            "n_rollouts": len(rollouts),
            "n_datums": len(datums),
            "ppo_steps": n_steps,
            "reward_mean": rew_mean,
            "reward_std": rew_std,
            "strict_or_full": strict,
            "kl_policy_ref": kl_val,
            "loss_fn": args.loss_fn,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  Iter {it+1}/{args.num_iterations} | "
              f"reward {rew_mean:.3f} ±{rew_std:.3f} | "
              f"full {strict:.3f} | "
              f"kl {kl_val:.4f} | "
              f"datums {len(datums)} | "
              f"{elapsed:.1f}s")
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        # Periodic ref-policy refresh (off by default)
        if (args.ref_refresh_every > 0 and (it + 1) % args.ref_refresh_every == 0
                and args.kl_coef > 0):
            ref_path = tc.save_weights_for_sampler(
                name=f"{args.checkpoint_name}_ref{it+1}"
            ).result().path
            ref_client = sc.create_sampling_client(model_path=ref_path)
            print(f"  Refreshed reference policy: {ref_path}")

        if (it + 1) % args.save_every == 0:
            tag = f"{args.checkpoint_name}_iter{it+1}"
            sp = tc.save_state(tag + "_state").result().path
            print(f"  Saved state: {sp}")

    # Final save + publish
    print(f"\nSaving final checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    state = tc.save_state(args.checkpoint_name + "_state").result()
    print(f"  Checkpoint: {ckpt.path}")
    print(f"  State:      {state.path}")

    if not args.no_publish:
        try:
            sc.create_rest_client().publish_checkpoint_from_tinker_path(ckpt.path).result()
            print("  Published.")
        except Exception as e:
            print(f"  Publish failed: {e}")

    info = {
        "checkpoint_path": ckpt.path,
        "state_path": state.path,
        "base_model": args.model,
        "task": args.task,
        "dataset": args.dataset_name or DEFAULT_DATASETS[args.task],
        "loss_fn": args.loss_fn,
        "kl_coef": args.kl_coef,
        "ppo_epochs": args.ppo_epochs,
        "history": history,
    }
    with open(os.path.join(EVAL_DIR, f"{args.checkpoint_name}_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log_f.close()
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
