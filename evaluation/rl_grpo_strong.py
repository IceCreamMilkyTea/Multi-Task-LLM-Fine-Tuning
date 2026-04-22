"""
Strong GRPO trainer for Tinker, modeled after grpo_ref/open-instruct/grpo_fast.py.

Key features vs. the baseline rl_ifeval.py / rl_train.py:

1. DAPO clipped surrogate objective with asymmetric clipping
       L = max(-A * ratio, -A * clip(ratio, 1 - eps_low, 1 + eps_high))
   and optional CISPO variant.

2. Per-token KL penalty against a frozen reference policy using k1/k2/k3
   estimators (Schulman's unbiased low-variance KL).

3. Multiple PPO epochs per rollout with cached "old" log-probs, so the
   update actually uses the ratio (our previous code did a single pass).

4. Token-level loss normalization ("token" denominator) vs. Dr-GRPO constant.

5. DAPO group filtering: drop prompts where every sample got the same reward
   (zero advantage, zero gradient signal, wasted compute).

6. Optional Truncated Importance Sampling (TIS) to correct drift between the
   sampler and the trainer.

7. Rich diagnostics: clipfrac, ratio mean/var, kl{1,2,3}, entropy,
   reward mean/std, group-filter fraction.

8. Periodic reference-policy refresh (ref_policy_update_freq).

Runs on Tinker's TrainingClient + SamplingClient. Reward functions are
pluggable (IFEval verifier / GSM8K verifier / HumanEval unit-test verifier).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tinker
from tinker import types
from tinker.types import tensor_data
from datasets import load_dataset

logger = logging.getLogger("rl_grpo_strong")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    # Model / checkpoint
    base_model: str = "Qwen/Qwen3-8B"
    load_checkpoint_path: Optional[str] = None  # e.g. tinker://.../sampler_weights/...

    # Task / dataset
    task: str = "ifeval"  # "ifeval" | "gsm8k" | "humaneval"
    dataset_name: Optional[str] = None  # overrides task default
    dataset_split: str = "train"
    max_prompt_length: int = 1024
    max_response_length: int = 1024

    # Sampling
    group_size: int = 8               # G samples per prompt (GRPO group)
    prompts_per_batch: int = 32       # number of prompts per rollout batch
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0

    # Optimization
    learning_rate: float = 1e-6
    num_iterations: int = 500
    ppo_epochs: int = 2               # epochs over each rollout batch
    mini_batches_per_rollout: int = 1 # splits within each epoch
    max_grad_norm: float = 1.0

    # DAPO / PPO clipping
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28       # DAPO asymmetric upper clip
    loss_type: str = "dapo"           # "dapo" | "cispo" | "vanilla"

    # KL to reference policy
    beta_kl: float = 0.01             # 0 disables KL term
    kl_estimator: str = "k3"          # "k1" | "k2" | "k3"
    ref_policy_update_freq: int = 0   # 0 = frozen ref; >0 = refresh every N iters

    # Loss normalization
    loss_normalizer: str = "token"    # "token" | "dr_grpo"
    dr_grpo_constant: float = 1.0

    # DAPO group filtering
    drop_zero_variance_groups: bool = True

    # Truncated importance sampling (sampler vs trainer drift)
    use_tis: bool = True
    tis_cap: float = 2.0

    # Rewards
    reward_format_bonus: float = 0.0  # small bonus for following output format

    # Logging / checkpoints
    log_dir: str = "rl_grpo_runs"
    save_every: int = 50
    seed: int = 42


# ---------------------------------------------------------------------------
# Reward functions (verifiers). Each returns float in [0, 1].
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    import re
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*([-+]?[0-9][0-9,\.]*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def reward_gsm8k(response: str, ground_truth: Dict[str, Any]) -> float:
    """GSM8K: parse the final numeric answer from response and match."""
    gold = str(ground_truth["answer"]).split("####")[-1].strip().replace(",", "")
    pred = _extract_boxed(response)
    if pred is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if pred == gold else 0.0


def reward_ifeval(response: str, ground_truth: Dict[str, Any]) -> float:
    """IFEval: run the exact instruction checkers; reward = fraction satisfied."""
    try:
        from instruction_following_eval import instructions_registry
    except Exception:
        return 0.0
    instr_ids: List[str] = ground_truth["instruction_id_list"]
    kwargs_list: List[Dict[str, Any]] = ground_truth.get("kwargs") or [{} for _ in instr_ids]
    if not instr_ids:
        return 0.0
    ok = 0
    for iid, kw in zip(instr_ids, kwargs_list):
        try:
            cls = instructions_registry.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            if inst.check_following(response):
                ok += 1
        except Exception:
            pass
    return ok / len(instr_ids)


def reward_humaneval(response: str, ground_truth: Dict[str, Any]) -> float:
    """Code: extract a python block and run the provided unit tests in a subprocess."""
    import re, subprocess, tempfile, textwrap
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = m.group(1) if m else response
    tests: str = ground_truth.get("test", "")
    entry_point: str = ground_truth.get("entry_point", "")
    if not tests:
        return 0.0
    script = f"{code}\n\n{tests}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=8)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        try: os.unlink(path)
        except OSError: pass


RewardFn = Callable[[str, Dict[str, Any]], float]

REWARDS: Dict[str, RewardFn] = {
    "gsm8k": reward_gsm8k,
    "ifeval": reward_ifeval,
    "humaneval": reward_humaneval,
}


# ---------------------------------------------------------------------------
# Dataset loaders — each yields {"prompt": str, "ground_truth": dict}
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = {
    "ifeval":    "allenai/RLVR-IFeval",                       # ~15K, IFEval verifier
    "gsm8k":     "allenai/RLVR-GSM",                          # 7.5K, numeric verifier
    "humaneval": "PrimeIntellect/verifiable-coding-problems", # has unit tests
}


def _format_rlvr_ifeval(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row["messages"][0]["content"] if "messages" in row else row["prompt"]
    gt = row.get("ground_truth")
    if isinstance(gt, str):
        try: gt = json.loads(gt)
        except Exception: gt = {"instruction_id_list": [], "kwargs": []}
    return {"prompt": prompt, "ground_truth": gt}


def _format_rlvr_gsm(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row["messages"][0]["content"] if "messages" in row else row["question"]
    gt = row.get("ground_truth") or {"answer": row.get("answer", "")}
    if isinstance(gt, str):
        gt = {"answer": gt}
    return {"prompt": prompt, "ground_truth": gt}


def _format_code(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row.get("prompt") or row.get("question") or row.get("problem", "")
    gt = {
        "test": row.get("test") or row.get("tests") or "",
        "entry_point": row.get("entry_point") or row.get("func_name", "solution"),
    }
    return {"prompt": prompt, "ground_truth": gt}


def load_rl_dataset(cfg: GRPOConfig) -> List[Dict[str, Any]]:
    name = cfg.dataset_name or DEFAULT_DATASETS[cfg.task]
    logger.info(f"Loading dataset {name} split={cfg.dataset_split}")
    ds = load_dataset(name, split=cfg.dataset_split)
    formatter = {
        "ifeval": _format_rlvr_ifeval,
        "gsm8k": _format_rlvr_gsm,
        "humaneval": _format_code,
    }[cfg.task]
    out = [formatter(r) for r in ds]
    out = [r for r in out if r["prompt"]]
    logger.info(f"Loaded {len(out)} examples")
    return out


def build_chat_prompt(tokenizer, user_prompt: str, task: str) -> List[int]:
    system = {
        "ifeval":   "Follow the user's instructions exactly. Output only the requested content.",
        "gsm8k":    "Solve the problem. Think step by step. Put the final numeric answer inside \\boxed{...}.",
        "humaneval":"Write a correct Python function. Return only a single ```python ... ``` code block.",
    }[task]
    msgs = [{"role": "system", "content": system},
            {"role": "user",   "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


# ---------------------------------------------------------------------------
# Core GRPO / DAPO loss
# ---------------------------------------------------------------------------

def compute_kl(logprob: np.ndarray, ref_logprob: np.ndarray, estimator: str) -> np.ndarray:
    """Per-token KL estimate between current (logprob) and reference (ref_logprob).
    Using Schulman's definitions:
        k1 = logprob - ref_logprob
        k2 = 0.5 * (logprob - ref_logprob) ** 2
        k3 = exp(ref_logprob - logprob) - (ref_logprob - logprob) - 1  (unbiased, >=0)
    """
    diff = logprob - ref_logprob
    if estimator == "k1":
        return diff
    if estimator == "k2":
        return 0.5 * diff * diff
    # k3
    return np.exp(-diff) + diff - 1.0


def grpo_loss_tokens(
    new_logprobs: np.ndarray,   # (T,) from forward pass
    old_logprobs: np.ndarray,   # (T,) cached when rollout was taken
    advantages: np.ndarray,     # (T,) broadcast per-sequence advantage
    ref_logprobs: Optional[np.ndarray],
    sampler_logprobs: Optional[np.ndarray],
    cfg: GRPOConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute per-token unreduced loss; caller sums/normalizes."""
    ratio = np.exp(new_logprobs - old_logprobs)

    if cfg.loss_type == "dapo":
        clipped = np.clip(ratio, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)
        loss_pg = -np.minimum(ratio * advantages, clipped * advantages)
        clipfrac = float(np.mean((ratio > 1 + cfg.clip_eps_high) |
                                 (ratio < 1 - cfg.clip_eps_low)))
    elif cfg.loss_type == "cispo":
        # Clip the importance weight itself, treat as constant on A's sign
        w = np.minimum(ratio, 1.0 + cfg.clip_eps_high)
        loss_pg = -w * advantages
        clipfrac = float(np.mean(ratio > 1 + cfg.clip_eps_high))
    else:  # vanilla
        loss_pg = -ratio * advantages
        clipfrac = 0.0

    # TIS correction: sampler_logprobs come from the (possibly stale) sampling
    # engine; cap the correction weight to avoid blowups.
    if cfg.use_tis and sampler_logprobs is not None:
        tis = np.exp(old_logprobs - sampler_logprobs)
        tis = np.minimum(tis, cfg.tis_cap)
        loss_pg = loss_pg * tis

    kl_term = np.zeros_like(loss_pg)
    kl_mean = 0.0
    if cfg.beta_kl > 0.0 and ref_logprobs is not None:
        kl_term = compute_kl(new_logprobs, ref_logprobs, cfg.kl_estimator)
        kl_mean = float(np.mean(kl_term))
        loss_pg = loss_pg + cfg.beta_kl * kl_term

    diag = {
        "ratio_mean": float(np.mean(ratio)),
        "ratio_max":  float(np.max(ratio)),
        "clipfrac":   clipfrac,
        "kl":         kl_mean,
    }
    return loss_pg, diag


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    prompt_tokens: List[int]
    response_tokens: List[int]
    sampler_logprobs: List[float]   # logprobs from the sampling engine
    reward: float
    advantage: float = 0.0          # filled after group-normalization
    group_id: int = 0


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed)

        self.service = tinker.ServiceClient()
        logger.info(f"Creating training client for {cfg.base_model}")
        self.train_client = self.service.create_lora_training_client(base_model=cfg.base_model)
        if cfg.load_checkpoint_path:
            logger.info(f"Loading checkpoint {cfg.load_checkpoint_path}")
            self.train_client.load_state(cfg.load_checkpoint_path).result()

        # Sampling client — points at current policy
        self.sampling_path = self.train_client.save_weights_for_sampler(
            name="grpo_init").result().path
        self.sampling_client = self.service.create_sampling_client(model_path=self.sampling_path)

        # Reference policy (frozen snapshot for KL regularization)
        self.ref_sampling_client = self.sampling_client if cfg.beta_kl > 0 else None

        self.tokenizer = self.train_client.get_tokenizer()
        self.reward_fn: RewardFn = REWARDS[cfg.task]
        self.dataset = load_rl_dataset(cfg)
        self._data_idx = 0

        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open(Path(cfg.log_dir) / "train.jsonl", "a")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _sample_group(self, prompt_tokens: List[int]) -> List[Tuple[List[int], List[float]]]:
        """Draw group_size responses from the current sampling client."""
        params = types.SamplingParams(
            max_tokens=self.cfg.max_response_length,
            temperature=self.cfg.sampling_temperature,
            top_p=self.cfg.sampling_top_p,
            stop=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            logprobs=True,
        )
        prompt = types.ModelInput.from_ints(prompt_tokens)
        fut = self.sampling_client.sample(
            prompt=prompt, num_samples=self.cfg.group_size, sampling_params=params,
        )
        resp = fut.result()
        out: List[Tuple[List[int], List[float]]] = []
        for s in resp.sequences:
            out.append((list(s.tokens), list(s.logprobs)))
        return out

    def _next_prompts(self) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(self.cfg.prompts_per_batch):
            batch.append(self.dataset[self._data_idx % len(self.dataset)])
            self._data_idx += 1
        return batch

    def collect_rollouts(self) -> List[Rollout]:
        """Sample G responses for each of prompts_per_batch prompts, score them,
        compute group-normalized advantages, and optionally drop zero-variance groups."""
        t0 = time.time()
        rollouts: List[Rollout] = []
        batch = self._next_prompts()
        rewards_by_group: List[List[float]] = []

        for gi, ex in enumerate(batch):
            try:
                prompt_toks = build_chat_prompt(self.tokenizer, ex["prompt"], self.cfg.task)
            except Exception as e:
                logger.warning(f"Prompt build failed: {e}")
                continue
            if len(prompt_toks) > self.cfg.max_prompt_length:
                prompt_toks = prompt_toks[-self.cfg.max_prompt_length:]

            samples = self._sample_group(prompt_toks)
            grp_rewards = []
            grp_rollouts = []
            for tokens, slps in samples:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                r = self.reward_fn(text, ex["ground_truth"])
                grp_rewards.append(r)
                grp_rollouts.append(Rollout(
                    prompt_tokens=prompt_toks,
                    response_tokens=tokens,
                    sampler_logprobs=slps,
                    reward=r,
                    group_id=gi,
                ))
            rewards_by_group.append(grp_rewards)

            mu = float(np.mean(grp_rewards))
            sd = float(np.std(grp_rewards))
            if self.cfg.drop_zero_variance_groups and sd < 1e-6:
                continue  # DAPO filter
            denom = sd + 1e-6
            for ro, r in zip(grp_rollouts, grp_rewards):
                ro.advantage = (r - mu) / denom
                rollouts.append(ro)

        dt = time.time() - t0
        all_r = [r for g in rewards_by_group for r in g]
        logger.info(
            f"Rollout: prompts={len(batch)} kept={len(rollouts)} "
            f"reward_mean={np.mean(all_r) if all_r else 0:.3f} "
            f"reward_std={np.std(all_r) if all_r else 0:.3f} ({dt:.1f}s)"
        )
        return rollouts

    # ------------------------------------------------------------------
    # Log-prob computation (sampler side) for old & reference policies
    # ------------------------------------------------------------------

    def _compute_logprobs(self, client, rollouts: List[Rollout]) -> List[np.ndarray]:
        """Return per-rollout response-token logprobs under the given sampler client.
        Uses sampling_client.compute_logprobs (teacher-forced) when available;
        otherwise falls back to the cached sampler_logprobs.
        """
        if not hasattr(client, "compute_logprobs"):
            return [np.asarray(r.sampler_logprobs, dtype=np.float32) for r in rollouts]
        inputs = []
        for r in rollouts:
            full = r.prompt_tokens + r.response_tokens
            inputs.append(types.ModelInput.from_ints(full))
        fut = client.compute_logprobs(inputs=inputs)
        res = fut.result()
        out: List[np.ndarray] = []
        for r, lp in zip(rollouts, res.logprobs):
            # Slice to the response portion only (last len(response_tokens) tokens).
            arr = np.asarray(lp, dtype=np.float32)
            out.append(arr[-len(r.response_tokens):])
        return out

    # ------------------------------------------------------------------
    # Training step — PPO-style multi-epoch, clipped surrogate + KL
    # ------------------------------------------------------------------

    def _build_training_examples(
        self,
        rollouts: List[Rollout],
        old_lps: List[np.ndarray],
        ref_lps: Optional[List[np.ndarray]],
    ) -> List[types.Datum]:
        """Build Tinker Datum objects carrying the per-token auxiliary tensors
        (old_logprobs, ref_logprobs, sampler_logprobs, advantages) that the
        custom-loss forward pass will consume.
        """
        data: List[types.Datum] = []
        for i, ro in enumerate(rollouts):
            prompt_len = len(ro.prompt_tokens)
            full = ro.prompt_tokens + ro.response_tokens
            L = len(full)

            # Loss mask: 1 on response tokens, 0 on prompt tokens.
            mask = np.zeros(L, dtype=np.float32)
            mask[prompt_len:] = 1.0

            adv = np.zeros(L, dtype=np.float32)
            adv[prompt_len:] = ro.advantage

            old_lp = np.zeros(L, dtype=np.float32)
            old_lp[prompt_len:] = old_lps[i]

            smp_lp = np.zeros(L, dtype=np.float32)
            smp_lp[prompt_len:] = np.asarray(ro.sampler_logprobs, dtype=np.float32)

            ref_lp = np.zeros(L, dtype=np.float32)
            if ref_lps is not None:
                ref_lp[prompt_len:] = ref_lps[i]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(full[:-1]),
                loss_fn_inputs={
                    "targets":          tensor_data.from_numpy(np.asarray(full[1:], dtype=np.int64)),
                    "mask":             tensor_data.from_numpy(mask[1:]),
                    "advantages":       tensor_data.from_numpy(adv[1:]),
                    "old_logprobs":     tensor_data.from_numpy(old_lp[1:]),
                    "sampler_logprobs": tensor_data.from_numpy(smp_lp[1:]),
                    "ref_logprobs":     tensor_data.from_numpy(ref_lp[1:]),
                },
            )
            data.append(datum)
        return data

    def _custom_loss(self, batch_outputs, batch_inputs):
        """Called by forward_backward_custom with a batch of per-example outputs.
        Implements DAPO/CISPO clipped objective with KL and TIS.
        batch_outputs[i].logprobs has per-token logprobs for the chosen targets.
        """
        cfg = self.cfg
        total_loss = 0.0
        total_tokens = 0.0
        agg = {"ratio_mean": 0.0, "ratio_max": 0.0, "clipfrac": 0.0, "kl": 0.0, "n": 0}

        for out, inp in zip(batch_outputs, batch_inputs):
            new_lp = out.logprobs                          # tensor-like (T,)
            lfi = inp.loss_fn_inputs
            mask = lfi["mask"].to_numpy()
            adv  = lfi["advantages"].to_numpy()
            old  = lfi["old_logprobs"].to_numpy()
            smp  = lfi["sampler_logprobs"].to_numpy()
            ref  = lfi["ref_logprobs"].to_numpy()

            new_np = new_lp.to_numpy() if hasattr(new_lp, "to_numpy") else np.asarray(new_lp)

            sel = mask > 0
            if not sel.any():
                continue

            tok_loss, diag = grpo_loss_tokens(
                new_logprobs=new_np[sel],
                old_logprobs=old[sel],
                advantages=adv[sel],
                ref_logprobs=ref[sel] if cfg.beta_kl > 0 else None,
                sampler_logprobs=smp[sel] if cfg.use_tis else None,
                cfg=cfg,
            )

            if cfg.loss_normalizer == "token":
                total_loss += float(tok_loss.sum())
                total_tokens += float(sel.sum())
            else:  # dr_grpo: constant denominator per sequence
                total_loss += float(tok_loss.sum()) / cfg.dr_grpo_constant
                total_tokens += 1.0

            for k in ("ratio_mean", "ratio_max", "clipfrac", "kl"):
                agg[k] += diag[k]
            agg["n"] += 1

        denom = max(total_tokens, 1.0)
        loss = total_loss / denom
        if agg["n"]:
            for k in ("ratio_mean", "ratio_max", "clipfrac", "kl"):
                agg[k] /= agg["n"]
        agg.pop("n")
        return loss, agg

    def train_step(self, rollouts: List[Rollout]) -> Dict[str, float]:
        cfg = self.cfg
        if not rollouts:
            return {"skipped": 1.0}

        # 1) Compute "old" log-probs (under the sampling policy) and reference
        #    log-probs (frozen ref policy) once per rollout batch.
        old_lps = self._compute_logprobs(self.sampling_client, rollouts)
        ref_lps = None
        if cfg.beta_kl > 0 and self.ref_sampling_client is not None:
            ref_lps = self._compute_logprobs(self.ref_sampling_client, rollouts)

        # 2) Run ppo_epochs * mini_batches passes over this data.
        diags_accum: Dict[str, float] = {}
        n_updates = 0
        indices = list(range(len(rollouts)))
        for epoch in range(cfg.ppo_epochs):
            random.shuffle(indices)
            chunks = np.array_split(indices, max(cfg.mini_batches_per_rollout, 1))
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                sub = [rollouts[i] for i in chunk]
                sub_old = [old_lps[i] for i in chunk]
                sub_ref = [ref_lps[i] for i in chunk] if ref_lps is not None else None
                data = self._build_training_examples(sub, sub_old, sub_ref)

                fb = self.train_client.forward_backward_custom(
                    data=data, loss_fn=self._custom_loss,
                ).result()
                self.train_client.optim_step(
                    types.AdamParams(learning_rate=cfg.learning_rate)
                ).result()

                # Aggregate diagnostics
                metrics = getattr(fb, "metrics", None) or {}
                for k, v in metrics.items():
                    diags_accum[k] = diags_accum.get(k, 0.0) + float(v)
                n_updates += 1

        if n_updates:
            for k in list(diags_accum):
                diags_accum[k] /= n_updates

        # 3) Refresh the sampler weights so the next rollout uses the updated policy.
        self.sampling_path = self.train_client.save_weights_for_sampler(
            name=f"grpo_step_{int(time.time())}",
        ).result().path
        self.sampling_client = self.service.create_sampling_client(model_path=self.sampling_path)

        return diags_accum

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        for it in range(1, cfg.num_iterations + 1):
            rollouts = self.collect_rollouts()
            stats = self.train_step(rollouts)

            # Reward stats for logging
            rewards = [ro.reward for ro in rollouts]
            row = {
                "iter": it,
                "n_rollouts": len(rollouts),
                "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
                "reward_std":  float(np.std(rewards))  if rewards else 0.0,
                **stats,
            }
            logger.info(json.dumps(row))
            self.log_file.write(json.dumps(row) + "\n"); self.log_file.flush()

            # Optional ref-policy refresh
            if (cfg.ref_policy_update_freq > 0
                    and it % cfg.ref_policy_update_freq == 0
                    and cfg.beta_kl > 0):
                p = self.train_client.save_weights_for_sampler(
                    name=f"ref_refresh_{it}").result().path
                self.ref_sampling_client = self.service.create_sampling_client(model_path=p)
                logger.info(f"Refreshed reference policy at iter {it}")

            # Checkpoint
            if it % cfg.save_every == 0:
                p = self.train_client.save_state(name=f"grpo_{cfg.task}_iter{it}").result().path
                logger.info(f"Saved state: {p}")

        # Final save
        p = self.train_client.save_state(name=f"grpo_{cfg.task}_final").result().path
        logger.info(f"Final checkpoint: {p}")
        self.log_file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser(description="Strong GRPO trainer for Tinker")
    p.add_argument("--task", choices=list(REWARDS), required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--load_checkpoint_path", default=None)

    p.add_argument("--num_iterations", type=int, default=500)
    p.add_argument("--prompts_per_batch", type=int, default=32)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--mini_batches_per_rollout", type=int, default=1)

    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--sampling_top_p", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_response_length", type=int, default=1024)

    p.add_argument("--clip_eps_low", type=float, default=0.2)
    p.add_argument("--clip_eps_high", type=float, default=0.28)
    p.add_argument("--loss_type", choices=["dapo", "cispo", "vanilla"], default="dapo")
    p.add_argument("--loss_normalizer", choices=["token", "dr_grpo"], default="token")

    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--kl_estimator", choices=["k1", "k2", "k3"], default="k3")
    p.add_argument("--ref_policy_update_freq", type=int, default=0)

    p.add_argument("--use_tis", action="store_true")
    p.add_argument("--no_tis", dest="use_tis", action="store_false")
    p.set_defaults(use_tis=True)
    p.add_argument("--tis_cap", type=float, default=2.0)

    p.add_argument("--drop_zero_variance_groups", action="store_true")
    p.add_argument("--keep_zero_variance_groups", dest="drop_zero_variance_groups", action="store_false")
    p.set_defaults(drop_zero_variance_groups=True)

    p.add_argument("--log_dir", default="rl_grpo_runs")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    return GRPOConfig(**vars(a))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args()
    log_sub = Path(cfg.log_dir) / f"{cfg.task}_{int(time.time())}"
    log_sub.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_sub)
    with open(log_sub / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    logger.info(f"Config: {cfg}")
    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
