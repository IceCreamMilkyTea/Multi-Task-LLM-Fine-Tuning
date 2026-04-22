"""
IFEval Constraint RL — GRPO-style RL aligned with IFEval's prompt_strict_acc.

Key design choices (vs. the prior single-constraint binary-reward version):
  1. Prompts carry 1-3 constraints at once (IFEval-style), sampled from the
     official `instruction_following_eval.instructions_registry` (25 types).
     This matches how IFEval builds real prompts and forces the model to
     satisfy multiple constraints simultaneously — the behavior the
     `prompt_strict_acc` metric actually measures.
  2. Reward is computed with the official `test_instruction_following`
     checker used by inspect_evals/ifeval, so training signal is identical
     to eval scoring.
  3. Partial credit: reward = n_satisfied / n_total + bonus * follow_all.
     The fraction term gives a dense gradient (2/3 > 1/3 > 0/3), the bonus
     keeps the optimum aligned with prompt_strict (follow_all = +bonus).

Usage:
    python evaluation/rl_ifeval.py \\
        --model meta-llama/Llama-3.2-3B \\
        --resume_from "tinker://..._state" \\
        --num_iterations 30 \\
        --checkpoint_name exp_ifeval_rl
"""

import argparse
import json
import os
import random

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import tinker
import torch
from tinker import types
from tinker.types import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from instruction_following_eval import instructions_registry
from instruction_following_eval.evaluation import (
    InputExample,
    ensure_nltk_resource,
    test_instruction_following,
)

MODEL_3B = "meta-llama/Llama-3.2-3B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# ============================================================
# Multi-constraint prompt generation using the official IFEval
# instructions_registry. Each instruction's build_description()
# auto-samples valid kwargs, so we don't need per-type sampling code.
# ============================================================

# Instruction types that don't compose well with free-form topic responses
# (they constrain the prompt itself or require specific content structure
# incompatible with a general "write about X" format).
_EXCLUDED_INST_IDS = {
    "combination:repeat_prompt",        # asks model to repeat the prompt verbatim
    "detectable_format:json_format",    # forces entire response to be JSON
    "detectable_format:constrained_response",  # "pick from (yes/no/maybe)"
    "language:response_language",       # non-English languages; conflicts with topic
}

INSTRUCTION_IDS = [
    iid for iid in instructions_registry.INSTRUCTION_DICT.keys()
    if iid not in _EXCLUDED_INST_IDS
]

TOPICS = [
    "Explain the water cycle",
    "Describe the benefits of exercise",
    "Write about the history of computers",
    "Explain how photosynthesis works",
    "Describe the solar system",
    "Write about healthy eating habits",
    "Explain the importance of recycling",
    "Describe how airplanes fly",
    "Explain how the internet works",
    "Describe the life cycle of a butterfly",
    "Write about different types of energy",
    "Explain what machine learning is",
    "Write about the importance of sleep",
    "Explain how vaccines work",
    "Explain the greenhouse effect",
    "Describe how electric cars work",
    "Explain the concept of gravity",
    "Describe the process of evolution",
    "Write about climate change",
    "Explain how batteries work",
]


def generate_constrained_prompt(rng, min_constraints=1, max_constraints=3):
    """Sample a topic + 1-3 distinct IFEval constraints; return prompt
    plus metadata needed to score with test_instruction_following."""
    topic = rng.choice(TOPICS)
    n = rng.randint(min_constraints, max_constraints)
    # Sample distinct instruction categories (family prefix before ':') to
    # avoid contradictory pairs like two different paragraph-count rules.
    chosen_ids = []
    chosen_families = set()
    shuffled = list(INSTRUCTION_IDS)
    rng.shuffle(shuffled)
    for iid in shuffled:
        fam = iid.split(":")[0]
        if fam in chosen_families:
            continue
        chosen_ids.append(iid)
        chosen_families.add(fam)
        if len(chosen_ids) >= n:
            break

    descriptions = []
    kwargs_list = []
    for iid in chosen_ids:
        cls = instructions_registry.INSTRUCTION_DICT[iid]
        # Per-instruction RNG seed so repeated (topic, iid) pairs get
        # different kwargs across iters.
        inst = cls(iid)
        desc = inst.build_description()          # auto-samples valid kwargs
        kwargs = inst.get_instruction_args() or {}
        # Drop None values (same cleanup inspect_evals does in record_to_sample)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        descriptions.append(desc)
        kwargs_list.append(kwargs)

    prompt = f"{topic}. " + " ".join(descriptions)
    return prompt, chosen_ids, kwargs_list


def score_response(response_text, instruction_id_list, kwargs_list, strict_bonus=0.5):
    """Return (partial_reward, strict_reward, combined_reward, n_satisfied, n_total).

    - partial_reward  = n_satisfied / n_total               (dense signal)
    - strict_reward   = 1.0 iff all constraints satisfied   (prompt_strict_acc aligned)
    - combined_reward = partial_reward + strict_bonus * strict_reward
    """
    try:
        ex = InputExample(
            key=0,
            instruction_id_list=instruction_id_list,
            prompt="",  # not used by checkers that don't reference prompt
            kwargs=kwargs_list,
        )
        out = test_instruction_following(ex, response_text, strict=True)
        flags = list(out.follow_instruction_list)
        n_total = len(flags)
        n_satisfied = sum(1 for f in flags if f)
        partial = n_satisfied / n_total if n_total > 0 else 0.0
        strict = 1.0 if out.follow_all_instructions else 0.0
        combined = partial + strict_bonus * strict
        return partial, strict, combined, n_satisfied, n_total
    except Exception:
        return 0.0, 0.0, 0.0, 0, len(instruction_id_list)


def main():
    parser = argparse.ArgumentParser(description="IFEval Constraint RL (prompt_strict-aligned)")
    parser.add_argument("--resume_from", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=30)
    parser.add_argument("--num_prompts_per_iter", type=int, default=8)
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint_name", type=str, default="exp_ifeval_rl")
    parser.add_argument("--no_publish", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_8B,
                        choices=[MODEL_3B, MODEL_8B])
    parser.add_argument("--min_constraints", type=int, default=1,
                        help="Min # of constraints per prompt")
    parser.add_argument("--max_constraints", type=int, default=3,
                        help="Max # of constraints per prompt (IFEval prompts typically have 1-3)")
    parser.add_argument("--strict_bonus", type=float, default=0.5,
                        help="Extra reward when ALL constraints satisfied (prompt_strict alignment). "
                             "reward = n_satisfied/n_total + strict_bonus * follow_all")
    args = parser.parse_args()

    MODEL = args.model
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    ensure_nltk_resource()  # required before calling test_instruction_following

    sc = tinker.ServiceClient()
    print(f"Resuming from: {args.resume_from}")
    tc = sc.create_training_client_from_state(args.resume_from)
    print("Training client ready")

    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    print(f"\nIFEval Constraint RL: {args.num_iterations} iterations")
    print(f"  Prompts per iter: {args.num_prompts_per_iter}")
    print(f"  Samples per prompt: {args.num_samples_per_prompt}")
    print(f"  Constraints per prompt: {args.min_constraints}-{args.max_constraints}")
    print(f"  Reward: n_satisfied/n_total + {args.strict_bonus} * follow_all  (prompt_strict aligned)")
    print(f"  Using {len(INSTRUCTION_IDS)} IFEval instruction types")
    print(f"  LR: {args.lr}, temperature: {args.temperature}")

    rewards_history = []      # combined reward (for logging continuity)
    strict_history = []       # fraction of samples that satisfy ALL constraints
    partial_history = []      # fraction of individual constraints satisfied
    rng = random.Random(SEED + 200)

    for iteration in range(args.num_iterations):
        # Get current model for sampling
        ckpt = tc.save_weights_for_sampler(
            name=f"{args.checkpoint_name}_iter{iteration}"
        ).result()
        sampling_client = sc.create_sampling_client(model_path=ckpt.path)

        all_datums = []
        iter_combined = []
        iter_strict = []
        iter_partial = []

        for _ in range(args.num_prompts_per_iter):
            prompt_text, inst_ids, kwargs_list = generate_constrained_prompt(
                rng,
                min_constraints=args.min_constraints,
                max_constraints=args.max_constraints,
            )

            # Build prompt tokens
            conversation = [{"role": "user", "content": prompt_text}]
            prompt_model_input = renderer.build_generation_prompt(conversation)
            prompt_tokens = list(prompt_model_input.to_ints())

            # Sample responses
            sampling_params = types.SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=1.0,
            )
            result = sampling_client.sample(
                prompt=prompt_model_input,
                sampling_params=sampling_params,
                num_samples=args.num_samples_per_prompt,
            ).result()

            # Compute rewards using the official IFEval checker
            sample_rewards = []
            sample_data = []
            for seq in result.sequences:
                response_tokens = list(seq.tokens)
                response_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(response_tokens)
                response_text = tokenizer.decode(response_tokens)
                partial, strict, combined, n_sat, n_tot = score_response(
                    response_text, inst_ids, kwargs_list, strict_bonus=args.strict_bonus
                )
                sample_rewards.append(combined)
                iter_partial.append(partial)
                iter_strict.append(strict)
                sample_data.append((response_tokens, response_logprobs))

            # GRPO advantage: normalize within the group of samples for this prompt
            mean_r = np.mean(sample_rewards)
            std_r = np.std(sample_rewards)
            if std_r > 1e-6:
                advantages = [(r - mean_r) / std_r for r in sample_rewards]
            else:
                # All samples got the same reward — shift so both directions are
                # explored. Center at the midpoint of the combined-reward range
                # [0, 1 + strict_bonus] so high uniform rewards still push up
                # while low uniform rewards push down.
                mid = (1.0 + args.strict_bonus) / 2.0
                advantages = [r - mid for r in sample_rewards]

            iter_combined.extend(sample_rewards)

            # Build datums
            for (response_tokens, response_logprobs), advantage in zip(sample_data, advantages):
                full_tokens = list(prompt_tokens) + response_tokens
                full_logprobs = [0.0] * len(prompt_tokens) + response_logprobs
                full_advs = [0.0] * len(prompt_tokens) + [float(advantage)] * len(response_tokens)

                if len(full_tokens) < 2:
                    continue
                input_tokens = full_tokens[:-1]
                target_tokens = full_tokens[1:]
                shifted_logprobs = full_logprobs[1:]
                shifted_advs = full_advs[1:]

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                        "logprobs": TensorData.from_torch(torch.tensor(shifted_logprobs, dtype=torch.float32)),
                        "advantages": TensorData.from_torch(torch.tensor(shifted_advs, dtype=torch.float32)),
                    },
                )
                all_datums.append(datum)

        # Train
        if len(all_datums) > 0:
            batch_size = 4
            for i in range(0, len(all_datums), batch_size):
                batch = all_datums[i:i + batch_size]
                tc.forward_backward(batch, loss_fn="importance_sampling").result()
                tc.optim_step(adam_params).result()

        avg_combined = float(np.mean(iter_combined)) if iter_combined else 0.0
        avg_strict = float(np.mean(iter_strict)) if iter_strict else 0.0
        avg_partial = float(np.mean(iter_partial)) if iter_partial else 0.0
        rewards_history.append(avg_combined)
        strict_history.append(avg_strict)
        partial_history.append(avg_partial)
        print(f"  Iter {iteration+1}/{args.num_iterations} | "
              f"prompt_strict(train): {avg_strict:.3f} | "
              f"inst_strict(train): {avg_partial:.3f} | "
              f"combined: {avg_combined:.3f} | "
              f"Datums: {len(all_datums)} | "
              f"Running strict: {np.mean(strict_history[-10:]):.3f}")

    # Save
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint: {checkpoint_path}")

    state_result = tc.save_state(args.checkpoint_name + "_state").result()
    state_path = state_result.path
    print(f"  State: {state_path}")

    if not args.no_publish:
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published!")

    info = {
        "checkpoint_path": checkpoint_path,
        "state_path": state_path,
        "base_model": MODEL,
        "rl_type": "ifeval_constraint_multi",
        "min_constraints": args.min_constraints,
        "max_constraints": args.max_constraints,
        "strict_bonus": args.strict_bonus,
        "final_prompt_strict": strict_history[-1] if strict_history else 0,
        "final_inst_strict": partial_history[-1] if partial_history else 0,
        "final_combined_reward": rewards_history[-1] if rewards_history else 0,
        "prompt_strict_history": strict_history,
        "inst_strict_history": partial_history,
        "rewards_history": rewards_history,
    }
    with open(os.path.join(EVAL_DIR, "checkpoint_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nEvaluate: python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL} --limit 100")


if __name__ == "__main__":
    main()
