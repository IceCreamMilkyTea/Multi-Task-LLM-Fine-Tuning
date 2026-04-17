"""
GRPO-style Reinforcement Learning training on GSM8K.

Samples N responses per problem, computes rewards (1 if answer correct, 0 otherwise),
computes group-relative advantages, and trains with importance_sampling loss.

Usage:
    python evaluation/rl_train.py \\
        --resume_from "tinker://..._state" \\
        --num_iterations 50 \\
        --checkpoint_name exp_rl_gsm8k
"""

import argparse
import json
import os
import random
import re

import numpy as np
import tinker
import torch
from datasets import load_dataset
from tinker import types
from tinker.types import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.2-3B"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def extract_answer(text):
    """Extract the final numeric answer from a model response."""
    # Match #### X format (GSM8K standard)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def compute_reward(response, ground_truth_answer):
    """Reward = 1 if extracted answer matches ground truth, 0 otherwise."""
    predicted = extract_answer(response)
    if predicted is None:
        return 0.0
    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth_answer)
        if abs(pred_num - gt_num) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        # String comparison fallback
        if str(predicted).strip() == str(ground_truth_answer).strip():
            return 1.0
    return 0.0


def extract_gt_answer(answer_text):
    """Extract ground truth answer from GSM8K formatted answer."""
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).strip()
    return None


def load_gsm8k_problems(num_samples=500):
    """Load GSM8K problems with ground truth answers."""
    print(f"Loading {num_samples} GSM8K problems...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))

    problems = []
    for example in ds:
        gt = extract_gt_answer(example["answer"])
        if gt is not None:
            problems.append({
                "question": example["question"],
                "answer": example["answer"],
                "gt_answer": gt,
            })
    print(f"  Loaded {len(problems)} problems with valid GT answers")
    return problems


def build_prompt_tokens(question, renderer, tokenizer):
    """Build input tokens for a question using the renderer."""
    conversation = [{"role": "user", "content": question}]
    model_input = renderer.build_generation_prompt(conversation)
    tokens = list(model_input.to_ints())
    return tokens, model_input


def main():
    parser = argparse.ArgumentParser(description="GRPO-style RL training on GSM8K")
    parser.add_argument("--resume_from", type=str, required=True,
                        help="Resume from training state checkpoint (tinker://..._state)")
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of RL iterations")
    parser.add_argument("--num_problems_per_iter", type=int, default=8,
                        help="Number of unique problems sampled per iteration")
    parser.add_argument("--num_samples_per_problem", type=int, default=4,
                        help="Number of response samples per problem (group size)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate for RL (should be low)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to sample per response")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for diversity")
    parser.add_argument("--checkpoint_name", type=str, default="exp_rl_gsm8k",
                        help="Checkpoint name")
    parser.add_argument("--num_train_problems", type=int, default=500,
                        help="Number of GSM8K problems to sample from")
    parser.add_argument("--no_publish", action="store_true")
    args = parser.parse_args()

    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load problems
    problems = load_gsm8k_problems(args.num_train_problems)

    # Create training client from resumed state
    sc = tinker.ServiceClient()
    print(f"Resuming from: {args.resume_from}")
    tc = sc.create_training_client_from_state(args.resume_from)
    print("Training client ready")

    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    print(f"\nRL training: {args.num_iterations} iterations")
    print(f"  Problems per iter: {args.num_problems_per_iter}")
    print(f"  Samples per problem: {args.num_samples_per_problem}")
    print(f"  LR: {args.lr}, temperature: {args.temperature}")

    # Stats tracking
    rewards_history = []

    random.seed(SEED)
    for iteration in range(args.num_iterations):
        # Sample problems for this iteration
        batch_problems = random.sample(problems, args.num_problems_per_iter)

        # For each problem, get sampling client and generate responses
        # First need a sampling client — use save_weights_for_sampler
        ckpt = tc.save_weights_for_sampler(
            name=f"{args.checkpoint_name}_iter{iteration}"
        ).result()
        sampling_client = sc.create_sampling_client(model_path=ckpt.path)

        # Sample responses for each problem
        all_datums = []
        iter_rewards = []

        for prob in batch_problems:
            prompt_tokens, prompt_model_input = build_prompt_tokens(
                prob["question"], renderer, tokenizer
            )

            # Sample G responses
            sampling_params = types.SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=1.0,
                stop=None,
            )

            result = sampling_client.sample(
                prompt=prompt_model_input,
                sampling_params=sampling_params,
                num_samples=args.num_samples_per_problem,
            ).result()

            # Compute rewards and collect samples
            sample_rewards = []
            sample_data = []
            for seq in result.sequences:
                response_tokens = list(seq.tokens)
                response_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(response_tokens)
                response_text = tokenizer.decode(response_tokens)
                reward = compute_reward(response_text, prob["gt_answer"])
                sample_rewards.append(reward)
                sample_data.append((response_tokens, response_logprobs))

            # Compute advantages (group-relative)
            mean_r = np.mean(sample_rewards)
            std_r = np.std(sample_rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in sample_rewards]

            # Skip if all rewards are same (no learning signal)
            if np.std(sample_rewards) < 1e-6:
                iter_rewards.extend(sample_rewards)
                continue

            iter_rewards.extend(sample_rewards)

            # Build datums for training — importance_sampling needs: target_tokens, logprobs, advantages
            for (response_tokens, response_logprobs), advantage in zip(sample_data, advantages):
                full_tokens = list(prompt_tokens) + response_tokens
                # logprobs: 0 for prompt tokens, actual logprobs for response
                full_logprobs = [0.0] * len(prompt_tokens) + response_logprobs
                # advantages: 0 for prompt (ignored), advantage for response
                full_advs = [0.0] * len(prompt_tokens) + [float(advantage)] * len(response_tokens)

                if len(full_tokens) < 2:
                    continue
                # Shift: input = tokens[:-1], targets = tokens[1:]
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

        # Train on collected datums in mini-batches
        if len(all_datums) > 0:
            batch_size = 4
            for i in range(0, len(all_datums), batch_size):
                batch = all_datums[i:i + batch_size]
                tc.forward_backward(batch, loss_fn="importance_sampling").result()
                tc.optim_step(adam_params).result()

        avg_reward = np.mean(iter_rewards) if iter_rewards else 0.0
        rewards_history.append(avg_reward)

        print(f"  Iter {iteration+1}/{args.num_iterations} | "
              f"Avg reward: {avg_reward:.3f} | "
              f"Datums: {len(all_datums)} | "
              f"Running avg (last 10): {np.mean(rewards_history[-10:]):.3f}")

    # Save final checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    print(f"Saving training state '{args.checkpoint_name}_state'...")
    state_result = tc.save_state(args.checkpoint_name + "_state").result()
    state_path = state_result.path
    print(f"  Training state saved: {state_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published!")

    # Save info
    info = {
        "checkpoint_path": checkpoint_path,
        "state_path": state_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "rl_training": {
            "resumed_from": args.resume_from,
            "num_iterations": args.num_iterations,
            "num_problems_per_iter": args.num_problems_per_iter,
            "num_samples_per_problem": args.num_samples_per_problem,
            "lr": args.lr,
            "final_avg_reward": rewards_history[-1] if rewards_history else 0.0,
            "rewards_history": rewards_history,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nEvaluate with:")
    print(f"  python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL} --limit 100")


if __name__ == "__main__":
    main()
