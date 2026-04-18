"""
IFEval Constraint RL — GRPO-style RL with programmatic constraint checking.

Instead of training on math correctness (GSM8K RL), this trains the model
to follow explicit formatting constraints like those in IFEval:
- Write in all caps / all lowercase
- Include exactly N bullet points / paragraphs / sentences
- Wrap in quotes, add postscript, section markers
- Word count constraints

The reward is 1 if ALL constraints are satisfied, 0 otherwise.
This teaches the model to follow arbitrary formatting instructions.

Usage:
    python evaluation/rl_ifeval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --resume_from "tinker://..._state" \\
        --num_iterations 30 \\
        --checkpoint_name exp_ifeval_rl
"""

import argparse
import json
import os
import random
import re

import numpy as np
import tinker
import torch
from tinker import types
from tinker.types import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_3B = "meta-llama/Llama-3.2-3B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# ============================================================
# Constraint definitions and checkers
# ============================================================

CONSTRAINTS = [
    {
        "template": "Your entire response should be in English, and in all capital letters.",
        "check": lambda r: r == r.upper() and len(r) > 20,
        "params": None,
    },
    {
        "template": "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
        "check": lambda r: r == r.lower() and len(r) > 20,
        "params": None,
    },
    {
        "template": "Your response must contain exactly {n} paragraphs. Paragraphs are separated by two newlines.",
        "check": lambda r, n: len([p for p in r.split('\n\n') if p.strip()]) == n,
        "params": [2, 3, 4, 5],
    },
    {
        "template": "Include exactly {n} bullet points in your response. Use the markdown bullet points such as: * This is a bullet point.",
        "check": lambda r, n: len(re.findall(r'^\* ', r, re.MULTILINE)) == n,
        "params": [3, 4, 5, 6],
    },
    {
        "template": "Wrap your entire response with double quotation marks.",
        "check": lambda r: r.strip().startswith('"') and r.strip().endswith('"'),
        "params": None,
    },
    {
        "template": "Finish your response with the exact phrase: Is there anything else I can help with?",
        "check": lambda r: r.strip().endswith("Is there anything else I can help with?"),
        "params": None,
    },
    {
        "template": "At the end of your response, please explicitly add a postscript starting with P.S.",
        "check": lambda r: "P.S." in r or "P.S " in r,
        "params": None,
    },
    {
        "template": "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
        "check": lambda r: "<<" in r and ">>" in r,
        "params": None,
    },
    {
        "template": "Your response must have {n} sections. Mark the beginning of each section with SECTION X.",
        "check": lambda r, n: len(re.findall(r'SECTION \d', r)) >= n,
        "params": [2, 3, 4],
    },
    {
        "template": "Give two different responses. Responses and only responses should be separated by 6 asterisks: ******.",
        "check": lambda r: "******" in r,
        "params": None,
    },
    {
        "template": "Answer with at least {n} words.",
        "check": lambda r, n: len(r.split()) >= n,
        "params": [50, 100, 200],
    },
    {
        "template": "Your response should contain {n} or fewer sentences.",
        "check": lambda r, n: len([s for s in re.split(r'[.!?]+', r) if s.strip()]) <= n,
        "params": [2, 3, 5],
    },
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


def check_constraint(response, constraint, param):
    """Check if a response satisfies a constraint. Returns reward (0 or 1)."""
    try:
        if param is None:
            return 1.0 if constraint["check"](response) else 0.0
        else:
            return 1.0 if constraint["check"](response, param) else 0.0
    except Exception:
        return 0.0


def generate_constrained_prompt(rng):
    """Generate a random prompt with a constraint."""
    topic = rng.choice(TOPICS)
    constraint = rng.choice(CONSTRAINTS)
    params = constraint["params"]
    param = rng.choice(params) if params else None

    if param is not None:
        constraint_text = constraint["template"].format(n=param)
    else:
        constraint_text = constraint["template"]

    prompt = f"{topic}. {constraint_text}"
    return prompt, constraint, param


def main():
    parser = argparse.ArgumentParser(description="IFEval Constraint RL")
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
    args = parser.parse_args()

    MODEL = args.model
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    sc = tinker.ServiceClient()
    print(f"Resuming from: {args.resume_from}")
    tc = sc.create_training_client_from_state(args.resume_from)
    print("Training client ready")

    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    print(f"\nIFEval Constraint RL: {args.num_iterations} iterations")
    print(f"  Prompts per iter: {args.num_prompts_per_iter}")
    print(f"  Samples per prompt: {args.num_samples_per_prompt}")
    print(f"  LR: {args.lr}, temperature: {args.temperature}")

    rewards_history = []
    rng = random.Random(SEED + 200)

    for iteration in range(args.num_iterations):
        # Get current model for sampling
        ckpt = tc.save_weights_for_sampler(
            name=f"{args.checkpoint_name}_iter{iteration}"
        ).result()
        sampling_client = sc.create_sampling_client(model_path=ckpt.path)

        all_datums = []
        iter_rewards = []

        for _ in range(args.num_prompts_per_iter):
            prompt_text, constraint, param = generate_constrained_prompt(rng)

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

            # Compute rewards
            sample_rewards = []
            sample_data = []
            for seq in result.sequences:
                response_tokens = list(seq.tokens)
                response_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(response_tokens)
                response_text = tokenizer.decode(response_tokens)
                reward = check_constraint(response_text, constraint, param)
                sample_rewards.append(reward)
                sample_data.append((response_tokens, response_logprobs))

            # Compute advantages
            mean_r = np.mean(sample_rewards)
            std_r = np.std(sample_rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in sample_rewards]

            if np.std(sample_rewards) < 1e-6:
                iter_rewards.extend(sample_rewards)
                continue

            iter_rewards.extend(sample_rewards)

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

        avg_reward = np.mean(iter_rewards) if iter_rewards else 0.0
        rewards_history.append(avg_reward)
        print(f"  Iter {iteration+1}/{args.num_iterations} | "
              f"Constraint satisfaction: {avg_reward:.3f} | "
              f"Datums: {len(all_datums)} | "
              f"Running avg: {np.mean(rewards_history[-10:]):.3f}")

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
        "rl_type": "ifeval_constraint",
        "final_constraint_satisfaction": rewards_history[-1] if rewards_history else 0,
        "rewards_history": rewards_history,
    }
    with open(os.path.join(EVAL_DIR, "checkpoint_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nEvaluate: python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL} --limit 100")


if __name__ == "__main__":
    main()
