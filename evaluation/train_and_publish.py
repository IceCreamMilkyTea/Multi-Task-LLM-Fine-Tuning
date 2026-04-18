"""
Train a model on multi-task data (GSM8K, Tulu-3 SFT, OpenCodeInstruct),
save checkpoint, and publish it.

Supports advanced methods:
  --filter_quality     Enable data quality filtering
  --curriculum         Enable curriculum learning (easy → hard)
  --stage2_from        Resume from checkpoint for stage 2 (focused training)

Usage:
    python evaluation/train_and_publish.py --checkpoint_name exp_01_baseline
    python evaluation/train_and_publish.py --num_steps 200 --checkpoint_name exp_02
    python evaluation/train_and_publish.py --filter_quality --curriculum --checkpoint_name exp_03
    python evaluation/train_and_publish.py --no_publish
"""

import argparse
import ast
import json
import os
import random
import re

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission
MODEL_3B = "meta-llama/Llama-3.2-3B"
MODEL_8B = "meta-llama/Llama-3.1-8B"

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42


# ============================================================
# Data Quality Filtering
# ============================================================

def filter_gsm8k_quality(conversations):
    """Filter GSM8K for clean step-by-step solutions.

    Keeps examples that:
    - Have a clear #### final answer marker
    - Have at least 2 reasoning steps
    - Are not excessively long (likely garbled)
    - Have a numeric final answer
    """
    filtered = []
    for convo in conversations:
        answer = convo[1]["content"]
        # Must have #### final answer marker
        if "####" not in answer:
            continue
        # Extract final answer — must be numeric
        parts = answer.split("####")
        final_ans = parts[-1].strip()
        if not re.search(r'\d', final_ans):
            continue
        # Must have at least 2 reasoning lines before the answer
        reasoning = parts[0].strip()
        lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        # Not too long (likely garbled or overly complex for the model)
        if len(answer) > 2000:
            continue
        # Not too short (likely missing reasoning)
        if len(reasoning) < 50:
            continue
        filtered.append(convo)
    return filtered


def filter_code_quality(conversations):
    """Filter code examples by syntax validity and quality.

    Keeps examples that:
    - Have substantial output (not trivially short)
    - Are not excessively long (hard to learn from)
    - Contain actual code patterns (def, class, import, etc.)
    - For Python: have valid syntax (parseable by ast)
    """
    filtered = []
    for convo in conversations:
        output = convo[1]["content"]
        prompt = convo[0]["content"]
        # Check minimum length
        if len(output) < 50:
            continue
        # Check maximum length
        if len(output) > 4000:
            continue
        # Must contain code-like patterns
        code_indicators = ["def ", "class ", "import ", "return ", "for ", "while ", "print("]
        has_code = any(indicator in output for indicator in code_indicators)
        if not has_code:
            continue
        # Try to extract and validate Python code blocks
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
        if code_blocks:
            # Validate at least one code block parses
            any_valid = False
            for block in code_blocks:
                try:
                    ast.parse(block)
                    any_valid = True
                    break
                except SyntaxError:
                    continue
            if not any_valid and len(code_blocks) > 0:
                continue
        filtered.append(convo)
    return filtered


def filter_tulu_quality(conversations):
    """Filter Tulu conversations by quality.

    Keeps examples that:
    - Are primarily English (ASCII ratio >= 85%)
    - Have at least one substantial assistant response (>= 100 chars)
    - Are not excessively long (would be truncated at max_length)
    - Are not trivially short
    - Have proper alternating user/assistant structure
    """
    filtered = []
    for convo in conversations:
        # Must start with user and have proper structure
        if not convo or convo[0]["role"] != "user":
            continue
        # English language check: high ASCII ratio in user message
        user_text = convo[0]["content"]
        ascii_chars = sum(1 for c in user_text if ord(c) < 128)
        if ascii_chars / max(len(user_text), 1) < 0.85:
            continue
        # Must have at least one substantial assistant response
        has_good_response = False
        for msg in convo:
            if msg["role"] == "assistant" and len(msg["content"]) >= 100:
                has_good_response = True
                break
        if not has_good_response:
            continue
        # Total conversation should not be too long (hard to learn, gets truncated)
        total_len = sum(len(msg["content"]) for msg in convo)
        if total_len > 3000:
            continue
        # Filter out very short conversations
        if total_len < 150:
            continue
        filtered.append(convo)
    return filtered


# ============================================================
# IFEval Data Augmentation
# ============================================================

IFEVAL_CONSTRAINT_TEMPLATES = [
    ("Your response must contain exactly {n} paragraphs. Paragraphs are separated by two newlines.", lambda n: "\n\n".join(["This is a paragraph about the topic." for _ in range(n)]), [2, 3, 4, 5]),
    ("Your entire response should be in English, and in all capital letters.", lambda _: "THIS IS MY RESPONSE IN ALL CAPITAL LETTERS.", [None]),
    ("Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.", lambda _: "this is my response in all lowercase letters.", [None]),
    ("Include exactly {n} bullet points in your response. Use the markdown bullet points such as: * This is a bullet point.", lambda n: "\n".join([f"* Bullet point {i+1}" for i in range(n)]), [3, 4, 5, 6]),
    ("Your response must contain at least {n} sentences.", lambda n: " ".join(["This is a sentence." for _ in range(n)]), [3, 5, 8, 10]),
    ("Wrap your entire response with double quotation marks.", lambda _: '"Here is my response wrapped in quotation marks."', [None]),
    ("Do not include keywords '{word}' in the response.", lambda w: "Here is my response without that specific word.", ["the", "is", "and"]),
    ("Your response should contain {n} or fewer sentences.", lambda n: " ".join(["Short answer." for _ in range(min(n, 3))]), [2, 3, 5]),
    ("Answer with at least {n} words.", lambda n: " ".join(["word" for _ in range(n + 5)]), [50, 100, 200]),
    ("Finish your response with the exact phrase: Is there anything else I can help with?", lambda _: "Here is my answer. Is there anything else I can help with?", [None]),
    ("In your response, the word '{word}' should appear at least {n} times.", lambda args: f"The {args[0]} is important. The {args[0]} matters. " * args[1], [("answer", 2), ("response", 3)]),
    ("Your response must have {n} sections. Mark the beginning of each section with 'SECTION {i}'.", lambda n: "\n\n".join([f"SECTION {i+1}\nContent for section {i+1}." for i in range(n)]), [2, 3, 4]),
]

IFEVAL_TOPICS = [
    "Explain the water cycle",
    "Describe the benefits of exercise",
    "Write about the history of computers",
    "Explain how photosynthesis works",
    "Describe the solar system",
    "Write about healthy eating habits",
    "Explain the importance of recycling",
    "Describe how airplanes fly",
    "Write about the role of teamwork",
    "Explain how the internet works",
    "Describe the life cycle of a butterfly",
    "Write about different types of energy",
    "Explain what machine learning is",
    "Describe the process of making bread",
    "Write about the importance of sleep",
    "Explain how vaccines work",
    "Describe the water treatment process",
    "Write about the history of music",
    "Explain the greenhouse effect",
    "Describe how electric cars work",
]


def generate_ifeval_augmented_data(num_samples=500):
    """Generate synthetic IFEval-style training data with explicit constraints."""
    import random as _rng
    _rng.seed(SEED + 100)

    conversations = []
    for i in range(num_samples):
        topic = _rng.choice(IFEVAL_TOPICS)
        template, response_fn, param_options = _rng.choice(IFEVAL_CONSTRAINT_TEMPLATES)
        param = _rng.choice(param_options)

        if param is None:
            constraint = template
            response = response_fn(None)
        elif isinstance(param, tuple):
            constraint = template.format(word=param[0], n=param[1])
            response = response_fn(param)
        elif isinstance(param, str):
            constraint = template.format(word=param)
            response = response_fn(param)
        else:
            constraint = template.format(n=param, i="X")
            response = response_fn(param)

        prompt = f"{topic}. {constraint}"
        conversations.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])

    print(f"  Generated {len(conversations)} IFEval-augmented conversations")
    return conversations


# ============================================================
# Curriculum Learning
# ============================================================

def gsm8k_difficulty(convo):
    """Score difficulty of a GSM8K problem. Lower = easier."""
    answer = convo[1]["content"]
    parts = answer.split("####")
    reasoning = parts[0] if len(parts) > 1 else answer
    # Count reasoning steps (non-empty lines)
    steps = len([l for l in reasoning.split("\n") if l.strip()])
    return steps


def code_difficulty(convo):
    """Score difficulty of a code problem. Lower = easier."""
    return len(convo[1]["content"])


def tulu_difficulty(convo):
    """Score difficulty of a Tulu conversation. Lower = easier."""
    return sum(len(msg["content"]) for msg in convo)


def sort_curriculum(gsm8k_convos, code_convos, tulu_convos):
    """Sort each dataset by difficulty (easy → hard), then interleave in curriculum order.

    Creates 3 difficulty tiers (easy/medium/hard) and mixes within each tier.
    This ensures the model sees easy examples from all tasks first.
    """
    # Sort each dataset by difficulty
    gsm8k_sorted = sorted(gsm8k_convos, key=gsm8k_difficulty)
    code_sorted = sorted(code_convos, key=code_difficulty)
    tulu_sorted = sorted(tulu_convos, key=tulu_difficulty)

    def split_thirds(lst):
        n = len(lst)
        return lst[:n//3], lst[n//3:2*n//3], lst[2*n//3:]

    gsm8k_tiers = split_thirds(gsm8k_sorted)
    code_tiers = split_thirds(code_sorted)
    tulu_tiers = split_thirds(tulu_sorted)

    # Build curriculum: easy tier (shuffled), then medium, then hard
    all_data = []
    for tier_idx in range(3):
        tier_data = list(gsm8k_tiers[tier_idx]) + list(code_tiers[tier_idx]) + list(tulu_tiers[tier_idx])
        random.shuffle(tier_data)
        all_data.extend(tier_data)

    return all_data


# ============================================================
# Data Loading (with optional filtering)
# ============================================================

def load_gsm8k_conversations(num_samples=7473, filter_quality=False):
    """Load GSM8K train split and format as conversations."""
    print(f"  Loading GSM8K (up to {num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if num_samples < len(ds):
        ds = ds.shuffle(seed=SEED).select(range(num_samples))

    conversations = []
    for example in ds:
        question = example["question"]
        answer = example["answer"]
        conversations.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
    print(f"    Loaded {len(conversations)} GSM8K conversations")

    if filter_quality:
        conversations = filter_gsm8k_quality(conversations)
        print(f"    After quality filtering: {len(conversations)} GSM8K conversations")

    return conversations


def load_tulu3_conversations(num_samples=5000, filter_quality=False, skip_first=0):
    """Load Tulu-3 SFT mixture and format as conversations.

    Args:
        skip_first: Skip this many samples from the start. Useful to get diverse sources
                    (first 5k are oasst1, after that mostly flan_v2 which is better for IFEval).
    """
    # Load extra samples if filtering, since we'll discard some
    load_count = int(num_samples * 1.5) if filter_quality else num_samples
    total_to_scan = load_count + skip_first
    print(f"  Loading Tulu-3 SFT mixture (skipping {skip_first}, up to {load_count} samples, target {num_samples})...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    conversations = []
    for i, example in enumerate(ds):
        if i >= total_to_scan:
            break
        if i < skip_first:
            continue
        messages = example["messages"]
        # Filter: must have at least user + assistant
        if len(messages) >= 2:
            convo = []
            for msg in messages:
                if msg["role"] in ("user", "assistant"):
                    convo.append({"role": msg["role"], "content": msg["content"]})
            if convo and convo[0]["role"] == "user":
                conversations.append(convo)

    print(f"    Loaded {len(conversations)} Tulu-3 conversations")

    if filter_quality:
        conversations = filter_tulu_quality(conversations)
        print(f"    After quality filtering: {len(conversations)} Tulu-3 conversations")
        # Trim to target
        if len(conversations) > num_samples:
            conversations = conversations[:num_samples]

    return conversations


def load_code_conversations(num_samples=5000, filter_quality=False):
    """Load OpenCodeInstruct and format as conversations.

    When filter_quality=True, uses the dataset's average_test_score field
    to only keep examples where the generated code passes >= 80% of unit tests.
    """
    # Load extra samples if filtering, since we'll discard some
    load_count = int(num_samples * 2) if filter_quality else num_samples
    print(f"  Loading OpenCodeInstruct (up to {load_count} samples, target {num_samples})...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    conversations = []
    for i, example in enumerate(ds):
        if i >= load_count:
            break
        input_text = example["input"]
        output_text = example["output"]
        if not input_text or not output_text:
            continue

        # Quality filtering using test scores from the dataset
        if filter_quality:
            test_score_raw = example.get("average_test_score", "0")
            try:
                test_score = float(test_score_raw) if test_score_raw else 0.0
            except (ValueError, TypeError):
                test_score = 0.0
            if test_score < 0.8:
                continue
            # Also length filter: not too short, not too long
            if len(output_text) < 50 or len(output_text) > 4000:
                continue

        conversations.append([
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ])

    print(f"    Loaded {len(conversations)} code conversations")

    # Trim to target
    if len(conversations) > num_samples:
        conversations = conversations[:num_samples]

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="experiment", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a save_state checkpoint (tinker:// path)")
    # Dataset sizes
    parser.add_argument("--gsm8k_samples", type=int, default=7473, help="Number of GSM8K samples")
    parser.add_argument("--tulu_samples", type=int, default=10000, help="Number of Tulu-3 samples")
    parser.add_argument("--code_samples", type=int, default=5000, help="Number of code samples")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--tulu_skip", type=int, default=0,
                        help="Skip first N Tulu samples to get diverse sources (flan_v2 starts ~5k)")
    # Advanced methods
    parser.add_argument("--filter_quality", action="store_true",
                        help="Enable data quality filtering (removes low-quality samples)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (easy → hard ordering)")
    parser.add_argument("--stage2_task", type=str, default=None, choices=["tulu", "gsm8k", "code"],
                        help="Stage 2: train primarily on this task (use with --resume_from)")
    parser.add_argument("--stage2_ratio", type=float, default=0.7,
                        help="Stage 2: fraction of data from the focused task (default: 0.7)")
    parser.add_argument("--ifeval_augment", type=int, default=0,
                        help="Number of synthetic IFEval-style augmented samples to add (0=disabled)")
    parser.add_argument("--model", type=str, default=MODEL_3B,
                        choices=[MODEL_3B, MODEL_8B],
                        help="Base model: 3B (default) or 8B (for final runs)")
    args = parser.parse_args()

    # Override MODEL if specified
    global MODEL
    MODEL = args.model

    # Setup
    print(f"Model: {MODEL}")
    print(f"Advanced methods: filter_quality={args.filter_quality}, curriculum={args.curriculum}, "
          f"stage2_task={args.stage2_task}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load datasets (with optional quality filtering)
    print("Loading training data...")
    gsm8k_convos = load_gsm8k_conversations(args.gsm8k_samples, filter_quality=args.filter_quality)
    tulu_convos = load_tulu3_conversations(args.tulu_samples, filter_quality=args.filter_quality,
                                              skip_first=args.tulu_skip)
    code_convos = load_code_conversations(args.code_samples, filter_quality=args.filter_quality)

    # Multi-stage training: Stage 2 rebalances data toward the weakest task
    if args.stage2_task:
        total_target = len(gsm8k_convos) + len(tulu_convos) + len(code_convos)
        focus_ratio = args.stage2_ratio
        other_ratio = (1.0 - focus_ratio) / 2.0
        focus_count = int(total_target * focus_ratio)
        other_count = int(total_target * other_ratio)

        task_map = {"gsm8k": gsm8k_convos, "tulu": tulu_convos, "code": code_convos}
        task_names = ["gsm8k", "tulu", "code"]

        for name in task_names:
            if name == args.stage2_task:
                # Upsample or keep the focus task
                convos = task_map[name]
                if len(convos) < focus_count:
                    # Upsample by repeating
                    repeats = (focus_count // len(convos)) + 1
                    convos = (convos * repeats)[:focus_count]
                else:
                    random.seed(SEED)
                    random.shuffle(convos)
                    convos = convos[:focus_count]
                task_map[name] = convos
            else:
                # Downsample other tasks
                convos = task_map[name]
                random.seed(SEED)
                random.shuffle(convos)
                task_map[name] = convos[:other_count]

        gsm8k_convos = task_map["gsm8k"]
        tulu_convos = task_map["tulu"]
        code_convos = task_map["code"]
        print(f"Stage 2 rebalanced (focus={args.stage2_task}, ratio={focus_ratio}): "
              f"GSM8K={len(gsm8k_convos)}, Tulu={len(tulu_convos)}, Code={len(code_convos)}")

    # IFEval data augmentation
    ifeval_augmented = []
    if args.ifeval_augment > 0:
        ifeval_augmented = generate_ifeval_augmented_data(args.ifeval_augment)

    # Combine: curriculum (easy→hard) or random shuffle
    if args.curriculum:
        print("Applying curriculum learning (easy → hard)...")
        all_convos = sort_curriculum(gsm8k_convos, code_convos, tulu_convos) + ifeval_augmented
    else:
        all_convos = gsm8k_convos + tulu_convos + code_convos + ifeval_augmented
        random.seed(SEED)
        random.shuffle(all_convos)

    print(f"Total conversations: {len(all_convos)} "
          f"(GSM8K: {len(gsm8k_convos)}, Tulu: {len(tulu_convos)}, Code: {len(code_convos)}, "
          f"IFEval-aug: {len(ifeval_augmented)})")

    # Convert to training data
    print("Preparing training data...")
    all_data = []
    skipped = 0
    for convo in all_convos:
        try:
            datum = conversation_to_datum(
                convo, renderer, max_length=args.max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            all_data.append(datum)
        except Exception as e:
            skipped += 1
    print(f"  {len(all_data)} training examples prepared ({skipped} skipped)")

    # Create training client
    sc = tinker.ServiceClient()
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        tc = sc.create_training_client_from_state(args.resume_from)
    else:
        print(f"Creating LoRA training client (rank={args.rank})...")
        tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        # Sample a random batch from all_data
        batch = [all_data[i % len(all_data)] for i in
                 random.sample(range(len(all_data)), min(args.batch_size, len(all_data)))]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

    # Save checkpoint (inference weights)
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Save training state (for resuming training later)
    print(f"Saving training state '{args.checkpoint_name}_state'...")
    state_result = tc.save_state(args.checkpoint_name + "_state").result()
    state_path = state_result.path
    print(f"  Training state saved: {state_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "state_path": state_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "gsm8k_samples": len(gsm8k_convos),
            "tulu_samples": len(tulu_convos),
            "code_samples": len(code_convos),
            "total_samples": len(all_data),
            "filter_quality": args.filter_quality,
            "curriculum": args.curriculum,
            "stage2_task": args.stage2_task,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()
