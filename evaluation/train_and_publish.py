"""
Train a model on multi-task data (GSM8K, Tulu-3 SFT, OpenCodeInstruct),
save checkpoint, and publish it.

Usage:
    python evaluation/train_and_publish.py --checkpoint_name exp_01_baseline
    python evaluation/train_and_publish.py --num_steps 200 --checkpoint_name exp_02
    python evaluation/train_and_publish.py --no_publish
"""

import argparse
import json
import os
import random

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42


def load_gsm8k_conversations(num_samples=7473):
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
    return conversations


def load_tulu3_conversations(num_samples=5000):
    """Load Tulu-3 SFT mixture and format as conversations."""
    print(f"  Loading Tulu-3 SFT mixture (up to {num_samples} samples)...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    conversations = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
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
    return conversations


def load_code_conversations(num_samples=5000):
    """Load OpenCodeInstruct and format as conversations."""
    print(f"  Loading OpenCodeInstruct (up to {num_samples} samples)...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    conversations = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        input_text = example["input"]
        output_text = example["output"]
        if input_text and output_text:
            conversations.append([
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ])

    print(f"    Loaded {len(conversations)} code conversations")
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="experiment", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    # Dataset sizes
    parser.add_argument("--gsm8k_samples", type=int, default=7473, help="Number of GSM8K samples")
    parser.add_argument("--tulu_samples", type=int, default=5000, help="Number of Tulu-3 samples")
    parser.add_argument("--code_samples", type=int, default=5000, help="Number of code samples")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load datasets
    print("Loading training data...")
    gsm8k_convos = load_gsm8k_conversations(args.gsm8k_samples)
    tulu_convos = load_tulu3_conversations(args.tulu_samples)
    code_convos = load_code_conversations(args.code_samples)

    # Combine all conversations
    all_convos = gsm8k_convos + tulu_convos + code_convos
    random.seed(SEED)
    random.shuffle(all_convos)
    print(f"Total conversations: {len(all_convos)} "
          f"(GSM8K: {len(gsm8k_convos)}, Tulu: {len(tulu_convos)}, Code: {len(code_convos)})")

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
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
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

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

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
