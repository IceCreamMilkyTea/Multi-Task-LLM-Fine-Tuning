"""
DPO training CLI for Tinker.

Wraps `tinker_cookbook.preference.train_dpo.main` with a `train_and_publish.py`-style
CLI and the IF-focused Tulu-3 preference builder from `dpo_data.py`.

Usage (run DPO on the best 8B SFT checkpoint #51):

    python evaluation/dpo_train.py \
        --model meta-llama/Llama-3.1-8B \
        --load_checkpoint_path "tinker://...:_state" \
        --checkpoint_name exp_0422_8b_dpo_if \
        --rank 64 --num_samples 30000 --num_epochs 1 \
        --lr 5e-6 --beta 0.1 --max_length 2048 --batch_size 4

The final sampler-weights + state paths are written to
`evaluation/checkpoint_info.json` so `eval_all.py` picks them up automatically.
"""

import argparse
import json
import os
import pathlib

from tinker_cookbook import model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dpo_data import Tulu3IFPreferenceBuilder

MODEL_3B = "meta-llama/Llama-3.2-3B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_final_checkpoint(log_path: str) -> tuple[str | None, str | None]:
    """Read the last record of checkpoints.jsonl and return (sampler_path, state_path)."""
    jsonl = pathlib.Path(log_path) / "checkpoints.jsonl"
    if not jsonl.exists():
        return None, None
    last = None
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if not last:
        return None, None
    return last.get("sampler_path"), last.get("state_path")


def main():
    p = argparse.ArgumentParser(description="DPO on Tulu-3 IF preferences")

    p.add_argument("--model", type=str, default=MODEL_8B, choices=[MODEL_3B, MODEL_8B])
    p.add_argument("--load_checkpoint_path", type=str, required=True,
                   help="SFT state checkpoint to initialize DPO (weights only, fresh Adam)")
    p.add_argument("--checkpoint_name", type=str, default="exp_dpo_if",
                   help="Run name; also used for wandb + log directory")

    # Data
    p.add_argument("--num_samples", type=int, default=30000,
                   help="Max IF preference pairs (filter yields up to ~65k)")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)

    # DPO / optim
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap training steps; default = num_epochs * num_batches")
    p.add_argument("--lr", type=float, default=5e-6,
                   help="DPO LR; keep well below SFT (which used 5e-5)")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL strength)")
    p.add_argument("--rank", type=int, default=64,
                   help="LoRA rank; match the SFT checkpoint's rank (8B best = 64)")
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--log_path", type=str, default=None,
                   help="Log dir for checkpoints.jsonl + trace; default /tmp/dpo_<name>")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    args = p.parse_args()

    renderer_name = model_info.get_recommended_renderer_name(args.model)
    log_path = args.log_path or f"/tmp/dpo_{args.checkpoint_name}"
    os.makedirs(log_path, exist_ok=True)

    # Data pipeline
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    comparison_builder = Tulu3IFPreferenceBuilder(num_samples=args.num_samples)
    dataset_builder = DPODatasetBuilderFromComparisons(
        comparison_builder=comparison_builder,
        common_config=common_config,
    )

    config = train_dpo.Config(
        log_path=log_path,
        model_name=args.model,
        dataset_builder=dataset_builder,
        load_checkpoint_path=args.load_checkpoint_path,
        renderer_name=renderer_name,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.num_epochs,
        dpo_beta=args.beta,
        lora_rank=args.rank,
        save_every=args.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        adam_beta2=args.adam_beta2,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or args.checkpoint_name,
        max_steps=args.max_steps,
    )

    print(f"Model: {args.model}")
    print(f"Renderer: {renderer_name}")
    print(f"Load from (SFT state): {args.load_checkpoint_path}")
    print(f"Log dir: {log_path}")
    print(f"DPO: beta={args.beta}, lr={args.lr}, rank={args.rank}, epochs={args.num_epochs}, "
          f"batch_size={args.batch_size}, max_length={args.max_length}")

    train_dpo.main(config)

    # After training: surface the final sampler_path + state_path so eval_all.py works
    sampler_path, state_path = _read_final_checkpoint(log_path)
    print("\n" + "=" * 60)
    if sampler_path:
        print(f"Final sampler weights: {sampler_path}")
    if state_path:
        print(f"Final training state:  {state_path}")
    print("=" * 60)

    # Write checkpoint_info.json in the same shape train_and_publish.py uses,
    # so eval_all.py auto-discovers the new checkpoint.
    info = {
        "checkpoint_path": sampler_path,
        "state_path": state_path,
        "base_model": args.model,
        "renderer_name": renderer_name,
        "training": {
            "kind": "dpo",
            "dpo_beta": args.beta,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_samples": args.num_samples,
            "data_sources": sorted(list({"allenai/tulu-3-sft-reused-if"})),
        },
        "resumed_from": args.load_checkpoint_path,
    }
    with open(os.path.join(EVAL_DIR, "checkpoint_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nWrote {EVAL_DIR}/checkpoint_info.json")

    if sampler_path:
        print(f"\nEvaluate with:")
        print(f"  python evaluation/eval_all.py \\")
        print(f'    --checkpoint_path "{sampler_path}" \\')
        print(f"    --base_model {args.model} --limit 300")


if __name__ == "__main__":
    main()
