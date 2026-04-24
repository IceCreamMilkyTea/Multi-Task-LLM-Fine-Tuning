"""
Run a resumable DPO experiment on Tulu preference data using Tinker cookbook.

Example:
    python evaluation/run_tulu_dpo.py \\
        --load_checkpoint_path "tinker://..._state" \\
        --log_path ~/dpo_runs/tulu_dpo_from_best_8b
"""

import argparse

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import Tulu38BComparisonBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config(args: argparse.Namespace) -> train_dpo.Config:
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=args.model_name,
        explicit_renderer_name=args.renderer_name,
        load_checkpoint_path=args.load_checkpoint_path,
        base_url=args.base_url,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=Tulu38BComparisonBuilder(),
    )

    return train_dpo.Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=args.load_checkpoint_path,
        renderer_name=renderer_name,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        dpo_beta=args.dpo_beta,
        lora_rank=args.lora_rank,
        num_replicas=args.num_replicas,
        base_url=args.base_url,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=args.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        ttl_seconds=None,
        rolling_save_every=args.rolling_save_every,
        rolling_ttl_seconds=args.rolling_ttl_seconds,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        wandb_project=None,
        wandb_name=args.wandb_name,
        enable_trace=False,
        span_chart_every=0,
        reference_model_name=args.reference_model_name,
        max_steps=args.max_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO on Tulu preference data")
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--load_checkpoint_path", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--renderer_name", default=None)
    parser.add_argument("--reference_model_name", default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_schedule", default="linear")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--rolling_save_every", type=int, default=10)
    parser.add_argument("--rolling_ttl_seconds", type=int, default=7200)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--num_replicas", type=int, default=8)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--base_url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    train_dpo.main(config)


if __name__ == "__main__":
    main()
