"""
Tulu-3 IF-focused preference data builder for Tinker DPO.

Loads `allenai/llama-3.1-tulu-3-8b-preference-mixture` and filters to the
`allenai/tulu-3-sft-reused-if` source (65,794 IF-targeted preference pairs —
the only IF-focused subset in the 8B preference mix; IF-Augmented / Persona-IF
in the Tulu-3 paper table are only in the 70B mix).

Plugs into `tinker_cookbook.preference.dpo_datasets.DPODatasetBuilderFromComparisons`,
which handles rendering/tokenisation downstream.
"""

import logging
import random
from typing import Any

import chz
import datasets

from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison

logger = logging.getLogger(__name__)

DATASET_NAME = "allenai/llama-3.1-tulu-3-8b-preference-mixture"

# Only 4 distinct sources in the 8B preference mix (confirmed by streaming 200k rows):
#   allenai/tulu-3-sft-reused-off-policy-8b      96,715 (math-heavy)
#   allenai/tulu-3-sft-reused-if                 65,794 (IF / constraint-following)
#   allenai/tulu-3-sft-reused-on-policy-8b       19,773 (math/reasoning)
#   allenai/tulu-3-wildchat-reused-on-policy-8b  17,718 (general chat)
IF_SOURCES = {"allenai/tulu-3-sft-reused-if"}

SEED = 42


def _normalize_messages(raw: Any) -> list[dict] | None:
    """Return a clean list[{'role','content'}] or None if malformed.

    The preference mix stores chosen/rejected as OpenAI message lists, but a
    few rows have stringified content or extra keys we don't care about.
    """
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for msg in raw:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant", "system") or not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    return out or None


def _strip_leading_user_if_dup(messages: list[dict], prompt_text: str) -> list[dict]:
    """If messages[0] is a user turn that duplicates the prompt string,
    drop it so we don't feed the prompt twice when we concatenate
    prompt_conversation + completion."""
    if messages and messages[0]["role"] == "user" and messages[0]["content"].strip() == prompt_text.strip():
        return messages[1:]
    return messages


@chz.chz
class Tulu3IFPreferenceBuilder(ComparisonDatasetBuilder):
    """IF-only subset of the Tulu-3 8B preference mixture, ready for DPO."""

    num_samples: int = 30000          # cap after filter; 65k IF rows available
    test_size: int = 0                # no held-out test split by default
    shuffle_seed: int = SEED
    swap: bool = False                # inherited from ComparisonDatasetBuilder

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        logger.info(f"Loading {DATASET_NAME} (full load — ~1GB)...")
        ds = datasets.load_dataset(DATASET_NAME, split="train")
        total = len(ds)
        logger.info(f"  Loaded {total} rows")

        # Filter by source in one pass
        keep_mask = [s in IF_SOURCES for s in ds["source"]]
        filtered = ds.select([i for i, k in enumerate(keep_mask) if k])
        logger.info(f"  After IF-source filter: {len(filtered)} rows (sources: {sorted(IF_SOURCES)})")

        # Shuffle + cap
        filtered = filtered.shuffle(seed=self.shuffle_seed)
        if self.num_samples and len(filtered) > self.num_samples:
            filtered = filtered.select(range(self.num_samples))
        logger.info(f"  Final training pool: {len(filtered)} rows")

        if self.test_size > 0 and len(filtered) > self.test_size:
            split = filtered.train_test_split(test_size=self.test_size, seed=self.shuffle_seed)
            return split["train"], split["test"]
        return filtered, None

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        if example.get("source") not in IF_SOURCES:
            return None

        prompt_text = example.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return None

        chosen = _normalize_messages(example.get("chosen"))
        rejected = _normalize_messages(example.get("rejected"))
        if chosen is None or rejected is None:
            return None

        # Strip duplicated leading user turn if the dataset embeds the prompt
        # at the start of chosen/rejected (some rows do, some don't).
        chosen = _strip_leading_user_if_dup(chosen, prompt_text)
        rejected = _strip_leading_user_if_dup(rejected, prompt_text)

        if not chosen or not rejected:
            return None
        # Must end in an assistant turn so the DPO loss weights align to generated tokens.
        if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
            return None

        prompt_conversation = [{"role": "user", "content": prompt_text}]
        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=prompt_conversation,
                completion_A=chosen,
                completion_B=rejected,
            ),
            label="A",
        )
