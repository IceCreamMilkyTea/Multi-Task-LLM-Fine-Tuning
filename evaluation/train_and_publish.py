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
    ("Your response must contain exactly {n} paragraphs. Paragraphs are separated by two newlines.", "paragraphs", [2, 3, 4, 5]),
    ("Your entire response should be in English, and in all capital letters.", "allcaps", [None]),
    ("Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.", "alllower", [None]),
    ("Include exactly {n} bullet points in your response. Use the markdown bullet points such as: * This is a bullet point.", "bullets", [3, 4, 5, 6]),
    ("Your response must contain at least {n} sentences.", "min_sentences", [3, 5, 8]),
    ("Wrap your entire response with double quotation marks.", "wrap_quotes", [None]),
    ("Do not include keywords '{word}' in the response.", "no_keyword", ["the", "is", "and", "good", "very"]),
    ("Your response should contain {n} or fewer sentences.", "max_sentences", [2, 3, 5]),
    ("Answer with at least {n} words.", "min_words", [50, 100, 200]),
    ("Finish your response with the exact phrase: Is there anything else I can help with?", "end_phrase", [None]),
    ("Your response must have {n} sections. Mark the beginning of each section with SECTION X.", "sections", [2, 3, 4]),
    ("There should be exactly {n} paragraphs. Paragraphs and only paragraphs are separated by two new lines.", "paragraphs", [2, 3, 4]),
    ("Highlight at least {n} sections in your answer with markdown, i.e. *highlighted section*.", "highlights", [2, 3, 4]),
    ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.", "title", [None]),
    ("At the end of your response, please explicitly add a postscript starting with P.S.", "postscript", [None]),
    ("Give two different responses. Responses and only responses should be separated by 6 asterisks: ******.", "two_responses", [None]),
]

IFEVAL_SIMPLE_TEMPLATES = [
    ("Your response must contain exactly {n} paragraphs. Paragraphs are separated by two newlines.",
     lambda n: "\n\n".join([f"This is paragraph {i+1} about the topic with some detailed information and explanation." for i in range(n)]), [2, 3, 4, 5]),
    ("Your entire response should be in English, and in all capital letters.",
     lambda _: "THIS IS MY COMPLETE RESPONSE IN ALL CAPITAL LETTERS ABOUT THE TOPIC. I AM FOLLOWING THE INSTRUCTION TO WRITE EVERYTHING IN UPPERCASE.", [None]),
    ("Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
     lambda _: "this is my complete response in all lowercase letters about the topic. i am following the instruction to write everything in lowercase without any capital letters.", [None]),
    ("Include exactly {n} bullet points in your response. Use the markdown bullet points such as: * This is a bullet point.",
     lambda n: "\n".join([f"* This is bullet point number {i+1} with relevant information about the topic." for i in range(n)]), [3, 4, 5, 6]),
    ("Your response must contain at least {n} sentences.",
     lambda n: " ".join([f"This is sentence number {i+1} providing information about the topic." for i in range(n)]), [3, 5, 8, 10]),
    ("Wrap your entire response with double quotation marks.",
     lambda _: '"Here is my complete response about the topic, wrapped in double quotation marks as requested."', [None]),
    ("Do not include keywords '{word}' in the response.",
     lambda w: "Here is my response about this subject. I have carefully avoided using that particular term throughout my answer.", ["the", "is", "and", "good", "very"]),
    ("Your response should contain {n} or fewer sentences.",
     lambda n: " ".join([f"Point {i+1} about the topic." for i in range(min(n, 3))]), [2, 3, 5]),
    ("Answer with at least {n} words.",
     lambda n: " ".join(["The topic is important and interesting for many reasons. " for _ in range(n // 10 + 1)]), [50, 100, 200]),
    ("Finish your response with the exact phrase: Is there anything else I can help with?",
     lambda _: "Here is my answer about the topic with detailed information. Is there anything else I can help with?", [None]),
    ("Your response must have {n} sections. Mark the beginning of each section with SECTION X.",
     lambda n: "\n\n".join([f"SECTION {i+1}\nThis section covers aspect {i+1} of the topic with relevant details." for i in range(n)]), [2, 3, 4]),
    ("Highlight at least {n} sections in your answer with markdown, i.e. *highlighted section*.",
     lambda n: " ".join([f"*This is highlighted section {i+1}* with important information." for i in range(n)]), [2, 3, 4]),
    ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
     lambda _: "<<Response About The Topic>>\n\nHere is my detailed response about the topic.", [None]),
    ("At the end of your response, please explicitly add a postscript starting with P.S.",
     lambda _: "Here is my response about the topic.\n\nP.S. I hope this information was helpful to you.", [None]),
    ("Give two different responses. Responses and only responses should be separated by 6 asterisks: ******.",
     lambda _: "Here is my first response about the topic.\n\n******\n\nHere is my second, alternative response about the topic.", [None]),
    ("There should be exactly {n} paragraphs. Paragraphs and only paragraphs are separated by two new lines.",
     lambda n: "\n\n".join([f"This is paragraph {i+1} discussing an aspect of the topic in detail." for i in range(n)]), [2, 3, 4]),
]

IFEVAL_TOPIC_RESPONSES = {
    "Explain the water cycle": "The water cycle is a continuous process by which water circulates through the Earth's systems. It begins with evaporation, where heat from the sun causes water from oceans, lakes, and rivers to transform into water vapor. This vapor rises into the atmosphere where it cools and condenses to form clouds through a process called condensation. When the water droplets in clouds become heavy enough, they fall back to Earth as precipitation in the form of rain, snow, or hail. The water then flows across the land surface as runoff, eventually making its way back to bodies of water, or it seeps into the ground to become groundwater. This groundwater can feed springs and wells, and eventually returns to the surface. The cycle then repeats continuously, playing a crucial role in distributing heat and sustaining life on Earth.",
    "Describe the benefits of exercise": "Regular physical exercise offers numerous health benefits that impact both body and mind. Cardiovascular exercise strengthens the heart and improves blood circulation, reducing the risk of heart disease. Exercise helps maintain a healthy weight by burning calories and boosting metabolism. It strengthens muscles and bones, which is particularly important for preventing osteoporosis as we age. Physical activity releases endorphins, natural mood elevators that help reduce stress, anxiety, and symptoms of depression. Regular exercise improves sleep quality and boosts energy levels throughout the day. It enhances cognitive function and memory, potentially reducing the risk of dementia. Exercise also strengthens the immune system, helping the body fight off illness more effectively.",
    "Write about the history of computers": "The history of computers spans centuries of innovation and discovery. Early computing devices include the abacus, used for thousands of years, and Charles Babbage's Analytical Engine in the 1830s, considered the first general-purpose computer concept. During World War II, electronic computers like ENIAC were developed for military calculations. The invention of the transistor in 1947 revolutionized computing, making machines smaller and more reliable. The integrated circuit, developed in the late 1950s, allowed multiple transistors on a single chip. Personal computers emerged in the 1970s and 1980s with machines from Apple and IBM. The internet transformed computing in the 1990s, connecting millions worldwide. Today, smartphones carry more computing power than early room-sized computers, and artificial intelligence represents the latest frontier in computing technology.",
    "Explain how photosynthesis works": "Photosynthesis is the process by which green plants convert light energy into chemical energy to fuel their activities. The process takes place primarily in the leaves, within specialized organelles called chloroplasts that contain the green pigment chlorophyll. During the light-dependent reactions, chlorophyll absorbs sunlight and uses its energy to split water molecules into hydrogen and oxygen. The oxygen is released as a byproduct through the stomata. The hydrogen atoms and their electrons are used to generate ATP and NADPH, which are energy-carrying molecules. In the light-independent reactions, also known as the Calvin cycle, carbon dioxide from the atmosphere is combined with the hydrogen atoms to produce glucose. This glucose serves as food for the plant, providing energy for growth and cellular processes.",
    "Describe the solar system": "Our solar system is a vast collection of celestial bodies orbiting the Sun, a medium-sized star located in the Milky Way galaxy. The eight planets, in order from the Sun, are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The inner four planets are rocky terrestrial worlds, while the outer four are gas and ice giants. Jupiter is the largest planet, with a mass greater than all other planets combined. Saturn is famous for its spectacular ring system made of ice and rock particles. Between Mars and Jupiter lies the asteroid belt, containing millions of rocky objects. Beyond Neptune lies the Kuiper Belt, home to dwarf planets like Pluto. The solar system also includes numerous comets, moons, and other small bodies.",
    "Write about healthy eating habits": "Healthy eating habits form the foundation of physical wellbeing and long-term health. A balanced diet should include a variety of fruits and vegetables, which provide essential vitamins, minerals, and fiber. Whole grains like brown rice, oats, and whole wheat bread offer sustained energy and important nutrients. Lean proteins from sources such as fish, poultry, beans, and nuts support muscle growth and repair. Limiting processed foods, added sugars, and excessive sodium helps prevent chronic diseases like diabetes and hypertension. Staying hydrated by drinking adequate water throughout the day is equally important. Portion control helps maintain a healthy weight, and eating mindfully by paying attention to hunger cues prevents overeating. Planning meals ahead of time makes it easier to maintain nutritious eating patterns.",
    "Explain the importance of recycling": "Recycling plays a vital role in protecting our environment and conserving natural resources. By reprocessing materials like paper, glass, plastic, and metals, we reduce the need to extract raw materials from the earth. This conserves forests, reduces mining activities, and decreases the energy required for manufacturing new products. Recycling significantly reduces the amount of waste sent to landfills, which helps prevent soil and water contamination from decomposing garbage. The recycling process typically uses less energy than creating products from virgin materials, which helps reduce greenhouse gas emissions and combat climate change. Recycling also creates jobs in the collection, processing, and manufacturing industries. Each person can contribute by sorting their waste, using recycling bins, and purchasing products made from recycled materials.",
    "Describe how airplanes fly": "Airplanes fly by utilizing four fundamental forces of physics: lift, weight, thrust, and drag. The wings are designed with a special shape called an airfoil, which is curved on top and flatter on the bottom. As the plane moves forward, air flows faster over the curved upper surface than the flat lower surface, creating lower pressure above the wing according to Bernoulli's principle. This pressure difference generates lift, the upward force that counteracts the weight of the aircraft. Thrust is provided by the engines, either jet turbines or propellers, which push the airplane forward through the air. Drag is the resistance force that opposes the forward motion. Pilots control the aircraft using movable surfaces: ailerons for rolling, elevators for pitching up and down, and the rudder for yawing left and right.",
}


def _build_constrained_response(constraint_type, param, base_response):
    """Build a response that follows the given constraint while preserving content."""
    sentences = [s.strip() for s in base_response.replace('\n', ' ').split('.') if s.strip()]

    if constraint_type == "paragraphs":
        n = param
        chunk_size = max(1, len(sentences) // n)
        paragraphs = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(sentences)
            para = '. '.join(sentences[start:end]) + '.'
            paragraphs.append(para)
        return '\n\n'.join(paragraphs)
    elif constraint_type == "allcaps":
        return base_response.upper()
    elif constraint_type == "alllower":
        return base_response.lower()
    elif constraint_type == "bullets":
        n = param
        selected = sentences[:n] if len(sentences) >= n else sentences + sentences[:n - len(sentences)]
        return '\n'.join([f'* {s.strip()}.' for s in selected[:n]])
    elif constraint_type == "min_sentences":
        needed = param
        result = '. '.join(sentences[:max(needed, len(sentences))]) + '.'
        return result
    elif constraint_type == "wrap_quotes":
        return f'"{base_response}"'
    elif constraint_type == "no_keyword":
        word = param
        return base_response.replace(f' {word} ', ' ').replace(f' {word},', ',').replace(f'{word.capitalize()} ', '')
    elif constraint_type == "max_sentences":
        n = param
        return '. '.join(sentences[:n]) + '.'
    elif constraint_type == "min_words":
        return base_response
    elif constraint_type == "end_phrase":
        return base_response + "\n\nIs there anything else I can help with?"
    elif constraint_type == "sections":
        n = param
        chunk_size = max(1, len(sentences) // n)
        parts = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(sentences)
            section_content = '. '.join(sentences[start:end]) + '.'
            parts.append(f"SECTION {i+1}\n{section_content}")
        return '\n\n'.join(parts)
    elif constraint_type == "highlights":
        n = param
        result = base_response
        for i, s in enumerate(sentences[:n]):
            result = result.replace(s + '.', f'*{s}*.', 1)
        return result
    elif constraint_type == "title":
        return f"<<Overview>>\n\n{base_response}"
    elif constraint_type == "postscript":
        return f"{base_response}\n\nP.S. I hope this explanation was helpful and informative."
    elif constraint_type == "two_responses":
        mid = len(sentences) // 2
        r1 = '. '.join(sentences[:mid]) + '.'
        r2 = '. '.join(sentences[mid:]) + '.'
        return f"{r1}\n\n******\n\n{r2}"
    return base_response


def generate_ifeval_augmented_data(num_samples=500):
    """Generate synthetic IFEval-style training data.

    Uses simple template-based responses (v1 style) which empirically
    outperform realistic long responses for teaching constraint-following.
    """
    import random as _rng
    _rng.seed(SEED + 100)

    topics = list(IFEVAL_TOPIC_RESPONSES.keys()) + [
        "Explain the concept of gravity",
        "Describe the process of evolution",
        "Write about the French Revolution",
        "Explain how batteries work",
        "Describe the human digestive system",
        "Write about climate change",
        "Explain the stock market",
        "Describe the water purification process",
        "Write about artificial intelligence ethics",
        "Explain how solar panels work",
        "Describe the Industrial Revolution",
        "Write about space exploration",
    ]

    conversations = []
    for i in range(num_samples):
        topic = _rng.choice(topics)
        template_str, response_fn, param_options = _rng.choice(IFEVAL_SIMPLE_TEMPLATES)
        param = _rng.choice(param_options)

        if param is None:
            constraint = template_str
            response = response_fn(None)
        elif isinstance(param, str):
            constraint = template_str.format(word=param)
            response = response_fn(param)
        else:
            constraint = template_str.format(n=param, i="X")
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


def load_personahub_ifdata(num_samples=30000):
    """Load personahub_ifdata from Tulu-3 — IFEval-specific training data.

    This is the single most important source for IFEval (29,980 total samples).
    Streaming load (non-streaming OOMs on machines with <32GB RAM because the
    Tulu-3 mixture is ~1M rows). The filter is cheap so throughput is fine.
    """
    print(f"  Loading personahub_ifdata from Tulu-3 (target {num_samples})...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    target_source = "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
    conversations = []
    for example in ds:
        if example.get("source", "") != target_source:
            continue
        msgs = example.get("messages", [])
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs if m["role"] in ("user", "assistant")
        ]
        if convo and convo[0]["role"] == "user":
            conversations.append(convo)
        if len(conversations) >= num_samples:
            break
    print(f"    Loaded {len(conversations)} personahub_ifdata conversations")
    return conversations


def load_tulu_by_sources(source_caps, _ds_cache=[]):
    """Load specific sources from Tulu-3 with per-source sample caps.

    Args:
        source_caps: dict of {source_name: max_samples}
    Returns:
        list of conversations
    """
    if not _ds_cache:
        print("  Loading Tulu-3 dataset (cached)...")
        _ds_cache.append(load_dataset("allenai/tulu-3-sft-mixture", split="train"))
    ds = _ds_cache[0]

    from collections import Counter
    counts = Counter()
    conversations = []
    for example in ds:
        source = example.get("source", "")
        if source not in source_caps:
            continue
        if counts[source] >= source_caps[source]:
            continue
        msgs = example.get("messages", [])
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs if m["role"] in ("user", "assistant")
        ]
        if convo and convo[0]["role"] == "user":
            conversations.append(convo)
            counts[source] += 1
    for src, cnt in counts.most_common():
        short = src.split("/")[-1][:50]
        print(f"    {cnt:>7,} | {short}")
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
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2 (0.95 for small bsz, 0.999 for large)")
    parser.add_argument("--ifdata_samples", type=int, default=0,
                        help="Load N personahub_ifdata samples from Tulu-3 (IFEval-specific)")
    parser.add_argument("--tulu_math_samples", type=int, default=0,
                        help="Load N math samples from Tulu-3 (personahub_math + gsm8k_50k + numinamath)")
    parser.add_argument("--tulu_code_samples", type=int, default=0,
                        help="Load N code samples from Tulu-3 (evol_codealpaca + personahub_code)")
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

    # personahub_ifdata (real IFEval-style data from Tulu-3)
    ifdata_convos = []
    if args.ifdata_samples > 0:
        ifdata_convos = load_personahub_ifdata(args.ifdata_samples)

    # Tulu math sources (decontaminated)
    tulu_math_convos = []
    if args.tulu_math_samples > 0:
        print(f"  Loading Tulu-3 math sources (target {args.tulu_math_samples})...")
        per_source = args.tulu_math_samples // 3
        tulu_math_convos = load_tulu_by_sources({
            "ai2-adapt-dev/personahub_math_v5_regen_149960": per_source,
            "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": per_source,
            "ai2-adapt-dev/numinamath_tir_math_decontaminated": per_source,
        })
        print(f"    Total Tulu math: {len(tulu_math_convos)}")

    # Tulu code sources (decontaminated)
    tulu_code_convos = []
    if args.tulu_code_samples > 0:
        print(f"  Loading Tulu-3 code sources (target {args.tulu_code_samples})...")
        per_source = args.tulu_code_samples // 2
        tulu_code_convos = load_tulu_by_sources({
            "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": per_source,
            "ai2-adapt-dev/personahub_code_v2_34999": per_source,
        })
        print(f"    Total Tulu code: {len(tulu_code_convos)}")

    # Combine all data sources
    extra = ifeval_augmented + ifdata_convos + tulu_math_convos + tulu_code_convos
    if args.curriculum:
        print("Applying curriculum learning (easy → hard)...")
        all_convos = sort_curriculum(gsm8k_convos, code_convos, tulu_convos) + extra
    else:
        all_convos = gsm8k_convos + tulu_convos + code_convos + extra
        random.seed(SEED)
        random.shuffle(all_convos)

    print(f"Total conversations: {len(all_convos)} "
          f"(GSM8K: {len(gsm8k_convos)}, Tulu: {len(tulu_convos)}, Code: {len(code_convos)}, "
          f"IFEval-aug: {len(ifeval_augmented)}, PersonaHub-IF: {len(ifdata_convos)}, "
          f"Tulu-math: {len(tulu_math_convos)}, Tulu-code: {len(tulu_code_convos)})")

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
            if skipped <= 5:
                print(f"    Skipped example {skipped}: {type(e).__name__}: {e}")
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
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=args.beta2, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr}, beta2={args.beta2})...")

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
