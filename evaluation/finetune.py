#!/usr/bin/env python
"""
Tulu-aligned SFT on Tinker
============================
目标:在 Tinker (LoRA-only) 的约束下,最大化接近 Tulu 3 8B SFT 效果。
聚焦 benchmark:GSM8K, HumanEval, IFEval

关键设计决策与 Tulu 3 的对应关系:
  - 数据:直接用 allenai/tulu-3-sft-mixture,按 source 过滤掉无关子集
         (多语言 Aya、安全 WildJailbreak/WildGuard/CoCoNot、表格/学术阅读)
  - Effective batch size 128:对齐 Tulu (通过 gradient accumulation 实现)
  - Max seq length 4096:对齐 Tulu (不要用 1024,会丢 30% 高质量长样本)
  - LoRA rank 128-256:Tinker 约束下尽量接近全参能力
  - LR 2e-5(LoRA 需要比全参 5e-6 高约 5-10x)
  - Linear warmup 3% + linear decay 到 0:对齐 Tulu
  - 2 epochs:对齐 Tulu
  - AdamW β1=0.9, β2=0.999:SFT 标准(不是 pretraining 的 0.95)

Usage:
    # 快速验证(5 万样本,~1 epoch)
    python train_sft_tulu_aligned.py --num_samples 50000 --num_epochs 1

    # 完整训练(30 万样本,2 epochs,对齐 Tulu)
    python train_sft_tulu_aligned.py --num_samples 300000 --num_epochs 2

    # 不发布,只存 checkpoint
    python train_sft_tulu_aligned.py --no_publish
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_3B = "meta-llama/Llama-3.2-3B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
SEED = 42

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Data Selection: 按 source 过滤 tulu-3-sft-mixture
# ============================================================
# 只保留对 GSM8K / HumanEval / IFEval 有贡献的子集
# 完全跳过:Aya(多语言)、WildJailbreak/WildGuardMix/CoCoNot(安全)、
#         SciRIFF(学术)、TableGPT(表格)
# ============================================================

KEEP_SOURCES = {
    # ===== 数学类 → GSM8K =====
    "ai2-adapt-dev/personahub_math_v5_regen_149960",              # Persona MATH
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",              # Persona GSM
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k", # Algebra
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",           # NuminaMath(已去 GSM8K 污染)
    "allenai/tulu-3-sft-personas-math-grade",                     # Math Grade
    
    # ===== 代码类 → HumanEval =====
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",         # 已去 HumanEval 污染
    "ai2-adapt-dev/personahub_code_v2_34999",                     # Persona Code
    
    # ===== 指令遵循 → IFEval =====
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",       # Persona IF(专门为 IFEval 合成)
    
    # ===== 通用能力(辅助所有 benchmark)=====
    "ai2-adapt-dev/flan_v2_converted",                            # FLAN v2
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k",                      # WildChat GPT-4
    "ai2-adapt-dev/oasst1_converted",                             # OpenAssistant
    "ai2-adapt-dev/no_robots_converted",                          # No Robots
}

# 每个 source 的采样上限(避免某类过大)
# 如果 balance_mode="capped",会按这个上限采样;否则保留全部
PER_SOURCE_CAP = {
    # 数学:留足够多,但不让任何单一 source 独占
    "ai2-adapt-dev/personahub_math_v5_regen_149960": 50_000,
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": 49_980,     # 全保留
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": 20_000,  # 全保留
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": 30_000,
    "allenai/tulu-3-sft-personas-math-grade": 30_000,
    
    # 代码
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": 50_000,
    "ai2-adapt-dev/personahub_code_v2_34999": 34_999,             # 全保留
    
    # IF:全保留,量本来就少
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980": 29_980,
    
    # 通用对话:适量保留,防止模型变成"只会解题"
    "ai2-adapt-dev/flan_v2_converted": 30_000,
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": 20_000,
    "ai2-adapt-dev/oasst1_converted": 7_132,                      # 全保留
    "ai2-adapt-dev/no_robots_converted": 9_500,                   # 全保留
}


def load_tulu3_targeted(max_samples=None, balance_mode="capped", verbose=True):
    """加载 Tulu 3 SFT mixture,只保留对目标 benchmark 有贡献的子集。
    
    Args:
        max_samples: 总样本上限。None = 不限制
        balance_mode: 
            "full":保留 KEEP_SOURCES 里所有样本(~760k)
            "capped":每个 source 按 PER_SOURCE_CAP 封顶(~310k)
        verbose: 打印每个 source 的计数
    """
    if verbose:
        print(f"Loading allenai/tulu-3-sft-mixture (mode={balance_mode})...")
    
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    
    conversations = []
    source_counts = Counter()
    scanned = 0

    # 计算所有source的cap总和，当全部达到cap时停止
    total_cap = sum(PER_SOURCE_CAP.get(s, 999_999) for s in KEEP_SOURCES) if balance_mode == "capped" else None

    for example in ds:
        scanned += 1

        # 当所有source都达到cap时提前停止
        if balance_mode == "capped" and len(conversations) >= (total_cap if total_cap else 999_999):
            break
        if max_samples and len(conversations) >= max_samples:
            break

        source = example.get("source", "")
        
        # 过滤 1:只保留目标 source
        if source not in KEEP_SOURCES:
            continue
        
        # 过滤 2:按 per-source 上限封顶
        if balance_mode == "capped":
            cap = PER_SOURCE_CAP.get(source, 999_999)
            if source_counts[source] >= cap:
                continue
        
        # 过滤 3:结构合法性
        msgs = example.get("messages", [])
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        
        # 只保留 user/assistant turn
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs if m["role"] in ("user", "assistant")
        ]
        if not convo or convo[0]["role"] != "user":
            continue
        
        conversations.append(convo)
        source_counts[source] += 1
    
    if verbose:
        print(f"\nScanned {scanned} examples, kept {len(conversations)}")
        print("Composition by source:")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            short_name = src.split("/")[-1][:55]
            print(f"  {count:>7,} | {short_name}")
    
    return conversations


# ============================================================
# LR Scheduler: 对齐 Tulu 3 的 linear warmup + linear decay
# ============================================================

def build_lr_schedule(step, total_steps, peak_lr, warmup_ratio=0.03):
    """Linear warmup 到 peak_lr,然后 linear decay 到 0。
    对应 Tulu 3 的 `lr_scheduler_type=linear`, `warmup_ratio=0.03`。
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    
    if step < warmup_steps:
        # Warmup: 0 → peak_lr
        return peak_lr * (step + 1) / warmup_steps
    
    # Decay: peak_lr → 0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * max(0.0, 1.0 - progress)


# ============================================================
# Training Data Preparation
# ============================================================

def prepare_training_data(conversations, renderer, max_length, verbose=True):
    """将 conversations 转成 tinker 的 training data 格式。
    
    关键:train_on_what=ALL_ASSISTANT_MESSAGES 确保只对 assistant 回答算 loss,
    user prompt 部分会被 mask 掉(weight=0)。
    """
    all_data = []
    skipped_reasons = Counter()
    
    for convo in conversations:
        try:
            datum = conversation_to_datum(
                convo, renderer, 
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            
            # 关键检查:确保至少有一个 token 参与 loss
            # 如果样本被 truncate 到只剩 prompt,所有 weights 会是 0,这种样本没价值
            weights = datum.loss_fn_inputs["weights"]
            if hasattr(weights, "tolist"):
                weights = weights.tolist()
            if sum(weights) < 1:
                skipped_reasons["empty_weights_after_truncation"] += 1
                continue
            
            all_data.append(datum)
        except Exception as e:
            skipped_reasons[type(e).__name__] += 1
    
    if verbose:
        print(f"\nPrepared {len(all_data)} training examples")
        if skipped_reasons:
            print(f"Skipped samples by reason:")
            for reason, count in skipped_reasons.most_common():
                print(f"  {count:>6,} | {reason}")
    
    return all_data


# ============================================================
# Batch Size Probing: 找 Tinker 上最优的 per-device batch
# ============================================================

def probe_per_device_batch_size(tc, all_data, target_effective_bsz=128):
    """自动探测 Tinker 上能跑的最大 per-device batch size,
    并计算达到 target effective batch size 需要的 gradient accumulation steps。
    """
    print("\n" + "=" * 60)
    print("Probing per-device batch size limit on Tinker...")
    print("=" * 60)
    
    # 从小到大试,出错就停
    candidate_sizes = [4, 8, 16]
    best_bsz = 4
    results = {}
    
    for bsz in candidate_sizes:
        if bsz > len(all_data):
            break
        try:
            batch = all_data[:bsz]
            # Warmup 一次
            _ = tc.forward_backward(batch, loss_fn="cross_entropy").result()
            
            # Timed run
            t0 = time.time()
            _ = tc.forward_backward(batch, loss_fn="cross_entropy").result()
            elapsed = time.time() - t0
            
            tokens = sum(len(d.model_input.to_ints()) for d in batch)
            throughput = tokens / elapsed
            results[bsz] = throughput
            
            print(f"  bsz={bsz:3d}: {elapsed:.2f}s | {tokens:5d} tok | "
                  f"{throughput:6.0f} tok/s")
            best_bsz = bsz
        except Exception as e:
            print(f"  bsz={bsz:3d}: FAILED ({type(e).__name__})")
            break
    
    # 如果 Tinker 支持更大,可以取最大可行值;否则用 best_bsz
    accum_steps = max(1, target_effective_bsz // best_bsz)
    actual_bsz = best_bsz * accum_steps
    
    print(f"\nRecommendation:")
    print(f"  per_device_batch_size: {best_bsz}")
    print(f"  grad_accum_steps:      {accum_steps}")
    print(f"  effective batch size:  {actual_bsz}")
    print("=" * 60)
    
    return best_bsz, accum_steps


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    
    # ===== 模型选择 =====
    parser.add_argument("--model", type=str, default=MODEL_8B,
                        choices=[MODEL_3B, MODEL_8B],
                        help="Base model (default: 8B for final runs)")
    
    # ===== 数据配置 =====
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Total sample cap. 0 = no limit")
    parser.add_argument("--balance_mode", type=str, default="full",
                        choices=["full", "capped"],
                        help="full=keep Tulu 3 original proportions; capped=per-source limit")
    
    # ===== 训练时长 =====
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Epochs (Tulu 3 uses 2)")
    
    # ===== Batch size(对齐 Tulu effective bsz=128)=====
    parser.add_argument("--per_device_batch_size", type=int, default=0,
                        help="Per-device batch. 0 = auto-probe")
    parser.add_argument("--grad_accum_steps", type=int, default=0,
                        help="Gradient accumulation steps. 0 = auto (bsz_eff=128)")
    parser.add_argument("--target_effective_bsz", type=int, default=128,
                        help="Target effective batch size (Tulu 3 = 128)")
    
    # ===== LoRA =====
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank (higher = closer to full-ft, Tinker cap ~256)")
    
    # ===== Learning rate =====
    parser.add_argument("--peak_lr", type=float, default=2e-5,
                        help="Peak LR (LoRA needs ~5-10x full-ft's 5e-6)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio (Tulu 3 = 0.03)")
    
    # ===== Sequence length =====
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max seq length (Tulu 3 = 4096, don't use 1024!)")
    
    # ===== Checkpointing =====
    parser.add_argument("--checkpoint_name", type=str, default="tulu_aligned_sft")
    parser.add_argument("--no_publish", action="store_true")
    parser.add_argument("--log_every", type=int, default=5,
                        help="Log loss every N optim steps")
    
    args = parser.parse_args()
    
    # ============================================================
    # Phase 0: Print config
    # ============================================================
    print("=" * 60)
    print("Tulu-aligned SFT Training Configuration")
    print("=" * 60)
    print(f"  Model:                {args.model}")
    print(f"  LoRA rank:            {args.lora_rank}")
    print(f"  Peak LR:              {args.peak_lr}")
    print(f"  Warmup ratio:         {args.warmup_ratio}")
    print(f"  Num samples (target): {args.num_samples:,}")
    print(f"  Balance mode:         {args.balance_mode}")
    print(f"  Num epochs:           {args.num_epochs}")
    print(f"  Max seq length:       {args.max_length}")
    print(f"  Target effective bsz: {args.target_effective_bsz}")
    print(f"  Checkpoint name:      {args.checkpoint_name}")
    print("=" * 60)
    
    # ============================================================
    # Phase 1: Load tokenizer & renderer
    # ============================================================
    print("\n[Phase 1] Setting up tokenizer & renderer...")
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"  Renderer: {renderer_name}")
    
    # ============================================================
    # Phase 2: Load & filter data
    # ============================================================
    print("\n[Phase 2] Loading Tulu 3 SFT mixture...")
    conversations = load_tulu3_targeted(
        max_samples=args.num_samples,
        balance_mode=args.balance_mode,
    )
    
    # ============================================================
    # Phase 3: Tokenize (apply chat template + masking)
    # ============================================================
    print("\n[Phase 3] Tokenizing conversations...")
    all_data = prepare_training_data(
        conversations, renderer, max_length=args.max_length
    )
    
    if len(all_data) < 100:
        raise RuntimeError(f"Too few training examples ({len(all_data)}). "
                           f"Check data loading or increase --num_samples.")
    
    # ============================================================
    # Phase 4: Create training client
    # ============================================================
    print(f"\n[Phase 4] Creating LoRA training client (rank={args.lora_rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    print("  Ready")
    
    # ============================================================
    # Phase 5: Determine batch size config
    # ============================================================
    if args.per_device_batch_size == 0 or args.grad_accum_steps == 0:
        # Auto-probe
        per_device, accum_steps = probe_per_device_batch_size(
            tc, all_data, target_effective_bsz=args.target_effective_bsz
        )
    else:
        per_device = args.per_device_batch_size
        accum_steps = args.grad_accum_steps
        print(f"\n[Phase 5] Using manual batch config:")
        print(f"  per_device: {per_device}, accum: {accum_steps}")
    
    effective_bsz = per_device * accum_steps
    
    # ============================================================
    # Phase 6: Compute training schedule
    # ============================================================
    # 一个 "logical step" = 一次参数更新 = accum_steps 次 forward_backward + 1 次 optim_step
    total_logical_steps = (len(all_data) * args.num_epochs) // effective_bsz
    total_fwd_bwd_calls = total_logical_steps * accum_steps
    
    print(f"\n[Phase 6] Training schedule:")
    print(f"  Total optim steps:       {total_logical_steps:,}")
    print(f"  Total fwd_bwd calls:     {total_fwd_bwd_calls:,}")
    print(f"  Effective batch size:    {effective_bsz}")
    print(f"  Total samples seen:      {total_fwd_bwd_calls * per_device:,}")
    
    # ============================================================
    # Phase 7: Training loop
    # ============================================================
    print(f"\n[Phase 7] Starting training...\n")
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 构造 epoch-wise 的 shuffle 索引
    # 好处:每个 epoch 内数据顺序不同,多 epoch 学到的泛化性更好
    all_indices = []
    for epoch_idx in range(args.num_epochs):
        epoch_indices = list(range(len(all_data)))
        random.Random(SEED + epoch_idx).shuffle(epoch_indices)
        all_indices.extend(epoch_indices)
    
    start_time = time.time()
    losses_log = []
    
    for logical_step in range(total_logical_steps):
        # 计算当前 lr(warmup + decay)
        current_lr = build_lr_schedule(
            logical_step, total_logical_steps, 
            args.peak_lr, args.warmup_ratio
        )
        # 关键:β2=0.999 是 SFT 的标准值,不是 pretraining 的 0.95
        adam_params = types.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )
        
        # ============================================
        # Gradient accumulation: 累积 accum_steps 次 forward_backward
        # 然后做 1 次 optim_step
        # ============================================
        step_loss_sum = 0.0
        step_weight_sum = 0.0
        
        for micro_step in range(accum_steps):
            # 从 all_indices 切出当前 micro-batch
            idx_start = (logical_step * accum_steps + micro_step) * per_device
            batch_indices = [
                all_indices[(idx_start + i) % len(all_indices)]
                for i in range(per_device)
            ]
            batch = [all_data[i] for i in batch_indices]
            
            # Forward + backward(梯度累积到 tc 内部状态)
            fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
            fwd_bwd_result = fwd_bwd_future.result()
            
            # 计算 loss(仅用于日志,不影响训练)
            logprobs = np.concatenate([
                o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs
            ])
            weights = np.concatenate([
                d.loss_fn_inputs["weights"].tolist() for d in batch
            ])
            step_loss_sum += -np.dot(logprobs, weights)
            step_weight_sum += weights.sum()
        
        # 累积完 accum_steps 次梯度后,一次性更新参数
        optim_future = tc.optim_step(adam_params)
        optim_future.result()
        
        avg_loss = step_loss_sum / max(step_weight_sum, 1)
        losses_log.append(avg_loss)
        
        # 日志
        if (logical_step + 1) % args.log_every == 0 or logical_step == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (logical_step + 1) / elapsed
            eta_sec = (total_logical_steps - logical_step - 1) / steps_per_sec
            
            # Moving average loss(更平滑)
            ma_window = min(10, len(losses_log))
            ma_loss = np.mean(losses_log[-ma_window:])
            
            print(f"  Step {logical_step+1:>5}/{total_logical_steps} | "
                  f"LR: {current_lr:.2e} | "
                  f"Loss: {avg_loss:.4f} (ma{ma_window}: {ma_loss:.4f}) | "
                  f"ETA: {eta_sec/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n[Training complete] Total time: {total_time/60:.1f} minutes")
    
    # ============================================================
    # Phase 8: Save checkpoint
    # ============================================================
    print(f"\n[Phase 8] Saving checkpoint '{args.checkpoint_name}'...")
    
    # Sampler checkpoint(用于推理/评估)
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    ckpt_path = ckpt.path
    print(f"  Sampler checkpoint: {ckpt_path}")
    
    # State checkpoint(用于未来继续训练,比如接 DPO)
    state = tc.save_state(args.checkpoint_name + "_state").result()
    state_path = state.path
    print(f"  Training state:     {state_path}")
    
    # ============================================================
    # Phase 9: Publish (optional)
    # ============================================================
    if not args.no_publish:
        print(f"\n[Phase 9] Publishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(ckpt_path).result()
        print("  Published!")
    
    # ============================================================
    # Save run metadata
    # ============================================================
    info = {
        "checkpoint_path": ckpt_path,
        "state_path": state_path,
        "base_model": args.model,
        "renderer_name": renderer_name,
        "training": {
            "num_samples": len(all_data),
            "num_epochs": args.num_epochs,
            "total_logical_steps": total_logical_steps,
            "per_device_batch_size": per_device,
            "grad_accum_steps": accum_steps,
            "effective_batch_size": effective_bsz,
            "peak_lr": args.peak_lr,
            "warmup_ratio": args.warmup_ratio,
            "lora_rank": args.lora_rank,
            "max_length": args.max_length,
            "balance_mode": args.balance_mode,
        },
        "final_loss_ma10": float(np.mean(losses_log[-10:])) if losses_log else None,
        "total_training_seconds": total_time,
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, f"{args.checkpoint_name}_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nRun info saved to {info_path}")
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"  1. Evaluate: python evaluation/eval_all.py "
          f"--checkpoint_path '{ckpt_path}' --base_model {args.model}")
    print(f"  2. Run DPO:  use state path '{state_path}' as load_checkpoint_path")
    print('=' * 60)


if __name__ == "__main__":
    main()