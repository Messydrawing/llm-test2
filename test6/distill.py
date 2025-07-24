# test1/distill.py  ── v6 2025-07-06 ───────────────────────────────────
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # 仅保留 ERROR

# ─── 内部模块 ───────────────────────────────────────────────────────────
from . import (
    config,
    dataset_builder,
    evaluate,
    teacher_labeler,
    train_lora,
)
from .inference import call_teacher as call_deepseek, call_gemini, call_qwen
from .clean_jsonl import main as clean_jsonl_once


# ╭───────────────────────── 工具函数 ─────────────────────────╮
def _clean(src: str | Path) -> str:
    """运行 clean_jsonl，并返回 cleaned_* 路径。"""
    src = Path(src)
    if not src.exists():
        sys.exit(f"[distill] ❌ 找不到 {src}")
    if src.name.endswith("-teacher.jsonl"):
        dst = src.with_name(src.name.replace("-teacher.jsonl", "-cleaned.jsonl"))
    else:
        dst = src.with_name(f"cleaned_{src.name}")
    print(f"[distill] 清洗 {src.name} → {dst.name}")
    clean_jsonl_once(src, dst)
    return str(dst)


# ╭────────────────────────── 主流程 ─────────────────────────╮
def run_pipeline(
    *,
    windows: int,
    val_ratio: float,
    max_tokens: int,
    max_len: int,
    output_dir: str,
    stock: str | None,
    skip_teacher: bool,
    overwrite: bool,
) -> None:

    # 1) 构造数据集
    stock_list = [stock] if stock else config.STOCK_CODES
    train_set, val_set = dataset_builder.build_dataset(
        stock_codes=stock_list,  # ← 修正形参名
        days=180,
        window=30,
        windows_per_stock=windows,
        val_ratio=val_ratio,
        tokenizer=None,
        max_tokens=max_tokens,
    )

    # Write all prompts to question.jsonl
    prompts_all = [
        dataset_builder.format_prompt(s) for s in train_set + val_set
    ]
    with open("question.jsonl", "w", encoding="utf-8") as f:
        for p in prompts_all:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

    # 2) Decide whether to call teachers
    need_label = True
    if skip_teacher and not overwrite:
        need_label = not (
            Path("D-teacher.jsonl").exists()
            and Path("G-teacher.jsonl").exists()
            and Path("Q-teacher.jsonl").exists()
        )

    # 3) Teacher labeling for multiple models
    if need_label:
        print("[distill] ▶ 教师模型标注 …")
        teacher_labeler.label_samples(
            prompts_all,
            "D-teacher.jsonl",
            call_teacher=call_deepseek,
        )
        teacher_labeler.label_samples(
            prompts_all,
            "G-teacher.jsonl",
            call_teacher=call_gemini,
        )
        teacher_labeler.label_samples(
            prompts_all,
            "Q-teacher.jsonl",
            call_teacher=call_qwen,
        )
    else:
        print("[distill] ▶ 跳过教师标注，使用现有 JSONL")

    # 4) 清洗各教师输出（仅 DeepSeek 结果用于训练）
    train_jsonl = _clean("D-teacher.jsonl")
    _clean("G-teacher.jsonl")
    _clean("Q-teacher.jsonl")
    val_jsonl = None

    # 5) LoRA 训练
    lora_cfg = train_lora.TrainConfig(
        data_path=train_jsonl,
        eval_path=val_jsonl,
        output_dir=output_dir,
        max_len=max_len,
        max_steps=200,
    )
    train_lora.main(lora_cfg)

    # 6) 评估
    prompts, refs = evaluate.load_dataset(val_jsonl or train_jsonl)
    metrics = evaluate.evaluate_model(output_dir, prompts, refs)
    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    Path(output_dir, "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False)
    )


# ╭────────────────────────── CLI ───────────────────────────╮
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Distill pipeline")
    ap.add_argument("--windows", type=int, default=1, help="每只股票样本数")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    ap.add_argument(
        "--max-tokens", type=int, default=1024, help="Prompt 最大 token 长度"
    )
    ap.add_argument("--max-len", type=int, default=1024, help="训练截断长度")
    ap.add_argument("--out", default="lora_adapter", help="输出目录")
    ap.add_argument("--stock", help="仅处理指定股票代码")
    ap.add_argument("--skip-teacher", action="store_true", help="跳过教师标注")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    args = ap.parse_args()

    run_pipeline(
        windows=args.windows,
        val_ratio=args.val_ratio,
        max_tokens=args.max_tokens,
        max_len=args.max_len,
        output_dir=args.out,
        stock=args.stock,
        skip_teacher=args.skip_teacher,
        overwrite=args.overwrite,
    )
