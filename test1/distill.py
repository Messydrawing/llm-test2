# test1/distill.py  ── v5 ───────────────────────────────────────────────
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()                      # 静音 INFO/WARNING

# 内部模块
from . import (
    config,
    dataset_builder,
    evaluate,
    teacher_labeler,
    train_lora,
)
from .clean_jsonl import main as clean_jsonl_once

# ╭───────────────────────── 辅助函数 ─────────────────────────╮
def _clean_once(src: str | Path) -> str:
    """调用 clean_jsonl 清洗文件，返回 cleaned_* 路径。"""
    src = Path(src)
    if not src.exists():
        sys.exit(f"[distill] ❌ 找不到 {src}")
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
    codes = [stock] if stock else config.STOCK_CODES
    train_s, val_s = dataset_builder.build_dataset(
        codes=codes,
        days=180,
        window=30,
        windows_per_stock=windows,
        val_ratio=val_ratio,
        tokenizer=None,         # 若内部需要可自行加载
        max_tokens=max_tokens,  # ★ Prompt 长度上限
    )

    # 2) 判断是否需要教师标注
    label_needed = True
    if skip_teacher and not overwrite:
        label_needed = not Path("labeled_data.jsonl").exists()
        if val_ratio:
            label_needed |= not Path("val_labeled_data.jsonl").exists()

    # 3) 教师模型打标
    if label_needed:
        print("[distill] ▶ 教师模型标注…")
        teacher_labeler.label_samples(
            [dataset_builder.format_prompt(s) for s in train_s],
            "labeled_data.jsonl",
        )
        if val_ratio:
            teacher_labeler.label_samples(
                [dataset_builder.format_prompt(s) for s in val_s],
                "val_labeled_data.jsonl",
            )
    else:
        print("[distill] ▶ 跳过教师标注，使用现有 JSONL")

    # 4) 清洗
    train_jsonl = _clean_once("labeled_data.jsonl")
    val_jsonl   = _clean_once("val_labeled_data.jsonl") if val_ratio else None

    # 5) LoRA 训练
    lora_cfg = train_lora.TrainConfig(
        data_path=train_jsonl,
        eval_path=val_jsonl,
        output_dir=output_dir,
        max_len=max_len,        # ★ 训练截断长度
        max_steps=200,
    )
    train_lora.main(lora_cfg)

    # 6) 评估
    prompts, refs = evaluate.load_dataset(val_jsonl or train_jsonl)
    metrics = evaluate.evaluate_model(output_dir, prompts, refs)
    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    Path(output_dir, "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))

# ╭─────────────────────────── CLI ───────────────────────────╮
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Distill pipeline")
    ap.add_argument("--windows", type=int, default=1, help="每只股票样本数")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Prompt 最大 token 长度")
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
