# test1/distill.py  ── v4: auto-clean JSONL  ─────────────────────────
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from . import (
    config,
    dataset_builder,
    evaluate,
    teacher_labeler,
    train_lora,
)

# === 引入清洗工具 ===
from .clean_jsonl import main as clean_jsonl_main  # 确保 clean_jsonl.py 位于 test1/ 目录
# --------------------------------------------------------------------


def _download_model(model_name: str, cache_dir: str = "models") -> str:
    """Ensure ``model_name`` is downloaded and return its local path."""
    dest = Path(cache_dir) / model_name.replace("/", "_")
    if not dest.exists():
        print(f"Downloading model {model_name} to {dest}...")
        snapshot_download(model_name, local_dir=dest, local_dir_use_symlinks=False)
    else:
        print(f"Model already available at {dest}")
    return str(dest)


def _clean_once(in_path: str | Path) -> str:
    """Run clean_jsonl and return cleaned file path."""
    inp = Path(in_path)
    out = inp.with_name(f"cleaned_{inp.name}")
    if not inp.exists():
        sys.exit(f"[distill] ❌  expected {inp} but not found")
    print(f"[distill] cleaning {inp.name} → {out.name}")
    clean_jsonl_main(inp, out)          # 调用清洗函数
    return str(out)


def main(
    *,
    stock: str | None = None,
    windows: int = 1,
    val_ratio: float = 0.2,
    skip_teacher: bool = False,
    overwrite: bool = False,
    output_dir: str = "lora_adapter",
) -> None:
    cfg = train_lora.TrainConfig(
        data_path="cleaned_labeled_data.jsonl",         # ← 训练始终用 cleaned_*
        eval_path="cleaned_val_labeled_data.jsonl",
        output_dir=output_dir,
        max_steps=200,
    )
    cfg.base_model = _download_model(cfg.base_model)

    # 1️⃣  构造窗口
    codes = config.STOCK_CODES if stock is None else [stock]
    train_samples, val_samples = dataset_builder.build_dataset(
        codes,
        days=180,
        window=30,
        windows_per_stock=windows,
        val_ratio=val_ratio,
    )

    # 2️⃣  判断是否需要重新向教师提问
    need_label = not skip_teacher
    need_label |= overwrite
    need_label |= not Path("labeled_data.jsonl").exists()
    need_label |= val_ratio > 0 and not Path("val_labeled_data.jsonl").exists()

    # 3️⃣  调教师模型
    if need_label:
        train_prompts = [dataset_builder.format_prompt(s) for s in train_samples]
        teacher_labeler.label_samples(train_prompts, "labeled_data.jsonl")

        if val_ratio > 0:
            val_prompts = [dataset_builder.format_prompt(s) for s in val_samples]
            teacher_labeler.label_samples(val_prompts, "val_labeled_data.jsonl")
    else:
        print("[distill] Skip teacher labeling – reuse existing JSONL")

    # 4️⃣  清洗（若 cleaned_ 文件不存在或 overwrite）
    train_jsonl = _clean_once("labeled_data.jsonl")
    val_jsonl = (
        _clean_once("val_labeled_data.jsonl") if val_ratio > 0 else None
    )

    # 5️⃣  LoRA 训练
    cfg.data_path = train_jsonl
    cfg.eval_path = val_jsonl
    train_lora.main(cfg)

    # 6️⃣  评估
    prompts, refs = evaluate.load_dataset(val_jsonl or train_jsonl)
    metrics = evaluate.evaluate_model(cfg.output_dir, prompts, refs)
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    Path("metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))


# ------------------------- CLI --------------------------
if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the distillation pipeline")
    parser.add_argument("--windows", type=int, default=1, help="Windows per stock")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--skip-teacher", action="store_true", help="Use existing JSONL, no teacher call")
    parser.add_argument("--overwrite", action="store_true", help="Force re-label / re-clean")
    parser.add_argument("--out", default="lora_adapter", help="Output directory")
    parser.add_argument("--stock", help="Single stock code to process")
    args = parser.parse_args()

    main(
        stock=args.stock,
        windows=args.windows,
        val_ratio=args.val_ratio,
        skip_teacher=args.skip_teacher,
        overwrite=args.overwrite,
        output_dir=args.out,
    )
