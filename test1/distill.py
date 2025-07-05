from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

import argparse

from . import config, dataset_builder, evaluate, teacher_labeler, train_lora


def _download_model(model_name: str, cache_dir: str = "models") -> str:
    """Ensure ``model_name`` is downloaded and return its local path."""
    dest = Path(cache_dir) / model_name.replace("/", "_")
    if not dest.exists():
        print(f"Downloading model {model_name} to {dest}...")
        snapshot_download(model_name, local_dir=dest, local_dir_use_symlinks=False)
    else:
        print(f"Model already available at {dest}")
    return str(dest)


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
        data_path="labeled_data.jsonl",
        eval_path="val_labeled_data.jsonl",
        output_dir=output_dir,
        max_steps=200,
    )
    cfg.base_model = _download_model(cfg.base_model)

    codes = config.STOCK_CODES if stock is None else [stock]
    train_samples, val_samples = dataset_builder.build_dataset(
       codes,            # ← 一次把 8 只股票全传进去
       days=180,
       window=30,
       windows_per_stock=windows,  # 仍然是“每股 50”
       val_ratio=val_ratio,
    )


    need_label = not skip_teacher
    need_label |= overwrite
    need_label |= not Path("labeled_data.jsonl").exists()
    need_label |= (val_ratio > 0 and not Path("val_labeled_data.jsonl").exists())

    if need_label:
        train_prompts = [dataset_builder.format_prompt(s) for s in train_samples]
        val_prompts = [dataset_builder.format_prompt(s) for s in val_samples]
        teacher_labeler.label_samples(train_prompts, "labeled_data.jsonl")
        if val_ratio > 0:
            teacher_labeler.label_samples(val_prompts, "val_labeled_data.jsonl")
        else:
            print("[distill] Skip teacher labeling - reuse existing JSONL")

    train_lora.main(cfg)

    prompts, refs = evaluate.load_dataset("val_labeled_data.jsonl")
    metrics = evaluate.evaluate_model(cfg.output_dir, prompts, refs)
    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    Path("metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Run the distillation pipeline"
    )
    parser.add_argument(
        "--windows", type=int, default=1, help="Windows per stock"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for validation",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Do NOT call teacher; use existiong labeled_data.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-label even if JSONL already exists",
    )
    parser.add_argument(
        "--out", default="lora_adapter", help="Output directory"
    )
    parser.add_argument("--stock", help="Stock code to process")
    args = parser.parse_args()
    main(
        stock=args.stock,
        windows=args.windows,
        val_ratio=args.val_ratio,
        skip_teacher=args.skip_teacher,
        overwrite=args.overwrite,
        output_dir=args.out,
    )