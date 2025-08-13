"""构建训练/验证集数据集并保存为 JSONL。

该脚本从配置文件读取股票代码、模板等信息，
调用 :func:`src.data.dataset_builder.build_dataset` 获取
训练集和验证集的 :class:`PromptItem` 列表，
并将结果写入 ``train.jsonl`` 与 ``val.jsonl``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.data.dataset_builder import build_dataset, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build dataset")
    parser.add_argument("--cfg", required=True, help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg.get("symbols", [])
    template = cfg["template"]
    days = cfg.get("days", 180)
    window = cfg.get("summary", {}).get("window_days", 30)
    val_ratio = cfg.get("val_ratio", 0.2)
    balance = cfg.get("balance", True)
    max_tokens = cfg.get("max_tokens", 1024)
    tokenizer_name = cfg.get("tokenizer")
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_items, val_items = build_dataset(
        symbols,
        days=days,
        window=window,
        val_ratio=val_ratio,
        balance=balance,
        template=template,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        seed=cfg.get("seed"),
    )

    out_dir = Path(cfg.get("save_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_items, out_dir / "train.jsonl")
    save_jsonl(val_items, out_dir / "val.jsonl")


if __name__ == "__main__":
    main()

