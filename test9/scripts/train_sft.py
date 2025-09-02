#!/usr/bin/env python
"""Simple supervised fine-tuning entry point.

The script wires together helper utilities defined in ``test9/src`` to
train a language model with Hugging Face's ``Trainer``.  It loads a YAML
configuration, prepares datasets and persists the resulting checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from model_utils import load_model, load_tokenizer, save_model
from data_utils import make_datasets

try:  # pragma: no cover - optional dependency
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
except Exception:  # pragma: no cover - transformers not installed
    Trainer = TrainingArguments = DataCollatorForLanguageModeling = None  # type: ignore


def train_sft(config_path: str, output_dir: str) -> None:
    """Train a model using supervised fine-tuning."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = Path(config_path).parent
    model_name = cfg["model_name"]
    train_path = (cfg_dir / cfg["train_path"]).resolve()
    val_path = (cfg_dir / cfg["val_path"]).resolve()
    training_args_cfg = cfg.get("training_args", {})

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    train_ds, val_ds = make_datasets(str(train_path), str(val_path), tokenizer)

    if Trainer is None:  # pragma: no cover - handled in environments without transformers
        raise ImportError("transformers library is required for training")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(output_dir=output_dir, **training_args_cfg)
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    save_model(model, tokenizer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning")
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "sft_config.yaml"
    default_out = Path(__file__).resolve().parent.parent / "models" / "sft"
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--output_dir", type=str, default=str(default_out))
    args = parser.parse_args()
    train_sft(args.config, args.output_dir)
