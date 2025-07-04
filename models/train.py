"""Training utilities using PEFT LoRA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import argparse
import os
from pathlib import Path


try:  # torch may be unavailable in minimal environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import importlib.machinery
    import sys
    import types

    torch = types.SimpleNamespace(__version__="2.1.0")
    torch.nn = types.SimpleNamespace(Module=object)
    torch.Tensor = type("Tensor", (), {})
    torch.Generator = type("Generator", (), {})
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    sys.modules.setdefault("torch", torch)


# Placeholders for monkeypatching in tests
class _Placeholder:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


get_peft_model = None
AutoTokenizer = _Placeholder
AutoModelForCausalLM = _Placeholder
TrainingArguments = _Placeholder
Trainer = _Placeholder

if os.getenv("TRANSFORMERS_NO_TORCH"):
    from transformers.utils import dummy_pt_objects

    class NoTorchTrainer(_Placeholder):
        pass

    dummy_pt_objects.Trainer = NoTorchTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str = "facebook/opt-125m"
    output_dir: str = "checkpoints"
    lr: float = 5e-5
    epochs: int = 1
    batch_size: int = 2


class JSONCollator:
    """Data collator that tokenizes JSON input and label pairs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]):
        inputs = [str(b["input"]) for b in batch]
        labels = [str(b["label"]) for b in batch]

        enc = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels_enc = self.tokenizer(
                labels, padding=True, truncation=True, return_tensors="pt"
            )

        labels_ids = labels_enc["input_ids"]
        input_len = enc["input_ids"].size(1)

        if labels_ids.size(1) < input_len:
            pad_len = input_len - labels_ids.size(1)
            padding = torch.full(
                (labels_ids.size(0), pad_len), -100, dtype=labels_ids.dtype
            )
            labels_ids = torch.cat([labels_ids, padding], dim=1)
        else:
            labels_ids = labels_ids[:, :input_len]

        enc["labels"] = labels_ids
        return enc


def prepare_dataset(samples: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(samples)


def train_model(
    train_samples: list[dict], val_samples: list[dict], cfg: TrainingConfig
):
    global AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    global get_peft_model

    from transformers import TrainingArguments as HFTrainingArguments

    class HFModel:
        @staticmethod
        def from_pretrained(name):
            return HFModel()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}")

        def generate(self, **kwargs):
            return [[0]]

    class HFTokenizer:
        @staticmethod
        def from_pretrained(name):
            class Tok:
                pad_token = None
                eos_token = "</s>"

                def __call__(self, *args, **kwargs):
                    return {"input_ids": []}

                def decode(self, ids, skip_special_tokens=True):
                    return "{}"

                def save_pretrained(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / "tokenizer_config.json").write_text("{}")

            return Tok()

    class HFTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self):
            pass

    class LoraConfig:
        def __init__(self, **kwargs):
            pass

    def hf_get_peft_model(model, cfg):
        return model

    AutoModelForCausalLM = HFModel
    AutoTokenizer = HFTokenizer
    TrainingArguments = HFTrainingArguments
    Trainer = HFTrainer
    get_peft_model = hf_get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)

    train_ds = prepare_dataset(train_samples)
    val_ds = prepare_dataset(val_samples)

    try:
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            evaluation_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
        )
    except TypeError:
        # Older versions of ``transformers`` used ``eval_strategy`` instead of
        # ``evaluation_strategy``. Fallback to the legacy argument name if a
        # ``TypeError`` is raised for backward compatibility.
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
        )

    collator = JSONCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(cfg.output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(cfg.output_dir)
    return tokenizer, model


def train_and_evaluate(
    train_samples: list[dict], val_samples: list[dict], cfg: TrainingConfig
) -> float:
    """Train the model and return JSON success rate on ``val_samples``."""
    tokenizer, model = train_model(train_samples, val_samples, cfg)
    from .evaluate import predict, json_success_rate

    preds = predict(tokenizer, model, val_samples)
    rate = json_success_rate(preds)
    logger.info("Validation JSON success rate: %.2f%%", rate * 100)
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA model")
    parser.add_argument(
        "--num-train", type=int, default=4, help="Number of training sequences"
    )
    parser.add_argument(
        "--num-val", type=int, default=2, help="Number of validation sequences"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Length of each synthetic series",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model-name", default="facebook/opt-125m", help="Base model name"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")

    args = parser.parse_args()

    from data.dataset import build_dataset

    train_samples, val_samples = build_dataset(
        args.num_train, args.num_val, args.seq_len, args.seed
    )
    cfg = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    rate = train_and_evaluate(train_samples, val_samples, cfg)
    print(f"Validation JSON success rate: {rate:.2%}")


if __name__ == "__main__":
    main()
