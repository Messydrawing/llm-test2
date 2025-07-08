from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen1.5-7B"
    data_path: str = "labeled_data.jsonl"
    eval_path: str | None = None
    output_dir: str = "lora_adapter"
    batch_size: int = 1
    lr: float = 2e-4
    epochs: int | None = 1
    max_steps: int | None = None
    grad_accum: int = 4
    max_len: int = 1024


def _load_dataset(path: str) -> Dataset:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"].strip()
            label = rec["label"]
            if isinstance(label, dict):
                label = json.dumps(label, ensure_ascii=False, sort_keys=True)
            else:
                label = str(label).strip()
            texts.append(f"{prompt}\n\n### 答案：{label}")
    return Dataset.from_dict({"text": texts})


def main(cfg: TrainConfig) -> None:
    train_ds = _load_dataset(cfg.data_path)
    eval_ds = _load_dataset(cfg.eval_path) if cfg.eval_path else None

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs or 1,
        max_steps=cfg.max_steps or -1,
        learning_rate=cfg.lr,
        logging_steps=1,
        evaluation_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_cfg,
        max_seq_length=cfg.max_len,
        dataset_text_field="text",
        args=args,
    )
    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled_data.jsonl")
    ap.add_argument("--eval-data")
    ap.add_argument("--base-model", default="Qwen/Qwen1.5-7B")
    ap.add_argument("--out", default="lora_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = TrainConfig(
        base_model=args.base_model,
        data_path=args.data,
        eval_path=args.eval_data,
        output_dir=args.out,
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    main(cfg)
