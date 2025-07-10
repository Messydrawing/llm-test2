from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import inspect

import torch
import torch.optim.lr_scheduler as lr_sched

if not hasattr(lr_sched, "LRScheduler"):
    lr_sched.LRScheduler = lr_sched._LRScheduler

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


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
    max_len: int = 4096


IGNORE_INDEX = -100


def _load_dataset(path: str) -> Dataset:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"].strip()
            label = rec["label"]
            if isinstance(label, dict):
                label = json.dumps(label, ensure_ascii=False, sort_keys=True)
            else:
                label = str(label).strip()
            recs.append(
                {
                    "prompt": prompt,
                    "label": label,
                    "target": rec.get("target", ""),
                }
            )
    return Dataset.from_list(recs)


class LabelCollator:
    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch):
        texts, prompt_lens = [], []
        for rec in batch:
            prompt = rec["prompt"].strip()
            label = rec["label"].strip()
            full = f"{prompt}\n\n### 答案：{label}"
            plen = len(self.tok(prompt + "\n\n### 答案：")["input_ids"])
            prompt_lens.append(plen)
            texts.append(full)

        enc = self.tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[:, :] = IGNORE_INDEX
        for i, plen in enumerate(prompt_lens):
            labels[i, plen : enc["input_ids"].size(1)] = enc["input_ids"][i, plen:]

        if (labels != IGNORE_INDEX).sum() == 0:
            raise ValueError(
                "\u5168\u90e8 label \u88ab\u622a\u6389\uff1b\u8bf7\u7f29\u77ed prompt \u6216\u589e\u5927 --max-len"
            )

        enc["labels"] = labels
        return enc


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

    args_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs or 1,
        max_steps=cfg.max_steps or -1,
        learning_rate=cfg.lr,
        logging_steps=1,
        remove_unused_columns=False,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        args_kwargs["evaluation_strategy"] = "epoch" if eval_ds else "no"
    if "save_strategy" in sig.parameters:
        args_kwargs["save_strategy"] = "epoch"

    args = TrainingArguments(**args_kwargs)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)

    collator = LabelCollator(tokenizer, cfg.max_len)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled_data.jsonl")
    ap.add_argument("--eval-data")
    ap.add_argument("--base-model", default="Qwen/Qwen1.5-7B")
    ap.add_argument("--out", default="lora_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-len", type=int, default=4096)
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
