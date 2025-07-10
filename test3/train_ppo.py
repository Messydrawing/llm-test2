from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


@dataclass
class RLConfig:
    model_path: str = "lora_adapter"
    data_path: str = "cleaned_labeled_data.jsonl"
    batch_size: int = 1
    epochs: int = 1
    lr: float = 1e-5
    max_len: int = 512


def _load_dataset(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append({"prompt": rec["prompt"], "target": rec.get("target", "")})
    return records


def compute_reward(answer: str, target: str) -> float:
    try:
        data = json.loads(answer)
        fmt = 1 if all(k in data for k in ("prediction", "analysis", "advice")) else 0
        pred = data.get("prediction", "")
    except Exception:
        fmt = 0
        pred = ""
    acc = 1 if pred == target else 0
    return fmt + acc


def main(cfg: RLConfig) -> None:
    data = _load_dataset(cfg.data_path)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.model_path,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.model_path,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )

    ppo_config = PPOConfig(batch_size=cfg.batch_size, learning_rate=cfg.lr)
    trainer = PPOTrainer(ppo_config, base_model, ref_model, tokenizer)

    for _ in range(cfg.epochs):
        for rec in data:
            query_tensors = tokenizer(rec["prompt"], return_tensors="pt").to(base_model.device)
            response = trainer.generate(query_tensors, max_new_tokens=128)
            answer = tokenizer.decode(response[0][query_tensors["input_ids"].size(1):], skip_special_tokens=True)
            reward = compute_reward(answer, rec["target"])
            trainer.step(query_tensors, response, torch.tensor([reward]).to(base_model.device))

    trainer.model.save_pretrained(cfg.model_path)
    tokenizer.save_pretrained(cfg.model_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="lora_adapter")
    ap.add_argument("--data", default="cleaned_labeled_data.jsonl")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = RLConfig(
        model_path=args.model_path,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    main(cfg)
