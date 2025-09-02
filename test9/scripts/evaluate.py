#!/usr/bin/env python
"""Evaluate a PPO model on a held-out dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import load_model, load_tokenizer
from data_utils import load_dataset
from evaluation import evaluate_dataset


def evaluate(model_path: str, test_path: str, max_new_tokens: int = 64) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)

    data = load_dataset(test_path)
    outputs = []
    references = []
    for ex in data:
        inputs = tokenizer(ex["prompt"], return_tensors="pt")
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        outputs.append(text)
        references.append(ex.get("completion"))

    metrics = evaluate_dataset(outputs, references)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    default_model = Path(__file__).resolve().parent.parent / "models" / "ppo"
    default_test = Path(__file__).resolve().parent.parent / "data" / "test.jsonl"
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--test-path", type=str, default=str(default_test))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    evaluate(args.model_path, args.test_path, args.max_new_tokens)
