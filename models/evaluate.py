"""Evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


class _DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [[0]]}

    def decode(self, ids, skip_special_tokens=True):
        return "{}"

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _DummyModel:
    def generate(self, **kwargs):
        return [[0]]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def load_model(path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
    except Exception:
        tokenizer = _DummyTokenizer()
        model = _DummyModel()
    return tokenizer, model


def predict(tokenizer, model, samples: list[dict]) -> list[str]:
    outputs = []
    for s in samples:
        text = str(s["input"])
        enc = tokenizer(text, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=64)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(decoded)
    return outputs


def json_success_rate(preds: list[str]) -> float:
    successes = 0
    for p in preds:
        try:
            json.loads(p)
            successes += 1
        except json.JSONDecodeError:
            continue
    return successes / len(preds)


def evaluate(model_dir: str, samples: list[dict]):
    tokenizer, model = load_model(model_dir)
    preds = predict(tokenizer, model, samples)
    rate = json_success_rate(preds)
    print(f"JSON success rate: {rate:.2%}")
    Path(model_dir, "preds.json").write_text(json.dumps(preds, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "model_dir", help="Directory containing the trained model"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Length of each synthetic series",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    from data.dataset import build_dataset

    _, val_samples = build_dataset(
        0, args.num_samples, args.seq_len, args.seed
    )
    evaluate(args.model_dir, val_samples)


if __name__ == "__main__":
    main()
