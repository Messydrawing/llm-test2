#!/usr/bin/env python
"""Inference script for PPO fine-tuned models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from model_utils import load_model, load_tokenizer


def generate(prompts: List[str], model_path: str, max_new_tokens: int = 64) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = {"text": text}
        print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a PPO model")
    default_model = Path(__file__).resolve().parent.parent / "models" / "ppo"
    parser.add_argument("prompt", nargs="*", help="Prompt strings")
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--input-file", type=str, help="File containing prompts", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    prompts = list(args.prompt)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        raise SystemExit("No prompts provided")

    generate(prompts, args.model_path, args.max_new_tokens)
