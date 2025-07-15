from __future__ import annotations

import argparse
import json
from typing import Any

from .inference import load_student, call_student


def load_sample(path: str, idx: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError("Sample index out of range")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and tuned model outputs for a dataset sample")
    parser.add_argument("--data", required=True, help="Path to labeled dataset in JSONL format")
    parser.add_argument("--index", type=int, default=0, help="Sample index to evaluate")
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-0.5B", help="Base student model name or path")
    parser.add_argument("--tuned-model", required=True, help="Fine-tuned model path")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for each model",
    )
    args = parser.parse_args()

    record = load_sample(args.data, args.index)
    prompt = record.get("prompt")
    if prompt is None:
        raise KeyError("Record does not contain 'prompt'")

    print(f"Prompt:\n{prompt}\n")
    label = record.get("label")
    if label is not None:
        print("Reference label:")
        print(json.dumps(label, ensure_ascii=False))
        print()

    base_tok, base_model = load_student(args.base_model)
    tuned_tok, tuned_model = load_student(args.tuned_model)

    base_out = call_student(
        base_tok, base_model, prompt, max_new_tokens=args.max_tokens
    )
    tuned_out = call_student(
        tuned_tok, tuned_model, prompt, max_new_tokens=args.max_tokens
    )

    print("Base model output:")
    print(base_out)
    print("\nTuned model output:")
    print(tuned_out)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()