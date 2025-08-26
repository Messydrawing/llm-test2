from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from test8.config import (
    DATA_DIR,
    BASE_MODEL_PATH,
    TREND_MODEL_PATH,
    ADVICE_MODEL_PATH,
    EXPLANATION_MODEL_PATH,
    MERGED_MODEL_PATH,
)

from datasets import load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load a JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_trend(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: List[Dict[str, str]]
) -> float:
    """Compute accuracy for trend prediction task."""
    if not data:
        return 0.0
    correct = 0
    for sample in data:
        prompt = sample.get("prompt", "")
        reference = sample.get("completion", "").strip().lower()
        prediction = generate_text(model, tokenizer, prompt).strip().lower()
        if prediction == reference:
            correct += 1
    return correct / len(data)


def evaluate_bleu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[Dict[str, str]],
    metric,
) -> float:
    """Compute BLEU score for advice/explanation tasks."""
    if not data:
        return 0.0
    predictions: List[List[str]] = []
    references: List[List[List[str]]] = []
    for sample in data:
        prompt = sample.get("prompt", "")
        reference = sample.get("completion", "")
        prediction = generate_text(model, tokenizer, prompt)
        predictions.append(prediction.split())
        references.append([reference.split()])
    bleu = metric.compute(predictions=predictions, references=references)
    return float(bleu["bleu"])


def load_model(path: Path) -> Tuple[AutoModelForCausalLM | None, AutoTokenizer | None]:
    """Load model and tokenizer from ``path``."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        return model, tokenizer
    except Exception as exc:  # pragma: no cover - graceful degradation
        print(f"Warning: failed to load model at {path}: {exc}")
        return None, None


def call_teacher_api(prompt: str) -> str:  # pragma: no cover - placeholder
    """Placeholder for Teacher_32B API call."""
    raise NotImplementedError("Implement API call for Teacher_32B model")


def main() -> None:  # pragma: no cover - script entry point
    data_dir = DATA_DIR
    trend_data = load_jsonl(data_dir / "test_trend.jsonl")
    advice_data = load_jsonl(data_dir / "test_advice.jsonl")
    explain_data = load_jsonl(data_dir / "test_explain.jsonl")

    metric = load_metric("bleu")

    model_paths = {
        "Base_7B": BASE_MODEL_PATH,
        "TrendModel_7B": TREND_MODEL_PATH,
        "AdviceModel_7B": ADVICE_MODEL_PATH,
        "ExplanationModel_7B": EXPLANATION_MODEL_PATH,
        "MergedModel": MERGED_MODEL_PATH,
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, path in model_paths.items():
        model, tokenizer = load_model(path)
        if model is None or tokenizer is None:
            continue
        results[name] = {
            "trend_accuracy": evaluate_trend(model, tokenizer, trend_data),
            "advice_bleu": evaluate_bleu(model, tokenizer, advice_data, metric),
            "explain_bleu": evaluate_bleu(model, tokenizer, explain_data, metric),
        }

    # Teacher model evaluation (placeholder)
    try:
        for sample in trend_data:
            call_teacher_api(sample.get("prompt", ""))
    except NotImplementedError:
        print("Teacher_32B evaluation skipped: API not implemented.")

    out_json = ROOT / "evaluation_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    header = "| Model | Trend Accuracy | Advice BLEU | Explain BLEU |\n|---|---|---|---|\n"
    rows = [
        f"| {name} | {res['trend_accuracy']:.4f} | {res['advice_bleu']:.4f} | {res['explain_bleu']:.4f} |"
        for name, res in results.items()
    ]
    table = header + "\n".join(rows)
    out_md = ROOT / "evaluation_results.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
