"""Evaluation script for student models."""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from .inference import load_student, call_student


def load_dataset(path: str) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    refs: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            label = rec.get("label", "")
            if isinstance(label, dict):
                refs.append(json.dumps(label, ensure_ascii=False))
            else:
                refs.append(str(label))
    return prompts, refs


def generate_predictions(model_name: str, prompts: Iterable[str]) -> list[str]:
    tokenizer, model = load_student(model_name)
    preds: list[str] = []
    for p in prompts:
        preds.append(call_student(tokenizer, model, p))
    return preds


def bleu_score(references: list[str], predictions: list[str]) -> float:
    smooth = SmoothingFunction().method4
    scores = []
    for ref, pred in zip(references, predictions):
        scores.append(
            sentence_bleu(
                [ref.split()], pred.split(), smoothing_function=smooth
            )
        )
    return sum(scores) / len(scores)


def rouge_l(references: list[str], predictions: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, pred in zip(references, predictions):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return sum(scores) / len(scores)


def embedding_similarity(
    references: list[str], predictions: list[str]
) -> float:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sims = []
    for ref, pred in zip(references, predictions):
        ref_emb = model.encode(ref, convert_to_tensor=True)
        pred_emb = model.encode(pred, convert_to_tensor=True)
        sims.append(util.cos_sim(ref_emb, pred_emb).item())
    return sum(sims) / len(sims)


def evaluate_model(
    model_name: str, prompts: list[str], refs: list[str]
) -> dict[str, float]:
    preds = generate_predictions(model_name, prompts)
    return {
        "bleu": bleu_score(refs, preds),
        "rougeL": rouge_l(refs, preds),
        "embed": embedding_similarity(refs, preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument(
        "--data", default="labeled_data.jsonl", help="Labeled prompts"
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen1.5-0.5B", help="Base student model"
    )
    parser.add_argument(
        "--tuned-model", required=True, help="Fine-tuned model path"
    )
    args = parser.parse_args()

    prompts, refs = load_dataset(args.data)

    print("Generating predictions with base model...")
    base_scores = evaluate_model(args.base_model, prompts, refs)
    print("Generating predictions with tuned model...")
    tuned_scores = evaluate_model(args.tuned_model, prompts, refs)

    def fmt(d: dict[str, float]) -> str:
        return ", ".join(f"{k}: {v:.4f}" for k, v in d.items())

    print("Before fine-tuning:", fmt(base_scores))
    print("After fine-tuning:", fmt(tuned_scores))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
