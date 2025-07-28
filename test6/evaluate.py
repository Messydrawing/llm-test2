"""Evaluation script for student models."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from .inference import load_student, call_student


def load_dataset(
    path: str,
    *,
    sample: int | None = None,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return prompts and references from ``path``.

    When ``sample`` is given, randomly select that many records using ``seed``.
    """
    prompts: list[str] = []
    refs: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            label = (
                rec.get("label")
                or rec.get("answer")
                or rec.get("reference")
                or ""
            )
            if isinstance(label, dict):
                refs.append(
                    json.dumps(label, ensure_ascii=False, sort_keys=True)
                )
            else:
                refs.append(str(label))

    if sample is not None and sample < len(prompts):
        rng = random.Random(seed)
        idxs = rng.sample(range(len(prompts)), sample)
        prompts = [prompts[i] for i in idxs]
        refs = [refs[i] for i in idxs]

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
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters")
    parser.add_argument(
        "--questions", default="question.jsonl", help="Prompt dataset"
    )
    parser.add_argument("--sample", type=int, default=20, help="Sample size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--models-dir",
        default="lora_adapter",
        help="Directory containing lora_D/G/Q subfolders",
    )
    args = parser.parse_args()

    prompts, refs = load_dataset(
        args.questions, sample=args.sample, seed=args.seed
    )

    results: dict[str, dict[str, float]] = {}
    for tag in ("D", "G", "Q"):
        model_path = os.path.join(args.models_dir, f"lora_{tag}")
        print(f"Evaluating {model_path} …")
        results[tag] = evaluate_model(model_path, prompts, refs)

    print(f"{'Model':<8}{'BLEU':>8}{'ROUGE-L':>10}{'Embed':>8}")
    for tag in ("D", "G", "Q"):
        sc = results[tag]
        print(
            f"{tag:<8}{sc['bleu']:>8.4f}{sc['rougeL']:>10.4f}{sc['embed']:>8.4f}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
