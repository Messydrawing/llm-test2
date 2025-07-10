from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any, Sequence

from .inference import call_teacher


def label_samples(
    samples: Sequence[dict[str, Any]],
    output_file: str | Path = "labeled_data.jsonl",
) -> list[dict[str, Any]]:
    """Label ``samples`` using the teacher model.

    Each sample is formatted with :func:`dataset_builder.format_prompt` and
    sent to :func:`call_teacher`. JSON answers are recorded along with the
    ground-truth ``target`` field.
    """
    from .dataset_builder import format_prompt

    path = Path(output_file)
    labeled: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            prompt = format_prompt(sample)
            ans = call_teacher(prompt)
            try:
                label = json.loads(ans["content"])
                if ans["reasoning"]:
                    label["reasoning"] = ans["reasoning"]
            except json.JSONDecodeError:
                label = {"raw": ans["content"], "reasoning": ans["reasoning"]}
            record = {
                "prompt": prompt,
                "label": label,
                "target": sample.get("target", ""),
            }
            labeled.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return labeled
