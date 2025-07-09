from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from .inference import call_teacher


def label_samples(
    prompts: Iterable[str], output_file: str | Path = "labeled_data.jsonl"
) -> list[dict[str, Any]]:
    """Label ``prompts`` using the teacher model.

    Each prompt is sent to :func:`call_teacher` and the JSON response is
    collected. Labeled records are written to ``output_file`` in JSON lines
    format and also returned.
    """
    path = Path(output_file)
    labeled: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            ans = call_teacher(prompt)
            try:
                label = json.loads(ans["content"])
                if ans["reasoning"]:
                    label["reasoning"] = ans["reasoning"]
            except json.JSONDecodeError:
                label = {"raw": ans["content"], "reasoning": ans["reasoning"]}
            record = {"prompt": prompt, "label": label}
            labeled.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return labeled
