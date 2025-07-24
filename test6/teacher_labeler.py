from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from .inference import call_teacher as _call_teacher


def label_samples(
    prompts: Iterable[str],
    output_file: str | Path = "labeled_data.jsonl",
    *,
    call_teacher=_call_teacher,
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
            answer = ans.get("content", ans) if isinstance(ans, dict) else ans
            try:
                label = json.loads(answer)
            except (TypeError, json.JSONDecodeError):
                label = {"raw": answer}
            if isinstance(ans, dict) and ans.get("reasoning"):
                label["reasoning"] = ans["reasoning"]
            record = {"prompt": prompt, "label": label}
            labeled.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return labeled
