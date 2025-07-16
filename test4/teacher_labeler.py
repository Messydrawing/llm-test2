from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from .inference import call_teacher


def label_samples(
    prompts: Iterable[str], output_file: str | Path = "labeled_data.jsonl"
) -> list[dict[str, Any]]:
    """Label ``prompts`` using the teacher model.

    Each prompt is sent to :func:`call_teacher`. The returned text is parsed
    as JSON if possible and validated to ensure it contains ``prediction``,
    ``analysis`` and ``advice`` fields. Missing fields are filled with empty
    strings and all values are coerced to ``str``.

    Cleaned records are written to ``output_file`` in JSON Lines format and a
    list of them is returned.
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

            if not isinstance(label, dict):
                label = {"raw": str(label)}

            # ensure required fields exist and are strings
            label["prediction"] = str(label.get("prediction", ""))
            label["analysis"] = str(label.get("analysis", ""))
            label["advice"] = str(label.get("advice", ""))

            if isinstance(ans, dict) and ans.get("reasoning"):
                label["reasoning"] = str(ans["reasoning"])

            record = {"prompt": prompt, "label": label}
            labeled.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return labeled
