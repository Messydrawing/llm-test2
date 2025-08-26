"""Automatic labeling of dataset samples using a teacher model or API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .dataset_builder import format_prompt


def call_teacher(prompt: str) -> dict[str, str]:
    """Call external teacher model if ``ARK_API_KEY`` is set.

    When no API key is found or a request fails, a fallback empty response is
    returned so the rest of the pipeline can continue offline.
    """

    api_key = os.getenv("ARK_API_KEY", "")
    if not api_key:
        return {"content": "{}", "reasoning": ""}
    try:  # pragma: no cover - network
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=api_key)
        resp = client.chat.completions.create(
            model="deepseek-r1-250528",
            messages=[{"role": "user", "content": prompt}],
        )
        msg = resp.choices[0].message
        content = msg.content.strip()
        reasoning = getattr(msg, "reasoning_content", "").strip()
        return {"content": content, "reasoning": reasoning}
    except Exception as e:
        return {"content": f"{{}}", "reasoning": f"[error: {e}]"}


def label_dataset(samples: Iterable[dict], out_dir: str | Path = ".") -> None:
    """Label ``samples`` and write multitask JSONL files.

    Three files are produced in ``out_dir``:
    ``train_trend.jsonl`` for prediction labels,
    ``train_advice.jsonl`` for advice generation and
    ``train_explain.jsonl`` for analysis/description tasks.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trend_f = (out_path / "train_trend.jsonl").open("w", encoding="utf-8")
    advice_f = (out_path / "train_advice.jsonl").open("w", encoding="utf-8")
    explain_f = (out_path / "train_explain.jsonl").open("w", encoding="utf-8")

    for sample in samples:
        prompt = format_prompt(sample)
        ans = call_teacher(prompt)
        try:
            data = json.loads(ans.get("content", "{}"))
        except json.JSONDecodeError:
            data = {}
        # Fallback for trend if teacher doesn't provide one
        if not data.get("prediction"):
            change = sample.get("change", 0)
            if change > 3:
                data["prediction"] = "up"
            elif change < -3:
                data["prediction"] = "down"
            else:
                data["prediction"] = "stable"
        trend_f.write(json.dumps({"prompt": prompt, "label": data.get("prediction", "")}, ensure_ascii=False) + "\n")
        advice_f.write(json.dumps({"prompt": prompt, "label": data.get("advice", "")}, ensure_ascii=False) + "\n")
        explain_label = data.get("analysis") or ans.get("reasoning", "")
        explain_f.write(json.dumps({"prompt": prompt, "label": explain_label}, ensure_ascii=False) + "\n")

    trend_f.close()
    advice_f.close()
    explain_f.close()
