"""Implementation of the Qwen3 teacher model API."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import TeacherModelAPI, load_teacher_config


class Qwen3API(TeacherModelAPI):
    """Teacher model wrapper for Qwen3 using the DashScope compatible API."""

    def __init__(self, config_path: str | None = None) -> None:
        cfg = load_teacher_config("qwen3", config_path)
        api_key: str | None = cfg.get("api_key")
        base_url: str | None = cfg.get("base_url")
        self.model: str = cfg.get("model", "")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using Qwen3's API."""
        try:
            resp: Any = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            message = resp.choices[0].message
            return getattr(message, "content", "")
        except Exception:
            return ""
