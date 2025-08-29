"""Implementation of the DeepSeek teacher model API."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import TeacherModelAPI, load_teacher_config


class DeepSeekAPI(TeacherModelAPI):
    """Teacher model wrapper for DeepSeek's OpenAI compatible API."""

    def __init__(self, config_path: str | None = None) -> None:
        cfg = load_teacher_config("deepseek", config_path)
        api_key: str | None = cfg.get("api_key")
        base_url: str | None = cfg.get("base_url")
        self.model: str = cfg.get("model", "")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using DeepSeek's API."""
        try:
            resp: Any = self.client.responses.create(model=self.model, input=prompt)
            return getattr(resp, "output_text", "")
        except Exception:
            # Return empty string on failure to keep interface simple
            return ""
