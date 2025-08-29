"""Implementation of the Gemini teacher model API."""

from __future__ import annotations

from typing import Any

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

from .base import TeacherModelAPI, load_teacher_config


class GeminiAPI(TeacherModelAPI):
    """Teacher model wrapper for Google's Gemini API."""

    def __init__(self, config_path: str | None = None) -> None:
        if genai is None:
            raise RuntimeError("google-genai package is required for GeminiAPI")
        cfg = load_teacher_config("gemini", config_path)
        api_key: str | None = cfg.get("api_key")
        self.model: str = cfg.get("model", "")
        self.client = genai.Client(api_key=api_key)

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using Gemini's API."""
        try:
            response: Any = self.client.models.generate_content(self.model, prompt)
            return getattr(response, "text", "")
        except Exception:
            return ""
