"""Factory for teacher model API implementations."""

from __future__ import annotations

from .base import TeacherModelAPI
from .deepseek_api import DeepSeekAPI
from .gemini_api import GeminiAPI
from .qwen3_api import Qwen3API


def get_teacher_model(name: str) -> TeacherModelAPI:
    """Return a concrete :class:`TeacherModelAPI` implementation by name."""
    normalized = name.lower()
    if normalized == "deepseek":
        return DeepSeekAPI()
    if normalized == "qwen3":
        return Qwen3API()
    if normalized == "gemini":
        return GeminiAPI()
    raise ValueError(f"Unknown teacher model: {name}")

__all__ = [
    "TeacherModelAPI",
    "DeepSeekAPI",
    "Qwen3API",
    "GeminiAPI",
    "get_teacher_model",
]
