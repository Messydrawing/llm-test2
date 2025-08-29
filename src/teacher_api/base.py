"""Base classes and utilities for teacher model APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import yaml

# Default configuration path two directories above this file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "teacher_models.yaml"


def load_teacher_config(name: str, config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration for a teacher model by name.

    Parameters
    ----------
    name: str
        Name of the teacher model.
    config_path: str | Path | None
        Optional path to the configuration file. If ``None`` the default
        ``configs/teacher_models.yaml`` relative to the project root is used.

    Returns
    -------
    dict
        Configuration dictionary for the requested teacher model.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for teacher in data.get("teachers", []):
        if teacher.get("name") == name:
            return teacher
    raise KeyError(f"Teacher model '{name}' not found in {path}")


class TeacherModelAPI(ABC):
    """Abstract interface for teacher model backends."""

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """Generate an answer for the given prompt."""
        raise NotImplementedError
