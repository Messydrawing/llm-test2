"""A lightweight stub of the `peft` package used for tests.

This module provides minimal implementations of the classes and
functions used in the training scripts so that the real dependency is
not required.  The goal is simply to satisfy imports; no parameter
efficient fine-tuning is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class PeftModel(nn.Module):
    """Trivial wrapper around a model.

    It forwards all calls to the wrapped model and exposes
    ``save_pretrained`` so that the ``Trainer`` can persist weights.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):  # pragma: no cover - passthrough
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        self.model.save_pretrained(save_directory, *args, **kwargs)


@dataclass
class LoraConfig:
    """Placeholder configuration container."""

    kwargs: dict[str, Any] = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def get_peft_model(model: nn.Module, config: LoraConfig) -> PeftModel:
    """Return the model wrapped in :class:`PeftModel`.

    The function mimics the real ``get_peft_model`` signature but does
    not modify the model parameters.
    """

    return PeftModel(model)


__all__ = ["PeftModel", "LoraConfig", "get_peft_model"]
