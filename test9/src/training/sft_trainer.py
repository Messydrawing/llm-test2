"""Light‑weight utilities for supervised fine‑tuning (SFT).

The real project uses :class:`~transformers.Trainer` with a custom data
collator.  For the unit tests in this kata we provide a greatly simplified
implementation that mirrors the public API while remaining importable even if
optional dependencies such as :mod:`torch` or :mod:`transformers` are missing.
The goal is to make the helpers easy to understand and dependable for small
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - transformers is an optional dependency
    from transformers import Trainer
except Exception:  # pragma: no cover - transformers not installed
    Trainer = object  # type: ignore

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


@dataclass
class SFTDataCollator:
    """Minimal data collator for supervised fine‑tuning.

    The collator simply pads ``input_ids`` using the provided tokenizer and
    copies them to ``labels`` so that the language model is trained to predict
    the next token.  If either :mod:`torch` or a tokenizer with a ``pad``
    method is unavailable, the features are returned unchanged to keep the
    behaviour predictable in the test environment.
    """

    tokenizer: Any

    def __call__(self, features: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        features = list(features)

        if getattr(self.tokenizer, "pad", None) and torch is not None:
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["labels"] = batch["input_ids"].clone()
            return batch

        # Fallback – return simple lists so tests can still inspect the output
        input_ids: List[Any] = [f["input_ids"] for f in features]
        return {"input_ids": input_ids, "labels": input_ids}


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


class SFTTrainer(Trainer):
    """Thin wrapper around :class:`~transformers.Trainer` with a default
    :class:`SFTDataCollator`.

    The wrapper injects :class:`SFTDataCollator` when a ``data_collator`` is not
    explicitly provided.  This keeps the public API similar to the real
    training code while remaining small enough for unit testing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover -
        # runtime behaviour depends on optional dependencies
        if Trainer is object:
            raise RuntimeError("transformers is required to use SFTTrainer")

        if "data_collator" not in kwargs and "tokenizer" in kwargs:
            kwargs["data_collator"] = SFTDataCollator(kwargs["tokenizer"])

        super().__init__(*args, **kwargs)  # type: ignore[misc]


__all__ = ["SFTDataCollator", "SFTTrainer"]

