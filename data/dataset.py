"""Utilities for preparing datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .generate import make_dataset
from .teacher import get_teacher_output


def build_dataset(
    num_train: int, num_val: int, seq_len: int, seed: int | None = None
) -> tuple[list[dict], list[dict]]:
    """Build training and validation datasets with teacher labels."""
    train_series, val_series = make_dataset(num_train, num_val, seq_len, seed)

    def make_samples(series_list: Iterable[np.ndarray]) -> list[dict]:
        samples = []
        for seq in series_list:
            label = get_teacher_output(seq.tolist())
            samples.append({"input": seq.tolist(), "label": label})
        return samples

    return make_samples(train_series), make_samples(val_series)
