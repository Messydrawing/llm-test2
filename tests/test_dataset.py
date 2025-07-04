import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from data.generate import generate_stock_series, make_dataset
from data.dataset import build_dataset


def test_make_dataset_shapes():
    train, val = make_dataset(3, 2, seq_len=5, seed=0)
    assert len(train) == 3
    assert len(val) == 2
    for seq in train + val:
        assert isinstance(seq, np.ndarray)
        assert len(seq) == 5


def test_build_dataset_labels():
    train_samples, val_samples = build_dataset(1, 1, seq_len=5, seed=0)
    assert len(train_samples) == 1
    assert len(val_samples) == 1
    sample = train_samples[0]
    assert "input" in sample and "label" in sample
    assert isinstance(sample["input"], list)
    assert isinstance(sample["label"], dict)
