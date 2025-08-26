import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from test8 import dataset_builder as db


def test_build_dataset_format_and_balance(monkeypatch, tmp_path):
    # prepare three windows producing up, down and stable trends
    window_up = pd.DataFrame(
        {
            "date": ["d1", "d2"],
            "open": [100, 102],
            "close": [100, 104],
            "high": [101, 105],
            "low": [99, 103],
            "volume": [1000, 1000],
            "MA5": [0, 0],
            "MA10": [0, 0],
            "RSI14": [0, 0],
            "MACD": [0, 0],
        }
    )
    window_down = pd.DataFrame(
        {
            "date": ["d1", "d2"],
            "open": [100, 98],
            "close": [100, 95],
            "high": [101, 99],
            "low": [99, 94],
            "volume": [1000, 1000],
            "MA5": [0, 0],
            "MA10": [0, 0],
            "RSI14": [0, 0],
            "MACD": [0, 0],
        }
    )
    window_stable = pd.DataFrame(
        {
            "date": ["d1", "d2"],
            "open": [100, 101],
            "close": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "volume": [1000, 1000],
            "MA5": [0, 0],
            "MA10": [0, 0],
            "RSI14": [0, 0],
            "MACD": [0, 0],
        }
    )
    windows = [window_up, window_down, window_stable]

    def fake_fetch(code, days, api):
        # ensure len(df) >= window inside build_dataset
        return pd.DataFrame({"close": [1, 2]})

    def fake_window_samples(df, window):
        for w in windows:
            yield w

    monkeypatch.setattr(db, "_fetch_kline", fake_fetch)
    monkeypatch.setattr(db, "_compute_indicators", lambda df: None)
    monkeypatch.setattr(db, "_window_samples", fake_window_samples)

    train, val, test = db.build_dataset(
        ["000001"], days=2, window=2, val_ratio=0, test_ratio=0, seed=0, out_dir=tmp_path
    )

    assert val == []
    assert test == []
    assert len(train) == 3
    # ensure files written
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    sample = train[0]
    assert set(sample.keys()) == {
        "stock_code",
        "change",
        "trend",
        "prediction",
        "analysis",
        "advice",
        "kline_summary",
    }
    counts = Counter(s["trend"] for s in train)
    assert counts["up"] == counts["down"] == counts["stable"] == 1
