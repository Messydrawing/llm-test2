"""Utilities for building stock K-line datasets with technical indicators."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

try:
    from .config import STOCK_CODES as DEFAULT_CODES
except Exception:  # pragma: no cover - optional
    DEFAULT_CODES = ["600000"]

from .data_loader import EastMoneyAPI


# ---------------------------------------------------------------------------
def _fetch_kline(code: str, days: int, api: EastMoneyAPI) -> Any:
    df = api.get_kline_data(code, num=days)
    if df is None:
        return None
    return df.tail(days).reset_index(drop=True)


def _compute_indicators(df) -> None:
    """Add MA5/MA10, RSI14 and MACD columns to ``df`` in-place."""
    import numpy as np
    df["pct_chg"] = df["close"].pct_change() * 100
    # Avoid chained-assignment warnings and upcoming pandas changes by
    # assigning the filled Series back to the DataFrame instead of using
    # ``inplace=True`` on the Series, which can operate on a copy.
    df["pct_chg"] = df["pct_chg"].fillna(0)
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    diffs = df["close"].diff()
    gains = diffs.clip(lower=0)
    losses = -diffs.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs_values = rs.to_numpy()
    avg_gain_values = avg_gain.to_numpy()
    avg_loss_values = avg_loss.to_numpy()
    rsi = np.where(
        avg_loss_values == 0,
        np.where(avg_gain_values == 0, 50, 100),
        100 - 100 / (1 + rs_values),
    )
    df["RSI14"] = rsi
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df[["MA5", "MA10", "RSI14", "MACD"]] = df[["MA5", "MA10", "RSI14", "MACD"]].round(2)


def _window_samples(df, window: int) -> Iterable[Any]:
    n = len(df)
    for i in range(n - window + 1):
        if (
            df["volume"].iloc[i : i + window].eq(0).any()
            or df["pct_chg"].iloc[i : i + window].abs().gt(20).any()
        ):
            continue
        yield df.iloc[i : i + window].reset_index(drop=True)


def _make_sample(window, code: str) -> dict[str, Any]:
    change = ((window["close"].iloc[-1] / window["close"].iloc[0]) - 1) * 100
    trend = "up" if change > 3 else "down" if change < -3 else "stable"
    summary = window[
        [
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "MA5",
            "MA10",
            "RSI14",
            "MACD",
        ]
    ].to_dict(orient="records")
    return {
        "stock_code": code,
        "change": round(change, 2),
        "trend": trend,
        "prediction": "",
        "analysis": "",
        "advice": "",
        "kline_summary": summary,
    }


def format_prompt(sample: dict[str, Any]) -> str:
    summary = json.dumps(sample["kline_summary"], ensure_ascii=False)
    return (
        f"股票 {sample['stock_code']} 近{len(sample['kline_summary'])}日K线数据: {summary}\n"
        f"涨跌幅: {sample['change']}%。请预测后市走势，给出简短分析和操作建议，"
        "并以 JSON 格式回复，包括 'prediction', 'analysis', 'advice' 三个字段。"
    )


# ---------------------------------------------------------------------------
def _save_jsonl(samples: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def build_dataset(
    stock_codes: Sequence[str] | None = None,
    *,
    days: int = 180,
    window: int = 30,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int | None = None,
    out_dir: str | Path | None = None,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build dataset and optionally save to JSONL files.

    Returns ``(train_samples, val_samples, test_samples)``. ``test_ratio``
    defaults to ``0.0`` so existing two-way splits continue to work.
    """

    codes = list(stock_codes) if stock_codes else list(DEFAULT_CODES)
    rng = random.Random(seed)
    api = EastMoneyAPI()
    samples: list[dict[str, Any]] = []

    for code in codes:
        df = _fetch_kline(code, days, api)
        if df is None or df.empty:
            continue
        _compute_indicators(df)
        if len(df) < window:
            continue
        for win in _window_samples(df, window):
            sample = _make_sample(win, code)
            samples.append(sample)

    rng.shuffle(samples)
    n_total = len(samples)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _save_jsonl(train, out_path / "train.jsonl")
        _save_jsonl(val, out_path / "val.jsonl")
        _save_jsonl(test, out_path / "test.jsonl")

    return train, val, test
