"""Dataset creation utilities for real K-line data."""

from __future__ import annotations

import json
import random
from typing import Any, Sequence, Tuple, Optional


def _fetch_kline(code: str, days: int):
    """Fetch ``days`` of kline data for ``code`` using :class:`EastMoneyAPI`."""
    from .data_loader import EastMoneyAPI  # local import to avoid hard dep

    api = EastMoneyAPI()
    df = api.get_kline_data(code, num=days)
    if df is None:
        return None
    return df.tail(days).reset_index(drop=True)


def _sample_windows(df, window: int, num: int, rng: random.Random):
    """Return all windows of length ``window`` from ``df``. (Parameter ``num`` is ignored.)"""
    if len(df) < window:
        return []
    return [
        df.iloc[i : i + window].reset_index(drop=True)
        for i in range(len(df) - window + 1)
    ]


def _make_prompt(window) -> dict[str, Any]:
    """Assemble structured prompt fields from a window of kline data."""
    change = ((window["close"].iloc[-1] / window["close"].iloc[0]) - 1) * 100
    summary = window.to_dict(orient="records")
    return {
        "change": round(change, 2),
        "prediction": "",
        "analysis": "",
        "advice": "",
        "kline_summary": summary,
    }


def format_prompt(sample: dict[str, Any]) -> str:
    """Convert a dataset sample into the textual prompt sent to the teacher."""
    summary = json.dumps(sample["kline_summary"], ensure_ascii=False)
    return (
        f"股票 {sample['stock_code']} 近30日K线数据: {summary}\n"
        f"涨跌幅: {sample['change']}%。请预测后市走势，"
        "给出简短分析和操作建议，"
        "并以 JSON 格式回复，包括 'prediction', 'analysis', 'advice' 三个字段。"
    )


def _trim_sample_tokens(
    sample: dict[str, Any], tokenizer, max_tokens: int
) -> None:
    """Trim ``kline_summary`` so ``format_prompt(sample)`` fits max_tokens."""
    if tokenizer is None:
        return
    while len(sample["kline_summary"]) > 1:
        text = format_prompt(sample)
        if (
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            <= max_tokens
        ):
            break
        sample["kline_summary"].pop(0)


def build_dataset(
    stock_codes: Sequence[str],
    *,
    days: int = 180,
    window: int = 30,
    windows_per_stock: int = 1,
    val_ratio: float = 0.2,
    seed: int | None = None,
    tokenizer=None,
    max_tokens: int = 1024,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build train/validation datasets from K-line data using sliding windows."""
    from .config import STOCK_CODES

    codes = list(stock_codes) if stock_codes else list(STOCK_CODES)
    rng = random.Random(seed)
    up_samples: list[dict[str, Any]] = []
    down_samples: list[dict[str, Any]] = []
    stable_samples: list[dict[str, Any]] = []

    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception as e:  # pragma: no cover - optional dep
        raise ImportError(
            "pandas and numpy are required for dataset building"
        ) from e

    for code in codes:
        df = _fetch_kline(code, days)
        if df is None or df.empty:
            continue
        # --- Compute technical indicators and mark anomalies ---
        df["pct_chg"] = df["close"].pct_change() * 100
        df["pct_chg"].fillna(0, inplace=True)
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        differences = df["close"].diff()
        gains = differences.clip(lower=0)
        losses = -differences.clip(upper=0)
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
        df["MA5"] = df["MA5"].round(2)
        df["MA10"] = df["MA10"].round(2)
        df["RSI14"] = df["RSI14"].round(2)
        df["MACD"] = df["MACD"].round(2)
        # --- Sliding window sample extraction ---
        n = len(df)
        if n < window:
            continue
        for i in range(n - window + 1):
            # Skip window if any day has volume=0 or daily change beyond ±20%
            if (
                df["volume"].iloc[i : i + window].eq(0).any()
                or df["pct_chg"].iloc[i : i + window].abs().gt(20).any()
            ):
                continue
            win = df.iloc[i : i + window][
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
            ].reset_index(drop=True)
            change_percent = (
                (win["close"].iloc[-1] / win["close"].iloc[0]) - 1
            ) * 100
            if change_percent > 3:
                category = "up"
            elif change_percent < -3:
                category = "down"
            else:
                category = "stable"
            prompt = _make_prompt(win)
            prompt["stock_code"] = code
            # --- Trim prompt to max_tokens if necessary ---
            if tokenizer:
                text = format_prompt(prompt)
                while (
                    len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    > max_tokens
                    and prompt["kline_summary"]
                ):
                    prompt["kline_summary"].pop(0)
                    text = format_prompt(prompt)
            if category == "up":
                up_samples.append(prompt)
            elif category == "down":
                down_samples.append(prompt)
            else:
                stable_samples.append(prompt)
    # --- Balance classes by downsampling ---
    if up_samples and down_samples and stable_samples:
        min_count = min(
            len(up_samples), len(down_samples), len(stable_samples)
        )
        rng.shuffle(up_samples)
        rng.shuffle(down_samples)
        rng.shuffle(stable_samples)
        up_samples = up_samples[:min_count]
        down_samples = down_samples[:min_count]
        stable_samples = stable_samples[:min_count]
    samples = up_samples + down_samples + stable_samples
    rng.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]
