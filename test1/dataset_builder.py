"""Dataset creation utilities for real K-line data."""

from __future__ import annotations

import json
import random
from typing import Any, Sequence, Tuple


def _fetch_kline(code: str, days: int):
    """Fetch ``days`` of kline data for ``code`` using :class:`EastMoneyAPI`."""
    from .data_loader import EastMoneyAPI  # local import to avoid hard dep

    api = EastMoneyAPI()
    df = api.get_kline_data(code, num=days)
    if df is None:
        return None
    return df.tail(days).reset_index(drop=True)


def _sample_windows(df, window: int, num: int, rng: random.Random):
    """Return ``num`` random windows of length ``window`` from ``df``."""
    if len(df) < window:
        return []
    max_start = len(df) - window
    starts = [rng.randint(0, max_start) for _ in range(num)]
    return [df.iloc[s : s + window].reset_index(drop=True) for s in starts]


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


def build_dataset(
    stock_codes: Sequence[str],
    *,
    days: int = 180,
    window: int = 30,
    windows_per_stock: int = 1,
    val_ratio: float = 0.2,
    seed: int | None = None,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build train/validation datasets from K-line data."""
    from .config import STOCK_CODES

    codes = list(stock_codes) if stock_codes else list(STOCK_CODES)
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    try:
        import pandas as pd  # noqa: F401
    except Exception as e:  # pragma: no cover - optional dep
        raise ImportError("pandas is required for dataset building") from e

    for code in codes:
        df = _fetch_kline(code, days)
        if df is None or df.empty:
            continue
        for win in _sample_windows(df, window, windows_per_stock, rng):
            prompt = _make_prompt(win)
            prompt["stock_code"] = code
            samples.append(prompt)

    rng.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]
