"""
将近30日K线转为摘要 summary 与涨跌幅 change。避免数据泄漏:
仅用过去窗口构造特征; 未来收益仅用于评测。
"""
from typing import Dict, List, Tuple
import numpy as np


def make_summary(kline: List[Dict], window: int=30) -> Tuple[str, float, Dict]:
    """
    :param kline: 近 N 日的 OHLCV 列表（按日期升序）
    :param window: 窗口长度，默认30
    :return:
      summary: str  # 例如 "近30日上涨X%，波动率Y，MA5/10/20..., 量能..."
      change: float # 近30日涨跌幅（或末日相对首日）
      extras: Dict  # 可附加中间指标，便于分析/评测
    """
    window_data = kline[-window:] if len(kline) >= window else kline
    closes = np.array([row["close"] for row in window_data], dtype=float)
    volumes = np.array([row.get("volume", 0.0) for row in window_data], dtype=float)
    change = (closes[-1] - closes[0]) / closes[0] * 100 if closes.size > 1 else 0.0
    returns = np.diff(closes) / closes[:-1] if closes.size > 1 else np.array([0.0])
    volatility = float(np.std(returns)) * 100
    avg_volume = float(np.mean(volumes))
    summary = (
        f"近{len(window_data)}日上涨{change:.2f}%，波动率{volatility:.2f}%，"
        f"平均成交量{avg_volume:.0f}"
    )
    extras = {"volatility": volatility, "avg_volume": avg_volume}
    return summary, change, extras
