"""
将近30日K线转为摘要 summary 与涨跌幅 change。避免数据泄漏:
仅用过去窗口构造特征; 未来收益仅用于评测。
"""
from typing import Dict, List, Tuple


def make_summary(kline: List[Dict], window: int=30) -> Tuple[str, float, Dict]:
    """
    :param kline: 近 N 日的 OHLCV 列表（按日期升序）
    :param window: 窗口长度，默认30
    :return:
      summary: str  # 例如 "近30日上涨X%，波动率Y，MA5/10/20..., 量能..."
      change: float # 近30日涨跌幅（或末日相对首日）
      extras: Dict  # 可附加中间指标，便于分析/评测
    """
    # TODO: 根据输入 K 线计算各类指标并生成文本摘要
    raise NotImplementedError
