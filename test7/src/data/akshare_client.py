"""
（可选）使用 akshare 直接拿日线数据，作为东财接口的替代/回退通道。
ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="YYYYMMDD", end_date="YYYYMMDD", adjust="")
注意接口变动较频繁，需做异常与重试。
"""
import pandas as pd


def fetch_hist(symbol: str, start_date: str, end_date: str, adjust: str="") -> pd.DataFrame:
    """
    返回包含 OHLCV 的 DataFrame, 列名与 parse_kline_json 对齐。
    """
    # TODO: 调用 akshare 接口获取历史行情并返回 DataFrame
    raise NotImplementedError
