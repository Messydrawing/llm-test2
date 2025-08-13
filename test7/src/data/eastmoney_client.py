"""
东方财富 K 线客户端
参考接口: https://push2his.eastmoney.com/api/qt/stock/kline/get
关键参数: secid, klt(101/102/103), fqt(0/1/2), fields1/fields2
参考实现: AkShare/efinance 使用相同端点与参数
"""
from typing import Dict, List, Literal, Tuple, Optional, Iterable
import datetime as dt
import json
import logging
import requests
import pandas as pd


def to_secid(symbol: str) -> str:
    """
    将交易所+代码转换为 secid 格式, 例如:
    - 上证: '1.600519'
    - 深证: '0.000001'
    具体映射策略可参考开源实现(akshare/efinance)或本地映射表。
    :param symbol: 形如 '600519' 或带交易所前缀 'SH600519'/'SZ000001'
    :return: 'ex.code' 形式的 secid 字符串
    """
    symbol = symbol.upper()
    if symbol.startswith("SH") or symbol.startswith("SZ"):
        code = symbol[-6:]
        prefix = symbol[:2]
    else:
        code = symbol[-6:]
        first = code[0]
        if first in {"5", "6", "9"}:
            prefix = "SH"
        else:
            prefix = "SZ"
    exch = "1" if prefix == "SH" else "0"
    return f"{exch}.{code}"


def fetch_kline(
    secid: str,
    beg: str, end: str,
    klt: Literal[101,102,103]=101,
    fqt: Literal[0,1,2]=1,
    fields1: str="f1,f2,f3,f4,f5,f6",
    fields2: str="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
    timeout: int=10
) -> Dict:
    """
    访问东财 push2his 接口，返回 JSON 响应（保持原样），失败需重试/限速。
    :param secid: 形如 '1.600519'
    :param beg/end: 'YYYYMMDD' 字符串
    :return: 原始 JSON 字典
    """
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "beg": beg,
        "end": end,
        "klt": klt,
        "fqt": fqt,
        "fields1": fields1,
        "fields2": fields2,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


logger = logging.getLogger(__name__)


def parse_kline_json(raw: Optional[Dict]) -> List[Dict]:
    """Parse K-line JSON from EastMoney or local cache.

    The function is tolerant to several shapes:

    - ``{"data": {"klines": [...]}}`` (official EastMoney response)
    - ``{"klines": [...]}``
    - ``[...]`` where the list is either strings or dict records
    """

    if isinstance(raw, dict):
        if isinstance(raw.get("data"), dict) and "klines" in raw["data"]:
            klines = raw["data"]["klines"]
        elif "klines" in raw:
            klines = raw["klines"]
        else:
            raise ValueError("JSON missing 'klines' list")
    elif isinstance(raw, list):
        klines = raw
    else:
        raise ValueError("raw JSON must be dict or list")

    if not isinstance(klines, list):
        raise ValueError("JSON missing 'klines' list")

    result: List[Dict] = []
    total = len(klines)
    for idx, item in enumerate(klines):
        if isinstance(item, str):
            parts = item.split(",")
            if len(parts) < 7:
                raise ValueError(f"kline entry {idx} malformed: {item}")
            date = parts[0]
            try:
                open_, close, high, low = map(float, parts[1:5])
                volume = float(parts[5])
                turnover = float(parts[6])
            except ValueError as e:
                raise ValueError(f"kline entry {idx} malformed: {item}") from e
        elif isinstance(item, dict):
            try:
                date = item["date"]
                open_ = float(item["open"])
                close = float(item["close"])
                high = float(item["high"])
                low = float(item["low"])
                volume = float(item.get("volume", 0))
                turnover = float(item.get("turnover", 0))
            except Exception as e:
                raise ValueError(f"kline entry {idx} malformed: {item}") from e
        else:
            raise ValueError(f"kline entry {idx} malformed: {item}")

        result.append(
            {
                "date": date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "turnover": turnover,
            }
        )

    valid = len(result)
    logger.info("Parsed %d/%d kline rows", valid, total)
    if valid == 0:
        logger.warning("No valid kline rows parsed")
    return result


class EastMoneyAPI:
    """基于上述函数的轻量封装，直接返回 :class:`pandas.DataFrame`。"""

    def get_kline_data(self, symbol: str, num: int = 1000) -> pd.DataFrame:
        """获取指定股票最近 ``num`` 天的日K线数据。

        :param symbol: 股票代码，例如 ``600519`` 或 ``SH600519``。
        :param num: 返回的最近天数。
        :return: ``pandas.DataFrame``，按日期升序排列。
        """

        secid = to_secid(symbol)
        end = dt.datetime.now().strftime("%Y%m%d")
        beg = (dt.datetime.now() - dt.timedelta(days=num * 2)).strftime("%Y%m%d")
        raw = fetch_kline(secid, beg, end)
        rows = parse_kline_json(raw)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df = df.tail(num)
        return df

    def get_recent_data(self, symbols: Iterable[str], days: int) -> Dict[str, pd.DataFrame]:
        """批量获取多只股票最近 ``days`` 天的数据。"""

        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.get_kline_data(sym, num=days)
            result[sym] = df
        return result
