"""
东方财富 K 线客户端
参考接口: https://push2his.eastmoney.com/api/qt/stock/kline/get
关键参数: secid, klt(101/102/103), fqt(0/1/2), fields1/fields2
参考实现: AkShare/efinance 使用相同端点与参数
"""
from typing import Dict, List, Literal, Tuple
import datetime as dt


def to_secid(symbol: str) -> str:
    """
    将交易所+代码转换为 secid 格式, 例如:
    - 上证: '1.600519'
    - 深证: '0.000001'
    具体映射策略可参考开源实现(akshare/efinance)或本地映射表。
    :param symbol: 形如 '600519' 或带交易所前缀 'SH600519'/'SZ000001'
    :return: 'ex.code' 形式的 secid 字符串
    """
    raise NotImplementedError


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
    raise NotImplementedError


def parse_kline_json(raw: Dict) -> List[Dict]:
    """
    将东财返回的 kline 数组解析为统一结构:
    [{'date': 'YYYY-MM-DD', 'open': float, 'high': float, 'low': float,
      'close': float, 'volume': float, 'turnover': float, ...}, ...]
    注意 fields2 对应字段含义按 akshare/efinance 源码比对。
    """
    raise NotImplementedError
