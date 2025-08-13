"""
拉取指定股票近N日K线，保存到 data/raw/*.json
入参: --symbols, --beg, --end, --klt, --fqt
依赖: src.data.eastmoney_client
"""
import argparse

import json
from pathlib import Path
import logging
import yaml

from src.data.eastmoney_client import (
    to_secid,
    fetch_kline,
    EastMoneyAPI,
    parse_kline_json,
)
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    parser.add_argument("--symbols", nargs="*", help="可覆盖配置中的股票代码")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = args.symbols or cfg.get("symbols", [])
    output_dir = Path(cfg.get("output_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    api = EastMoneyAPI()
    for sym in symbols:
        # 连通性测试：尝试获取最近1天数据
        try:
            test_df = api.get_kline_data(sym, num=1)
        except Exception as e:  # 网络或解析异常
            logger.error(f"[测试] 股票{sym} 最新行情获取失败: {e}")
            continue
        if test_df.empty:
            logger.warning(f"[测试] 股票{sym} 最新行情获取失败！")
        else:
            latest = test_df.iloc[-1]
            logger.info(
                f"[测试] 股票{sym} 最新日期: {latest['date'].date()}, 收盘价: {latest['close']}"
            )

        # 仍使用原始接口抓取并保存原始 JSON，以保持下游兼容
        secid = to_secid(sym)
        data = fetch_kline(
            secid,
            cfg.get("beg"),
            cfg.get("end"),
            cfg.get("klt", 101),
            cfg.get("fqt", 1),
        )
        try:
            rows = parse_kline_json(data)
        except ValueError as e:
            logger.error(f"股票{sym} 数据解析失败: {e}")
            continue
        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["close", "volume"], inplace=True)
        if df.empty:
            logger.warning(f"股票{sym} 无有效数据，已跳过保存")
            continue
        # 仅保留必要字段，重建成解析器期望的结构
        klines = [
            f"{r['date']},{r['open']},{r['close']},{r['high']},{r['low']},{r['volume']},{r['turnover']}"
            for r in df.to_dict(orient="records")
        ]
        payload = {"data": {"klines": klines}}
        with open(output_dir / f"{sym}.json", "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)


if __name__ == "__main__":
    main()
