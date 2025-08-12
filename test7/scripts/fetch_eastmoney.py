"""
拉取指定股票近N日K线，保存到 data/raw/*.json
入参: --symbols, --beg, --end, --klt, --fqt
依赖: src.data.eastmoney_client
"""
import argparse

import json
from pathlib import Path
import yaml

from src.data.eastmoney_client import to_secid, fetch_kline, EastMoneyAPI


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    parser.add_argument("--symbols", nargs="*", help="可覆盖配置中的股票代码")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = args.symbols or cfg.get("symbols", [])
    output_dir = Path(cfg.get("output_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    api = EastMoneyAPI()
    for sym in symbols:
        # 连通性测试：尝试获取最近1天数据
        test_df = api.get_kline_data(sym, num=1)
        if test_df.empty:
            print(f"[测试] 股票{sym} 最新行情获取失败！")
        else:
            latest = test_df.iloc[-1]
            print(
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
        with open(output_dir / f"{sym}.json", "w", encoding="utf-8") as fw:
            json.dump(data, fw, ensure_ascii=False)


if __name__ == "__main__":
    main()
