"""
拉取指定股票近N日K线，保存到 data/raw/*.json
入参: --symbols, --beg, --end, --klt, --fqt
依赖: src.data.eastmoney_client
"""
import argparse

import json
from pathlib import Path
import yaml

from src.data.eastmoney_client import to_secid, fetch_kline


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
    for sym in symbols:
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
