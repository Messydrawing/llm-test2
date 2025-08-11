"""
将 raw K 线转为 processed 样本，生成 prompt/summary/change
并时间滚动切分，导出 data/processed/*.jsonl 与 data/splits/*
"""
import argparse

import json
from pathlib import Path
import yaml

from src.utils.io import read_jsonl, write_jsonl
from src.data.dataset_builder import build_prompts_from_kline, time_roll_split
from src.data.eastmoney_client import parse_kline_json


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    parser.add_argument("--raw", help="原始K线数据目录", nargs="?")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(args.raw or cfg.get("raw_dir", "data/raw"))
    template = cfg["template"]
    all_items = []
    for path in raw_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as fr:
            raw = json.load(fr)
        kline = parse_kline_json(raw)
        for rec in kline:
            rec["stock_code"] = path.stem
        item = build_prompts_from_kline(kline, template)
        all_items.append(item)

    df = cfg.get("dataframe")
    if df is not None:
        subsets = time_roll_split(df, cfg.get("splits", {}))
    else:
        subsets = {"all": all_items}

    output_dir = Path(cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "all" in subsets:
        write_jsonl([i.model_dump() for i in all_items], output_dir / "all.jsonl")
    else:
        for name, subset in subsets.items():
            write_jsonl([i.model_dump() for i in subset], output_dir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
