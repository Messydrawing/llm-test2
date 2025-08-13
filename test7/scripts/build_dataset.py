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

    raw_paths = list(raw_dir.glob("*.json"))
    if not raw_paths:
        print(f"[警告] 在 {raw_dir} 未找到本地JSON，尝试在线获取…")
        from src.data.eastmoney_client import EastMoneyAPI
        import pandas as pd

        api = EastMoneyAPI()
        symbols = cfg.get("symbols", [])
        for sym in symbols:
            df = api.get_kline_data(sym)
            if df is None or df.empty:
                print(f"[警告] 股票 {sym} 在线获取失败，已跳过")
                continue
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            kline = df.to_dict(orient="records")
            for rec in kline:
                rec["stock_code"] = sym
            try:
                item = build_prompts_from_kline(kline, template)
            except ValueError as e:
                print(f"[警告] {sym} 数据异常: {e}，已跳过")
                continue
            all_items.append(item)
        if not all_items:
            print("[警告] 未能在线获取任何K线数据")
    else:
        for path in raw_paths:
            with open(path, "r", encoding="utf-8") as fr:
                raw = json.load(fr)
            kline = parse_kline_json(raw)
            if not kline:
                # 若原始 JSON 中无有效 K 线数据，则跳过该文件，避免生成空 prompt
                print(f"[警告] {path.name} 无有效K线数据，已跳过")
                continue
            for rec in kline:
                rec["stock_code"] = path.stem
            try:
                item = build_prompts_from_kline(kline, template)
            except ValueError as e:
                # 如果K线数据不足或含有NaN/inf等异常，跳过该样本
                print(f"[警告] {path.name} 数据异常: {e}，已跳过")
                continue
            all_items.append(item)

    if not all_items:
        print("[警告] 未能从本地或远程获取任何样本")

    df = cfg.get("dataframe")
    if df is not None:
        subsets = time_roll_split(df, cfg.get("splits", {}))
    else:
        subsets = {"all": all_items}

    output_dir = Path(cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "all" in subsets:
        # 当未提供显式划分时，默认生成 all.jsonl 和 train.jsonl
        items = [i.model_dump() for i in all_items]
        write_jsonl(items, output_dir / "all.jsonl")
        # 兼容下游默认读取的 train.jsonl
        write_jsonl(items, output_dir / "train.jsonl")
    else:
        for name, subset in subsets.items():
            write_jsonl([i.model_dump() for i in subset], output_dir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
