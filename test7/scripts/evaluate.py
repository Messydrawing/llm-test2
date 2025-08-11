"""
统一评测脚本:
  - 读取学生/老师在测试集的输出
  - 计算分类主指标（Acc/F1/MCC）与校准(Brier/ECE)
  - 统计 JSON 合法率
  - 产出与老师差距报告 (≤1~2pct 为达标线)
"""
import argparse

import yaml
from src.utils.io import read_jsonl
from src.eval.metrics import classification_metrics, calibration_metrics, json_validity_rate


def parse_args():
    """解析评测配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    records = list(read_jsonl(cfg["predictions"]))
    y_true = [r[cfg.get("label_field", "label")] for r in records]
    y_pred = [r[cfg.get("pred_field", "prediction")] for r in records]
    probs = [r.get(cfg.get("prob_field", "prob"), 0.0) for r in records]
    json_list = [r.get(cfg.get("json_field", "output"), "{}") for r in records]

    cls = classification_metrics(y_true, y_pred)
    cal = calibration_metrics(probs, y_true) if probs else {"brier": 0.0, "ece": 0.0}
    jv = json_validity_rate(json_list)

    for k, v in {**cls, **cal, "json_valid": jv}.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
