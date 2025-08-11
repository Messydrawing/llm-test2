"""
使用 vLLM Offline Inference 批推老师输出（严格 JSON），
保存到 data/teacher_outputs/*.jsonl
生成参数来自 configs/teacher_infer.yaml
"""
import argparse

import yaml
from pathlib import Path

from src.teacher.vllm_offline import run_offline_infer, ensure_json
from src.utils.io import read_jsonl, write_jsonl
from src.teacher.postprocess import normalize_teacher_json


def parse_args():
    """解析配置文件路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts = []
    for obj in read_jsonl(cfg["prompts"]):
        if isinstance(obj, dict):
            prompts.append(obj.get("prompt", ""))
        else:
            prompts.append(str(obj))

    results = []
    for out in run_offline_infer(
        cfg["model"],
        prompts,
        cfg.get("temperature", 0.7),
        cfg.get("top_p", 0.9),
        cfg.get("max_new_tokens", 512),
        cfg.get("stop", []),
        cfg.get("tp_size", 1),
        cfg.get("max_model_len", 32768),
        True,
    ):
        cleaned = ensure_json(out["output"], cfg["schema"])
        normalized = normalize_teacher_json(cleaned)
        results.append(
            {
                "prompt": out["prompt"],
                "teacher": normalized.model_dump(),
                "meta": out.get("meta", {}),
            }
        )

    Path(cfg["output"]).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, cfg["output"])


if __name__ == "__main__":
    main()
