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

    # Structured config sections
    data_cfg = cfg.get("data", {})
    gen_cfg = cfg.get("generation", {})
    meta_cfg = cfg.get("meta", {})

    # Read input prompts
    prompts = []
    for obj in read_jsonl(data_cfg["input_jsonl"]):
        if isinstance(obj, dict):
            p = obj.get("prompt", "")
        else:
            p = str(obj)
        if not p:
            # 如果存在空 prompt，则忽略并提示，避免教师模型输出默认值
            print("[警告] 检测到空 prompt，已跳过")
            continue
        prompts.append(p)

    results = []
    for out in run_offline_infer(
        cfg["model_name"],
        prompts,
        gen_cfg.get("temperature", 0.7),
        gen_cfg.get("top_p", 0.9),
        gen_cfg.get("max_new_tokens", 512),
        gen_cfg.get("stop", []),
        cfg.get("tensor_parallel_size", 1),
        cfg.get("max_model_len", 32768),
        meta_cfg.get("record_sampling", True),
    ):
        cleaned = ensure_json(out["output"], data_cfg["json_schema"])
        normalized = normalize_teacher_json(cleaned)
        results.append(
            {
                "prompt": out["prompt"],
                "teacher": normalized.model_dump(),
                "meta": out.get("meta", {}),
            }
        )

    Path(data_cfg["output_jsonl"]).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, data_cfg["output_jsonl"])


if __name__ == "__main__":
    main()
