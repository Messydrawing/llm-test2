"""
对 DistillKit 的 distil_logits.py 做包装。
同词表(Qwen->Qwen)场景: KL(logits)+CE 组合, T/alpha 可配。
"""
from typing import Dict
import json
import subprocess


def launch_logits_kd(config: Dict):
    """
    :param config: 读取 configs/distill_logits.yaml
    调用 accelerate + DistillKit 的 distil_logits.py
    """
    cfg_path = config.get("config_path", "configs/distill_logits.yaml")
    cmd = [
        "accelerate",
        "launch",
        config.get("script", "distil_logits.py"),
        "--config",
        cfg_path,
    ]
    extra = config.get("extra_args") or []
    if isinstance(extra, dict):
        for k, v in extra.items():
            cmd.extend([f"--{k}", str(v)])
    elif isinstance(extra, list):
        cmd.extend(map(str, extra))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("accelerate not found, command would be:", " ".join(cmd))
