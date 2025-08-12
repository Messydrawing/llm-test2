"""
对 DistillKit 的 distil_hidden.py 做轻量包装，统一配置/日志。
支持: 层映射策略、MSE/Cosine 混合损失、BF16/FlashAttn/DeepSpeed。
"""
from typing import Dict
import subprocess


def launch_hidden_kd(config: Dict):
    """
    :param config: 读取 configs/distill_hidden.yaml
    调用 accelerate + DistillKit 的 distil_hidden.py
    """
    cfg_path = config.get("config_path", "configs/accelerate_ds_zero3.yaml")
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        cfg_path,
        config.get("script", "distil_hidden.py"),
    ]
    distill_cfg = config.get("distill_config", "configs/distill_hidden.yaml")
    if distill_cfg:
        cmd.extend(["--config", distill_cfg])
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
