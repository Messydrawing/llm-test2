"""
对 DistillKit 的 distil_hidden.py 做轻量包装，统一配置/日志。
支持: 层映射策略、MSE/Cosine 混合损失、BF16/FlashAttn/DeepSpeed。
"""
from typing import Dict
import subprocess
import argparse
from pathlib import Path
import yaml


def launch_hidden_kd(config: Dict):
    """
    :param config: 读取 configs/distill_hidden.yaml
    调用 accelerate + DistillKit 的 distil_hidden.py
    """
    cfg_path = config.get("config_path", "configs/accelerate_ds_zero3.yaml")
    script = config.get("script", "distil_hidden.py")
    if script == "distil_hidden.py":
        # 尝试在已安装的 DistillKit 包中定位该脚本
        try:
            import distillkit
            pkg_dir = Path(distillkit.__file__).resolve().parent
            candidate = pkg_dir / "distil_hidden.py"
            if not candidate.exists():
                candidate = pkg_dir / "scripts" / "distil_hidden.py"
            script = str(candidate)
        except Exception as exc:  # pragma: no cover - 网络/安装环境差异
            raise FileNotFoundError(
                "distil_hidden.py not found. Please ensure DistillKit is installed."
            ) from exc

    cmd = ["accelerate", "launch", "--config_file", cfg_path, script]
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


def main():  # pragma: no cover - CLI 简单包装
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/distill_hidden.yaml", help="DistillKit 配置文件"
    )
    parser.add_argument(
        "--accelerate-config", default="configs/accelerate_ds_zero3.yaml", help="accelerate 配置"
    )
    args, extra = parser.parse_known_args()
    with open(args.config, "r", encoding="utf-8") as f:
        distill_cfg = yaml.safe_load(f)
    cfg = {
        "config_path": args.accelerate_config,
        "distill_config": args.config,
        "extra_args": extra,
    }
    cfg.update(distill_cfg)
    launch_hidden_kd(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
