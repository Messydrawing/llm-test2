"""
对 DistillKit 的 distil_logits.py 做包装。
同词表(Qwen->Qwen)场景: KL(logits)+CE 组合, T/alpha 可配。
"""
from typing import Dict


def launch_logits_kd(config: Dict):
    """
    :param config: 读取 configs/distill_logits.yaml
    调用 accelerate + DistillKit 的 distil_logits.py
    """
    # TODO: 根据配置启动对数蒸馏训练
    raise NotImplementedError
