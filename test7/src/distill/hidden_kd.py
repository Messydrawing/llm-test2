"""
对 DistillKit 的 distil_hidden.py 做轻量包装，统一配置/日志。
支持: 层映射策略、MSE/Cosine 混合损失、BF16/FlashAttn/DeepSpeed。
"""
from typing import Dict


def launch_hidden_kd(config: Dict):
    """
    :param config: 读取 configs/distill_hidden.yaml
    调用 accelerate + DistillKit 的 distil_hidden.py
    """
    # TODO: 根据配置构造命令并启动隐藏态蒸馏
    raise NotImplementedError
