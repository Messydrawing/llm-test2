"""
TRL Generalized KD (GKD) 的一轮 on-policy 蒸馏（学生自采样+老师分布监督）。
"""
from typing import Dict


def run_gkd(config: Dict):
    """
    读取 configs/gkd.yaml:
      - lambda: 学生自生成样本占比
      - beta:   前/反向KL插值系数（可网格搜索）
      - seq_kd: 序列级 KD
    核心调用: TRL.GKDTrainer
    """
    # TODO: 构造 GKDTrainer 并执行一轮 on-policy 蒸馏
    raise NotImplementedError
