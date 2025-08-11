"""随机种子相关工具"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """设置 python/numpy/torch 的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
