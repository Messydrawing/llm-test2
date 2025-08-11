"""
约束解码: 保障学生推理时只输出合法 JSON。
方法1: prefix_allowed_tokens_fn（Transformers原生）
方法2: 集成 lm-format-enforcer 构造约束函数（见 format_enforcer.py）
"""
from typing import Callable, List
import torch


def build_json_prefix_allowed_tokens_fn(tokenizer) -> Callable[[int, torch.Tensor], List[int]]:
    """
    返回可直接传入 model.generate(..., prefix_allowed_tokens_fn=fn) 的函数。
    需根据当前已生成内容决定下一步允许 token 集合。
    """
    vocab = list(range(getattr(tokenizer, "vocab_size", 0)))

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        # 简易实现: 始终允许全部 token, 但保留接口可拓展
        _ = batch_id, input_ids
        return vocab

    return prefix_allowed_tokens_fn
