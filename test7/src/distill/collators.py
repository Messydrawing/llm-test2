"""
蒸馏场景专用 collator:
  - 拼接 prompt 与 目标 JSON
  - 生成 teacher logit/hidden 对齐的 masks
"""


def kd_collate_fn(batch, tokenizer, max_length: int):
    """
    :param batch: 包含 {prompt, teacher_json, ...}
    :return: 模型输入张量、注意力mask、labels等
    """
    # TODO: 对 batch 中字段进行编码并生成对齐的张量
    raise NotImplementedError
