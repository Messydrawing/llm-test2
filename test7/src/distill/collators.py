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
    import torch

    input_ids = []
    attention_mask = []
    labels = []
    for sample in batch:
        prompt = sample.get("prompt", "")
        target = sample.get("teacher_json", "")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        lab = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

        if len(ids) > max_length:
            ids = ids[:max_length]
            lab = lab[:max_length]

        input_ids.append(ids)
        labels.append(lab)
        attention_mask.append([1] * len(ids))

    # padding
    pad_id = tokenizer.pad_token_id or 0
    max_len = max(len(ids) for ids in input_ids)
    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])
        input_ids[i] += [pad_id] * pad_len
        attention_mask[i] += [0] * pad_len
        labels[i] += [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
