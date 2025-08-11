"""
（可选）使用 lm-format-enforcer 对接 Transformers，
保证 JSON schema 级别的强约束。
"""


def build_enforced_generate(tokenizer, schema: dict):
    """
    返回一个包装好的 generate 调用器，内部集成前缀约束。
    """
    # TODO: 利用 lm-format-enforcer 结合 tokenizer 构造约束生成函数
    raise NotImplementedError
