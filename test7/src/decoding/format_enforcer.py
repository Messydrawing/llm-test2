"""
（可选）使用 lm-format-enforcer 对接 Transformers，
保证 JSON schema 级别的强约束。
"""


def build_enforced_generate(tokenizer, schema: dict):
    """
    返回一个包装好的 generate 调用器，内部集成前缀约束。
    """
    try:
        from lmformatenforcer import JsonSchemaEnforcer
        from lmformatenforcer.integrations.transformers import (
            build_transformers_textual_constraint,
        )

        constraint = JsonSchemaEnforcer(schema)
        constraint_fn = build_transformers_textual_constraint(tokenizer, constraint)

        def generate(model, *args, **kwargs):
            kwargs.setdefault("prefix_allowed_tokens_fn", constraint_fn)
            return model.generate(*args, **kwargs)

        return generate
    except Exception:
        def generate(model, *args, **kwargs):
            return model.generate(*args, **kwargs)

        return generate
