"""测试约束解码函数是否可调用"""
from src.decoding.constraints import build_json_prefix_allowed_tokens_fn


def test_build_fn_exists():
    assert callable(build_json_prefix_allowed_tokens_fn)
