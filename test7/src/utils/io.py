"""输入输出相关工具函数"""
from typing import Iterable, Any


def read_jsonl(path: str) -> Iterable[Any]:
    """读取 JSONL 文件, 返回对象迭代器"""
    raise NotImplementedError


def write_jsonl(items: Iterable[Any], path: str):
    """将对象列表写入 JSONL 文件"""
    raise NotImplementedError
