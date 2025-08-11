"""输入输出相关工具函数"""
from typing import Iterable, Any


def read_jsonl(path: str) -> Iterable[Any]:
    """读取 JSONL 文件, 返回对象迭代器"""
    # TODO: 打开指定路径, 按行解析 JSON 并生成对象
    raise NotImplementedError


def write_jsonl(items: Iterable[Any], path: str):
    """将对象列表写入 JSONL 文件"""
    # TODO: 将对象序列逐行转为 JSON 字符串后写入文件
    raise NotImplementedError
