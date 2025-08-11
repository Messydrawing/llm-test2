"""输入输出相关工具函数"""
import json
from typing import Iterable, Any


def read_jsonl(path: str) -> Iterable[Any]:
    """读取 JSONL 文件, 返回对象迭代器"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(items: Iterable[Any], path: str):
    """将对象列表写入 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            if hasattr(obj, "model_dump"):
                obj = obj.model_dump()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
