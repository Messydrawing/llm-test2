"""
老师输出的 JSON 清洗、字段补齐、与 prompt 对齐。
"""
from .schema import TeacherJSON


def normalize_teacher_json(obj: dict) -> TeacherJSON:
    """
    规则：
      - 严格仅保留 prediction/analysis/advice
      - prediction 归一至 up/down/flat
      - 去除额外文本/前后缀
    """
    raise NotImplementedError
