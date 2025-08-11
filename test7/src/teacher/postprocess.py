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
    pred = str(obj.get("prediction", "")).lower()
    mapping = {
        "up": "up",
        "increase": "up",
        "涨": "up",
        "down": "down",
        "decrease": "down",
        "跌": "down",
        "flat": "flat",
        "neutral": "flat",
        "震荡": "flat",
    }
    norm_pred = mapping.get(pred, "flat")
    analysis = str(obj.get("analysis", "")).strip()
    advice = str(obj.get("advice", "")).strip()
    return TeacherJSON(prediction=norm_pred, analysis=analysis, advice=advice)
