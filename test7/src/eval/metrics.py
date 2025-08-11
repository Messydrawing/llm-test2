"""
评测主指标与从指标:
  主: Accuracy / Macro-F1 / MCC / Brier / ECE（将prediction离散类转概率）
  从: JSON 合法率、关键要点覆盖率（基于规则/关键词）
"""


def classification_metrics(y_true, y_pred) -> dict:
    """计算分类相关指标"""
    raise NotImplementedError


def calibration_metrics(probs, y_true) -> dict:
    """计算校准指标"""
    raise NotImplementedError


def json_validity_rate(json_list) -> float:
    """统计 JSON 合法率"""
    raise NotImplementedError
