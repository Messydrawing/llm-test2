"""测试指标函数的占位实现"""
from src.eval.metrics import classification_metrics


def test_classification_metrics_stub():
    try:
        classification_metrics([], [])
    except NotImplementedError:
        pass
