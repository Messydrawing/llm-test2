"""
评测主指标与从指标:
  主: Accuracy / Macro-F1 / MCC / Brier / ECE（将prediction离散类转概率）
  从: JSON 合法率、关键要点覆盖率（基于规则/关键词）
"""


def classification_metrics(y_true, y_pred) -> dict:
    """计算分类相关指标"""
    if not y_true:
        return {"accuracy": 0.0, "f1": 0.0, "mcc": 0.0}
    labels = list(set(y_true) | set(y_pred))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    import numpy as np

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    accuracy = cm.trace() / cm.sum()

    f1s = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))

    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    c = cm.trace()
    s = np.dot(t_sum, p_sum)
    n_samples = cm.sum()
    denom = np.sqrt((n_samples**2 - np.sum(p_sum**2)) * (n_samples**2 - np.sum(t_sum**2)))
    mcc = (c * n_samples - s) / denom if denom > 0 else 0.0
    return {"accuracy": float(accuracy), "f1": f1_macro, "mcc": float(mcc)}


def calibration_metrics(probs, y_true) -> dict:
    """计算校准指标"""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    brier = float(np.mean((probs - y_true) ** 2))
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.any():
            conf = probs[mask].mean()
            acc = y_true[mask].mean()
            ece += mask.mean() * abs(conf - acc)
    return {"brier": brier, "ece": float(ece)}


def json_validity_rate(json_list) -> float:
    """统计 JSON 合法率"""
    import json

    if not json_list:
        return 0.0
    ok = 0
    for s in json_list:
        try:
            json.loads(s)
            ok += 1
        except Exception:
            pass
    return ok / len(json_list)
