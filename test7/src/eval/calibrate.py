"""温度缩放/Platt 校准（针对 prediction 概率输出）。"""

import numpy as np


def temperature_scaling(logits, labels):
    """返回最佳温度T与校准后概率"""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    temps = np.linspace(0.5, 5.0, 20)
    best_t = 1.0
    best_nll = float("inf")
    best_probs = softmax(logits)
    for t in temps:
        probs = softmax(logits / t)
        nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        if nll < best_nll:
            best_nll = nll
            best_t = t
            best_probs = probs
    return best_t, best_probs
