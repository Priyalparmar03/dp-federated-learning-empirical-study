"""Standard classification metrics, computed on the held-out global
val/test set against the aggregated global model each round.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float

    def to_dict(self):
        return asdict(self)


def evaluate(model, X: np.ndarray, y: np.ndarray) -> ClassificationMetrics:
    probs = model.predict_proba(X)
    preds = probs.argmax(axis=1)

    n = len(y)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y] = 1.0
    loss = float(-np.sum(y_onehot * np.log(np.clip(probs, 1e-9, 1.0))) / n)

    avg = "binary" if probs.shape[1] == 2 else "macro"
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y, preds)),
        precision=float(precision_score(y, preds, average=avg, zero_division=0)),
        recall=float(recall_score(y, preds, average=avg, zero_division=0)),
        f1=float(f1_score(y, preds, average=avg, zero_division=0)),
        loss=loss,
    )


def confusion(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    preds = model.predict(X)
    return confusion_matrix(y, preds)
