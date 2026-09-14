"""Performance metrics utilities."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Calculate core classification metrics and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
