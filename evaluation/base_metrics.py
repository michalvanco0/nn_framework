"""Binary and multiclass metrics listed in the project's Prehľad table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_hausdorff_distance(
    reference_points: Sequence[Sequence[float]],
    predicted_points: Sequence[Sequence[float]],
) -> float:
    """Calculate the symmetric Hausdorff distance between coordinate sets."""
    reference = np.asarray(reference_points, dtype=float)
    predicted = np.asarray(predicted_points, dtype=float)
    if reference.ndim != 2 or predicted.ndim != 2:
        raise ValueError("Hausdorff inputs must be two-dimensional coordinate arrays")
    if reference.shape[1] != predicted.shape[1]:
        raise ValueError("Hausdorff coordinate dimensions must match")
    if len(reference) == 0 or len(predicted) == 0:
        raise ValueError("Hausdorff inputs must each contain at least one point")
    return float(
        max(
            directed_hausdorff(reference, predicted)[0],
            directed_hausdorff(predicted, reference)[0],
        )
    )


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _binary_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    positive_index: int,
    positive_label: str,
) -> dict[str, Any]:
    binary_targets = targets == positive_index
    binary_predictions = predictions == positive_index
    true_positive = int(np.sum(binary_targets & binary_predictions))
    false_positive = int(np.sum(~binary_targets & binary_predictions))
    false_negative = int(np.sum(binary_targets & ~binary_predictions))
    true_negative = int(np.sum(~binary_targets & ~binary_predictions))

    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    npv = _safe_divide(true_negative, true_negative + false_negative)
    f1 = _safe_divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    balanced = (
        float((sensitivity + specificity) / 2)
        if sensitivity is not None and specificity is not None
        else None
    )
    youden = (
        float(sensitivity + specificity - 1)
        if sensitivity is not None and specificity is not None
        else None
    )
    if np.unique(binary_targets).size < 2:
        auroc = None
        auprc = None
    else:
        class_scores = probabilities[:, positive_index]
        auroc = float(roc_auc_score(binary_targets, class_scores))
        auprc = float(average_precision_score(binary_targets, class_scores))

    return {
        "status": "measured",
        "positive_label": positive_label,
        "accuracy": float(accuracy_score(binary_targets, binary_predictions)),
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1_score": f1,
        "balanced_accuracy": balanced,
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(binary_targets, binary_predictions)
        ),
        "youden_statistic": youden,
        "auroc": auroc,
        "auprc": auprc,
    }


def _class_overlap_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_indices: np.ndarray,
) -> dict[str, Any]:
    dice_values = []
    iou_values = []
    volume_values = []
    for class_index in class_indices:
        true_mask = targets == class_index
        predicted_mask = predictions == class_index
        true_count = int(np.sum(true_mask))
        predicted_count = int(np.sum(predicted_mask))
        intersection = int(np.sum(true_mask & predicted_mask))
        union = true_count + predicted_count - intersection

        dice_denominator = true_count + predicted_count
        dice_values.append(
            float(2 * intersection / dice_denominator) if dice_denominator else 1.0
        )
        iou_values.append(float(intersection / union) if union else 1.0)
        volume_denominator = true_count + predicted_count
        volume_values.append(
            float(1 - abs(true_count - predicted_count) / volume_denominator)
            if volume_denominator
            else 1.0
        )

    return {
        "dice_similarity_coefficient_macro": float(np.mean(dice_values)),
        "intersection_over_union_jaccard_index_macro": float(np.mean(iou_values)),
        "volume_similarity_macro": float(np.mean(volume_values)),
    }


def calculate_base_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    class_names: Sequence[str],
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Calculate the applicable binary or multiclass table metrics.

    Multiclass MAE/MSE/RMSE and Pearson correlation compare the one-hot target
    matrix with the probability matrix. Dice, IoU and volume similarity compare
    the per-class sets of correctly assigned examples. Hausdorff requires
    coordinate sets, which the current classification datasets do not provide.
    """
    targets = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    class_indices = np.arange(len(class_names), dtype=int)

    if len(class_indices) < 2:
        raise ValueError("At least two classes are required")
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(targets) == 0 or len(targets) != len(predictions):
        raise ValueError("y_true and y_pred must have equal, non-zero length")
    if scores.shape != (len(targets), len(class_indices)):
        raise ValueError("probabilities must have one column per class_name")
    if not np.all(np.isin(targets, class_indices)) or not np.all(
        np.isin(predictions, class_indices)
    ):
        raise ValueError("y_true and y_pred must use class_names indices")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1")

    binary = {
        "status": "not_applicable",
        "reason": "The task is not binary with a named positive outcome.",
    }
    if positive_label is not None:
        if len(class_indices) != 2:
            raise ValueError("The current binary metrics require exactly two classes")
        if positive_label not in class_names:
            raise ValueError(f"Unknown positive_label {positive_label!r}")
        positive_index = list(class_names).index(positive_label)
        binary = _binary_metrics(
            targets,
            predictions,
            scores,
            positive_index=positive_index,
            positive_label=positive_label,
        )

    multiclass: dict[str, Any] = {
        "status": "not_applicable",
        "reason": "The task has two classes; multiclass-only metrics are omitted.",
    }
    if len(class_indices) > 2:
        one_hot = np.column_stack([targets == index for index in class_indices]).astype(float)
        errors = one_hot - scores
        flattened_targets = one_hot.ravel()
        flattened_scores = scores.ravel()
        pearson = (
            float(np.corrcoef(flattened_targets, flattened_scores)[0, 1])
            if np.std(flattened_targets) > 0 and np.std(flattened_scores) > 0
            else None
        )
        multiclass = {
            "status": "measured",
            "multiclass_accuracy": float(accuracy_score(targets, predictions)),
            "averaged_metrics": {
                average: {
                    "precision": float(
                        precision_score(
                            targets,
                            predictions,
                            labels=class_indices,
                            average=average,
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            targets,
                            predictions,
                            labels=class_indices,
                            average=average,
                            zero_division=0,
                        )
                    ),
                    "f1_score": float(
                        f1_score(
                            targets,
                            predictions,
                            labels=class_indices,
                            average=average,
                            zero_division=0,
                        )
                    ),
                }
                for average in ("macro", "micro", "weighted")
            },
            "mean_absolute_error": float(np.mean(np.abs(errors))),
            "mean_squared_error": float(np.mean(errors**2)),
            "root_mean_square_error": float(np.sqrt(np.mean(errors**2))),
            "pearson_correlation_coefficient": pearson,
            **_class_overlap_metrics(targets, predictions, class_indices),
            "hausdorff_distance": {
                "value": None,
                "status": "not_applicable",
                "reason": (
                    "The current classification datasets have no coordinate sets. "
                    "Use calculate_hausdorff_distance with spatial points for a task "
                    "that supplies them."
                ),
            },
        }

    return {
        "binary_classification": binary,
        "multiclass_classification": multiclass,
    }
