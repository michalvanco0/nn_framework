from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd


def prepare_feature_groups(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    numeric_bins: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if numeric_bins < 1:
        raise ValueError("numeric_bins must be at least 1")
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test feature columns must match in the same order")
    if train_features.empty or test_features.empty:
        raise ValueError("Train and test features must both contain rows")

    groups: dict[str, np.ndarray] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for column in train_features.columns:
        train_column = train_features[column]
        test_column = test_features[column]

        if pd.api.types.is_numeric_dtype(train_column):
            train_numeric = pd.to_numeric(train_column, errors="coerce")
            observed = train_numeric.dropna().to_numpy(dtype=float)
            if observed.size:
                quantiles = np.linspace(0.0, 1.0, numeric_bins + 1)[1:-1]
                edges = np.unique(np.quantile(observed, quantiles)).tolist()
            else:
                edges = []
            numeric_test = pd.to_numeric(test_column, errors="coerce")
            labels = [f"bin_{index + 1}" for index in range(len(edges) + 1)]
            binned = pd.cut(
                numeric_test,
                bins=[-np.inf, *edges, np.inf],
                labels=labels,
                include_lowest=True,
            ).astype("string")
            values = binned.fillna("Missing").to_numpy(dtype=object)
            definitions[column] = {
                "type": "numeric_quantile_bins",
                "requested_bins": numeric_bins,
                "actual_bins": len(labels),
                "training_quantile_edges": [float(edge) for edge in edges],
            }
        else:
            normalized = test_column.astype("string").str.strip()
            normalized = normalized.replace("", pd.NA).fillna("Missing")
            values = normalized.to_numpy(dtype=object)
            definitions[column] = {"type": "categorical_values"}

        groups[str(column)] = values

    return groups, definitions


def _gap(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return float(max(measured) - min(measured)) if len(measured) > 1 else None


def _rates_by_group(
    targets: np.ndarray,
    predictions: np.ndarray,
    group_values: Sequence[Any],
    *,
    positive_class_index: int,
) -> dict[str, Any]:
    raw_groups = np.asarray(group_values, dtype=object)
    if raw_groups.ndim != 1 or len(raw_groups) != len(targets):
        raise ValueError("Each feature group vector must match the label length")

    groups = np.asarray(
        ["Missing" if value is None else str(value) for value in raw_groups],
        dtype=object,
    )
    by_group: dict[str, Any] = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        actual_positive = targets[mask] == positive_class_index
        actual_negative = ~actual_positive
        predicted_positive = predictions[mask] == positive_class_index
        positive_count = int(np.sum(actual_positive))
        negative_count = int(np.sum(actual_negative))
        by_group[name] = {
            "sample_count": int(np.sum(mask)),
            "true_positive_rate": (
                float(np.sum(predicted_positive & actual_positive) / positive_count)
                if positive_count
                else None
            ),
            "false_positive_rate": (
                float(np.sum(predicted_positive & actual_negative) / negative_count)
                if negative_count
                else None
            ),
        }

    tpr_gap = _gap([value["true_positive_rate"] for value in by_group.values()])
    fpr_gap = _gap([value["false_positive_rate"] for value in by_group.values()])
    components = [value for value in (tpr_gap, fpr_gap) if value is not None]
    return {
        "groups": by_group,
        "true_positive_rate_gap": tpr_gap,
        "false_positive_rate_gap": fpr_gap,
        "equal_opportunity": tpr_gap,
        "equalized_odds": max(components) if components else None,
    }


def calculate_fairness_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    feature_groups: Mapping[str, Sequence[Any]],
    *,
    positive_class_index: int,
) -> dict[str, Any]:
    targets = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional")
    if len(targets) == 0 or len(targets) != len(predictions):
        raise ValueError("Labels and predictions must have equal, non-zero length")
    if positive_class_index < 0 or not np.any(targets == positive_class_index):
        raise ValueError("positive_class_index must be present in y_true")
    if not feature_groups:
        raise ValueError("At least one input feature group vector is required")

    feature_results = {
        name: _rates_by_group(
            targets,
            predictions,
            values,
            positive_class_index=positive_class_index,
        )
        for name, values in feature_groups.items()
    }
    return {
        "status": "measured",
        "features_audited": list(feature_groups),
        "gap_unit": "fraction",
        "features": feature_results,
    }
