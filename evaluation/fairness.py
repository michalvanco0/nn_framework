from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_1d(values: Any, name: str) -> np.ndarray:
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()

    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        return values

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of sample values")

    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc

    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_value(value: Any) -> Any:
    value = _python_value(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _is_missing(value: Any) -> bool:
    value = _python_value(value)
    if value is None:
        return True
    if type(value).__name__ == "NAType" and type(value).__module__.startswith("pandas"):
        return True
    try:
        result = np.isnan(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _same_value(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _mask_for_label(values: np.ndarray, label: Any) -> np.ndarray:
    return np.fromiter(
        (_same_value(value, label) for value in values),
        dtype=bool,
        count=len(values),
    )


def _unique_values(*sequences: np.ndarray) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for sequence in sequences:
        for raw_value in sequence:
            value = _python_value(raw_value)
            if _is_missing(value):
                raise ValueError("y_true and evaluated y_pred cannot contain missing labels")
            try:
                if value not in seen:
                    seen.add(value)
                    result.append(value)
            except TypeError as exc:
                raise ValueError("Class labels must be hashable scalar values") from exc
    return result


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(y_true) == 0:
        empty = {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
        }
        per_class = [
            {
                "class_label": _json_value(label),
                "support": 0,
                "predicted_count": 0,
                "selection_rate": None,
                "true_positive_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "true_negative_rate": None,
                "positive_predictive_value": None,
                "negative_predictive_value": None,
                "f1": None,
            }
            for label in class_labels
        ]
        return empty, per_class

    correct = np.fromiter(
        (_same_value(actual, predicted) for actual, predicted in zip(y_true, y_pred)),
        dtype=bool,
        count=len(y_true),
    )

    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for label in class_labels:
        actual_positive = _mask_for_label(y_true, label)
        predicted_positive = _mask_for_label(y_pred, label)
        tp = int(np.sum(actual_positive & predicted_positive))
        fp = int(np.sum(~actual_positive & predicted_positive))
        fn = int(np.sum(actual_positive & ~predicted_positive))
        tn = int(np.sum(~actual_positive & ~predicted_positive))

        tpr = _safe_divide(tp, tp + fn)
        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, tp + fn)
        tnr = _safe_divide(tn, tn + fp)
        ppv = _safe_divide(tp, tp + fp)
        npv = _safe_divide(tn, tn + fn)
        selection_rate = float(np.mean(predicted_positive))
        precision_for_macro = 0.0 if ppv is None else ppv
        recall_for_macro = 0.0 if tpr is None else tpr
        f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        f1_for_macro = 0.0 if f1 is None else f1

        precisions.append(precision_for_macro)
        if tpr is not None:
            recalls.append(tpr)
        f1_scores.append(f1_for_macro)

        per_class.append(
            {
                "class_label": _json_value(label),
                "support": int(tp + fn),
                "predicted_count": int(tp + fp),
                "selection_rate": selection_rate,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "true_negative_rate": tnr,
                "positive_predictive_value": ppv,
                "negative_predictive_value": npv,
                "f1": f1,
            }
        )

    metrics = {
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
    }
    return metrics, per_class


def _class_distribution(values: np.ndarray, class_labels: list[Any]) -> list[dict[str, Any]]:
    total = len(values)
    distribution = []
    for label in class_labels:
        count = int(np.sum(_mask_for_label(values, label)))
        distribution.append(
            {
                "class_label": _json_value(label),
                "count": count,
                "rate": float(count / total) if total else None,
            }
        )
    return distribution


def _probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[Any],
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "brier_score": None,
            "log_loss": None,
            "classwise_roc_auc": [],
        }

    label_to_index = {label: index for index, label in enumerate(class_labels)}
    true_indices = np.fromiter(
        (label_to_index[_python_value(value)] for value in y_true),
        dtype=int,
        count=len(y_true),
    )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), true_indices] = 1.0

    clipped = np.clip(probabilities, 1e-15, 1.0)
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(clipped[np.arange(len(y_true)), true_indices])))

    classwise_roc_auc = []
    for class_index, label in enumerate(class_labels):
        binary_truth = one_hot[:, class_index].astype(int)
        roc_auc = (
            float(roc_auc_score(binary_truth, probabilities[:, class_index]))
            if len(np.unique(binary_truth)) == 2
            else None
        )
        classwise_roc_auc.append(
            {
                "class_label": _json_value(label),
                "roc_auc": roc_auc,
            }
        )

    return {
        "brier_score": brier_score,
        "log_loss": log_loss,
        "classwise_roc_auc": classwise_roc_auc,
    }


def _gap(values: list[tuple[Any, float | None]]) -> dict[str, Any]:
    valid = [(group, float(value)) for group, value in values if value is not None]
    if len(valid) < 2:
        return {
            "max_minus_min": None,
            "minimum": None,
            "maximum": None,
            "minimum_groups": [],
            "maximum_groups": [],
            "valid_group_count": len(valid),
            "reason": "At least two groups need a defined value.",
        }

    minimum = min(value for _, value in valid)
    maximum = max(value for _, value in valid)
    return {
        "max_minus_min": float(maximum - minimum),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "minimum_groups": [
            _json_value(group)
            for group, value in valid
            if math.isclose(value, minimum, rel_tol=1e-12, abs_tol=1e-12)
        ],
        "maximum_groups": [
            _json_value(group)
            for group, value in valid
            if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)
        ],
        "valid_group_count": len(valid),
    }


def _attribute_columns(sensitive_features: Any) -> dict[str, Any]:
    if isinstance(sensitive_features, Mapping):
        columns = {str(name): values for name, values in sensitive_features.items()}
    elif hasattr(sensitive_features, "columns") and hasattr(sensitive_features, "__getitem__"):
        columns = {
            str(name): sensitive_features[name]
            for name in sensitive_features.columns
        }
    else:
        columns = {"sensitive_attribute": sensitive_features}

    if not columns:
        raise ValueError("sensitive_features must contain at least one attribute")
    return columns


def _group_ranges(
    group_results: list[dict[str, Any]],
    class_labels: list[Any],
) -> dict[str, Any]:
    standard_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    result: dict[str, Any] = {
        metric: _gap(
            [
                (group["group"], group["metrics"][metric])
                for group in group_results
            ]
        )
        for metric in standard_metrics
    }

    per_class_ranges = []
    class_rate_names = (
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "false_negative_rate",
        "true_negative_rate",
        "positive_predictive_value",
        "negative_predictive_value",
        "f1",
    )
    for class_index, label in enumerate(class_labels):
        ranges = {
            rate_name: _gap(
                [
                    (
                        group["group"],
                        group["per_class"][class_index][rate_name],
                    )
                    for group in group_results
                ]
            )
            for rate_name in class_rate_names
        }
        per_class_ranges.append(
            {
                "class_label": _json_value(label),
                "metrics": {
                    **ranges,
                    "statistical_parity_gap": ranges["selection_rate"],
                    "equal_opportunity_gap": ranges["true_positive_rate"],
                    "equalized_odds_gaps": {
                        "true_positive_rate": ranges["true_positive_rate"],
                        "false_positive_rate": ranges["false_positive_rate"],
                    },
                    "predictive_parity_gap": ranges["positive_predictive_value"],
                    "negative_predictive_parity_gap": ranges["negative_predictive_value"],
                },
            }
        )
    result["per_class"] = per_class_ranges

    probability_metrics = ("brier_score", "log_loss")
    result["probability_quality"] = {
        metric: _gap(
            [
                (
                    group["group"],
                    None
                    if group["probability_metrics"] is None
                    else group["probability_metrics"][metric],
                )
                for group in group_results
            ]
        )
        for metric in probability_metrics
    }
    result["probability_quality"]["classwise_roc_auc_gap"] = [
        {
            "class_label": _json_value(label),
            "gap": _gap(
                [
                    (
                        group["group"],
                        None
                        if group["probability_metrics"] is None
                        else group["probability_metrics"]["classwise_roc_auc"][class_index]["roc_auc"],
                    )
                    for group in group_results
                ]
            ),
        }
        for class_index, label in enumerate(class_labels)
    ]

    result["selective"] = {
        "coverage": _gap(
            [
                (group["group"], group["coverage"])
                for group in group_results
            ]
        ),
        "selective_accuracy": _gap(
            [
                (group["group"], group["selective_accuracy"])
                for group in group_results
            ]
        ),
    }
    return result


def _reference_comparisons(
    group_results: list[dict[str, Any]],
    reference_value: Any,
    class_labels: list[Any],
) -> dict[str, Any]:
    reference = next(
        (
            group
            for group in group_results
            if _same_value(group["_raw_group"], reference_value)
        ),
        None,
    )
    if reference is None:
        raise ValueError(
            f"Reference group {_json_value(reference_value)!r} was not found "
            "among the non-missing values of this sensitive attribute"
        )

    comparisons = []
    for group in group_results:
        if _same_value(group["_raw_group"], reference_value):
            continue
        per_class = []
        for class_index, label in enumerate(class_labels):
            current_rates = group["per_class"][class_index]
            reference_rates = reference["per_class"][class_index]

            def difference(metric: str) -> float | None:
                current = current_rates[metric]
                baseline = reference_rates[metric]
                if current is None or baseline is None:
                    return None
                return float(current - baseline)

            selection_difference = difference("selection_rate")
            selection_ratio = None
            reference_selection = reference_rates["selection_rate"]
            current_selection = current_rates["selection_rate"]
            if (
                current_selection is not None
                and reference_selection is not None
                and reference_selection > 0
            ):
                selection_ratio = float(
                    current_selection / reference_selection
                )

            tpr_difference = difference("true_positive_rate")
            fpr_difference = difference("false_positive_rate")
            average_odds_difference = None
            if tpr_difference is not None and fpr_difference is not None:
                average_odds_difference = float(
                    (tpr_difference + fpr_difference) / 2.0
                )

            per_class.append(
                {
                    "class_label": _json_value(label),
                    "statistical_parity_difference": selection_difference,
                    "disparate_impact_ratio": selection_ratio,
                    "equal_opportunity_difference": tpr_difference,
                    "false_positive_rate_difference": fpr_difference,
                    "average_odds_difference": average_odds_difference,
                    "predictive_parity_difference": difference(
                        "positive_predictive_value"
                    ),
                    "negative_predictive_parity_difference": difference(
                        "negative_predictive_value"
                    ),
                }
            )

        comparisons.append(
            {
                "group": group["group"],
                "per_class": per_class,
            }
        )

    return {
        "reference_group": _json_value(reference_value),
        "comparison_direction": "each reported group minus the reference group",
        "disparate_impact_ratio_definition": "group selection rate / reference selection rate",
        "comparisons": comparisons,
    }


def _abroca_by_class(
    group_results: list[dict[str, Any]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[Any],
) -> list[dict[str, Any]]:
    """Compare group ROC curves pairwise using absolute area between curves."""
    results = []
    for class_index, label in enumerate(class_labels):
        group_curves = []
        for group in group_results:
            indices = group["_indices"]
            binary_truth = _mask_for_label(y_true[indices], label).astype(int)
            if len(np.unique(binary_truth)) < 2:
                continue
            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_truth,
                probabilities[indices, class_index],
            )
            group_curves.append(
                (
                    group["group"],
                    false_positive_rate,
                    true_positive_rate,
                )
            )

        pairwise = []
        for left_index in range(len(group_curves)):
            left_group, left_fpr, left_tpr = group_curves[left_index]
            for right_index in range(left_index + 1, len(group_curves)):
                right_group, right_fpr, right_tpr = group_curves[right_index]
                fpr_grid = np.unique(np.concatenate((left_fpr, right_fpr)))
                left_interpolated = np.interp(fpr_grid, left_fpr, left_tpr)
                right_interpolated = np.interp(fpr_grid, right_fpr, right_tpr)
                signed_distance = left_interpolated - right_interpolated
                segment_areas = []
                for index, width in enumerate(np.diff(fpr_grid)):
                    left_distance = float(signed_distance[index])
                    right_distance = float(signed_distance[index + 1])
                    if left_distance * right_distance < 0.0:
                        denominator = abs(left_distance) + abs(right_distance)
                        segment_area = float(width) * (
                            left_distance**2 + right_distance**2
                        ) / (2.0 * denominator)
                    else:
                        segment_area = float(width) * (
                            abs(left_distance) + abs(right_distance)
                        ) / 2.0
                    segment_areas.append(segment_area)
                abroca = float(np.sum(segment_areas))
                pairwise.append(
                    {
                        "group_a": _json_value(left_group),
                        "group_b": _json_value(right_group),
                        "abroca": abroca,
                    }
                )

        results.append(
            {
                "class_label": _json_value(label),
                "pairwise_comparisons": pairwise,
                "valid_pair_count": len(pairwise),
                "reason": (
                    None
                    if pairwise
                    else "Each compared group needs both positive and negative examples."
                ),
            }
        )
    return results


def calculate_fairness_metrics(
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any,
    *,
    probabilities: Any | None = None,
    class_labels: Any | None = None,
    positive_label: Any | None = None,
    reference_groups: Mapping[str, Any] | None = None,
    accepted_mask: Any | None = None,
    small_group_warning_threshold: int | None = 30,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    true_values = _as_1d(y_true, "y_true")
    predicted_values = _as_1d(y_pred, "y_pred")
    if len(true_values) == 0:
        raise ValueError("At least one sample is required")
    if len(true_values) != len(predicted_values):
        raise ValueError("y_true and y_pred must contain the same number of samples")

    if accepted_mask is None:
        accepted = np.ones(len(true_values), dtype=bool)
    else:
        accepted = _as_1d(accepted_mask, "accepted_mask").astype(bool)
        if len(accepted) != len(true_values):
            raise ValueError("accepted_mask must contain one value per sample")
    if not np.any(accepted):
        raise ValueError("accepted_mask must accept at least one sample")

    evaluated_predictions = predicted_values[accepted]
    labels = (
        [_python_value(value) for value in _as_1d(class_labels, "class_labels")]
        if class_labels is not None
        else _unique_values(true_values, evaluated_predictions)
    )
    if len(labels) < 2:
        raise ValueError("Classification fairness metrics require at least two classes")
    try:
        if len(set(labels)) != len(labels):
            raise ValueError("class_labels must not contain duplicates")
    except TypeError as exc:
        raise ValueError("Class labels must be hashable scalar values") from exc
    if any(_is_missing(label) for label in labels):
        raise ValueError("class_labels cannot contain missing values")

    label_set = set(labels)
    for value in true_values:
        if _is_missing(value):
            raise ValueError("y_true cannot contain missing labels")
        if _python_value(value) not in label_set:
            raise ValueError(f"Ground-truth label {_json_value(value)!r} is absent from class_labels")
    for value in evaluated_predictions:
        if _is_missing(value):
            raise ValueError("Evaluated y_pred values cannot contain missing labels")
        if _python_value(value) not in label_set:
            raise ValueError(f"Predicted label {_json_value(value)!r} is absent from class_labels")

    if positive_label is not None and positive_label not in label_set:
        raise ValueError("positive_label must be included in class_labels")
    if positive_label is None and len(labels) == 2 and set(labels) == {0, 1}:
        positive_label = 1

    if small_group_warning_threshold is not None and small_group_warning_threshold < 1:
        raise ValueError("small_group_warning_threshold must be positive or None")

    probability_values = None
    if probabilities is not None:
        if hasattr(probabilities, "detach") and hasattr(probabilities, "cpu"):
            probabilities = probabilities.detach().cpu().numpy()
        probability_values = np.asarray(probabilities, dtype=float)
        is_positive_probability_vector = probability_values.ndim == 1
        if probability_values.ndim == 1:
            if len(labels) != 2 or positive_label is None:
                raise ValueError(
                    "One-dimensional probabilities require a binary task and positive_label"
                )
            if len(probability_values) != len(true_values):
                raise ValueError("probabilities must contain one row per sample")
            positive_index = labels.index(positive_label)
            expanded = np.empty((len(probability_values), 2), dtype=float)
            expanded[:, positive_index] = probability_values
            expanded[:, 1 - positive_index] = 1.0 - probability_values
            probability_values = expanded
        if probability_values.ndim != 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes)")
        if not is_positive_probability_vector and class_labels is None:
            raise ValueError(
                "Pass class_labels in probability-column order when probabilities are two-dimensional"
            )
        if probability_values.shape != (len(true_values), len(labels)):
            raise ValueError(
                "probabilities must have one row per sample and one column per class_label"
            )
        if not np.all(np.isfinite(probability_values)):
            raise ValueError("probabilities must be finite")
        if np.any(probability_values < 0.0) or np.any(probability_values > 1.0):
            raise ValueError("probabilities must be between 0 and 1")
        if not np.allclose(np.sum(probability_values, axis=1), 1.0, atol=1e-2):
            raise ValueError("Each probability row must sum to 1")

    columns = _attribute_columns(sensitive_features)
    references = {str(name): value for name, value in (reference_groups or {}).items()}
    unknown_references = set(references) - set(columns)
    if unknown_references:
        raise ValueError(
            "reference_groups names unknown sensitive attributes: "
            + ", ".join(sorted(unknown_references))
        )

    attributes: dict[str, Any] = {}
    for name, raw_values in columns.items():
        group_values = _as_1d(raw_values, f"sensitive_features[{name!r}]")
        if len(group_values) != len(true_values):
            raise ValueError(
                f"Sensitive attribute {name!r} must contain one value per sample"
            )

        group_indices: dict[Any, list[int]] = {}
        missing_count = 0
        for index, raw_value in enumerate(group_values):
            if _is_missing(raw_value):
                missing_count += 1
                continue
            value = _python_value(raw_value)
            try:
                group_indices.setdefault(value, []).append(index)
            except TypeError as exc:
                raise ValueError(
                    f"Values in sensitive attribute {name!r} must be hashable"
                ) from exc

        group_results = []
        for raw_group, indices_list in group_indices.items():
            indices = np.asarray(indices_list, dtype=int)
            evaluated_indices = indices[accepted[indices]]
            group_true = true_values[evaluated_indices]
            group_pred = predicted_values[evaluated_indices]
            metrics, per_class = _class_metrics(group_true, group_pred, labels)
            group_correct = np.fromiter(
                (_same_value(actual, predicted) for actual, predicted in zip(group_true, group_pred)),
                dtype=bool,
                count=len(group_true),
            )

            probability_result = None
            if probability_values is not None:
                probability_result = _probability_metrics(
                    true_values[indices],
                    probability_values[indices],
                    labels,
                )

            coverage = float(len(evaluated_indices) / len(indices))
            selective_accuracy = (
                float(np.mean(group_correct)) if len(group_correct) else None
            )
            group_results.append(
                {
                    "group": _json_value(raw_group),
                    "_raw_group": raw_group,
                    "_indices": indices,
                    "sample_count": int(len(indices)),
                    "evaluated_count": int(len(evaluated_indices)),
                    "coverage": coverage,
                    "selective_accuracy": selective_accuracy,
                    "true_class_distribution": _class_distribution(
                        true_values[indices], labels
                    ),
                    "predicted_class_distribution": _class_distribution(
                        group_pred, labels
                    ),
                    "metrics": metrics,
                    "per_class": per_class,
                    "probability_metrics": probability_result,
                }
            )

        reference_comparison = None
        if name in references:
            reference_comparison = _reference_comparisons(
                group_results,
                references[name],
                labels,
            )

        warnings = []
        if missing_count:
            warnings.append(
                f"{missing_count} samples have missing values for {name!r}; "
                "they are counted but excluded from that attribute's group comparisons."
            )
        if len(group_results) < 2:
            warnings.append(
                "At least two non-missing groups are needed to calculate between-group gaps."
            )
        for group in group_results:
            if (
                small_group_warning_threshold is not None
                and group["sample_count"] < small_group_warning_threshold
            ):
                warnings.append(
                    f"Group {group['group']!r} has only {group['sample_count']} samples; "
                    "its metric estimates may be unstable."
                )

        public_groups = [
            {
                key: value
                for key, value in group.items()
                if key not in {"_raw_group", "_indices"}
            }
            for group in group_results
        ]
        roc_curve_comparisons = (
            None
            if probability_values is None
            else _abroca_by_class(
                group_results,
                true_values,
                probability_values,
                labels,
            )
        )
        attributes[name] = {
            "sample_count": int(len(true_values)),
            "included_sample_count": int(len(true_values) - missing_count),
            "missing_sensitive_value_count": int(missing_count),
            "group_count": int(len(group_results)),
            "group_metrics": public_groups,
            "across_group_gaps": _group_ranges(group_results, labels),
            "roc_curve_comparisons": roc_curve_comparisons,
            "reference_group_comparisons": reference_comparison,
            "warnings": warnings,
        }

    overall_metrics, overall_per_class = _class_metrics(
        true_values[accepted],
        predicted_values[accepted],
        labels,
    )
    overall_probability_metrics = (
        None
        if probability_values is None
        else _probability_metrics(true_values, probability_values, labels)
    )

    return {
        "task": "classification",
        "metadata": _json_value(dict(metadata)) if metadata is not None else {},
        "sample_count": int(len(true_values)),
        "overall_metrics": {
            **overall_metrics,
            "per_class": overall_per_class,
            "coverage": float(np.mean(accepted)),
            "probability_metrics": overall_probability_metrics,
        },
        "class_labels": [_json_value(label) for label in labels],
        "positive_label": _json_value(positive_label),
        "probability_metrics_included": probability_values is not None,
        "selective_metrics_included": accepted_mask is not None,
        "small_group_warning_threshold": small_group_warning_threshold,
        "classification_metric_population": (
            "accepted samples only" if accepted_mask is not None else "all samples"
        ),
        "probability_metric_population": "all samples",
        "report_semantics": (
            "Descriptive measurements only. No aggregate fairness score, threshold, "
            "ranking, or fair/unfair verdict is applied."
        ),
        "attributes": attributes,
        "limitations": [
            "Metric meaning depends on the selected task, class, sensitive attributes, and context.",
            "Group gaps describe observed differences and do not establish their cause or social meaning.",
            "Different fairness criteria can disagree; the report keeps their measurements separate.",
            "Probability-quality metrics require predicted probabilities. Calibration error is evaluated separately with ACE in the calibration module.",
            "No confidence intervals or statistical significance tests are calculated.",
            "Reference-based differences are emitted only for reference groups supplied by the caller.",
            "This module covers classification; causal, individual, ranking, and regression fairness require task-specific inputs and assumptions.",
        ],
        "not_measured": [
            "Sensitive attributes not supplied by the caller.",
            "Causal explanations for observed group differences.",
            "Individual or counterfactual fairness.",
            "Regression or ranking fairness.",
            "Whether any measured difference is acceptable for a particular use.",
        ],
    }
