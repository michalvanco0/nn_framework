from __future__ import annotations
import csv
import json
import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from evaluation.fairness import calculate_fairness_metrics


def run_fairness_audit(
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any | None = None,
    *,
    audit_features: Any | None = None,
    probabilities: Any | None = None,
    class_labels: Any | None = None,
    positive_label: Any | None = None,
    reference_groups: Mapping[str, Any] | None = None,
    accepted_mask: Any | None = None,
    small_group_warning_threshold: int | None = 30,
    roc_curve_group_limit: int | None = 100,
    metadata: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if audit_features is None:
        audit_features = sensitive_features
    elif sensitive_features is not None:
        raise ValueError("Pass audit_features or sensitive_features, not both")
    if audit_features is None:
        raise ValueError("audit_features must contain at least one feature/group column")

    report = calculate_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        audit_features=audit_features,
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
        reference_groups=reference_groups,
        accepted_mask=accepted_mask,
        small_group_warning_threshold=small_group_warning_threshold,
        roc_curve_group_limit=roc_curve_group_limit,
        metadata=metadata,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)

    return report


def build_feature_audit_groups(
    training_features: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    numeric_bins: int = 5,
    max_categories: int = 25,
    include_intersections: bool = True,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Create comparable audit groups for raw features and feature pairs.

    Numeric cut points and high-cardinality category levels are learned from
    the training partition and then applied unchanged to the evaluation rows.
    """
    if numeric_bins < 2:
        raise ValueError("numeric_bins must be at least 2")
    if max_categories < 2:
        raise ValueError("max_categories must be at least 2")
    selected_columns = list(columns or training_features.columns)
    if not selected_columns:
        raise ValueError("At least one feature column is required for group audits")
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("Feature audit column names must be unique")
    missing = [
        column
        for column in selected_columns
        if column not in training_features or column not in evaluation_features
    ]
    if missing:
        raise ValueError(f"Feature data is missing audit columns: {missing}")

    def is_missing(value: Any) -> bool:
        try:
            missing_value = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing_value) if isinstance(missing_value, (bool, np.bool_)) else False

    def python_scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    groups: dict[str, list[Any]] = {}
    feature_metadata: dict[str, Any] = {}

    for column in selected_columns:
        train_values = training_features[column]
        evaluation_values = evaluation_features[column]
        if pd.api.types.is_numeric_dtype(train_values):
            numeric_train = (
                pd.to_numeric(train_values, errors="coerce")
                .astype("float64")
                .to_numpy()
            )
            finite_train = numeric_train[np.isfinite(numeric_train)]
            distinct_count = int(np.unique(finite_train).size)
            if distinct_count > 1:
                quantile_count = min(numeric_bins, distinct_count)
                _, cut_edges = pd.qcut(
                    finite_train,
                    q=quantile_count,
                    retbins=True,
                    duplicates="drop",
                )
                cut_edges = np.unique(np.asarray(cut_edges, dtype=float))
            else:
                cut_edges = np.asarray([], dtype=float)

            if len(cut_edges) >= 3:
                bin_edges = cut_edges.copy()
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                numeric_evaluation = pd.to_numeric(
                    evaluation_values, errors="coerce"
                ).astype("float64").to_numpy()
                binned = pd.cut(
                    pd.Series(numeric_evaluation).where(np.isfinite(numeric_evaluation)),
                    bins=bin_edges,
                    include_lowest=True,
                )
                groups[column] = [
                    None if is_missing(value) else str(value)
                    for value in binned.tolist()
                ]
                feature_metadata[column] = {
                    "kind": "numeric",
                    "grouping": "quantile_bins",
                    "bin_count": len(bin_edges) - 1,
                    "training_cut_points": [float(value) for value in cut_edges[1:-1]],
                    "cut_points_source": "training_partition",
                }
            else:
                numeric_evaluation = pd.to_numeric(
                    evaluation_values, errors="coerce"
                ).astype("float64")
                groups[column] = [
                    None if not math.isfinite(value) else python_scalar(value)
                    for value in numeric_evaluation.tolist()
                ]
                feature_metadata[column] = {
                    "kind": "numeric",
                    "grouping": "observed_value",
                    "reason": "The training partition has fewer than two distinct finite values.",
                }
            continue

        train_nonmissing = [
            python_scalar(value)
            for value in train_values.tolist()
            if not is_missing(value)
        ]
        unique_train_values = set(train_nonmissing)
        collapsed_levels = len(unique_train_values) > max_categories
        other_label = "__OTHER_LEVELS__"
        while other_label in unique_train_values:
            other_label += "_"
        if collapsed_levels:
            counts = pd.Series(train_nonmissing, dtype=object).value_counts()
            retained_values = set(counts.index[: max_categories - 1].tolist())

            def normalize_category(value: Any) -> Any:
                if is_missing(value):
                    return None
                value = python_scalar(value)
                return value if value in retained_values else other_label

            grouping = "top_levels_plus_other"
            retained_count = len(retained_values)
        else:
            def normalize_category(value: Any) -> Any:
                return None if is_missing(value) else python_scalar(value)

            grouping = "observed_category"
            retained_count = len(unique_train_values)

        groups[column] = [normalize_category(value) for value in evaluation_values.tolist()]
        feature_metadata[column] = {
            "kind": "categorical",
            "grouping": grouping,
            "training_level_count": len(unique_train_values),
            "retained_level_count": retained_count,
            "max_categories": max_categories if collapsed_levels else None,
            "other_level_label": other_label if collapsed_levels else None,
            "level_selection_source": "training_partition",
        }

    intersections = []
    if include_intersections:
        single_feature_groups = {name: list(values) for name, values in groups.items()}
        for left_name, right_name in combinations(selected_columns, 2):
            interaction_name = f"intersection::{left_name} × {right_name}"
            left_values = single_feature_groups[left_name]
            right_values = single_feature_groups[right_name]
            groups[interaction_name] = [
                None if left is None or right is None else (left, right)
                for left, right in zip(left_values, right_values)
            ]
            intersections.append(
                {
                    "attribute": interaction_name,
                    "features": [left_name, right_name],
                }
            )

    setup = {
        "method": "Audit each selected feature and each pairwise feature intersection",
        "numeric_grouping": (
            f"Up to {numeric_bins} quantile bins; cut points are learned from the training partition."
        ),
        "categorical_grouping": (
            f"Observed categories are retained up to {max_categories}; higher-cardinality "
            "features keep the most frequent training categories and combine the rest as other."
        ),
        "features": feature_metadata,
        "intersections": intersections,
    }
    return groups, setup


def write_disparity_ranking(report: Mapping[str, Any], output_path: Path) -> Path:
    """Write the JSON report's sorted disparity summary as a flat CSV."""
    output_path = Path(output_path)
    rows = report["disparity_ranking"]["ranked_attributes"]
    fieldnames = [
        "rank",
        "attribute",
        "attribute_type",
        "disparity_score",
        "class_label",
        "equal_opportunity_gap",
        "true_positive_rate_gap",
        "false_positive_rate_gap",
        "accuracy_gap",
        "sample_count",
        "group_count",
        "minimum_group_sample_count",
        "small_group_count",
        "most_disparate_groups",
        "class_results",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if key in {"class_results", "most_disparate_groups"}
                    else row.get(key)
                    for key in fieldnames
                }
            )
    return output_path
