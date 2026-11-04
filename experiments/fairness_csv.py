from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.fairness_plots import (
    save_group_metric_panels,
    save_group_roc_curve_comparisons,
)
from experiments.fairness import run_fairness_audit


def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Input CSV column names must be unique")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {input_path}")
    return rows, list(reader.fieldnames)


def _get_column(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    column: str,
    *,
    allow_missing: bool = False,
) -> list[str | None]:
    if column not in fieldnames:
        raise ValueError(f"Input CSV is missing column {column!r}")

    values = []
    for row_index, row in enumerate(rows, start=2):
        raw_value = row.get(column)
        value = raw_value.strip() if raw_value is not None else ""
        if not value and not allow_missing:
            raise ValueError(
                f"Column {column!r} contains a missing value on CSV row {row_index}"
            )
        values.append(value or None)
    return values


def _parse_reference_group(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Reference groups must use ATTRIBUTE=VALUE syntax"
        )
    attribute, group = value.split("=", maxsplit=1)
    if not attribute or not group:
        raise argparse.ArgumentTypeError(
            "Both the reference attribute and group value are required"
        )
    return attribute, group


def _parse_accepted_values(values: list[str | None], column: str) -> np.ndarray:
    accepted = []
    true_values = {"1", "true", "yes", "accepted"}
    false_values = {"0", "false", "no", "rejected"}
    for row_index, value in enumerate(values, start=2):
        normalized = (value or "").strip().lower()
        if normalized in true_values:
            accepted.append(True)
        elif normalized in false_values:
            accepted.append(False)
        else:
            raise ValueError(
                f"Column {column!r} must contain a boolean value on CSV row "
                f"{row_index}; got {value!r}"
            )
    return np.asarray(accepted, dtype=bool)


def run_fairness_csv_audit(
    input_path: Path,
    *,
    label_column: str,
    prediction_column: str,
    group_columns: list[str],
    probability_columns: list[str] | None = None,
    class_labels: list[str] | None = None,
    positive_label: str | None = None,
    reference_groups: dict[str, str] | None = None,
    accepted_column: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    split_name: str | None = None,
    output_path: Path = Path("results/fairness.json"),
) -> dict[str, Any]:
    if not group_columns:
        raise ValueError("At least one group column must be supplied")
    if len(set(group_columns)) != len(group_columns):
        raise ValueError("Group column names must be unique")
    if label_column in group_columns:
        raise ValueError(
            "The target-label column cannot also be used as a group attribute"
        )

    rows, fieldnames = _read_rows(input_path)
    y_true = _get_column(rows, fieldnames, label_column)
    y_pred = _get_column(rows, fieldnames, prediction_column)
    groups = {
        column: _get_column(rows, fieldnames, column, allow_missing=True)
        for column in group_columns
    }

    probabilities = None
    if probability_columns:
        if class_labels is None:
            raise ValueError(
                "Supply --class-labels in the same order as --probability-columns"
            )
        if len(probability_columns) not in {1, len(class_labels)}:
            raise ValueError(
                "Supply one probability column for a binary positive class, "
                "or one column per class"
            )
        if len(probability_columns) == 1 and len(class_labels) != 2:
            raise ValueError(
                "A single probability column is supported only for binary classification"
            )
        if len(probability_columns) == 1 and positive_label is None:
            raise ValueError(
                "Specify --positive-label when supplying one binary probability column"
            )
        if len(set(probability_columns)) != len(probability_columns):
            raise ValueError("Probability column names must be unique")
        probability_values = [
            _get_column(rows, fieldnames, column)
            for column in probability_columns
        ]
        probabilities = np.asarray(probability_values, dtype=float).T
        if len(probability_columns) == 1:
            probabilities = probabilities[:, 0]

    accepted_mask = None
    if accepted_column is not None:
        accepted_values = _get_column(rows, fieldnames, accepted_column)
        accepted_mask = _parse_accepted_values(accepted_values, accepted_column)

    metadata = {
        key: value
        for key, value in {
            "model": model_name,
            "dataset": dataset_name,
            "split": split_name,
            "predictions_file": str(input_path),
            "row_alignment": "all columns refer to the same example on each CSV row",
        }.items()
        if value is not None
    }
    report = run_fairness_audit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
        reference_groups=reference_groups,
        accepted_mask=accepted_mask,
        metadata=metadata,
    )

    chart_paths = {}
    roc_chart_paths = {}
    for index, attribute_name in enumerate(group_columns):
        if report["attributes"][attribute_name]["group_metrics"]:
            chart_paths[attribute_name] = save_group_metric_panels(
                report,
                sensitive_attribute=attribute_name,
                output_stem=output_path.with_name(
                    f"{output_path.stem}_group_{index + 1}_metrics"
                ),
            )
        else:
            chart_paths[attribute_name] = {}
        if probabilities is not None:
            roc_chart_paths[attribute_name] = save_group_roc_curve_comparisons(
                y_true=y_true,
                group_values=groups[attribute_name],
                probabilities=probabilities,
                class_labels=class_labels,
                positive_label=positive_label,
                output_stem=output_path.with_name(
                    f"{output_path.stem}_group_{index + 1}_roc"
                ),
            )
    report["artifacts"] = {
        "group_metric_charts": {
            attribute_name: {
                extension: path.name for extension, path in paths.items()
            }
            for attribute_name, paths in chart_paths.items()
        }
    }
    if roc_chart_paths:
        report["artifacts"]["group_roc_charts"] = {
            attribute_name: {
                class_label: {
                    extension: path.name for extension, path in paths.items()
                }
                for class_label, paths in classes.items()
            }
            for attribute_name, classes in roc_chart_paths.items()
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)

    print(f"Overall accuracy: {report['overall_metrics']['accuracy']:.4f}")
    for attribute_name in group_columns:
        attribute = report["attributes"][attribute_name]
        print(f"\nResults by group for {attribute_name!r}:")
        for group in attribute["group_metrics"]:
            print(
                f"  {group['group']}: n={group['sample_count']}, "
                f"accuracy={group['metrics']['accuracy']:.4f}, "
                f"balanced_accuracy={group['metrics']['balanced_accuracy']:.4f}"
            )
        gap = attribute["across_group_gaps"]["accuracy"]["max_minus_min"]
        if gap is not None:
            print(f"  accuracy range (max - min): {gap:.4f} ({gap * 100:.2f} pp)")
        if "png" in chart_paths[attribute_name]:
            print(f"  chart: {chart_paths[attribute_name]['png']}")
        else:
            print("  chart: skipped because the attribute has no observed groups")
        for class_label, paths in roc_chart_paths.get(attribute_name, {}).items():
            print(f"  ROC comparison for {class_label}: {paths['png']}")
    print(f"Detailed report: {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run model-agnostic classification group metrics from a CSV of "
            "aligned labels, predictions, group attributes, and optional scores"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--group-columns", nargs="+", required=True)
    parser.add_argument("--probability-columns", nargs="+")
    parser.add_argument(
        "--class-labels",
        nargs="+",
        help="Class order; required with probabilities and must match their columns",
    )
    parser.add_argument("--positive-label")
    parser.add_argument(
        "--reference-group",
        action="append",
        type=_parse_reference_group,
        default=[],
        metavar="ATTRIBUTE=VALUE",
    )
    parser.add_argument("--accepted-column")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split-name")
    parser.add_argument("--output", type=Path, default=Path("results/fairness.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fairness_csv_audit(
        input_path=args.input,
        label_column=args.label_column,
        prediction_column=args.prediction_column,
        group_columns=args.group_columns,
        probability_columns=args.probability_columns,
        class_labels=args.class_labels,
        positive_label=args.positive_label,
        reference_groups=dict(args.reference_group),
        accepted_column=args.accepted_column,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        output_path=args.output,
    )
