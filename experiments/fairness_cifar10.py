from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.evaluator import evaluate
from evaluation.fairness_plots import (
    save_group_metric_panels,
    save_group_roc_curve_comparisons,
)
from experiments.fairness import run_fairness_audit
from models.resnet import create_resnet18


def _load_resnet18(
    checkpoint_path: Path,
    class_count: int,
    device: torch.device,
    architecture: str,
) -> tuple[nn.Module, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if architecture == "auto":
        conv1_weight = checkpoint.get("conv1.weight")
        if conv1_weight is None or conv1_weight.ndim != 4:
            raise ValueError(
                "Cannot infer ResNet-18 architecture: checkpoint must be a state "
                "dict containing conv1.weight."
            )
        kernel_size = conv1_weight.shape[-1]
        if kernel_size == 7:
            architecture = "torchvision"
        elif kernel_size == 3:
            architecture = "cifar"
        else:
            raise ValueError(f"Unsupported ResNet-18 conv1 kernel size: {kernel_size}")

    if architecture == "torchvision":
        model = torchvision_resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    else:
        model = create_resnet18(num_classes=class_count)

    model = model.to(device)
    model.load_state_dict(checkpoint)
    return model, architecture


def _load_group_metadata(
    metadata_path: Path,
    group_columns: list[str],
    expected_count: int,
) -> dict[str, list[str | None]]:
    if not group_columns:
        raise ValueError("At least one group column must be provided")
    if len(set(group_columns)) != len(group_columns):
        raise ValueError("Group column names must be unique")

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None:
            raise ValueError(f"Group metadata CSV is empty: {metadata_path}")
        missing_columns = set(group_columns) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Group metadata is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows = list(reader)

    if len(rows) != expected_count:
        raise ValueError(
            f"Group metadata has {len(rows)} rows, but the evaluated test split "
            f"has {expected_count} samples."
        )

    return {
        column: [
            ((row.get(column) or "").strip() or None)
            for row in rows
        ]
        for column in group_columns
    }


def run_cifar10_fairness_audit(
    checkpoint_path: Path = Path("results/resnet18_cifar10.pt"),
    data_dir: Path = Path("data"),
    output_path: Path = Path("results/fairness_cifar10.json"),
    group_metadata_path: Path | None = None,
    group_columns: list[str] | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    architecture: str = "auto",
) -> dict[str, Any]:
    if group_metadata_path is None or not group_columns:
        raise ValueError(
            "CIFAR-10 has no demographic or other group metadata. Provide an "
            "external CSV aligned row-for-row with the CIFAR-10 test split and "
            "the group column names; the script will not derive groups from "
            "ground-truth classes."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}")

    _, test_loader, class_names = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    groups = _load_group_metadata(
        metadata_path=group_metadata_path,
        group_columns=group_columns,
        expected_count=len(test_loader.dataset),
    )
    model, resolved_architecture = _load_resnet18(
        checkpoint_path,
        len(class_names),
        device,
        architecture,
    )
    print(f"Model: ResNet-18 ({resolved_architecture})")
    print(f"Checkpoint: {checkpoint_path}")

    predictions = evaluate(model, test_loader, device)
    y_true_names = [class_names[int(label)] for label in predictions["y_true"]]
    y_pred_names = [class_names[int(label)] for label in predictions["y_pred"]]

    report = run_fairness_audit(
        y_true=y_true_names,
        y_pred=y_pred_names,
        sensitive_features=groups,
        probabilities=predictions["probabilities"],
        class_labels=class_names,
        metadata={
            "model": "ResNet-18",
            "architecture": resolved_architecture,
            "dataset": "CIFAR-10",
            "split": "test",
            "group_attributes": group_columns,
            "group_metadata_source": str(group_metadata_path),
            "group_metadata_alignment": "CSV row order matches CIFAR-10 test split order",
        },
        output_path=output_path,
    )

    group_plot_paths = {}
    roc_plot_paths = {}
    for index, attribute_name in enumerate(group_columns):
        group_plot_paths[attribute_name] = save_group_metric_panels(
            report,
            sensitive_attribute=attribute_name,
            output_stem=output_path.with_name(
                f"{output_path.stem}_group_{index + 1}_metrics"
            ),
        )
        roc_plot_paths[attribute_name] = save_group_roc_curve_comparisons(
            y_true=y_true_names,
            group_values=groups[attribute_name],
            probabilities=predictions["probabilities"],
            class_labels=class_names,
            output_stem=output_path.with_name(
                f"{output_path.stem}_group_{index + 1}_roc"
            ),
        )
    report["artifacts"] = {
        "group_metric_charts": {
            attribute_name: {
                extension: path.name for extension, path in paths.items()
            }
            for attribute_name, paths in group_plot_paths.items()
        },
    }
    report["artifacts"]["group_roc_charts"] = {
        attribute_name: {
            class_label: {
                extension: path.name for extension, path in paths.items()
            }
            for class_label, paths in classes.items()
        }
        for attribute_name, classes in roc_plot_paths.items()
    }
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    print(f"  overall accuracy: {report['overall_metrics']['accuracy']:.4f}")
    for attribute_name in group_columns:
        attribute = report["attributes"][attribute_name]
        print(f"\nResults by group for {attribute_name!r}:")
        for group in attribute["group_metrics"]:
            metrics = group["metrics"]
            print(
                f"  {group['group']}: n={group['sample_count']}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
            )
        accuracy_gap = attribute["across_group_gaps"]["accuracy"]["max_minus_min"]
        if accuracy_gap is not None:
            print(
                f"  accuracy range (max - min): {accuracy_gap:.4f} "
                f"({accuracy_gap * 100:.2f} percentage points)"
            )
    print(f"Detailed report: {output_path}")
    for attribute_name, paths in group_plot_paths.items():
        print(f"Group metrics chart ({attribute_name}): {paths['png']}")
        for class_label, class_paths in roc_plot_paths[attribute_name].items():
            print(
                f"Group ROC comparison ({attribute_name}, {class_label}): "
                f"{class_paths['png']}"
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a CIFAR-10 ResNet-18 checkpoint using externally supplied "
            "group metadata"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/resnet18_cifar10.pt"),
        help="Path to a ResNet-18 state-dict checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/fairness_cifar10.json"))
    parser.add_argument(
        "--group-metadata",
        type=Path,
        required=True,
        help=(
            "CSV with one row per CIFAR-10 test example in exactly the test "
            "loader's order and one or more group columns"
        ),
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        required=True,
        help="CSV columns to audit, for example --group-columns sex age_band",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("auto", "torchvision", "cifar"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cifar10_fairness_audit(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        group_metadata_path=args.group_metadata,
        group_columns=args.group_columns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        architecture=args.architecture,
    )
