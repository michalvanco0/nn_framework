from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.evaluator import evaluate
from evaluation.fairness_plots import (
    save_binary_roc_curve,
    save_group_metric_panels,
)
from experiments.fairness import run_fairness_audit
from models.resnet import create_resnet18


VEHICLE_CLASSES = {"airplane", "automobile", "ship", "truck"}
ANIMAL_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}


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


def run_cifar10_fairness_audit(
    checkpoint_path: Path = Path("results/resnet18_cifar10.pt"),
    data_dir: Path = Path("data"),
    output_path: Path = Path("results/fairness_cifar10_superclasses.json"),
    batch_size: int = 128,
    num_workers: int = 0,
    architecture: str = "auto",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}")

    _, test_loader, class_names = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
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
    class_to_superclass = {
        **{name: "animals" for name in ANIMAL_CLASSES},
        **{name: "vehicles" for name in VEHICLE_CLASSES},
    }
    missing_classes = set(class_names) - set(class_to_superclass)
    if missing_classes:
        raise ValueError(
            "The CIFAR-10 superclass mapping is missing classes: "
            + ", ".join(sorted(missing_classes))
        )
    y_true_names = [class_names[int(label)] for label in predictions["y_true"]]
    y_pred_names = [class_names[int(label)] for label in predictions["y_pred"]]
    groups = [class_to_superclass[label] for label in y_true_names]

    report = run_fairness_audit(
        y_true=y_true_names,
        y_pred=y_pred_names,
        sensitive_features={"target_superclass": groups},
        probabilities=predictions["probabilities"],
        class_labels=class_names,
        metadata={
            "model": "ResNet-18",
            "architecture": resolved_architecture,
            "dataset": "CIFAR-10",
            "split": "test",
            "grouping": "animals vs vehicles, derived from the ground-truth class",
            "grouping_is_protected_demographic_attribute": False,
            "interpretation": (
                "Exploratory class-superclass performance profile only; CIFAR-10 "
                "contains no demographic group attributes."
            ),
        },
        output_path=output_path,
    )

    group_plot_paths = save_group_metric_panels(
        report,
        sensitive_attribute="target_superclass",
        output_stem=output_path.with_name(f"{output_path.stem}_group_metrics"),
    )
    vehicle_class_indices = [
        index for index, name in enumerate(class_names) if name in VEHICLE_CLASSES
    ]
    vehicle_scores = np.asarray(predictions["probabilities"], dtype=float)[
        :, vehicle_class_indices
    ].sum(axis=1)
    superclass_roc_auc, roc_plot_paths = save_binary_roc_curve(
        y_true=groups,
        positive_scores=vehicle_scores,
        positive_label="vehicles",
        output_stem=output_path.with_name(f"{output_path.stem}_superclass_roc"),
        title="ROC curve: vehicle vs animal superclasses",
    )
    report["artifacts"] = {
        "group_metric_chart": {
            extension: path.name for extension, path in group_plot_paths.items()
        },
        "superclass_roc_curve": {
            extension: path.name for extension, path in roc_plot_paths.items()
        },
        "superclass_roc_auc": superclass_roc_auc,
    }
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    attribute = report["attributes"]["target_superclass"]
    print("\nResults by group (CIFAR-10 test set):")
    print(f"  overall accuracy: {report['overall_metrics']['accuracy']:.4f}")
    for group in attribute["group_metrics"]:
        metrics = group["metrics"]
        print(
            f"  {group['group']}: n={group['sample_count']}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )

    accuracy_gap = attribute["across_group_gaps"]["accuracy"]["max_minus_min"]
    accuracy_gap_text = (
        f"{accuracy_gap:.4f} ({accuracy_gap * 100:.2f} percentage points)"
        if accuracy_gap is not None
        else "undefined"
    )
    print(f"Accuracy gap (max - min): {accuracy_gap_text}")
    print(f"Superclass ROC AUC (vehicles vs animals): {superclass_roc_auc:.4f}")
    print(f"Detailed report: {output_path}")
    print(f"Group metrics chart: {group_plot_paths['png']}")
    print(f"Superclass ROC chart: {roc_plot_paths['png']}")
    print(
        "Note: groups are derived from the target class; this is not a "
        "demographic fairness measurement."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a class-superclass audit on the CIFAR-10 ResNet-18 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/resnet18_cifar10.pt"),
        help="Path to a ResNet-18 state-dict checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fairness_cifar10_superclasses.json"),
        help="Where to save the JSON report",
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        architecture=args.architecture,
    )
