from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18
from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.calibration import (
    calculate_ace,
    calculate_ece,
    save_classwise_ace_reliability_diagram,
    save_classwise_reliability_diagram,
    save_ece_reliability_diagram,
)
from evaluation.evaluator import evaluate
from models.resnet import create_resnet18


def _num_calibration_bins(num_samples: int) -> int:
    # Sturges' rule
    if num_samples < 1:
        raise ValueError("Calibration requires at least one sample")
    return max(1, math.ceil(math.log2(num_samples) + 1))


def _resolve_class_index(class_name: str, class_names) -> int:
    requested = str(class_name).strip().casefold()
    matches = [
        index
        for index, available_name in enumerate(class_names)
        if available_name.casefold() == requested
    ]
    if matches:
        return matches[0]
    try:
        class_index = int(requested)
    except ValueError as error:
        raise ValueError(
            f"Unknown comparison class {class_name!r}. Choose one of: "
            f"{', '.join(class_names)}"
        ) from error
    if not 0 <= class_index < len(class_names):
        raise ValueError(
            f"Comparison class index must be between 0 and {len(class_names) - 1}"
        )
    return class_index


def run_calibration(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    batch_size: int = 128,
    num_workers: int = 2,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    architecture: str = "auto",
    comparison_class: str = "cat",
    ece_num_bins: int = 10,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    _, test_loader, class_names = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if checkpoint_path is None:
        checkpoint_path = results_dir / "resnet18_cifar10.pt"
    if output_dir is None:
        output_dir = results_dir

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if architecture == "auto":
        conv1_weight = checkpoint.get("conv1.weight")
        if conv1_weight is None or conv1_weight.ndim != 4:
            raise ValueError(
                "Cannot infer ResNet-18 architecture from checkpoint: "
                "expected a state dict containing conv1.weight."
            )
        kernel_size = conv1_weight.shape[-1]
        if kernel_size == 7:
            architecture = "torchvision"
        elif kernel_size == 3:
            architecture = "cifar"
        else:
            raise ValueError(
                f"Unsupported ResNet-18 conv1 kernel size: {kernel_size}"
            )

    if architecture == "torchvision":
        model = torchvision_resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        print("Using standard torchvision ResNet-18 architecture (7x7 stem).")
    elif architecture == "cifar":
        model = create_resnet18(num_classes=len(class_names))
        print("Using CIFAR ResNet-18 architecture (3x3 stem, no maxpool).")
    else:
        raise ValueError(
            "architecture must be one of: 'auto', 'torchvision', 'cifar'"
        )

    model = model.to(device)
    model.load_state_dict(checkpoint)

    print(f"Loaded model from: {checkpoint_path}")

    evaluation = evaluate(
        model,
        test_loader,
        device,
    )

    top_label_ece, top_label_ece_bins = calculate_ece(
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        confidence=evaluation["confidence"],
        num_bins=ece_num_bins,
    )
    num_bins = _num_calibration_bins(len(evaluation["y_true"]))
    ace, classwise_bins = calculate_ace(
        y_true=evaluation["y_true"],
        probabilities=evaluation["probabilities"],
        num_bins=num_bins,
    )

    comparison_class_index = _resolve_class_index(
        comparison_class,
        class_names,
    )
    comparison_class_name = class_names[comparison_class_index]
    comparison_class_result = classwise_bins[comparison_class_index]
    comparison_class_ace = comparison_class_result["ace"]

    print(f"Top-label ECE ({ece_num_bins} fixed-width bins): {top_label_ece:.4f}")
    print(f"ACE: {ace:.4f}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagram_path = (
        output_dir / "reliability_diagram.pdf"
    )
    ece_diagram_path = output_dir / "ece_reliability_diagram.pdf"
    classwise_ace_diagram_path = (
        output_dir / "classwise_ace_reliability_diagram.pdf"
    )

    save_classwise_reliability_diagram(
        classwise_bins=classwise_bins,
        output_path=diagram_path,
        class_names=class_names,
    )
    save_ece_reliability_diagram(
        bins=top_label_ece_bins,
        ece=top_label_ece,
        output_path=ece_diagram_path,
    )
    save_classwise_ace_reliability_diagram(
        class_result=comparison_class_result,
        class_name=comparison_class_name,
        output_path=classwise_ace_diagram_path,
    )

    result = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "class_names": list(class_names),
        "metric": "ACE",
        "ace": ace,
        "top_label_ece": top_label_ece,
        "top_label_ece_num_bins": ece_num_bins,
        "top_label_ece_binning": "fixed-width confidence bins",
        "top_label_ece_bins": top_label_ece_bins,
        "comparison_class": {
            "name": comparison_class_name,
            "index": comparison_class_index,
            "adaptive_classwise_ace": comparison_class_ace,
        },
        "num_bins": num_bins,
        "num_bins_rule": "Sturges' rule: ceil(log2(number of samples) + 1)",
        "binning": "class-wise adaptive equal-count bins",
        "reliability_diagram": diagram_path.name,
        "top_label_ece_reliability_diagram": ece_diagram_path.name,
        "comparison_class_ace_reliability_diagram": classwise_ace_diagram_path.name,
        "classwise_bins": classwise_bins,
    }

    result_path = (
        output_dir / "calibration.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"Calibration results saved to: {result_path}"
    )

    print(
        f"Reliability diagram saved to: {diagram_path}"
    )
    print(f"Top-label ECE diagram saved to: {ece_diagram_path}")
    print(f"Class-wise ACE diagram saved to: {classwise_ace_diagram_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 ResNet-18 calibration"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/resnet18_cifar10.pt"),
        help="Path to a ResNet-18 state-dict checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for calibration.json and calibration diagrams",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="CIFAR-10 dataset directory",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("auto", "torchvision", "cifar"),
        default="auto",
        help="ResNet-18 variant; auto selects from conv1 kernel size",
    )
    parser.add_argument(
        "--comparison-class",
        type=str,
        default="cat",
        help="CIFAR-10 class name or index for the class-wise ACE diagram",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of fixed-width bins used for ECE (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration(
        data_dir=args.data_dir,
        results_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        architecture=args.architecture,
        comparison_class=args.comparison_class,
        ece_num_bins=args.ece_bins,
    )
