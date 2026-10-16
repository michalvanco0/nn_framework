from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18
from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.calibration import (
    calculate_ace,
    save_classwise_ace_bar_chart,
    save_classwise_reliability_diagram,
)
from evaluation.evaluator import evaluate
from models.resnet import create_resnet18


def _num_calibration_bins(num_samples: int) -> int:
    # Sturges' rule
    if num_samples < 1:
        raise ValueError("Calibration requires at least one sample")
    return max(1, math.ceil(math.log2(num_samples) + 1))


def run_calibration(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    batch_size: int = 128,
    num_workers: int = 2,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    architecture: str = "auto",
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

    num_bins = _num_calibration_bins(len(evaluation["y_true"]))
    ace, classwise_bins = calculate_ace(
        y_true=evaluation["y_true"],
        probabilities=evaluation["probabilities"],
        num_bins=num_bins,
    )

    print(f"ACE: {ace:.4f}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagram_path = (
        output_dir / "reliability_diagram.pdf"
    )
    ace_chart_path = (
        output_dir / "classwise_ace.pdf"
    )

    save_classwise_reliability_diagram(
        classwise_bins=classwise_bins,
        output_path=diagram_path,
    )
    save_classwise_ace_bar_chart(
        classwise_bins=classwise_bins,
        output_path=ace_chart_path,
        class_names=class_names,
    )

    result = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "metric": "ACE",
        "ace": ace,
        "num_bins": num_bins,
        "num_bins_rule": "Sturges' rule: ceil(log2(number of samples) + 1)",
        "binning": "class-wise adaptive equal-count bins",
        "reliability_diagram": diagram_path.name,
        "classwise_ace_chart": ace_chart_path.name,
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
    print(
        f"Class-wise ACE chart saved to: {ace_chart_path}"
    )

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
        help=(
            "Directory for calibration.json, reliability_diagram.pdf, "
            "and classwise_ace.pdf"
        ),
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
    )
