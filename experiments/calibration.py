from __future__ import annotations

import json
from pathlib import Path

import torch

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.calibration import (
    calculate_ece,
    save_reliability_diagram,
)
from evaluation.evaluator import evaluate
from models.resnet import create_resnet18


def run_calibration(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    batch_size: int = 128,
    num_workers: int = 2,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load test dataset
    _, test_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create model
    model = create_resnet18(num_classes=10).to(device)

    # Load already trained model
    checkpoint_path = results_dir / "resnet18_cifar10.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint)

    print(f"Loaded model from: {checkpoint_path}")

    # Evaluate model
    evaluation = evaluate(
        model,
        test_loader,
        device,
    )

    # Calculate ECE
    ece, bins = calculate_ece(
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        confidence=evaluation["confidence"],
        num_bins=10,
    )

    print(f"ECE: {ece:.4f}")

    # Save reliability diagram
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagram_path = (
        results_dir / "reliability_diagram.png"
    )

    save_reliability_diagram(
        bins=bins,
        output_path=diagram_path,
    )

    # Save numerical results
    result = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "ece": ece,
        "num_bins": 10,
        "bins": bins,
    }

    result_path = (
        results_dir / "calibration.json"
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

    return result


if __name__ == "__main__":
    run_calibration()