"""Selective prediction / abstention experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.evaluator import evaluate
from evaluation.selective import calculate_risk_coverage_curve
from models.resnet import create_resnet18


def run_selective_prediction(
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

    # Load trained checkpoint
    checkpoint_path = (
        results_dir / "resnet18_cifar10.pt"
    )

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

    print(
        f"Loaded model from: {checkpoint_path}"
    )

    # Evaluate model
    evaluation = evaluate(
        model,
        test_loader,
        device,
    )

    # Calculate risk-coverage curve
    results = calculate_risk_coverage_curve(
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        confidence=evaluation["confidence"],
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save numerical results
    result_path = (
        results_dir
        / "selective_prediction.json"
    )

    output = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "results": results,
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    # Prepare data for plot
    coverage = [
        item["coverage"]
        for item in results
    ]

    risk = [
        item["risk"]
        for item in results
    ]

    # Create risk-coverage curve
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        coverage,
        risk,
        marker="o",
    )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(
        "Risk-Coverage Curve"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True)

    fig.tight_layout()

    plot_path = (
        results_dir
        / "risk_coverage_curve.png"
    )

    fig.savefig(plot_path)

    plt.close(fig)

    print(
        f"Selective prediction results saved to: "
        f"{result_path}"
    )

    print(
        f"Risk-coverage curve saved to: "
        f"{plot_path}"
    )

    # Print selected thresholds
    print("\nSelected thresholds:")

    for item in results:
        threshold = item["threshold"]

        if threshold in {
            0.5,
            0.7,
            0.8,
            0.9,
        }:
            print(
                f"threshold={threshold:.2f} | "
                f"coverage={item['coverage']:.3f} | "
                f"risk={item['risk']:.3f}"
            )

    return output


if __name__ == "__main__":
    run_selective_prediction()