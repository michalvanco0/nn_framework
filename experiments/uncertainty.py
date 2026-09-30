"""Confidence and uncertainty analysis experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.evaluator import evaluate
from evaluation.uncertainty import (
    calculate_confidence_bins,
    calculate_confidence_statistics,
)
from models.resnet import create_resnet18


def run_uncertainty_analysis(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    batch_size: int = 128,
    num_workers: int = 2,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load test dataset
    _, test_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create model
    model = create_resnet18(
        num_classes=10
    ).to(device)

    # Load trained model
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

    # Calculate statistics
    statistics = calculate_confidence_statistics(
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        confidence=evaluation["confidence"],
    )

    bins = calculate_confidence_bins(
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
        confidence=evaluation["confidence"],
        num_bins=10,
    )

    # Print results
    print("\nConfidence statistics:")

    for key, value in statistics.items():
        print(f"{key}: {value:.4f}"
              if isinstance(value, float)
              else f"{key}: {value}")

    # Save JSON
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "statistics": statistics,
        "confidence_bins": bins,
    }

    result_path = (
        results_dir
        / "uncertainty.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    # Create confidence vs accuracy plot
    confidences = [
        item["confidence"]
        for item in bins
    ]

    accuracies = [
        item["accuracy"]
        for item in bins
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        confidences,
        accuracies,
        marker="o",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.set_xlabel(
        "Mean confidence"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Confidence vs. Accuracy"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    plot_path = (
        results_dir
        / "confidence_accuracy.png"
    )

    fig.savefig(plot_path)

    plt.close(fig)

    print(
        f"\nUncertainty results saved to: "
        f"{result_path}"
    )

    print(
        f"Confidence plot saved to: "
        f"{plot_path}"
    )

    return output


if __name__ == "__main__":
    run_uncertainty_analysis()