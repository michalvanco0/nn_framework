"""Baseline training and evaluation experiment."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.evaluator import Evaluator
from evaluation.performance import calculate_classification_metrics
from models.resnet import create_resnet18


@dataclass
class BaselineConfig:
    """Configurable hyperparameters for the baseline experiment."""

    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_workers: int = 2
    seed: int = 42
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def save_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Save a confusion matrix image for baseline diagnostics."""
    cm = np.array(confusion_matrix)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_baseline(config: BaselineConfig) -> dict[str, float | str | list[list[int]]]:
    """Run baseline train/evaluate pipeline and persist outputs."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, class_names = get_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = create_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f}")

    evaluator = Evaluator()
    y_true, y_pred = evaluator.evaluate(model, test_loader, device)
    metrics = calculate_classification_metrics(y_true, y_pred)

    result = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    config.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.results_dir / "resnet18_cifar10.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = config.results_dir / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    save_confusion_matrix_plot(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=config.results_dir / "confusion_matrix.png",
    )

    return result
