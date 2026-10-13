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

from datasets.cifar10 import get_cifar10_train_val_test_dataloaders
from evaluation.evaluator import evaluate as evaluate_predictions
from evaluation.performance import calculate_classification_metrics
from models.resnet import create_resnet18


@dataclass
class BaselineConfig:
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    num_workers: int = 2
    seed: int = 42
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
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


def run_baseline(
    config: BaselineConfig,
) -> dict[str, float | int | str | list[list[int]]]:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = (
        get_cifar10_train_val_test_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    )

    model = create_resnet18(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.results_dir / "resnet18_cifar10.pt"
    best_val_accuracy = -1.0
    best_epoch = 0

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_predictions = evaluate_predictions(model, val_loader, device)
        val_targets = val_predictions["y_true"]
        val_predictions_list = val_predictions["y_pred"]
        val_accuracy = sum(
            target == prediction
            for target, prediction in zip(val_targets, val_predictions_list)
        ) / len(val_targets)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- loss: {avg_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f} "
            f"- best_val_accuracy: {best_val_accuracy:.4f} "
            f"- lr: {current_lr:.6f}"
        )
        scheduler.step()

    if best_epoch == 0:
        raise RuntimeError("Training ended without producing a model checkpoint")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_predictions = evaluate_predictions(model, test_loader, device)
    metrics = calculate_classification_metrics(
        test_predictions["y_true"],
        test_predictions["y_pred"],
    )

    result = {
        "model": "ResNet18",
        "dataset": "CIFAR10",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    metrics_path = config.results_dir / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    save_confusion_matrix_plot(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=config.results_dir / "confusion_matrix.png",
    )

    print(
        f"Best validation accuracy: {best_val_accuracy:.4f} "
        f"(epoch {best_epoch})"
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    return result
