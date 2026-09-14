"""Model evaluation helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluate classification models on dataloaders."""

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> tuple[list[int], list[int]]:
        """Run model inference and return true/predicted labels."""
        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                y_true.extend(targets.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        return y_true, y_pred
