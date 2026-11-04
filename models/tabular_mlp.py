"""A small fully connected classifier for preprocessed tabular features."""

from __future__ import annotations

import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim < 1:
                raise ValueError("hidden layer sizes must be positive")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)
