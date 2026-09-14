"""ResNet model definitions for baseline experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes: int = 10) -> nn.Module:
    """Create a ResNet-18 model configured for the target number of classes."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
