from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, tuple[str, ...]]:
    """Return the original train/test loader pair used by evaluation tools."""
    transform = _evaluation_transform()

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, tuple(train_dataset.classes)


def get_cifar10_train_val_test_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Return augmented training, clean validation, and clean test loaders."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = _evaluation_transform()

    train_aug_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )

    train_eval_dataset = copy.copy(train_aug_dataset)
    train_eval_dataset.transform = eval_transform

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_aug_dataset), generator=generator).tolist()
    val_indices = indices[:5_000]
    train_indices = indices[5_000:]

    train_dataset = Subset(train_aug_dataset, train_indices)
    val_dataset = Subset(train_eval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tuple(train_aug_dataset.classes)
