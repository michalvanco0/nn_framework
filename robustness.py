from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.resnet import create_resnet18
from datasets.cifar10 import get_cifar10_dataloaders


def add_gaussian_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    noisy_images = images + noise

    return torch.clamp(noisy_images, -3.0, 3.0)


def add_brightness(images, factor=1.2):
    return images * factor


def add_contrast(images, factor=1.2):
    mean = images.mean(dim=(2, 3), keepdim=True)

    return (images - mean) * factor + mean


def add_blur(images, kernel_size=5):
    return F.avg_pool2d(
        images,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


def evaluate_model(model, dataloader, device, transform=None):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            if transform is not None:
                images = transform(images)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy


def run_robustness_test(
    model_path="results/resnet18_cifar10.pt",
    batch_size=128,
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = create_resnet18(num_classes=10)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)

    _, test_loader, _ = get_cifar10_dataloaders(
        data_dir="data",
        batch_size=batch_size
    )

    clean_accuracy = evaluate_model(
        model,
        test_loader,
        device,
    )

    noise_accuracy = evaluate_model(
        model,
        test_loader,
        device,
        transform=lambda x: add_gaussian_noise(x, std=0.1),
    )

    blur_accuracy = evaluate_model(
        model,
        test_loader,
        device,
        transform=lambda x: add_blur(x, kernel_size=5),
    )

    brightness_accuracy = evaluate_model(
        model,
        test_loader,
        device,
        transform=lambda x: add_brightness(x, factor=1.2),
    )

    contrast_accuracy = evaluate_model(
        model,
        test_loader,
        device,
        transform=lambda x: add_contrast(x, factor=1.2),
    )

    results = {
        "clean": clean_accuracy,
        "gaussian_noise": noise_accuracy,
        "blur": blur_accuracy,
        "brightness": brightness_accuracy,
        "contrast": contrast_accuracy,
    }

    print("\nRobustness results:")

    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

    return results


if __name__ == "__main__":
    run_robustness_test()