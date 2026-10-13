from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datasets.cifar10 import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_dataloaders,
)
from evaluation.evaluator import evaluate
from evaluation.explainability import GradCAM
from models.resnet import create_resnet18


DATA_DIR = Path("data")
MODEL_PATH = Path("results/resnet18_cifar10.pt")
OUTPUT_DIR = Path("results/explainability")

BATCH_SIZE = 128
NUM_IMAGES = 5

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def denormalize(image):
    """Convert a normalized CIFAR-10 image back to [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    image = image.cpu() * std + mean
    return torch.clamp(image, 0, 1)


def create_visualization(
    image,
    heatmap,
    true_class,
    predicted_class,
    confidence,
    output_path,
):
    
    image = denormalize(image)

    image_np = image.permute(1, 2, 0).numpy()

    heatmap_tensor = torch.from_numpy(
        np.asarray(heatmap, dtype=np.float32)
    )

    if heatmap_tensor.ndim != 2:
        raise ValueError(
            f"Expected 2D heatmap, got shape {heatmap_tensor.shape}"
        )

    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)

    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )

    heatmap_resized = heatmap_resized[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(image_np)
    axes[0].set_title(
        f"Original\nTrue: {true_class}"
    )
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.5,
    )
    axes[2].set_title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )
    axes[2].axis("off")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    _, test_loader, _ = get_cifar10_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    model = create_resnet18(num_classes=10)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")

    grad_cam = GradCAM(
        model=model,
        target_layer="layer4",
    )

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(min(NUM_IMAGES, len(images))):

        input_image = images[i:i + 1]

        predicted_class = predictions[i].item()
        true_class = labels[i].item()
        confidence = confidences[i].item()

        heatmap = grad_cam.generate_heatmap(
            input_image=input_image,
            class_index=predicted_class,
        )

        output_path = OUTPUT_DIR / f"example_{i}.png"

        create_visualization(
            image=images[i],
            heatmap=heatmap,
            true_class=CLASS_NAMES[true_class],
            predicted_class=CLASS_NAMES[predicted_class],
            confidence=confidence,
            output_path=output_path,
        )

        print(
            f"Saved {output_path} | "
            f"true={CLASS_NAMES[true_class]}, "
            f"predicted={CLASS_NAMES[predicted_class]}, "
            f"confidence={confidence:.4f}"
        )


if __name__ == "__main__":
    main()