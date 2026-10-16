from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def calculate_ace(
    y_true,
    probabilities,
    num_bins: int = 10,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] != len(y_true):
        raise ValueError("y_true and probabilities must contain the same samples")
    if probabilities.shape[1] < 2:
        raise ValueError("ACE requires at least two classes")
    if num_bins < 1 or num_bins > len(y_true):
        raise ValueError("num_bins must be between 1 and the number of samples")

    num_classes = probabilities.shape[1]
    classwise_bins = []
    class_errors = []

    # K - class_index 
    # R - num_bins
    # acc(r,k) - bin_accuracy
    # conf(r,k) - bin_confidence
    for class_index in range(num_classes):
        class_probabilities = probabilities[:, class_index]
        sorted_indices = np.argsort(class_probabilities, kind="stable")
        adaptive_bins = np.array_split(sorted_indices, num_bins)
        bin_details = []
        bin_errors = []

        for indices in adaptive_bins:
            bin_probabilities = class_probabilities[indices]
            bin_accuracy = float(np.mean(y_true[indices] == class_index))
            bin_confidence = float(np.mean(bin_probabilities))
            bin_error = abs(bin_accuracy - bin_confidence)

            bin_errors.append(bin_error)
            bin_details.append(
                {
                    "lower": float(np.min(bin_probabilities)),
                    "upper": float(np.max(bin_probabilities)),
                    "count": int(len(indices)),
                    "accuracy": bin_accuracy,
                    "confidence": bin_confidence,
                    "calibration_error": float(bin_error),
                }
            )

        class_ace = float(np.mean(bin_errors))
        class_errors.append(class_ace)
        classwise_bins.append(
            {
                "class_index": class_index,
                "ace": class_ace,
                "bins": bin_details,
            }
        )

    return float(np.mean(class_errors)), classwise_bins


def save_classwise_reliability_diagram(
    classwise_bins,
    output_path: Path,
):
    if not classwise_bins:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    for class_result in classwise_bins:
        bins = class_result["bins"]
        ax.plot(
            [item["confidence"] for item in bins],
            [item["accuracy"] for item in bins],
            marker="o",
            label=(
                f"Class {class_result['class_index']} "
                f"(ACE={class_result['ace']:.3f})"
            ),
        )

    ax.set_xlabel("Mean class probability")
    ax.set_ylabel("Empirical class frequency")
    ax.set_title("Class-wise Adaptive Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(title="Class", fontsize="small")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_classwise_ace_bar_chart(
    classwise_bins,
    output_path: Path,
    class_names=None,
    max_classes: int = 20,
):
    if not classwise_bins:
        return
    if class_names is not None and len(class_names) != len(classwise_bins):
        raise ValueError("class_names must match the number of classes")
    if max_classes < 1:
        raise ValueError("max_classes must be at least 1")

    ranked_classes = sorted(
        classwise_bins,
        key=lambda item: item["ace"],
        reverse=True,
    )[:max_classes]
    class_indices = [item["class_index"] for item in ranked_classes]
    labels = (
        [class_names[class_index] for class_index in class_indices]
        if class_names is not None
        else [str(class_index) for class_index in class_indices]
    )
    ace_values = [item["ace"] for item in ranked_classes]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.35)))
    bars = ax.barh(labels[::-1], ace_values[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Adaptive calibration error (ACE)")
    ax.set_ylabel("Class")
    ax.set_title(
        f"Worst-calibrated Classes (top {len(ranked_classes)})"
    )
    ax.set_xlim(0, max(0.1, max(ace_values) * 1.15))
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
