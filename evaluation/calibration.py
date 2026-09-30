from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calculate_ece(
    y_true,
    y_pred,
    confidence,
    num_bins: int = 10,
):
    """
    Calculate Expected Calibration Error (ECE).

    Predictions are divided into confidence bins.
    ECE measures the weighted difference between
    accuracy and confidence in each bin.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bins = []

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if not np.any(mask):
            continue

        bin_confidence = confidence[mask].mean()
        bin_accuracy = np.mean(
            y_true[mask] == y_pred[mask]
        )

        bin_size = np.sum(mask)

        ece += (
            bin_size / len(confidence)
            * abs(bin_accuracy - bin_confidence)
        )

        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(bin_size),
                "accuracy": float(bin_accuracy),
                "confidence": float(bin_confidence),
            }
        )

    return float(ece), bins


def save_reliability_diagram(
    bins,
    output_path: Path,
):
    """
    Save a reliability diagram.
    """

    if not bins:
        return

    accuracies = [
        item["accuracy"]
        for item in bins
    ]

    confidences = [
        item["confidence"]
        for item in bins
    ]

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        confidences,
        accuracies,
        marker="o",
        label="Model",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(output_path)
    plt.close()