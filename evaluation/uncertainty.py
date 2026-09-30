from __future__ import annotations

import numpy as np


def calculate_confidence_statistics(
    y_true,
    y_pred,
    confidence,
) -> dict[str, float | int]:
    """
    Calculate basic confidence statistics.

    Compares confidence of correct and incorrect predictions.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    correct = y_true == y_pred
    incorrect = ~correct

    correct_count = int(np.sum(correct))
    incorrect_count = int(np.sum(incorrect))

    mean_confidence = float(
        np.mean(confidence)
    )

    if correct_count > 0:
        correct_confidence = float(
            np.mean(confidence[correct])
        )
    else:
        correct_confidence = 0.0

    if incorrect_count > 0:
        incorrect_confidence = float(
            np.mean(confidence[incorrect])
        )
    else:
        incorrect_confidence = 0.0

    return {
        "mean_confidence": mean_confidence,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "correct_mean_confidence": correct_confidence,
        "incorrect_mean_confidence": incorrect_confidence,
    }


def calculate_confidence_bins(
    y_true,
    y_pred,
    confidence,
    num_bins: int = 10,
):
    """
    Calculate accuracy and confidence
    for confidence intervals.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    results = []

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        bin_accuracy = float(
            np.mean(
                y_true[mask] == y_pred[mask]
            )
        )

        bin_confidence = float(
            np.mean(confidence[mask])
        )

        results.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
            }
        )

    return results