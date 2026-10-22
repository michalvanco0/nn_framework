from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def _save_figure_formats(fig: plt.Figure, output_stem: Path) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        extension: output_stem.with_suffix(f".{extension}")
        for extension in ("png", "pdf", "svg")
    }
    for extension, path in outputs.items():
        options = {"dpi": 180} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", **options)
    plt.close(fig)
    return outputs


def save_group_metric_panels(
    report: dict[str, Any],
    sensitive_attribute: str,
    output_stem: Path,
) -> dict[str, Path]:
    """Plot group accuracy and probability-quality metrics from an audit report."""
    attribute = report["attributes"][sensitive_attribute]
    groups = attribute["group_metrics"]
    if not groups:
        raise ValueError(f"No groups found for sensitive attribute {sensitive_attribute!r}")

    metric_specs = [
        ("accuracy", "Accuracy", "Accuracy", (0.0, 1.0)),
        ("balanced_accuracy", "Balanced accuracy", "Balanced accuracy", (0.0, 1.0)),
    ]
    first_probability_metrics = groups[0]["probability_metrics"]
    if first_probability_metrics is not None:
        metric_specs.extend(
            [
                ("brier_score", "Brier score", "Brier score", None),
                ("log_loss", "Log loss", "Log loss", None),
                ("top_label_ece", "Top-label ECE", "Top-label ECE", (0.0, 1.0)),
            ]
        )

    rows = int(np.ceil(len(metric_specs) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.2 * rows), squeeze=False)
    labels = [f"{item['group']}\n(n={item['sample_count']})" for item in groups]
    colors = plt.get_cmap("tab10").colors

    for axis, (metric_key, title, ylabel, limits) in zip(axes.flat, metric_specs):
        if metric_key in {"accuracy", "balanced_accuracy"}:
            values = [item["metrics"][metric_key] for item in groups]
        else:
            values = [item["probability_metrics"][metric_key] for item in groups]

        positions = np.arange(len(groups))
        bars = axis.bar(
            positions,
            [np.nan if value is None else value for value in values],
            color=[colors[index % len(colors)] for index in range(len(groups))],
        )
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if limits is not None:
            axis.set_ylim(*limits)
        else:
            finite_values = [float(value) for value in values if value is not None]
            upper = max(finite_values, default=1.0) * 1.18
            axis.set_ylim(0.0, max(upper, 0.1))

        for bar, value in zip(bars, values):
            if value is not None:
                axis.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            else:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    for axis in axes.flat[len(metric_specs) :]:
        axis.set_visible(False)

    fig.suptitle(
        "Observed model metrics by supplied group",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure_formats(fig, Path(output_stem))


def save_binary_roc_curve(
    y_true: Any,
    positive_scores: Any,
    positive_label: Any,
    output_stem: Path,
    title: str,
) -> tuple[float, dict[str, Path]]:
    """Plot one binary ROC curve and return its AUC and saved file paths."""
    labels = np.asarray(y_true)
    scores = np.asarray(positive_scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("y_true and positive_scores must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("positive_scores must be finite")

    binary_truth = labels == positive_label
    if len(np.unique(binary_truth)) != 2:
        raise ValueError("A binary ROC curve requires positive and negative examples")

    false_positive_rate, true_positive_rate, _ = roc_curve(binary_truth, scores)
    roc_auc = float(auc(false_positive_rate, true_positive_rate))

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2.4,
        label=f"Model (AUC = {roc_auc:.3f})",
        color="#3274A1",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        linewidth=1.3,
        label="Chance",
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    paths = _save_figure_formats(fig, Path(output_stem))
    return roc_auc, paths
