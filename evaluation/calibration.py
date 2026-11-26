from math import ceil, log2
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np


ECE_NUM_BINS = 10


def calibration_bin_count(num_samples: int) -> int:
    """Return the adaptive per-class bin count used for ACE (Sturges' rule)."""
    if num_samples < 1:
        raise ValueError("Calibration requires at least one sample")
    return min(num_samples, max(1, ceil(log2(num_samples) + 1)))


def calculate_ece(
    y_true,
    y_pred,
    confidence,
    num_bins: int = 10,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or confidence.ndim != 1:
        raise ValueError("y_true, y_pred and confidence must be one-dimensional")
    if not (len(y_true) == len(y_pred) == len(confidence)):
        raise ValueError("Inputs must contain the same number of samples")
    if len(y_true) == 0:
        raise ValueError("ECE requires at least one sample")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    if not np.all(np.isfinite(confidence)) or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("confidence values must be finite and between 0 and 1")

    bin_indices = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    correct = y_true == y_pred
    bin_details = []
    ece = 0.0

    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        if count == 0:
            bin_details.append(
                {
                    "lower": bin_index / num_bins,
                    "upper": (bin_index + 1) / num_bins,
                    "count": 0,
                    "accuracy": None,
                    "confidence": None,
                    "calibration_error": None,
                }
            )
            continue
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidence[mask]))
        bin_error = abs(bin_accuracy - bin_confidence)
        ece += (count / len(y_true)) * bin_error
        bin_details.append(
            {
                "lower": bin_index / num_bins,
                "upper": (bin_index + 1) / num_bins,
                "count": count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "calibration_error": float(bin_error),
            }
        )

    return float(ece), bin_details


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


def calculate_calibration_metrics(
    y_true,
    y_pred,
    confidence,
    probabilities,
    *,
    ece_num_bins: int = ECE_NUM_BINS,
    ace_num_bins: int | None = None,
) -> dict:
    y_true_array = np.asarray(y_true)
    if ace_num_bins is None:
        ace_num_bins = calibration_bin_count(len(y_true_array))
    ece, ece_bins = calculate_ece(
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        num_bins=ece_num_bins,
    )
    ace, classwise_bins = calculate_ace(
        y_true=y_true,
        probabilities=probabilities,
        num_bins=ace_num_bins,
    )
    return {
        "top_label_ece": ece,
        "classwise_ace": ace,
        "top_label_ece_num_bins": ece_num_bins,
        "classwise_ace_num_bins": ace_num_bins,
        "top_label_ece_binning": "fixed-width confidence bins",
        "classwise_ace_binning": "adaptive equal-count bins per class",
        "classwise_ace_bin_count_rule": "Sturges' rule: ceil(log2(number of samples) + 1)",
        "ece_bins": ece_bins,
        "classwise_bins": classwise_bins,
    }


def save_classwise_reliability_diagram(
    classwise_bins,
    output_path: Path,
    class_names=None,
):
    if not classwise_bins:
        return
    if class_names is not None and len(class_names) != len(classwise_bins):
        raise ValueError("class_names must match the number of classes")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    colors = plt.get_cmap("tab10")
    for class_result in classwise_bins:
        bins = class_result["bins"]
        class_index = class_result["class_index"]
        color = colors(class_index % 10)
        class_label = (
            class_names[class_index]
            if class_names is not None
            else f"Class {class_index}"
        )
        ax.plot(
            [item["confidence"] for item in bins],
            [item["accuracy"] for item in bins],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=f"{class_label} (ACE={class_result['ace']:.3f})",
        )

    ax.set_xlabel("Mean class probability")
    ax.set_ylabel("Empirical class frequency")
    ax.set_title("Class-wise Adaptive Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(
        title="Class",
        fontsize="small",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    zooms = (
        {
            "bounds": [0.08, 0.57, 0.32, 0.32],
            "xlim": (0.0, 0.05),
            "ylim": (0.0, 0.05),
            "title": "Lower-left detail",
        },
        {
            "bounds": [0.58, 0.08, 0.32, 0.32],
            "xlim": (0.95, 1.0),
            "ylim": (0.95, 1.0),
            "title": "Upper-right detail",
        },
    )
    for zoom in zooms:
        inset = ax.inset_axes(zoom["bounds"])
        inset.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.9)
        for class_result in classwise_bins:
            bins = class_result["bins"]
            inset.plot(
                [item["confidence"] for item in bins],
                [item["accuracy"] for item in bins],
                marker="o",
                markersize=2.5,
                linewidth=0.9,
                color=colors(class_result["class_index"] % 10),
            )
        inset.set_xlim(*zoom["xlim"])
        inset.set_ylim(*zoom["ylim"])
        inset.text(
            0.04,
            0.96,
            zoom["title"],
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1},
        )
        inset.tick_params(labelsize=7)
        inset.grid(True, linewidth=0.5, alpha=0.5)
        xmin, xmax = zoom["xlim"]
        ymin, ymax = zoom["ylim"]
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="0.25",
                linewidth=0.9,
                linestyle=":",
            )
        )

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _save_bin_gap_diagram(
    bins,
    metric_value: float,
    metric_label: str,
    output_path: Path,
    title: str,
    adaptive: bool,
    class_name: str | None = None,
):
    if not bins:
        return

    fig, ax = plt.subplots(figsize=(10, 6) if adaptive else (7.5, 6))
    bins = list(bins)
    non_empty_bins = [item for item in bins if item["count"] > 0]
    if not non_empty_bins:
        plt.close(fig)
        return

    if adaptive:
        accuracies = np.asarray([item["accuracy"] for item in non_empty_bins])
        confidences = np.asarray([item["confidence"] for item in non_empty_bins])
        positions = np.arange(len(non_empty_bins))
        width = 0.76
        tick_labels = [
            f"{index + 1}\n({confidence:.1e})"
            for index, confidence in enumerate(confidences)
        ]
        ax.set_xticks(positions, tick_labels, fontsize=8)
        ax.set_xlabel("Adaptive bin index (mean class probability in parentheses)")
        ax.set_xlim(-0.6, len(non_empty_bins) - 0.4)
        ax.scatter(
            positions,
            confidences,
            marker="_",
            s=100,
            color="black",
            linewidths=1.5,
            label="Mean class probability",
            zorder=4,
        )
    else:
        bin_lowers = np.asarray([item["lower"] for item in bins])
        bin_uppers = np.asarray([item["upper"] for item in bins])
        bin_widths = bin_uppers - bin_lowers
        bin_centers = (bin_lowers + bin_uppers) / 2
        positions = bin_centers
        width = bin_widths * 0.9
        ax.set_xlabel("Top-label confidence bin")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        for index, item in enumerate(bins):
            if item["count"] == 0:
                ax.bar(
                    bin_centers[index],
                    1.0,
                    width=width[index],
                    color="#F0F0F0",
                    edgecolor="#888888",
                    linewidth=0.7,
                    hatch="///",
                    zorder=0,
                )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="0.35",
            linewidth=1.0,
            label="Perfect calibration",
            zorder=1,
        )

    if adaptive:
        accuracies = np.asarray([item["accuracy"] for item in non_empty_bins])
        confidences = np.asarray([item["confidence"] for item in non_empty_bins])
        plot_bins = non_empty_bins
        plot_positions = positions
        plot_widths = np.full(len(plot_bins), width)
    else:
        plot_bins = [item for item in bins if item["count"] > 0]
        plot_positions = np.asarray(
            [(item["lower"] + item["upper"]) / 2 for item in plot_bins]
        )
        plot_widths = np.asarray(
            [(item["upper"] - item["lower"]) * 0.9 for item in plot_bins]
        )
        accuracies = np.asarray([item["accuracy"] for item in plot_bins])
        confidences = np.asarray([item["confidence"] for item in plot_bins])

    ax.bar(
        plot_positions,
        accuracies,
        width=plot_widths,
        color="#2457D6",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for position, bar_width, accuracy, confidence in zip(
        plot_positions, plot_widths, accuracies, confidences
    ):
        ax.bar(
            position,
            abs(accuracy - confidence),
            bottom=min(accuracy, confidence),
            width=bar_width,
            color="#F3A0A0",
            edgecolor="#D62728",
            linewidth=0.7,
            hatch="///",
            zorder=3,
        )

    metric_text = (
        f"{metric_label} ({class_name}) = {metric_value:.4f}"
        if class_name is not None
        else f"{metric_label} = {metric_value:.4f}"
    )
    ax.text(
        0.02,
        0.97,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.4", "alpha": 0.95},
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(
        "Empirical class frequency" if adaptive else "Top-label accuracy"
    )
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)
    if adaptive:
        legend_handles = [
            Line2D(
                [], [], color="black", marker="_", linestyle="None",
                markersize=12, label="Mean class probability"
            ),
            Patch(facecolor="#2457D6", edgecolor="black", label="Bin accuracy"),
            Patch(
                facecolor="#F3A0A0", edgecolor="#D62728", hatch="///",
                label="Calibration gap"
            ),
        ]
    else:
        legend_handles = [
            Line2D(
                [], [], color="0.35", linestyle="--",
                label="Perfect calibration"
            ),
            Patch(facecolor="#2457D6", edgecolor="black", label="Bin accuracy"),
            Patch(
                facecolor="#F3A0A0", edgecolor="#D62728", hatch="///",
                label="Calibration gap"
            ),
            Patch(
                facecolor="#F0F0F0", edgecolor="#888888", hatch="///",
                label="No samples"
            ),
        ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.88),
        fontsize="small",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_ece_reliability_diagram(
    bins,
    ece: float,
    output_path: Path,
):
    _save_bin_gap_diagram(
        bins=bins,
        metric_value=ece,
        metric_label="ECE",
        output_path=output_path,
        title="Top-label ECE Reliability Diagram",
        adaptive=False,
    )


def save_classwise_ace_reliability_diagram(
    class_result,
    class_name: str,
    output_path: Path,
):
    _save_bin_gap_diagram(
        bins=class_result["bins"],
        metric_value=class_result["ace"],
        metric_label="ACE",
        output_path=output_path,
        title=f"Class-wise ACE Reliability Diagram — {class_name} class",
        adaptive=True,
        class_name=class_name,
    )


def save_calibration_plots(
    calibration: dict,
    class_names: list[str] | tuple[str, ...],
    output_dir: Path,
    *,
    comparison_class_index: int | None = None,
) -> dict[str, str]:
    classwise_bins = calibration["classwise_bins"]
    if not classwise_bins:
        raise ValueError("Calibration plots require per-class ACE bins")
    if len(class_names) != len(classwise_bins):
        raise ValueError("class_names must match the number of probability classes")

    if comparison_class_index is None:
        comparison_class_index = max(classwise_bins, key=lambda item: item["ace"])[
            "class_index"
        ]
    if not 0 <= comparison_class_index < len(class_names):
        raise ValueError("comparison_class_index is outside class_names")

    output_dir = Path(output_dir)
    classwise_path = output_dir / "classwise_reliability_diagram.pdf"
    ece_path = output_dir / "ece_reliability_diagram.pdf"
    ace_path = output_dir / "comparison_class_ace_reliability_diagram.pdf"
    class_result = classwise_bins[comparison_class_index]

    save_classwise_reliability_diagram(
        classwise_bins=classwise_bins,
        output_path=classwise_path,
        class_names=class_names,
    )
    save_ece_reliability_diagram(
        bins=calibration["ece_bins"],
        ece=calibration["top_label_ece"],
        output_path=ece_path,
    )
    save_classwise_ace_reliability_diagram(
        class_result=class_result,
        class_name=class_names[comparison_class_index],
        output_path=ace_path,
    )
    return {
        "classwise_reliability_diagram": classwise_path.name,
        "ece_reliability_diagram": ece_path.name,
        "comparison_class_ace_reliability_diagram": ace_path.name,
        "comparison_class": class_names[comparison_class_index],
    }
