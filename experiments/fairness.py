from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from evaluation.fairness import calculate_fairness_metrics


def run_fairness_audit(
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any,
    *,
    probabilities: Any | None = None,
    class_labels: Any | None = None,
    positive_label: Any | None = None,
    reference_groups: Mapping[str, Any] | None = None,
    accepted_mask: Any | None = None,
    num_bins: int = 10,
    small_group_warning_threshold: int | None = 30,
    metadata: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report = calculate_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
        reference_groups=reference_groups,
        accepted_mask=accepted_mask,
        num_bins=num_bins,
        small_group_warning_threshold=small_group_warning_threshold,
        metadata=metadata,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)

    return report
