from __future__ import annotations
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets.adult_income import load_adult_income
from datasets.tabular import clean_features, load_tabular_csv, normalize_target
from evaluation.calibration import calculate_ace, calculate_ece
from evaluation.evaluator import evaluate
from evaluation.fairness_plots import (
    save_group_metric_panels,
    save_group_roc_curve_comparisons,
)
from evaluation.performance import calculate_classification_metrics
from evaluation.uncertainty import calculate_confidence_statistics
from experiments.fairness import (
    build_feature_audit_groups,
    run_fairness_audit,
    write_disparity_ranking,
)
from models.tabular_mlp import TabularMLP


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No model features remain after excluding the target and groups")
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    return preprocessor, numeric_columns, categorical_columns


def _to_float32(matrix: Any) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float32)
    if result.ndim != 2 or not np.all(np.isfinite(result.data)):
        raise ValueError("Preprocessing produced non-finite or invalid model features")
    return result


def _split_indices(
    labels: np.ndarray,
    *,
    seed: int,
    official_train_count: int | None = None,
    split_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels))
    if split_values is not None:
        aliases = {
            "train": "train",
            "training": "train",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
        }
        normalized_splits = np.asarray(
            [aliases.get(str(value).strip().lower(), "") for value in split_values]
        )
        unknown_values = sorted(set(normalized_splits) - {"train", "validation", "test"})
        if unknown_values:
            raise ValueError(
                "Split column values must be train, validation/val, or test; "
                f"unrecognized values: {unknown_values}"
            )
        train_pool = all_indices[normalized_splits == "train"]
        test_indices = all_indices[normalized_splits == "test"]
        val_indices = all_indices[normalized_splits == "validation"]
        if len(train_pool) == 0 or len(test_indices) == 0:
            raise ValueError("A split column must contain at least one train and one test row")
        if len(val_indices) == 0:
            train_indices, val_indices = train_test_split(
                train_pool,
                test_size=0.15,
                random_state=seed,
                stratify=labels[train_pool],
            )
        else:
            train_indices = train_pool
        return (
            np.asarray(train_indices),
            np.asarray(val_indices),
            np.asarray(test_indices),
        )
    if official_train_count is not None:
        train_pool = all_indices[:official_train_count]
        test_indices = all_indices[official_train_count:]
        train_indices, val_indices = train_test_split(
            train_pool,
            test_size=0.15,
            random_state=seed,
            stratify=labels[train_pool],
        )
    else:
        train_indices, held_out = train_test_split(
            all_indices,
            test_size=0.30,
            random_state=seed,
            stratify=labels,
        )
        val_indices, test_indices = train_test_split(
            held_out,
            test_size=0.50,
            random_state=seed,
            stratify=labels[held_out],
        )
    return (
        np.asarray(train_indices),
        np.asarray(val_indices),
        np.asarray(test_indices),
    )


class _TabularDataset(Dataset):
    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, int(self.labels[index])


class _SparseBatchCollator:
    def __init__(self, features: sparse.csr_matrix) -> None:
        self.features = features

    def __call__(self, samples: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        indices, labels = zip(*samples)
        dense_batch = self.features[list(indices)].toarray().astype(np.float32, copy=False)
        return (
            torch.from_numpy(dense_batch),
            torch.tensor(labels, dtype=torch.int64),
        )


def _make_loader(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        _TabularDataset(labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_SparseBatchCollator(features),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(labels)
        sample_count += len(labels)
    return loss_sum / max(sample_count, 1)


def _save_confusion_matrix(
    values: list[list[int]],
    class_labels: list[str],
    output_path: Path,
) -> None:
    matrix = np.asarray(values)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_predictions(
    path: Path,
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[list[float]],
    class_labels: list[str],
    groups: dict[str, list[Any]],
) -> None:
    probability_names = [f"probability_{index}_{label}" for index, label in enumerate(class_labels)]
    fieldnames = ["y_true", "y_pred", *probability_names, *groups.keys()]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (true_label, predicted_label, scores) in enumerate(
            zip(y_true, y_pred, probabilities)
        ):
            row: dict[str, Any] = {"y_true": true_label, "y_pred": predicted_label}
            row.update(
                {
                    name: score
                    for name, score in zip(probability_names, scores)
                }
            )
            row.update(
                {
                    name: values[row_index]
                    for name, values in groups.items()
                }
            )
            writer.writerow(row)


def _save_permutation_importance(
    importance: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not importance:
        return
    ordered = sorted(
        importance,
        key=lambda item: item["balanced_accuracy_decrease_mean"],
    )
    names = [item["feature"] for item in ordered]
    means = [item["balanced_accuracy_decrease_mean"] for item in ordered]
    deviations = [item["balanced_accuracy_decrease_std"] for item in ordered]
    figure_height = max(5.0, 0.34 * len(ordered))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    positions = np.arange(len(ordered))
    axis.barh(positions, means, xerr=deviations, color="#3274A1", alpha=0.9)
    axis.axvline(0.0, color="#555555", linewidth=1.0)
    axis.set_yticks(positions, names)
    axis.set_xlabel("Decrease in balanced accuracy after shuffling")
    axis.set_title("Global permutation importance")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _calculate_permutation_importance(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    raw_test_features: pd.DataFrame,
    y_true_indices: np.ndarray,
    baseline_prediction: dict[str, list],
    class_count: int,
    *,
    batch_size: int,
    num_workers: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if repeats < 1:
        raise ValueError("importance_repeats must be at least 1")

    baseline_pred = np.asarray(baseline_prediction["y_pred"], dtype=np.int64)
    baseline_probabilities = np.asarray(baseline_prediction["probabilities"], dtype=float)
    baseline_accuracy = float(np.mean(y_true_indices == baseline_pred))
    baseline_balanced_accuracy = float(
        balanced_accuracy_score(y_true_indices, baseline_pred)
    )
    baseline_log_loss = float(
        log_loss(
            y_true_indices,
            baseline_probabilities,
            labels=np.arange(class_count),
        )
    )
    random_generator = np.random.default_rng(seed)
    importance: list[dict[str, Any]] = []

    for feature_index, column in enumerate(raw_test_features.columns, start=1):
        print(
            f"Permutation importance {feature_index}/{len(raw_test_features.columns)}: {column}"
        )
        accuracy_decreases = []
        balanced_accuracy_decreases = []
        log_loss_increases = []
        original_values = raw_test_features[column].to_numpy(copy=True)

        for _ in range(repeats):
            shuffled_features = raw_test_features.copy()
            shuffled_features[column] = original_values[
                random_generator.permutation(len(original_values))
            ]
            transformed = _to_float32(
                preprocessor.transform(clean_features(shuffled_features))
            )
            shuffled_loader = _make_loader(
                transformed,
                y_true_indices,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            shuffled_prediction = evaluate(model, shuffled_loader, device)
            shuffled_pred = np.asarray(shuffled_prediction["y_pred"], dtype=np.int64)
            shuffled_probabilities = np.asarray(
                shuffled_prediction["probabilities"], dtype=float
            )
            accuracy_decreases.append(
                baseline_accuracy - float(np.mean(y_true_indices == shuffled_pred))
            )
            balanced_accuracy_decreases.append(
                baseline_balanced_accuracy
                - float(balanced_accuracy_score(y_true_indices, shuffled_pred))
            )
            log_loss_increases.append(
                float(
                    log_loss(
                        y_true_indices,
                        shuffled_probabilities,
                        labels=np.arange(class_count),
                    )
                    - baseline_log_loss
                )
            )

        importance.append(
            {
                "feature": str(column),
                "accuracy_decrease_mean": float(np.mean(accuracy_decreases)),
                "accuracy_decrease_std": float(np.std(accuracy_decreases)),
                "balanced_accuracy_decrease_mean": float(
                    np.mean(balanced_accuracy_decreases)
                ),
                "balanced_accuracy_decrease_std": float(
                    np.std(balanced_accuracy_decreases)
                ),
                "log_loss_increase_mean": float(np.mean(log_loss_increases)),
                "log_loss_increase_std": float(np.std(log_loss_increases)),
                "repeats": repeats,
            }
        )
    return sorted(
        importance,
        key=lambda item: item["balanced_accuracy_decrease_mean"],
        reverse=True,
    )


def run_tabular_baseline(
    *,
    dataset: str | None = None,
    csv_path: Path | None = None,
    target_column: str | None = None,
    audit_columns: list[str] | None = None,
    sensitive_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    split_column: str | None = None,
    positive_label: str | None = None,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("results/tabular"),
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    importance_repeats: int = 5,
    basic_only: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    if (dataset == "adult") == (csv_path is not None):
        raise ValueError("Choose exactly one input: --dataset adult or --csv FILE")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or importance_repeats < 1:
        raise ValueError(
            "epochs, batch_size, learning_rate, and importance_repeats must be positive"
        )
    _set_seed(seed)
    if audit_columns is not None and sensitive_columns is not None:
        raise ValueError("Pass audit_columns or sensitive_columns, not both")
    requested_audit_columns = (
        audit_columns if audit_columns is not None else sensitive_columns
    )

    if dataset == "adult":
        if split_column is not None:
            raise ValueError("Adult already supplies an official split; omit --split-column")
        train_frame, test_frame = load_adult_income(data_dir)
        target_column = "income"
        # age_band is available as an additional audit feature; all original
        # Adult attributes remain model inputs by default.
        frame = pd.concat([train_frame, test_frame], ignore_index=True)
        official_train_count = len(train_frame)
        dataset_name = "UCI Adult (Census Income)"
        if positive_label is None:
            positive_label = ">50K"
    elif csv_path is not None:
        frame = load_tabular_csv(csv_path)
        if target_column is None:
            raise ValueError("--target-column is required when using --csv")
        frame = normalize_target(frame, target_column)
        official_train_count = None
        dataset_name = csv_path.stem
        drop_columns = list(drop_columns or [])
    else:
        raise ValueError("Only the built-in 'adult' dataset or --csv is currently supported")

    assert target_column is not None
    frame = normalize_target(frame, target_column)
    drop_columns = list(drop_columns or [])
    missing_columns = [column for column in drop_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing requested drop columns: {missing_columns}")

    class_labels = sorted(frame[target_column].astype(str).unique().tolist())
    if positive_label is not None and positive_label not in class_labels:
        raise ValueError(
            f"positive_label {positive_label!r} is not among the classes {class_labels}"
        )
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    labels = frame[target_column].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    split_values = None
    if split_column is not None:
        if dataset == "adult":
            raise ValueError("Adult already supplies an official split; omit --split-column")
        if split_column not in frame.columns:
            raise ValueError(f"Dataset does not contain split column {split_column!r}")
        split_values = frame[split_column].to_numpy()

    excluded = {target_column, *drop_columns}
    if dataset == "adult":
        excluded.add("age_band")
    if split_column is not None:
        excluded.add(split_column)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No model features remain after excluding the target and explicit drops")
    if basic_only:
        audit_columns = []
    elif requested_audit_columns is None:
        audit_columns = list(feature_columns)
        if dataset == "adult" and "age_band" in frame.columns:
            audit_columns.append("age_band")
    else:
        audit_columns = list(requested_audit_columns)
    if len(set(audit_columns)) != len(audit_columns):
        raise ValueError("Audit column names must be unique")
    missing_audit_columns = [
        column for column in audit_columns if column not in frame.columns
    ]
    if missing_audit_columns:
        raise ValueError(f"Dataset is missing requested audit columns: {missing_audit_columns}")
    if target_column in audit_columns:
        raise ValueError("The target column cannot also be an audit group")
    if split_column is not None and split_column in audit_columns:
        raise ValueError("The split column cannot be used as an audit group")
    features = clean_features(frame[feature_columns])

    train_indices, val_indices, test_indices = _split_indices(
        labels,
        seed=seed,
        official_train_count=official_train_count,
        split_values=split_values,
    )
    if set(labels[train_indices]) != set(range(len(class_labels))):
        raise ValueError("The training split must contain at least one row from every class")
    preprocessor, numeric_columns, categorical_columns = _make_preprocessor(
        features.iloc[train_indices]
    )
    x_train = _to_float32(preprocessor.fit_transform(features.iloc[train_indices]))
    x_val = _to_float32(preprocessor.transform(features.iloc[val_indices]))
    x_test = _to_float32(preprocessor.transform(features.iloc[test_indices]))

    train_loader = _make_loader(
        x_train, labels[train_indices], batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = _make_loader(
        x_val, labels[val_indices], batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = _make_loader(
        x_test, labels[test_indices], batch_size, shuffle=False, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_name} | device: {device}")
    print(
        f"Rows: train={len(train_indices)}, validation={len(val_indices)}, "
        f"test={len(test_indices)} | encoded features={x_train.shape[1]}"
    )
    model = TabularMLP(input_dim=x_train.shape[1], num_classes=len(class_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tabular_mlp.pt"
    preprocessor_path = output_dir / "preprocessor.joblib"
    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        validation = evaluate(model, val_loader, device)
        val_accuracy = float(
            np.mean(np.asarray(validation["y_true"]) == np.asarray(validation["y_pred"]))
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_accuracy": val_accuracy}
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(x_train.shape[1]),
                    "num_classes": len(class_labels),
                    "hidden_dims": [128, 64],
                    "dropout": 0.2,
                    "class_labels": class_labels,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "dataset": dataset_name,
                },
                checkpoint_path,
            )
        print(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
            f"- val_accuracy={val_accuracy:.4f} "
            f"- best_val_accuracy={best_val_accuracy:.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, device)
    y_true_indices = np.asarray(test_result["y_true"], dtype=np.int64)
    y_pred_indices = np.asarray(test_result["y_pred"], dtype=np.int64)
    probability_array = np.asarray(test_result["probabilities"], dtype=float)
    y_true = [class_labels[index] for index in y_true_indices]
    y_pred = [class_labels[index] for index in y_pred_indices]
    confidence = np.max(probability_array, axis=1)

    classification = calculate_classification_metrics(
        y_true_indices.tolist(), y_pred_indices.tolist()
    )
    classification["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true_indices, y_pred_indices)
    )
    classification["confusion_matrix"] = confusion_matrix(
        y_true_indices,
        y_pred_indices,
        labels=np.arange(len(class_labels)),
    ).tolist()
    metrics = {
        "model": "TabularMLP",
        "dataset": dataset_name,
        "target_column": target_column,
        "class_labels": class_labels,
        "positive_label": positive_label,
        "split_strategy": (
            "UCI official train/test split with 15% of training rows reserved for validation"
            if dataset == "adult"
            else f"provided split column: {split_column}"
            if split_column is not None
            else "stratified random 70/15/15 split"
        ),
        "split_sizes": {
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
        "feature_count_before_encoding": len(feature_columns),
        "feature_count_after_encoding": int(x_train.shape[1]),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "model_feature_columns": feature_columns,
        "analysis_mode": "basic classification evaluation" if basic_only else "full evaluation",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test_metrics": classification,
        "training_history": history,
    }
    if not basic_only:
        raw_test_features = features.iloc[test_indices].reset_index(drop=True)
        permutation_importance = _calculate_permutation_importance(
            model,
            preprocessor,
            raw_test_features,
            y_true_indices,
            test_result,
            len(class_labels),
            batch_size=batch_size,
            num_workers=num_workers,
            repeats=importance_repeats,
            seed=seed,
            device=device,
        )
        permutation_path = output_dir / "permutation_importance.json"
        with permutation_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "method": "global permutation importance on held-out test rows",
                    "metric_direction": {
                        "accuracy_decrease": "positive means the metric fell after shuffling",
                        "balanced_accuracy_decrease": "positive means the metric fell after shuffling",
                        "log_loss_increase": "positive means log loss rose after shuffling",
                    },
                    "interpretation_limitations": [
                        "This is predictive association, not a causal effect.",
                        "Correlated inputs can share or mask permutation importance.",
                        "Negative values mean shuffling improved the measured metric in these repeats.",
                    ],
                    "importance": permutation_importance,
                },
                output_file,
                indent=2,
                ensure_ascii=False,
            )
        _save_permutation_importance(
            permutation_importance,
            output_dir / "permutation_importance.png",
        )
        metrics["interpretability"] = {
            "method": "global permutation importance",
            "report": permutation_path.name,
            "plot": "permutation_importance.png",
        }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False)
    joblib.dump(preprocessor, preprocessor_path)
    _save_confusion_matrix(
        classification["confusion_matrix"],
        class_labels,
        output_dir / "confusion_matrix.png",
    )

    if not basic_only:
        calibration_bins = min(
            max(1, math.ceil(math.log2(len(y_true_indices)) + 1)), len(y_true_indices)
        )
        ece_bins = min(10, calibration_bins)
        ece, ece_details = calculate_ece(
            y_true_indices, y_pred_indices, confidence, num_bins=ece_bins
        )
        ace, classwise_bins = calculate_ace(
            y_true_indices,
            probability_array,
            num_bins=calibration_bins,
        )
        calibration = {
            "top_label_ece": ece,
            "classwise_ace": ace,
            "top_label_ece_num_bins": ece_bins,
            "classwise_ace_num_bins": calibration_bins,
            "ece_bins": ece_details,
            "classwise_bins": classwise_bins,
        }
        with (output_dir / "calibration.json").open("w", encoding="utf-8") as output_file:
            json.dump(calibration, output_file, indent=2, ensure_ascii=False)
        uncertainty = calculate_confidence_statistics(
            y_true_indices, y_pred_indices, confidence
        )
        with (output_dir / "uncertainty.json").open("w", encoding="utf-8") as output_file:
            json.dump(uncertainty, output_file, indent=2, ensure_ascii=False)

    test_groups: dict[str, list[Any]] = {}
    audit_grouping: dict[str, Any] = {}
    if not basic_only and audit_columns:
        test_groups, audit_grouping = build_feature_audit_groups(
            frame.iloc[train_indices][audit_columns].reset_index(drop=True),
            frame.iloc[test_indices][audit_columns].reset_index(drop=True),
            columns=audit_columns,
            include_intersections=True,
        )
    predictions_path = output_dir / "predictions.csv"
    _write_predictions(
        predictions_path,
        y_true,
        y_pred,
        probability_array.tolist(),
        class_labels,
        test_groups,
    )

    if not basic_only and audit_columns:
        fairness_path = output_dir / "fairness.json"
        fairness = run_fairness_audit(
            y_true=y_true,
            y_pred=y_pred,
            audit_features=test_groups,
            probabilities=probability_array,
            class_labels=class_labels,
            positive_label=positive_label,
            metadata={
                "model": "TabularMLP",
                "dataset": dataset_name,
                "split": "test",
                "target_column": target_column,
                "audit_attributes_used_as_inputs": [
                    column for column in audit_columns if column in feature_columns
                ],
                "audit_setup": audit_grouping,
                "training_seed": seed,
            },
            output_path=fairness_path,
        )
        ranking_path = write_disparity_ranking(
            fairness,
            output_dir / "fairness_ranking.csv",
        )
        charts: dict[str, Any] = {}
        roc_charts: dict[str, Any] = {}
        for index, attribute in enumerate(audit_columns):
            if fairness["attributes"][attribute]["group_metrics"]:
                charts[attribute] = save_group_metric_panels(
                    fairness,
                    sensitive_attribute=attribute,
                    output_stem=output_dir / f"fairness_group_{index + 1}_metrics",
                )
            roc_charts[attribute] = save_group_roc_curve_comparisons(
                y_true=y_true,
                group_values=test_groups[attribute],
                probabilities=probability_array,
                class_labels=class_labels,
                positive_label=positive_label,
                output_stem=output_dir / f"fairness_group_{index + 1}_roc",
            )
        fairness["artifacts"] = {
            "predictions_csv": predictions_path.name,
            "disparity_ranking_csv": ranking_path.name,
            "group_metric_charts": {
                name: {extension: path.name for extension, path in files.items()}
                for name, files in charts.items()
            },
            "group_roc_charts": {
                name: {
                    label: {extension: path.name for extension, path in files.items()}
                    for label, files in classes.items()
                }
                for name, classes in roc_charts.items()
            },
        }
        with fairness_path.open("w", encoding="utf-8") as output_file:
            json.dump(fairness, output_file, indent=2, ensure_ascii=False)
        print(f"Fairness report: {fairness_path}")
        print("Largest equalized-odds gaps:")
        for item in fairness["disparity_ranking"]["ranked_attributes"][:10]:
            if item["disparity_score"] is None:
                print(f"  {item['attribute']}: score unavailable")
            else:
                print(
                    f"  {item['attribute']}: {item['disparity_score']:.4f} "
                    f"({item['disparity_score'] * 100:.1f} pp), "
                    f"class={item['class_label']}, min_group_n={item['minimum_group_sample_count']}"
                )
    elif not basic_only:
        print("No feature columns supplied for the fairness audit.")

    metrics_line = (
        f"Test accuracy={classification['accuracy']:.4f} "
        f"balanced_accuracy={classification['balanced_accuracy']:.4f} "
        f"F1={classification['f1']:.4f}"
    )
    if not basic_only:
        metrics_line += f" | ECE={ece:.4f} | ACE={ace:.4f}"
    print(metrics_line)
    print(f"Artifacts: {output_dir.resolve()}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an MLP on Adult or a generic tabular CSV"
    )
    parser.add_argument("--dataset", choices=["adult"])
    parser.add_argument("--csv", type=Path, help="One CSV containing a tabular dataset")
    parser.add_argument("--target-column", help="Target label column for --csv")
    parser.add_argument(
        "--audit-columns",
        "--sensitive-columns",
        nargs="+",
        dest="audit_columns",
        help=(
            "Optional columns to audit instead of all model input features; "
            "numeric columns are binned and pairwise intersections are included"
        ),
    )
    parser.add_argument(
        "--drop-columns",
        nargs="+",
        default=[],
        help="Additional columns to omit from model features, such as an ID",
    )
    parser.add_argument(
        "--split-column",
        help="Optional CSV column containing train/validation/test row assignments",
    )
    parser.add_argument("--positive-label", help="Positive outcome label for binary fairness metrics")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adult_income"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=5,
        help="Repeats for each feature in the global permutation importance analysis",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Train and report standard classification metrics only; skip fairness and interpretation",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_tabular_baseline(
        dataset=args.dataset,
        csv_path=args.csv,
        target_column=args.target_column,
        audit_columns=args.audit_columns,
        drop_columns=args.drop_columns,
        split_column=args.split_column,
        positive_label=args.positive_label,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        importance_repeats=args.importance_repeats,
        basic_only=args.basic_only,
        seed=args.seed,
    )
