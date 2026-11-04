"""Train and evaluate a neural classifier on Adult or a user supplied CSV."""

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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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
from experiments.fairness import run_fairness_audit
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


def run_tabular_baseline(
    *,
    dataset: str | None = None,
    csv_path: Path | None = None,
    target_column: str | None = None,
    sensitive_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    split_column: str | None = None,
    include_sensitive_features: bool = False,
    positive_label: str | None = None,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("results/tabular"),
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    if (dataset == "adult") == (csv_path is not None):
        raise ValueError("Choose exactly one input: --dataset adult or --csv FILE")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    _set_seed(seed)

    if dataset == "adult":
        if split_column is not None:
            raise ValueError("Adult already supplies an official split; omit --split-column")
        train_frame, test_frame = load_adult_income(data_dir)
        target_column = "income"
        if sensitive_columns is None:
            sensitive_columns = ["sex", "race", "age_band"]
        # Age is audited in bands; fnlwgt is survey-weight metadata rather than
        # an ordinary person-level predictor.
        drop_columns = list(drop_columns or []) + ["age", "fnlwgt"]
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
        sensitive_columns = list(sensitive_columns or [])
        drop_columns = list(drop_columns or [])
    else:
        raise ValueError("Only the built-in 'adult' dataset or --csv is currently supported")

    assert target_column is not None
    frame = normalize_target(frame, target_column)
    sensitive_columns = list(sensitive_columns or [])
    drop_columns = list(drop_columns or [])
    if len(set(sensitive_columns)) != len(sensitive_columns):
        raise ValueError("Sensitive column names must be unique")
    missing_columns = [
        column
        for column in [*sensitive_columns, *drop_columns]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing requested columns: {missing_columns}")
    if target_column in sensitive_columns:
        raise ValueError("The target column cannot also be a sensitive group")

    class_labels = sorted(frame[target_column].astype(str).unique().tolist())
    if positive_label is not None and positive_label not in class_labels:
        raise ValueError(
            f"positive_label {positive_label!r} is not among the classes {class_labels}"
        )
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    labels = frame[target_column].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    group_frame = frame[sensitive_columns].copy()
    split_values = None
    if split_column is not None:
        if dataset == "adult":
            raise ValueError("Adult already supplies an official split; omit --split-column")
        if split_column not in frame.columns:
            raise ValueError(f"Dataset does not contain split column {split_column!r}")
        split_values = frame[split_column].to_numpy()

    excluded = {target_column, *drop_columns}
    if split_column is not None:
        excluded.add(split_column)
    if not include_sensitive_features:
        excluded.update(sensitive_columns)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No model features remain after excluding the target and groups")
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
        "sensitive_columns": sensitive_columns,
        "sensitive_features_in_model": include_sensitive_features,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test_metrics": classification,
        "training_history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False)
    joblib.dump(preprocessor, preprocessor_path)
    _save_confusion_matrix(
        classification["confusion_matrix"],
        class_labels,
        output_dir / "confusion_matrix.png",
    )

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
    for column in sensitive_columns:
        raw_values = group_frame.iloc[test_indices][column].tolist()
        test_groups[column] = [
            None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            for value in raw_values
        ]
    predictions_path = output_dir / "predictions.csv"
    _write_predictions(
        predictions_path,
        y_true,
        y_pred,
        probability_array.tolist(),
        class_labels,
        test_groups,
    )

    if sensitive_columns:
        fairness_path = output_dir / "fairness.json"
        fairness = run_fairness_audit(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=test_groups,
            probabilities=probability_array,
            class_labels=class_labels,
            positive_label=positive_label,
            metadata={
                "model": "TabularMLP",
                "dataset": dataset_name,
                "split": "test",
                "target_column": target_column,
                "sensitive_features_in_model": include_sensitive_features,
                "training_seed": seed,
            },
            output_path=fairness_path,
        )
        charts: dict[str, Any] = {}
        roc_charts: dict[str, Any] = {}
        for index, attribute in enumerate(sensitive_columns):
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
    else:
        print("No sensitive columns supplied; fairness report was not generated.")

    print(
        f"Test accuracy={classification['accuracy']:.4f} "
        f"balanced_accuracy={classification['balanced_accuracy']:.4f} "
        f"F1={classification['f1']:.4f} | ECE={ece:.4f} | ACE={ace:.4f}"
    )
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
        "--sensitive-columns",
        nargs="+",
        help="Categorical group columns to audit; omitted from model features by default",
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
    parser.add_argument(
        "--include-sensitive-features",
        action="store_true",
        help="Allow sensitive group columns as model features",
    )
    parser.add_argument("--positive-label", help="Positive outcome label for binary fairness metrics")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adult_income"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_tabular_baseline(
        dataset=args.dataset,
        csv_path=args.csv,
        target_column=args.target_column,
        sensitive_columns=args.sensitive_columns,
        drop_columns=args.drop_columns,
        split_column=args.split_column,
        include_sensitive_features=args.include_sensitive_features,
        positive_label=args.positive_label,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
    )
