from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18 as torchvision_resnet18
from torch import nn
from datasets.adult_income import load_adult_income
from datasets.cifar10 import get_cifar10_dataloaders
from evaluation.base_metrics import calculate_base_metrics
from evaluation.calibration import calculate_calibration_metrics, save_calibration_plots
from evaluation.evaluator import evaluate
from evaluation.fairness import calculate_fairness_metrics, prepare_feature_groups
from models.resnet import create_resnet18
from models.tabular_mlp import TabularMLP


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def _save_metric_files(
    *,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    class_names: list[str] | tuple[str, ...],
    positive_label: str | None,
    predictions: dict[str, list],
    checkpoint_path: Path,
    preprocessing_note: str | None = None,
    fairness_groups: dict[str, np.ndarray] | None = None,
    fairness_group_definitions: dict[str, dict[str, Any]] | None = None,
) -> None:
    class_names = list(class_names)
    base_metrics = calculate_base_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["probabilities"],
        class_names=class_names,
        positive_label=positive_label,
    )
    calibration = calculate_calibration_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["confidence"],
        predictions["probabilities"],
    )

    comparison_index = (
        class_names.index(positive_label)
        if positive_label in class_names
        else class_names.index("cat")
        if "cat" in class_names
        else None
    )
    calibration_artifacts = save_calibration_plots(
        calibration,
        class_names,
        output_dir,
        comparison_class_index=comparison_index,
    )

    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "checkpoint": str(checkpoint_path),
        "split": "official test",
        "sample_count": len(predictions["y_true"]),
        "class_names": class_names,
        "positive_label": positive_label,
    }
    if preprocessing_note is not None:
        metadata["preprocessing"] = preprocessing_note

    _write_json(
        output_dir / "base_metrics.json",
        {**metadata, "metrics": base_metrics},
    )
    _write_json(
        output_dir / "calibration.json",
        {**metadata, "metrics": calibration, "plots": calibration_artifacts},
    )
    if fairness_groups is None:
        fairness = {
            "status": "not_applicable",
            "reason": "CIFAR-10 has no input group attributes for fairness metrics.",
        }
    else:
        if positive_label is None:
            raise ValueError("Fairness metrics require a named positive class")
        fairness_metrics = calculate_fairness_metrics(
            predictions["y_true"],
            predictions["y_pred"],
            fairness_groups,
            positive_class_index=class_names.index(positive_label),
        )
        fairness = {
            **fairness_metrics,
            "group_definitions": fairness_group_definitions or {},
        }
    _write_json(output_dir / "fairness.json", {**metadata, **fairness})


def evaluate_cifar10_checkpoint(
    checkpoint_path: Path,
    *,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    architecture: str = "auto",
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CIFAR-10 checkpoint not found: {checkpoint_path}")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, class_names = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        loaded.get("model_state_dict", loaded)
        if isinstance(loaded, dict)
        else loaded
    )
    if architecture == "auto":
        stem = state_dict.get("conv1.weight")
        if stem is None or stem.ndim != 4:
            raise ValueError("Cannot infer ResNet-18 variant from checkpoint conv1.weight")
        if stem.shape[-1] == 3:
            architecture = "cifar"
        elif stem.shape[-1] == 7:
            architecture = "torchvision"
        else:
            raise ValueError(f"Unsupported ResNet stem kernel size: {stem.shape[-1]}")

    if architecture == "cifar":
        model = create_resnet18(num_classes=len(class_names))
        model_name = "CIFAR ResNet-18"
    elif architecture == "torchvision":
        model = torchvision_resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model_name = "Torchvision ResNet-18"
    else:
        raise ValueError("architecture must be 'auto', 'cifar', or 'torchvision'")
    model.load_state_dict(state_dict)
    model.to(device)
    predictions = evaluate(model, test_loader, device)
    _save_metric_files(
        output_dir=output_dir,
        model_name=model_name,
        dataset_name="CIFAR-10",
        class_names=class_names,
        positive_label=None,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
    )
    print(f"CIFAR-10 metric files: {output_dir.resolve()}")


def _clean_adult_features(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        values = clean[column].astype("string").str.strip()
        clean[column] = values.replace("", pd.NA).replace("?", pd.NA).astype(object)
    for column in clean.columns:
        if clean[column].dtype == object:
            clean[column] = clean[column].where(clean[column].notna(), np.nan)
    return clean


def _rebuild_adult_preprocessor(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    *,
    seed: int,
) -> Any:
    train_indices, _ = train_test_split(
        np.arange(len(train_labels)),
        test_size=0.15,
        random_state=seed,
        stratify=train_labels,
    )
    numeric_columns = train_features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in train_features.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    preprocessor.fit(train_features.iloc[train_indices])
    return preprocessor


def evaluate_adult_checkpoint(
    checkpoint_path: Path,
    *,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 256,
    num_workers: int = 0,
    preprocessor_seed: int = 42,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adult checkpoint not found: {checkpoint_path}")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_labels")
    feature_columns = checkpoint.get("feature_columns")
    if not class_names or not feature_columns:
        raise ValueError("Adult checkpoint must store class_labels and feature_columns")

    train_frame, test_frame = load_adult_income(data_dir)
    if any(
        column not in train_frame.columns or column not in test_frame.columns
        for column in feature_columns
    ):
        raise ValueError("Adult checkpoint feature_columns do not match the loaded dataset")
    target_column = checkpoint.get("target_column", "income")
    label_to_index = {label: index for index, label in enumerate(class_names)}
    train_labels = train_frame[target_column].astype(str).map(label_to_index)
    test_labels = test_frame[target_column].astype(str).map(label_to_index)
    if train_labels.isna().any() or test_labels.isna().any():
        raise ValueError("Adult labels do not match the checkpoint class_labels")

    train_features = _clean_adult_features(train_frame[feature_columns])
    test_features = _clean_adult_features(test_frame[feature_columns])
    fairness_groups, fairness_group_definitions = prepare_feature_groups(
        train_features,
        test_features,
        numeric_bins=5,
    )
    preprocessor_path = checkpoint_path.with_name("preprocessor.joblib")
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        preprocessing_note = f"loaded from {preprocessor_path}"
    else:
        preprocessor = _rebuild_adult_preprocessor(
            train_features,
            train_labels.to_numpy(dtype=np.int64),
            seed=preprocessor_seed,
        )
        preprocessing_note = (
            "reconstructed from official training rows; the checkpoint contains "
            f"no fitted preprocessor, so seed {preprocessor_seed} was assumed"
        )

    transformed_test = preprocessor.transform(test_features)
    if sparse.issparse(transformed_test):
        transformed_test = transformed_test.toarray()
    transformed_test = np.asarray(transformed_test, dtype=np.float32)
    expected_input_dim = int(checkpoint["input_dim"])
    if transformed_test.ndim != 2 or transformed_test.shape[1] != expected_input_dim:
        received_input_dim = (
            transformed_test.shape[-1] if transformed_test.ndim == 2 else "invalid"
        )
        raise ValueError(
            "Rebuilt Adult preprocessing does not match the checkpoint input size: "
            f"expected {expected_input_dim}, received {received_input_dim}"
        )

    model = TabularMLP(
        input_dim=expected_input_dim,
        num_classes=int(checkpoint["num_classes"]),
        hidden_dims=tuple(checkpoint.get("hidden_dims", [128, 64])),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_dataset = TensorDataset(
        torch.from_numpy(transformed_test),
        torch.from_numpy(test_labels.to_numpy(dtype=np.int64).copy()),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = evaluate(model, test_loader, device)
    positive_label = ">50K" if ">50K" in class_names else None
    _save_metric_files(
        output_dir=output_dir,
        model_name="TabularMLP",
        dataset_name="UCI Adult",
        class_names=class_names,
        positive_label=positive_label,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
        preprocessing_note=preprocessing_note,
        fairness_groups=fairness_groups,
        fairness_group_definitions=fairness_group_definitions,
    )
    print(f"Adult metric files: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing model checkpoints; this command does not train models"
    )
    parser.add_argument("--model", choices=("cifar10", "adult", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument(
        "--cifar-checkpoint",
        type=Path,
        default=Path("trainedModels/resnet18_cifar10.pt"),
    )
    parser.add_argument(
        "--adult-checkpoint",
        type=Path,
        default=Path("trainedModels/tabular_mlp.pt"),
    )
    parser.add_argument("--cifar-batch-size", type=int, default=128)
    parser.add_argument("--adult-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preprocessor-seed", type=int, default=42)
    parser.add_argument(
        "--cifar-architecture",
        choices=("auto", "cifar", "torchvision"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.model in ("cifar10", "both"):
        evaluate_cifar10_checkpoint(
            arguments.cifar_checkpoint,
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir / "cifar10",
            batch_size=arguments.cifar_batch_size,
            num_workers=arguments.num_workers,
            architecture=arguments.cifar_architecture,
        )
    if arguments.model in ("adult", "both"):
        evaluate_adult_checkpoint(
            arguments.adult_checkpoint,
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir / "adult",
            batch_size=arguments.adult_batch_size,
            num_workers=arguments.num_workers,
            preprocessor_seed=arguments.preprocessor_seed,
        )


if __name__ == "__main__":
    main()
