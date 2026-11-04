"""Shared input helpers for supervised tabular classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_tabular_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while treating common blank and question-mark markers as missing."""
    frame = pd.read_csv(
        path,
        skipinitialspace=True,
        na_values=["?", "NA", "N/A", ""],
        keep_default_na=True,
    )
    if frame.empty:
        raise ValueError(f"CSV contains no data rows: {path}")
    if frame.columns.has_duplicates:
        raise ValueError("CSV column names must be unique")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.replace("", pd.NA).replace("?", pd.NA)
    return frame


def normalize_target(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in frame.columns:
        raise ValueError(f"Dataset does not contain target column {target_column!r}")
    clean = frame.dropna(subset=[target_column]).copy()
    clean[target_column] = clean[target_column].astype("string").str.strip()
    clean = clean[clean[target_column].notna() & (clean[target_column] != "")]
    if clean[target_column].nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    return clean.reset_index(drop=True)


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature missing values for scikit-learn imputers."""
    clean = frame.copy()
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        values = clean[column].astype("string").str.strip()
        clean[column] = values.replace("", pd.NA).replace("?", pd.NA).astype(object)
    for column in clean.columns:
        if clean[column].dtype == object:
            clean[column] = clean[column].where(clean[column].notna(), np.nan)
    return clean
