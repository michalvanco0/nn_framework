"""Download and read the official UCI Adult income train/test files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)


def _download_adult_files(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if train_path.exists() and test_path.exists():
        return train_path, test_path

    print(f"Downloading UCI Adult dataset from {ADULT_URL}")
    with urlopen(ADULT_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        members = {Path(name).name.lower(): name for name in archive.namelist()}
        for filename, destination in (
            ("adult.data", train_path),
            ("adult.test", test_path),
        ):
            member = members.get(filename)
            if member is None:
                raise RuntimeError(
                    f"The UCI archive does not contain the expected file {filename}"
                )
            destination.write_bytes(archive.read(member))

    return train_path, test_path


def _read_adult_file(path: Path, *, is_test: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        skiprows=1 if is_test else 0,
        skipinitialspace=True,
        na_values=["?"],
        keep_default_na=True,
    )
    frame["income"] = frame["income"].astype("string").str.strip().str.rstrip(".")
    frame = frame.dropna(subset=["income"]).copy()
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["age"], errors="coerce"),
        bins=[0, 29, 39, 49, 59, np.inf],
        labels=["17-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype("string")
    return frame.reset_index(drop=True)


def load_adult_income(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official training and test splits, including ``age_band``."""
    train_path, test_path = _download_adult_files(Path(data_dir))
    return (
        _read_adult_file(train_path, is_test=False),
        _read_adult_file(test_path, is_test=True),
    )
