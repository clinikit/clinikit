"""Fetch and normalise the UCI benchmark CSVs bundled with clinikit.

Run once during initial setup or when a dataset needs refreshing:

    python scripts/fetch_datasets.py

Outputs are written to ``src/clinikit/datasets/data/`` and committed
to the repository so the package works after a clean
``pip install clinikit`` with no network access.

Datasets
--------
- PIMA Indians Diabetes (UCI / NIDDK, public domain).
- Wisconsin Breast Cancer (Diagnostic) — taken from the copy bundled
  with scikit-learn (``sklearn.datasets.load_breast_cancer``).
- Heart Disease (Cleveland subset, UCI).
"""

from __future__ import annotations

import io
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "clinikit" / "datasets" / "data"

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
PIMA_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
]

HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def _http_get(url: str, *, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "clinikit-bootstrap/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def fetch_pima() -> pd.DataFrame:
    raw = _http_get(PIMA_URL)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=PIMA_COLUMNS)
    df["outcome"] = df["outcome"].astype(np.int64)
    return df


def fetch_wisconsin() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].astype(np.int64)
    return df


def fetch_heart() -> pd.DataFrame:
    raw = _http_get(HEART_URL)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=HEART_COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(np.int64)
    df = df.drop(columns=["num"])
    return df


DATASET_BUILDERS = {
    "pima.csv": fetch_pima,
    "wisconsin.csv": fetch_wisconsin,
    "heart.csv": fetch_heart,
}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in DATASET_BUILDERS.items():
        target = DATA_DIR / filename
        print(f"-> {filename}: fetching ...", flush=True)
        try:
            df = builder()
        except Exception as exc:
            print(f"   FAILED: {exc}", file=sys.stderr)
            return 1
        df.to_csv(target, index=False, lineterminator="\n")
        size_kb = target.stat().st_size / 1024
        print(f"   wrote {target.relative_to(REPO_ROOT)} ({df.shape}, {size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
