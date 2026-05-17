"""Tests for clinikit.datasets — bundled UCI benchmarks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinikit.datasets import (
    DatasetBunch,
    load_heart,
    load_pima,
    load_wisconsin,
)

DATASET_SHAPES = {
    "pima": (768, 8),
    "wisconsin": (569, 30),
    "heart": (303, 13),
}

LOADERS = {
    "pima": load_pima,
    "wisconsin": load_wisconsin,
    "heart": load_heart,
}

TARGET_NAMES = {
    "pima": "outcome",
    "wisconsin": "diagnosis",
    "heart": "target",
}


# ---------------------------------------------------------------------------
# Shape and surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DATASET_SHAPES))
def test_return_x_y_shapes(name: str) -> None:
    loader = LOADERS[name]
    X, y = loader(return_X_y=True, as_frame=True)
    n_rows, n_features = DATASET_SHAPES[name]
    assert X.shape == (n_rows, n_features)
    assert y.shape == (n_rows,)


@pytest.mark.parametrize("name", list(DATASET_SHAPES))
def test_return_x_y_numpy(name: str) -> None:
    loader = LOADERS[name]
    X, y = loader(return_X_y=True, as_frame=False)
    n_rows, n_features = DATASET_SHAPES[name]
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_rows, n_features)
    assert y.shape == (n_rows,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64


@pytest.mark.parametrize("name", list(DATASET_SHAPES))
def test_return_bunch(name: str) -> None:
    loader = LOADERS[name]
    bunch = loader()  # default: return_X_y=False, as_frame=True
    assert isinstance(bunch, DatasetBunch)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)
    assert bunch.target_name == TARGET_NAMES[name]
    assert bunch.feature_names == list(bunch.data.columns)
    assert TARGET_NAMES[name] not in bunch.feature_names
    assert bunch.description
    assert bunch.source_url.startswith("https://")
    assert bunch.frame is not None
    assert TARGET_NAMES[name] in bunch.frame.columns


@pytest.mark.parametrize("name", list(DATASET_SHAPES))
def test_target_is_binary(name: str) -> None:
    _, y = LOADERS[name](return_X_y=True, as_frame=True)
    unique = set(pd.unique(y).tolist())
    assert unique <= {0, 1}, f"{name!r}: target labels are not {{0,1}}: {unique!r}"


# ---------------------------------------------------------------------------
# Dataset-specific sanity
# ---------------------------------------------------------------------------


def test_pima_column_names() -> None:
    bunch = load_pima()
    assert "glucose" in bunch.feature_names
    assert "age" in bunch.feature_names
    assert "bmi" in bunch.feature_names


def test_wisconsin_no_missing_values() -> None:
    X, _ = load_wisconsin(return_X_y=True, as_frame=True)
    assert not X.isna().any().any()


def test_heart_preserves_missing_values() -> None:
    # The Cleveland 'ca' and 'thal' columns contain '?' in the source
    # which our loader maps to NaN.
    X, _ = load_heart(return_X_y=True, as_frame=True)
    assert X[["ca", "thal"]].isna().any().any()


def test_as_frame_false_drops_nans_to_nan_floats() -> None:
    X_arr, _ = load_heart(return_X_y=True, as_frame=False)
    # NaNs must survive the conversion to a float ndarray.
    assert np.isnan(X_arr).any()


# ---------------------------------------------------------------------------
# Portability — loaders must work with importlib.resources, not paths.
# ---------------------------------------------------------------------------


def test_loaders_do_not_touch_user_paths(tmp_path: Path) -> None:
    # If a loader were to read from a hardcoded path, switching CWD
    # would break it. This test runs each loader from an unrelated
    # temp directory.
    original = Path.cwd()
    try:
        os.chdir(tmp_path)
        for loader in LOADERS.values():
            X, y = loader(return_X_y=True)
            assert X is not None
            assert y is not None
    finally:
        os.chdir(original)
