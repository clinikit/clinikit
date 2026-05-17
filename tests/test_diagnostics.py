"""Tests for clinikit.diagnostics."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from clinikit.diagnostics import (
    SeedStabilityResult,
    cleanlab_label_issues,
    loo_influence,
    neighborhood_conflict,
    seed_stability,
)


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Cleanlab (skipped when the optional dep is missing)
# ---------------------------------------------------------------------------


cleanlab_available = importlib.util.find_spec("cleanlab") is not None


@pytest.mark.skipif(cleanlab_available, reason="run only when cleanlab is missing")
def test_cleanlab_raises_clear_import_error_when_missing(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    with pytest.raises(ImportError, match="cleanlab"):
        cleanlab_label_issues(X, y)


@pytest.mark.skipif(not cleanlab_available, reason="cleanlab is not installed")
def test_cleanlab_returns_indices(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    y = y.copy()
    y[0] = 1 - y[0]  # planted noisy label
    idx = cleanlab_label_issues(X, y)
    assert isinstance(idx, np.ndarray)
    assert idx.dtype == np.int64


# ---------------------------------------------------------------------------
# Neighborhood conflict
# ---------------------------------------------------------------------------


def test_neighborhood_conflict_shape_and_range(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    conflict = neighborhood_conflict(X, y, n_neighbors=5)
    assert conflict.shape == (X.shape[0],)
    assert np.all(conflict >= 0.0) and np.all(conflict <= 1.0)


def test_neighborhood_conflict_clean_data_low() -> None:
    # Two well-separated clusters: every sample's neighbours share its class.
    X = np.vstack([np.zeros((20, 2)) + 5.0, np.zeros((20, 2)) - 5.0])
    y = np.array([0] * 20 + [1] * 20)
    conflict = neighborhood_conflict(X, y, n_neighbors=3)
    assert conflict.max() == 0.0


def test_neighborhood_conflict_noisy_label_high() -> None:
    X = np.vstack([np.zeros((20, 2)) + 5.0, np.zeros((20, 2)) - 5.0])
    y = np.array([0] * 20 + [1] * 20)
    y[0] = 1  # flip one label in cluster 0
    conflict = neighborhood_conflict(X, y, n_neighbors=3)
    # The flipped point is now surrounded by class-0 neighbours but
    # labelled 1, so its conflict should be 1.0.
    assert conflict[0] == pytest.approx(1.0)


def test_neighborhood_conflict_invalid_n_neighbors() -> None:
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        neighborhood_conflict(np.zeros((10, 2)), np.zeros(10), n_neighbors=0)


def test_neighborhood_conflict_too_many_neighbors() -> None:
    with pytest.raises(ValueError, match="smaller than n_samples"):
        neighborhood_conflict(np.zeros((5, 2)), np.zeros(5), n_neighbors=10)


# ---------------------------------------------------------------------------
# LOO influence
# ---------------------------------------------------------------------------


def test_loo_influence_returns_correct_shape(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    inf = loo_influence(X, y, max_samples=20, random_state=0)
    assert inf.shape == (X.shape[0],)
    # Unsampled rows are NaN.
    assert np.isnan(inf).sum() == X.shape[0] - 20


def test_loo_influence_invalid_max_samples() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        loo_influence(np.zeros((10, 2)), np.zeros(10), max_samples=0)


def test_loo_influence_custom_metric(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    inf = loo_influence(X, y, metric=accuracy_score, max_samples=10, random_state=0)
    assert inf.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# Seed stability
# ---------------------------------------------------------------------------


def test_seed_stability_returns_per_seed_scores(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    result = seed_stability(X, y, LogisticRegression(), seeds=range(5))
    assert isinstance(result, SeedStabilityResult)
    assert result.scores.shape == (5,)
    assert result.seeds.shape == (5,)
    assert result.mean == pytest.approx(result.scores.mean())


def test_seed_stability_low_std_for_deterministic_data() -> None:
    # Perfectly separable data: every seed should give ~1.0 accuracy.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    X[:, 0] = np.where(rng.uniform(size=100) > 0.5, X[:, 0] + 10, X[:, 0] - 10)
    y = (X[:, 0] > 0).astype(int)
    result = seed_stability(X, y, LogisticRegression(), seeds=range(5))
    assert result.std < 0.05


def test_seed_stability_empty_seeds_rejected(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    with pytest.raises(ValueError, match="seeds"):
        seed_stability(X, y, LogisticRegression(), seeds=[])


def test_seed_stability_invalid_test_size(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    with pytest.raises(ValueError, match="test_size"):
        seed_stability(X, y, LogisticRegression(), test_size=1.5)
