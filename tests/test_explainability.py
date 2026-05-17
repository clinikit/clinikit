"""Tests for clinikit.explainability."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from clinikit.explainability import (
    PartialDependenceResult,
    lime_explain_instance,
    partial_dependence,
    shap_summary,
)

shap_available = importlib.util.find_spec("shap") is not None
lime_available = importlib.util.find_spec("lime") is not None


@pytest.fixture
def fitted_lr(rng: np.random.Generator) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# shap_summary
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shap_available, reason="run only when shap is missing")
def test_shap_raises_clear_import_error(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    with pytest.raises(ImportError, match="shap"):
        shap_summary(model, X)


@pytest.mark.skipif(not shap_available, reason="shap not installed")
def test_shap_returns_per_feature_importance(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    result = shap_summary(model, X, max_samples=30, random_state=0)
    assert result.feature_importance.shape == (X.shape[1],)
    assert np.all(result.feature_importance >= 0)


# ---------------------------------------------------------------------------
# lime_explain_instance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(lime_available, reason="run only when lime is missing")
def test_lime_raises_clear_import_error(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    with pytest.raises(ImportError, match="lime"):
        lime_explain_instance(model, X, instance_idx=0)


@pytest.mark.skipif(not lime_available, reason="lime not installed")
def test_lime_returns_feature_weights(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    weights = lime_explain_instance(model, X, instance_idx=0, random_state=0)
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_lime_rejects_no_proba_model(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    # Only matters when lime is installed; mocking a no-proba model.
    from sklearn.svm import LinearSVC

    if not lime_available:
        pytest.skip("lime not installed")
    model = LinearSVC().fit(fitted_lr[1], fitted_lr[2])
    with pytest.raises(ValueError, match="predict_proba"):
        lime_explain_instance(model, fitted_lr[1], instance_idx=0)


def test_lime_invalid_instance_idx(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    if not lime_available:
        pytest.skip("lime not installed")
    model, X, _ = fitted_lr
    with pytest.raises(ValueError, match="instance_idx"):
        lime_explain_instance(model, X, instance_idx=10_000)


# ---------------------------------------------------------------------------
# partial_dependence (no optional dep)
# ---------------------------------------------------------------------------


def test_partial_dependence_single_feature(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    out = partial_dependence(model, X, features=0)
    assert len(out) == 1
    result = out[0]
    assert isinstance(result, PartialDependenceResult)
    assert result.feature == 0
    assert result.grid.ndim == 1
    assert result.averaged_predictions.shape == result.grid.shape


def test_partial_dependence_multiple_features(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    out = partial_dependence(model, X, features=[0, 1, 2])
    assert [r.feature for r in out] == [0, 1, 2]


def test_partial_dependence_grid_resolution(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    out = partial_dependence(model, X, features=0, grid_resolution=20)
    assert out[0].grid.shape[0] <= 20


def test_partial_dependence_empty_features(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    with pytest.raises(ValueError, match="at least one"):
        partial_dependence(model, X, features=[])


def test_partial_dependence_out_of_range_feature(fitted_lr) -> None:  # type: ignore[no-untyped-def]
    model, X, _ = fitted_lr
    with pytest.raises(ValueError, match="out of range"):
        partial_dependence(model, X, features=99)
