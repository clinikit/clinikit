"""Tests for clinikit.automl."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from clinikit.automl import AutoMLResult, run_autogluon, run_flaml, run_tabpfn

tabpfn_available = importlib.util.find_spec("tabpfn") is not None
flaml_available = importlib.util.find_spec("flaml") is not None
autogluon_available = importlib.util.find_spec("autogluon") is not None


@pytest.fixture
def small_split(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X[:45], y[:45], X[45:], y[45:]


# ---------------------------------------------------------------------------
# Validation (no backend required)
# ---------------------------------------------------------------------------


def test_run_flaml_rejects_non_positive_budget(small_split) -> None:  # type: ignore[no-untyped-def]
    if not flaml_available:
        pytest.skip("flaml not installed")
    X_tr, y_tr, X_te, _ = small_split
    with pytest.raises(ValueError, match="time_budget"):
        run_flaml(X_tr, y_tr, X_te, time_budget=0.0)


def test_run_autogluon_rejects_non_positive_budget(small_split) -> None:  # type: ignore[no-untyped-def]
    if not autogluon_available:
        pytest.skip("autogluon not installed")
    X_tr, y_tr, X_te, _ = small_split
    with pytest.raises(ValueError, match="time_budget"):
        run_autogluon(X_tr, y_tr, X_te, time_budget=0.0)


def test_validate_split_feature_mismatch(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    bad_X_te = X_te[:, :2]
    if tabpfn_available:
        with pytest.raises(ValueError, match="features"):
            run_tabpfn(X_tr, y_tr, bad_X_te)
    elif flaml_available:
        with pytest.raises(ValueError, match="features"):
            run_flaml(X_tr, y_tr, bad_X_te)
    elif autogluon_available:
        with pytest.raises(ValueError, match="features"):
            run_autogluon(X_tr, y_tr, bad_X_te)
    else:
        pytest.skip("no automl backend installed; validation runs after import")


# ---------------------------------------------------------------------------
# Import errors when backend is missing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(tabpfn_available, reason="run only when tabpfn is missing")
def test_run_tabpfn_raises_clear_import_error(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    with pytest.raises(ImportError, match="tabpfn"):
        run_tabpfn(X_tr, y_tr, X_te)


@pytest.mark.skipif(flaml_available, reason="run only when flaml is missing")
def test_run_flaml_raises_clear_import_error(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    with pytest.raises(ImportError, match="flaml"):
        run_flaml(X_tr, y_tr, X_te, time_budget=5.0)


@pytest.mark.skipif(autogluon_available, reason="run only when autogluon is missing")
def test_run_autogluon_raises_clear_import_error(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    with pytest.raises(ImportError, match="autogluon"):
        run_autogluon(X_tr, y_tr, X_te, time_budget=5.0)


# ---------------------------------------------------------------------------
# Smoke tests when backend is available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not tabpfn_available, reason="tabpfn not installed")
def test_run_tabpfn_smoke(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    result = run_tabpfn(X_tr, y_tr, X_te, random_state=0)
    assert isinstance(result, AutoMLResult)
    assert result.backend == "tabpfn"
    assert result.y_pred.shape[0] == X_te.shape[0]


@pytest.mark.skipif(not flaml_available, reason="flaml not installed")
def test_run_flaml_smoke(small_split) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    result = run_flaml(X_tr, y_tr, X_te, time_budget=5.0, random_state=0)
    assert isinstance(result, AutoMLResult)
    assert result.backend == "flaml"
    assert result.y_pred.shape[0] == X_te.shape[0]
    assert "best_estimator" in result.metadata


@pytest.mark.skipif(not autogluon_available, reason="autogluon not installed")
def test_run_autogluon_smoke(small_split, tmp_path) -> None:  # type: ignore[no-untyped-def]
    X_tr, y_tr, X_te, _ = small_split
    result = run_autogluon(X_tr, y_tr, X_te, time_budget=10.0, work_dir=tmp_path, random_state=0)
    assert isinstance(result, AutoMLResult)
    assert result.backend == "autogluon"
    assert result.y_pred.shape[0] == X_te.shape[0]
