"""Tests for clinikit.active_learning."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from clinikit.active_learning import ActiveLearningResult, run_active_learning

modal_available = importlib.util.find_spec("modAL") is not None


@pytest.fixture
def pool(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Import-error path (no modAL)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(modal_available, reason="run only when modAL is missing")
def test_run_active_learning_raises_clear_import_error(pool) -> None:  # type: ignore[no-untyped-def]
    X, y = pool
    init = np.array([0, 1, 2, 3], dtype=np.int64)
    with pytest.raises(ImportError, match="modAL"):
        run_active_learning(
            LogisticRegression(),
            X,
            oracle=lambda Xs: y[: Xs.shape[0]],
            initial_indices=init,
            initial_labels=y[init],
            n_queries=3,
        )


# ---------------------------------------------------------------------------
# Validation (always run; ValueError fires before the modAL import in some
# paths, after in others — guarded by skipif where needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not modal_available, reason="modAL not installed")
def test_invalid_n_queries(pool) -> None:  # type: ignore[no-untyped-def]
    X, y = pool
    init = np.array([0, 1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="n_queries"):
        run_active_learning(
            LogisticRegression(),
            X,
            oracle=lambda Xs: y[: Xs.shape[0]],
            initial_indices=init,
            initial_labels=y[init],
            n_queries=0,
        )


@pytest.mark.skipif(not modal_available, reason="modAL not installed")
def test_unknown_strategy_raises(pool) -> None:  # type: ignore[no-untyped-def]
    X, y = pool
    init = np.array([0, 1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="query_strategy"):
        run_active_learning(
            LogisticRegression(),
            X,
            oracle=lambda Xs: y[: Xs.shape[0]],
            initial_indices=init,
            initial_labels=y[init],
            query_strategy="nope",
            n_queries=2,
        )


@pytest.mark.skipif(not modal_available, reason="modAL not installed")
def test_initial_length_mismatch(pool) -> None:  # type: ignore[no-untyped-def]
    X, y = pool
    init = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="initial_indices"):
        run_active_learning(
            LogisticRegression(),
            X,
            oracle=lambda Xs: y[: Xs.shape[0]],
            initial_indices=init,
            initial_labels=y[:2],
            n_queries=2,
        )


# ---------------------------------------------------------------------------
# Smoke test (modAL installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not modal_available, reason="modAL not installed")
def test_loop_queries_n_samples(pool) -> None:  # type: ignore[no-untyped-def]
    X, y = pool
    init = np.array([0, 1, 30, 31], dtype=np.int64)

    def oracle(X_query: np.ndarray) -> np.ndarray:
        # Look up labels by row equality.
        idx = [int(np.argmin(np.linalg.norm(X - row, axis=1))) for row in X_query]
        return y[idx]

    result = run_active_learning(
        LogisticRegression(),
        X,
        oracle=oracle,
        initial_indices=init,
        initial_labels=y[init],
        n_queries=5,
    )
    assert isinstance(result, ActiveLearningResult)
    assert result.n_queries == 5
    assert result.queried_indices.shape[0] == 5
    # No duplicates with the seed set.
    assert len(set(result.queried_indices.tolist())) == 5
    assert not set(result.queried_indices.tolist()) & set(init.tolist())
