"""Active-learning labelling loop on top of modAL.

modAL is an optional dependency — install with
``pip install "clinikit[active]"``. The single public function
:func:`run_active_learning` queries the most-informative samples from
an unlabelled pool, asks a caller-supplied oracle for the truth, and
re-trains until ``n_queries`` annotations have been gathered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

__all__ = [
    "ActiveLearningResult",
    "run_active_learning",
]


@dataclass(slots=True)
class ActiveLearningResult:
    """Output of :func:`run_active_learning`."""

    estimator: BaseEstimator
    queried_indices: NDArray[np.int64]
    queried_labels: NDArray
    n_queries: int


_QUERY_STRATEGY_REGISTRY: dict[str, Any] = {}


def _load_strategy(name: str) -> Any:
    """Lazy-resolve modAL query strategies."""
    if name in _QUERY_STRATEGY_REGISTRY:
        return _QUERY_STRATEGY_REGISTRY[name]
    try:
        from modAL.uncertainty import (  # noqa: PLC0415
            entropy_sampling,
            margin_sampling,
            uncertainty_sampling,
        )
    except ImportError as exc:
        raise ImportError(
            "modAL is required. Install with: pip install 'clinikit[active]'."
        ) from exc

    table = {
        "uncertainty": uncertainty_sampling,
        "margin": margin_sampling,
        "entropy": entropy_sampling,
    }
    _QUERY_STRATEGY_REGISTRY.update(table)
    if name not in table:
        raise ValueError(f"Unknown query_strategy {name!r}. Available: {sorted(table)}.")
    return table[name]


def run_active_learning(
    estimator: BaseEstimator,
    X_pool: ArrayLike,
    *,
    oracle: Callable[[NDArray], NDArray],
    initial_indices: ArrayLike,
    initial_labels: ArrayLike,
    query_strategy: str = "uncertainty",
    n_queries: int = 10,
) -> ActiveLearningResult:
    """Run a pool-based active-learning loop.

    Parameters
    ----------
    estimator : sklearn classifier
        Cloned by modAL inside its ActiveLearner; must support
        ``predict_proba``.
    X_pool : array-like of shape (n_samples, n_features)
        Full pool — labelled and unlabelled rows together.
    oracle : callable
        ``oracle(X) -> y`` where ``X`` is the *queried* slice. In real
        deployments this prompts a human annotator; in tests it is
        usually a ground-truth lookup.
    initial_indices : array-like of int
        Indices into ``X_pool`` that already have labels.
    initial_labels : array-like
        Labels for ``initial_indices``.
    query_strategy : {"uncertainty", "margin", "entropy"}, default
        ``"uncertainty"``.
    n_queries : int, default 10
        Number of additional samples to query.

    Returns
    -------
    ActiveLearningResult

    Raises
    ------
    ImportError
        If modAL is not installed.
    """
    try:
        from modAL.models import ActiveLearner  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "modAL is required. Install with: pip install 'clinikit[active]'."
        ) from exc

    if n_queries < 1:
        raise ValueError(f"n_queries must be >= 1; got {n_queries!r}.")
    strategy = _load_strategy(query_strategy)

    X_pool_arr = np.asarray(X_pool)
    init_idx = np.asarray(initial_indices, dtype=np.int64)
    init_lbl = np.asarray(initial_labels)
    if init_idx.shape[0] != init_lbl.shape[0]:
        raise ValueError("initial_indices and initial_labels must align in length.")

    learner = ActiveLearner(
        estimator=estimator,
        query_strategy=strategy,
        X_training=X_pool_arr[init_idx],
        y_training=init_lbl,
    )

    queried_idx: list[int] = []
    queried_lbl: list[Any] = []
    used = set(init_idx.tolist())
    for _ in range(n_queries):
        candidate_idx = np.array(
            [i for i in range(X_pool_arr.shape[0]) if i not in used], dtype=np.int64
        )
        if candidate_idx.size == 0:
            break
        candidate_X = X_pool_arr[candidate_idx]
        query_idx, _ = learner.query(candidate_X)
        chosen = int(candidate_idx[query_idx][0])
        chosen_X = X_pool_arr[chosen : chosen + 1]
        chosen_y = oracle(chosen_X)
        learner.teach(chosen_X, np.asarray(chosen_y))
        queried_idx.append(chosen)
        queried_lbl.append(np.asarray(chosen_y).ravel()[0])
        used.add(chosen)

    return ActiveLearningResult(
        estimator=learner.estimator,
        queried_indices=np.asarray(queried_idx, dtype=np.int64),
        queried_labels=np.asarray(queried_lbl),
        n_queries=len(queried_idx),
    )
