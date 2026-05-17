"""Label-noise and stability diagnostics.

- :func:`cleanlab_label_issues` — Cleanlab-based label-noise detection
  (lazy-imports the optional ``cleanlab`` dependency).
- :func:`neighborhood_conflict` — k-nearest-neighbour disagreement
  rate per sample.
- :func:`loo_influence` — leave-one-out metric influence per sample
  (computed on a random subset for efficiency on large data).
- :func:`seed_stability` — distribution of an evaluation metric across
  multiple training seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "SeedStabilityResult",
    "cleanlab_label_issues",
    "loo_influence",
    "neighborhood_conflict",
    "seed_stability",
]


# ---------------------------------------------------------------------------
# Cleanlab (lazy import)
# ---------------------------------------------------------------------------


def cleanlab_label_issues(
    X: ArrayLike,
    y: ArrayLike,
    base_estimator: ClassifierMixin | None = None,
    *,
    return_indices: bool = True,
    **cleanlab_kwargs: object,
) -> NDArray[np.int64] | NDArray[np.bool_]:
    """Detect probable label noise using Cleanlab.

    Requires the optional ``[diagnostics]`` extra
    (``pip install "clinikit[diagnostics]"``).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    base_estimator : sklearn classifier, optional
        Cloned and fitted under the hood. Default
        :class:`~sklearn.linear_model.LogisticRegression`.
    return_indices : bool, default True
        If True, return a 1-D ndarray of integer indices of likely
        label issues. If False, return a 1-D boolean mask of the same
        length as ``y``.
    **cleanlab_kwargs : passed to ``cleanlab.classification.CleanLearning``.

    Returns
    -------
    ndarray
        Either integer indices of issues or a boolean mask, controlled
        by ``return_indices``.

    Raises
    ------
    ImportError
        If Cleanlab is not installed.

    Examples
    --------
    >>> from clinikit.diagnostics import cleanlab_label_issues  # doctest: +SKIP
    >>> idx = cleanlab_label_issues(X, y)  # doctest: +SKIP
    """
    try:
        from cleanlab.classification import CleanLearning
    except ImportError as exc:
        raise ImportError(
            "cleanlab is required. Install with: pip install 'clinikit[diagnostics]'."
        ) from exc

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    check_consistent_length(X_arr, y_arr)

    if base_estimator is None:
        base_estimator = LogisticRegression()
    cl = CleanLearning(clf=clone(base_estimator), **cleanlab_kwargs)
    issues = cl.find_label_issues(X_arr, y_arr)
    # Cleanlab returns a DataFrame with an "is_label_issue" boolean column.
    mask = np.asarray(issues["is_label_issue"].to_numpy(), dtype=bool)
    if return_indices:
        return np.where(mask)[0].astype(np.int64)
    return mask


# ---------------------------------------------------------------------------
# Neighborhood conflict
# ---------------------------------------------------------------------------


def neighborhood_conflict(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> NDArray[np.float64]:
    """Per-sample k-NN label-disagreement rate.

    For each sample, computes the fraction of its ``n_neighbors``
    nearest neighbours whose label differs. High values flag samples
    that sit in a region the rest of the data labels differently —
    candidates for label noise or decision-boundary cases.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_neighbors : int, default 10
    metric : str, default "euclidean"
        Passed to :class:`~sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    ndarray of shape (n_samples,)
        Conflict rate per sample, in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.diagnostics import neighborhood_conflict
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> conflict = neighborhood_conflict(X, y, n_neighbors=5)
    >>> conflict.shape
    (60,)
    """
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1; got {n_neighbors!r}.")
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    check_consistent_length(X_arr, y_arr)

    n_samples = X_arr.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError(f"n_neighbors={n_neighbors} must be smaller than n_samples={n_samples}.")

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(X_arr)
    _, indices = nn.kneighbors(X_arr)
    # Drop the self-match in column 0.
    neighbour_idx = indices[:, 1:]
    neighbour_labels = y_arr[neighbour_idx]
    same = neighbour_labels == y_arr[:, None]
    return np.asarray(1.0 - same.mean(axis=1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Leave-one-out influence
# ---------------------------------------------------------------------------


def loo_influence(
    X: ArrayLike,
    y: ArrayLike,
    base_estimator: ClassifierMixin | None = None,
    *,
    metric: Callable[..., float] = accuracy_score,
    max_samples: int = 100,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Approximate leave-one-out training influence per sample.

    For each chosen training index ``i``, the estimator is retrained
    on ``X`` with row ``i`` removed and the metric is re-evaluated on
    the *full* training set. The influence is
    ``metric_full - metric_loo``: positive means dropping ``i`` hurt
    the metric (the sample was useful); negative means dropping ``i``
    helped (the sample may be harmful / noisy).

    Parameters
    ----------
    X, y : array-like of shape (n_samples, n_features) and (n_samples,)
    base_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    metric : callable, default :func:`sklearn.metrics.accuracy_score`
        Signature ``metric(y_true, y_pred) -> float``.
    max_samples : int, default 100
        Cap on the number of LOO retrains; samples are drawn uniformly
        without replacement when ``max_samples < n_samples``. Indices
        not selected get an influence of ``np.nan``.
    random_state : int, optional

    Returns
    -------
    ndarray of shape (n_samples,)
        Per-sample influence; ``np.nan`` for unsampled indices.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    check_consistent_length(X_arr, y_arr)
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1; got {max_samples!r}.")

    rng = np.random.default_rng(random_state)
    n = X_arr.shape[0]
    if base_estimator is None:
        base_estimator = LogisticRegression()

    sample_size = min(max_samples, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    full_est = clone(base_estimator).fit(X_arr, y_arr)
    full_pred = full_est.predict(X_arr)
    full_metric = float(metric(y_arr, full_pred))

    influences = np.full(n, np.nan, dtype=np.float64)
    for i in sample_idx:
        mask = np.ones(n, dtype=bool)
        mask[int(i)] = False
        loo_est = clone(base_estimator).fit(X_arr[mask], y_arr[mask])
        loo_pred = loo_est.predict(X_arr)
        influences[int(i)] = full_metric - float(metric(y_arr, loo_pred))
    return influences


# ---------------------------------------------------------------------------
# Seed stability
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SeedStabilityResult:
    """Output of :func:`seed_stability`."""

    seeds: NDArray[np.int64]
    scores: NDArray[np.float64]
    mean: float
    std: float
    min: float
    max: float


def seed_stability(
    X: ArrayLike,
    y: ArrayLike,
    estimator: BaseEstimator,
    *,
    seeds: Iterable[int] = range(10),
    test_size: float = 0.2,
    metric: Callable[..., float] = accuracy_score,
    stratify: bool = True,
) -> SeedStabilityResult:
    """Measure metric stability across multiple random seeds.

    For each seed the estimator is cloned (and the seed forwarded to
    ``random_state`` when the estimator exposes one), then fitted on a
    fresh stratified train/test split derived from the seed. The
    resulting per-seed metric vector lets you compare expected
    performance with run-to-run variability.

    Parameters
    ----------
    X, y : array-like
    estimator : sklearn estimator
        Cloned once per seed.
    seeds : iterable of int, default range(10)
    test_size : float in (0, 1), default 0.2
    metric : callable, default :func:`sklearn.metrics.accuracy_score`
    stratify : bool, default True
        Stratify the train/test split by ``y``.

    Returns
    -------
    SeedStabilityResult
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    check_consistent_length(X_arr, y_arr)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}.")

    seed_list = np.asarray(list(seeds), dtype=np.int64)
    if seed_list.size == 0:
        raise ValueError("seeds must be non-empty.")

    scores = np.empty(seed_list.size, dtype=np.float64)
    has_random_state = "random_state" in estimator.get_params()

    for k, seed in enumerate(seed_list):
        seed_int = int(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=seed_int,
            stratify=y_arr if stratify else None,
        )
        est = clone(estimator)
        if has_random_state:
            est.set_params(random_state=seed_int)
        est.fit(X_tr, y_tr)
        scores[k] = float(metric(y_te, est.predict(X_te)))

    return SeedStabilityResult(
        seeds=seed_list,
        scores=scores,
        mean=float(scores.mean()),
        std=float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        min=float(scores.min()),
        max=float(scores.max()),
    )
