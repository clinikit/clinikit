"""Drift detection and performance monitoring.

- :func:`ks_drift`           — two-sample Kolmogorov-Smirnov test.
- :func:`wasserstein_drift`  — 1-Wasserstein distance.
- :func:`psi_drift`          — Population Stability Index.
- :func:`track_performance`  — rolling-window metric over a
  prediction/ground-truth log.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "DriftResult",
    "PerformanceTrack",
    "ks_drift",
    "psi_drift",
    "track_performance",
    "wasserstein_drift",
]


@dataclass(slots=True)
class DriftResult:
    """Output of every drift-detection helper.

    Attributes
    ----------
    statistic : float
        Test-statistic value (KS distance, Wasserstein distance, or PSI).
    p_value : float or None
        Two-sided p-value when the test has one (KS); ``None`` for
        distance-style measures.
    drift_detected : bool
        ``True`` when the statistic exceeds the configured threshold
        (or ``p_value`` falls below ``alpha`` for KS).
    threshold : float
        Threshold used to flag drift.
    test_name : str
    """

    statistic: float
    p_value: float | None
    drift_detected: bool
    threshold: float
    test_name: str


def _validate_pair(
    reference: ArrayLike,
    current: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Both reference and current must be non-empty.")
    return ref, cur


def ks_drift(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    alpha: float = 0.05,
) -> DriftResult:
    """Two-sample Kolmogorov-Smirnov drift test.

    Parameters
    ----------
    reference, current : array-like
        1-D samples from the two distributions being compared.
    alpha : float in (0, 1), default 0.05
        Significance level — ``drift_detected = p_value < alpha``.

    Returns
    -------
    DriftResult
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
    ref, cur = _validate_pair(reference, current)
    result = stats.ks_2samp(ref, cur)
    return DriftResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        drift_detected=float(result.pvalue) < alpha,
        threshold=alpha,
        test_name="ks_2samp",
    )


def wasserstein_drift(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    threshold: float = 0.1,
) -> DriftResult:
    """1-Wasserstein distance between two 1-D samples.

    Parameters
    ----------
    reference, current : array-like
    threshold : float >= 0, default 0.1
        ``drift_detected`` is ``True`` when the distance exceeds this
        value.

    Returns
    -------
    DriftResult
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0; got {threshold!r}.")
    ref, cur = _validate_pair(reference, current)
    distance = float(stats.wasserstein_distance(ref, cur))
    return DriftResult(
        statistic=distance,
        p_value=None,
        drift_detected=distance > threshold,
        threshold=threshold,
        test_name="wasserstein_distance",
    )


def psi_drift(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    n_bins: int = 10,
    threshold: float = 0.25,
    epsilon: float = 1e-6,
) -> DriftResult:
    """Population Stability Index.

    PSI bins the reference distribution into ``n_bins`` equal-count
    buckets and measures::

        PSI = sum_b (p_curr_b - p_ref_b) * log(p_curr_b / p_ref_b)

    Conventional cutoffs (Karakoulas 2004): PSI < 0.1 = no shift,
    0.1-0.25 = moderate, > 0.25 = significant. ``threshold`` defaults
    to the conservative 0.25.

    Parameters
    ----------
    reference, current : array-like
    n_bins : int, default 10
    threshold : float >= 0, default 0.25
    epsilon : float > 0, default 1e-6
        Floor applied to empty bins to keep ``log`` finite.

    Returns
    -------
    DriftResult
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins!r}.")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0; got {threshold!r}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon!r}.")
    ref, cur = _validate_pair(reference, current)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    # Squash trailing collisions so np.searchsorted has strictly
    # increasing internal edges.
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError(
            "Reference distribution has too few unique values for the requested n_bins."
        )

    def _proportions(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        bin_idx = np.clip(np.searchsorted(edges[1:-1], arr, side="right"), 0, len(edges) - 2)
        counts = np.bincount(bin_idx, minlength=len(edges) - 1)
        return np.asarray(np.maximum(counts / counts.sum(), epsilon), dtype=np.float64)

    p_ref = _proportions(ref)
    p_cur = _proportions(cur)
    psi = float(((p_cur - p_ref) * np.log(p_cur / p_ref)).sum())
    return DriftResult(
        statistic=psi,
        p_value=None,
        drift_detected=psi > threshold,
        threshold=threshold,
        test_name="psi",
    )


# ---------------------------------------------------------------------------
# Rolling-window performance tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PerformanceTrack:
    """Output of :func:`track_performance`."""

    window_size: int
    metric_name: str
    window_scores: NDArray[np.float64]
    rolling_mean: NDArray[np.float64]


def track_performance(
    predictions: ArrayLike,
    ground_truth: ArrayLike,
    *,
    window: int = 100,
    metric: Callable[..., float] = accuracy_score,
    metric_name: str | None = None,
) -> PerformanceTrack:
    """Compute a rolling metric across a prediction log.

    Parameters
    ----------
    predictions, ground_truth : array-like of shape (n_samples,)
        Logs aligned by time.
    window : int, default 100
        Window size in samples. Must be ``>= 2`` and ``<=`` log
        length.
    metric : callable, default :func:`sklearn.metrics.accuracy_score`
        ``metric(y_true, y_pred) -> float``.
    metric_name : str, optional
        Display name. Defaults to ``metric.__name__``.

    Returns
    -------
    PerformanceTrack
    """
    pred = np.asarray(predictions)
    truth = np.asarray(ground_truth)
    check_consistent_length(pred, truth)
    n = pred.shape[0]
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window!r}.")
    if window > n:
        raise ValueError(f"window={window!r} exceeds log length {n}.")

    n_windows = n - window + 1
    scores = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        scores[i] = float(metric(truth[i : i + window], pred[i : i + window]))

    rolling_mean = np.cumsum(scores) / (np.arange(n_windows) + 1)
    resolved_name: str = metric_name or str(getattr(metric, "__name__", "metric"))
    return PerformanceTrack(
        window_size=window,
        metric_name=resolved_name,
        window_scores=scores,
        rolling_mean=rolling_mean,
    )
