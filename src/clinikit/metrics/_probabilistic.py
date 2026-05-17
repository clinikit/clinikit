"""Probability-based metrics.

These metrics consume predicted probabilities for the positive class
rather than hard predictions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "brier_score",
    "expected_calibration_error",
]

BinStrategy = Literal["uniform", "quantile"]


def _validate_binary_targets(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Coerce inputs to a binary indicator vector + probability vector."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    check_consistent_length(y_true_arr, y_prob_arr)

    if y_prob_arr.ndim != 1:
        raise ValueError(
            f"y_prob must be 1-D probabilities for the positive class; "
            f"got shape {y_prob_arr.shape!r}. For multiclass calibration "
            "use sklearn.metrics directly."
        )
    if np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("y_prob must lie in [0, 1].")

    y_binary = (y_true_arr == pos_label).astype(np.int64)
    return y_binary, y_prob_arr


def brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Brier score: mean squared error between probabilities and labels.

    ``Brier = mean((y_prob - y_binary)^2)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    pos_label : int, str, or bool, default 1
        Which class to treat as the positive one.
    sample_weight : array-like of shape (n_samples,), optional
        Per-sample weights.

    Returns
    -------
    float
        Brier score in ``[0, 1]``; lower is better.

    Examples
    --------
    >>> from clinikit.metrics import brier_score
    >>> round(brier_score([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]), 4)
    0.025
    """
    y_binary, p = _validate_binary_targets(y_true, y_prob, pos_label=pos_label)
    sq_err = (p - y_binary) ** 2
    if sample_weight is None:
        return float(np.mean(sq_err))
    w = np.asarray(sample_weight, dtype=np.float64)
    check_consistent_length(y_binary, w)
    return float(np.average(sq_err, weights=w))


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Expected Calibration Error (ECE).

    ECE is the weighted average of the absolute difference between
    average predicted probability and empirical accuracy across bins::

        ECE = sum_b (|bin_b| / N) * |acc(bin_b) - conf(bin_b)|

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of bins. Must be ``>= 1``.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy. ``"uniform"`` uses equal-width bins over
        ``[0, 1]``; ``"quantile"`` uses equal-count bins.
    pos_label : int, str, or bool, default 1
    sample_weight : array-like, optional

    Returns
    -------
    float
        ECE in ``[0, 1]``; lower is better.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.metrics import expected_calibration_error
    >>> rng = np.random.default_rng(0)
    >>> y_true = rng.integers(0, 2, size=200)
    >>> y_prob = rng.uniform(0, 1, size=200)
    >>> 0.0 <= expected_calibration_error(y_true, y_prob, n_bins=10) <= 1.0
    True
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    y_binary, p = _validate_binary_targets(y_true, y_prob, pos_label=pos_label)

    if sample_weight is None:
        w = np.ones_like(p, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        check_consistent_length(y_binary, w)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, quantiles)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}.")

    bin_ids = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)

    total_weight = float(np.sum(w))
    if total_weight == 0.0:
        return 0.0

    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_weight = float(np.sum(w[mask]))
        if bin_weight == 0.0:
            continue
        avg_conf = float(np.average(p[mask], weights=w[mask]))
        avg_acc = float(np.average(y_binary[mask], weights=w[mask]))
        ece += (bin_weight / total_weight) * abs(avg_conf - avg_acc)

    return ece
