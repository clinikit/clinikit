"""Threshold-selection strategies for binary classification.

Each strategy maps ground-truth labels and predicted probabilities to a
single threshold (or per-cluster mapping). The :func:`pick_threshold`
dispatcher in :mod:`clinikit.thresholds` routes by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_consistent_length

from clinikit.metrics import sensitivity, specificity

__all__ = [
    "accuracy_constrained",
    "accuracy_max",
    "cluster_specific",
    "evaluate_threshold",
    "recall_constrained",
    "two_stage",
]

_GRID_SIZE = 256


def _candidate_grid(y_prob: NDArray[np.float64]) -> NDArray[np.float64]:
    """Union of the unique data points and a uniform grid on ``[0, 1]``.

    The uniform grid guarantees we always have some candidate even
    when ``y_prob`` is degenerate (constant, or only takes a few
    values).
    """
    uniform = np.linspace(0.0, 1.0, _GRID_SIZE)
    data = np.unique(y_prob)
    return np.unique(np.concatenate([uniform, data]))


def _validate_inputs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    check_consistent_length(y_true_arr, y_prob_arr)
    if y_prob_arr.ndim != 1:
        raise ValueError(
            f"y_prob must be 1-D probabilities for the positive class; "
            f"got shape {y_prob_arr.shape!r}."
        )
    if np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("y_prob must lie in [0, 1].")
    # Coerce labels to int but accept any binary representation.
    unique = np.unique(y_true_arr)
    if len(unique) > 2:
        raise ValueError(f"Threshold strategies are binary-only; got {len(unique)} unique labels.")
    pos_label = unique[-1] if len(unique) > 0 else 1
    y_binary = (y_true_arr == pos_label).astype(np.int64)
    return y_binary, y_prob_arr


def accuracy_max(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> float:
    """Pick the threshold that maximises plain accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of the positive class.

    Returns
    -------
    float
        Selected threshold in ``[0, 1]``. Ties are broken by choosing
        the smallest threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.thresholds import accuracy_max
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, size=200)
    >>> p = rng.uniform(0, 1, size=200)
    >>> 0.0 <= accuracy_max(y, p) <= 1.0
    True
    """
    y_binary, p = _validate_inputs(y_true, y_prob)
    grid = _candidate_grid(p)
    accs = np.array([accuracy_score(y_binary, (p >= t).astype(np.int64)) for t in grid])
    return float(grid[int(np.argmax(accs))])


def recall_constrained(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    min_recall: float = 0.9,
) -> float:
    """Pick the threshold that maximises accuracy subject to a recall floor.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    min_recall : float in [0, 1], default 0.9
        Lower bound on sensitivity (true positive rate).

    Returns
    -------
    float
        The highest threshold that still achieves ``recall >= min_recall``
        while maximising accuracy. Falls back to ``0.0`` if no
        threshold satisfies the constraint.
    """
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError(f"min_recall must be in [0, 1]; got {min_recall!r}.")
    y_binary, p = _validate_inputs(y_true, y_prob)
    grid = _candidate_grid(p)

    best_t = 0.0
    best_acc = -1.0
    for t in grid:
        pred = (p >= t).astype(np.int64)
        if sensitivity(y_binary, pred) < min_recall:
            continue
        acc = accuracy_score(y_binary, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def accuracy_constrained(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    min_accuracy: float = 0.7,
) -> float:
    """Pick the threshold that maximises recall subject to an accuracy floor.

    The mirror of :func:`recall_constrained`. Falls back to ``0.5`` if
    no threshold satisfies the constraint.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    min_accuracy : float in [0, 1], default 0.7
        Lower bound on overall accuracy.
    """
    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError(f"min_accuracy must be in [0, 1]; got {min_accuracy!r}.")
    y_binary, p = _validate_inputs(y_true, y_prob)
    grid = _candidate_grid(p)

    best_t = 0.5
    best_rec = -1.0
    for t in grid:
        pred = (p >= t).astype(np.int64)
        if accuracy_score(y_binary, pred) < min_accuracy:
            continue
        rec = sensitivity(y_binary, pred)
        if rec > best_rec:
            best_rec = rec
            best_t = float(t)
    return best_t


def cluster_specific(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    cluster_labels: ArrayLike,
    *,
    min_cluster_size: int = 5,
    default_threshold: float = 0.5,
) -> dict[int, float]:
    """Pick one threshold per cluster, maximising per-cluster accuracy.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    cluster_labels : array-like of shape (n_samples,)
        Integer cluster assignment per sample.
    min_cluster_size : int, default 5
        Clusters smaller than this fall back to ``default_threshold``
        (avoids overfitting on degenerate slices).
    default_threshold : float, default 0.5
        Used for tiny or single-class clusters.

    Returns
    -------
    dict[int, float]
        ``{cluster_id: threshold}``.
    """
    y_binary, p = _validate_inputs(y_true, y_prob)
    labels = np.asarray(cluster_labels)
    check_consistent_length(y_binary, labels)

    out: dict[int, float] = {}
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() < min_cluster_size or len(np.unique(y_binary[mask])) < 2:
            out[int(c)] = float(default_threshold)
            continue
        out[int(c)] = accuracy_max(y_binary[mask], p[mask])
    return out


def two_stage(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    min_specificity: float = 0.95,
    min_sensitivity: float = 0.95,
) -> tuple[float, float]:
    """Pick two thresholds for a high-confidence / high-confidence pair.

    Returns ``(t_low, t_high)`` such that:

    - predictions with ``y_prob >= t_high`` reach the ``min_specificity``
      target when calling them positive,
    - predictions with ``y_prob <= t_low`` reach the ``min_sensitivity``
      target when calling them negative,
    - everything in between is treated as "abstain" downstream.

    If the constraints cannot be met, the corresponding bound is
    snapped to ``1.0`` (no positives accepted) or ``0.0`` (no negatives
    accepted).

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    min_specificity : float in [0, 1], default 0.95
    min_sensitivity : float in [0, 1], default 0.95

    Returns
    -------
    (t_low, t_high) : tuple[float, float]
        With ``t_low <= t_high``.
    """
    for name, val in [
        ("min_specificity", min_specificity),
        ("min_sensitivity", min_sensitivity),
    ]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {val!r}.")

    y_binary, p = _validate_inputs(y_true, y_prob)
    grid = _candidate_grid(p)

    # t_high: smallest threshold such that specificity at this threshold meets the floor.
    t_high = 1.0
    for t in grid:
        pred = (p >= t).astype(np.int64)
        if specificity(y_binary, pred) >= min_specificity:
            t_high = float(t)
            break

    # t_low: largest threshold such that sensitivity at this threshold meets the floor.
    t_low = 0.0
    for t in grid[::-1]:
        pred = (p >= t).astype(np.int64)
        if sensitivity(y_binary, pred) >= min_sensitivity:
            t_low = float(t)
            break

    if t_low > t_high:
        # Infeasible constraints — collapse to a single point in the middle.
        mid = 0.5 * (t_low + t_high)
        t_low = t_high = mid
    return t_low, t_high


def evaluate_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Compute headline metrics at a given threshold.

    Returns a dict with ``accuracy``, ``sensitivity``, ``specificity``,
    and ``positive_rate`` (the fraction of samples classified positive).

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    threshold : float in [0, 1]

    Returns
    -------
    dict[str, float]

    Examples
    --------
    >>> from clinikit.thresholds import evaluate_threshold
    >>> y = [0, 0, 1, 1]
    >>> p = [0.1, 0.4, 0.6, 0.9]
    >>> out = evaluate_threshold(y, p, 0.5)
    >>> sorted(out.keys())
    ['accuracy', 'positive_rate', 'sensitivity', 'specificity']
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}.")
    y_binary, p = _validate_inputs(y_true, y_prob)
    pred = (p >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_binary, pred)),
        "sensitivity": sensitivity(y_binary, pred),
        "specificity": specificity(y_binary, pred),
        "positive_rate": float(pred.mean()),
    }


def pick_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    strategy: str = "accuracy_max",
    **kwargs: Any,
) -> float | tuple[float, float] | dict[int, float]:
    """Dispatch to the named threshold strategy.

    See module docstring for the available strategies and their
    extra keyword arguments.
    """
    table = {
        "accuracy_max": accuracy_max,
        "recall_constrained": recall_constrained,
        "accuracy_constrained": accuracy_constrained,
        "cluster_specific": cluster_specific,
        "two_stage": two_stage,
    }
    if strategy not in table:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {sorted(table)}.")
    result: float | tuple[float, float] | dict[int, float] = table[strategy](  # type: ignore[operator]
        y_true, y_prob, **kwargs
    )
    return result
