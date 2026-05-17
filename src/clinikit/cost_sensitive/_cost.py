"""Cost-sensitive utilities for binary classification.

Three classical helpers tied together by a simple
:math:`\\text{cost}(\\text{FP}, \\text{FN}) = c_{FP} \\cdot \\text{FP} + c_{FN} \\cdot \\text{FN}` model:

- :func:`weighted_loss`           — total weighted misclassification cost.
- :func:`bayes_optimal_threshold` — analytical decision threshold under
  asymmetric costs (Elkan 2001).
- :func:`cost_curve`              — total cost across a threshold grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "CostCurveResult",
    "bayes_optimal_threshold",
    "cost_curve",
    "weighted_loss",
]


def _binary_targets(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    check_consistent_length(y_true_arr, y_pred_arr)
    unique = set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
    if len(unique) > 2:
        raise ValueError(
            f"Cost-sensitive utilities are binary-only; got labels {sorted(unique)!r}."
        )
    return y_true_arr.astype(np.int64), y_pred_arr.astype(np.int64)


def weighted_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    pos_label: int | str | bool = 1,
) -> float:
    """Total weighted misclassification cost.

    ``loss = cost_fp * FP + cost_fn * FN``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Hard predictions (not probabilities).
    cost_fp : float, default 1.0
        Cost charged for each false positive.
    cost_fn : float, default 1.0
        Cost charged for each false negative.
    pos_label : default 1
        Which label counts as positive.

    Returns
    -------
    float
        Non-negative total cost.

    Examples
    --------
    >>> from clinikit.cost_sensitive import weighted_loss
    >>> weighted_loss([0, 1, 0, 1], [1, 1, 0, 0], cost_fp=2.0, cost_fn=5.0)
    7.0
    """
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError(
            f"costs must be non-negative; got cost_fp={cost_fp!r}, cost_fn={cost_fn!r}."
        )
    y_true_arr, y_pred_arr = _binary_targets(y_true, y_pred)
    y_true_pos = y_true_arr == pos_label
    y_pred_pos = y_pred_arr == pos_label
    fp = int(((~y_true_pos) & y_pred_pos).sum())
    fn = int((y_true_pos & (~y_pred_pos)).sum())
    return float(cost_fp * fp + cost_fn * fn)


def bayes_optimal_threshold(
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """Decision-theoretic optimal threshold under asymmetric costs.

    Assuming zero cost for correct classifications and additive costs
    for the two error types, the threshold that minimises expected
    cost is::

        threshold* = cost_fp / (cost_fp + cost_fn)

    Symmetric costs (the default) recover the conventional ``0.5``
    threshold; raising ``cost_fn`` lowers the threshold (predict
    positive more often, false negatives are expensive); raising
    ``cost_fp`` raises it.

    Parameters
    ----------
    cost_fp : float, default 1.0
    cost_fn : float, default 1.0

    Returns
    -------
    float
        Threshold in ``[0, 1]``.

    Examples
    --------
    >>> from clinikit.cost_sensitive import bayes_optimal_threshold
    >>> bayes_optimal_threshold()
    0.5
    >>> bayes_optimal_threshold(cost_fp=1.0, cost_fn=4.0)
    0.2
    """
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError(
            f"costs must be non-negative; got cost_fp={cost_fp!r}, cost_fn={cost_fn!r}."
        )
    total = cost_fp + cost_fn
    if total == 0:
        # Degenerate "every prediction is free" case — collapse to 0.5
        # so the caller still gets a usable cutoff.
        return 0.5
    return float(cost_fp / total)


@dataclass(slots=True)
class CostCurveResult:
    """Output of :func:`cost_curve`."""

    thresholds: NDArray[np.float64]
    total_cost: NDArray[np.float64]
    false_positives: NDArray[np.int64]
    false_negatives: NDArray[np.int64]

    @property
    def optimal_threshold(self) -> float:
        """Threshold achieving the minimum cost on the supplied grid."""
        return float(self.thresholds[int(np.argmin(self.total_cost))])

    @property
    def minimum_cost(self) -> float:
        return float(self.total_cost.min())


def cost_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    thresholds: ArrayLike | None = None,
    pos_label: int | str | bool = 1,
) -> CostCurveResult:
    """Sweep thresholds and report the total weighted cost at each one.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    cost_fp, cost_fn : float, default 1.0
        Per-error costs.
    thresholds : array-like, optional
        Thresholds to evaluate. Defaults to ``np.linspace(0.01, 0.99, 99)``.
    pos_label : default 1

    Returns
    -------
    CostCurveResult
        ``.optimal_threshold`` returns the grid point with the lowest
        total cost.
    """
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError(
            f"costs must be non-negative; got cost_fp={cost_fp!r}, cost_fn={cost_fn!r}."
        )
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    check_consistent_length(y_true_arr, y_prob_arr)
    if y_prob_arr.ndim != 1:
        raise ValueError(f"y_prob must be 1-D; got shape {y_prob_arr.shape!r}.")
    if np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("y_prob must lie in [0, 1].")

    if thresholds is None:
        thresholds_arr = np.linspace(0.01, 0.99, 99)
    else:
        thresholds_arr = np.asarray(thresholds, dtype=np.float64)
        if np.any((thresholds_arr < 0) | (thresholds_arr > 1)):
            raise ValueError("thresholds must lie in [0, 1].")

    y_true_pos = y_true_arr == pos_label
    fps = np.zeros(thresholds_arr.size, dtype=np.int64)
    fns = np.zeros(thresholds_arr.size, dtype=np.int64)
    total = np.zeros(thresholds_arr.size, dtype=np.float64)
    for i, t in enumerate(thresholds_arr):
        pred_pos = y_prob_arr >= t
        fp = int(((~y_true_pos) & pred_pos).sum())
        fn = int((y_true_pos & (~pred_pos)).sum())
        fps[i] = fp
        fns[i] = fn
        total[i] = cost_fp * fp + cost_fn * fn

    return CostCurveResult(
        thresholds=thresholds_arr,
        total_cost=total,
        false_positives=fps,
        false_negatives=fns,
    )
