"""Diagnostic curves for binary classifiers.

Four canonical analyses, each returning a small dataclass with the
underlying numeric arrays plus a ``plot()`` helper that delegates to
``clinikit.plots``:

- :func:`roc_curve_data` — ROC curve and AUC.
- :func:`precision_recall_curve_data` — PR curve and average precision.
- :func:`calibration_curve_data` — reliability bins.
- :func:`decision_curve_data` — net benefit per threshold
  (Vickers & Elkin 2006).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as _sk_auc,
)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.utils.validation import check_consistent_length

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "CalibrationCurve",
    "DecisionCurve",
    "PrecisionRecallCurve",
    "ROCCurve",
    "calibration_curve_data",
    "decision_curve_data",
    "precision_recall_curve_data",
    "roc_curve_data",
]


def _validate_binary(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
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
    if len(np.unique(y_true_arr)) > 2:
        raise ValueError("curves are binary-only; got >2 unique labels in y_true.")
    y_binary = (y_true_arr == pos_label).astype(np.int64)
    return y_binary, y_prob_arr


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ROCCurve:
    """Receiver-operating-characteristic curve plus AUC.

    Attributes
    ----------
    fpr : ndarray
        False-positive rate at each threshold.
    tpr : ndarray
        True-positive rate at each threshold.
    thresholds : ndarray
        Decreasing list of probability thresholds.
    auc : float
        Area under the ROC curve in ``[0, 1]``.
    """

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    auc: float

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        from clinikit.plots import plot_roc

        return plot_roc(self, ax=ax, **kwargs)


def roc_curve_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> ROCCurve:
    """Compute the ROC curve and AUC.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    pos_label : default 1
    sample_weight : array-like, optional

    Returns
    -------
    ROCCurve
    """
    y_binary, p = _validate_binary(y_true, y_prob, pos_label=pos_label)
    fpr, tpr, thresholds = roc_curve(y_binary, p, pos_label=1, sample_weight=sample_weight)
    return ROCCurve(
        fpr=np.asarray(fpr, dtype=np.float64),
        tpr=np.asarray(tpr, dtype=np.float64),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        auc=float(_sk_auc(fpr, tpr)),
    )


# ---------------------------------------------------------------------------
# Precision-Recall
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PrecisionRecallCurve:
    """Precision-recall curve plus average precision.

    Attributes
    ----------
    precision : ndarray
    recall : ndarray
    thresholds : ndarray
    average_precision : float
    """

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    average_precision: float

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        from clinikit.plots import plot_pr

        return plot_pr(self, ax=ax, **kwargs)


def precision_recall_curve_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> PrecisionRecallCurve:
    """Compute the precision-recall curve and average precision."""
    y_binary, p = _validate_binary(y_true, y_prob, pos_label=pos_label)
    precision, recall, thresholds = precision_recall_curve(
        y_binary, p, pos_label=1, sample_weight=sample_weight
    )
    ap = float(average_precision_score(y_binary, p, pos_label=1, sample_weight=sample_weight))
    return PrecisionRecallCurve(
        precision=np.asarray(precision, dtype=np.float64),
        recall=np.asarray(recall, dtype=np.float64),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        average_precision=ap,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CalibrationCurve:
    """Reliability diagram data.

    Attributes
    ----------
    mean_predicted : ndarray of shape (n_bins,)
        Average predicted probability per bin.
    fraction_positive : ndarray of shape (n_bins,)
        Empirical positive rate per bin.
    bin_counts : ndarray of shape (n_bins,)
        Number of samples in each bin (informative — sklearn's
        ``calibration_curve`` does not return this).
    n_bins : int
    strategy : str
        ``"uniform"`` or ``"quantile"``.
    """

    mean_predicted: NDArray[np.float64]
    fraction_positive: NDArray[np.float64]
    bin_counts: NDArray[np.int64]
    n_bins: int
    strategy: str

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        from clinikit.plots import plot_calibration

        return plot_calibration(self, ax=ax, **kwargs)


def calibration_curve_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    pos_label: int | str | bool = 1,
) -> CalibrationCurve:
    """Compute reliability-diagram bins.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    n_bins : int, default 10
    strategy : {"uniform", "quantile"}, default "uniform"
    pos_label : default 1

    Returns
    -------
    CalibrationCurve
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins!r}.")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile'; got {strategy!r}.")
    y_binary, p = _validate_binary(y_true, y_prob, pos_label=pos_label)

    fraction_pos, mean_pred = calibration_curve(y_binary, p, n_bins=n_bins, strategy=strategy)

    # Re-derive per-bin counts (sklearn drops empty bins).
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    bin_ids = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1)
    # Keep only the bins sklearn kept (the ones with samples).
    kept = counts[counts > 0]

    return CalibrationCurve(
        mean_predicted=np.asarray(mean_pred, dtype=np.float64),
        fraction_positive=np.asarray(fraction_pos, dtype=np.float64),
        bin_counts=np.asarray(kept, dtype=np.int64),
        n_bins=n_bins,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Decision Curve Analysis (Vickers & Elkin 2006)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecisionCurve:
    """Decision-curve-analysis output.

    Attributes
    ----------
    thresholds : ndarray of shape (n_thresholds,)
        Threshold probabilities :math:`p_t`.
    net_benefit : ndarray of shape (n_thresholds,)
        Model's net benefit at each threshold.
    treat_all_net_benefit : ndarray of shape (n_thresholds,)
        Net benefit of the "treat everyone" strategy (the model must
        beat this curve to be useful).
    """

    thresholds: NDArray[np.float64]
    net_benefit: NDArray[np.float64]
    treat_all_net_benefit: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        from clinikit.plots import plot_decision_curve

        return plot_decision_curve(self, ax=ax, **kwargs)


def decision_curve_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
    pos_label: int | str | bool = 1,
) -> DecisionCurve:
    """Compute net benefit per threshold (Vickers & Elkin 2006).

    The net benefit at threshold :math:`p_t` is

    .. math:: \\text{NB}(p_t) = \\frac{TP}{N} - \\frac{FP}{N} \\cdot \\frac{p_t}{1-p_t}

    where TP and FP are computed by classifying as positive when
    ``y_prob >= p_t``.

    Parameters
    ----------
    y_true, y_prob : array-like of shape (n_samples,)
    thresholds : array-like, optional
        Probabilities to evaluate. Defaults to ``np.linspace(0.01, 0.99, 99)``.
    pos_label : default 1

    Returns
    -------
    DecisionCurve
    """
    y_binary, p = _validate_binary(y_true, y_prob, pos_label=pos_label)
    if thresholds is None:
        thresholds_arr = np.linspace(0.01, 0.99, 99)
    else:
        thresholds_arr = np.asarray(thresholds, dtype=np.float64)
        if np.any((thresholds_arr <= 0) | (thresholds_arr >= 1)):
            raise ValueError("thresholds must lie strictly inside (0, 1).")

    n = float(y_binary.shape[0])
    prevalence = float(y_binary.mean())

    net_benefit = np.zeros_like(thresholds_arr)
    treat_all = np.zeros_like(thresholds_arr)
    for i, t in enumerate(thresholds_arr):
        pred = p >= t
        tp = float(((y_binary == 1) & pred).sum())
        fp = float(((y_binary == 0) & pred).sum())
        ratio = t / (1.0 - t)
        net_benefit[i] = tp / n - (fp / n) * ratio
        treat_all[i] = prevalence - (1.0 - prevalence) * ratio

    return DecisionCurve(
        thresholds=thresholds_arr,
        net_benefit=net_benefit,
        treat_all_net_benefit=treat_all,
    )
