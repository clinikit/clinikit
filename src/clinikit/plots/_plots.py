"""Matplotlib helpers for the diagnostic curves and threshold sweeps.

Each function accepts either a fitted :mod:`clinikit.curves` dataclass
or raw ``(y_true, y_prob)`` arrays and returns the
:class:`matplotlib.axes.Axes` it drew into. The user owns the figure
(create with ``plt.subplots`` or pass an existing ``ax``); no
``plt.show()`` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from clinikit.curves import (
        CalibrationCurve,
        DecisionCurve,
        PrecisionRecallCurve,
        ROCCurve,
    )

__all__ = [
    "plot_calibration",
    "plot_decision_curve",
    "plot_pr",
    "plot_roc",
    "plot_threshold_sweep",
]


def _resolve_ax(ax: Axes | None) -> Axes:
    if ax is None:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots(figsize=(6, 5))
    return ax


def plot_roc(
    curve_or_y_true: ROCCurve | ArrayLike,
    y_prob: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    show_diagonal: bool = True,
) -> Axes:
    """Plot a ROC curve.

    Pass either a :class:`~clinikit.curves.ROCCurve` (positional) or
    raw ``(y_true, y_prob)`` arrays.
    """
    from clinikit.curves import ROCCurve, roc_curve_data

    ax = _resolve_ax(ax)

    if isinstance(curve_or_y_true, ROCCurve):
        curve = curve_or_y_true
    else:
        if y_prob is None:
            raise ValueError("Pass either a ROCCurve or (y_true, y_prob).")
        curve = roc_curve_data(curve_or_y_true, y_prob)

    auc_label = f" (AUC = {curve.auc:.3f})"
    ax.plot(
        curve.fpr,
        curve.tpr,
        label=(label or "ROC") + auc_label,
    )
    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("Receiver operating characteristic")
    return ax


def plot_pr(
    curve_or_y_true: PrecisionRecallCurve | ArrayLike,
    y_prob: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot a precision-recall curve."""
    from clinikit.curves import PrecisionRecallCurve, precision_recall_curve_data

    ax = _resolve_ax(ax)

    if isinstance(curve_or_y_true, PrecisionRecallCurve):
        curve = curve_or_y_true
    else:
        if y_prob is None:
            raise ValueError("Pass either a PrecisionRecallCurve or (y_true, y_prob).")
        curve = precision_recall_curve_data(curve_or_y_true, y_prob)

    ap_label = f" (AP = {curve.average_precision:.3f})"
    ax.plot(curve.recall, curve.precision, label=(label or "PR") + ap_label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Precision-recall curve")
    return ax


def plot_calibration(
    curve_or_y_true: CalibrationCurve | ArrayLike,
    y_prob: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    label: str | None = None,
) -> Axes:
    """Plot a reliability diagram."""
    from clinikit.curves import CalibrationCurve, calibration_curve_data

    ax = _resolve_ax(ax)

    if isinstance(curve_or_y_true, CalibrationCurve):
        curve = curve_or_y_true
    else:
        if y_prob is None:
            raise ValueError("Pass either a CalibrationCurve or (y_true, y_prob).")
        curve = calibration_curve_data(curve_or_y_true, y_prob, n_bins=n_bins, strategy=strategy)

    ax.plot(
        curve.mean_predicted,
        curve.fraction_positive,
        marker="o",
        label=label or "model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper left")
    ax.set_title(f"Reliability diagram ({curve.strategy} bins)")
    return ax


def plot_decision_curve(
    curve_or_y_true: DecisionCurve | ArrayLike,
    y_prob: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    show_treat_all: bool = True,
    show_treat_none: bool = True,
) -> Axes:
    """Plot Decision Curve Analysis (Vickers & Elkin 2006)."""
    from clinikit.curves import DecisionCurve, decision_curve_data

    ax = _resolve_ax(ax)

    if isinstance(curve_or_y_true, DecisionCurve):
        curve = curve_or_y_true
    else:
        if y_prob is None:
            raise ValueError("Pass either a DecisionCurve or (y_true, y_prob).")
        curve = decision_curve_data(curve_or_y_true, y_prob)

    ax.plot(curve.thresholds, curve.net_benefit, label=label or "model")
    if show_treat_all:
        ax.plot(
            curve.thresholds,
            curve.treat_all_net_benefit,
            linestyle="--",
            color="orange",
            label="treat all",
        )
    if show_treat_none:
        ax.axhline(0.0, linestyle="--", color="gray", label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(curve.thresholds.min(), curve.thresholds.max())
    ax.legend(loc="upper right")
    ax.set_title("Decision curve analysis")
    return ax


def plot_threshold_sweep(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    ax: Axes | None = None,
    metrics: tuple[str, ...] = ("sensitivity", "specificity", "ppv"),
    n_points: int = 99,
) -> Axes:
    """Plot how the requested metrics behave as the threshold sweeps ``(0, 1)``.

    Useful for choosing operating points visually.
    """
    from clinikit.metrics import (
        npv as _npv,
    )
    from clinikit.metrics import (
        ppv as _ppv,
    )
    from clinikit.metrics import (
        sensitivity as _sens,
    )
    from clinikit.metrics import (
        specificity as _spec,
    )

    metric_table: dict[str, Any] = {
        "sensitivity": _sens,
        "specificity": _spec,
        "ppv": _ppv,
        "npv": _npv,
    }
    unknown = [m for m in metrics if m not in metric_table]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown!r}; available: {sorted(metric_table)}.")

    ax = _resolve_ax(ax)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.linspace(0.01, 0.99, n_points)

    for name in metrics:
        fn = metric_table[name]
        values = np.array([fn(y_true_arr, (y_prob_arr >= t).astype(int)) for t in thresholds])
        ax.plot(thresholds, values, label=name)

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best")
    ax.set_title("Threshold sweep")
    return ax
