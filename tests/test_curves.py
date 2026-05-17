"""Tests for clinikit.curves."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from clinikit.curves import (
    CalibrationCurve,
    DecisionCurve,
    PrecisionRecallCurve,
    ROCCurve,
    calibration_curve_data,
    decision_curve_data,
    precision_recall_curve_data,
    roc_curve_data,
)


@pytest.fixture
def binary_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0, 1, size=300)
    return y, p


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def test_roc_curve_data_shapes_and_auc(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    curve = roc_curve_data(y, p)
    assert isinstance(curve, ROCCurve)
    assert curve.fpr.ndim == 1 and curve.tpr.ndim == 1
    assert curve.fpr.shape == curve.tpr.shape
    assert 0.0 <= curve.auc <= 1.0
    # Parity with sklearn.
    assert curve.auc == pytest.approx(roc_auc_score(y, p))


def test_roc_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    curve = roc_curve_data(y, p)
    assert curve.auc == pytest.approx(1.0)


def test_roc_curve_plot_returns_axes(binary_data) -> None:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    y, p = binary_data
    ax = roc_curve_data(y, p).plot()
    assert ax is not None


# ---------------------------------------------------------------------------
# Precision-recall
# ---------------------------------------------------------------------------


def test_pr_curve_data_shapes_and_ap(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    curve = precision_recall_curve_data(y, p)
    assert isinstance(curve, PrecisionRecallCurve)
    assert curve.precision.ndim == 1
    assert curve.recall.ndim == 1
    assert 0.0 <= curve.average_precision <= 1.0
    # Parity with sklearn.
    assert curve.average_precision == pytest.approx(average_precision_score(y, p))


def test_pr_curve_plot_returns_axes(binary_data) -> None:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    y, p = binary_data
    ax = precision_recall_curve_data(y, p).plot()
    assert ax is not None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_curve_data_basic(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    curve = calibration_curve_data(y, p, n_bins=10)
    assert isinstance(curve, CalibrationCurve)
    assert curve.mean_predicted.shape == curve.fraction_positive.shape
    assert curve.bin_counts.sum() == y.size
    assert curve.n_bins == 10


def test_calibration_curve_quantile_strategy(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    curve = calibration_curve_data(y, p, n_bins=5, strategy="quantile")
    assert curve.strategy == "quantile"
    assert curve.mean_predicted.size <= 5


def test_calibration_curve_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        calibration_curve_data([0, 1], [0.1, 0.9], strategy="bad")


def test_calibration_curve_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve_data([0, 1], [0.1, 0.9], n_bins=0)


def test_calibration_curve_plot_returns_axes(binary_data) -> None:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    y, p = binary_data
    ax = calibration_curve_data(y, p, n_bins=5).plot()
    assert ax is not None


# ---------------------------------------------------------------------------
# Decision curve
# ---------------------------------------------------------------------------


def test_decision_curve_data_basic(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    curve = decision_curve_data(y, p)
    assert isinstance(curve, DecisionCurve)
    assert curve.thresholds.shape == curve.net_benefit.shape
    assert curve.thresholds.shape == curve.treat_all_net_benefit.shape


def test_decision_curve_custom_thresholds() -> None:
    y = np.array([0, 0, 1, 1] * 25)
    p = np.linspace(0.0, 1.0, 100)
    curve = decision_curve_data(y, p, thresholds=[0.1, 0.3, 0.5, 0.7])
    assert curve.thresholds.size == 4


def test_decision_curve_thresholds_outside_range_rejected() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        decision_curve_data([0, 1], [0.1, 0.9], thresholds=[0.5, 1.0])


def test_decision_curve_plot_returns_axes(binary_data) -> None:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    y, p = binary_data
    ax = decision_curve_data(y, p).plot()
    assert ax is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_curves_reject_2d_y_prob() -> None:
    with pytest.raises(ValueError, match="1-D"):
        roc_curve_data([0, 1], [[0.1, 0.9], [0.2, 0.8]])


def test_curves_reject_out_of_range_prob() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        roc_curve_data([0, 1], [0.1, 1.5])


def test_curves_reject_multiclass() -> None:
    with pytest.raises(ValueError, match="binary-only"):
        roc_curve_data([0, 1, 2], [0.1, 0.5, 0.9])
