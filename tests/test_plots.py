"""Tests for clinikit.plots.

We don't compare pixels — we verify that:
- each helper returns a matplotlib Axes,
- both call shapes (curve dataclass or raw arrays) work,
- bad inputs raise.

Matplotlib's "Agg" backend is forced so the tests are headless.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinikit.curves import (
    calibration_curve_data,
    decision_curve_data,
    precision_recall_curve_data,
    roc_curve_data,
)
from clinikit.plots import (
    plot_calibration,
    plot_decision_curve,
    plot_pr,
    plot_roc,
    plot_threshold_sweep,
)


@pytest.fixture
def binary_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    return y, p


@pytest.fixture
def ax() -> object:
    _fig, ax = plt.subplots()
    yield ax
    plt.close(_fig)


# ---------------------------------------------------------------------------
# Both call shapes
# ---------------------------------------------------------------------------


def test_plot_roc_from_curve(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    result = plot_roc(roc_curve_data(y, p), ax=ax)
    assert result is ax


def test_plot_roc_from_raw(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    result = plot_roc(y, p, ax=ax)
    assert result is ax


def test_plot_roc_raw_requires_two_args(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, _ = binary_data
    with pytest.raises(ValueError, match="y_true, y_prob"):
        plot_roc(y, ax=ax)


def test_plot_pr_both_shapes(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    assert plot_pr(precision_recall_curve_data(y, p), ax=ax) is ax
    plt.close("all")
    _fig, ax2 = plt.subplots()
    assert plot_pr(y, p, ax=ax2) is ax2
    plt.close(_fig)


def test_plot_calibration_both_shapes(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    assert plot_calibration(calibration_curve_data(y, p, n_bins=5), ax=ax) is ax


def test_plot_decision_curve_both_shapes(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    assert plot_decision_curve(decision_curve_data(y, p), ax=ax) is ax


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def test_plot_threshold_sweep(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    result = plot_threshold_sweep(y, p, ax=ax, metrics=("sensitivity", "specificity"))
    assert result is ax


def test_plot_threshold_sweep_rejects_unknown_metric(binary_data, ax) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    with pytest.raises(ValueError, match="Unknown metric"):
        plot_threshold_sweep(y, p, ax=ax, metrics=("nonsense",))


# ---------------------------------------------------------------------------
# Default axis creation
# ---------------------------------------------------------------------------


def test_plot_roc_creates_axes_when_none(binary_data) -> None:  # type: ignore[no-untyped-def]
    y, p = binary_data
    result = plot_roc(y, p)  # no ax supplied
    assert result is not None
    plt.close("all")
