"""Tests for clinikit.monitor."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from clinikit.monitor import (
    DriftResult,
    PerformanceTrack,
    ks_drift,
    psi_drift,
    track_performance,
    wasserstein_drift,
)

# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------


def test_ks_drift_same_distribution_no_drift() -> None:
    # Fixed local seed so the test is deterministic on every platform.
    # KS p-value is uniformly distributed under H0, so a session-scoped
    # rng plus differing test orders on Windows used to flip this test
    # at ~5% rate; an explicit seed pins ref/cur to a sample we know
    # KS classifies as non-drifting.
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(500)
    cur = rng.standard_normal(500)
    result = ks_drift(ref, cur)
    assert isinstance(result, DriftResult)
    assert result.test_name == "ks_2samp"
    assert result.drift_detected is False


def test_ks_drift_shifted_distribution_detected() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(500)
    cur = rng.standard_normal(500) + 2.0  # mean shift
    result = ks_drift(ref, cur)
    assert result.drift_detected is True
    assert result.p_value < 0.05


def test_ks_drift_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ks_drift([0, 1], [0, 1], alpha=1.5)


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------


def test_wasserstein_drift_zero_for_identical() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    result = wasserstein_drift(ref, ref)
    assert result.statistic == pytest.approx(0.0)
    assert result.drift_detected is False


def test_wasserstein_drift_shift_increases_distance() -> None:
    ref = np.linspace(0, 1, 100)
    cur = ref + 5.0
    result = wasserstein_drift(ref, cur, threshold=0.1)
    assert result.statistic == pytest.approx(5.0)
    assert result.drift_detected is True


def test_wasserstein_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        wasserstein_drift([0, 1], [0, 1], threshold=-0.1)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def test_psi_no_shift_is_small(rng: np.random.Generator) -> None:
    ref = rng.standard_normal(1000)
    cur = rng.standard_normal(1000)
    result = psi_drift(ref, cur, n_bins=10)
    assert result.statistic < 0.1
    assert result.drift_detected is False


def test_psi_large_shift_detected() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(500)
    cur = rng.standard_normal(500) + 3.0
    result = psi_drift(ref, cur)
    assert result.statistic > 0.25
    assert result.drift_detected is True


def test_psi_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        psi_drift([0, 1, 2], [0, 1, 2], n_bins=1)


def test_psi_too_few_unique_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        psi_drift([5, 5, 5, 5], [5, 5, 5, 5], n_bins=4)


# ---------------------------------------------------------------------------
# Track performance
# ---------------------------------------------------------------------------


def test_track_performance_returns_rolling_scores() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    pred = y.copy()
    pred[rng.choice(n, size=30, replace=False)] = 1 - pred[rng.choice(n, size=30, replace=False)]
    result = track_performance(pred, y, window=50, metric=accuracy_score)
    assert isinstance(result, PerformanceTrack)
    assert result.window_size == 50
    assert result.window_scores.shape == (n - 50 + 1,)
    assert 0.0 <= result.window_scores.min() <= 1.0


def test_track_performance_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must"):
        track_performance([0, 1, 0], [0, 1, 0], window=1)


def test_track_performance_window_exceeds_log() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        track_performance([0, 1, 0], [0, 1, 0], window=10)


def test_track_performance_metric_name_default() -> None:
    result = track_performance([0, 1] * 50, [0, 1] * 50, window=10, metric=accuracy_score)
    assert result.metric_name == "accuracy_score"
