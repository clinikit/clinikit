"""Tests for clinikit.cost_sensitive."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.cost_sensitive import (
    CostCurveResult,
    bayes_optimal_threshold,
    cost_curve,
    weighted_loss,
)

# ---------------------------------------------------------------------------
# weighted_loss
# ---------------------------------------------------------------------------


def test_weighted_loss_simple() -> None:
    # y_true=[0,1,0,1], y_pred=[1,1,0,0] -> FP=1, FN=1
    # With cost_fp=2, cost_fn=5 -> 2 + 5 = 7
    assert weighted_loss([0, 1, 0, 1], [1, 1, 0, 0], cost_fp=2.0, cost_fn=5.0) == 7.0


def test_weighted_loss_perfect_zero() -> None:
    assert weighted_loss([0, 1, 0, 1], [0, 1, 0, 1]) == 0.0


def test_weighted_loss_rejects_negative_costs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        weighted_loss([0, 1], [0, 1], cost_fp=-1.0)


def test_weighted_loss_rejects_multiclass() -> None:
    with pytest.raises(ValueError, match="binary-only"):
        weighted_loss([0, 1, 2], [0, 1, 2])


# ---------------------------------------------------------------------------
# bayes_optimal_threshold
# ---------------------------------------------------------------------------


def test_bayes_optimal_threshold_symmetric_default() -> None:
    assert bayes_optimal_threshold() == 0.5


def test_bayes_optimal_threshold_asymmetric() -> None:
    # cost_fn=4*cost_fp -> threshold = 1/(1+4) = 0.2
    assert bayes_optimal_threshold(cost_fp=1.0, cost_fn=4.0) == pytest.approx(0.2)


def test_bayes_optimal_threshold_high_fp_cost() -> None:
    # cost_fp >> cost_fn -> high threshold (predict positive rarely)
    t = bayes_optimal_threshold(cost_fp=9.0, cost_fn=1.0)
    assert t == pytest.approx(0.9)


def test_bayes_optimal_threshold_zero_costs_returns_half() -> None:
    assert bayes_optimal_threshold(cost_fp=0.0, cost_fn=0.0) == 0.5


def test_bayes_optimal_threshold_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        bayes_optimal_threshold(cost_fp=-1.0)


# ---------------------------------------------------------------------------
# cost_curve
# ---------------------------------------------------------------------------


def test_cost_curve_returns_result(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=100)
    p = rng.uniform(0, 1, size=100)
    result = cost_curve(y, p, cost_fp=1.0, cost_fn=2.0)
    assert isinstance(result, CostCurveResult)
    assert result.thresholds.size == result.total_cost.size
    assert result.total_cost.min() >= 0.0


def test_cost_curve_optimal_threshold_matches_min() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    result = cost_curve(y, p)
    expected_idx = int(np.argmin(result.total_cost))
    assert result.optimal_threshold == result.thresholds[expected_idx]
    assert result.minimum_cost == result.total_cost[expected_idx]


def test_cost_curve_custom_thresholds() -> None:
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    result = cost_curve(y, p, thresholds=[0.2, 0.5, 0.8])
    assert result.thresholds.size == 3


def test_cost_curve_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds must"):
        cost_curve([0, 1], [0.1, 0.9], thresholds=[0.5, 1.5])


def test_cost_curve_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cost_curve([0, 1], [0.1, 1.5])


def test_cost_curve_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cost_curve([0, 1], [0.1, 0.9], cost_fp=-1.0)
