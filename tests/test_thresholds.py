"""Tests for clinikit.thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.thresholds import (
    accuracy_constrained,
    accuracy_max,
    cluster_specific,
    evaluate_threshold,
    pick_threshold,
    recall_constrained,
    two_stage,
)

# ---------------------------------------------------------------------------
# accuracy_max
# ---------------------------------------------------------------------------


def test_accuracy_max_perfectly_separable() -> None:
    # Probabilities perfectly rank labels — any cut between 0.4 and 0.6 is optimal.
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    t = accuracy_max(y, p)
    pred = (np.asarray(p) >= t).astype(int)
    assert np.array_equal(pred, y)


def test_accuracy_max_returns_in_unit_interval(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    t = accuracy_max(y, p)
    assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# recall_constrained / accuracy_constrained
# ---------------------------------------------------------------------------


def test_recall_constrained_meets_minimum() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0, 1, size=n)
    t = recall_constrained(y, p, min_recall=0.85)
    pred = (p >= t).astype(int)
    # Recall at the selected threshold must clear the floor.
    tp = int(((y == 1) & (pred == 1)).sum())
    pos = int((y == 1).sum())
    achieved = tp / max(pos, 1)
    assert achieved >= 0.85 or t == 0.0  # 0.0 means "all positive" -> recall=1


def test_accuracy_constrained_returns_valid() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    t = accuracy_constrained(y, p, min_accuracy=0.55)
    assert 0.0 <= t <= 1.0


def test_recall_constrained_validates_input() -> None:
    with pytest.raises(ValueError, match="min_recall"):
        recall_constrained([0, 1], [0.1, 0.9], min_recall=1.5)


def test_accuracy_constrained_validates_input() -> None:
    with pytest.raises(ValueError, match="min_accuracy"):
        accuracy_constrained([0, 1], [0.1, 0.9], min_accuracy=-0.1)


# ---------------------------------------------------------------------------
# cluster_specific
# ---------------------------------------------------------------------------


def test_cluster_specific_returns_per_cluster_thresholds() -> None:
    rng = np.random.default_rng(2)
    n = 60
    clusters = np.repeat([0, 1, 2], n // 3)
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0, 1, size=n)
    out = cluster_specific(y, p, clusters)
    assert set(out) == {0, 1, 2}
    assert all(0.0 <= t <= 1.0 for t in out.values())


def test_cluster_specific_uses_default_for_tiny_cluster() -> None:
    # Cluster with fewer samples than min_cluster_size falls back to default.
    y = [0, 1, 0, 1, 1, 1, 0, 0]
    p = [0.1, 0.9, 0.2, 0.7, 0.8, 0.6, 0.3, 0.4]
    clusters = [0, 0, 0, 0, 0, 0, 1, 1]  # cluster 1 has only 2 samples
    out = cluster_specific(y, p, clusters, min_cluster_size=5, default_threshold=0.5)
    assert out[1] == 0.5


# ---------------------------------------------------------------------------
# two_stage
# ---------------------------------------------------------------------------


def test_two_stage_returns_ordered_pair() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    t_low, t_high = two_stage(y, p, min_specificity=0.8, min_sensitivity=0.8)
    assert 0.0 <= t_low <= t_high <= 1.0


def test_two_stage_validates_inputs() -> None:
    with pytest.raises(ValueError, match="min_specificity"):
        two_stage([0, 1], [0.1, 0.9], min_specificity=1.5)
    with pytest.raises(ValueError, match="min_sensitivity"):
        two_stage([0, 1], [0.1, 0.9], min_sensitivity=-0.1)


def test_two_stage_handles_perfectly_separable() -> None:
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    t_low, t_high = two_stage(y, p, min_specificity=1.0, min_sensitivity=1.0)
    assert t_low <= t_high


# ---------------------------------------------------------------------------
# evaluate_threshold
# ---------------------------------------------------------------------------


def test_evaluate_threshold_returns_expected_keys() -> None:
    out = evaluate_threshold([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.7], 0.5)
    assert set(out) == {"accuracy", "sensitivity", "specificity", "positive_rate"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_evaluate_threshold_validates_range() -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_threshold([0, 1], [0.1, 0.9], 1.5)


# ---------------------------------------------------------------------------
# pick_threshold dispatcher
# ---------------------------------------------------------------------------


def test_pick_threshold_default_is_accuracy_max() -> None:
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    assert pick_threshold(y, p) == accuracy_max(y, p)


def test_pick_threshold_routes_by_name() -> None:
    y = [0, 0, 1, 1, 1]
    p = [0.1, 0.4, 0.55, 0.7, 0.9]
    assert pick_threshold(y, p, strategy="recall_constrained", min_recall=0.9) == (
        recall_constrained(y, p, min_recall=0.9)
    )


def test_pick_threshold_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        pick_threshold([0, 1], [0.1, 0.9], strategy="nonsense")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_strategies_reject_2d_y_prob() -> None:
    with pytest.raises(ValueError, match="1-D"):
        accuracy_max([0, 1], [[0.1, 0.9], [0.2, 0.8]])


def test_strategies_reject_out_of_range_prob() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        accuracy_max([0, 1], [0.1, 1.5])


def test_strategies_reject_multiclass() -> None:
    with pytest.raises(ValueError, match="binary-only"):
        accuracy_max([0, 1, 2], [0.1, 0.5, 0.9])
