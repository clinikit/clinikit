"""Tests for clinikit.statistics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from clinikit.statistics import (
    BootstrapCI,
    DelongResult,
    McNemarResult,
    bootstrap_ci,
    delong_test,
    mcnemar_test,
)

# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------


def test_delong_test_returns_result(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=200)
    pa = rng.uniform(0, 1, size=200)
    pb = rng.uniform(0, 1, size=200)
    result = delong_test(y, pa, pb)
    assert isinstance(result, DelongResult)
    assert 0.0 <= result.p_value <= 1.0
    # AUC should match sklearn's calculation.
    assert result.auc_a == pytest.approx(roc_auc_score(y, pa), abs=1e-3)
    assert result.auc_b == pytest.approx(roc_auc_score(y, pb), abs=1e-3)


def test_delong_identical_scorers_p_value_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.uniform(0, 1, size=100)
    result = delong_test(y, p, p)
    # Identical scorers -> AUC equal -> Z=0 -> p=1.
    assert result.p_value == pytest.approx(1.0)
    assert result.z == pytest.approx(0.0)


def test_delong_rejects_multiclass() -> None:
    with pytest.raises(ValueError, match="binary-only"):
        delong_test([0, 1, 2], [0.1, 0.5, 0.9], [0.2, 0.4, 0.8])


def test_delong_rejects_2d_scores() -> None:
    with pytest.raises(ValueError, match="1-D"):
        delong_test([0, 1], [[0.1, 0.9]], [0.2, 0.4])


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_returns_ordered_interval(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=120)
    pred = rng.integers(0, 2, size=120)
    ci = bootstrap_ci(y, pred, accuracy_score, n_resamples=200, random_state=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.n_resamples == 200


def test_bootstrap_ci_alpha_widens_interval(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=120)
    pred = rng.integers(0, 2, size=120)
    narrow = bootstrap_ci(y, pred, accuracy_score, alpha=0.10, n_resamples=300, random_state=0)
    wide = bootstrap_ci(y, pred, accuracy_score, alpha=0.05, n_resamples=300, random_state=0)
    assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)


def test_bootstrap_ci_invalid_args() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([0, 1], [0, 1], accuracy_score, n_resamples=1)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci([0, 1], [0, 1], accuracy_score, alpha=0.0)


def test_bootstrap_ci_unstratified_runs(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=80)
    pred = rng.integers(0, 2, size=80)
    ci = bootstrap_ci(y, pred, accuracy_score, n_resamples=50, stratify=False, random_state=0)
    assert ci.lower <= ci.upper


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_identical_predictions() -> None:
    # Two identical classifiers -> b=c=0 -> p=1 by convention.
    y = [0, 1, 0, 1, 1, 0]
    pred = [0, 1, 0, 0, 1, 0]
    result = mcnemar_test(y, pred, pred)
    assert isinstance(result, McNemarResult)
    assert result.b == 0
    assert result.c == 0
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_strongly_different() -> None:
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pred_a = [0, 0, 0, 0, 1, 1, 1, 1]  # all correct
    pred_b = [1, 1, 1, 1, 0, 0, 0, 0]  # all wrong
    result = mcnemar_test(y, pred_a, pred_b)
    assert result.b == 8
    assert result.c == 0
    assert result.p_value < 0.05


def test_mcnemar_without_continuity_correction() -> None:
    # Engineer b=4 (pred_a-only correct), c=1 (pred_b-only correct) so the
    # continuity correction has visible bite: with corr -> (|4-1|-1)^2/5 =
    # 0.8; without -> (4-1)^2/5 = 1.8.
    y = [1, 1, 1, 1, 1, 0]
    pred_a = [1, 1, 1, 1, 0, 1]
    pred_b = [0, 0, 0, 0, 1, 0]
    r_with = mcnemar_test(y, pred_a, pred_b, continuity_correction=True)
    r_without = mcnemar_test(y, pred_a, pred_b, continuity_correction=False)
    assert r_without.statistic > r_with.statistic
