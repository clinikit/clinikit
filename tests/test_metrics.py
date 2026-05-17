"""Tests for clinikit.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score as sk_balanced_accuracy,
)
from sklearn.metrics import (
    brier_score_loss as sk_brier,
)
from sklearn.metrics import (
    fbeta_score as sk_fbeta,
)
from sklearn.metrics import (
    matthews_corrcoef as sk_mcc,
)
from sklearn.metrics import (
    precision_score as sk_precision,
)
from sklearn.metrics import (
    recall_score as sk_recall,
)

from clinikit.metrics import (
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    f2_score,
    mcc,
    npv,
    ppv,
    sensitivity,
    specificity,
)

# ---------------------------------------------------------------------------
# Hand-checked tiny cases
# ---------------------------------------------------------------------------


def test_sensitivity_handworked() -> None:
    # TP=1, FN=1 -> 0.5
    assert sensitivity([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_specificity_handworked() -> None:
    # TN=2, FP=1 -> 2/3
    assert specificity([0, 0, 0, 1], [0, 1, 0, 1]) == pytest.approx(2 / 3)


def test_ppv_handworked() -> None:
    # TP=2, FP=1 -> 2/3
    assert ppv([1, 1, 0, 0], [1, 1, 1, 0]) == pytest.approx(2 / 3)


def test_npv_handworked() -> None:
    # y_true=[1,0,0,0], y_pred=[0,0,0,1] -> TN=2, FN=1 -> 2/3
    assert npv([1, 0, 0, 0], [0, 0, 0, 1]) == pytest.approx(2 / 3)


def test_f2_score_handworked() -> None:
    # Precision=2/3, recall=1 -> F2 = 5 * (2/3) * 1 / (4*(2/3) + 1) = (10/3)/(11/3) = 10/11
    assert f2_score([1, 1, 0, 0], [1, 1, 1, 0]) == pytest.approx(10 / 11)


def test_balanced_accuracy_handworked() -> None:
    # sens=0.5, spec=2/3 -> ba = (0.5 + 2/3)/2 = 7/12
    assert balanced_accuracy([1, 1, 0, 0, 0], [1, 0, 0, 1, 0]) == pytest.approx(7 / 12)


# ---------------------------------------------------------------------------
# Parity with sklearn on random binary data
# ---------------------------------------------------------------------------


@pytest.fixture
def random_binary(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.uniform(0, 1, size=500)
    y_pred = (y_prob > 0.5).astype(int)
    return y_true, y_pred, y_prob


def test_sensitivity_matches_sklearn_recall(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, _ = random_binary
    assert sensitivity(y_true, y_pred) == pytest.approx(
        sk_recall(y_true, y_pred, pos_label=1, zero_division=0)
    )


def test_ppv_matches_sklearn_precision(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, _ = random_binary
    assert ppv(y_true, y_pred) == pytest.approx(
        sk_precision(y_true, y_pred, pos_label=1, zero_division=0)
    )


def test_f2_matches_sklearn_fbeta(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, _ = random_binary
    assert f2_score(y_true, y_pred) == pytest.approx(
        sk_fbeta(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
    )


def test_mcc_matches_sklearn(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, _ = random_binary
    assert mcc(y_true, y_pred) == pytest.approx(sk_mcc(y_true, y_pred))


def test_balanced_accuracy_matches_sklearn(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, _ = random_binary
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(sk_balanced_accuracy(y_true, y_pred))


def test_brier_matches_sklearn(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, _, y_prob = random_binary
    assert brier_score(y_true, y_prob) == pytest.approx(sk_brier(y_true, y_prob, pos_label=1))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "y_true", "y_pred"),
    [
        # No actual positives -> sensitivity denominator is 0
        (sensitivity, [0, 0, 0], [0, 0, 0]),
        # No actual negatives -> specificity denominator is 0
        (specificity, [1, 1, 1], [1, 1, 1]),
        # No predicted positives -> ppv denominator is 0
        (ppv, [0, 1, 0], [0, 0, 0]),
        # No predicted negatives -> npv denominator is 0
        (npv, [0, 1, 0], [1, 1, 1]),
        # No actual positives -> f2 collapses to 0/0 via ppv+sensitivity
        (f2_score, [0, 0, 0], [0, 0, 0]),
    ],
)
def test_zero_division_returns_configured_value(fn, y_true, y_pred) -> None:  # type: ignore[no-untyped-def]
    assert fn(y_true, y_pred, zero_division=0.5) == pytest.approx(0.5)


def test_sample_weight_propagates() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    weighted = sensitivity(y_true, y_pred, sample_weight=[2.0, 0.0, 1.0, 1.0])
    # Effective: 2 TP, 0 FN — weighted sensitivity = 1.0
    assert weighted == pytest.approx(1.0)


def test_binary_confusion_raises_on_multiclass() -> None:
    with pytest.raises(ValueError, match=">2 unique labels"):
        sensitivity([0, 1, 2], [0, 1, 2])


# ---------------------------------------------------------------------------
# ECE-specific tests
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero() -> None:
    # If probability exactly equals empirical frequency in each bin,
    # ECE collapses to 0.
    y_true = np.array([0] * 100 + [1] * 100)
    y_prob = np.array([0.0] * 100 + [1.0] * 100)
    assert expected_calibration_error(y_true, y_prob, n_bins=2) == pytest.approx(0.0)


def test_ece_worst_calibration_is_one() -> None:
    # All predictions confident-wrong: prob=1.0 but always y=0
    y_true = np.zeros(50, dtype=int)
    y_prob = np.ones(50)
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(1.0)


def test_ece_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="strategy"):
        expected_calibration_error([0, 1], [0.1, 0.9], strategy="nonsense")  # type: ignore[arg-type]


def test_ece_invalid_n_bins_raises() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0, 1], [0.1, 0.9], n_bins=0)


def test_ece_invalid_probabilities_raise() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error([0, 1], [0.1, 1.5])


def test_ece_quantile_strategy_runs(rng: np.random.Generator) -> None:
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
    assert 0.0 <= ece <= 1.0


def test_probabilistic_metrics_reject_2d_y_prob() -> None:
    # The package's binary-only API rejects 2-D probability arrays so
    # users do not accidentally pass a (n, n_classes) sklearn-style
    # output without choosing a positive column.
    y_true = [0, 1, 0, 1]
    y_prob_2d = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    with pytest.raises(ValueError, match="1-D"):
        brier_score(y_true, y_prob_2d)
    with pytest.raises(ValueError, match="1-D"):
        expected_calibration_error(y_true, y_prob_2d)


def test_brier_score_respects_sample_weight() -> None:
    # Manually weighted Brier: rows 0 and 1 carry all the mass and
    # match exactly (zero error); row 2 has weight 0 so the perfect
    # mis-prediction does not count.
    y_true = [0, 1, 1]
    y_prob = [0.0, 1.0, 0.0]
    weighted = brier_score(y_true, y_prob, sample_weight=[1.0, 1.0, 0.0])
    assert weighted == pytest.approx(0.0)
    # Sanity: unweighted is not zero because of the wrong row 2.
    assert brier_score(y_true, y_prob) > 0.0


def test_ece_respects_sample_weight() -> None:
    # Two perfectly-calibrated points carry positive weight; a third
    # confident-wrong point has zero weight, so ECE collapses to 0.
    y_true = [0, 1, 0]
    y_prob = [0.0, 1.0, 1.0]
    weighted = expected_calibration_error(y_true, y_prob, n_bins=2, sample_weight=[1.0, 1.0, 0.0])
    assert weighted == pytest.approx(0.0)
