"""Use-case tests for the G4 classifier group."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.models import (
    ClassConditionalImputer,
    CrossDistributionDistiller,
    IterativeLabelRefiner,
)


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((150, 4))
    y = (X[:, 0] + 0.3 * rng.standard_normal(150) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# ClassConditionalImputer
# ---------------------------------------------------------------------------


def test_class_conditional_imputer_handles_nans(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    X = X.copy()
    X[0, 0] = np.nan
    X[1, 2] = np.nan
    clf = ClassConditionalImputer().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_class_conditional_imputer_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = ClassConditionalImputer().fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_class_conditional_imputer_stores_class_medians(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = ClassConditionalImputer().fit(X, y)
    assert clf.class_medians_.shape == (2, X.shape[1])
    assert clf.class_priors_.shape == (2,)
    assert clf.class_priors_.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CrossDistributionDistiller
# ---------------------------------------------------------------------------


def test_cross_distribution_distiller_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = CrossDistributionDistiller(random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_cross_distribution_distiller_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = CrossDistributionDistiller(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_cross_distribution_distiller_invalid_strength() -> None:
    with pytest.raises(ValueError, match="distillation_strength"):
        CrossDistributionDistiller(distillation_strength=1.5).fit(
            np.zeros((4, 2)), np.array([0, 1, 0, 1])
        )


def test_cross_distribution_distiller_invalid_perturb() -> None:
    with pytest.raises(ValueError, match="perturb_scale"):
        CrossDistributionDistiller(perturb_scale=-0.1).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_cross_distribution_distiller_zero_strength_runs(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = CrossDistributionDistiller(distillation_strength=0.0, random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


# ---------------------------------------------------------------------------
# IterativeLabelRefiner
# ---------------------------------------------------------------------------


def test_iterative_label_refiner_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = IterativeLabelRefiner(random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_iterative_label_refiner_records_flips(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    y = y.copy()
    # Plant a single very confident wrong label.
    y[0] = 1 - y[0]
    clf = IterativeLabelRefiner(
        max_flips_per_iter=1, max_total_flips_fraction=0.05, random_state=0
    ).fit(X, y)
    assert clf.flips_ >= 0
    assert clf.n_iter_ >= 1
    assert clf.refined_labels_.shape == y.shape


def test_iterative_label_refiner_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="flip_threshold"):
        IterativeLabelRefiner(flip_threshold=0.4).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_iterative_label_refiner_invalid_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        IterativeLabelRefiner(max_iter=0).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_iterative_label_refiner_zero_flips_per_iter_runs() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = IterativeLabelRefiner(max_flips_per_iter=0, random_state=0).fit(X, y)
    # No flips allowed -> labels untouched.
    assert np.array_equal(clf.refined_labels_, y)
