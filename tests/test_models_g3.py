"""Use-case tests for the G3 classifier group."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.models import (
    ErrorAwareCalibrator,
    MonotonicBooster,
    SelectiveClassifier,
)


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + 0.3 * rng.standard_normal(200) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# ErrorAwareCalibrator
# ---------------------------------------------------------------------------


def test_error_aware_calibrator_predict_proba_in_unit_interval(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = ErrorAwareCalibrator(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_error_aware_calibrator_predict_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = ErrorAwareCalibrator(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_error_aware_calibrator_invalid_n_clusters() -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        ErrorAwareCalibrator(n_clusters=0).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_error_aware_calibrator_invalid_cv() -> None:
    with pytest.raises(ValueError, match="cv"):
        ErrorAwareCalibrator(cv=1).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# MonotonicBooster
# ---------------------------------------------------------------------------


def test_monotonic_booster_default_no_constraints(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = MonotonicBooster(random_state=0).fit(X, y)
    assert np.all(clf.monotonic_vector_ == 0)


def test_monotonic_booster_applies_constraint(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = MonotonicBooster(monotonic_constraints={0: 1, 2: -1}, random_state=0).fit(X, y)
    assert clf.monotonic_vector_[0] == 1
    assert clf.monotonic_vector_[2] == -1
    assert clf.monotonic_vector_[1] == 0
    assert clf.monotonic_vector_[3] == 0


def test_monotonic_booster_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        MonotonicBooster(monotonic_constraints={0: 2}).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_monotonic_booster_invalid_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        MonotonicBooster(monotonic_constraints={9: 1}).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_monotonic_booster_handles_nan_input(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    X = X.copy()
    X[0, 0] = np.nan
    clf = MonotonicBooster(random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


# ---------------------------------------------------------------------------
# SelectiveClassifier
# ---------------------------------------------------------------------------


def test_selective_classifier_predict_matches_base(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = SelectiveClassifier(coverage=0.9).fit(X, y)
    # Standard predict ignores abstention — must be a normal label.
    pred = clf.predict(X)
    assert set(np.unique(pred).tolist()) <= set(clf.classes_.tolist())


def test_selective_classifier_predict_with_abstain_uses_minus_one(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = SelectiveClassifier(coverage=0.85).fit(X, y)
    out = clf.predict_with_abstain(X)
    labels = set(np.unique(out).tolist())
    assert labels <= {-1, 0, 1}
    # With coverage<1, at least some samples should abstain.
    assert (out == -1).any()


def test_selective_classifier_coverage_one_never_abstains(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = SelectiveClassifier(coverage=1.0).fit(X, y)
    out = clf.predict_with_abstain(X)
    # coverage=1.0 collapses the abstain band to a single point — only
    # the unlikely exact-0.5 case can abstain. With continuous data
    # this should essentially never happen.
    assert (out == -1).sum() <= 1


def test_selective_classifier_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        SelectiveClassifier(coverage=0.4).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
