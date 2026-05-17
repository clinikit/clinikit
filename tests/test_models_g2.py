"""Use-case tests for the G2 classifier group.

(check_estimator coverage lives in tests/test_sklearn_compatibility.py.)
"""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.models import (
    DialecticalEnsemble,
    DualViewCoTrainer,
    HardSampleWeightedEnsemble,
    LatentSubtypeRouter,
)


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((200, 6))
    logits = X @ rng.standard_normal(6)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# HardSampleWeightedEnsemble
# ---------------------------------------------------------------------------


def test_hard_sample_weighted_ensemble_fits_n_members(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = HardSampleWeightedEnsemble(n_estimators=4, random_state=0).fit(X, y)
    assert len(clf.estimators_) == 4


def test_hard_sample_weighted_ensemble_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = HardSampleWeightedEnsemble(n_estimators=3, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_hard_sample_weighted_ensemble_invalid_growth() -> None:
    with pytest.raises(ValueError, match="weight_growth"):
        HardSampleWeightedEnsemble(weight_growth=0.9).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_hard_sample_weighted_ensemble_invalid_n_estimators() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        HardSampleWeightedEnsemble(n_estimators=0).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# DialecticalEnsemble
# ---------------------------------------------------------------------------


def test_dialectical_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = DialecticalEnsemble(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_dialectical_thesis_and_antithesis_present(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = DialecticalEnsemble(random_state=0).fit(X, y)
    assert clf.thesis_ is not None
    assert clf.antithesis_ is not None
    # Arbiter may or may not exist depending on disagreement count.


def test_dialectical_rejects_multiclass() -> None:
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        DialecticalEnsemble().fit(X, y)


# ---------------------------------------------------------------------------
# LatentSubtypeRouter
# ---------------------------------------------------------------------------


def test_latent_subtype_router_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = LatentSubtypeRouter(n_subtypes=2, random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_latent_subtype_router_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = LatentSubtypeRouter(n_subtypes=2, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_latent_subtype_router_invalid_n_subtypes() -> None:
    with pytest.raises(ValueError, match="n_subtypes"):
        LatentSubtypeRouter(n_subtypes=0).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_latent_subtype_router_invalid_min_size() -> None:
    with pytest.raises(ValueError, match="min_subtype_size"):
        LatentSubtypeRouter(min_subtype_size=1).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_latent_subtype_router_falls_back_for_tiny_subtypes() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] > 0).astype(int)
    # 4 subtypes for 40 samples → some will fall below default min_subtype_size=8.
    clf = LatentSubtypeRouter(n_subtypes=8, min_subtype_size=20, random_state=0).fit(X, y)
    # Every subtype should be small enough that experts are empty.
    assert len(clf.experts_) == 0


# ---------------------------------------------------------------------------
# DualViewCoTrainer
# ---------------------------------------------------------------------------


def test_dual_view_default_splits_features_in_half(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = DualViewCoTrainer(random_state=0).fit(X, y)
    assert len(clf.view1_features_) == 3
    assert len(clf.view2_features_) == 3
    assert np.intersect1d(clf.view1_features_, clf.view2_features_).size == 0


def test_dual_view_explicit_features(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = DualViewCoTrainer(view1_features=[0, 2, 4], view2_features=[1, 3, 5]).fit(X, y)
    assert list(clf.view1_features_) == [0, 2, 4]
    assert list(clf.view2_features_) == [1, 3, 5]


def test_dual_view_rejects_overlapping_views(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    with pytest.raises(ValueError, match="disjoint"):
        DualViewCoTrainer(view1_features=[0, 1], view2_features=[1, 2]).fit(X, y)


def test_dual_view_rejects_one_sided_explicit() -> None:
    X = np.zeros((6, 4))
    y = np.array([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="together"):
        DualViewCoTrainer(view1_features=[0, 1]).fit(X, y)


def test_dual_view_predict_proba_consistent(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = DualViewCoTrainer(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)
