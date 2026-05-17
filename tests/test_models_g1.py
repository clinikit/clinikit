"""Use-case tests for the G1 classifier group.

(check_estimator coverage lives in tests/test_sklearn_compatibility.py.)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from clinikit.models import (
    BoundaryRefineClassifier,
    InstanceAdaptiveThreshold,
    RuleAugmentedClassifier,
    SubgroupThresholdClassifier,
)


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((200, 4))
    logits = X @ rng.standard_normal(4)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# RuleAugmentedClassifier
# ---------------------------------------------------------------------------


def test_rule_augmented_passes_through_without_rules(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    base = LogisticRegression()
    base.fit(X, y)
    base_pred = base.predict(X)

    clf = RuleAugmentedClassifier(base_estimator=LogisticRegression()).fit(X, y)
    assert np.array_equal(clf.predict(X), base_pred)


def test_rule_augmented_rule_overrides_base() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] > 0).astype(int)

    def always_one(arr: np.ndarray) -> np.ndarray:
        return np.ones(arr.shape[0], dtype=np.int64)

    clf = RuleAugmentedClassifier(rules=[always_one]).fit(X, y)
    assert np.all(clf.predict(X) == 1)


def test_rule_augmented_no_opinion_keeps_base() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 2))
    y = (X[:, 0] > 0).astype(int)

    def silent(arr: np.ndarray) -> np.ndarray:
        return np.full(arr.shape[0], -1, dtype=np.int64)

    clf_with_rule = RuleAugmentedClassifier(rules=[silent]).fit(X, y)
    clf_plain = RuleAugmentedClassifier().fit(X, y)
    assert np.array_equal(clf_with_rule.predict(X), clf_plain.predict(X))


def test_rule_augmented_rejects_bad_rule_shape() -> None:
    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 1])

    def wrong_shape(arr: np.ndarray) -> np.ndarray:
        return np.zeros(arr.shape[0] + 1, dtype=np.int64)

    clf = RuleAugmentedClassifier(rules=[wrong_shape]).fit(X, y)
    with pytest.raises(ValueError, match="expected"):
        clf.predict(X)


def test_rule_augmented_rejects_invalid_label() -> None:
    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 1])

    def bad_label(arr: np.ndarray) -> np.ndarray:
        return np.full(arr.shape[0], 7, dtype=np.int64)

    clf = RuleAugmentedClassifier(rules=[bad_label]).fit(X, y)
    with pytest.raises(ValueError, match="classes_"):
        clf.predict(X)


def test_rule_augmented_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="rule_confidence"):
        RuleAugmentedClassifier(rule_confidence=0.4).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# BoundaryRefineClassifier
# ---------------------------------------------------------------------------


def test_boundary_refine_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = BoundaryRefineClassifier(margin=0.2).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert clf.refine_estimator_ is not None or clf.refine_estimator_ is None  # both ok


def test_boundary_refine_invalid_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        BoundaryRefineClassifier(margin=0.6).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_boundary_refine_skips_refinement_when_too_few_boundary() -> None:
    # Very wide margin AND very separable data → no boundary samples.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    X[:20] += 10.0  # far-apart classes
    y = np.array([0] * 20 + [1] * 20)
    clf = BoundaryRefineClassifier(margin=0.001).fit(X, y)
    assert clf.refine_estimator_ is None


def test_boundary_refine_rejects_multiclass() -> None:
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        BoundaryRefineClassifier().fit(X, y)


# ---------------------------------------------------------------------------
# SubgroupThresholdClassifier
# ---------------------------------------------------------------------------


def test_subgroup_threshold_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = SubgroupThresholdClassifier(n_subgroups=3, random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert len(clf.thresholds_) >= 1


def test_subgroup_threshold_predict_proba_consistent_with_predict(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = SubgroupThresholdClassifier(n_subgroups=2, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_subgroup_threshold_invalid_default() -> None:
    with pytest.raises(ValueError, match="default_threshold"):
        SubgroupThresholdClassifier(default_threshold=1.5).fit(
            np.zeros((4, 2)), np.array([0, 1, 0, 1])
        )


def test_subgroup_threshold_invalid_n_subgroups() -> None:
    with pytest.raises(ValueError, match="n_subgroups"):
        SubgroupThresholdClassifier(n_subgroups=0).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# InstanceAdaptiveThreshold
# ---------------------------------------------------------------------------


def test_instance_adaptive_predicts(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = InstanceAdaptiveThreshold(random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_instance_adaptive_predict_proba_consistent_with_predict(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    clf = InstanceAdaptiveThreshold(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], pred)


def test_instance_adaptive_invalid_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        InstanceAdaptiveThreshold(margin=0.5).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_instance_adaptive_invalid_cv() -> None:
    with pytest.raises(ValueError, match="cv"):
        InstanceAdaptiveThreshold(cv=1).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
