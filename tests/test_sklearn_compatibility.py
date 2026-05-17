"""sklearn estimator compatibility tests.

Every public estimator listed in ``CLINIKIT_PUBLIC_ESTIMATORS`` is
parameterised through ``sklearn.utils.estimator_checks.check_estimator``.

This list starts empty during the foundation phase and grows as
estimators land. A new estimator is not considered shipped until its
entry is added here and the test passes on the CI matrix.
"""

from __future__ import annotations

from typing import Any

import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

from clinikit.models import (
    BoundaryRefineClassifier,
    ClassConditionalImputer,
    CrossDistributionDistiller,
    DialecticalEnsemble,
    DualViewCoTrainer,
    ErrorAwareCalibrator,
    HardSampleWeightedEnsemble,
    InstanceAdaptiveThreshold,
    IterativeLabelRefiner,
    LatentSubtypeRouter,
    MonotonicBooster,
    RuleAugmentedClassifier,
    SelectiveClassifier,
    SubgroupThresholdClassifier,
)

CLINIKIT_PUBLIC_ESTIMATORS: list[type[BaseEstimator]] = [
    # G1 — threshold family
    RuleAugmentedClassifier,
    BoundaryRefineClassifier,
    SubgroupThresholdClassifier,
    InstanceAdaptiveThreshold,
    # G2 — ensemble / routing family
    HardSampleWeightedEnsemble,
    DialecticalEnsemble,
    LatentSubtypeRouter,
    DualViewCoTrainer,
    # G3 — calibration / monotonic / abstention
    ErrorAwareCalibrator,
    MonotonicBooster,
    SelectiveClassifier,
    # G4 — imputation / distillation / label refinement
    ClassConditionalImputer,
    CrossDistributionDistiller,
    IterativeLabelRefiner,
]


@pytest.mark.sklearn_api
@pytest.mark.parametrize(
    "estimator_cls",
    CLINIKIT_PUBLIC_ESTIMATORS,
    ids=[cls.__name__ for cls in CLINIKIT_PUBLIC_ESTIMATORS],
)
def test_check_estimator(estimator_cls: type[BaseEstimator]) -> None:
    """Run sklearn's ``check_estimator`` on a public estimator class.

    The estimator must be instantiable with no required arguments
    (sklearn convention) and must pass every default check.
    """
    estimator: Any = estimator_cls()
    check_estimator(estimator)


def test_public_estimator_list_exists() -> None:
    """The registry of public estimators must be a list (possibly empty)."""
    assert isinstance(CLINIKIT_PUBLIC_ESTIMATORS, list)
