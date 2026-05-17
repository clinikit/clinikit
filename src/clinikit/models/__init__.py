"""clinikit.models — 14 sklearn-compatible hybrid classifiers.

Every public class in this subpackage inherits from
``sklearn.base.BaseEstimator`` and ``sklearn.base.ClassifierMixin``,
passes ``sklearn.utils.estimator_checks.check_estimator``, and stores
fitted attributes with a trailing underscore.

Public classes
--------------
RuleAugmentedClassifier
BoundaryRefineClassifier
SubgroupThresholdClassifier
ErrorAwareCalibrator
MonotonicBooster
HardSampleWeightedEnsemble
ClassConditionalImputer
CrossDistributionDistiller
SelectiveClassifier
InstanceAdaptiveThreshold
DialecticalEnsemble
LatentSubtypeRouter
IterativeLabelRefiner
DualViewCoTrainer
"""

from __future__ import annotations

__all__: list[str] = []
