"""clinikit.models — 14 sklearn-compatible hybrid classifiers.

Every public class in this subpackage inherits from
``sklearn.base.BaseEstimator`` and ``sklearn.base.ClassifierMixin``,
passes ``sklearn.utils.estimator_checks.check_estimator``, and stores
fitted attributes with a trailing underscore.

Public classes (G1 — threshold family)
--------------------------------------
- :class:`RuleAugmentedClassifier`
- :class:`BoundaryRefineClassifier`
- :class:`SubgroupThresholdClassifier`
- :class:`InstanceAdaptiveThreshold`

Remaining classes ship in subsequent groups; see ``roadmap.md``.
"""

from __future__ import annotations

from clinikit.models.boundary_refine import BoundaryRefineClassifier
from clinikit.models.instance_adaptive_threshold import InstanceAdaptiveThreshold
from clinikit.models.rule_augmented import RuleAugmentedClassifier
from clinikit.models.subgroup_threshold import SubgroupThresholdClassifier

__all__ = [
    "BoundaryRefineClassifier",
    "InstanceAdaptiveThreshold",
    "RuleAugmentedClassifier",
    "SubgroupThresholdClassifier",
]
