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

Public classes (G2 — ensemble / routing family)
-----------------------------------------------
- :class:`HardSampleWeightedEnsemble`
- :class:`DialecticalEnsemble`
- :class:`LatentSubtypeRouter`
- :class:`DualViewCoTrainer`

Remaining classes ship in subsequent groups; see ``roadmap.md``.
"""

from __future__ import annotations

from clinikit.models.boundary_refine import BoundaryRefineClassifier
from clinikit.models.dialectical_ensemble import DialecticalEnsemble
from clinikit.models.dual_view_cotrainer import DualViewCoTrainer
from clinikit.models.hard_sample_weighted_ensemble import HardSampleWeightedEnsemble
from clinikit.models.instance_adaptive_threshold import InstanceAdaptiveThreshold
from clinikit.models.latent_subtype_router import LatentSubtypeRouter
from clinikit.models.rule_augmented import RuleAugmentedClassifier
from clinikit.models.subgroup_threshold import SubgroupThresholdClassifier

__all__ = [
    "BoundaryRefineClassifier",
    "DialecticalEnsemble",
    "DualViewCoTrainer",
    "HardSampleWeightedEnsemble",
    "InstanceAdaptiveThreshold",
    "LatentSubtypeRouter",
    "RuleAugmentedClassifier",
    "SubgroupThresholdClassifier",
]
