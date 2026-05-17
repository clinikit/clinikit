"""The five concrete experiment protocols.

Every protocol is a thin configuration over :class:`BaseProtocol`:
the recipe differs in preprocessing, threshold strategy, and
defaults; the run loop is shared.

Protocols
---------
- :class:`DefensibleProtocol`   — conservative, recall-floored.
- :class:`MaxScoreProtocol`     — aggressive, accuracy-maximised.
- :class:`OriginalOnlyProtocol` — baseline, no preprocessing.
- :class:`DeploymentProtocol`   — production-style holdout split.
- :class:`AuditProtocol`        — large-test-size, recall-floored,
  intended for compliance reviews.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator, ClassifierMixin

from clinikit.preprocessing import RobustScaler
from clinikit.protocols._base import BaseProtocol

__all__ = [
    "AuditProtocol",
    "DefensibleProtocol",
    "DeploymentProtocol",
    "MaxScoreProtocol",
    "OriginalOnlyProtocol",
]


@dataclass
class DefensibleProtocol(BaseProtocol):
    """Conservative recipe — recall floor and robust scaling.

    Recall-floored threshold and a robust scaler so the protocol
    behaves the same on heavy-tailed inputs. Use this when the cost
    of a missed positive is non-trivial.
    """

    name: str = "defensible"
    threshold_strategy: str = "recall_constrained"
    test_size: float = 0.25

    def __post_init__(self) -> None:
        if self.preprocessing is None:
            self.preprocessing = RobustScaler()
        if self.threshold_kwargs is None:
            self.threshold_kwargs = {"min_recall": 0.9}


@dataclass
class MaxScoreProtocol(BaseProtocol):
    """Aggressive recipe — maximise raw accuracy on a held-out split.

    Picks the accuracy-maximising threshold on the training fold.
    Suitable for leaderboard chasing or shoot-out evaluations where
    headline accuracy is the only metric.
    """

    name: str = "max_score"
    threshold_strategy: str = "accuracy_max"
    test_size: float = 0.2


@dataclass
class OriginalOnlyProtocol(BaseProtocol):
    """Baseline recipe — no preprocessing, default 0.5 threshold.

    Use as a reference point when measuring how much value
    preprocessing and threshold tuning add.
    """

    name: str = "original_only"
    threshold_strategy: str = "accuracy_max"  # picks 0.5 if the data is balanced
    test_size: float = 0.2

    def __post_init__(self) -> None:
        # No preprocessing — feed raw features straight to the
        # classifier. A passthrough is encoded as a no-op transformer.
        if self.preprocessing is None:
            self.preprocessing = _Identity()


@dataclass
class DeploymentProtocol(BaseProtocol):
    """Production-style recipe — strict test split for deployment review.

    Larger test fold and recall-aware threshold tuning, matching the
    deployment-readiness checks described in
    :mod:`clinikit.governance`.
    """

    name: str = "deployment"
    threshold_strategy: str = "recall_constrained"
    test_size: float = 0.3

    def __post_init__(self) -> None:
        if self.threshold_kwargs is None:
            self.threshold_kwargs = {"min_recall": 0.85}


@dataclass
class AuditProtocol(BaseProtocol):
    """Compliance-review recipe — large test set, recall floor.

    Mirrors the strict evaluation guideline in
    :mod:`clinikit.audit`: a sizeable test fold so error estimates
    are tighter and a high recall floor so false negatives are
    visible.
    """

    name: str = "audit"
    threshold_strategy: str = "recall_constrained"
    test_size: float = 0.4

    def __post_init__(self) -> None:
        if self.threshold_kwargs is None:
            self.threshold_kwargs = {"min_recall": 0.95}


class _Identity(BaseEstimator):
    """No-op transformer used by :class:`OriginalOnlyProtocol`."""

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        del y
        return X


# Re-export so the type-checker sees the variable as a classifier-shaped
# protocol for downstream wrappers.
_ClassifierShape = ClassifierMixin
