"""clinikit.protocols — five experiment protocols.

A *protocol* bundles preprocessing, a base classifier, a threshold
strategy, and split / reporting choices into a single reproducible
recipe. Each protocol exposes a single :meth:`run` method that
returns a :class:`ProtocolResult` with headline metrics and a
pre-built leaderboard row.

Public classes
--------------
- :class:`DefensibleProtocol`   — conservative, recall-floored.
- :class:`MaxScoreProtocol`     — aggressive, accuracy-maximised.
- :class:`OriginalOnlyProtocol` — baseline, no preprocessing.
- :class:`DeploymentProtocol`   — production-style holdout.
- :class:`AuditProtocol`        — compliance-grade, large test fold.

Plus the shared base + result dataclass:

- :class:`BaseProtocol`
- :class:`ProtocolResult`
"""

from __future__ import annotations

from clinikit.protocols._base import BaseProtocol, ProtocolResult
from clinikit.protocols._protocols import (
    AuditProtocol,
    DefensibleProtocol,
    DeploymentProtocol,
    MaxScoreProtocol,
    OriginalOnlyProtocol,
)

__all__ = [
    "AuditProtocol",
    "BaseProtocol",
    "DefensibleProtocol",
    "DeploymentProtocol",
    "MaxScoreProtocol",
    "OriginalOnlyProtocol",
    "ProtocolResult",
]
