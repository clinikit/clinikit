"""clinikit.protocols — five experiment protocols.

Protocols
---------
- DefensibleProtocol
- MaxScoreProtocol
- OriginalOnlyProtocol
- DeploymentProtocol
- AuditProtocol

A protocol bundles a preprocessing pipeline, a model class, a CV
strategy, a threshold strategy, and a reporting configuration into a
single reproducible recipe.
"""

from __future__ import annotations

__all__: list[str] = []
