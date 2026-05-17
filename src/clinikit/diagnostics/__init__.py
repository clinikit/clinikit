"""clinikit.diagnostics — label-noise and training-set stability tools.

Public functions
----------------
- :func:`cleanlab_label_issues`  — Cleanlab-based noisy-label detection
  (lazy-imports the optional ``[diagnostics]`` extra).
- :func:`neighborhood_conflict`  — per-sample k-NN label-disagreement rate.
- :func:`loo_influence`          — leave-one-out training influence (subsampled).
- :func:`seed_stability`         — distribution of an evaluation metric
  across multiple random seeds.

Public dataclasses
------------------
- :class:`SeedStabilityResult`
"""

from __future__ import annotations

from clinikit.diagnostics._diagnostics import (
    SeedStabilityResult,
    cleanlab_label_issues,
    loo_influence,
    neighborhood_conflict,
    seed_stability,
)

__all__ = [
    "SeedStabilityResult",
    "cleanlab_label_issues",
    "loo_influence",
    "neighborhood_conflict",
    "seed_stability",
]
