"""clinikit.monitor — drift detection and rolling performance.

Public functions
----------------
- :func:`ks_drift`           — two-sample Kolmogorov-Smirnov test.
- :func:`wasserstein_drift`  — 1-Wasserstein distance.
- :func:`psi_drift`          — Population Stability Index.
- :func:`track_performance`  — rolling-window metric across a log.

Public dataclasses
------------------
- :class:`DriftResult`
- :class:`PerformanceTrack`
"""

from __future__ import annotations

from clinikit.monitor._drift import (
    DriftResult,
    PerformanceTrack,
    ks_drift,
    psi_drift,
    track_performance,
    wasserstein_drift,
)

__all__ = [
    "DriftResult",
    "PerformanceTrack",
    "ks_drift",
    "psi_drift",
    "track_performance",
    "wasserstein_drift",
]
