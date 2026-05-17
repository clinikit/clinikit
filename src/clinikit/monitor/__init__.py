"""clinikit.monitor — drift detection and performance monitoring.

Drift tests
-----------
- ks_drift(reference, current, *, alpha=0.05)
- wasserstein_drift(reference, current)
- psi_drift(reference, current, *, n_bins=10)

Performance monitor
-------------------
- track_performance(predictions_log, ground_truth_log)
"""

from __future__ import annotations

__all__: list[str] = []
