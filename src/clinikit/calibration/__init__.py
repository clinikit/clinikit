"""clinikit.calibration — post-hoc probability calibration.

Calibrators
-----------
- PlattCalibrator
- IsotonicCalibrator
- TemperatureScaler

All calibrators follow the sklearn ``fit`` / ``transform`` /
``predict_proba`` API and may be plugged into pipelines.

Public functions
----------------
- reliability_diagram(y_true, y_prob, *, n_bins=10, ax=None)
"""

from __future__ import annotations

__all__: list[str] = []
