"""clinikit.calibration — post-hoc probability calibration.

Each calibrator fits on uncalibrated probabilities + binary labels
and produces a calibrated probability vector. They operate on a 1-D
probability vector — apply them *after* a base classifier's
:meth:`predict_proba`, not inside an sklearn feature pipeline.

Public classes
--------------
- :class:`PlattCalibrator`     — sigmoid fit on logits.
- :class:`IsotonicCalibrator`  — non-parametric monotonic fit.
- :class:`TemperatureScaler`   — single-parameter logit scaling.

The reliability-diagram helper lives in :mod:`clinikit.curves`
(:func:`~clinikit.curves.calibration_curve_data`) and is plotted via
:func:`clinikit.plots.plot_calibration`.
"""

from __future__ import annotations

from clinikit.calibration._calibrators import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
)

__all__ = [
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureScaler",
]
