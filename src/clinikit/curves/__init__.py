"""clinikit.curves — ROC, PR, calibration, and Decision Curve Analysis.

Each helper returns a small dataclass with the underlying numeric
arrays plus a ``plot()`` method that delegates to
:mod:`clinikit.plots`.

Public dataclasses
------------------
- :class:`ROCCurve`
- :class:`PrecisionRecallCurve`
- :class:`CalibrationCurve`
- :class:`DecisionCurve`

Public functions
----------------
- :func:`roc_curve_data`
- :func:`precision_recall_curve_data`
- :func:`calibration_curve_data`
- :func:`decision_curve_data`
"""

from __future__ import annotations

from clinikit.curves._curves import (
    CalibrationCurve,
    DecisionCurve,
    PrecisionRecallCurve,
    ROCCurve,
    calibration_curve_data,
    decision_curve_data,
    precision_recall_curve_data,
    roc_curve_data,
)

__all__ = [
    "CalibrationCurve",
    "DecisionCurve",
    "PrecisionRecallCurve",
    "ROCCurve",
    "calibration_curve_data",
    "decision_curve_data",
    "precision_recall_curve_data",
    "roc_curve_data",
]
