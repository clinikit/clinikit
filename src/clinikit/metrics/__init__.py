"""clinikit.metrics — classification metrics for tabular models.

Public functions
----------------
Confusion-matrix-derived (point predictions):

- :func:`sensitivity`
- :func:`specificity`
- :func:`ppv`
- :func:`npv`
- :func:`f2_score`
- :func:`mcc`
- :func:`balanced_accuracy`

Probability-based (predicted probabilities):

- :func:`brier_score`
- :func:`expected_calibration_error`

All metric functions follow the sklearn convention of taking
``y_true`` as the first positional argument and ``y_pred`` (or
``y_prob`` for probabilistic metrics) as the second. Other arguments
are keyword-only.
"""

from __future__ import annotations

from clinikit.metrics._classification import (
    balanced_accuracy,
    f2_score,
    mcc,
    npv,
    ppv,
    sensitivity,
    specificity,
)
from clinikit.metrics._probabilistic import (
    brier_score,
    expected_calibration_error,
)

__all__ = [
    "balanced_accuracy",
    "brier_score",
    "expected_calibration_error",
    "f2_score",
    "mcc",
    "npv",
    "ppv",
    "sensitivity",
    "specificity",
]
