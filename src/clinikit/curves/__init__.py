"""clinikit.curves — ROC, PR, calibration, and Decision Curve Analysis.

Public functions
----------------
- roc_curve_data
- precision_recall_curve_data
- calibration_curve_data
- decision_curve_data

Each returns a dataclass with numeric arrays plus a ``plot()`` helper
that delegates to matplotlib.
"""

from __future__ import annotations

__all__: list[str] = []
