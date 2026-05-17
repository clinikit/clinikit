"""clinikit.cost_sensitive — asymmetric-error utilities.

Three classical helpers for cost-sensitive binary classification:

- :func:`weighted_loss`           — total weighted misclassification cost.
- :func:`bayes_optimal_threshold` — analytical optimum under
  asymmetric costs (Elkan 2001).
- :func:`cost_curve`              — total cost across a threshold grid,
  plus the grid-optimal threshold via :attr:`CostCurveResult.optimal_threshold`.
"""

from __future__ import annotations

from clinikit.cost_sensitive._cost import (
    CostCurveResult,
    bayes_optimal_threshold,
    cost_curve,
    weighted_loss,
)

__all__ = [
    "CostCurveResult",
    "bayes_optimal_threshold",
    "cost_curve",
    "weighted_loss",
]
