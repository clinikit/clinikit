"""clinikit.cost_sensitive — asymmetric error weighting.

Public functions
----------------
- weighted_loss(y_true, y_pred, *, cost_fp, cost_fn)
- bayes_optimal_threshold(*, cost_fp, cost_fn, p_positive)
- cost_curve(y_true, y_prob, *, cost_grid)
"""

from __future__ import annotations

__all__: list[str] = []
