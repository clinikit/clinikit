"""clinikit.statistics — statistical tests for model comparison.

Public functions
----------------
- delong_test(y_true, y_prob_a, y_prob_b)
- bootstrap_ci(scores, *, n_resamples=1000, alpha=0.05, random_state=None)
- mcnemar_test(y_true, y_pred_a, y_pred_b)
"""

from __future__ import annotations

__all__: list[str] = []
