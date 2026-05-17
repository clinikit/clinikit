"""clinikit.thresholds — five threshold-selection strategies.

Strategies
----------
- accuracy_max
- recall_constrained(min_recall)
- accuracy_constrained(min_accuracy)
- cluster_specific(clusters)
- two_stage(low, high)

Public functions
----------------
- pick_threshold(y_true, y_prob, *, strategy="accuracy_max", **kwargs)
- evaluate_threshold(y_true, y_prob, threshold)
"""

from __future__ import annotations

__all__: list[str] = []
