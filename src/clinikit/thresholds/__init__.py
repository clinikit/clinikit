"""clinikit.thresholds — threshold-selection strategies for binary models.

The five built-in strategies are:

- :func:`accuracy_max`         — maximise raw accuracy.
- :func:`recall_constrained`   — maximise accuracy subject to a recall floor.
- :func:`accuracy_constrained` — maximise recall subject to an accuracy floor.
- :func:`cluster_specific`     — pick one threshold per supplied cluster.
- :func:`two_stage`            — pick a low/high pair with abstention between.

:func:`pick_threshold` dispatches by name; :func:`evaluate_threshold`
reports headline metrics at a given threshold.

All strategies treat the second element of ``np.unique(y_true)`` as the
positive class — supply ``{0, 1}`` labels for the unambiguous case.
"""

from __future__ import annotations

from clinikit.thresholds._strategies import (
    accuracy_constrained,
    accuracy_max,
    cluster_specific,
    evaluate_threshold,
    pick_threshold,
    recall_constrained,
    two_stage,
)

__all__ = [
    "accuracy_constrained",
    "accuracy_max",
    "cluster_specific",
    "evaluate_threshold",
    "pick_threshold",
    "recall_constrained",
    "two_stage",
]
