"""clinikit.plots — matplotlib wrappers for common figures.

Public functions
----------------
- plot_roc(y_true, y_score, *, ax=None)
- plot_pr(y_true, y_score, *, ax=None)
- plot_calibration(y_true, y_prob, *, n_bins=10, ax=None)
- plot_decision_curve(y_true, y_prob, *, ax=None)
- plot_threshold_sweep(y_true, y_prob, *, ax=None)

Each function accepts an optional ``ax`` and returns the
``matplotlib.axes.Axes`` it drew into.
"""

from __future__ import annotations

__all__: list[str] = []
