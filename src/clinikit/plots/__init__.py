"""clinikit.plots — matplotlib wrappers for common diagnostic figures.

Each helper accepts either a :mod:`clinikit.curves` dataclass or raw
``(y_true, y_prob)`` arrays and returns the
:class:`matplotlib.axes.Axes` it drew into. The caller owns the
figure; no ``plt.show()`` is called.

Public functions
----------------
- :func:`plot_roc`
- :func:`plot_pr`
- :func:`plot_calibration`
- :func:`plot_decision_curve`
- :func:`plot_threshold_sweep`
"""

from __future__ import annotations

from clinikit.plots._plots import (
    plot_calibration,
    plot_decision_curve,
    plot_pr,
    plot_roc,
    plot_threshold_sweep,
)

__all__ = [
    "plot_calibration",
    "plot_decision_curve",
    "plot_pr",
    "plot_roc",
    "plot_threshold_sweep",
]
