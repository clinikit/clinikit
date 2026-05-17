"""clinikit.diagnostics — label-noise and stability diagnostics.

Public functions
----------------
- cleanlab_label_issues(X, y, base_estimator)
- neighborhood_conflict(X, y, *, n_neighbors=10)
- loo_influence(X, y, base_estimator)
- seed_stability(X, y, estimator_factory, *, seeds, scoring)

``cleanlab_label_issues`` requires the optional ``[diagnostics]``
extra; the function raises a helpful ``ImportError`` if Cleanlab is
not installed.
"""

from __future__ import annotations

__all__: list[str] = []
