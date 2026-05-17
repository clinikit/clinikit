"""clinikit.cross_val — group-aware cross-validation utilities.

Public splitters
----------------
- GroupStratifiedKFold
- TimeSeriesGroupSplit
- NestedCV

Public helpers
--------------
- cross_validate_with_groups(estimator, X, y, *, groups, cv, scoring)
"""

from __future__ import annotations

__all__: list[str] = []
