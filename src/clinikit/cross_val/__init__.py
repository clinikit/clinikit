"""clinikit.cross_val — group-aware cross-validation utilities.

Public splitters
----------------
- :class:`GroupStratifiedKFold` — sklearn StratifiedGroupKFold with
  consistent defaults (``shuffle=True``, seeded ``random_state``).
- :class:`TimeSeriesGroupSplit` — chronological folds that respect
  group membership.

Public helper
-------------
- :func:`cross_validate_with_groups` — runs cross-validation,
  returning a :class:`CrossValResult` with per-fold metrics, plus
  mean and std summaries.
"""

from __future__ import annotations

from clinikit.cross_val._splitters import (
    CrossValResult,
    GroupStratifiedKFold,
    TimeSeriesGroupSplit,
    cross_validate_with_groups,
)

__all__ = [
    "CrossValResult",
    "GroupStratifiedKFold",
    "TimeSeriesGroupSplit",
    "cross_validate_with_groups",
]
