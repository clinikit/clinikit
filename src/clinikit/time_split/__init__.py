"""clinikit.time_split — chronological train/test split utilities.

Two helpers, both DataFrame-first:

- :func:`chronological_split`   — single train/test cut by time order.
- :func:`rolling_origin_splits` — expanding-window iterator.

For group-aware time splits (same subject across rows must stay
together) use :class:`clinikit.cross_val.TimeSeriesGroupSplit`
instead.
"""

from __future__ import annotations

from clinikit.time_split._splits import (
    chronological_split,
    rolling_origin_splits,
)

__all__ = [
    "chronological_split",
    "rolling_origin_splits",
]
