"""clinikit.leaderboard — experiment tracking CSV with a fixed schema.

One row per experiment run, 38 fixed columns. The schema is exposed
as :data:`LEADERBOARD_COLUMNS` for downstream tooling.

Public functions
----------------
- :func:`new_leaderboard`  — create an empty CSV with the header.
- :func:`append_run`       — append a single run row.
- :func:`read_leaderboard` — read the CSV into a ``pandas.DataFrame``.
"""

from __future__ import annotations

from clinikit.leaderboard._leaderboard import (
    LEADERBOARD_COLUMNS,
    append_run,
    new_leaderboard,
    read_leaderboard,
)

__all__ = [
    "LEADERBOARD_COLUMNS",
    "append_run",
    "new_leaderboard",
    "read_leaderboard",
]
