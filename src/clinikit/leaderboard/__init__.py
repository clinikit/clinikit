"""clinikit.leaderboard — experiment-tracking CSV with a fixed schema.

The leaderboard stores one row per experiment run with 38 columns
covering dataset, model, hyperparameters, split, metrics, calibration,
and reproducibility metadata.

Public functions
----------------
- new_leaderboard(path)
- append_run(path, row)
- read_leaderboard(path)
"""

from __future__ import annotations

__all__: list[str] = []
