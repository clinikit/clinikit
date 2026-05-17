"""clinikit.automl — thin AutoML wrappers (safe-mode).

Three optional back-ends, each behind a small, opinionated wrapper:

- :func:`run_tabpfn`    — TabPFN small-tabular prior network.
- :func:`run_flaml`     — FLAML time-budgeted AutoML.
- :func:`run_autogluon` — AutoGluon-Tabular ensemble AutoML.

All three are optional. Install the extra with
``pip install "clinikit[automl]"``. The wrappers do not auto-upload
data, do not install other packages on the fly, and forward
``random_state`` where the back-end supports one.

Public dataclass
----------------
- :class:`AutoMLResult`
"""

from __future__ import annotations

from clinikit.automl._runners import (
    AutoMLResult,
    run_autogluon,
    run_flaml,
    run_tabpfn,
)

__all__ = [
    "AutoMLResult",
    "run_autogluon",
    "run_flaml",
    "run_tabpfn",
]
