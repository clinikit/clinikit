"""clinikit.active_learning — modAL-based labelling loops.

modAL is an optional dependency. Install the extra with
``pip install "clinikit[active]"``.

Public function
---------------
- :func:`run_active_learning` — pool-based query loop with the
  caller's own oracle.

Public dataclass
----------------
- :class:`ActiveLearningResult`
"""

from __future__ import annotations

from clinikit.active_learning._loop import (
    ActiveLearningResult,
    run_active_learning,
)

__all__ = [
    "ActiveLearningResult",
    "run_active_learning",
]
