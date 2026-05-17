"""clinikit.statistics — statistical tests for model comparison.

Public results
--------------
- :class:`DelongResult`
- :class:`BootstrapCI`
- :class:`McNemarResult`

Public functions
----------------
- :func:`delong_test`         — paired AUC comparison
- :func:`bootstrap_ci`        — percentile-bootstrap CI for any binary metric
- :func:`mcnemar_test`        — paired hard-prediction comparison
"""

from __future__ import annotations

from clinikit.statistics._tests import (
    BootstrapCI,
    DelongResult,
    McNemarResult,
    bootstrap_ci,
    delong_test,
    mcnemar_test,
)

__all__ = [
    "BootstrapCI",
    "DelongResult",
    "McNemarResult",
    "bootstrap_ci",
    "delong_test",
    "mcnemar_test",
]
