"""clinikit.external_val — cross-dataset validation framework.

Public functions
----------------
- :func:`validate_external`   — fit on one dataset, evaluate on many.
- :func:`domain_shift_report` — per-feature drift between two matrices
  via KS + PSI.
"""

from __future__ import annotations

from clinikit.external_val._external import (
    domain_shift_report,
    validate_external,
)

__all__ = [
    "domain_shift_report",
    "validate_external",
]
