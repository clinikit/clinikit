"""clinikit.synthetic — CTGAN / TVAE wrappers with a TSTR safety gate.

Synthetic data generators include a Train-on-Synthetic-Test-on-Real
(TSTR) safety check: a downstream classifier is trained on synthetic
samples and evaluated on real held-out data. If the gap between TSTR
and the real-on-real baseline exceeds a configurable tolerance, the
synthetic dataset is flagged and not used by default.

Install the extra with ``pip install "clinikit[synthetic]"``. The
TSTR helper is pure-Python and works without the extra.

Public functions
----------------
- :func:`fit_ctgan`        — fit a CTGAN synthesizer (requires SDV).
- :func:`fit_tvae`         — fit a TVAE synthesizer (requires SDV).
- :func:`tstr_safety_gate` — TSTR vs RTRT downstream safety check.

Public dataclass
----------------
- :class:`TSTRReport`
"""

from __future__ import annotations

from clinikit.synthetic._synth import (
    TSTRReport,
    fit_ctgan,
    fit_tvae,
    tstr_safety_gate,
)

__all__ = [
    "TSTRReport",
    "fit_ctgan",
    "fit_tvae",
    "tstr_safety_gate",
]
