"""clinikit.synthetic — CTGAN / TVAE wrappers with a TSTR safety gate.

Synthetic data generators include a Train-on-Synthetic-Test-on-Real
(TSTR) safety check: a downstream classifier is trained on synthetic
samples and evaluated on real held-out data. If the gap between TSTR
and the real-on-real baseline exceeds a configurable tolerance, the
synthetic dataset is flagged and not used by default.

Install the extra with ``pip install "clinikit[synthetic]"``.

Public functions
----------------
- fit_ctgan(X, *, epochs=300, random_state=None)
- fit_tvae(X, *, epochs=300, random_state=None)
- tstr_safety_gate(real_X, real_y, synth_X, synth_y, *, tolerance)
"""

from __future__ import annotations

__all__: list[str] = []
