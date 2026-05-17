"""clinikit.explainability — SHAP, LIME, and partial-dependence helpers.

SHAP and LIME are *optional* dependencies — install the
``[explain]`` extra to enable them
(``pip install "clinikit[explain]"``). The module imports cleanly
without them; calling the SHAP or LIME helpers without the extra
raises a friendly :class:`ImportError`. Partial dependence is
implemented on top of :mod:`sklearn.inspection` and works with the
core install.

Public functions
----------------
- :func:`shap_summary`            — mean-absolute SHAP feature importance.
- :func:`lime_explain_instance`   — local LIME explanation for one row.
- :func:`partial_dependence`      — sklearn-backed partial dependence.

Public dataclass
----------------
- :class:`PartialDependenceResult`
"""

from __future__ import annotations

from clinikit.explainability._explain import (
    PartialDependenceResult,
    lime_explain_instance,
    partial_dependence,
    shap_summary,
)

__all__ = [
    "PartialDependenceResult",
    "lime_explain_instance",
    "partial_dependence",
    "shap_summary",
]
