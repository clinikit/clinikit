"""clinikit.explainability — SHAP, LIME, and partial-dependence helpers.

All functions in this module lazily import their respective backends.
Install the extra with ``pip install "clinikit[explain]"``.

Public functions
----------------
- shap_summary(model, X)
- lime_explain_instance(model, X, instance_idx)
- partial_dependence(model, X, features)
"""

from __future__ import annotations

__all__: list[str] = []
