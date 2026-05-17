"""clinikit.audit — leakage detection, subgroup fairness, doc checks.

Public functions
----------------
- detect_leakage(X_train, X_test, y_train, y_test)
- subgroup_fairness(y_true, y_pred, groups)
- documentation_checklist(run)

These utilities are documentation aids for research workflows; they
do not replace formal compliance review.
"""

from __future__ import annotations

__all__: list[str] = []
