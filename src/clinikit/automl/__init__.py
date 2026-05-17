"""clinikit.automl — thin AutoML wrappers (safe-mode).

The wrappers are intentionally conservative: bounded budgets, fixed
seeds, no auto-uploading of data, and no silent dependency-install
side effects. Install the extra with
``pip install "clinikit[automl]"``.

Public functions
----------------
- run_tabpfn(X_train, y_train, X_test)
- run_flaml(X_train, y_train, *, time_budget)
- run_autogluon(X_train, y_train, *, time_budget)
"""

from __future__ import annotations

__all__: list[str] = []
