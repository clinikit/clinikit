"""clinikit.datasets — bundled UCI benchmark datasets.

Datasets are shipped as CSV files under
``clinikit/datasets/data/`` and loaded via
``importlib.resources.files()`` so the package works after a clean
``pip install clinikit`` on any supported platform.

Public loaders
--------------
- :func:`load_pima`        — PIMA Indians Diabetes
- :func:`load_wisconsin`   — Wisconsin Breast Cancer (Diagnostic)
- :func:`load_heart`       — UCI Heart Disease (Cleveland)

Each loader returns either a :class:`DatasetBunch` (sklearn-style)
or a tuple ``(X, y)`` when called with ``return_X_y=True``.

The Frankfurt Diabetes dataset listed in the project plan is
deferred to a later release pending license review.
"""

from __future__ import annotations

from clinikit.datasets._loaders import (
    DatasetBunch,
    load_heart,
    load_pima,
    load_wisconsin,
)

__all__ = [
    "DatasetBunch",
    "load_heart",
    "load_pima",
    "load_wisconsin",
]
