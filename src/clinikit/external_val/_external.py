"""Cross-dataset validation framework.

- :func:`validate_external`   — fit on one dataset, score on many.
- :func:`domain_shift_report` — per-feature drift summary using KS
  and PSI, returning a single DataFrame keyed by feature index.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score

from clinikit.metrics import sensitivity, specificity
from clinikit.monitor import ks_drift, psi_drift

__all__ = [
    "domain_shift_report",
    "validate_external",
]


def _coerce_xy(pair: tuple[ArrayLike, ArrayLike]) -> tuple[np.ndarray, np.ndarray]:
    X, y = pair
    return np.asarray(X), np.asarray(y)


_DEFAULT_METRICS: dict[str, Callable[..., float]] = {
    "accuracy": accuracy_score,
    "sensitivity": sensitivity,
    "specificity": specificity,
}


def validate_external(
    model: BaseEstimator,
    train_dataset: tuple[ArrayLike, ArrayLike],
    external_datasets: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    metrics: Mapping[str, Callable[..., float]] | None = None,
) -> pd.DataFrame:
    """Fit ``model`` on ``train_dataset`` and score it on each external dataset.

    Parameters
    ----------
    model : sklearn estimator
        Cloned before fit so the input is not mutated.
    train_dataset : (X, y)
        Training data. Must match the feature width of every external
        dataset.
    external_datasets : mapping[str, (X, y)]
        One named ``(X, y)`` pair per evaluation cohort. The model is
        evaluated on each separately.
    metrics : mapping, optional
        Map of metric name to callable with sklearn signature
        ``metric(y_true, y_pred) -> float``. Defaults to accuracy,
        sensitivity, and specificity.

    Returns
    -------
    pandas.DataFrame
        One row per external dataset; columns are
        ``["n_samples"] + list(metrics)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from clinikit.external_val import validate_external
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> table = validate_external(
    ...     LogisticRegression(),
    ...     train_dataset=(X, y),
    ...     external_datasets={"holdout": (X[:30], y[:30])},
    ... )
    >>> set(table.columns) >= {"accuracy", "sensitivity", "specificity", "n_samples"}
    True
    """
    if not external_datasets:
        raise ValueError("external_datasets must contain at least one entry.")

    used_metrics: Mapping[str, Callable[..., float]] = metrics or _DEFAULT_METRICS

    X_train, y_train = _coerce_xy(train_dataset)
    fitted = clone(model)
    fitted.fit(X_train, y_train)

    rows: list[dict[str, Any]] = []
    for name, pair in external_datasets.items():
        X_ext, y_ext = _coerce_xy(pair)
        if X_ext.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"external dataset {name!r} has {X_ext.shape[1]} features; "
                f"train has {X_train.shape[1]}."
            )
        y_pred = fitted.predict(X_ext)
        row: dict[str, Any] = {"dataset": name, "n_samples": int(X_ext.shape[0])}
        for metric_name, fn in used_metrics.items():
            row[metric_name] = float(fn(y_ext, y_pred))
        rows.append(row)

    return pd.DataFrame(rows).set_index("dataset")


def domain_shift_report(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    psi_n_bins: int = 10,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.25,
) -> pd.DataFrame:
    """Per-feature drift summary between two matrices.

    Runs the KS test and PSI on each feature column and returns a
    DataFrame with one row per feature.

    Parameters
    ----------
    reference, target : array-like of shape (n_samples, n_features)
    psi_n_bins : int, default 10
    ks_alpha : float, default 0.05
    psi_threshold : float, default 0.25

    Returns
    -------
    pandas.DataFrame
        Columns: ``["ks_statistic", "ks_p_value", "ks_drift",
        "psi_statistic", "psi_drift"]``. Indexed by feature index.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if tgt.ndim == 1:
        tgt = tgt.reshape(-1, 1)
    if ref.shape[1] != tgt.shape[1]:
        raise ValueError(f"reference has {ref.shape[1]} features; target has {tgt.shape[1]}.")

    rows: list[dict[str, Any]] = []
    for j in range(ref.shape[1]):
        ref_col = ref[:, j]
        tgt_col = tgt[:, j]
        try:
            ks = ks_drift(ref_col, tgt_col, alpha=ks_alpha)
            ks_stat = ks.statistic
            ks_p = ks.p_value
            ks_flag = ks.drift_detected
        except (ValueError, RuntimeError):
            ks_stat, ks_p, ks_flag = float("nan"), float("nan"), False

        try:
            psi = psi_drift(ref_col, tgt_col, n_bins=psi_n_bins, threshold=psi_threshold)
            psi_stat = psi.statistic
            psi_flag = psi.drift_detected
        except (ValueError, RuntimeError):
            psi_stat, psi_flag = float("nan"), False

        rows.append(
            {
                "feature": j,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "ks_drift": bool(ks_flag),
                "psi_statistic": psi_stat,
                "psi_drift": bool(psi_flag),
            }
        )

    return pd.DataFrame(rows).set_index("feature")
