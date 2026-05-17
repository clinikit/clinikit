"""Pre-deployment audit helpers — leakage, subgroup fairness, doc checks.

Three diagnostics that surface common problems before a model is
treated as ready to ship:

- :func:`detect_leakage`        — overlap and distribution checks
  between train and test folds.
- :func:`subgroup_fairness`     — per-group accuracy / sensitivity /
  specificity with worst/best ratios.
- :func:`documentation_checklist` — checklist of fields a run record
  ought to contain before it can be filed.

These are *documentation aids* for research workflows; they do not
replace formal compliance review.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.utils.validation import check_consistent_length

from clinikit.metrics import sensitivity, specificity

__all__ = [
    "DocumentationCheckResult",
    "LeakageReport",
    "SubgroupFairnessReport",
    "detect_leakage",
    "documentation_checklist",
    "subgroup_fairness",
]


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LeakageReport:
    """Output of :func:`detect_leakage`."""

    duplicate_rows: int
    duplicate_row_rate: float
    feature_distribution_alerts: list[str]
    label_distribution_alert: bool
    summary: str


def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 1- or 2-D; got ndim={arr.ndim}.")
    return arr


def detect_leakage(
    X_train: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
    *,
    distribution_alpha: float = 1e-3,
    label_ratio_tolerance: float = 0.1,
) -> LeakageReport:
    """Flag potential train/test contamination.

    The function performs three checks:

    1. **Row-level overlap** — count rows of ``X_test`` that appear
       exactly in ``X_train``.
    2. **Per-feature KS test** — for every numeric column, run a
       two-sample Kolmogorov-Smirnov test; columns whose p-value is
       *below* ``distribution_alpha`` are flagged as suspiciously
       similar (the conventional alpha direction is reversed because
       leakage usually shows up as artificially-matched
       distributions).
    3. **Label-balance drift** — compare the positive class rate in
       ``y_train`` vs ``y_test``; a gap larger than
       ``label_ratio_tolerance`` is flagged.

    Parameters
    ----------
    X_train, X_test : array-like of shape (n, n_features)
    y_train, y_test : array-like of shape (n,)
    distribution_alpha : float in (0, 1), default 1e-3
        Low-p-value threshold for the per-feature KS check.
    label_ratio_tolerance : float in (0, 1), default 0.1

    Returns
    -------
    LeakageReport
    """
    X_train_arr = _as_2d(X_train)
    X_test_arr = _as_2d(X_test)
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    check_consistent_length(X_train_arr, y_train_arr)
    check_consistent_length(X_test_arr, y_test_arr)
    if X_train_arr.shape[1] != X_test_arr.shape[1]:
        raise ValueError(
            f"X_train and X_test must share n_features; got {X_train_arr.shape[1]} vs "
            f"{X_test_arr.shape[1]}."
        )

    # 1. Row overlap via sorted hashes — exact-byte equality.
    train_hashes = {bytes(row.tobytes()) for row in X_train_arr}
    n_test = X_test_arr.shape[0]
    duplicates = int(sum(1 for row in X_test_arr if bytes(row.tobytes()) in train_hashes))
    dup_rate = duplicates / n_test if n_test > 0 else 0.0

    # 2. Per-feature KS test.
    feature_alerts: list[str] = []
    for j in range(X_train_arr.shape[1]):
        col_a = X_train_arr[:, j]
        col_b = X_test_arr[:, j]
        try:
            p = float(stats.ks_2samp(col_a, col_b).pvalue)
        except (ValueError, TypeError):
            continue
        if p < distribution_alpha:
            feature_alerts.append(f"feature[{j}]: KS p-value={p:.2e}")

    # 3. Label-balance drift.
    train_rate = float(np.mean(y_train_arr == np.max(y_train_arr)))
    test_rate = float(np.mean(y_test_arr == np.max(y_test_arr)))
    label_alert = abs(train_rate - test_rate) > label_ratio_tolerance

    parts: list[str] = []
    if duplicates:
        parts.append(f"{duplicates} duplicate rows ({dup_rate:.1%})")
    if feature_alerts:
        parts.append(f"{len(feature_alerts)} feature distribution alert(s)")
    if label_alert:
        parts.append(f"label-balance drift |train={train_rate:.2f} test={test_rate:.2f}|")
    summary = "; ".join(parts) if parts else "no leakage signals detected"

    return LeakageReport(
        duplicate_rows=duplicates,
        duplicate_row_rate=dup_rate,
        feature_distribution_alerts=feature_alerts,
        label_distribution_alert=label_alert,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Subgroup fairness
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubgroupFairnessReport:
    """Output of :func:`subgroup_fairness`."""

    per_group: pd.DataFrame
    accuracy_disparity: float
    sensitivity_disparity: float
    specificity_disparity: float


def subgroup_fairness(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
) -> SubgroupFairnessReport:
    """Report per-group accuracy / sensitivity / specificity + disparities.

    Parameters
    ----------
    y_true, y_pred, groups : array-like of shape (n_samples,)
    pos_label : default 1

    Returns
    -------
    SubgroupFairnessReport
        ``per_group`` is a DataFrame indexed by group id with columns
        ``n_samples``, ``accuracy``, ``sensitivity``, ``specificity``.
        Each ``*_disparity`` is ``best - worst`` across groups (so
        ``0.0`` means perfect parity and larger values indicate
        widening gaps).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    groups_arr = np.asarray(groups)
    check_consistent_length(y_true_arr, y_pred_arr)
    check_consistent_length(y_true_arr, groups_arr)

    unique_groups = np.unique(groups_arr)
    if unique_groups.size < 1:
        raise ValueError("groups must contain at least one unique value.")

    rows = []
    for g in unique_groups:
        mask = groups_arr == g
        size = int(mask.sum())
        y_t = y_true_arr[mask]
        y_p = y_pred_arr[mask]
        if size == 0 or len(np.unique(y_t)) < 1:
            rows.append(
                {
                    "group": g,
                    "n_samples": size,
                    "accuracy": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                }
            )
            continue
        acc = float(np.mean(y_t == y_p))
        try:
            sens = sensitivity(y_t, y_p, pos_label=pos_label)
        except ValueError:
            sens = float("nan")
        try:
            spec = specificity(y_t, y_p, pos_label=pos_label)
        except ValueError:
            spec = float("nan")
        rows.append(
            {
                "group": g,
                "n_samples": size,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
            }
        )
    df = pd.DataFrame(rows).set_index("group")

    def _disparity(col: str) -> float:
        vals = df[col].dropna()
        if vals.empty:
            return float("nan")
        return float(vals.max() - vals.min())

    return SubgroupFairnessReport(
        per_group=df,
        accuracy_disparity=_disparity("accuracy"),
        sensitivity_disparity=_disparity("sensitivity"),
        specificity_disparity=_disparity("specificity"),
    )


# ---------------------------------------------------------------------------
# Documentation checklist
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DocumentationCheckResult:
    """Output of :func:`documentation_checklist`."""

    required_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    empty_fields: tuple[str, ...]
    complete: bool


_REQUIRED_RUN_FIELDS: tuple[str, ...] = (
    "run_id",
    "dataset_name",
    "dataset_hash",
    "model_class",
    "random_state",
    "cv_strategy",
    "threshold_strategy",
    "threshold",
    "accuracy",
    "sensitivity",
    "specificity",
    "notes",
)


def documentation_checklist(
    run: Mapping[str, Any],
    *,
    required_fields: tuple[str, ...] | None = None,
) -> DocumentationCheckResult:
    """Check that a run record contains the fields a research log expects.

    Parameters
    ----------
    run : mapping
        Run record — typically a dict (leaderboard row, manifest, etc.).
    required_fields : tuple of str, optional
        Defaults to a curated set covering identity, dataset, model,
        split, threshold, and headline metrics. Override when the
        downstream archive needs different fields.

    Returns
    -------
    DocumentationCheckResult
    """
    fields = required_fields or _REQUIRED_RUN_FIELDS
    missing = tuple(f for f in fields if f not in run)

    def _is_empty(value: Any) -> bool:
        if value is None or value == "":
            return True
        # NaN check via numpy (handles float NaN cleanly).
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    empty = tuple(f for f in fields if f in run and _is_empty(run[f]))
    return DocumentationCheckResult(
        required_fields=fields,
        missing_fields=missing,
        empty_fields=empty,
        complete=not (missing or empty),
    )
