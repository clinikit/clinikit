"""Confusion-matrix-derived classification metrics.

All public functions in this module follow a uniform signature::

    metric(y_true, y_pred, *, pos_label=1, sample_weight=None,
           zero_division=0.0) -> float

For binary problems ``pos_label`` selects which label is treated as
the positive class. Multiclass labels are accepted via macro
averaging in ``balanced_accuracy`` and ``mcc`` (which is defined for
multiclass natively); the per-class metrics
(``sensitivity``, ``specificity``, ``ppv``, ``npv``, ``f2_score``)
are binary by construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import matthews_corrcoef as _sklearn_matthews_corrcoef
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "balanced_accuracy",
    "f2_score",
    "mcc",
    "npv",
    "ppv",
    "sensitivity",
    "specificity",
]


def _binary_confusion(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(tn, fp, fn, tp)`` for a binary problem.

    Computed manually with numpy so that the 2x2 layout is well-defined
    even when the input contains only one of the two classes — which
    is precisely when the zero-division branches of the public metrics
    must trigger.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    check_consistent_length(y_true_arr, y_pred_arr)

    unique = set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
    if len(unique) > 2:
        raise ValueError(
            f"Binary metric received >2 unique labels: {sorted(unique)!r}. "
            "Use a multiclass-capable metric or filter the inputs."
        )

    if sample_weight is None:
        w = np.ones(y_true_arr.shape[0], dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        check_consistent_length(y_true_arr, w)

    y_true_pos = y_true_arr == pos_label
    y_pred_pos = y_pred_arr == pos_label

    tp = float(np.sum(w[y_true_pos & y_pred_pos]))
    fp = float(np.sum(w[~y_true_pos & y_pred_pos]))
    fn = float(np.sum(w[y_true_pos & ~y_pred_pos]))
    tn = float(np.sum(w[~y_true_pos & ~y_pred_pos]))
    return tn, fp, fn, tp


def _safe_divide(num: float, den: float, zero_division: float) -> float:
    if den == 0:
        return float(zero_division)
    return float(num / den)


def sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Sensitivity, a.k.a. recall or true-positive rate.

    ``sensitivity = TP / (TP + FN)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, str, or bool, default 1
        Label of the positive class.
    sample_weight : array-like of shape (n_samples,), optional
        Per-sample weights.
    zero_division : float, default 0.0
        Value returned when ``TP + FN == 0``.

    Returns
    -------
    float
        Sensitivity in ``[0, 1]``.

    Examples
    --------
    >>> from clinikit.metrics import sensitivity
    >>> sensitivity([1, 1, 0, 0], [1, 0, 0, 0])
    0.5
    """
    _, _, fn, tp = _binary_confusion(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
    )
    return _safe_divide(tp, tp + fn, zero_division)


def specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Specificity, a.k.a. true-negative rate.

    ``specificity = TN / (TN + FP)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Ground-truth and predicted labels.
    pos_label : int, str, or bool, default 1
        Label of the positive class.
    sample_weight : array-like, optional
    zero_division : float, default 0.0
        Value returned when ``TN + FP == 0``.

    Returns
    -------
    float
        Specificity in ``[0, 1]``.

    Examples
    --------
    >>> from clinikit.metrics import specificity
    >>> specificity([0, 0, 0, 1], [0, 1, 0, 1])
    0.6666666666666666
    """
    tn, fp, _, _ = _binary_confusion(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
    )
    return _safe_divide(tn, tn + fp, zero_division)


def ppv(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Positive predictive value (precision).

    ``ppv = TP / (TP + FP)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    pos_label : int, str, or bool, default 1
    sample_weight : array-like, optional
    zero_division : float, default 0.0
        Value returned when ``TP + FP == 0``.

    Returns
    -------
    float
        PPV in ``[0, 1]``.
    """
    _, fp, _, tp = _binary_confusion(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
    )
    return _safe_divide(tp, tp + fp, zero_division)


def npv(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Negative predictive value.

    ``npv = TN / (TN + FN)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    pos_label : int, str, or bool, default 1
    sample_weight : array-like, optional
    zero_division : float, default 0.0
        Value returned when ``TN + FN == 0``.

    Returns
    -------
    float
        NPV in ``[0, 1]``.
    """
    tn, _, fn, _ = _binary_confusion(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
    )
    return _safe_divide(tn, tn + fn, zero_division)


def f2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """F-beta score with ``beta = 2``.

    F2 weights recall four times as heavily as precision::

        F2 = 5 * P * R / (4 * P + R)

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    pos_label : int, str, or bool, default 1
    sample_weight : array-like, optional
    zero_division : float, default 0.0
        Value returned when both precision and recall are zero.

    Returns
    -------
    float
        F2 in ``[0, 1]``.
    """
    p = ppv(
        y_true,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    r = sensitivity(
        y_true,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    beta_sq = 4.0
    return _safe_divide((1.0 + beta_sq) * p * r, beta_sq * p + r, zero_division)


def mcc(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Matthews correlation coefficient.

    Equivalent to ``sklearn.metrics.matthews_corrcoef`` but returns a
    Python float. Defined for both binary and multiclass problems.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    sample_weight : array-like, optional

    Returns
    -------
    float
        MCC in ``[-1, 1]``.
    """
    return float(
        _sklearn_matthews_corrcoef(
            np.asarray(y_true),
            np.asarray(y_pred),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )
    )


def balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: int | str | bool = 1,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Balanced accuracy for binary problems.

    Defined as the arithmetic mean of sensitivity and specificity.
    For multiclass problems, use ``sklearn.metrics.balanced_accuracy_score``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    pos_label : int, str, or bool, default 1
    sample_weight : array-like, optional

    Returns
    -------
    float
        Balanced accuracy in ``[0, 1]``.
    """
    sens = sensitivity(
        y_true,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
        zero_division=0.0,
    )
    spec = specificity(
        y_true,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
        zero_division=0.0,
    )
    return 0.5 * (sens + spec)
