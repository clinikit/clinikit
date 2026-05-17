"""Statistical tests for classifier comparison.

Three tests, all classical and well-documented:

- :func:`delong_test` — paired comparison of two ROC curves
  (DeLong, DeLong & Clarke-Pearson 1988; fast variant via Sun & Xu 2014).
- :func:`bootstrap_ci` — percentile bootstrap confidence interval for
  an arbitrary binary metric.
- :func:`mcnemar_test` — paired contingency-table test for two
  hard-prediction classifiers on the same labelled data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "BootstrapCI",
    "DelongResult",
    "McNemarResult",
    "bootstrap_ci",
    "delong_test",
    "mcnemar_test",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_paired_scores(
    y_true: ArrayLike,
    y_prob_a: ArrayLike,
    y_prob_b: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true)
    a = np.asarray(y_prob_a, dtype=np.float64)
    b = np.asarray(y_prob_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("y_prob_a and y_prob_b must be 1-D arrays.")
    check_consistent_length(y_true_arr, a)
    check_consistent_length(y_true_arr, b)
    unique = np.unique(y_true_arr)
    if len(unique) != 2:
        raise ValueError(f"delong_test is binary-only; got {len(unique)} unique labels.")
    pos_label = unique[-1]
    y_binary = (y_true_arr == pos_label).astype(np.int64)
    return y_binary, a, b


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DelongResult:
    """Output of :func:`delong_test`."""

    auc_a: float
    auc_b: float
    z: float
    p_value: float
    var_a: float
    var_b: float
    covariance: float


def _delong_components(
    pos_scores: NDArray[np.float64],
    neg_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mann-Whitney midrank components for one scorer.

    Returns
    -------
    V10 : ndarray of shape (n_pos,)
        For each positive sample, average over negatives of
        ``I(pos>neg) + 0.5 I(pos==neg)``.
    V01 : ndarray of shape (n_neg,)
        Symmetric — for each negative sample, averaged over positives.
    """
    n_pos = pos_scores.size
    n_neg = neg_scores.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong requires at least one sample per class.")
    V10 = np.empty(n_pos, dtype=np.float64)
    V01 = np.empty(n_neg, dtype=np.float64)
    for i, s in enumerate(pos_scores):
        gt = float((s > neg_scores).sum())
        eq = float((s == neg_scores).sum())
        V10[i] = (gt + 0.5 * eq) / n_neg
    for j, s in enumerate(neg_scores):
        gt = float((pos_scores > s).sum())
        eq = float((pos_scores == s).sum())
        V01[j] = (gt + 0.5 * eq) / n_pos
    return V10, V01


def delong_test(
    y_true: ArrayLike,
    y_prob_a: ArrayLike,
    y_prob_b: ArrayLike,
) -> DelongResult:
    """Paired DeLong test for two ROC AUCs on the same labels.

    Parameters
    ----------
    y_true, y_prob_a, y_prob_b : array-like of shape (n_samples,)

    Returns
    -------
    DelongResult
        ``z`` and ``p_value`` are the two-sided test of
        ``AUC_a == AUC_b``.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.statistics import delong_test
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, size=200)
    >>> pa = rng.uniform(0, 1, size=200)
    >>> pb = rng.uniform(0, 1, size=200)
    >>> r = delong_test(y, pa, pb)
    >>> 0.0 <= r.p_value <= 1.0
    True
    """
    y_binary, a, b = _validate_paired_scores(y_true, y_prob_a, y_prob_b)
    pos_mask = y_binary == 1
    pos_a, neg_a = a[pos_mask], a[~pos_mask]
    pos_b, neg_b = b[pos_mask], b[~pos_mask]

    V10_a, V01_a = _delong_components(pos_a, neg_a)
    V10_b, V01_b = _delong_components(pos_b, neg_b)

    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    n_pos = V10_a.size
    n_neg = V01_a.size
    S_pos = np.cov(np.vstack([V10_a, V10_b]), ddof=1) / n_pos
    S_neg = np.cov(np.vstack([V01_a, V01_b]), ddof=1) / n_neg
    S = S_pos + S_neg
    var_a = float(S[0, 0])
    var_b = float(S[1, 1])
    cov_ab = float(S[0, 1])

    denom = var_a + var_b - 2.0 * cov_ab
    z = 0.0 if denom <= 0 else (auc_a - auc_b) / float(np.sqrt(denom))
    p_value = 2.0 * float(stats.norm.sf(abs(z)))

    return DelongResult(
        auc_a=auc_a,
        auc_b=auc_b,
        z=z,
        p_value=p_value,
        var_a=var_a,
        var_b=var_b,
        covariance=cov_ab,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BootstrapCI:
    """Output of :func:`bootstrap_ci`."""

    point: float
    lower: float
    upper: float
    alpha: float
    n_resamples: int


def bootstrap_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metric: Callable[..., float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
    stratify: bool = True,
) -> BootstrapCI:
    """Percentile-bootstrap CI for a binary metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_score : array-like of shape (n_samples,)
        Either probabilities or hard predictions, depending on
        ``metric``.
    metric : callable
        ``metric(y_true_resampled, y_score_resampled) -> float``.
    n_resamples : int, default 1000
    alpha : float in (0, 1), default 0.05
        Two-sided coverage gap; CI level is ``1 - alpha``.
    random_state : int, optional
    stratify : bool, default True
        Stratify each resample by the original class proportions.

    Returns
    -------
    BootstrapCI
    """
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be >= 2; got {n_resamples!r}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    check_consistent_length(y_true_arr, y_score_arr)
    rng = np.random.default_rng(random_state)
    n = y_true_arr.shape[0]

    point = float(metric(y_true_arr, y_score_arr))

    samples = np.empty(n_resamples, dtype=np.float64)
    if stratify:
        unique = np.unique(y_true_arr)
        idx_per_class = {c: np.where(y_true_arr == c)[0] for c in unique}
    else:
        idx_per_class = None

    for i in range(n_resamples):
        if stratify and idx_per_class is not None:
            chosen: NDArray[np.int64] = np.concatenate(
                [
                    rng.choice(idx_per_class[c], size=idx_per_class[c].size, replace=True)
                    for c in idx_per_class
                ]
            )
        else:
            chosen = np.asarray(rng.integers(0, n, size=n), dtype=np.int64)
        samples[i] = float(metric(y_true_arr[chosen], y_score_arr[chosen]))

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(point=point, lower=lower, upper=upper, alpha=alpha, n_resamples=n_resamples)


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class McNemarResult:
    """Output of :func:`mcnemar_test`."""

    statistic: float
    p_value: float
    b: int
    c: int


def mcnemar_test(
    y_true: ArrayLike,
    y_pred_a: ArrayLike,
    y_pred_b: ArrayLike,
    *,
    continuity_correction: bool = True,
) -> McNemarResult:
    """Paired McNemar test for two classifiers on the same labels.

    The :math:`b` and :math:`c` cells of the contingency table count
    samples where exactly one of the two classifiers was correct.

    Parameters
    ----------
    y_true, y_pred_a, y_pred_b : array-like of shape (n_samples,)
    continuity_correction : bool, default True
        Apply Yates's correction (recommended for small samples).

    Returns
    -------
    McNemarResult
    """
    y_true_arr = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b_arr = np.asarray(y_pred_b)
    check_consistent_length(y_true_arr, a)
    check_consistent_length(y_true_arr, b_arr)

    correct_a = a == y_true_arr
    correct_b = b_arr == y_true_arr
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, b=b, c=c)

    if continuity_correction:
        statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    else:
        statistic = float((b - c) ** 2) / (b + c)
    p_value = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(statistic=float(statistic), p_value=p_value, b=b, c=c)
