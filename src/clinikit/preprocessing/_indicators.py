"""Indicator transformers — flag outliers and missing values.

- :class:`MissingIndicator` is a thin re-export of sklearn's own
  :class:`sklearn.impute.MissingIndicator`.
- :class:`OutlierFlag` is a small custom transformer that produces a
  per-column 0/1 indicator using either the interquartile-range or
  z-score rule.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator as _SkMissingIndicator
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = [
    "MissingIndicator",
    "OutlierFlag",
]

OutlierMethod = Literal["iqr", "zscore"]


class MissingIndicator(_SkMissingIndicator):
    """Indicate which values in a dataset are missing.

    Thin wrapper around :class:`sklearn.impute.MissingIndicator`.
    """


class OutlierFlag(TransformerMixin, BaseEstimator):
    """Produce a 0/1 outlier indicator for every input feature.

    The fitted transformer learns per-column thresholds and at
    ``transform`` time emits an array of the same shape as the input
    where ``1`` marks an out-of-range value and ``0`` marks an in-range
    one. NaN values are treated as in-range (``0``).

    Parameters
    ----------
    method : {"iqr", "zscore"}, default "iqr"
        Outlier rule. ``"iqr"`` flags values outside
        ``[Q1 - k*IQR, Q3 + k*IQR]``; ``"zscore"`` flags values whose
        absolute z-score exceeds ``k``.
    k : float, default 1.5 (for "iqr") / 3.0 (for "zscore")
        Threshold multiplier. ``None`` picks a sensible default for
        the chosen ``method``.

    Attributes
    ----------
    n_features_in_ : int
    feature_names_in_ : ndarray of shape (n_features,)
    lower_bounds_ : ndarray of shape (n_features,)
    upper_bounds_ : ndarray of shape (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.preprocessing import OutlierFlag
    >>> X = np.array([[1.0], [1.0], [1.0], [1.0], [100.0]])
    >>> flag = OutlierFlag(method="iqr").fit_transform(X)
    >>> flag.ravel().tolist()
    [0, 0, 0, 0, 1]
    """

    def __init__(self, *, method: OutlierMethod = "iqr", k: float | None = None) -> None:
        self.method = method
        self.k = k

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        # OutlierFlag emits int64 indicators, not the input float dtype.
        tags.transformer_tags.preserves_dtype = []
        return tags

    def _resolved_k(self) -> float:
        if self.k is not None:
            return float(self.k)
        return 1.5 if self.method == "iqr" else 3.0

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> OutlierFlag:
        if self.method not in ("iqr", "zscore"):
            raise ValueError(f"Unknown method: {self.method!r}. Use 'iqr' or 'zscore'.")

        X_arr = validate_data(
            self,
            X,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            reset=True,
        )
        k = self._resolved_k()

        if self.method == "iqr":
            q1 = np.nanpercentile(X_arr, 25, axis=0)
            q3 = np.nanpercentile(X_arr, 75, axis=0)
            iqr = q3 - q1
            self.lower_bounds_ = q1 - k * iqr
            self.upper_bounds_ = q3 + k * iqr
        else:  # "zscore"
            mean = np.nanmean(X_arr, axis=0)
            std = np.nanstd(X_arr, axis=0, ddof=0)
            std = np.where(std == 0.0, 1.0, std)
            self.lower_bounds_ = mean - k * std
            self.upper_bounds_ = mean + k * std

        return self

    def transform(self, X: ArrayLike) -> NDArray[np.int64]:
        check_is_fitted(self, "lower_bounds_")
        X_arr = validate_data(
            self,
            X,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            reset=False,
        )

        below = X_arr < self.lower_bounds_
        above = X_arr > self.upper_bounds_
        out: NDArray[np.int64] = (below | above).astype(np.int64)
        # NaN values: comparisons return False on both sides so they
        # naturally become 0 — explicit for clarity.
        out[np.isnan(X_arr)] = 0
        return out

    def get_feature_names_out(
        self,
        input_features: ArrayLike | None = None,
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "lower_bounds_")
        if hasattr(self, "feature_names_in_"):
            names = list(self.feature_names_in_)
        else:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray([f"{n}_outlier" for n in names], dtype=object)
