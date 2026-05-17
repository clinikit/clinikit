"""Imputers — three thin wrappers around scikit-learn plus two custom transformers.

Thin wrappers
-------------
- :class:`MedianImputer`  — ``SimpleImputer(strategy="median")`` shortcut.
- :class:`KNNImputer`     — sklearn's nearest-neighbours imputer.
- :class:`MICEImputer`    — sklearn's :class:`IterativeImputer`.

Custom
------
- :class:`MissForestImputer`  — :class:`IterativeImputer` driven by a
  Random Forest regressor (the "MissForest" recipe).
- :class:`DomainAwareImputer` — per-column strategies specified by the
  caller (string column names for DataFrames, integer indices for
  ndarrays).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as _SkIterativeImputer
from sklearn.impute import KNNImputer as _SkKNNImputer
from sklearn.impute import SimpleImputer as _SkSimpleImputer
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = [
    "DomainAwareImputer",
    "KNNImputer",
    "MICEImputer",
    "MedianImputer",
    "MissForestImputer",
]

SimpleStrategy = Literal["mean", "median", "most_frequent", "constant"]


class MedianImputer(_SkSimpleImputer):
    """Impute missing values with the per-column median.

    Convenience constructor for
    :class:`sklearn.impute.SimpleImputer` pinned to
    ``strategy="median"``.
    """

    def __init__(
        self,
        *,
        missing_values: float = np.nan,
        fill_value: float | None = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
    ) -> None:
        super().__init__(
            missing_values=missing_values,
            strategy="median",
            fill_value=fill_value,
            copy=copy,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )


class KNNImputer(_SkKNNImputer):
    """Impute missing values using k-nearest neighbours.

    Thin wrapper around :class:`sklearn.impute.KNNImputer`. See the
    upstream docs for parameter details.
    """


class MICEImputer(_SkIterativeImputer):
    """Multivariate imputation by chained equations (sklearn's IterativeImputer).

    Thin wrapper around :class:`sklearn.impute.IterativeImputer`. The
    underlying class is still flagged ``experimental`` in scikit-learn
    — re-export here gives us a stable name even if the upstream API
    moves.
    """


class MissForestImputer(_SkIterativeImputer):
    """IterativeImputer driven by a Random Forest regressor.

    Implements the "MissForest" recipe (Stekhoven & Bühlmann, 2012) by
    fixing the per-feature estimator to a
    :class:`~sklearn.ensemble.RandomForestRegressor` and adjusting
    sensible defaults. All other parameters are forwarded to
    :class:`sklearn.impute.IterativeImputer`.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees per forest.
    max_features : str, default "sqrt"
        Forest split-feature subsample policy.
    max_iter : int, default 10
        Maximum number of imputation rounds.
    random_state : int, optional
        Seed forwarded to both the imputer and the forest.

    Notes
    -----
    Random Forests are robust to monotonic transformations of the
    inputs, so unlike :class:`MICEImputer` you do not need to scale
    features beforehand.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_features: str | float | int = "sqrt",
        max_iter: int = 10,
        random_state: int | None = None,
        tol: float = 1e-3,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        super().__init__(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=random_state,
                n_jobs=1,
            ),
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )


class DomainAwareImputer(TransformerMixin, BaseEstimator):
    """Per-column imputer driven by a caller-supplied strategy map.

    The user supplies a mapping ``feature -> strategy`` where the key
    may be a column name (when ``X`` is a DataFrame) or an integer
    column index (when ``X`` is an ndarray). The value is one of the
    standard sklearn :class:`SimpleImputer` strategies: ``"mean"``,
    ``"median"``, ``"most_frequent"``, or ``("constant", <fill>)``.

    Columns not present in the map fall back to ``default_strategy``.

    Parameters
    ----------
    strategies : dict[str | int, str | tuple[str, Any]]
        Per-feature strategy map. Use ``"mean"``, ``"median"``,
        ``"most_frequent"``, or ``("constant", fill_value)``.
    default_strategy : str or tuple[str, Any], default "median"
        Strategy for features not mentioned in ``strategies``.
    missing_values : float, default ``np.nan``
        Value treated as missing.

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features,)
        Column names (for DataFrame input) or generated names.
    n_features_in_ : int
    column_imputers_ : list[SimpleImputer]
        One fitted :class:`sklearn.impute.SimpleImputer` per column.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from clinikit.preprocessing import DomainAwareImputer
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 0, 1]})
    >>> imp = DomainAwareImputer(strategies={"a": "median", "b": "most_frequent"})
    >>> out = imp.fit_transform(df)
    >>> out.shape
    (3, 2)
    """

    def __init__(
        self,
        strategies: dict[str | int, SimpleStrategy | tuple[str, Any]] | None = None,
        *,
        default_strategy: SimpleStrategy | tuple[str, Any] = "median",
        missing_values: float = np.nan,
    ) -> None:
        self.strategies = strategies
        self.default_strategy = default_strategy
        self.missing_values = missing_values

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_strategy(
        s: SimpleStrategy | tuple[str, Any],
    ) -> tuple[SimpleStrategy, Any]:
        if isinstance(s, tuple):
            if len(s) != 2 or s[0] != "constant":
                raise ValueError(
                    f"Tuple strategy must be of the form ('constant', fill_value); got {s!r}."
                )
            return "constant", s[1]
        return s, None

    def _column_strategy(self, col_key: str | int) -> tuple[SimpleStrategy, Any]:
        strategies = self.strategies or {}
        if col_key in strategies:
            return self._parse_strategy(strategies[col_key])
        return self._parse_strategy(self.default_strategy)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> DomainAwareImputer:
        had_columns = isinstance(X, pd.DataFrame)
        X_arr = validate_data(
            self,
            X,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            reset=True,
        )

        if had_columns:
            feature_names = list(self.feature_names_in_)
        else:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        self.column_imputers_ = []
        for j, name in enumerate(feature_names):
            key: str | int = name if had_columns else j
            strategy, fill_value = self._column_strategy(key)
            imputer = _SkSimpleImputer(
                missing_values=self.missing_values,
                strategy=strategy,
                fill_value=fill_value,
                keep_empty_features=True,
            )
            imputer.fit(X_arr[:, [j]])
            self.column_imputers_.append(imputer)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "column_imputers_")
        X_arr = validate_data(
            self,
            X,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            reset=False,
        )

        out = np.empty_like(X_arr, dtype=np.float64)
        for j, imputer in enumerate(self.column_imputers_):
            out[:, [j]] = imputer.transform(X_arr[:, [j]])
        return out

    def get_feature_names_out(
        self,
        input_features: ArrayLike | None = None,
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "column_imputers_")
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
