"""Monotonic gradient-boosted binary classifier.

A thin wrapper around
:class:`~sklearn.ensemble.HistGradientBoostingClassifier` that exposes
per-feature monotonic constraints as a small, declarative parameter.
The classifier is binary-only and sklearn-compatible.

Why this exists: sklearn's HistGradientBoostingClassifier already
supports monotonic constraints, but the constraint vector must be the
exact length of the input — easy to misconfigure when feature columns
move. This wrapper accepts a dict ``{feature_index: direction}`` and
expands it at fit time, so missing entries default to "no constraint"
and a wrong constraint shape raises a helpful error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["MonotonicBooster"]

MonotonicSpec = dict[int, int]


class MonotonicBooster(ClassifierMixin, BaseEstimator):
    """Gradient boosting with declarative monotonic feature constraints.

    Parameters
    ----------
    monotonic_constraints : dict[int, int], optional
        Maps feature index to ``+1`` (non-decreasing) or ``-1``
        (non-increasing); ``0`` (unconstrained) is the default for
        features not listed. Out-of-range indices raise.
    learning_rate : float, default 0.1
    max_iter : int, default 100
    max_depth : int, optional
    random_state : int, optional

    Attributes
    ----------
    estimator_ : fitted ``HistGradientBoostingClassifier``.
    monotonic_vector_ : ndarray of shape (n_features,)
        The constraint vector actually passed to the underlying model.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import MonotonicBooster
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = MonotonicBooster(
    ...     monotonic_constraints={0: 1}, random_state=0
    ... ).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        monotonic_constraints: MonotonicSpec | None = None,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.monotonic_constraints = monotonic_constraints
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_constraint_vector(self, n_features: int) -> NDArray[np.int64]:
        vec = np.zeros(n_features, dtype=np.int64)
        if not self.monotonic_constraints:
            return vec
        for idx, direction in self.monotonic_constraints.items():
            if not isinstance(idx, int) or idx < 0 or idx >= n_features:
                raise ValueError(
                    f"monotonic_constraints index {idx!r} is out of range for "
                    f"n_features = {n_features}."
                )
            if direction not in (-1, 0, 1):
                raise ValueError(
                    f"monotonic_constraints direction must be -1, 0, or 1; "
                    f"got {direction!r} for feature {idx}."
                )
            vec[idx] = direction
        return vec

    def fit(self, X: ArrayLike, y: ArrayLike) -> MonotonicBooster:
        X_arr, y_arr = validate_data(self, X, y, reset=True, ensure_all_finite="allow-nan")
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        self.classes_ = np.unique(y_arr)
        self.monotonic_vector_ = self._build_constraint_vector(X_arr.shape[1])

        self.estimator_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            monotonic_cst=self.monotonic_vector_.tolist(),
            random_state=self.random_state,
        )
        self.estimator_.fit(X_arr, y_arr)
        return self

    @property
    def n_iter_(self) -> int:
        check_is_fitted(self, "estimator_")
        return int(getattr(self.estimator_, "n_iter_", self.max_iter))

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "estimator_")
        X_arr = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        proba = self.estimator_.predict_proba(X_arr)
        col_order = [
            int(np.where(np.asarray(self.estimator_.classes_) == c)[0][0]) for c in self.classes_
        ]
        return np.asarray(proba[:, col_order], dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = True
        return tags
