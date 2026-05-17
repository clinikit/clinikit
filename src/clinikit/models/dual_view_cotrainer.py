"""Dual-view co-trained binary classifier.

Splits the feature space into two disjoint *views* and trains a
separate classifier on each view of the same labels. The two views
should capture complementary information; at predict time their
probabilities are averaged. This is a supervised-only specialisation
of Blum & Mitchell's (1998) co-training algorithm — without an
unlabeled-pool exchange step.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["DualViewCoTrainer"]


class DualViewCoTrainer(ClassifierMixin, BaseEstimator):
    """Average two classifiers fit on disjoint feature views.

    Parameters
    ----------
    view1_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    view2_estimator : sklearn classifier, optional
        Default same as ``view1_estimator``.
    view1_features : list[int] or None, optional
        Column indices for view 1. If ``None``, columns are split at
        the midpoint (first half to view 1, second half to view 2).
    view2_features : list[int] or None, optional
        Column indices for view 2. Required only when
        ``view1_features`` is explicitly supplied.
    random_state : int, optional

    Attributes
    ----------
    view1_estimator_, view2_estimator_ : fitted classifiers.
    view1_features_, view2_features_ : ndarray of int
        The actual feature subsets used at fit time.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import DualViewCoTrainer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 6))
    >>> y = (X[:, 0] + X[:, 3] > 0).astype(int)
    >>> clf = DualViewCoTrainer(random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        view1_estimator: ClassifierMixin | None = None,
        view2_estimator: ClassifierMixin | None = None,
        *,
        view1_features: list[int] | None = None,
        view2_features: list[int] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.view1_estimator = view1_estimator
        self.view2_estimator = view2_estimator
        self.view1_features = view1_features
        self.view2_features = view2_features
        self.random_state = random_state

    def _aligned_proba(
        self,
        estimator: ClassifierMixin,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        proba = estimator.predict_proba(X)
        col_order = [
            int(np.where(np.asarray(estimator.classes_) == c)[0][0]) for c in self.classes_
        ]
        return np.asarray(proba[:, col_order], dtype=np.float64)

    def _resolve_views(self, n_features: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if self.view1_features is None and self.view2_features is None:
            mid = max(n_features // 2, 1)
            return (
                np.arange(0, mid, dtype=np.int64),
                np.arange(mid, n_features, dtype=np.int64),
            )
        if self.view1_features is None or self.view2_features is None:
            raise ValueError(
                "view1_features and view2_features must be supplied together or both left as None."
            )
        v1 = np.asarray(self.view1_features, dtype=np.int64)
        v2 = np.asarray(self.view2_features, dtype=np.int64)
        if v1.size == 0 or v2.size == 0:
            raise ValueError("Each view must contain at least one feature index.")
        if np.intersect1d(v1, v2).size != 0:
            raise ValueError("view1_features and view2_features must be disjoint.")
        if v1.max() >= n_features or v2.max() >= n_features or v1.min() < 0 or v2.min() < 0:
            raise ValueError(f"Feature indices out of range for X with {n_features} columns.")
        return v1, v2

    def fit(self, X: ArrayLike, y: ArrayLike) -> DualViewCoTrainer:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        self.classes_ = np.unique(y_arr)
        n_features = X_arr.shape[1]
        if n_features < 2:
            raise ValueError(
                f"DualViewCoTrainer needs at least 2 features; got n_features = {n_features}."
            )

        self.view1_features_, self.view2_features_ = self._resolve_views(n_features)

        view1 = self.view1_estimator if self.view1_estimator is not None else LogisticRegression()
        view2 = self.view2_estimator if self.view2_estimator is not None else LogisticRegression()
        self.view1_estimator_ = clone(view1)
        self.view2_estimator_ = clone(view2)
        self.view1_estimator_.fit(X_arr[:, self.view1_features_], y_arr)
        self.view2_estimator_.fit(X_arr[:, self.view2_features_], y_arr)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "view1_estimator_")
        X_arr = validate_data(self, X, reset=False)
        p1 = self._aligned_proba(self.view1_estimator_, X_arr[:, self.view1_features_])
        p2 = self._aligned_proba(self.view2_estimator_, X_arr[:, self.view2_features_])
        return np.asarray(0.5 * (p1 + p2), dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "view1_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
