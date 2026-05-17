"""Error-aware calibrator binary classifier.

A post-hoc calibrator that conditions on regions of the input space
where the base estimator is historically prone to error. The recipe:

1. Fit a base classifier on the training data.
2. Get its out-of-fold probabilities so the calibration signal is
   not leaked from the training set.
3. Cluster the inputs and fit a separate isotonic regression
   calibrator per cluster, with a global fallback.

At predict time each sample is routed to its cluster's isotonic
calibrator (or the fallback) before producing the final probability.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, clone
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["ErrorAwareCalibrator"]


class ErrorAwareCalibrator(ClassifierMixin, BaseEstimator):
    """Per-cluster isotonic calibration on top of a base classifier.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    clusterer : sklearn ClusterMixin, optional
        Default ``KMeans(n_clusters=n_clusters)``.
    n_clusters : int, default 3
    min_cluster_size : int, default 10
        Clusters smaller than this use the global fallback calibrator.
    cv : int, default 5
        Folds for out-of-fold base probabilities.
    random_state : int, optional

    Attributes
    ----------
    base_estimator_ : fitted base classifier.
    clusterer_ : fitted clusterer.
    fallback_calibrator_ : fitted isotonic calibrator on all OOF data.
    cluster_calibrators_ : dict[int, IsotonicRegression]
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import ErrorAwareCalibrator
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((120, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = ErrorAwareCalibrator(n_clusters=2, random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        clusterer: ClusterMixin | None = None,
        *,
        n_clusters: int = 3,
        min_cluster_size: int = 10,
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.cv = cv
        self.random_state = random_state

    def _positive_proba(
        self,
        estimator: ClassifierMixin,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        proba = estimator.predict_proba(X)
        pos_idx = int(np.where(estimator.classes_ == self.classes_[-1])[0][0])
        return np.asarray(proba[:, pos_idx], dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike) -> ErrorAwareCalibrator:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1; got {self.n_clusters!r}.")
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2; got {self.min_cluster_size!r}.")
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2; got {self.cv!r}.")

        self.classes_ = np.unique(y_arr)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")

        # OOF probabilities of the positive class.
        y_pos = (y_arr == self.classes_[-1]).astype(np.int64)
        min_class = int(np.bincount(y_pos).min())
        effective_cv = max(2, min(self.cv, min_class))
        splitter = StratifiedKFold(
            n_splits=effective_cv, shuffle=True, random_state=self.random_state
        )
        oof = cross_val_predict(clone(base), X_arr, y_arr, cv=splitter, method="predict_proba")
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)
        pos_col = int(np.where(self.base_estimator_.classes_ == self.classes_[-1])[0][0])
        p_oof = oof[:, pos_col]

        # Global fallback calibrator on all OOF data.
        self.fallback_calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.fallback_calibrator_.fit(p_oof, y_pos)

        # Cluster the inputs.
        effective_k = min(self.n_clusters, X_arr.shape[0])
        if self.clusterer is None:
            self.clusterer_ = KMeans(
                n_clusters=max(effective_k, 1),
                n_init=10,
                random_state=self.random_state,
            )
        else:
            self.clusterer_ = clone(self.clusterer)
        X_for_cluster = np.ascontiguousarray(X_arr, dtype=np.float64).copy()
        self.clusterer_.fit(X_for_cluster)
        labels = (
            np.asarray(self.clusterer_.labels_, dtype=np.int64)
            if hasattr(self.clusterer_, "labels_")
            else np.asarray(self.clusterer_.predict(X_for_cluster), dtype=np.int64)
        )

        self.cluster_calibrators_: dict[int, IsotonicRegression] = {}
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() < self.min_cluster_size or len(np.unique(y_pos[mask])) < 2:
                continue
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            cal.fit(p_oof[mask], y_pos[mask])
            self.cluster_calibrators_[int(c)] = cal
        return self

    def _assign_clusters(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X_writable = np.ascontiguousarray(X, dtype=np.float64).copy()
        if hasattr(self.clusterer_, "predict"):
            return np.asarray(self.clusterer_.predict(X_writable), dtype=np.int64)
        if hasattr(self.clusterer_, "cluster_centers_"):
            d = cdist(X_writable, self.clusterer_.cluster_centers_)
            return np.asarray(d.argmin(axis=1), dtype=np.int64)
        return np.zeros(X.shape[0], dtype=np.int64)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        p_pos = self._positive_proba(self.base_estimator_, X_arr)
        clusters = self._assign_clusters(X_arr)

        # Start with fallback-calibrated positive probability.
        p_cal = self.fallback_calibrator_.transform(p_pos)

        for c, cal in self.cluster_calibrators_.items():
            mask = clusters == c
            if not mask.any():
                continue
            p_cal[mask] = cal.transform(p_pos[mask])

        p_cal = np.clip(p_cal, 0.0, 1.0)
        out = np.empty((X_arr.shape[0], 2), dtype=np.float64)
        pos_col = int(np.where(self.classes_ == self.classes_[-1])[0][0])
        out[:, pos_col] = p_cal
        out[:, 1 - pos_col] = 1.0 - p_cal
        return out

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
