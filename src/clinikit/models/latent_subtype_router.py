"""Latent-subtype router (mixture-of-experts) binary classifier.

Inputs are first grouped into latent subtypes by an unsupervised
clusterer (default :class:`~sklearn.cluster.KMeans`). One *expert*
classifier is then fitted per subtype on its own slice of the data.
At predict time the clusterer assigns a subtype and the matching
expert produces the probability; subtypes with too few samples in
training fall back to a global model.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["LatentSubtypeRouter"]


class LatentSubtypeRouter(ClassifierMixin, BaseEstimator):
    """Cluster X into latent subtypes and fit one expert per subtype.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Cloned per subtype and as the fallback global model. Default:
        :class:`~sklearn.linear_model.LogisticRegression`.
    clusterer : sklearn ClusterMixin, optional
        Used to assign subtypes. Default: ``KMeans(n_clusters=n_subtypes)``.
    n_subtypes : int, default 3
    min_subtype_size : int, default 8
        Subtypes smaller than this (or single-class) are routed to the
        fallback global model instead of fitting their own expert.
    random_state : int, optional

    Attributes
    ----------
    clusterer_ : fitted clusterer.
    experts_ : dict[int, fitted classifier]
        Subtypes that received their own expert.
    fallback_ : fitted classifier
        Trained on the full data; used for subtypes without an expert
        and for unfamiliar predict-time clusters.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import LatentSubtypeRouter
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = LatentSubtypeRouter(n_subtypes=2, random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        clusterer: ClusterMixin | None = None,
        *,
        n_subtypes: int = 3,
        min_subtype_size: int = 8,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.clusterer = clusterer
        self.n_subtypes = n_subtypes
        self.min_subtype_size = min_subtype_size
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> LatentSubtypeRouter:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if self.n_subtypes < 1:
            raise ValueError(f"n_subtypes must be >= 1; got {self.n_subtypes!r}.")
        if self.min_subtype_size < 2:
            raise ValueError(f"min_subtype_size must be >= 2; got {self.min_subtype_size!r}.")

        self.classes_ = np.unique(y_arr)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")

        # Always have a fallback global model.
        self.fallback_ = clone(base)
        self.fallback_.fit(X_arr, y_arr)

        # Fit the clusterer.
        effective_k = min(self.n_subtypes, X_arr.shape[0])
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

        self.experts_: dict[int, ClassifierMixin] = {}
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() < self.min_subtype_size or len(np.unique(y_arr[mask])) < 2:
                continue
            expert = clone(base)
            expert.fit(X_arr[mask], y_arr[mask])
            self.experts_[int(c)] = expert
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
        check_is_fitted(self, "fallback_")
        X_arr = validate_data(self, X, reset=False)
        clusters = self._assign_clusters(X_arr)

        # Default: fallback for every sample.
        out = self._aligned_proba(self.fallback_, X_arr)

        for c, expert in self.experts_.items():
            mask = clusters == c
            if not mask.any():
                continue
            out[mask] = self._aligned_proba(expert, X_arr[mask])
        return np.asarray(out, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "fallback_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
