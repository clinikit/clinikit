"""Per-subgroup threshold binary classifier.

A meta-estimator that wraps a base classifier and learns a separate
decision threshold for every subgroup discovered in the training
data. Subgroups are inferred at fit time by a clusterer (default
:class:`~sklearn.cluster.KMeans`) on the raw feature matrix; at
predict time the clusterer assigns each new sample to its nearest
subgroup, which selects the threshold to apply.

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

from clinikit.thresholds import pick_threshold

__all__ = ["SubgroupThresholdClassifier"]


class SubgroupThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a base classifier with per-subgroup decision thresholds.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default: :class:`~sklearn.linear_model.LogisticRegression`.
    clusterer : sklearn ClusterMixin, optional
        Used to discover subgroups. Default:
        ``KMeans(n_clusters=n_subgroups)`` with the supplied
        ``random_state``.
    n_subgroups : int, default 3
        Used only when ``clusterer`` is ``None``.
    threshold_strategy : str, default "accuracy_max"
        Routed through :func:`clinikit.thresholds.pick_threshold`.
    threshold_strategy_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen strategy.
    min_subgroup_size : int, default 5
        Subgroups smaller than this fall back to ``default_threshold``.
    default_threshold : float, default 0.5
    random_state : int, optional
        Seed for the default :class:`KMeans` clusterer.

    Attributes
    ----------
    base_estimator_ : fitted base estimator.
    clusterer_ : fitted clusterer.
    thresholds_ : dict[int, float]
        Per-subgroup decision thresholds.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import SubgroupThresholdClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = SubgroupThresholdClassifier(n_subgroups=2, random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        clusterer: ClusterMixin | None = None,
        *,
        n_subgroups: int = 3,
        threshold_strategy: str = "accuracy_max",
        threshold_strategy_kwargs: dict | None = None,
        min_subgroup_size: int = 5,
        default_threshold: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.clusterer = clusterer
        self.n_subgroups = n_subgroups
        self.threshold_strategy = threshold_strategy
        self.threshold_strategy_kwargs = threshold_strategy_kwargs
        self.min_subgroup_size = min_subgroup_size
        self.default_threshold = default_threshold
        self.random_state = random_state

    def _positive_proba(
        self,
        estimator: ClassifierMixin,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        proba = estimator.predict_proba(X)
        pos_idx = int(np.where(estimator.classes_ == self.classes_[-1])[0][0])
        return np.asarray(proba[:, pos_idx], dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike) -> SubgroupThresholdClassifier:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if self.n_subgroups < 1:
            raise ValueError(f"n_subgroups must be >= 1; got {self.n_subgroups!r}.")
        if not 0.0 <= self.default_threshold <= 1.0:
            raise ValueError(
                f"default_threshold must be in [0, 1]; got {self.default_threshold!r}."
            )

        self.classes_ = np.unique(y_arr)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)
        if not hasattr(self.base_estimator_, "predict_proba"):
            raise ValueError(
                f"Base estimator {type(self.base_estimator_).__name__} must support predict_proba."
            )

        # Pick a clusterer with a sample-size-respecting fallback.
        effective_k = min(self.n_subgroups, X_arr.shape[0])
        if self.clusterer is None:
            self.clusterer_ = KMeans(
                n_clusters=max(effective_k, 1),
                n_init=10,
                random_state=self.random_state,
            )
        else:
            self.clusterer_ = clone(self.clusterer)
        # KMeans (and some clusterers) need a writable, contiguous, float64
        # buffer; check_estimator deliberately passes read-only views.
        X_for_cluster = np.ascontiguousarray(X_arr, dtype=np.float64).copy()
        self.clusterer_.fit(X_for_cluster)

        if hasattr(self.clusterer_, "labels_"):
            cluster_labels = np.asarray(self.clusterer_.labels_, dtype=np.int64)
        else:
            cluster_labels = np.asarray(self.clusterer_.predict(X_arr), dtype=np.int64)

        proba = self._positive_proba(self.base_estimator_, X_arr)
        kwargs = self.threshold_strategy_kwargs or {}
        y_pos = (y_arr == self.classes_[-1]).astype(np.int64)

        self.thresholds_: dict[int, float] = {}
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            if mask.sum() < self.min_subgroup_size or len(np.unique(y_pos[mask])) < 2:
                self.thresholds_[int(c)] = float(self.default_threshold)
                continue
            t = pick_threshold(y_pos[mask], proba[mask], strategy=self.threshold_strategy, **kwargs)
            if not isinstance(t, float):
                # cluster_specific / two_stage produce non-scalar outputs we
                # cannot use here — guard against them being routed in.
                raise ValueError(
                    f"threshold_strategy={self.threshold_strategy!r} produced a "
                    f"non-scalar threshold {t!r}; choose a strategy that returns a float."
                )
            self.thresholds_[int(c)] = t
        return self

    def _assign_clusters(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X_writable = np.ascontiguousarray(X, dtype=np.float64).copy()
        if hasattr(self.clusterer_, "predict"):
            return np.asarray(self.clusterer_.predict(X_writable), dtype=np.int64)
        # Fall back to nearest-centroid mapping when the clusterer is
        # transductive (e.g. AgglomerativeClustering).
        if hasattr(self.clusterer_, "cluster_centers_"):
            d = cdist(X_writable, self.clusterer_.cluster_centers_)
            return np.asarray(d.argmin(axis=1), dtype=np.int64)
        # Last resort: collapse to a single cluster.
        return np.zeros(X.shape[0], dtype=np.int64)

    def _per_sample_thresholds(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        clusters = self._assign_clusters(X)
        return np.array(
            [self.thresholds_.get(int(c), self.default_threshold) for c in clusters],
            dtype=np.float64,
        )

    @staticmethod
    def _rescale_proba(
        proba_pos: NDArray[np.float64],
        thresholds: NDArray[np.float64],
        eps: float = 1e-9,
    ) -> NDArray[np.float64]:
        """Map ``proba_pos`` so that ``proba_pos >= threshold`` becomes ``>= 0.5``.

        Two linear pieces glued at the per-sample threshold:
        ``[0, t] -> [0, 0.5]`` and ``[t, 1] -> [0.5, 1]``.
        """
        out = np.empty_like(proba_pos, dtype=np.float64)
        upper = proba_pos >= thresholds
        # Upper piece
        denom_upper = np.maximum(1.0 - thresholds, eps)
        out[upper] = 0.5 + 0.5 * (proba_pos[upper] - thresholds[upper]) / denom_upper[upper]
        # Lower piece
        denom_lower = np.maximum(thresholds, eps)
        out[~upper] = 0.5 * proba_pos[~upper] / denom_lower[~upper]
        return np.clip(out, 0.0, 1.0)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        raw_proba = self.base_estimator_.predict_proba(X_arr).astype(np.float64)
        p_pos = self._positive_proba(self.base_estimator_, X_arr)
        ts = self._per_sample_thresholds(X_arr)
        p_pos_new = self._rescale_proba(p_pos, ts)

        # Reconstruct full probability matrix in self.classes_ ordering.
        pos_col = int(np.where(self.classes_ == self.classes_[-1])[0][0])
        out = raw_proba.copy()
        out[:, pos_col] = p_pos_new
        # Renormalise the negative column(s) to keep rows summing to 1.
        if out.shape[1] == 2:
            out[:, 1 - pos_col] = 1.0 - p_pos_new
        else:
            mask = np.arange(out.shape[1]) != pos_col
            row_sum = out[:, mask].sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum > 0, row_sum, 1.0)
            out[:, mask] = out[:, mask] * ((1.0 - p_pos_new[:, None]) / row_sum)
        return np.asarray(out, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
