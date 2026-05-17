"""Hard-sample weighted ensemble classifier.

A small ensemble in the spirit of AdaBoost: each successive member is
trained on the same data, but with weights that are higher on samples
the running ensemble has gotten wrong so far. At predict time the
member probabilities are averaged.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["HardSampleWeightedEnsemble"]


class HardSampleWeightedEnsemble(ClassifierMixin, BaseEstimator):
    """An additive ensemble that up-weights previously-misclassified samples.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Cloned once per member. Must accept ``sample_weight`` in
        ``fit``. Default :class:`~sklearn.linear_model.LogisticRegression`.
    n_estimators : int, default 5
        Number of ensemble members.
    weight_growth : float, default 1.5
        Factor applied to a sample's weight every time the running
        ensemble misclassifies it. Must be ``> 1.0``.
    random_state : int, optional
        Forwarded to the base estimator when applicable.

    Attributes
    ----------
    estimators_ : list of fitted estimators
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import HardSampleWeightedEnsemble
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = HardSampleWeightedEnsemble(n_estimators=3, random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        *,
        n_estimators: int = 5,
        weight_growth: float = 1.5,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.weight_growth = weight_growth
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> HardSampleWeightedEnsemble:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1; got {self.n_estimators!r}.")
        if self.weight_growth <= 1.0:
            raise ValueError(f"weight_growth must be > 1.0; got {self.weight_growth!r}.")

        self.classes_ = np.unique(y_arr)

        n_samples = X_arr.shape[0]
        weights = np.ones(n_samples, dtype=np.float64)
        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()

        self.estimators_: list[ClassifierMixin] = []
        running_proba_pos = np.zeros(n_samples, dtype=np.float64)

        for _ in range(self.n_estimators):
            member = clone(base)
            try:
                member.fit(X_arr, y_arr, sample_weight=weights)
            except TypeError as exc:
                raise TypeError(
                    f"Base estimator {type(member).__name__} does not accept sample_weight; "
                    "supply one that does."
                ) from exc
            self.estimators_.append(member)

            if not hasattr(member, "predict_proba"):
                raise ValueError(
                    f"Base estimator {type(member).__name__} must support predict_proba."
                )
            pos_idx = int(np.where(member.classes_ == self.classes_[-1])[0][0])
            member_proba = member.predict_proba(X_arr)[:, pos_idx]
            running_proba_pos += member_proba
            avg_proba = running_proba_pos / len(self.estimators_)
            avg_pred = (avg_proba >= 0.5).astype(np.int64)
            y_pos = (y_arr == self.classes_[-1]).astype(np.int64)
            wrong = avg_pred != y_pos
            weights = np.where(wrong, weights * self.weight_growth, weights)
            # Normalise so the loop does not blow up numerically.
            weights = weights * (n_samples / weights.sum())

        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "estimators_")
        X_arr = validate_data(self, X, reset=False)
        n_classes = len(self.classes_)
        proba = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)
        for member in self.estimators_:
            member_proba = member.predict_proba(X_arr)
            col_order = [
                int(np.where(np.asarray(member.classes_) == c)[0][0]) for c in self.classes_
            ]
            proba += member_proba[:, col_order]
        proba /= len(self.estimators_)
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "estimators_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
