"""Class-conditional imputation binary classifier.

A meta-classifier that handles missing values *during training* by
filling them with class-conditional medians, then fits a base
classifier on the imputed data. At predict time the unknown-class
medians are blended (weighted by class priors) since the true label
is — by definition — what we are trying to predict.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["ClassConditionalImputer"]


class ClassConditionalImputer(ClassifierMixin, BaseEstimator):
    """Impute missing values using class-conditional medians, then classify.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.

    Attributes
    ----------
    base_estimator_ : fitted base classifier on the imputed data.
    class_medians_ : ndarray of shape (n_classes, n_features)
        Per-class, per-feature training medians, computed ignoring
        NaNs.
    class_priors_ : ndarray of shape (n_classes,)
        Empirical class priors used to blend predict-time imputations.
    overall_medians_ : ndarray of shape (n_features,)
        Fallback medians for features where a class had only NaNs.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import ClassConditionalImputer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 4))
    >>> X[0, 0] = np.nan
    >>> y = (X[:, 1] > 0).astype(int)
    >>> clf = ClassConditionalImputer().fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None) -> None:
        self.base_estimator = base_estimator

    def _impute_with_class(
        self,
        X: NDArray[np.float64],
        class_idx: int,
    ) -> NDArray[np.float64]:
        medians = self.class_medians_[class_idx]
        # Fall back to overall median for features where the class
        # median is NaN (e.g. an entirely-missing column for one class).
        nan_in_class = np.isnan(medians)
        medians = np.where(nan_in_class, self.overall_medians_, medians)
        out = X.copy()
        mask = np.isnan(out)
        col_idx = np.where(mask)
        out[col_idx] = medians[col_idx[1]]
        return np.asarray(out, dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike) -> ClassConditionalImputer:
        X_arr, y_arr = validate_data(self, X, y, reset=True, ensure_all_finite="allow-nan")
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        self.classes_ = np.unique(y_arr)
        n_features = X_arr.shape[1]

        self.overall_medians_ = np.nanmedian(X_arr, axis=0)
        # If a feature is entirely NaN, fall back to 0 to keep the
        # estimator well-defined.
        self.overall_medians_ = np.where(
            np.isnan(self.overall_medians_), 0.0, self.overall_medians_
        )

        self.class_medians_ = np.zeros((len(self.classes_), n_features), dtype=np.float64)
        self.class_priors_ = np.zeros(len(self.classes_), dtype=np.float64)
        for k, c in enumerate(self.classes_):
            mask = y_arr == c
            self.class_priors_[k] = mask.mean()
            if mask.any():
                self.class_medians_[k] = np.nanmedian(X_arr[mask], axis=0)
            else:
                self.class_medians_[k] = self.overall_medians_

        # Train base on globally-imputed data: impute each row with the
        # known training-time class median, so the base estimator sees a
        # finite matrix even before label-conditional inference is needed.
        imputed = np.empty_like(X_arr, dtype=np.float64)
        for k, c in enumerate(self.classes_):
            rows = y_arr == c
            if rows.any():
                imputed[rows] = self._impute_with_class(X_arr[rows], k)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(imputed, y_arr)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")

        # Each test sample's true class is unknown; produce a probability
        # for every possible imputation and weight by class prior.
        n_classes = len(self.classes_)
        accum = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)
        col_order = [
            int(np.where(np.asarray(self.base_estimator_.classes_) == c)[0][0])
            for c in self.classes_
        ]
        for k in range(n_classes):
            imputed_k = self._impute_with_class(X_arr, k)
            proba_k = self.base_estimator_.predict_proba(imputed_k)[:, col_order]
            accum += self.class_priors_[k] * proba_k

        # Re-normalise rows (they already sum to 1 in expectation, but
        # rounding may drift).
        row_sums = accum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return np.asarray(accum / row_sums, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = True
        return tags
