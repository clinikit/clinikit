"""Selective binary classifier with an abstention option.

Wraps a base classifier and exposes a separate :meth:`predict_with_abstain`
that returns ``-1`` for low-confidence inputs (those whose positive-class
probability falls within ``[1 - coverage, coverage]`` around 0.5 after
calibration on the training data).

The standard :meth:`predict` is left exactly equivalent to the base
classifier's so the class still passes ``check_estimator`` — abstention
is an opt-in feature exposed via the extra method.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["SelectiveClassifier"]


class SelectiveClassifier(ClassifierMixin, BaseEstimator):
    """A classifier that may abstain on low-confidence inputs.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    coverage : float in (0.5, 1.0], default 0.95
        Target coverage rate over the training data. The class
        calibrates an abstention band whose centre is 0.5 and whose
        width gives roughly ``coverage`` of training samples a
        non-abstain decision.

    Attributes
    ----------
    base_estimator_ : fitted base classifier.
    abstain_low_ : float
    abstain_high_ : float
        Probability range ``[abstain_low_, abstain_high_]`` is the
        abstention band; samples whose positive-class probability
        lands inside the band are abstained on by
        :meth:`predict_with_abstain`.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import SelectiveClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((120, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = SelectiveClassifier(coverage=0.9).fit(X, y)
    >>> preds = clf.predict_with_abstain(X[:10])
    >>> set(preds.tolist()) <= {-1, 0, 1}
    True
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        *,
        coverage: float = 0.95,
    ) -> None:
        self.base_estimator = base_estimator
        self.coverage = coverage

    def _positive_proba(
        self,
        estimator: ClassifierMixin,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        proba = estimator.predict_proba(X)
        pos_idx = int(np.where(estimator.classes_ == self.classes_[-1])[0][0])
        return np.asarray(proba[:, pos_idx], dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike) -> SelectiveClassifier:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if not 0.5 < self.coverage <= 1.0:
            raise ValueError(f"coverage must be in (0.5, 1.0]; got {self.coverage!r}.")

        self.classes_ = np.unique(y_arr)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)

        p_pos = self._positive_proba(self.base_estimator_, X_arr)
        # Centre the abstention band on 0.5; choose its half-width so
        # roughly (1 - coverage) of training samples fall inside it.
        distances = np.abs(p_pos - 0.5)
        # Quantile of (1 - coverage) gives the threshold below which
        # distances are smallest (those samples abstain). Coverage=1
        # collapses the band to a point.
        half = (
            0.0
            if self.coverage == 1.0
            else float(np.quantile(distances, 1.0 - self.coverage))
        )
        self.abstain_low_ = max(0.0, 0.5 - half)
        self.abstain_high_ = min(1.0, 0.5 + half)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        proba = self.base_estimator_.predict_proba(X_arr)
        col_order = [
            int(np.where(np.asarray(self.base_estimator_.classes_) == c)[0][0])
            for c in self.classes_
        ]
        return np.asarray(proba[:, col_order], dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def predict_with_abstain(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return predictions in ``self.classes_`` or ``-1`` for abstentions.

        A sample abstains when its positive-class probability lies in
        ``[abstain_low_, abstain_high_]``.
        """
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        proba_pos = self._positive_proba(self.base_estimator_, X_arr)
        out = np.where(proba_pos >= 0.5, self.classes_[-1], self.classes_[0]).astype(np.int64)
        abstain_mask = (proba_pos >= self.abstain_low_) & (proba_pos <= self.abstain_high_)
        out = np.where(abstain_mask, -1, out)
        return np.asarray(out, dtype=np.int64)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
