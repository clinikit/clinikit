"""Boundary-refining binary classifier.

A two-stage meta-estimator: a *primary* estimator is fit on the whole
training set; samples whose primary-positive-class probability falls
within a configurable margin around 0.5 are then re-classified by a
*refinement* estimator trained only on that boundary slice. The aim
is to spend more model capacity exactly where the decision is hardest.

The class is binary-only and sklearn-compatible: it passes
``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["BoundaryRefineClassifier"]


class BoundaryRefineClassifier(ClassifierMixin, BaseEstimator):
    """Refine predictions near the decision boundary with a second model.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Primary estimator covering the easy cases. Default:
        :class:`~sklearn.linear_model.LogisticRegression`.
    refine_estimator : sklearn classifier, optional
        Boundary refiner. Default:
        ``LogisticRegression(C=10.0)`` — a slightly under-regularised
        copy of the primary, good for sharpening the local decision.
    margin : float in (0, 0.5], default 0.15
        Sample is considered "on the boundary" when
        ``|p_positive - 0.5| < margin``.
    min_boundary_samples : int, default 6
        Refinement is only attempted when at least this many boundary
        samples exist and span both classes. Below the threshold the
        primary estimator is used everywhere.

    Attributes
    ----------
    base_estimator_ : fitted primary estimator.
    refine_estimator_ : fitted refinement estimator, or ``None`` when
        the boundary slice was too small.
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import BoundaryRefineClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] + 0.5 * rng.standard_normal(100) > 0).astype(int)
    >>> clf = BoundaryRefineClassifier().fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        refine_estimator: ClassifierMixin | None = None,
        *,
        margin: float = 0.15,
        min_boundary_samples: int = 6,
    ) -> None:
        self.base_estimator = base_estimator
        self.refine_estimator = refine_estimator
        self.margin = margin
        self.min_boundary_samples = min_boundary_samples

    def fit(self, X: ArrayLike, y: ArrayLike) -> BoundaryRefineClassifier:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if not 0.0 < self.margin <= 0.5:
            raise ValueError(f"margin must lie in (0, 0.5]; got {self.margin!r}.")
        if self.min_boundary_samples < 2:
            raise ValueError(
                f"min_boundary_samples must be >= 2; got {self.min_boundary_samples!r}."
            )

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Classifier requires more than one class. Got 1 class in y.")

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)

        self.refine_estimator_: ClassifierMixin | None = None
        if not hasattr(self.base_estimator_, "predict_proba"):
            # Cannot identify boundary samples without probabilities.
            return self

        boundary_mask = self._boundary_mask(X_arr)
        if (
            boundary_mask.sum() >= self.min_boundary_samples
            and len(np.unique(y_arr[boundary_mask])) >= 2
        ):
            refine = (
                self.refine_estimator
                if self.refine_estimator is not None
                else LogisticRegression(C=10.0)
            )
            self.refine_estimator_ = clone(refine)
            self.refine_estimator_.fit(X_arr[boundary_mask], y_arr[boundary_mask])
        return self

    def _positive_proba(
        self,
        estimator: ClassifierMixin,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        proba = estimator.predict_proba(X)
        # Always select the column matching the largest class label
        # (consistent with sklearn's convention).
        pos_idx = int(np.where(estimator.classes_ == self.classes_[-1])[0][0])
        return np.asarray(proba[:, pos_idx], dtype=np.float64)

    def _boundary_mask(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        p_pos = self._positive_proba(self.base_estimator_, X)
        return np.abs(p_pos - 0.5) < self.margin

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        if not hasattr(self.base_estimator_, "predict_proba"):
            raise NotImplementedError(
                f"Base estimator {type(self.base_estimator_).__name__} has no predict_proba."
            )

        proba: NDArray[np.float64] = self.base_estimator_.predict_proba(X_arr).astype(np.float64)
        if self.refine_estimator_ is None:
            return proba

        boundary_mask = self._boundary_mask(X_arr)
        if not boundary_mask.any():
            return proba

        refined = self.refine_estimator_.predict_proba(X_arr[boundary_mask])
        # Reorder refined columns to match self.classes_ ordering.
        refine_classes = np.asarray(self.refine_estimator_.classes_)
        col_order = [int(np.where(refine_classes == c)[0][0]) for c in self.classes_]
        proba[boundary_mask] = refined[:, col_order].astype(np.float64)
        return proba

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        if hasattr(self.base_estimator_, "predict_proba"):
            proba = self.predict_proba(X)
            return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)
        # No probabilities; fall back to the primary estimator's decision.
        X_arr = validate_data(self, X, reset=False)
        return np.asarray(self.base_estimator_.predict(X_arr))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
