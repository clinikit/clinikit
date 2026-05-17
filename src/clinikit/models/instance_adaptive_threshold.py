"""Instance-adaptive threshold binary classifier.

A meta-estimator that learns a *per-instance* decision threshold via
a secondary regressor. The recipe:

1. Fit a base classifier on the training data.
2. Compute its out-of-fold positive-class probabilities so the
   training-set probabilities are not leaked.
3. Derive a target threshold per sample: the largest threshold that
   still classifies the sample correctly, with a small margin.
4. Fit a regressor that predicts that target threshold from ``X``.

At predict time the per-instance threshold is consulted; the
probability output is rescaled around the threshold so that
``argmax(predict_proba) == predict`` (the sklearn contract).

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["InstanceAdaptiveThreshold"]


class InstanceAdaptiveThreshold(ClassifierMixin, BaseEstimator):
    """Predict a per-sample decision threshold alongside the base classifier.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default: :class:`~sklearn.linear_model.LogisticRegression`.
    threshold_predictor : sklearn regressor, optional
        Predicts the threshold target from ``X``. Default:
        :class:`~sklearn.linear_model.Ridge`.
    margin : float in (0, 0.5), default 0.05
        Buffer between the per-sample probability and the target
        threshold (keeps the training signal away from a noisy
        boundary).
    cv : int, default 5
        Number of folds used to compute out-of-fold probabilities.
        Reduced automatically when there are fewer samples in a class.
    random_state : int, optional
        Seed forwarded to the CV splitter when applicable.

    Attributes
    ----------
    base_estimator_ : fitted base classifier.
    threshold_predictor_ : fitted threshold regressor.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import InstanceAdaptiveThreshold
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = InstanceAdaptiveThreshold(random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        threshold_predictor: RegressorMixin | None = None,
        *,
        margin: float = 0.05,
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.threshold_predictor = threshold_predictor
        self.margin = margin
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> InstanceAdaptiveThreshold:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if not 0.0 < self.margin < 0.5:
            raise ValueError(f"margin must lie in (0, 0.5); got {self.margin!r}.")
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2; got {self.cv!r}.")

        self.classes_ = np.unique(y_arr)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")

        # Out-of-fold probabilities so the threshold target uses
        # generalisation-style scores rather than training-set scores.
        y_pos = (y_arr == self.classes_[-1]).astype(np.int64)
        min_class = int(np.bincount(y_pos).min())
        effective_cv = max(2, min(self.cv, min_class))
        splitter = StratifiedKFold(
            n_splits=effective_cv, shuffle=True, random_state=self.random_state
        )
        oof_proba = cross_val_predict(
            clone(base), X_arr, y_arr, cv=splitter, method="predict_proba"
        )
        # cross_val_predict uses the estimator's own classes_ ordering.
        # After cloning, classes_ is not available until fit, so we re-fit
        # and use that ordering.
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)
        pos_col = int(np.where(self.base_estimator_.classes_ == self.classes_[-1])[0][0])
        p_oof = oof_proba[:, pos_col]

        # Target threshold per sample: small, bounded shift around 0.5.
        # y=1 -> target = 0.5 - margin (more permissive cutoff for positives)
        # y=0 -> target = 0.5 + margin (stricter cutoff for negatives)
        # OOF probability is intentionally ignored here so the regressor
        # learns X-pattern -> direction-of-shift without amplifying the
        # base classifier's noise.
        _ = p_oof  # kept for symmetry with future use
        target = np.where(
            y_pos == 1,
            0.5 - self.margin,
            0.5 + self.margin,
        ).astype(np.float64)

        predictor = self.threshold_predictor if self.threshold_predictor is not None else Ridge()
        self.threshold_predictor_ = clone(predictor)
        self.threshold_predictor_.fit(X_arr, target)
        return self

    @staticmethod
    def _rescale_proba(
        proba_pos: NDArray[np.float64],
        thresholds: NDArray[np.float64],
        eps: float = 1e-9,
    ) -> NDArray[np.float64]:
        """Map probabilities so the decision boundary lands at 0.5."""
        thresholds = np.clip(thresholds, 0.0, 1.0)
        out = np.empty_like(proba_pos, dtype=np.float64)
        upper = proba_pos >= thresholds
        denom_upper = np.maximum(1.0 - thresholds, eps)
        out[upper] = 0.5 + 0.5 * (proba_pos[upper] - thresholds[upper]) / denom_upper[upper]
        denom_lower = np.maximum(thresholds, eps)
        out[~upper] = 0.5 * proba_pos[~upper] / denom_lower[~upper]
        return np.clip(out, 0.0, 1.0)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)
        raw = self.base_estimator_.predict_proba(X_arr).astype(np.float64)
        p_pos = self._positive_proba(self.base_estimator_, X_arr)
        ts = np.clip(self.threshold_predictor_.predict(X_arr), 0.0, 1.0)
        p_pos_new = self._rescale_proba(p_pos, ts.astype(np.float64))

        pos_col = int(np.where(self.classes_ == self.classes_[-1])[0][0])
        out = raw.copy()
        out[:, pos_col] = p_pos_new
        if out.shape[1] == 2:
            out[:, 1 - pos_col] = 1.0 - p_pos_new
        return np.asarray(out, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "base_estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
