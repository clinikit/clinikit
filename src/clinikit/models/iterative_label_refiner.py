"""Iterative label-refinement binary classifier.

Self-training-style loop that, at each iteration, retrains the base
classifier and flips a small number of training labels for which the
model is highly confident but the training label disagrees. The
intuition is that a small fraction of training labels may be noisy
and re-labelling the most-disagreed-with rows can improve test-time
calibration.

The number of flips per iteration is bounded by ``max_flips_per_iter``
and the total flips are bounded by ``max_total_flips``, both small by
default so the refinement does not run away.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["IterativeLabelRefiner"]


class IterativeLabelRefiner(ClassifierMixin, BaseEstimator):
    """Iteratively re-label the most confidently-wrong training samples.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    max_iter : int, default 3
    flip_threshold : float in (0.5, 1.0], default 0.85
        Only flip a label when the model's confidence in the *other*
        class is at least this large.
    max_flips_per_iter : int, default 5
        Cap on how many labels can be flipped per iteration.
    max_total_flips_fraction : float in (0, 0.5], default 0.1
        Hard ceiling on the fraction of the training set that may be
        flipped across all iterations combined.
    random_state : int, optional

    Attributes
    ----------
    estimator_ : the classifier fitted on the (possibly refined) labels.
    refined_labels_ : ndarray
        The labels actually used in the final fit (possibly different
        from the labels supplied to fit).
    flips_ : int
        Total number of label flips applied across all iterations.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import IterativeLabelRefiner
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> y[0] = 1 - y[0]  # planted noisy label
    >>> clf = IterativeLabelRefiner(random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        *,
        max_iter: int = 3,
        flip_threshold: float = 0.85,
        max_flips_per_iter: int = 5,
        max_total_flips_fraction: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.max_iter = max_iter
        self.flip_threshold = flip_threshold
        self.max_flips_per_iter = max_flips_per_iter
        self.max_total_flips_fraction = max_total_flips_fraction
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> IterativeLabelRefiner:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter!r}.")
        if not 0.5 < self.flip_threshold <= 1.0:
            raise ValueError(f"flip_threshold must be in (0.5, 1.0]; got {self.flip_threshold!r}.")
        if self.max_flips_per_iter < 0:
            raise ValueError(f"max_flips_per_iter must be >= 0; got {self.max_flips_per_iter!r}.")
        if not 0.0 < self.max_total_flips_fraction <= 0.5:
            raise ValueError(
                f"max_total_flips_fraction must be in (0, 0.5]; "
                f"got {self.max_total_flips_fraction!r}."
            )

        self.classes_ = np.unique(y_arr)
        n_samples = X_arr.shape[0]
        max_total = max(int(np.floor(self.max_total_flips_fraction * n_samples)), 0)

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        if not hasattr(base, "predict_proba"):
            raise ValueError(f"Base estimator {type(base).__name__} must support predict_proba.")

        labels = y_arr.copy()
        self.flips_ = 0
        iterations_run = 0

        for _ in range(self.max_iter):
            iterations_run += 1
            self.estimator_ = clone(base)
            self.estimator_.fit(X_arr, labels)

            # Stop if every class is no longer represented after flips.
            if len(np.unique(labels)) < 2:
                break

            proba = self._aligned_proba(self.estimator_, X_arr)
            pred_idx = np.argmax(proba, axis=1)
            pred_labels = self.classes_[pred_idx]
            disagree = pred_labels != labels
            confidences = proba[np.arange(n_samples), pred_idx]

            candidate_idx = np.where(disagree & (confidences >= self.flip_threshold))[0]
            if candidate_idx.size == 0:
                break

            # Flip the most confident disagreements first, up to the
            # per-iteration cap and the global cap.
            order = np.argsort(-confidences[candidate_idx])
            allowed_now = min(
                self.max_flips_per_iter,
                max_total - self.flips_,
                candidate_idx.size,
            )
            if allowed_now <= 0:
                break
            chosen = candidate_idx[order[:allowed_now]]
            labels[chosen] = pred_labels[chosen]
            self.flips_ += int(allowed_now)

        # Final fit on whatever labels survived the loop.
        self.estimator_ = clone(base)
        self.estimator_.fit(X_arr, labels)
        self.refined_labels_ = labels
        self.n_iter_ = max(iterations_run, 1)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "estimator_")
        X_arr = validate_data(self, X, reset=False)
        return self._aligned_proba(self.estimator_, X_arr)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "estimator_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
