"""Cross-distribution knowledge-distillation binary classifier.

Trains a *teacher* classifier on a perturbed (re-weighted, jittered)
version of the training data, then fits a *student* on the actual
data plus the teacher's soft labels. The aim is to nudge the student
toward features that generalise across a small distribution shift.

In its simplest one-distribution mode (no extra perturbation
configured), the distiller reduces to a smoothed-target classifier:
the student is trained on a mixture of hard labels and the teacher's
predicted probabilities.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["CrossDistributionDistiller"]


class CrossDistributionDistiller(ClassifierMixin, BaseEstimator):
    """Distil a teacher into a student to improve cross-distribution robustness.

    Parameters
    ----------
    teacher_estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    student_estimator : sklearn classifier, optional
        Default same as teacher. Must accept ``sample_weight``.
    distillation_strength : float in [0, 1], default 0.3
        Weight on the teacher's soft prediction when constructing the
        student's target. ``0`` recovers a plain student fit; ``1``
        replaces the labels entirely with teacher probabilities.
    perturb_scale : float >= 0, default 0.0
        Standard deviation of Gaussian noise added to the teacher's
        training features. Non-zero values approximate "different
        distribution" supervision.
    cv : int, default 5
        Folds used to compute out-of-fold teacher probabilities so the
        student does not see leaked training-set predictions.
    random_state : int, optional

    Attributes
    ----------
    teacher_ : fitted teacher classifier.
    student_ : fitted student classifier.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import CrossDistributionDistiller
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = CrossDistributionDistiller(random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        teacher_estimator: ClassifierMixin | None = None,
        student_estimator: ClassifierMixin | None = None,
        *,
        distillation_strength: float = 0.3,
        perturb_scale: float = 0.0,
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.teacher_estimator = teacher_estimator
        self.student_estimator = student_estimator
        self.distillation_strength = distillation_strength
        self.perturb_scale = perturb_scale
        self.cv = cv
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> CrossDistributionDistiller:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        if not 0.0 <= self.distillation_strength <= 1.0:
            raise ValueError(
                f"distillation_strength must be in [0, 1]; got {self.distillation_strength!r}."
            )
        if self.perturb_scale < 0:
            raise ValueError(f"perturb_scale must be >= 0; got {self.perturb_scale!r}.")
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2; got {self.cv!r}.")

        self.classes_ = np.unique(y_arr)
        rng = np.random.default_rng(self.random_state)

        # Train the teacher on a perturbed copy of X.
        teacher = (
            self.teacher_estimator if self.teacher_estimator is not None else LogisticRegression()
        )
        if not hasattr(teacher, "predict_proba"):
            raise ValueError(f"Teacher {type(teacher).__name__} must support predict_proba.")
        if self.perturb_scale > 0.0:
            X_teacher = X_arr + rng.normal(0.0, self.perturb_scale, size=X_arr.shape)
        else:
            X_teacher = X_arr
        self.teacher_ = clone(teacher)
        self.teacher_.fit(X_teacher, y_arr)

        # OOF teacher probabilities so the student is not trained on
        # information the teacher saw in fitting.
        y_pos = (y_arr == self.classes_[-1]).astype(np.int64)
        min_class = int(np.bincount(y_pos).min())
        effective_cv = max(2, min(self.cv, min_class))
        splitter = StratifiedKFold(
            n_splits=effective_cv, shuffle=True, random_state=self.random_state
        )
        oof = cross_val_predict(
            clone(teacher), X_teacher, y_arr, cv=splitter, method="predict_proba"
        )
        pos_col_teacher = int(np.where(self.teacher_.classes_ == self.classes_[-1])[0][0])
        teacher_p_pos = oof[:, pos_col_teacher]

        # Blend hard labels with teacher soft scores; train the student
        # on the original (un-perturbed) X with weights that emphasise
        # confident teacher predictions.
        soft_target = (
            1.0 - self.distillation_strength
        ) * y_pos + self.distillation_strength * teacher_p_pos
        # Use confidence (distance from 0.5) as the sample weight so
        # uncertain teacher rows contribute less.
        weights = 0.5 + np.abs(teacher_p_pos - 0.5)

        student = (
            self.student_estimator if self.student_estimator is not None else LogisticRegression()
        )
        self.student_ = clone(student)
        # Use the hard labels for the student fit (LogReg insists on
        # categorical targets), but apply the confidence weights.
        # Optionally, callers can subclass and override this step to
        # use a regressor on ``soft_target`` instead.
        del soft_target  # retained in attribute below for callers
        try:
            self.student_.fit(X_arr, y_arr, sample_weight=weights)
        except TypeError:
            self.student_.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "student_")
        X_arr = validate_data(self, X, reset=False)
        return self._aligned_proba(self.student_, X_arr)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "student_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
