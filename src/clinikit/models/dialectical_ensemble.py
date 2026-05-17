"""Dialectical-ensemble binary classifier.

Two sub-models with different inductive biases (the *thesis* and
*antithesis*) are trained on the same data. A third model — the
*arbiter* — is trained on the cases where thesis and antithesis
disagree. At predict time:

- If thesis and antithesis agree, average their probabilities.
- If they disagree, use the arbiter to resolve.

Binary-only; passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["DialecticalEnsemble"]


class DialecticalEnsemble(ClassifierMixin, BaseEstimator):
    """Pair two opposing models and let an arbiter resolve disagreements.

    Parameters
    ----------
    thesis_estimator : sklearn classifier, optional
        Default: :class:`~sklearn.linear_model.LogisticRegression`.
    antithesis_estimator : sklearn classifier, optional
        Default:
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.
    arbiter_estimator : sklearn classifier, optional
        Resolves disagreements. Default same as thesis.
    random_state : int, optional
        Forwarded where applicable.

    Attributes
    ----------
    thesis_ : fitted thesis estimator.
    antithesis_ : fitted antithesis estimator.
    arbiter_ : fitted arbiter or ``None`` when there were no
        disagreements during training.
    classes_ : ndarray of shape (2,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import DialecticalEnsemble
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = DialecticalEnsemble(random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        thesis_estimator: ClassifierMixin | None = None,
        antithesis_estimator: ClassifierMixin | None = None,
        arbiter_estimator: ClassifierMixin | None = None,
        *,
        random_state: int | None = None,
    ) -> None:
        self.thesis_estimator = thesis_estimator
        self.antithesis_estimator = antithesis_estimator
        self.arbiter_estimator = arbiter_estimator
        self.random_state = random_state

    def _pos_col(self, estimator: ClassifierMixin) -> int:
        return int(np.where(np.asarray(estimator.classes_) == self.classes_[-1])[0][0])

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

    def fit(self, X: ArrayLike, y: ArrayLike) -> DialecticalEnsemble:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        y_type = type_of_target(y_arr, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        self.classes_ = np.unique(y_arr)

        thesis = (
            self.thesis_estimator if self.thesis_estimator is not None else LogisticRegression()
        )
        antithesis = (
            self.antithesis_estimator
            if self.antithesis_estimator is not None
            else GradientBoostingClassifier(random_state=self.random_state)
        )
        arbiter = (
            self.arbiter_estimator if self.arbiter_estimator is not None else LogisticRegression()
        )

        self.thesis_ = clone(thesis)
        self.antithesis_ = clone(antithesis)
        self.thesis_.fit(X_arr, y_arr)
        self.antithesis_.fit(X_arr, y_arr)

        thesis_pred = self.thesis_.predict(X_arr)
        antithesis_pred = self.antithesis_.predict(X_arr)
        disagree = thesis_pred != antithesis_pred

        self.arbiter_: ClassifierMixin | None = None
        if disagree.sum() >= 4 and len(np.unique(y_arr[disagree])) >= 2:
            self.arbiter_ = clone(arbiter)
            self.arbiter_.fit(X_arr[disagree], y_arr[disagree])
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "thesis_")
        X_arr = validate_data(self, X, reset=False)
        thesis_proba = self._aligned_proba(self.thesis_, X_arr)
        antithesis_proba = self._aligned_proba(self.antithesis_, X_arr)

        thesis_pred = self.classes_[np.argmax(thesis_proba, axis=1)]
        antithesis_pred = self.classes_[np.argmax(antithesis_proba, axis=1)]
        agree = thesis_pred == antithesis_pred

        out = 0.5 * (thesis_proba + antithesis_proba)
        if self.arbiter_ is not None and (~agree).any():
            arbiter_proba = self._aligned_proba(self.arbiter_, X_arr[~agree])
            out[~agree] = arbiter_proba
        return np.asarray(out, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self, "thesis_")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=self.classes_.dtype)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
