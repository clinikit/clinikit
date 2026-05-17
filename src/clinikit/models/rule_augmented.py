"""Rule-augmented binary classifier.

A meta-estimator that wraps a base classifier and lets the caller
override its predictions with a list of deterministic rules. Each
rule is a callable ``rule(X) -> np.ndarray`` returning class labels
where ``-1`` means "this rule has no opinion on this sample". Rules
are checked in order; the first rule with an opinion wins.

The class is sklearn-compatible: ``RuleAugmentedClassifier()`` (no
rules, default :class:`~sklearn.linear_model.LogisticRegression` base)
passes ``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["RuleAugmentedClassifier"]

Rule = Callable[[np.ndarray], np.ndarray]


class RuleAugmentedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a base classifier with deterministic rule overrides.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Estimator used when no rule has an opinion. Cloned at fit time;
        the default is
        :class:`sklearn.linear_model.LogisticRegression`.
    rules : list of callable, optional
        Each rule maps ``X`` to an array of length ``n_samples``
        containing class labels (drawn from
        :attr:`classes_`) or ``-1`` for "no opinion".
        Rules are evaluated in order; the first non-``-1`` opinion is
        kept.
    rule_confidence : float in (0.5, 1.0], default 0.95
        Probability assigned to a rule-overridden prediction in
        :meth:`predict_proba`. Lower values keep more weight on the
        base estimator's calibration.

    Attributes
    ----------
    base_estimator_ : sklearn classifier
        The fitted clone of ``base_estimator``.
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    feature_names_in_ : ndarray, optional
        Set only when ``fit`` is called with a pandas DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.models import RuleAugmentedClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = RuleAugmentedClassifier().fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        rules: list[Rule] | None = None,
        *,
        rule_confidence: float = 0.95,
    ) -> None:
        self.base_estimator = base_estimator
        self.rules = rules
        self.rule_confidence = rule_confidence

    def fit(self, X: ArrayLike, y: ArrayLike) -> RuleAugmentedClassifier:
        X_arr, y_arr = validate_data(self, X, y, reset=True)
        check_classification_targets(y_arr)

        if not 0.5 < self.rule_confidence <= 1.0:
            raise ValueError(
                f"rule_confidence must lie in (0.5, 1.0]; got {self.rule_confidence!r}."
            )

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Classifier requires more than one class. Got 1 class in y.")

        base = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_arr, y_arr)
        return self

    def _apply_rules(self, X_arr: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return per-sample rule opinion, with ``-1`` for "no rule fired"."""
        out = np.full(X_arr.shape[0], -1, dtype=np.int64)
        if not self.rules:
            return out
        valid_labels = set(self.classes_.tolist())
        for rule in self.rules:
            opinion = np.asarray(rule(X_arr))
            if opinion.shape != (X_arr.shape[0],):
                raise ValueError(
                    f"Rule {rule!r} returned shape {opinion.shape}; expected ({X_arr.shape[0]},)."
                )
            # Update only samples that don't yet have an opinion.
            unresolved = out == -1
            takes = (opinion != -1) & unresolved
            if takes.any():
                # Validate the labels being applied.
                proposed = opinion[takes]
                bad = [lbl for lbl in proposed.tolist() if lbl not in valid_labels]
                if bad:
                    raise ValueError(
                        f"Rule {rule!r} produced labels {bad!r} not in classes_={self.classes_!r}."
                    )
                out[takes] = proposed
        return out

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)

        base_pred = self.base_estimator_.predict(X_arr)
        rule_pred = self._apply_rules(X_arr)
        mask = rule_pred != -1
        final = np.where(mask, rule_pred, base_pred).astype(self.classes_.dtype)
        return final

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "base_estimator_")
        X_arr = validate_data(self, X, reset=False)

        if not hasattr(self.base_estimator_, "predict_proba"):
            raise NotImplementedError(
                f"Base estimator {type(self.base_estimator_).__name__} has no predict_proba."
            )

        proba: NDArray[np.float64] = self.base_estimator_.predict_proba(X_arr).astype(np.float64)
        rule_pred = self._apply_rules(X_arr)

        if (rule_pred == -1).all():
            return proba

        # For every sample where a rule fired, replace its probability
        # row with a peak at the rule's chosen class.
        residual = (1.0 - self.rule_confidence) / max(len(self.classes_) - 1, 1)
        for i, lbl in enumerate(rule_pred):
            if lbl == -1:
                continue
            row = np.full(len(self.classes_), residual, dtype=np.float64)
            row[int(np.where(self.classes_ == lbl)[0][0])] = self.rule_confidence
            proba[i] = row
        return proba

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        return tags
