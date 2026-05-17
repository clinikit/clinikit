"""Shared base for the five experiment protocols.

A *protocol* bundles preprocessing, a base classifier, a CV strategy,
a threshold strategy, and reporting choices into a single recipe.
Each concrete protocol in :mod:`clinikit.protocols._protocols`
subclasses :class:`BaseProtocol` and overrides only the
configuration; the run loop lives here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinikit.metrics import (
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    f2_score,
    mcc,
    npv,
    ppv,
    sensitivity,
    specificity,
)
from clinikit.thresholds import pick_threshold

__all__ = [
    "BaseProtocol",
    "ProtocolResult",
]


@dataclass(slots=True)
class ProtocolResult:
    """Output of a protocol's :meth:`run`.

    Attributes
    ----------
    protocol_name : str
    metrics : dict[str, float]
        Headline test-set metrics (accuracy, sensitivity, ...).
    leaderboard_row : dict[str, Any]
        Pre-built row suitable for ``clinikit.leaderboard.append_run``.
        Caller is responsible for adding identity fields
        (``run_id``, ``git_commit``, etc.) before persisting.
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing + classifier pipeline.
    threshold : float
        Decision threshold actually used.
    duration_seconds : float
        Wall-clock time of the run.
    """

    protocol_name: str
    metrics: dict[str, float]
    leaderboard_row: dict[str, Any]
    fitted_pipeline: Pipeline
    threshold: float
    duration_seconds: float = field(default=0.0)


@dataclass
class BaseProtocol:
    """Common configuration + run loop for every concrete protocol.

    Subclasses set the ``name`` and override the constructor's
    defaults to encode their recipe.
    """

    name: str = "base"
    base_estimator: ClassifierMixin | None = None
    preprocessing: BaseEstimator | None = None
    threshold_strategy: str = "accuracy_max"
    threshold_kwargs: dict[str, Any] | None = None
    test_size: float = 0.2
    random_state: int | None = None

    def _build_pipeline(self) -> Pipeline:
        estimator = self.base_estimator if self.base_estimator is not None else LogisticRegression()
        preprocessor = self.preprocessing if self.preprocessing is not None else StandardScaler()
        return Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("estimator", clone(estimator)),
            ]
        )

    def _select_threshold(self, y_true: NDArray[np.int64], y_prob: NDArray[np.float64]) -> float:
        kwargs = self.threshold_kwargs or {}
        result = pick_threshold(y_true, y_prob, strategy=self.threshold_strategy, **kwargs)
        if isinstance(result, float):
            return result
        # cluster_specific / two_stage return non-scalar — collapse to 0.5
        # to keep the run loop deterministic.
        return 0.5

    def _compute_metrics(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.int64],
        y_prob: NDArray[np.float64],
    ) -> dict[str, float]:
        scorers: dict[str, Callable[..., float]] = {
            "accuracy": accuracy_score,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f2_score": f2_score,
            "mcc": mcc,
            "balanced_accuracy": balanced_accuracy,
        }
        out = {name: float(fn(y_true, y_pred)) for name, fn in scorers.items()}
        out["brier_score"] = float(brier_score(y_true, y_prob))
        out["expected_calibration_error"] = float(
            expected_calibration_error(y_true, y_prob, n_bins=10)
        )
        return out

    def run(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> ProtocolResult:
        """Execute the protocol on ``(X, y)`` and return the headline result.

        Parameters
        ----------
        X, y : array-like
        groups : array-like, optional
            Accepted for API consistency; subclasses may use it for
            group-aware CV.

        Returns
        -------
        ProtocolResult
        """
        del groups  # base protocol uses a plain train/test split
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if len(classes) != 2:
            raise ValueError(f"Protocols are binary-only; got {len(classes)} unique labels.")

        start = time.perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_arr,
        )

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        estimator = pipeline.named_steps["estimator"]
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(f"Estimator {type(estimator).__name__} must support predict_proba.")

        pos_col = int(np.where(estimator.classes_ == classes[-1])[0][0])
        y_prob_train = pipeline.predict_proba(X_train)[:, pos_col]
        y_prob_test = pipeline.predict_proba(X_test)[:, pos_col]

        # Choose threshold on the training fold (the test set must stay
        # untouched).
        y_train_binary = (y_train == classes[-1]).astype(np.int64)
        threshold = self._select_threshold(y_train_binary, y_prob_train)

        y_pred_test = (y_prob_test >= threshold).astype(np.int64)
        y_test_binary = (y_test == classes[-1]).astype(np.int64)
        metrics = self._compute_metrics(y_test_binary, y_pred_test, y_prob_test)

        duration = float(time.perf_counter() - start)

        leaderboard_row: dict[str, Any] = {
            "n_samples_train": int(X_train.shape[0]),
            "n_samples_test": int(X_test.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "class_balance": float(y_arr.mean()) if y_arr.dtype.kind in "iub" else float("nan"),
            "model_class": type(estimator).__name__,
            "random_state": ("" if self.random_state is None else int(self.random_state)),
            "cv_strategy": f"holdout(test_size={self.test_size})",
            "test_size": self.test_size,
            "duration_seconds": round(duration, 4),
            "threshold_strategy": self.threshold_strategy,
            "threshold": round(threshold, 6),
            "accuracy": round(metrics["accuracy"], 6),
            "sensitivity": round(metrics["sensitivity"], 6),
            "specificity": round(metrics["specificity"], 6),
            "ppv": round(metrics["ppv"], 6),
            "npv": round(metrics["npv"], 6),
            "f2_score": round(metrics["f2_score"], 6),
            "mcc": round(metrics["mcc"], 6),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
            "brier_score": round(metrics["brier_score"], 6),
            "ece": round(metrics["expected_calibration_error"], 6),
            "notes": self.name,
        }

        return ProtocolResult(
            protocol_name=self.name,
            metrics=metrics,
            leaderboard_row=leaderboard_row,
            fitted_pipeline=pipeline,
            threshold=threshold,
            duration_seconds=duration,
        )
