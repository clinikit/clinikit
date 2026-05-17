"""Post-hoc probability calibrators.

Each calibrator fits on a vector of *uncalibrated* probabilities plus
the true binary labels and produces a calibrated probability vector
at transform time. They are lightweight — designed to be applied
after a base classifier's :meth:`predict_proba` rather than as
sklearn pipeline transformers (calibration operates on a 1-D
probability vector, not on the feature matrix).

- :class:`PlattCalibrator`     — sigmoid fit on logits.
- :class:`IsotonicCalibrator`  — non-parametric monotonic fit.
- :class:`TemperatureScaler`   — single-parameter logit scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_consistent_length, check_is_fitted

__all__ = [
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureScaler",
]

_EPS = 1e-12


def _validate_probabilities(p: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(p, dtype=np.float64).ravel()
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return arr


def _validate_pair(p: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    p_arr = _validate_probabilities(p)
    y_arr = np.asarray(y).ravel()
    check_consistent_length(p_arr, y_arr)
    unique = np.unique(y_arr)
    if len(unique) > 2:
        raise ValueError("Calibrators are binary-only.")
    pos_label = unique[-1] if len(unique) > 0 else 1
    y_binary = (y_arr == pos_label).astype(np.int64)
    return p_arr, y_binary


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped)).astype(np.float64)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------


class PlattCalibrator(BaseEstimator):
    """Platt sigmoid calibration.

    Fits a logistic regression on the logits of the base
    probabilities. Equivalent to the original Platt (1999) recipe.

    Attributes
    ----------
    coef_ : float
    intercept_ : float

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.calibration import PlattCalibrator
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0, 1, size=200)
    >>> y = (rng.uniform(0, 1, size=200) < p).astype(int)
    >>> cal = PlattCalibrator().fit(p, y)
    >>> cal.transform([0.1, 0.5, 0.9]).shape
    (3,)
    """

    def fit(self, p: ArrayLike, y: ArrayLike) -> PlattCalibrator:
        p_arr, y_binary = _validate_pair(p, y)
        if len(np.unique(y_binary)) < 2:
            raise ValueError("PlattCalibrator requires both classes in y.")
        logits = _logit(p_arr).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(logits, y_binary)
        # LogisticRegression scores the positive class via coef * x + b.
        self.coef_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])
        return self

    def transform(self, p: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "coef_")
        p_arr = _validate_probabilities(p)
        logits = _logit(p_arr)
        return _sigmoid(self.coef_ * logits + self.intercept_)

    def fit_transform(self, p: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        return self.fit(p, y).transform(p)


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------


class IsotonicCalibrator(BaseEstimator):
    """Non-parametric monotonic calibration.

    Thin wrapper around :class:`sklearn.isotonic.IsotonicRegression`
    with sensible defaults (``y_min=0``, ``y_max=1``, clip
    out-of-bounds inputs).

    Attributes
    ----------
    isotonic_ : fitted ``IsotonicRegression``.
    """

    def fit(self, p: ArrayLike, y: ArrayLike) -> IsotonicCalibrator:
        p_arr, y_binary = _validate_pair(p, y)
        self.isotonic_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.isotonic_.fit(p_arr, y_binary)
        return self

    def transform(self, p: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "isotonic_")
        p_arr = _validate_probabilities(p)
        return np.asarray(self.isotonic_.transform(p_arr), dtype=np.float64)

    def fit_transform(self, p: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        return self.fit(p, y).transform(p)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


class TemperatureScaler(BaseEstimator):
    """Single-parameter logit-scaling calibration (Guo et al. 2017).

    Scales the logits by ``1/T`` where ``T > 0`` is optimised by
    minimising negative log-likelihood on the calibration set.

    Attributes
    ----------
    temperature_ : float
        The fitted temperature. ``T < 1`` sharpens probabilities,
        ``T > 1`` softens them.
    """

    def __init__(self, *, t_bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.t_bounds = t_bounds

    def fit(self, p: ArrayLike, y: ArrayLike) -> TemperatureScaler:
        p_arr, y_binary = _validate_pair(p, y)
        if len(np.unique(y_binary)) < 2:
            raise ValueError("TemperatureScaler requires both classes in y.")
        logits = _logit(p_arr)

        def _neg_log_likelihood(t: float) -> float:
            t = max(float(t), 1e-9)
            scaled = logits / t
            log_p = np.where(
                scaled >= 0,
                -np.log1p(np.exp(-scaled)),
                scaled - np.log1p(np.exp(scaled)),
            )
            # log(1 - sigmoid(x)) = -log1p(exp(x)) for stability.
            log_1mp = np.where(
                scaled >= 0,
                -scaled - np.log1p(np.exp(-scaled)),
                -np.log1p(np.exp(scaled)),
            )
            nll = -(y_binary * log_p + (1 - y_binary) * log_1mp).mean()
            return float(nll)

        result = minimize_scalar(_neg_log_likelihood, bounds=self.t_bounds, method="bounded")
        self.temperature_ = float(result.x)
        return self

    def transform(self, p: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "temperature_")
        p_arr = _validate_probabilities(p)
        logits = _logit(p_arr)
        return _sigmoid(logits / self.temperature_)

    def fit_transform(self, p: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        return self.fit(p, y).transform(p)
