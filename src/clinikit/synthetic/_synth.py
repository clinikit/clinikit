"""CTGAN / TVAE generators and a TSTR safety gate.

CTGAN and TVAE come from the SDV (``sdv``) package, an optional
dependency installed with the ``[synthetic]`` extra. The TSTR
safety gate is pure-Python and always available — it compares a
downstream classifier's *train-on-synthetic-test-on-real* performance
against a real-on-real baseline and flags synthetic datasets that
underperform by more than a configurable tolerance.

The safety-gate workflow protects against a common failure mode of
generative models: they reproduce the marginal distributions well
enough to pass eyeballed checks, but the joint structure is degraded
enough that downstream models trained on synthetic data crash on
real data. The TSTR test catches this cheaply.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

__all__ = [
    "TSTRReport",
    "fit_ctgan",
    "fit_tvae",
    "tstr_safety_gate",
]


# ---------------------------------------------------------------------------
# Generators (lazy)
# ---------------------------------------------------------------------------


def _check_dataframe(X: ArrayLike, name: str) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got ndim={arr.ndim}.")
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _load_sdv_synthesizer(name: str) -> Any:
    """Import either CTGAN or TVAE from SDV lazily.

    SDV's API has shifted a couple of times between releases — try the
    current ``sdv.single_table`` import first, fall back to the older
    ``sdv.tabular`` location.
    """
    try:
        if name == "ctgan":
            try:
                from sdv.single_table import CTGANSynthesizer  # noqa: PLC0415

                return CTGANSynthesizer
            except ImportError:
                from sdv.tabular import CTGAN as _CTGAN  # noqa: PLC0415

                return _CTGAN
        if name == "tvae":
            try:
                from sdv.single_table import TVAESynthesizer  # noqa: PLC0415

                return TVAESynthesizer
            except ImportError:
                from sdv.tabular import TVAE as _TVAE  # noqa: PLC0415

                return _TVAE
        raise ValueError(f"Unknown synthesizer {name!r}.")
    except ImportError as exc:
        raise ImportError(
            "sdv is required. Install with: pip install 'clinikit[synthetic]'."
        ) from exc


def _fit_synthesizer(
    cls: Any,
    df: pd.DataFrame,
    *,
    epochs: int,
    random_state: int | None,
) -> Any:
    """Construct + fit either flavour of SDV synthesizer.

    SDV's newer API requires a ``Metadata`` object; the older one
    accepts a DataFrame directly. We try both.
    """
    try:
        from sdv.metadata import SingleTableMetadata  # noqa: PLC0415

        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        synth = cls(metadata=meta, epochs=epochs)
        synth.fit(df)
        return synth
    except ImportError:
        synth = cls(epochs=epochs)
        synth.fit(df)
        return synth
    except TypeError:
        # Some SDV versions take random_state as a kwarg.
        synth = cls(epochs=epochs, random_state=random_state)
        synth.fit(df)
        return synth


def fit_ctgan(
    X: ArrayLike,
    *,
    epochs: int = 300,
    random_state: int | None = None,
) -> Any:
    """Fit a CTGAN synthesizer on ``X`` and return the fitted object.

    Requires ``pip install "clinikit[synthetic]"``.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
    epochs : int, default 300
    random_state : int, optional

    Returns
    -------
    sdv synthesizer
        Fitted object with a ``.sample(n)`` method.
    """
    cls = _load_sdv_synthesizer("ctgan")
    df = _check_dataframe(X, "X")
    return _fit_synthesizer(cls, df, epochs=epochs, random_state=random_state)


def fit_tvae(
    X: ArrayLike,
    *,
    epochs: int = 300,
    random_state: int | None = None,
) -> Any:
    """Fit a TVAE synthesizer on ``X``. Same contract as :func:`fit_ctgan`."""
    cls = _load_sdv_synthesizer("tvae")
    df = _check_dataframe(X, "X")
    return _fit_synthesizer(cls, df, epochs=epochs, random_state=random_state)


# ---------------------------------------------------------------------------
# TSTR safety gate (pure logic — always available)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TSTRReport:
    """Output of :func:`tstr_safety_gate`.

    Attributes
    ----------
    real_score : float
        ``metric(y_real_test, model_trained_on_real.predict(X_real_test))``.
    synth_score : float
        ``metric(y_real_test, model_trained_on_synth.predict(X_real_test))``.
    gap : float
        ``real_score - synth_score``.
    tolerance : float
    passed : bool
        ``gap <= tolerance``. ``True`` means the synthetic dataset is
        good enough for downstream use under this metric.
    """

    real_score: float
    synth_score: float
    gap: float
    tolerance: float
    passed: bool


def tstr_safety_gate(
    real_X: ArrayLike,
    real_y: ArrayLike,
    synth_X: ArrayLike,
    synth_y: ArrayLike,
    *,
    tolerance: float = 0.05,
    estimator: BaseEstimator | None = None,
    metric: Callable[..., float] = accuracy_score,
    test_size: float = 0.3,
    random_state: int | None = None,
) -> TSTRReport:
    """Train-on-Synthetic-Test-on-Real safety check.

    Splits ``(real_X, real_y)`` into train/test, fits the supplied
    estimator twice — once on the real train fold, once on the
    synthetic data — and evaluates both on the same real test fold.
    A synthetic dataset passes the gate when its TSTR score is within
    ``tolerance`` of the real-on-real baseline.

    Parameters
    ----------
    real_X, real_y : array-like
        Real-data train+test pool.
    synth_X, synth_y : array-like
        Synthetic-data train pool.
    tolerance : float >= 0, default 0.05
        Maximum allowed drop in the metric.
    estimator : sklearn classifier, optional
        Default :class:`~sklearn.linear_model.LogisticRegression`.
    metric : callable, default :func:`sklearn.metrics.accuracy_score`
    test_size : float in (0, 1), default 0.3
    random_state : int, optional

    Returns
    -------
    TSTRReport
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0; got {tolerance!r}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}.")

    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    real_X_arr = np.asarray(real_X)
    real_y_arr = np.asarray(real_y)
    synth_X_arr = np.asarray(synth_X)
    synth_y_arr = np.asarray(synth_y)
    if real_X_arr.shape[1] != synth_X_arr.shape[1]:
        raise ValueError(
            f"real and synth feature counts differ: "
            f"{real_X_arr.shape[1]} vs {synth_X_arr.shape[1]}."
        )

    base = estimator if estimator is not None else LogisticRegression()

    X_real_tr, X_real_te, y_real_tr, y_real_te = train_test_split(
        real_X_arr,
        real_y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=real_y_arr,
    )

    real_model = clone(base).fit(X_real_tr, y_real_tr)
    synth_model = clone(base).fit(synth_X_arr, synth_y_arr)

    real_score = float(metric(y_real_te, real_model.predict(X_real_te)))
    synth_score = float(metric(y_real_te, synth_model.predict(X_real_te)))
    gap = real_score - synth_score
    return TSTRReport(
        real_score=real_score,
        synth_score=synth_score,
        gap=gap,
        tolerance=tolerance,
        passed=gap <= tolerance,
    )
