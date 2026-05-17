"""SHAP, LIME, and partial-dependence helpers.

SHAP and LIME are *optional* dependencies (the ``[explain]`` extra)
and are imported lazily inside the helpers. A clean
``pip install clinikit`` is enough to import this module; calling
:func:`shap_summary` or :func:`lime_explain_instance` without the
extra installed raises an :class:`ImportError` with a friendly hint.

Partial dependence is computed via :mod:`sklearn.inspection`, which
ships with clinikit's core dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

__all__ = [
    "PartialDependenceResult",
    "lime_explain_instance",
    "partial_dependence",
    "shap_summary",
]


# ---------------------------------------------------------------------------
# SHAP (lazy)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ShapResult:
    """Lightweight container around a SHAP Explanation."""

    feature_importance: NDArray[np.float64]
    values: NDArray[np.float64]
    expected_value: float | NDArray[np.float64]


def shap_summary(
    model: BaseEstimator,
    X: ArrayLike,
    *,
    max_samples: int = 100,
    random_state: int | None = None,
) -> _ShapResult:
    """Return mean-absolute SHAP feature importance for ``model`` on ``X``.

    Requires the optional ``[explain]`` extra
    (``pip install "clinikit[explain]"``).

    Parameters
    ----------
    model : sklearn estimator
        Already fitted. Must support either ``predict_proba`` or
        ``predict``.
    X : array-like of shape (n_samples, n_features)
        Reference data — SHAP estimates background distribution from
        these rows (capped at ``max_samples``).
    max_samples : int, default 100
        Random subset of ``X`` used as the SHAP background.
    random_state : int, optional

    Returns
    -------
    _ShapResult
        ``feature_importance`` is the mean absolute SHAP value per
        feature; ``values`` is the full per-sample matrix;
        ``expected_value`` is the model's average baseline.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "shap is required. Install with: pip install 'clinikit[explain]'."
        ) from exc

    rng = np.random.default_rng(random_state)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.shape[0] > max_samples:
        idx = rng.choice(X_arr.shape[0], size=max_samples, replace=False)
        background = X_arr[idx]
    else:
        background = X_arr

    # Explainer chooses the right strategy (tree / linear / kernel)
    # automatically based on the model type.
    explainer = shap.Explainer(model, background)
    explanation = explainer(background)

    raw_values = np.asarray(explanation.values)
    # For binary classification SHAP returns shape (n_samples, n_features, 2)
    # in some versions; collapse to (n_samples, n_features) by taking the
    # positive-class column.
    if raw_values.ndim == 3:
        raw_values = raw_values[:, :, -1]

    return _ShapResult(
        feature_importance=np.abs(raw_values).mean(axis=0).astype(np.float64),
        values=raw_values.astype(np.float64),
        expected_value=float(np.asarray(explanation.base_values).mean()),
    )


# ---------------------------------------------------------------------------
# LIME (lazy)
# ---------------------------------------------------------------------------


def lime_explain_instance(
    model: BaseEstimator,
    X: ArrayLike,
    instance_idx: int,
    *,
    feature_names: list[str] | None = None,
    num_features: int = 10,
    random_state: int | None = None,
) -> dict[str, float]:
    """Local LIME explanation for a single sample.

    Requires the optional ``[explain]`` extra.

    Parameters
    ----------
    model : sklearn estimator
        Already fitted; must expose ``predict_proba``.
    X : array-like of shape (n_samples, n_features)
        Reference data — LIME perturbs around the chosen instance.
    instance_idx : int
        Index of the row to explain.
    feature_names : list[str], optional
        Names for the table; defaults to ``["feature_0", ...]``.
    num_features : int, default 10
        Top-N features to return.
    random_state : int, optional

    Returns
    -------
    dict[str, float]
        Mapping from feature name to its signed LIME weight for the
        positive class.

    Raises
    ------
    ImportError
        If ``lime`` is not installed.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lime is required. Install with: pip install 'clinikit[explain]'."
        ) from exc

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"LIME requires a probabilistic classifier; "
            f"{type(model).__name__} has no predict_proba."
        )

    X_arr = np.asarray(X, dtype=np.float64)
    if not 0 <= instance_idx < X_arr.shape[0]:
        raise ValueError(
            f"instance_idx={instance_idx!r} out of range for n_samples={X_arr.shape[0]}."
        )

    names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
    explainer = LimeTabularExplainer(
        training_data=X_arr,
        feature_names=names,
        class_names=["class_0", "class_1"],
        mode="classification",
        random_state=random_state,
    )
    explanation = explainer.explain_instance(
        X_arr[instance_idx],
        model.predict_proba,
        num_features=num_features,
    )
    return dict(explanation.as_list())


# ---------------------------------------------------------------------------
# Partial dependence (no extra deps)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PartialDependenceResult:
    """Output of :func:`partial_dependence` for one feature."""

    feature: int
    grid: NDArray[np.float64]
    averaged_predictions: NDArray[np.float64]


def partial_dependence(
    model: BaseEstimator,
    X: ArrayLike,
    features: list[int] | int,
    *,
    grid_resolution: int = 50,
) -> list[PartialDependenceResult]:
    """Sklearn-backed partial dependence for one or more features.

    Thin wrapper around
    :func:`sklearn.inspection.partial_dependence` that returns the
    arrays in a self-describing dataclass per feature.

    Parameters
    ----------
    model : sklearn estimator
        Already fitted.
    X : array-like of shape (n_samples, n_features)
    features : int or list of int
        Column index (or indices) to compute partial dependence for.
    grid_resolution : int, default 50
        Number of evenly-spaced grid points to evaluate at.

    Returns
    -------
    list[PartialDependenceResult]
        One entry per feature index.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from clinikit.explainability import partial_dependence
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = LogisticRegression().fit(X, y)
    >>> pd_result = partial_dependence(model, X, features=[0, 1])
    >>> len(pd_result)
    2
    """
    from sklearn.inspection import partial_dependence as _sk_pd  # noqa: PLC0415

    feature_list = [features] if isinstance(features, int) else list(features)
    if not feature_list:
        raise ValueError("features must contain at least one column index.")
    X_arr = np.asarray(X, dtype=np.float64)

    out: list[PartialDependenceResult] = []
    for f in feature_list:
        if not 0 <= f < X_arr.shape[1]:
            raise ValueError(f"feature index {f!r} out of range for n_features={X_arr.shape[1]}.")
        sk_result: Any = _sk_pd(
            model,
            X_arr,
            features=[f],
            grid_resolution=grid_resolution,
            kind="average",
        )
        grid = np.asarray(sk_result["grid_values"][0], dtype=np.float64)
        avg = np.asarray(sk_result["average"][0], dtype=np.float64)
        # For binary classifiers sklearn returns (n_classes,) along the
        # first axis when output is multi-class; collapse to the
        # positive-class slice if needed.
        if avg.ndim > 1:
            avg = avg[-1]
        out.append(
            PartialDependenceResult(
                feature=int(f),
                grid=grid,
                averaged_predictions=avg.astype(np.float64),
            )
        )
    return out
