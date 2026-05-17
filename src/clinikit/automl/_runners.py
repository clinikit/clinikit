"""Thin AutoML wrappers around TabPFN, FLAML, and AutoGluon-Tabular.

All three back-ends are optional dependencies. The wrappers are
intentionally conservative: bounded budgets, fixed seeds where the
back-end exposes one, and no implicit data uploads or
remote-installer side effects.

Install the extra with ``pip install "clinikit[automl]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AutoMLResult",
    "run_autogluon",
    "run_flaml",
    "run_tabpfn",
]


@dataclass(slots=True)
class AutoMLResult:
    """Output of every AutoML runner.

    Attributes
    ----------
    backend : str
        ``"tabpfn"`` / ``"flaml"`` / ``"autogluon"``.
    y_pred : ndarray of shape (n_test,)
        Hard predictions for the supplied test set.
    y_proba : ndarray of shape (n_test,) or None
        Positive-class probabilities, when the back-end produces them.
    metadata : dict
        Free-form back-end-specific information (selected model name,
        training time, etc.).
    """

    backend: str
    y_pred: NDArray
    y_proba: NDArray[np.float64] | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_split(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    X_tr = np.asarray(X_train)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test)
    if X_tr.ndim != 2 or X_te.ndim != 2:
        raise ValueError("X_train and X_test must be 2-D arrays.")
    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError(f"X_train has {X_tr.shape[1]} features; X_test has {X_te.shape[1]}.")
    if X_tr.shape[0] != y_tr.shape[0]:
        raise ValueError(
            f"X_train and y_train length mismatch: {X_tr.shape[0]} vs {y_tr.shape[0]}."
        )
    return X_tr, y_tr, X_te


def run_tabpfn(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    *,
    random_state: int | None = None,
) -> AutoMLResult:
    """Run TabPFN on the supplied split.

    TabPFN is bounded to small tabular problems (~1000 rows x 100
    features in practice). Inputs outside that envelope may slow it
    down significantly; the wrapper does not check this.

    Requires ``pip install "clinikit[automl]"``.

    Parameters
    ----------
    X_train, X_test : array-like
        2-D feature matrices with the same number of columns.
    y_train : array-like
        Training labels.
    random_state : int, optional

    Returns
    -------
    AutoMLResult

    Raises
    ------
    ImportError
        If TabPFN is not installed.
    """
    try:
        from tabpfn import TabPFNClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tabpfn is required. Install with: pip install 'clinikit[automl]'."
        ) from exc

    X_tr, y_tr, X_te = _validate_split(X_train, y_train, X_test)
    clf = TabPFNClassifier(random_state=random_state)
    clf.fit(X_tr, y_tr)
    y_pred = np.asarray(clf.predict(X_te))
    y_proba: NDArray[np.float64] | None
    if hasattr(clf, "predict_proba"):
        proba_full = np.asarray(clf.predict_proba(X_te), dtype=np.float64)
        y_proba = proba_full[:, -1] if proba_full.ndim == 2 else None
    else:
        y_proba = None
    return AutoMLResult(
        backend="tabpfn",
        y_pred=y_pred,
        y_proba=y_proba,
        metadata={"random_state": random_state},
    )


def run_flaml(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    *,
    time_budget: float = 30.0,
    metric: str = "roc_auc",
    random_state: int | None = None,
) -> AutoMLResult:
    """Run FLAML AutoML on the supplied split.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train : array-like
    time_budget : float, default 30.0
        Wall-clock budget in seconds. Hard cap — FLAML stops when
        exceeded.
    metric : str, default ``"roc_auc"``
        FLAML metric name.
    random_state : int, optional

    Returns
    -------
    AutoMLResult

    Raises
    ------
    ImportError
        If FLAML is not installed.
    """
    try:
        from flaml import AutoML  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "flaml is required. Install with: pip install 'clinikit[automl]'."
        ) from exc

    if time_budget <= 0:
        raise ValueError(f"time_budget must be > 0; got {time_budget!r}.")

    X_tr, y_tr, X_te = _validate_split(X_train, y_train, X_test)
    automl = AutoML()
    automl.fit(
        X_train=X_tr,
        y_train=y_tr,
        task="classification",
        time_budget=time_budget,
        metric=metric,
        seed=random_state,
        verbose=0,
    )
    y_pred = np.asarray(automl.predict(X_te))
    y_proba: NDArray[np.float64] | None
    if hasattr(automl, "predict_proba"):
        proba_full = np.asarray(automl.predict_proba(X_te), dtype=np.float64)
        y_proba = proba_full[:, -1] if proba_full.ndim == 2 else None
    else:
        y_proba = None
    return AutoMLResult(
        backend="flaml",
        y_pred=y_pred,
        y_proba=y_proba,
        metadata={
            "best_estimator": str(automl.best_estimator),
            "time_budget": time_budget,
            "metric": metric,
        },
    )


def run_autogluon(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    *,
    time_budget: float = 60.0,
    label_column: str = "label",
    work_dir: str | Path | None = None,
    random_state: int | None = None,
) -> AutoMLResult:
    """Run AutoGluon-Tabular on the supplied split.

    AutoGluon writes intermediate artifacts to ``work_dir`` (defaults
    to a fresh temp directory). The wrapper deletes nothing; callers
    are responsible for cleanup if they care about disk space.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train : array-like
    time_budget : float, default 60.0
        Wall-clock budget in seconds.
    label_column : str, default ``"label"``
        Column name AutoGluon uses internally; must not collide with
        feature columns.
    work_dir : str or pathlib.Path, optional
        Output directory; defaults to a fresh ``tempfile`` location.
    random_state : int, optional

    Returns
    -------
    AutoMLResult

    Raises
    ------
    ImportError
        If AutoGluon is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415
        from autogluon.tabular import TabularPredictor  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "autogluon is required. Install with: pip install 'clinikit[automl]'."
        ) from exc

    if time_budget <= 0:
        raise ValueError(f"time_budget must be > 0; got {time_budget!r}.")

    X_tr, y_tr, X_te = _validate_split(X_train, y_train, X_test)
    feature_cols = [f"x{i}" for i in range(X_tr.shape[1])]
    if label_column in feature_cols:
        raise ValueError(f"label_column {label_column!r} collides with synthetic feature names.")
    df_train = pd.DataFrame(X_tr, columns=feature_cols)
    df_train[label_column] = np.asarray(y_tr)
    df_test = pd.DataFrame(X_te, columns=feature_cols)

    if work_dir is None:
        import tempfile  # noqa: PLC0415

        work_dir = tempfile.mkdtemp(prefix="clinikit_autogluon_")
    out_dir = Path(work_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=label_column,
        path=str(out_dir),
        verbosity=0,
    ).fit(df_train, time_limit=time_budget, seed=random_state)
    y_pred = np.asarray(predictor.predict(df_test))
    try:
        proba_df = predictor.predict_proba(df_test)
        y_proba: NDArray[np.float64] | None = np.asarray(proba_df.iloc[:, -1], dtype=np.float64)
    except (AttributeError, ValueError):
        y_proba = None
    return AutoMLResult(
        backend="autogluon",
        y_pred=y_pred,
        y_proba=y_proba,
        metadata={
            "best_model": str(getattr(predictor, "model_best", "")),
            "work_dir": str(out_dir),
            "time_budget": time_budget,
        },
    )
