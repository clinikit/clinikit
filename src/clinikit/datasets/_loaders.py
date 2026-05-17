"""Loaders for UCI benchmark datasets bundled with clinikit.

CSVs are shipped under ``clinikit/datasets/data/`` and resolved via
``importlib.resources.files`` so the loaders work after a clean
``pip install clinikit`` on any platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

__all__ = [
    "DatasetBunch",
    "load_heart",
    "load_pima",
    "load_wisconsin",
]


@dataclass(slots=True)
class DatasetBunch:
    """sklearn-style container returned by ``load_*`` when ``return_X_y=False``.

    Attributes
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    target : pandas.Series or numpy.ndarray
        Target labels (integer-coded).
    feature_names : list[str]
        Column names corresponding to ``data``.
    target_name : str
        Name of the target column in the source CSV.
    frame : pandas.DataFrame, optional
        The combined ``data`` + ``target`` frame, present only when
        the loader was called with ``as_frame=True`` (the default).
    description : str
        Short human-readable description of the dataset.
    source_url : str
        Canonical URL the bundled CSV was derived from.
    """

    data: pd.DataFrame | NDArray[np.float64]
    target: pd.Series | NDArray[np.int64]
    feature_names: list[str]
    target_name: str
    description: str
    source_url: str
    frame: pd.DataFrame | None = field(default=None)


def _resource_csv(name: str) -> Traversable:
    return files("clinikit.datasets") / "data" / name


def _load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(_resource_csv(filename).open("rb"))


def _as_bunch_or_xy(
    frame: pd.DataFrame,
    *,
    target_name: str,
    description: str,
    source_url: str,
    return_X_y: bool,
    as_frame: bool,
) -> DatasetBunch | tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]:
    feature_names = [c for c in frame.columns if c != target_name]
    if as_frame:
        X: pd.DataFrame | NDArray[np.float64] = frame[feature_names].copy()
        y: pd.Series | NDArray[np.int64] = frame[target_name].astype(np.int64).copy()
    else:
        X = frame[feature_names].to_numpy(dtype=np.float64)
        y = frame[target_name].to_numpy(dtype=np.int64)

    if return_X_y:
        return X, y

    return DatasetBunch(
        data=X,
        target=y,
        feature_names=feature_names,
        target_name=target_name,
        description=description,
        source_url=source_url,
        frame=frame.copy() if as_frame else None,
    )


# ---------------------------------------------------------------------------
# PIMA Indians Diabetes
# ---------------------------------------------------------------------------

_PIMA_DESCRIPTION = (
    "PIMA Indians Diabetes (UCI / NIDDK). 768 samples, 8 numeric "
    "features, binary target 'outcome' (1 = diabetes diagnosed)."
)
_PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)


@overload
def load_pima(
    *, return_X_y: Literal[True], as_frame: bool = ...
) -> tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]: ...


@overload
def load_pima(*, return_X_y: Literal[False] = False, as_frame: bool = ...) -> DatasetBunch: ...


def load_pima(
    *,
    return_X_y: bool = False,
    as_frame: bool = True,
) -> DatasetBunch | tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]:
    """Load the bundled PIMA Indians Diabetes dataset.

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return a ``(X, y)`` tuple instead of a
        :class:`DatasetBunch`.
    as_frame : bool, default True
        If True, ``X`` is a ``pandas.DataFrame`` and ``y`` is a
        ``pandas.Series``. If False, both are ``numpy`` arrays.

    Returns
    -------
    DatasetBunch or (X, y) tuple

    Examples
    --------
    >>> from clinikit.datasets import load_pima
    >>> X, y = load_pima(return_X_y=True)
    >>> X.shape
    (768, 8)
    """
    frame = _load_csv("pima.csv")
    return _as_bunch_or_xy(
        frame,
        target_name="outcome",
        description=_PIMA_DESCRIPTION,
        source_url=_PIMA_URL,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


# ---------------------------------------------------------------------------
# Wisconsin Breast Cancer (Diagnostic)
# ---------------------------------------------------------------------------

_WISCONSIN_DESCRIPTION = (
    "Wisconsin Breast Cancer (Diagnostic, UCI). 569 samples, 30 "
    "numeric features. Target 'diagnosis': 0 = malignant, 1 = benign "
    "(matches scikit-learn's load_breast_cancer encoding)."
)
_WISCONSIN_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"


@overload
def load_wisconsin(
    *, return_X_y: Literal[True], as_frame: bool = ...
) -> tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]: ...


@overload
def load_wisconsin(*, return_X_y: Literal[False] = False, as_frame: bool = ...) -> DatasetBunch: ...


def load_wisconsin(
    *,
    return_X_y: bool = False,
    as_frame: bool = True,
) -> DatasetBunch | tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]:
    """Load the bundled Wisconsin Breast Cancer (Diagnostic) dataset.

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return a ``(X, y)`` tuple instead of a
        :class:`DatasetBunch`.
    as_frame : bool, default True
        If True, ``X`` is a ``pandas.DataFrame`` and ``y`` is a
        ``pandas.Series``. If False, both are ``numpy`` arrays.

    Returns
    -------
    DatasetBunch or (X, y) tuple

    Examples
    --------
    >>> from clinikit.datasets import load_wisconsin
    >>> X, y = load_wisconsin(return_X_y=True)
    >>> X.shape
    (569, 30)
    """
    frame = _load_csv("wisconsin.csv")
    return _as_bunch_or_xy(
        frame,
        target_name="diagnosis",
        description=_WISCONSIN_DESCRIPTION,
        source_url=_WISCONSIN_URL,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


# ---------------------------------------------------------------------------
# UCI Heart Disease (Cleveland)
# ---------------------------------------------------------------------------

_HEART_DESCRIPTION = (
    "UCI Heart Disease (Cleveland subset). 303 samples, 13 features. "
    "Target binarised from the original 0..4 severity score: "
    "0 = no disease, 1 = any disease. Missing values preserved as NaN "
    "in 'ca' and 'thal' — feed through an imputer before training."
)
_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


@overload
def load_heart(
    *, return_X_y: Literal[True], as_frame: bool = ...
) -> tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]: ...


@overload
def load_heart(*, return_X_y: Literal[False] = False, as_frame: bool = ...) -> DatasetBunch: ...


def load_heart(
    *,
    return_X_y: bool = False,
    as_frame: bool = True,
) -> DatasetBunch | tuple[pd.DataFrame | NDArray[np.float64], pd.Series | NDArray[np.int64]]:
    """Load the bundled UCI Heart Disease (Cleveland) dataset.

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return a ``(X, y)`` tuple instead of a
        :class:`DatasetBunch`.
    as_frame : bool, default True
        If True, ``X`` is a ``pandas.DataFrame`` and ``y`` is a
        ``pandas.Series``. If False, both are ``numpy`` arrays.

    Returns
    -------
    DatasetBunch or (X, y) tuple

    Notes
    -----
    Some rows contain missing values in the ``ca`` and ``thal``
    columns. When ``as_frame=False`` these are preserved as
    ``np.nan``; if ``as_frame=False`` is combined with downstream code
    that does not handle NaN, run the frame through a
    :mod:`clinikit.preprocessing` imputer first.

    Examples
    --------
    >>> from clinikit.datasets import load_heart
    >>> X, y = load_heart(return_X_y=True)
    >>> X.shape
    (303, 13)
    """
    frame = _load_csv("heart.csv")
    return _as_bunch_or_xy(
        frame,
        target_name="target",
        description=_HEART_DESCRIPTION,
        source_url=_HEART_URL,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )
