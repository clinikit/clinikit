"""Chronological train/test split utilities.

Two helpers for time-aware evaluation that do not require a group
key (see :mod:`clinikit.cross_val` for grouped variants):

- :func:`chronological_split`   — single train/test cut.
- :func:`rolling_origin_splits` — expanding-window iterator.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "chronological_split",
    "rolling_origin_splits",
]


def _sort_by_time(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column!r} not in DataFrame columns.")
    return df.sort_values(time_column, kind="mergesort").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    *,
    time_column: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train and test.

    Sorts by ``time_column`` (stable sort), then takes the last
    ``test_size`` fraction of rows as the test set and the rest as
    train. The two frames preserve the time ordering.

    Parameters
    ----------
    df : pandas.DataFrame
    time_column : str
        Column name to sort on. Any orderable dtype is accepted —
        datetimes, ints, strings.
    test_size : float in (0, 1), default 0.2

    Returns
    -------
    (train, test) : tuple of DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> from clinikit.time_split import chronological_split
    >>> df = pd.DataFrame({"t": range(10), "v": range(10)})
    >>> train, test = chronological_split(df, time_column="t", test_size=0.3)
    >>> len(train), len(test)
    (7, 3)
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}.")
    ordered = _sort_by_time(df, time_column)
    n = len(ordered)
    if n < 2:
        raise ValueError(f"Need at least 2 rows; got {n}.")
    cut = max(1, int(np.floor(n * (1.0 - test_size))))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def rolling_origin_splits(
    df: pd.DataFrame,
    *,
    time_column: str,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int = 1,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window splits along time.

    The earliest ``min_train_size`` rows seed the first train fold;
    each subsequent fold extends the train window by ``test_size``
    rows and re-evaluates on the next ``test_size`` rows. Yields
    exactly ``n_splits`` folds.

    Parameters
    ----------
    df : pandas.DataFrame
    time_column : str
    n_splits : int, default 5
    min_train_size : int, optional
        Train window for the *first* fold. Defaults to
        ``len(df) - n_splits * test_size``.
    test_size : int, default 1
        Number of rows per test fold.

    Yields
    ------
    (train, test) : tuple of DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> from clinikit.time_split import rolling_origin_splits
    >>> df = pd.DataFrame({"t": range(20), "v": range(20)})
    >>> folds = list(rolling_origin_splits(df, time_column="t", n_splits=4, test_size=3))
    >>> len(folds)
    4
    >>> [(len(tr), len(te)) for tr, te in folds]
    [(8, 3), (11, 3), (14, 3), (17, 3)]
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1; got {n_splits!r}.")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1; got {test_size!r}.")

    ordered = _sort_by_time(df, time_column)
    n = len(ordered)
    if min_train_size is None:
        min_train_size = n - n_splits * test_size
    if min_train_size < 1:
        raise ValueError(
            f"min_train_size must be >= 1; computed {min_train_size} for "
            f"n={n}, n_splits={n_splits}, test_size={test_size}. "
            "Either supply min_train_size explicitly or shrink n_splits / test_size."
        )
    if min_train_size + n_splits * test_size > n:
        raise ValueError(
            f"Not enough rows: need at least {min_train_size + n_splits * test_size}, got {n}."
        )

    for k in range(n_splits):
        train_high = min_train_size + k * test_size
        test_low = train_high
        test_high = test_low + test_size
        yield (
            ordered.iloc[:train_high].copy(),
            ordered.iloc[test_low:test_high].copy(),
        )
