"""Tests for clinikit.time_split."""

from __future__ import annotations

import pandas as pd
import pytest

from clinikit.time_split import chronological_split, rolling_origin_splits


@pytest.fixture
def time_df() -> pd.DataFrame:
    return pd.DataFrame({"t": list(range(20)), "v": list(range(100, 120))})


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_basic(time_df: pd.DataFrame) -> None:
    train, test = chronological_split(time_df, time_column="t", test_size=0.3)
    assert len(train) == 14
    assert len(test) == 6
    # Test rows must all come after train rows by time.
    assert train["t"].max() < test["t"].min()


def test_chronological_split_unordered_input_sorts(time_df: pd.DataFrame) -> None:
    shuffled = time_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    train, test = chronological_split(shuffled, time_column="t", test_size=0.2)
    assert train["t"].is_monotonic_increasing
    assert test["t"].is_monotonic_increasing


def test_chronological_split_invalid_test_size(time_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="test_size"):
        chronological_split(time_df, time_column="t", test_size=1.5)


def test_chronological_split_unknown_column(time_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="time_column"):
        chronological_split(time_df, time_column="nope", test_size=0.2)


def test_chronological_split_tiny_input() -> None:
    df = pd.DataFrame({"t": [0]})
    with pytest.raises(ValueError, match="at least 2"):
        chronological_split(df, time_column="t", test_size=0.5)


# ---------------------------------------------------------------------------
# rolling_origin_splits
# ---------------------------------------------------------------------------


def test_rolling_origin_default_sizing(time_df: pd.DataFrame) -> None:
    folds = list(rolling_origin_splits(time_df, time_column="t", n_splits=4, test_size=3))
    assert len(folds) == 4
    for train, test in folds:
        assert train["t"].max() < test["t"].min()
        assert len(test) == 3


def test_rolling_origin_expanding_train(time_df: pd.DataFrame) -> None:
    folds = list(rolling_origin_splits(time_df, time_column="t", n_splits=3, test_size=2))
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[1] - train_sizes[0] == 2
    assert train_sizes[2] - train_sizes[1] == 2


def test_rolling_origin_explicit_min_train(time_df: pd.DataFrame) -> None:
    folds = list(
        rolling_origin_splits(time_df, time_column="t", n_splits=2, test_size=3, min_train_size=8)
    )
    assert len(folds[0][0]) == 8
    assert len(folds[1][0]) == 11


def test_rolling_origin_invalid_n_splits(time_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        list(rolling_origin_splits(time_df, time_column="t", n_splits=0))


def test_rolling_origin_invalid_test_size(time_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="test_size"):
        list(rolling_origin_splits(time_df, time_column="t", n_splits=2, test_size=0))


def test_rolling_origin_not_enough_rows() -> None:
    df = pd.DataFrame({"t": list(range(5))})
    with pytest.raises(ValueError, match=r"Not enough rows|min_train_size"):
        list(rolling_origin_splits(df, time_column="t", n_splits=5, test_size=2))
