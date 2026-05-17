"""Tests for clinikit.cross_val."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from clinikit.cross_val import (
    CrossValResult,
    GroupStratifiedKFold,
    TimeSeriesGroupSplit,
    cross_validate_with_groups,
)


@pytest.fixture
def grouped_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_groups = 10
    per_group = 8
    n = n_groups * per_group
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(n_groups), per_group)
    return X, y, groups


# ---------------------------------------------------------------------------
# GroupStratifiedKFold
# ---------------------------------------------------------------------------


def test_group_stratified_kfold_respects_groups(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    X, y, groups = grouped_xy
    splitter = GroupStratifiedKFold(n_splits=5, random_state=0)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        # No group appears on both sides.
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_group_stratified_kfold_n_splits(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    X, y, groups = grouped_xy
    splitter = GroupStratifiedKFold(n_splits=4, random_state=0)
    splits = list(splitter.split(X, y, groups=groups))
    assert len(splits) == 4


# ---------------------------------------------------------------------------
# TimeSeriesGroupSplit
# ---------------------------------------------------------------------------


def test_timeseries_group_split_chronological() -> None:
    n_groups = 10
    per_group = 5
    n = n_groups * per_group
    X = np.arange(n).reshape(-1, 1)
    groups = np.repeat(np.arange(n_groups), per_group)
    tscv = TimeSeriesGroupSplit(n_splits=3)
    splits = list(tscv.split(X, groups=groups))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        # All test groups come strictly after all train groups.
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)
        if train_groups and test_groups:
            assert max(train_groups) < min(test_groups)


def test_timeseries_group_split_requires_groups() -> None:
    X = np.arange(10).reshape(-1, 1)
    tscv = TimeSeriesGroupSplit(n_splits=2)
    with pytest.raises(ValueError, match="groups"):
        next(iter(tscv.split(X)))


def test_timeseries_group_split_invalid_n_splits() -> None:
    X = np.arange(10).reshape(-1, 1)
    groups = np.repeat([0, 1, 2, 3, 4], 2)
    tscv = TimeSeriesGroupSplit(n_splits=0)
    with pytest.raises(ValueError, match="n_splits"):
        next(iter(tscv.split(X, groups=groups)))


def test_timeseries_group_split_too_many_splits() -> None:
    X = np.arange(6).reshape(-1, 1)
    groups = np.array([0, 0, 1, 1, 2, 2])
    tscv = TimeSeriesGroupSplit(n_splits=3)
    with pytest.raises(ValueError, match="n_groups"):
        list(tscv.split(X, groups=groups))


def test_timeseries_group_split_get_n_splits() -> None:
    assert TimeSeriesGroupSplit(n_splits=4).get_n_splits() == 4


def test_timeseries_group_split_max_train_size() -> None:
    n_groups = 8
    per_group = 5
    X = np.arange(n_groups * per_group).reshape(-1, 1)
    groups = np.repeat(np.arange(n_groups), per_group)
    tscv = TimeSeriesGroupSplit(n_splits=2, max_train_size=10)
    for train_idx, _ in tscv.split(X, groups=groups):
        assert train_idx.size <= 10


# ---------------------------------------------------------------------------
# cross_validate_with_groups
# ---------------------------------------------------------------------------


def test_cross_validate_with_groups_default(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    X, y, groups = grouped_xy
    result = cross_validate_with_groups(LogisticRegression(), X, y, groups=groups)
    assert isinstance(result, CrossValResult)
    assert result.splits == 5
    assert "accuracy" in result.fold_scores
    assert result.fold_scores["accuracy"].size == 5
    assert 0.0 <= result.mean_scores["accuracy"] <= 1.0


def test_cross_validate_with_groups_custom_metrics(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    X, y, groups = grouped_xy
    result = cross_validate_with_groups(
        LogisticRegression(),
        X,
        y,
        groups=groups,
        metrics={"accuracy": accuracy_score, "balanced": balanced_accuracy_score},
    )
    assert set(result.metric_names) == {"accuracy", "balanced"}


def test_cross_validate_with_groups_use_proba(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    X, y, groups = grouped_xy
    from sklearn.metrics import roc_auc_score

    result = cross_validate_with_groups(
        LogisticRegression(),
        X,
        y,
        groups=groups,
        metrics={"auc": roc_auc_score},
        use_proba=True,
    )
    assert 0.0 <= result.mean_scores["auc"] <= 1.0


def test_cross_validate_with_groups_no_groups(grouped_xy) -> None:  # type: ignore[no-untyped-def]
    # When groups is None, defaults to StratifiedKFold.
    X, y, _ = grouped_xy
    result = cross_validate_with_groups(LogisticRegression(), X, y)
    assert result.splits == 5
