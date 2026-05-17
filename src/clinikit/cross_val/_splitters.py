"""Group-aware cross-validation utilities.

Three helpers:

- :class:`GroupStratifiedKFold` — wraps
  :class:`sklearn.model_selection.StratifiedGroupKFold` so we have a
  stable name and consistent defaults.
- :class:`TimeSeriesGroupSplit` — chronological splits that keep all
  rows belonging to one group on the same side of every split.
- :func:`cross_validate_with_groups` — convenience around
  :func:`sklearn.model_selection.cross_validate` that forwards
  ``groups`` to a group-aware splitter and accepts a ``metrics`` dict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "CrossValResult",
    "GroupStratifiedKFold",
    "TimeSeriesGroupSplit",
    "cross_validate_with_groups",
]


class GroupStratifiedKFold(StratifiedGroupKFold):
    """Group-aware *and* class-stratified K-fold.

    Thin re-export of
    :class:`sklearn.model_selection.StratifiedGroupKFold`. The wrapper
    gives the splitter a stable name inside ``clinikit`` and keeps the
    defaults consistent with the rest of the toolkit (``shuffle=True``
    with a seeded ``random_state``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


@dataclass(slots=True)
class TimeSeriesGroupSplit:
    """Chronological folds that respect group membership.

    Rows belonging to the same group always end up on the same side of
    every fold boundary. Each fold's test set is a *later* time slice
    than its train set, so the splitter can be used to validate
    deployment-style "train on past, evaluate on future" workflows.

    Parameters
    ----------
    n_splits : int, default 5
    max_train_size : int, optional
        Cap the train set to the most recent ``max_train_size`` rows
        (after time ordering). ``None`` keeps the full history.

    Examples
    --------
    >>> import numpy as np
    >>> from clinikit.cross_val import TimeSeriesGroupSplit
    >>> times = np.arange(50)
    >>> groups = np.repeat(np.arange(10), 5)
    >>> tscv = TimeSeriesGroupSplit(n_splits=3)
    >>> splits = list(tscv.split(times, groups=groups))
    >>> len(splits)
    3
    """

    n_splits: int = 5
    max_train_size: int | None = None

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        del X, y, groups
        return self.n_splits

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        del y
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1; got {self.n_splits!r}.")
        X_arr = np.asarray(X)
        n = X_arr.shape[0]
        if groups is None:
            raise ValueError("TimeSeriesGroupSplit requires `groups`.")
        groups_arr = np.asarray(groups)
        check_consistent_length(X_arr, groups_arr)

        # Order each group by its first appearance — that determines
        # the chronological rank of the whole group.
        original_order = np.arange(n)
        first_seen = np.full(groups_arr.shape, fill_value=n + 1, dtype=np.int64)
        seen: dict[Any, int] = {}
        for i, g in enumerate(groups_arr.tolist()):
            if g not in seen:
                seen[g] = i
            first_seen[i] = seen[g]
        sort_idx = np.lexsort((original_order, first_seen))

        # Build n_splits + 1 cut points so we get n_splits expanding
        # train windows with disjoint test slices.
        unique_groups = [g for g, _ in sorted(seen.items(), key=lambda kv: kv[1])]
        n_groups = len(unique_groups)
        if self.n_splits >= n_groups:
            raise ValueError(f"n_splits={self.n_splits} must be < n_groups={n_groups}.")
        group_edges = np.linspace(0, n_groups, self.n_splits + 2, dtype=np.int64)
        # Map each row to its group's rank (position in unique_groups).
        group_rank = np.empty(n, dtype=np.int64)
        rank_lookup = {g: i for i, g in enumerate(unique_groups)}
        for i, g in enumerate(groups_arr.tolist()):
            group_rank[i] = rank_lookup[g]

        for k in range(self.n_splits):
            train_high = int(group_edges[k + 1])
            test_low = train_high
            test_high = int(group_edges[k + 2])
            if test_low == test_high:
                continue
            train_mask = group_rank < train_high
            test_mask = (group_rank >= test_low) & (group_rank < test_high)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if self.max_train_size is not None and train_idx.size > self.max_train_size:
                train_idx = train_idx[-self.max_train_size :]
            # Preserve the chronological order of rows.
            train_idx = train_idx[np.argsort(sort_idx[train_idx])]
            test_idx = test_idx[np.argsort(sort_idx[test_idx])]
            yield train_idx, test_idx


@dataclass(slots=True)
class CrossValResult:
    """Output of :func:`cross_validate_with_groups`."""

    splits: int
    metric_names: tuple[str, ...]
    fold_scores: dict[str, NDArray[np.float64]]
    mean_scores: dict[str, float]
    std_scores: dict[str, float]


def cross_validate_with_groups(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    cv: Any = None,
    metrics: dict[str, Callable[..., float]] | None = None,
    use_proba: bool = False,
) -> CrossValResult:
    """Run group-aware cross-validation and report per-fold metrics.

    Parameters
    ----------
    estimator : sklearn estimator
        Cloned once per fold.
    X, y : array-like
    groups : array-like, optional
        Group label per sample. Required when ``cv`` is a group-aware
        splitter; ignored otherwise.
    cv : sklearn splitter, optional
        Defaults to :class:`GroupStratifiedKFold(n_splits=5)` if
        ``groups`` is supplied; otherwise five-fold stratified CV.
    metrics : dict[str, callable], optional
        Map of name -> metric function with signature
        ``metric(y_true, y_pred_or_score) -> float``. Defaults to a
        single ``"accuracy"`` entry.
    use_proba : bool, default False
        When True, metrics are called with positive-class
        probabilities (``predict_proba(X)[:, pos_idx]``) instead of
        hard predictions. Requires the estimator to support
        ``predict_proba``.

    Returns
    -------
    CrossValResult
        ``fold_scores`` is keyed by metric name; ``mean_scores`` and
        ``std_scores`` summarise across folds.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    check_consistent_length(X_arr, y_arr)
    if metrics is None:
        metrics = {"accuracy": accuracy_score}

    if cv is None:
        cv = (
            GroupStratifiedKFold(n_splits=5, random_state=0)
            if groups is not None
            else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        )

    if groups is not None:
        groups_arr: NDArray | None = np.asarray(groups)
        check_consistent_length(X_arr, groups_arr)
    else:
        groups_arr = None

    splitter_kwargs: dict[str, Any] = {"X": X_arr, "y": y_arr}
    if groups_arr is not None:
        splitter_kwargs["groups"] = groups_arr

    fold_scores: dict[str, list[float]] = {name: [] for name in metrics}
    n_splits = 0
    for train_idx, test_idx in cv.split(**splitter_kwargs):
        n_splits += 1
        model = clone(estimator)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        if use_proba:
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    f"use_proba=True but estimator {type(model).__name__} has no predict_proba."
                )
            classes = np.asarray(model.classes_)
            pos_col = int(np.where(classes == classes[-1])[0][0])
            y_score = model.predict_proba(X_arr[test_idx])[:, pos_col]
        else:
            y_score = model.predict(X_arr[test_idx])
        for name, fn in metrics.items():
            fold_scores[name].append(float(fn(y_arr[test_idx], y_score)))

    fold_arrays = {name: np.asarray(v, dtype=np.float64) for name, v in fold_scores.items()}
    return CrossValResult(
        splits=n_splits,
        metric_names=tuple(metrics.keys()),
        fold_scores=fold_arrays,
        mean_scores={name: float(arr.mean()) for name, arr in fold_arrays.items()},
        std_scores={
            name: float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            for name, arr in fold_arrays.items()
        },
    )
