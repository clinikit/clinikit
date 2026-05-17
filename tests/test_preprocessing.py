"""Tests for clinikit.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_estimator

from clinikit.preprocessing import (
    DomainAwareImputer,
    KNNImputer,
    MedianImputer,
    MICEImputer,
    MinMaxScaler,
    MissForestImputer,
    MissingIndicator,
    OutlierFlag,
    QuantileScaler,
    RobustScaler,
    StandardScaler,
    YeoJohnsonScaler,
)

# ---------------------------------------------------------------------------
# Smoke: every public transformer fit/transforms a small numeric matrix.
# ---------------------------------------------------------------------------

ALL_TRANSFORMERS = [
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    QuantileScaler,
    YeoJohnsonScaler,
    MedianImputer,
    KNNImputer,
    MICEImputer,
    MissForestImputer,
    OutlierFlag,
]


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS)
def test_fit_transform_smoke(cls, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
    X = rng.standard_normal((30, 4))
    # Inject some NaNs for imputers and outliers; scalers handle them
    # via sklearn's own validation.
    if "Imputer" in cls.__name__:
        X[0, 0] = np.nan
        X[5, 2] = np.nan
    out = cls().fit_transform(X)
    assert out is not None
    assert np.asarray(out).shape[0] == X.shape[0]


# ---------------------------------------------------------------------------
# Scalers — quick numeric sanity checks
# ---------------------------------------------------------------------------


def test_standard_scaler_zero_mean_unit_std(rng: np.random.Generator) -> None:
    X = rng.standard_normal((200, 3)) * 5.0 + 10.0
    Xt = StandardScaler().fit_transform(X)
    assert Xt.mean(axis=0) == pytest.approx(0.0, abs=1e-10)
    assert Xt.std(axis=0, ddof=0) == pytest.approx(1.0, abs=1e-10)


def test_minmax_scaler_default_range(rng: np.random.Generator) -> None:
    X = rng.standard_normal((50, 2)) * 10.0
    Xt = MinMaxScaler().fit_transform(X)
    assert Xt.min(axis=0) == pytest.approx(0.0)
    assert Xt.max(axis=0) == pytest.approx(1.0)


def test_quantile_scaler_normal_distribution_default(rng: np.random.Generator) -> None:
    X = rng.exponential(scale=2.0, size=(500, 1))
    Xt = QuantileScaler(n_quantiles=200, random_state=0).fit_transform(X)
    # Should be approximately N(0, 1) after the quantile transform.
    assert abs(Xt.mean()) < 0.2
    assert 0.8 < Xt.std() < 1.2


def test_yeojohnson_scaler_handles_negatives() -> None:
    X = np.array([[-5.0], [-1.0], [0.0], [1.0], [5.0]])
    YeoJohnsonScaler().fit_transform(X)  # must not raise


# ---------------------------------------------------------------------------
# Imputers
# ---------------------------------------------------------------------------


def test_median_imputer_fills_with_median() -> None:
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    Xt = MedianImputer().fit_transform(X)
    # Column 0 median is 2.0 (of [1, 3]); column 1 median is 3.0 (of [2, 4]).
    assert Xt[1, 0] == pytest.approx(2.0)
    assert Xt[2, 1] == pytest.approx(3.0)


def test_knn_imputer_runs(rng: np.random.Generator) -> None:
    X = rng.standard_normal((20, 3))
    X[0, 0] = np.nan
    Xt = KNNImputer(n_neighbors=3).fit_transform(X)
    assert not np.isnan(Xt).any()


def test_mice_imputer_runs(rng: np.random.Generator) -> None:
    X = rng.standard_normal((30, 4))
    X[0, 0] = np.nan
    X[5, 2] = np.nan
    Xt = MICEImputer(max_iter=5, random_state=0).fit_transform(X)
    assert not np.isnan(Xt).any()


def test_missforest_imputer_runs(rng: np.random.Generator) -> None:
    X = rng.standard_normal((30, 4))
    X[0, 0] = np.nan
    Xt = MissForestImputer(n_estimators=10, max_iter=2, random_state=0).fit_transform(X)
    assert not np.isnan(Xt).any()


# ---------------------------------------------------------------------------
# DomainAwareImputer
# ---------------------------------------------------------------------------


def test_domain_aware_imputer_with_dataframe() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 0.0, 1.0]})
    imp = DomainAwareImputer(strategies={"a": "median", "b": "most_frequent"})
    Xt = imp.fit_transform(df)
    assert Xt[1, 0] == pytest.approx(2.0)  # median of [1, 3]
    # most_frequent on [nan, 0, 1] picks the smallest mode -> 0.0
    assert Xt[0, 1] == pytest.approx(0.0)


def test_domain_aware_imputer_with_ndarray_uses_int_keys() -> None:
    X = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, 7.0]])
    imp = DomainAwareImputer(strategies={0: "median", 1: "mean"})
    Xt = imp.fit_transform(X)
    assert Xt[1, 0] == pytest.approx(2.0)
    assert Xt[0, 1] == pytest.approx(6.0)


def test_domain_aware_imputer_default_strategy() -> None:
    # Both columns fall back to the default median.
    X = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, 7.0]])
    Xt = DomainAwareImputer().fit_transform(X)
    assert Xt[1, 0] == pytest.approx(2.0)
    assert Xt[0, 1] == pytest.approx(6.0)


def test_domain_aware_imputer_constant_strategy() -> None:
    X = np.array([[1.0], [np.nan], [3.0]])
    Xt = DomainAwareImputer(strategies={0: ("constant", -1.0)}).fit_transform(X)
    assert Xt[1, 0] == pytest.approx(-1.0)


def test_domain_aware_imputer_invalid_tuple_strategy() -> None:
    X = np.array([[1.0], [2.0]])
    bad = DomainAwareImputer(strategies={0: ("median", None)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="constant"):
        bad.fit(X)


def test_domain_aware_imputer_feature_mismatch_at_transform() -> None:
    X_train = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, 7.0]])
    X_test = np.array([[1.0]])
    imp = DomainAwareImputer().fit(X_train)
    with pytest.raises(ValueError, match="features"):
        imp.transform(X_test)


def test_domain_aware_imputer_get_feature_names_out_with_dataframe() -> None:
    df = pd.DataFrame({"age": [1.0, 2.0], "bmi": [3.0, 4.0]})
    imp = DomainAwareImputer().fit(df)
    assert list(imp.get_feature_names_out()) == ["age", "bmi"]


def test_domain_aware_imputer_get_feature_names_out_with_ndarray() -> None:
    # When fitted on an ndarray, validate_data does not set
    # feature_names_in_, so the fallback must synthesise "x0", "x1", ...
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    imp = DomainAwareImputer().fit(X)
    assert list(imp.get_feature_names_out()) == ["x0", "x1"]


# ---------------------------------------------------------------------------
# OutlierFlag
# ---------------------------------------------------------------------------


def test_outlier_flag_iqr_marks_extreme() -> None:
    X = np.array([[1.0], [1.0], [1.0], [1.0], [100.0]])
    flag = OutlierFlag(method="iqr").fit_transform(X)
    assert flag.ravel().tolist() == [0, 0, 0, 0, 1]


def test_outlier_flag_zscore_marks_extreme() -> None:
    X = np.array([[0.0]] * 99 + [[10.0]])
    flag = OutlierFlag(method="zscore", k=3.0).fit_transform(X)
    assert flag[-1, 0] == 1
    assert flag[0, 0] == 0


def test_outlier_flag_preserves_shape(rng: np.random.Generator) -> None:
    X = rng.standard_normal((50, 4))
    flag = OutlierFlag().fit_transform(X)
    assert flag.shape == X.shape
    assert flag.dtype == np.int64


def test_outlier_flag_treats_nan_as_inrange() -> None:
    X = np.array([[1.0]] * 5 + [[np.nan]] + [[100.0]])
    flag = OutlierFlag().fit_transform(X)
    assert flag[5, 0] == 0  # NaN row -> not flagged
    assert flag[6, 0] == 1  # extreme value -> flagged


def test_outlier_flag_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        OutlierFlag(method="nonsense").fit(np.zeros((10, 1)))  # type: ignore[arg-type]


def test_outlier_flag_feature_mismatch_at_transform(rng: np.random.Generator) -> None:
    flag = OutlierFlag().fit(rng.standard_normal((30, 3)))
    with pytest.raises(ValueError, match="features"):
        flag.transform(rng.standard_normal((5, 2)))


def test_outlier_flag_feature_names_out_with_dataframe() -> None:
    df = pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0], "bmi": [10.0, 20.0, 30.0, 40.0]})
    flag = OutlierFlag().fit(df)
    assert list(flag.get_feature_names_out()) == ["age_outlier", "bmi_outlier"]


def test_outlier_flag_feature_names_out_with_ndarray() -> None:
    # Fitted on an ndarray, so feature_names_in_ is absent and the
    # fallback should synthesise "x0_outlier", "x1_outlier".
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    flag = OutlierFlag().fit(X)
    assert list(flag.get_feature_names_out()) == ["x0_outlier", "x1_outlier"]


# ---------------------------------------------------------------------------
# MissingIndicator — thin sklearn re-export, smoke only
# ---------------------------------------------------------------------------


def test_missing_indicator_runs() -> None:
    X = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, 7.0]])
    out = MissingIndicator(features="all").fit_transform(X)
    assert out.shape == X.shape


# ---------------------------------------------------------------------------
# sklearn API compliance — check_estimator on each public transformer.
# ---------------------------------------------------------------------------


SKLEARN_COMPAT_TRANSFORMERS = [
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    QuantileScaler,
    YeoJohnsonScaler,
    MedianImputer,
    KNNImputer,
    MICEImputer,
    MissForestImputer,
    OutlierFlag,
    DomainAwareImputer,
]


@pytest.mark.sklearn_api
@pytest.mark.parametrize(
    "cls",
    SKLEARN_COMPAT_TRANSFORMERS,
    ids=[c.__name__ for c in SKLEARN_COMPAT_TRANSFORMERS],
)
def test_check_estimator(cls) -> None:  # type: ignore[no-untyped-def]
    check_estimator(cls())
