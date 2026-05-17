"""Tests for clinikit.external_val."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from clinikit.external_val import domain_shift_report, validate_external

# ---------------------------------------------------------------------------
# validate_external
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_validate_external_returns_one_row_per_external(synthetic_pair) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_pair
    table = validate_external(
        LogisticRegression(),
        train_dataset=(X, y),
        external_datasets={
            "site_a": (X[:30], y[:30]),
            "site_b": (X[30:60], y[30:60]),
        },
    )
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["site_a", "site_b"]
    assert set(table.columns) >= {"n_samples", "accuracy", "sensitivity", "specificity"}


def test_validate_external_custom_metrics(synthetic_pair) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_pair
    table = validate_external(
        LogisticRegression(),
        train_dataset=(X, y),
        external_datasets={"hold": (X, y)},
        metrics={"acc": accuracy_score},
    )
    assert "acc" in table.columns
    assert 0.0 <= table.loc["hold", "acc"] <= 1.0


def test_validate_external_rejects_mismatched_features(synthetic_pair) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_pair
    with pytest.raises(ValueError, match="features"):
        validate_external(
            LogisticRegression(),
            train_dataset=(X, y),
            external_datasets={"bad": (X[:, :2], y)},
        )


def test_validate_external_empty_externals(synthetic_pair) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_pair
    with pytest.raises(ValueError, match="at least one"):
        validate_external(LogisticRegression(), train_dataset=(X, y), external_datasets={})


# ---------------------------------------------------------------------------
# domain_shift_report
# ---------------------------------------------------------------------------


def test_domain_shift_report_clean(rng: np.random.Generator) -> None:
    ref = rng.standard_normal((500, 3))
    tgt = rng.standard_normal((500, 3))
    report = domain_shift_report(ref, tgt)
    assert isinstance(report, pd.DataFrame)
    assert report.shape == (3, 5)
    assert {"ks_statistic", "ks_p_value", "ks_drift", "psi_statistic", "psi_drift"} == set(
        report.columns
    )


def test_domain_shift_report_flags_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((500, 3))
    tgt = ref + np.array([0.0, 0.0, 5.0])  # third column shifted
    report = domain_shift_report(ref, tgt)
    assert report.loc[2, "ks_drift"] is np.True_ or bool(report.loc[2, "ks_drift"]) is True
    assert report.loc[2, "psi_drift"] is np.True_ or bool(report.loc[2, "psi_drift"]) is True


def test_domain_shift_report_mismatched_features() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((100, 3))
    tgt = rng.standard_normal((100, 4))
    with pytest.raises(ValueError, match="features"):
        domain_shift_report(ref, tgt)


def test_domain_shift_report_handles_1d_input(rng: np.random.Generator) -> None:
    ref = rng.standard_normal(200)
    tgt = rng.standard_normal(200)
    report = domain_shift_report(ref, tgt)
    assert report.shape == (1, 5)
