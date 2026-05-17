"""Tests for clinikit.audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinikit.audit import (
    DocumentationCheckResult,
    LeakageReport,
    SubgroupFairnessReport,
    detect_leakage,
    documentation_checklist,
    subgroup_fairness,
)

# ---------------------------------------------------------------------------
# detect_leakage
# ---------------------------------------------------------------------------


def test_detect_leakage_clean(rng: np.random.Generator) -> None:
    X_train = rng.standard_normal((200, 4))
    X_test = rng.standard_normal((100, 4))
    y_train = rng.integers(0, 2, size=200)
    y_test = rng.integers(0, 2, size=100)
    report = detect_leakage(X_train, X_test, y_train, y_test)
    assert isinstance(report, LeakageReport)
    assert report.duplicate_rows == 0
    assert "no leakage" in report.summary.lower()


def test_detect_leakage_duplicate_rows_flagged() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 4))
    # Plant 10 duplicates from train into the test set.
    X_test = np.vstack([X_train[:10], rng.standard_normal((90, 4))])
    y_train = rng.integers(0, 2, size=100)
    y_test = rng.integers(0, 2, size=100)
    report = detect_leakage(X_train, X_test, y_train, y_test)
    assert report.duplicate_rows == 10
    assert report.duplicate_row_rate == pytest.approx(0.1)


def test_detect_leakage_label_drift() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 3))
    X_test = rng.standard_normal((100, 3))
    y_train = np.zeros(100, dtype=int)
    y_train[:90] = 1  # 90% positive
    y_test = np.zeros(100, dtype=int)
    y_test[:10] = 1  # 10% positive
    report = detect_leakage(X_train, X_test, y_train, y_test)
    assert report.label_distribution_alert is True


def test_detect_leakage_mismatched_feature_count() -> None:
    with pytest.raises(ValueError, match="n_features"):
        detect_leakage(np.zeros((4, 3)), np.zeros((4, 2)), [0, 1, 0, 1], [0, 1, 0, 1])


# ---------------------------------------------------------------------------
# subgroup_fairness
# ---------------------------------------------------------------------------


def test_subgroup_fairness_returns_per_group(rng: np.random.Generator) -> None:
    y = rng.integers(0, 2, size=120)
    pred = rng.integers(0, 2, size=120)
    groups = rng.integers(0, 3, size=120)
    report = subgroup_fairness(y, pred, groups)
    assert isinstance(report, SubgroupFairnessReport)
    assert isinstance(report.per_group, pd.DataFrame)
    assert set(report.per_group.columns) >= {"accuracy", "sensitivity", "specificity"}


def test_subgroup_fairness_perfectly_fair() -> None:
    # When every group has the same metrics, all disparities are 0.
    y = [0, 1, 0, 1] * 6
    pred = [0, 1, 0, 1] * 6
    groups = [0, 0, 1, 1] * 6
    report = subgroup_fairness(y, pred, groups)
    assert report.accuracy_disparity == 0.0


def test_subgroup_fairness_strong_disparity() -> None:
    # Group 0 perfect, group 1 perfectly wrong -> accuracy_disparity = 1.0
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    pred = [0, 1, 0, 1, 1, 0, 1, 0]
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    report = subgroup_fairness(y, pred, groups)
    assert report.accuracy_disparity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# documentation_checklist
# ---------------------------------------------------------------------------


def test_documentation_checklist_complete() -> None:
    run = {
        "run_id": "r1",
        "dataset_name": "pima",
        "dataset_hash": "sha256:abc",
        "model_class": "LogisticRegression",
        "random_state": 42,
        "cv_strategy": "stratified_kfold",
        "threshold_strategy": "accuracy_max",
        "threshold": 0.45,
        "accuracy": 0.83,
        "sensitivity": 0.79,
        "specificity": 0.88,
        "notes": "smoke run",
    }
    result = documentation_checklist(run)
    assert isinstance(result, DocumentationCheckResult)
    assert result.complete is True
    assert result.missing_fields == ()
    assert result.empty_fields == ()


def test_documentation_checklist_missing_fields() -> None:
    result = documentation_checklist({"run_id": "r1"})
    assert result.complete is False
    assert "dataset_name" in result.missing_fields


def test_documentation_checklist_empty_values() -> None:
    run = {"run_id": "r1", "dataset_name": ""}
    result = documentation_checklist(run, required_fields=("run_id", "dataset_name"))
    assert "dataset_name" in result.empty_fields
    assert result.complete is False


def test_documentation_checklist_custom_required() -> None:
    result = documentation_checklist({"foo": "bar"}, required_fields=("foo",))
    assert result.complete is True
