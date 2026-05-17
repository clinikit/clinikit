"""Tests for clinikit.synthetic."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from clinikit.synthetic import TSTRReport, fit_ctgan, fit_tvae, tstr_safety_gate

sdv_available = importlib.util.find_spec("sdv") is not None


@pytest.fixture
def real_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((150, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# fit_ctgan / fit_tvae — import-error path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sdv_available, reason="run only when sdv is missing")
def test_fit_ctgan_raises_clear_import_error(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, _ = real_dataset
    with pytest.raises(ImportError, match="sdv"):
        fit_ctgan(X, epochs=1)


@pytest.mark.skipif(sdv_available, reason="run only when sdv is missing")
def test_fit_tvae_raises_clear_import_error(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, _ = real_dataset
    with pytest.raises(ImportError, match="sdv"):
        fit_tvae(X, epochs=1)


# ---------------------------------------------------------------------------
# fit_ctgan / fit_tvae — smoke when SDV is installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sdv_available, reason="sdv not installed")
def test_fit_ctgan_smoke(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, _ = real_dataset
    synth = fit_ctgan(X, epochs=1, random_state=0)
    assert hasattr(synth, "sample")


@pytest.mark.skipif(not sdv_available, reason="sdv not installed")
def test_fit_tvae_smoke(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, _ = real_dataset
    synth = fit_tvae(X, epochs=1, random_state=0)
    assert hasattr(synth, "sample")


# ---------------------------------------------------------------------------
# tstr_safety_gate (pure logic — no optional dep)
# ---------------------------------------------------------------------------


def test_tstr_safety_gate_returns_report(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    # Use the real data as a faux-synthetic set — should pass.
    report = tstr_safety_gate(X, y, X, y, tolerance=0.1, random_state=0)
    assert isinstance(report, TSTRReport)
    assert 0.0 <= report.real_score <= 1.0
    assert 0.0 <= report.synth_score <= 1.0
    assert report.tolerance == 0.1


def test_tstr_safety_gate_passes_when_synth_matches_real(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    report = tstr_safety_gate(X, y, X, y, tolerance=0.1, random_state=0)
    assert report.passed
    assert report.gap == pytest.approx(report.real_score - report.synth_score)


def test_tstr_safety_gate_flags_useless_synth(real_dataset, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    junk_X = rng.standard_normal(X.shape)
    junk_y = rng.integers(0, 2, size=y.shape)
    report = tstr_safety_gate(X, y, junk_X, junk_y, tolerance=0.01, random_state=0)
    # Random synthetic data shouldn't recover the real-on-real baseline
    # under such a tight tolerance.
    assert not report.passed
    assert report.gap > 0.01


def test_tstr_safety_gate_custom_estimator_and_metric(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    report = tstr_safety_gate(
        X,
        y,
        X,
        y,
        tolerance=0.2,
        estimator=LogisticRegression(max_iter=200),
        metric=balanced_accuracy_score,
        random_state=0,
    )
    assert isinstance(report, TSTRReport)


def test_tstr_safety_gate_rejects_negative_tolerance(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    with pytest.raises(ValueError, match="tolerance"):
        tstr_safety_gate(X, y, X, y, tolerance=-0.05)


def test_tstr_safety_gate_rejects_bad_test_size(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    with pytest.raises(ValueError, match="test_size"):
        tstr_safety_gate(X, y, X, y, test_size=1.5)


def test_tstr_safety_gate_feature_count_mismatch(real_dataset) -> None:  # type: ignore[no-untyped-def]
    X, y = real_dataset
    with pytest.raises(ValueError, match="feature counts"):
        tstr_safety_gate(X, y, X[:, :2], y)
