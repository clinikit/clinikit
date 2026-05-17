"""Tests for clinikit.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
)
from clinikit.metrics import expected_calibration_error


@pytest.fixture
def miscalibrated(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Miscalibrated probabilities: true positives have higher base than 0.5
    but the raw probabilities are pushed even further out."""
    n = 400
    y = rng.integers(0, 2, size=n)
    # Sharper-than-true probabilities — overconfident.
    base = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    sharp = np.where(y == 1, 0.5 + (base - 0.5) ** 0.5, 0.5 - (0.5 - base) ** 0.5)
    return y, np.clip(sharp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Surface and basic invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calibrator_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureScaler])
def test_fit_transform_returns_unit_interval(
    calibrator_cls, miscalibrated: tuple[np.ndarray, np.ndarray]
) -> None:  # type: ignore[no-untyped-def]
    y, p = miscalibrated
    cal = calibrator_cls().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


@pytest.mark.parametrize("calibrator_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureScaler])
def test_fit_transform_convenience(
    calibrator_cls, miscalibrated: tuple[np.ndarray, np.ndarray]
) -> None:  # type: ignore[no-untyped-def]
    y, p = miscalibrated
    direct = calibrator_cls().fit_transform(p, y)
    chained = calibrator_cls().fit(p, y).transform(p)
    assert np.allclose(direct, chained)


@pytest.mark.parametrize("calibrator_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureScaler])
def test_rejects_out_of_range_probabilities(calibrator_cls) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator_cls().fit([0.1, 1.5], [0, 1])


@pytest.mark.parametrize("calibrator_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureScaler])
def test_rejects_multiclass(calibrator_cls) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="binary-only"):
        calibrator_cls().fit([0.1, 0.5, 0.9], [0, 1, 2])


# ---------------------------------------------------------------------------
# Calibration improves ECE (loose check — the synthetic dataset is
# overconfident, so any sane calibrator should not make ECE worse).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calibrator_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureScaler])
def test_calibration_does_not_worsen_ece(
    calibrator_cls, miscalibrated: tuple[np.ndarray, np.ndarray]
) -> None:  # type: ignore[no-untyped-def]
    y, p = miscalibrated
    base_ece = expected_calibration_error(y, p, n_bins=10)
    calibrated = calibrator_cls().fit_transform(p, y)
    cal_ece = expected_calibration_error(y, calibrated, n_bins=10)
    # Allow a small slack — bin counts shift around with calibration.
    assert cal_ece <= base_ece + 0.05


# ---------------------------------------------------------------------------
# Calibrator-specific tests
# ---------------------------------------------------------------------------


def test_platt_calibrator_coef_intercept(miscalibrated: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = miscalibrated
    cal = PlattCalibrator().fit(p, y)
    assert isinstance(cal.coef_, float)
    assert isinstance(cal.intercept_, float)


def test_platt_calibrator_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        PlattCalibrator().fit([0.1, 0.5, 0.9], [0, 0, 0])


def test_temperature_scaler_fits_positive_temperature(
    miscalibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = miscalibrated
    cal = TemperatureScaler().fit(p, y)
    assert cal.temperature_ > 0.0


def test_isotonic_calibrator_is_monotone(
    miscalibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = miscalibrated
    cal = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.0, 1.0, 50)
    out = cal.transform(grid)
    # Isotonic regression guarantees a monotone non-decreasing map.
    assert np.all(np.diff(out) >= -1e-12)
