"""Shared pytest fixtures for the clinikit test suite."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

DEFAULT_SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A deterministic numpy generator shared across tests."""
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def binary_classification_data(
    rng: np.random.Generator,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64]]]:
    """Small synthetic binary-classification dataset.

    Returns
    -------
    (X, y)
        ``X`` has shape ``(200, 5)``; ``y`` has shape ``(200,)`` with
        roughly balanced labels.
    """
    n_samples = 200
    n_features = 5
    X = rng.standard_normal((n_samples, n_features))
    linear = X @ rng.standard_normal(n_features)
    y = (linear > 0).astype(np.int64)
    yield X, y
