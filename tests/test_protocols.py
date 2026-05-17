"""Tests for clinikit.protocols."""

from __future__ import annotations

import numpy as np
import pytest

from clinikit.leaderboard import LEADERBOARD_COLUMNS
from clinikit.protocols import (
    AuditProtocol,
    BaseProtocol,
    DefensibleProtocol,
    DeploymentProtocol,
    MaxScoreProtocol,
    OriginalOnlyProtocol,
    ProtocolResult,
)

PROTOCOL_CLASSES = [
    DefensibleProtocol,
    MaxScoreProtocol,
    OriginalOnlyProtocol,
    DeploymentProtocol,
    AuditProtocol,
]


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + 0.3 * rng.standard_normal(200) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# All five protocols obey the BaseProtocol contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", PROTOCOL_CLASSES)
def test_protocol_run_returns_result(protocol_cls, binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    proto = protocol_cls(random_state=0)
    result = proto.run(X, y)
    assert isinstance(result, ProtocolResult)
    assert result.protocol_name == proto.name
    assert 0.0 <= result.threshold <= 1.0
    assert result.duration_seconds >= 0.0


@pytest.mark.parametrize("protocol_cls", PROTOCOL_CLASSES)
def test_protocol_metrics_complete(protocol_cls, binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    result = protocol_cls(random_state=0).run(X, y)
    expected_keys = {
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f2_score",
        "mcc",
        "balanced_accuracy",
        "brier_score",
        "expected_calibration_error",
    }
    assert expected_keys <= set(result.metrics)
    for v in result.metrics.values():
        assert isinstance(v, float)


@pytest.mark.parametrize("protocol_cls", PROTOCOL_CLASSES)
def test_protocol_leaderboard_row_uses_known_columns(protocol_cls, binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    result = protocol_cls(random_state=0).run(X, y)
    extras = [k for k in result.leaderboard_row if k not in LEADERBOARD_COLUMNS]
    assert not extras, f"unexpected leaderboard columns: {extras!r}"


@pytest.mark.parametrize("protocol_cls", PROTOCOL_CLASSES)
def test_protocol_rejects_multiclass(protocol_cls) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = np.tile([0, 1, 2], 10)
    with pytest.raises(ValueError, match="binary-only"):
        protocol_cls().run(X, y)


# ---------------------------------------------------------------------------
# Per-protocol identity checks
# ---------------------------------------------------------------------------


def test_defensible_uses_recall_constrained() -> None:
    p = DefensibleProtocol()
    assert p.threshold_strategy == "recall_constrained"
    assert p.threshold_kwargs == {"min_recall": 0.9}


def test_max_score_uses_accuracy_max() -> None:
    assert MaxScoreProtocol().threshold_strategy == "accuracy_max"


def test_original_only_passes_through_features() -> None:
    p = OriginalOnlyProtocol()
    # The identity preprocessor should not change the feature scale.
    assert type(p.preprocessing).__name__ == "_Identity"


def test_deployment_has_wider_test_split() -> None:
    assert DeploymentProtocol().test_size == 0.3


def test_audit_has_largest_test_split() -> None:
    assert AuditProtocol().test_size == 0.4
    assert AuditProtocol().threshold_kwargs == {"min_recall": 0.95}


# ---------------------------------------------------------------------------
# Determinism + threshold range
# ---------------------------------------------------------------------------


def test_protocol_is_deterministic(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    p1 = MaxScoreProtocol(random_state=0).run(X, y)
    p2 = MaxScoreProtocol(random_state=0).run(X, y)
    assert p1.metrics == p2.metrics
    assert p1.threshold == p2.threshold


def test_base_protocol_run_smoke(binary_xy) -> None:  # type: ignore[no-untyped-def]
    X, y = binary_xy
    result = BaseProtocol(random_state=0).run(X, y)
    assert isinstance(result, ProtocolResult)
