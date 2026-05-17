"""Smoke tests: the package imports and basic surface is present."""

from __future__ import annotations

import importlib

import pytest

import clinikit
from clinikit.cli import app as cli_app

PUBLIC_SUBMODULES = [
    "clinikit",
    "clinikit.models",
    "clinikit.preprocessing",
    "clinikit.metrics",
    "clinikit.curves",
    "clinikit.protocols",
    "clinikit.leaderboard",
    "clinikit.report",
    "clinikit.audit",
    "clinikit.governance",
    "clinikit.reproducibility",
    "clinikit.datasets",
    "clinikit.cli",
    "clinikit.plots",
    "clinikit.thresholds",
    "clinikit.calibration",
    "clinikit.statistics",
    "clinikit.diagnostics",
    "clinikit.cost_sensitive",
    "clinikit.monitor",
    "clinikit.modelcard",
    "clinikit.cross_val",
    "clinikit.explainability",
    "clinikit.automl",
    "clinikit.external_val",
    "clinikit.time_split",
    "clinikit.active_learning",
    "clinikit.synthetic",
]


@pytest.mark.parametrize("name", PUBLIC_SUBMODULES)
def test_submodule_imports(name: str) -> None:
    """Every advertised submodule must be importable."""
    module = importlib.import_module(name)
    assert module is not None


def test_top_level_metadata() -> None:
    """The top-level package exposes version and author metadata."""
    assert isinstance(clinikit.__version__, str) and clinikit.__version__
    assert isinstance(clinikit.__author__, str) and clinikit.__author__
    assert clinikit.__license__ == "MIT"


def test_cli_app_present() -> None:
    """The Typer app object is exported from clinikit.cli."""
    assert cli_app is not None
    assert cli_app.info.name == "clinikit"
