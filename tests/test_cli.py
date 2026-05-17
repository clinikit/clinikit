"""End-to-end tests for the clinikit CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from clinikit.cli import app

runner = CliRunner()


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Tiny binary-classification CSV with a `target` column."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 80
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0).astype(int)
    g = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "group": g,
            "target": y,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "clinikit" in result.stdout


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("train", "benchmark", "audit", "validate", "report"):
        assert cmd in result.stdout


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_yaml_config(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "protocol: max_score\ndataset: pima\nrandom_state: 0\nnotes: cli smoke\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.joblib"
    lb_path = tmp_path / "runs.csv"
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(cfg),
            "--output",
            str(model_path),
            "--leaderboard",
            str(lb_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert model_path.is_file()
    assert lb_path.is_file()
    assert "accuracy=" in result.stdout


def test_train_json_config(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"protocol": "max_score", "dataset": "wisconsin", "random_state": 0}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0, result.stdout


def test_train_unknown_protocol_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("protocol: nonsense\ndataset: pima\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code != 0


def test_train_unsupported_config_suffix(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("not a config", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runs_default_grid(tmp_path: Path) -> None:
    lb_path = tmp_path / "runs.csv"
    result = runner.invoke(
        app,
        ["benchmark", "--dataset", "wisconsin", "--leaderboard", str(lb_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert lb_path.is_file()
    df = pd.read_csv(lb_path)
    # All five protocols should have produced rows.
    assert len(df) == 5


def test_benchmark_subset_of_protocols(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dataset",
            "pima",
            "--models",
            "max_score,defensible",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.count("[max_score]") == 1
    assert result.stdout.count("[defensible]") == 1


def test_benchmark_unknown_dataset_rejected() -> None:
    result = runner.invoke(app, ["benchmark", "--dataset", "nonsense"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_audit_runs_leakage_check(sample_csv: Path) -> None:
    result = runner.invoke(
        app,
        ["audit", "--data", str(sample_csv), "--target", "target"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Leakage:" in result.stdout
    assert "duplicate_rows" in result.stdout


def test_audit_with_groups(sample_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "audit",
            "--data",
            str(sample_csv),
            "--target",
            "target",
            "--groups",
            "group",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Subgroup fairness" in result.stdout


def test_audit_writes_html_report(sample_csv: Path, tmp_path: Path) -> None:
    report = tmp_path / "audit.html"
    result = runner.invoke(
        app,
        [
            "audit",
            "--data",
            str(sample_csv),
            "--target",
            "target",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Audit findings" in text


def test_audit_missing_target_rejected(sample_csv: Path) -> None:
    result = runner.invoke(app, ["audit", "--data", str(sample_csv), "--target", "no_such_column"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_reports_metrics(tmp_path: Path) -> None:
    # Train on wisconsin (clean dataset without missing-value sentinels).
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("protocol: max_score\ndataset: wisconsin\nrandom_state: 0\n", encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    train_result = runner.invoke(app, ["train", "--config", str(cfg), "--output", str(model_path)])
    assert train_result.exit_code == 0, train_result.stdout

    import numpy as np

    from clinikit.datasets import load_wisconsin

    X, y = load_wisconsin(return_X_y=True)
    # load_wisconsin returns a DataFrame; coerce to numpy before wrapping
    # in a new frame with renamed columns (otherwise pandas reindexes by
    # column label and every cell becomes NaN).
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    df = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
    df["target"] = y_arr
    val_path = tmp_path / "val.csv"
    df.head(100).to_csv(val_path, index=False)

    result = runner.invoke(
        app,
        ["validate", "--model", str(model_path), "--data", str(val_path), "--target", "target"],
    )
    assert result.exit_code == 0, result.stdout
    assert "accuracy" in result.stdout
    assert "sensitivity" in result.stdout


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_renders_from_leaderboard(tmp_path: Path) -> None:
    lb_path = tmp_path / "runs.csv"
    runner.invoke(app, ["benchmark", "--dataset", "pima", "--leaderboard", str(lb_path)])
    out_path = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["report", "--leaderboard", str(lb_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.is_file()
    text = out_path.read_text(encoding="utf-8")
    assert "Leaderboard" in text
