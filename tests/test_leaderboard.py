"""Tests for clinikit.leaderboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinikit.leaderboard import (
    LEADERBOARD_COLUMNS,
    append_run,
    new_leaderboard,
    read_leaderboard,
)


def test_schema_is_38_columns() -> None:
    assert len(LEADERBOARD_COLUMNS) == 38
    # No duplicates.
    assert len(set(LEADERBOARD_COLUMNS)) == 38


# ---------------------------------------------------------------------------
# new_leaderboard
# ---------------------------------------------------------------------------


def test_new_leaderboard_creates_empty_csv(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    out = new_leaderboard(p)
    assert out == p.resolve()
    df = pd.read_csv(out)
    assert list(df.columns) == list(LEADERBOARD_COLUMNS)
    assert len(df) == 0


def test_new_leaderboard_refuses_to_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    with pytest.raises(FileExistsError, match="overwrite"):
        new_leaderboard(p)


def test_new_leaderboard_overwrite_flag(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    out = new_leaderboard(p, overwrite=True)
    assert out.is_file()


def test_new_leaderboard_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "runs.csv"
    out = new_leaderboard(p)
    assert out.is_file()


# ---------------------------------------------------------------------------
# append_run
# ---------------------------------------------------------------------------


def test_append_run_basic(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    append_run(
        p,
        {
            "run_id": "demo-1",
            "model_class": "LogisticRegression",
            "accuracy": 0.81,
            "sensitivity": 0.77,
        },
    )
    df = read_leaderboard(p)
    assert len(df) == 1
    assert df.iloc[0]["model_class"] == "LogisticRegression"
    assert df.iloc[0]["accuracy"] == pytest.approx(0.81)


def test_append_run_fills_defaults(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    append_run(p, {"run_id": "demo-2"})
    df = read_leaderboard(p)
    assert df.iloc[0]["created_at"]  # auto-set
    assert df.iloc[0]["leaderboard_schema_version"] == 1


def test_append_run_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    with pytest.raises(KeyError, match="Unknown leaderboard columns"):
        append_run(p, {"run_id": "x", "nonsense_column": 42})


def test_append_run_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        append_run(tmp_path / "no_such.csv", {"run_id": "x"})


def test_append_multiple_rows_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    for i in range(3):
        append_run(p, {"run_id": f"r{i}", "accuracy": 0.7 + 0.05 * i})
    df = read_leaderboard(p)
    assert list(df["run_id"]) == ["r0", "r1", "r2"]


# ---------------------------------------------------------------------------
# read_leaderboard
# ---------------------------------------------------------------------------


def test_read_leaderboard_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_leaderboard(tmp_path / "missing.csv")


def test_read_leaderboard_schema_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        read_leaderboard(p)


def test_round_trip_preserves_column_order(tmp_path: Path) -> None:
    p = tmp_path / "runs.csv"
    new_leaderboard(p)
    append_run(p, {"run_id": "x", "accuracy": 0.5})
    df = read_leaderboard(p)
    assert list(df.columns) == list(LEADERBOARD_COLUMNS)
