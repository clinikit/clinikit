"""Tests for clinikit.report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinikit.leaderboard import LEADERBOARD_COLUMNS, append_run, new_leaderboard
from clinikit.report import render_report, render_report_from_leaderboard

_DEMO_RUN = {
    "run_id": "demo-1",
    "created_at": "2026-05-17T12:00:00+00:00",
    "clinikit_version": "0.1.0",
    "git_commit": "deadbeef",
    "model_class": "LogisticRegression",
    "dataset_name": "pima",
    "dataset_hash": "sha256:abc",
    "n_samples_train": 614,
    "n_samples_test": 154,
    "n_features": 8,
    "threshold_strategy": "accuracy_max",
    "threshold": 0.48,
    "cv_strategy": "holdout(test_size=0.2)",
    "random_state": 42,
    "accuracy": 0.812,
    "sensitivity": 0.764,
    "specificity": 0.835,
    "ppv": 0.701,
    "npv": 0.872,
    "f2_score": 0.749,
    "mcc": 0.582,
    "balanced_accuracy": 0.799,
    "brier_score": 0.146,
    "roc_auc": 0.874,
    "ece": 0.041,
}


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_report_writes_file(tmp_path: Path) -> None:
    out = render_report(_DEMO_RUN, tmp_path / "report.html")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "LogisticRegression on pima" in text


def test_render_report_includes_metrics(tmp_path: Path) -> None:
    out = render_report(_DEMO_RUN, tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    # Headline metrics card.
    assert "accuracy" in text
    assert "0.8120" in text
    assert "sensitivity" in text


def test_render_report_includes_disclaimer(tmp_path: Path) -> None:
    out = render_report(_DEMO_RUN, tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    assert "research-and-development summary" in text.lower()


def test_render_report_custom_title(tmp_path: Path) -> None:
    out = render_report(_DEMO_RUN, tmp_path / "report.html", title="Q3 evaluation")
    text = out.read_text(encoding="utf-8")
    assert "Q3 evaluation" in text


def test_render_report_default_title_fallback(tmp_path: Path) -> None:
    out = render_report({}, tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    assert "clinikit report" in text


def test_render_report_with_audit_section(tmp_path: Path) -> None:
    out = render_report(
        _DEMO_RUN,
        tmp_path / "report.html",
        audit={"duplicate_rows": 0, "summary": "no leakage signals detected"},
    )
    text = out.read_text(encoding="utf-8")
    assert "Audit findings" in text
    assert "no leakage" in text


def test_render_report_creates_parent_dirs(tmp_path: Path) -> None:
    out = render_report(_DEMO_RUN, tmp_path / "deep" / "nested" / "report.html")
    assert out.is_file()


# ---------------------------------------------------------------------------
# Leaderboard section
# ---------------------------------------------------------------------------


def test_render_report_with_leaderboard_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame([_DEMO_RUN, {**_DEMO_RUN, "run_id": "demo-2", "accuracy": 0.85}])
    out = render_report(_DEMO_RUN, tmp_path / "r.html", leaderboard=df)
    text = out.read_text(encoding="utf-8")
    assert "Leaderboard" in text
    assert "demo-2" in text


def test_render_report_from_leaderboard_csv(tmp_path: Path) -> None:
    lb_path = tmp_path / "runs.csv"
    new_leaderboard(lb_path)
    append_run(lb_path, {"run_id": "r1", "model_class": "LogReg", "accuracy": 0.81})
    append_run(lb_path, {"run_id": "r2", "model_class": "LogReg", "accuracy": 0.83})

    out = render_report_from_leaderboard(lb_path, tmp_path / "r.html")
    text = out.read_text(encoding="utf-8")
    assert "Leaderboard" in text
    assert "r1" in text
    assert "r2" in text
    # Every leaderboard column should appear as a header at least.
    for col in LEADERBOARD_COLUMNS:
        assert col in text
