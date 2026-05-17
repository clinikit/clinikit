"""Tests for clinikit.modelcard."""

from __future__ import annotations

from pathlib import Path

from clinikit.modelcard import generate_modelcard, render_modelcard

_DEMO_RUN = {
    "run_id": "demo-1",
    "model_class": "LogisticRegression",
    "dataset_name": "pima",
    "dataset_hash": "sha256:abc",
    "n_samples_train": 614,
    "n_features": 8,
    "threshold_strategy": "accuracy_max",
    "threshold": 0.48,
    "cv_strategy": "holdout(test_size=0.2)",
    "random_state": 42,
    "clinikit_version": "0.1.0",
    "accuracy": 0.812,
    "sensitivity": 0.764,
    "specificity": 0.835,
    "ppv": 0.701,
    "npv": 0.872,
    "f1_score": 0.731,
    "f2_score": 0.749,
    "mcc": 0.582,
    "balanced_accuracy": 0.799,
    "brier_score": 0.146,
    "roc_auc": 0.874,
    "ece": 0.041,
}


# ---------------------------------------------------------------------------
# render_modelcard
# ---------------------------------------------------------------------------


def test_render_modelcard_includes_yaml_frontmatter() -> None:
    card = render_modelcard(_DEMO_RUN)
    assert card.startswith("---\n")
    assert "\n---\n" in card  # frontmatter close
    assert "license: mit" in card
    assert "language: en" in card


def test_render_modelcard_lists_tags() -> None:
    card = render_modelcard(_DEMO_RUN, tags=("tabular", "binary"))
    assert "  - tabular" in card
    assert "  - binary" in card


def test_render_modelcard_includes_model_name() -> None:
    card = render_modelcard(_DEMO_RUN)
    assert "name: LogisticRegression" in card
    assert "Model card for `LogisticRegression`" in card


def test_render_modelcard_metric_table() -> None:
    card = render_modelcard(_DEMO_RUN)
    assert "| `accuracy` | 0.8120 |" in card
    assert "| `sensitivity` | 0.7640 |" in card
    # Metric YAML block too.
    assert "metrics:" in card
    assert "          - name: accuracy" in card


def test_render_modelcard_uses_dataset_name_fallback() -> None:
    card = render_modelcard({"model_class": "Foo"})
    assert "name: unknown" in card  # dataset_name fallback


def test_render_modelcard_includes_intended_use_and_limitations() -> None:
    card = render_modelcard(_DEMO_RUN)
    assert "Research and development use only" in card
    assert "CITATIONS.md" in card


def test_render_modelcard_skips_missing_metrics() -> None:
    run = {"model_class": "Foo", "accuracy": 0.5}  # only one metric
    card = render_modelcard(run)
    assert "| `accuracy` | 0.5000 |" in card
    # No accidental missing-metric rows.
    assert "| `sensitivity` |" not in card


# ---------------------------------------------------------------------------
# generate_modelcard
# ---------------------------------------------------------------------------


def test_generate_modelcard_writes_file(tmp_path: Path) -> None:
    out = generate_modelcard(_DEMO_RUN, tmp_path / "card.md")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Model card for `LogisticRegression`" in text


def test_generate_modelcard_creates_parents(tmp_path: Path) -> None:
    out = generate_modelcard(_DEMO_RUN, tmp_path / "deep" / "nested" / "card.md")
    assert out.is_file()
