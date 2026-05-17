"""Tests for clinikit.governance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinikit.governance import (
    GOVERNANCE_DISCLAIMER,
    MANIFEST_REQUIRED_FIELDS,
    generate_manifest,
    list_required_fields,
    render_manifest_markdown,
)


def test_disclaimer_text() -> None:
    assert "research documentation only" in GOVERNANCE_DISCLAIMER.lower()


def test_list_required_fields_returns_tuple() -> None:
    fields = list_required_fields()
    assert isinstance(fields, tuple)
    assert fields == MANIFEST_REQUIRED_FIELDS
    assert "run_id" in fields


# ---------------------------------------------------------------------------
# generate_manifest — JSON
# ---------------------------------------------------------------------------


def test_generate_manifest_json(tmp_path: Path) -> None:
    out = generate_manifest(
        {"run_id": "demo", "accuracy": 0.81},
        tmp_path / "manifest.json",
        template="json",
    )
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["disclaimer"] == GOVERNANCE_DISCLAIMER
    assert data["run"]["run_id"] == "demo"
    assert data["run"]["accuracy"] == 0.81
    assert "manifest_written_at" in data


def test_generate_manifest_creates_parents(tmp_path: Path) -> None:
    out = generate_manifest(
        {"run_id": "demo"},
        tmp_path / "deep" / "nested" / "manifest.json",
    )
    assert out.is_file()


def test_generate_manifest_invalid_template(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="template"):
        generate_manifest({"run_id": "demo"}, tmp_path / "out.json", template="bad")


def test_generate_manifest_sorts_json_keys(tmp_path: Path) -> None:
    out = generate_manifest({"zeta": 1, "alpha": 2}, tmp_path / "manifest.json")
    text = out.read_text(encoding="utf-8")
    assert text.index('"disclaimer"') < text.index('"manifest_written_at"')


# ---------------------------------------------------------------------------
# generate_manifest — Markdown
# ---------------------------------------------------------------------------


def test_generate_manifest_markdown(tmp_path: Path) -> None:
    out = generate_manifest(
        {"run_id": "demo", "accuracy": 0.81},
        tmp_path / "manifest.md",
        template="markdown",
    )
    text = out.read_text(encoding="utf-8")
    assert "clinikit run manifest" in text
    assert GOVERNANCE_DISCLAIMER in text
    assert "`run_id`" in text


def test_render_manifest_markdown_includes_disclaimer() -> None:
    md = render_manifest_markdown({"run": {"run_id": "demo"}})
    assert GOVERNANCE_DISCLAIMER in md
    assert "`run_id`" in md


def test_render_manifest_markdown_escapes_pipe() -> None:
    md = render_manifest_markdown({"run": {"notes": "a | b"}})
    assert r"a \| b" in md
