"""Audit-trail manifest templates (research documentation only).

The outputs of this module are *templates* — JSON / Markdown
artifacts whose first lines explicitly state that they are intended
for research documentation and not for regulated decision-making.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "GOVERNANCE_DISCLAIMER",
    "MANIFEST_REQUIRED_FIELDS",
    "generate_manifest",
    "list_required_fields",
    "render_manifest_markdown",
]

GOVERNANCE_DISCLAIMER = "Templates are for research documentation only."

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "created_at",
    "clinikit_version",
    "dataset_name",
    "dataset_hash",
    "model_class",
    "model_params_hash",
    "cv_strategy",
    "threshold_strategy",
    "threshold",
    "accuracy",
    "sensitivity",
    "specificity",
    "notes",
)


def list_required_fields() -> tuple[str, ...]:
    """Return the field names a complete manifest is expected to carry."""
    return MANIFEST_REQUIRED_FIELDS


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def generate_manifest(
    run: Mapping[str, Any],
    out_path: str | Path,
    *,
    template: str = "json",
) -> Path:
    """Write a research-documentation manifest for a single run.

    Parameters
    ----------
    run : mapping
        Run record (e.g. a leaderboard row, the manifest from
        :mod:`clinikit.reproducibility`, or both merged).
    out_path : str or pathlib.Path
        Destination file. Parent directories are created if missing.
    template : {"json", "markdown"}, default "json"
        Output format. ``"json"`` produces a sorted-key, two-space
        indented file with a top-level ``disclaimer`` key. ``"markdown"``
        produces a human-readable summary with the disclaimer in a
        block-quote at the top.

    Returns
    -------
    pathlib.Path
        Resolved absolute path of the written file.

    Raises
    ------
    ValueError
        If ``template`` is unknown.
    """
    if template not in ("json", "markdown"):
        raise ValueError(f"template must be 'json' or 'markdown'; got {template!r}.")
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "disclaimer": GOVERNANCE_DISCLAIMER,
        "manifest_written_at": _now_iso(),
        "run": dict(run),
    }

    if template == "json":
        out.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        out.write_text(render_manifest_markdown(payload), encoding="utf-8")
    return out


def render_manifest_markdown(payload: Mapping[str, Any]) -> str:
    """Render the manifest payload as a Markdown document."""
    run = payload.get("run", {})
    written_at = payload.get("manifest_written_at", _now_iso())

    lines: list[str] = [
        "# clinikit run manifest",
        "",
        f"> {GOVERNANCE_DISCLAIMER}",
        "",
        f"Manifest written at `{written_at}`.",
        "",
        "## Run record",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in sorted(run):
        value = run[key]
        cell = value.replace("|", "\\|") if isinstance(value, str) else repr(value)
        lines.append(f"| `{key}` | {cell} |")
    lines.append("")
    return "\n".join(lines)
