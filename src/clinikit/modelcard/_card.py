"""Hugging Face Model Card generator.

Produces a YAML-frontmatter + Markdown file that follows the
Hugging Face Hub ``model-card`` schema. The output is a *template*:
it contains placeholders for the author to fill in (intended use,
limitations, ethical considerations), and a populated metrics table
derived from a run dict.

Reference: Mitchell et al., 2019, "Model Cards for Model Reporting".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "generate_modelcard",
    "render_modelcard",
]


def _yaml_block(
    *,
    model_name: str,
    license_id: str,
    language: str,
    tags: tuple[str, ...],
    task_name: str,
    task_type: str,
    dataset_name: str,
    metric_rows: list[tuple[str, str, float]],
) -> str:
    """Build the YAML frontmatter (no PyYAML — we hand-render to keep
    the dependency surface small and to enforce the HF order)."""
    lines: list[str] = ["---"]
    lines.append(f"language: {language}")
    lines.append(f"license: {license_id}")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("model-index:")
    lines.append(f"  - name: {model_name}")
    lines.append("    results:")
    lines.append("      - task:")
    lines.append(f"          name: {task_name}")
    lines.append(f"          type: {task_type}")
    lines.append("        dataset:")
    lines.append(f"          name: {dataset_name}")
    lines.append(f"          type: {dataset_name}")
    if metric_rows:
        lines.append("        metrics:")
        for name, m_type, value in metric_rows:
            lines.append(f"          - name: {name}")
            lines.append(f"            type: {m_type}")
            lines.append(f"            value: {value}")
    lines.append("---")
    return "\n".join(lines)


_METRIC_NAME_TO_TYPE: dict[str, str] = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": "specificity",
    "ppv": "precision",
    "npv": "npv",
    "f1_score": "f1",
    "f2_score": "f2",
    "mcc": "mcc",
    "balanced_accuracy": "balanced_accuracy",
    "brier_score": "brier",
    "roc_auc": "auc",
    "average_precision": "average_precision",
    "ece": "expected_calibration_error",
}


def _extract_metric_rows(run: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for name, m_type in _METRIC_NAME_TO_TYPE.items():
        if name in run and run[name] not in ("", None):
            try:
                rows.append((name, m_type, float(run[name])))
            except (TypeError, ValueError):
                continue
    return rows


def render_modelcard(
    run: Mapping[str, Any],
    *,
    model_name: str | None = None,
    license_id: str = "mit",
    language: str = "en",
    tags: tuple[str, ...] = ("tabular", "classification", "clinikit"),
    task_name: str = "Binary Classification",
    task_type: str = "classification",
    dataset_name: str | None = None,
    intended_use: str = "Research and development use only. Not a regulated product.",
    limitations: str = "See `CITATIONS.md` for the academic basis of the bundled methods.",
) -> str:
    """Render a model card as a single Markdown string.

    Parameters
    ----------
    run : mapping
        Run record (e.g. a leaderboard row). The function reads metric
        fields by name (``accuracy``, ``sensitivity``, ...) and turns
        them into the YAML metrics block.
    model_name : str, optional
        Defaults to ``run["model_class"]`` or ``"clinikit-model"``.
    license_id, language, tags : YAML frontmatter fields.
    task_name, task_type : YAML task-block fields.
    dataset_name : str, optional
        Defaults to ``run["dataset_name"]`` or ``"unknown"``.
    intended_use, limitations : free-text sections.

    Returns
    -------
    str
        The full model card (frontmatter + Markdown body).
    """
    mname = model_name or str(run.get("model_class") or "clinikit-model")
    dname = dataset_name or str(run.get("dataset_name") or "unknown")
    metric_rows = _extract_metric_rows(run)

    yaml = _yaml_block(
        model_name=mname,
        license_id=license_id,
        language=language,
        tags=tags,
        task_name=task_name,
        task_type=task_type,
        dataset_name=dname,
        metric_rows=metric_rows,
    )

    body_lines = [
        "",
        f"# Model card for `{mname}`",
        "",
        "## Model description",
        "",
        f"`{mname}` is a binary classifier produced with the **clinikit** "
        "toolkit. See the toolkit README for the integration framing — "
        "this card documents one specific trained instance.",
        "",
        "## Intended use",
        "",
        intended_use,
        "",
        "## Training data",
        "",
        f"- Dataset name: `{dname}`",
        f"- Dataset hash: `{run.get('dataset_hash', 'unknown')}`",
        f"- Train samples: `{run.get('n_samples_train', 'unknown')}`",
        f"- Features: `{run.get('n_features', 'unknown')}`",
        "",
        "## Evaluation",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for name, _type, value in metric_rows:
        body_lines.append(f"| `{name}` | {value:.4f} |")
    body_lines.extend(
        [
            "",
            "## Limitations",
            "",
            limitations,
            "",
            "## Reproducibility",
            "",
            f"- Threshold strategy: `{run.get('threshold_strategy', 'unknown')}`",
            f"- Threshold: `{run.get('threshold', 'unknown')}`",
            f"- CV strategy: `{run.get('cv_strategy', 'unknown')}`",
            f"- Random state: `{run.get('random_state', 'unknown')}`",
            f"- clinikit version: `{run.get('clinikit_version', 'unknown')}`",
            "",
        ]
    )

    return yaml + "\n" + "\n".join(body_lines)


def generate_modelcard(
    run: Mapping[str, Any],
    out_path: str | Path,
    **render_kwargs: Any,
) -> Path:
    """Write the rendered model card to ``out_path``.

    Parameters
    ----------
    run : mapping
        Run record.
    out_path : str or pathlib.Path
        Destination ``.md`` file. Parent directories are created on
        demand.
    **render_kwargs : forwarded to :func:`render_modelcard`.

    Returns
    -------
    pathlib.Path
        Resolved absolute path of the written file.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_modelcard(run, **render_kwargs), encoding="utf-8")
    return out
