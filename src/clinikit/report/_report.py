"""HTML structured report generator.

Renders a single self-contained HTML report from a *run record*
(dict-like object — typically a leaderboard row, a protocol result,
or a merge of the two). Templates live alongside this module under
``templates/``; the default template inlines all CSS so the report
opens cleanly when emailed or zipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from clinikit import __version__ as _clinikit_version

__all__ = [
    "render_report",
    "render_report_from_leaderboard",
]


_NUMERIC_LEADERBOARD_COLUMNS: frozenset[str] = frozenset(
    {
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1_score",
        "f2_score",
        "mcc",
        "balanced_accuracy",
        "brier_score",
        "ece",
        "roc_auc",
        "average_precision",
        "accuracy_ci_lower",
        "accuracy_ci_upper",
        "post_cal_ece",
        "threshold",
        "class_balance",
        "duration_seconds",
        "test_size",
    }
)

_DATASET_KEYS = ("dataset_name", "dataset_hash", "n_samples_train", "n_samples_test", "n_features")
_MODEL_KEYS = (
    "model_class",
    "model_params_hash",
    "random_state",
    "cv_strategy",
    "threshold_strategy",
    "threshold",
)
_METRIC_KEYS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1_score",
    "f2_score",
    "mcc",
    "balanced_accuracy",
    "brier_score",
    "ece",
    "roc_auc",
    "average_precision",
)
_REPRO_KEYS = ("run_id", "created_at", "clinikit_version", "git_commit")


def _templates_dir() -> Path:
    return Path(str(files("clinikit.report").joinpath("templates")))


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_templates_dir())),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def _split_run(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Bucket a flat run mapping into the template's named sections."""

    def _slice(keys: Iterable[str]) -> dict[str, Any]:
        return {k: run[k] for k in keys if k in run and run[k] not in ("", None)}

    return {
        "metrics": {k: run[k] for k in _METRIC_KEYS if k in run and run[k] not in ("", None)},
        "dataset": _slice(_DATASET_KEYS),
        "model": _slice(_MODEL_KEYS),
        "reproducibility": _slice(_REPRO_KEYS),
        "audit": {},
    }


def render_report(
    run: Mapping[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
    template: str = "default.html.j2",
    leaderboard: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    audit: Mapping[str, Any] | None = None,
) -> Path:
    """Render a single-run HTML report and write it to ``out_path``.

    Parameters
    ----------
    run : mapping
        Run record. Recognised keys: ``run_id``, ``created_at``,
        ``model_class``, ``dataset_name``, ``threshold``, plus any of
        the standard metric column names. Unknown keys are ignored.
    out_path : str or pathlib.Path
        Destination ``.html`` file. Parents are created if missing.
    title : str, optional
        Page title; defaults to ``"{model_class} on {dataset_name}"``
        or ``"clinikit report"``.
    template : str, default ``"default.html.j2"``
        Filename inside ``clinikit/report/templates``.
    leaderboard : DataFrame or iterable of mappings, optional
        If supplied, an extra "Leaderboard" section is rendered with
        the full table.
    audit : mapping, optional
        Free-form key-value pairs surfaced in an "Audit findings"
        section (e.g. output of ``clinikit.audit.detect_leakage``).

    Returns
    -------
    pathlib.Path
        Resolved absolute path of the written file.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sections = _split_run(run)
    if audit:
        sections["audit"] = {str(k): v for k, v in audit.items()}

    if leaderboard is not None:
        if isinstance(leaderboard, pd.DataFrame):
            lb_records = leaderboard.to_dict(orient="records")
            lb_columns: list[str] = list(leaderboard.columns)
        else:
            lb_records = [dict(row) for row in leaderboard]
            lb_columns = list(lb_records[0].keys()) if lb_records else []
    else:
        lb_records = []
        lb_columns = []

    resolved_title = title or (
        f"{run['model_class']} on {run['dataset_name']}"
        if run.get("model_class") and run.get("dataset_name")
        else "clinikit report"
    )

    env = _jinja_env()
    tmpl = env.get_template(template)
    html = tmpl.render(
        title=resolved_title,
        clinikit_version=_clinikit_version,
        generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        metrics=sections["metrics"],
        dataset=sections["dataset"],
        model=sections["model"],
        reproducibility=sections["reproducibility"],
        audit=sections["audit"],
        leaderboard=lb_records,
        leaderboard_columns=lb_columns,
        numeric_columns=_NUMERIC_LEADERBOARD_COLUMNS,
    )
    out.write_text(html, encoding="utf-8")
    return out


def render_report_from_leaderboard(
    leaderboard_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
    template: str = "default.html.j2",
) -> Path:
    """Render a leaderboard-only report.

    Reads a CSV produced by ``clinikit.leaderboard.new_leaderboard``
    and renders it as a single HTML table without per-run sections.

    Parameters
    ----------
    leaderboard_path : str or pathlib.Path
    out_path : str or pathlib.Path
    title : str, optional
    template : str, default ``"default.html.j2"``

    Returns
    -------
    pathlib.Path
    """
    df = pd.read_csv(leaderboard_path)
    return render_report(
        run={},
        out_path=out_path,
        title=title or "clinikit leaderboard",
        template=template,
        leaderboard=df,
    )
