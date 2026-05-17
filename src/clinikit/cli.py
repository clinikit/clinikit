"""clinikit.cli — Typer-based command-line interface.

Five subcommands tied to the corresponding library modules:

- ``clinikit train``      — run one protocol from a config file.
- ``clinikit benchmark``  — run a protocol grid against a bundled dataset.
- ``clinikit audit``      — run leakage and subgroup-fairness checks.
- ``clinikit validate``   — evaluate a saved model on a new dataset.
- ``clinikit report``     — render an HTML report from a leaderboard CSV.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from clinikit import __version__
from clinikit.audit import detect_leakage, subgroup_fairness
from clinikit.datasets import load_heart, load_pima, load_wisconsin
from clinikit.leaderboard import append_run, new_leaderboard
from clinikit.metrics import (
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    sensitivity,
    specificity,
)
from clinikit.protocols import (
    AuditProtocol,
    BaseProtocol,
    DefensibleProtocol,
    DeploymentProtocol,
    MaxScoreProtocol,
    OriginalOnlyProtocol,
)
from clinikit.report import render_report, render_report_from_leaderboard

app = typer.Typer(
    name="clinikit",
    help="A lightweight, sklearn-compatible toolkit for tabular ML.",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Registries (keep CLI surface declarative)
# ---------------------------------------------------------------------------


_PROTOCOL_REGISTRY: dict[str, type[BaseProtocol]] = {
    "defensible": DefensibleProtocol,
    "max_score": MaxScoreProtocol,
    "original_only": OriginalOnlyProtocol,
    "deployment": DeploymentProtocol,
    "audit": AuditProtocol,
}

_DATASET_REGISTRY: dict[str, Callable[..., Any]] = {
    "pima": load_pima,
    "wisconsin": load_wisconsin,
    "heart": load_heart,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        parsed = yaml.safe_load(text) or {}
        return dict(parsed)
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        return dict(parsed)
    raise typer.BadParameter(f"Config file must be .yaml, .yml, or .json; got {path.suffix!r}.")


def _load_bundled_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    loader = _DATASET_REGISTRY.get(name)
    if loader is None:
        raise typer.BadParameter(
            f"Unknown bundled dataset {name!r}. Available: {sorted(_DATASET_REGISTRY)}."
        )
    X, y = loader(return_X_y=True)
    return np.asarray(X), np.asarray(y)


def _load_csv_xy(path: Path, target: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise typer.BadParameter(
            f"target column {target!r} not in CSV columns {list(df.columns)!r}."
        )
    y = np.asarray(df[target].to_numpy())
    X = np.asarray(df.drop(columns=[target]).to_numpy(dtype=np.float64))
    return X, y


def _resolve_protocol(name: str) -> type[BaseProtocol]:
    cls = _PROTOCOL_REGISTRY.get(name)
    if cls is None:
        raise typer.BadParameter(
            f"Unknown protocol {name!r}. Available: {sorted(_PROTOCOL_REGISTRY)}."
        )
    return cls


def _ensure_leaderboard(path: Path) -> None:
    if not path.exists():
        new_leaderboard(path)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"clinikit {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the clinikit version and exit.",
        ),
    ] = None,
) -> None:
    """clinikit command-line entry point."""
    del version


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML / JSON config file.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to save the fitted model (joblib)."),
    ] = None,
    leaderboard: Annotated[
        Path | None,
        typer.Option("--leaderboard", "-l", help="Append the run to this leaderboard CSV."),
    ] = None,
) -> None:
    """Train a single protocol from a YAML or JSON config file.

    The config must specify ``protocol`` (one of defensible / max_score /
    original_only / deployment / audit) and ``dataset`` (one of pima /
    wisconsin / heart). Optional keys: ``random_state``, ``notes``.
    """
    cfg = _load_config(config)
    protocol_name = cfg.get("protocol", "defensible")
    dataset_name = cfg.get("dataset", "pima")
    random_state = int(cfg.get("random_state", 42))
    notes = str(cfg.get("notes", "") or "")

    protocol_cls = _resolve_protocol(protocol_name)
    X, y = _load_bundled_dataset(dataset_name)

    proto = protocol_cls(random_state=random_state)
    result = proto.run(X, y)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.fitted_pipeline, output)
        typer.echo(f"Saved model to {output}")

    if leaderboard is not None:
        _ensure_leaderboard(leaderboard)
        row = {
            **result.leaderboard_row,
            "run_id": _new_run_id(),
            "dataset_name": dataset_name,
            "clinikit_version": __version__,
            "notes": notes or proto.name,
        }
        append_run(leaderboard, row)
        typer.echo(f"Appended run to {leaderboard}")

    metrics = result.metrics
    typer.echo(
        f"[{proto.name}] dataset={dataset_name} accuracy={metrics['accuracy']:.4f} "
        f"sensitivity={metrics['sensitivity']:.4f} specificity={metrics['specificity']:.4f}"
    )


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    dataset: Annotated[
        str, typer.Option(help="Bundled dataset name (pima, wisconsin, heart).")
    ] = "pima",
    models: Annotated[str, typer.Option(help="Comma-separated protocol names or 'all'.")] = "all",
    leaderboard: Annotated[
        Path | None,
        typer.Option("--leaderboard", "-l", help="Output leaderboard CSV (created if missing)."),
    ] = None,
    random_state: Annotated[int, typer.Option(help="Forwarded to each protocol.")] = 42,
) -> None:
    """Run a grid of protocols against a bundled dataset."""
    X, y = _load_bundled_dataset(dataset)
    if models == "all":
        chosen = list(_PROTOCOL_REGISTRY.values())
    else:
        names = [m.strip() for m in models.split(",") if m.strip()]
        chosen = [_resolve_protocol(n) for n in names]

    if leaderboard is not None:
        _ensure_leaderboard(leaderboard)

    for proto_cls in chosen:
        proto = proto_cls(random_state=random_state)
        result = proto.run(X, y)
        if leaderboard is not None:
            row = {
                **result.leaderboard_row,
                "run_id": _new_run_id(),
                "dataset_name": dataset,
                "clinikit_version": __version__,
                "notes": proto.name,
            }
            append_run(leaderboard, row)
        metrics = result.metrics
        typer.echo(
            f"[{proto.name}] accuracy={metrics['accuracy']:.4f} "
            f"sensitivity={metrics['sensitivity']:.4f} "
            f"specificity={metrics['specificity']:.4f}"
        )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@app.command()
def audit(
    data: Annotated[Path, typer.Option("--data", "-d", help="Input CSV path.")],
    target: Annotated[str, typer.Option("--target", "-t", help="Label column name.")],
    groups: Annotated[
        str | None,
        typer.Option("--groups", "-g", help="Optional column for subgroup fairness analysis."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", "-r", help="Write an HTML audit report to this path."),
    ] = None,
    test_size: Annotated[float, typer.Option(help="Split fraction used for leakage check.")] = 0.2,
    random_state: Annotated[int, typer.Option(help="Seed for the leakage train/test split.")] = 42,
) -> None:
    """Run leakage and (optionally) subgroup-fairness diagnostics on a CSV."""
    df = pd.read_csv(data)
    if target not in df.columns:
        raise typer.BadParameter(f"target {target!r} not in CSV columns.")
    if groups is not None and groups not in df.columns:
        raise typer.BadParameter(f"groups column {groups!r} not in CSV columns.")

    drop = [target] + ([groups] if groups else [])
    X = df.drop(columns=drop).to_numpy(dtype=np.float64)
    y = df[target].to_numpy()
    g = df[groups].to_numpy() if groups else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    leakage = detect_leakage(X_train, X_test, y_train, y_test)
    typer.echo(f"Leakage: {leakage.summary}")
    typer.echo(f"  duplicate_rows = {leakage.duplicate_rows} ({leakage.duplicate_row_rate:.2%})")
    typer.echo(f"  feature alerts = {len(leakage.feature_distribution_alerts)}")
    typer.echo(f"  label drift    = {leakage.label_distribution_alert}")

    fairness_dict: dict[str, Any] = {}
    if g is not None:
        # Use trivial "predict positive class" as placeholder; the audit
        # is on data distribution, not on a specific model.
        majority = int(pd.Series(y).mode().iloc[0])
        pred = np.full_like(y, majority)
        sf = subgroup_fairness(y, pred, g)
        typer.echo("Subgroup fairness (majority-baseline classifier):")
        typer.echo(f"  accuracy_disparity    = {sf.accuracy_disparity:.4f}")
        typer.echo(f"  sensitivity_disparity = {sf.sensitivity_disparity:.4f}")
        typer.echo(f"  specificity_disparity = {sf.specificity_disparity:.4f}")
        fairness_dict = {
            "accuracy_disparity": round(sf.accuracy_disparity, 4),
            "sensitivity_disparity": round(sf.sensitivity_disparity, 4),
            "specificity_disparity": round(sf.specificity_disparity, 4),
            "n_groups": len(sf.per_group),
        }

    if report is not None:
        audit_section: dict[str, Any] = {
            "leakage_summary": leakage.summary,
            "duplicate_rows": leakage.duplicate_rows,
            "duplicate_row_rate": round(leakage.duplicate_row_rate, 4),
            "feature_distribution_alerts": len(leakage.feature_distribution_alerts),
            "label_distribution_alert": leakage.label_distribution_alert,
            **fairness_dict,
        }
        render_report(
            run={"dataset_name": data.name, "model_class": "(audit-only)"},
            out_path=report,
            title=f"Audit report — {data.name}",
            audit=audit_section,
        )
        typer.echo(f"Audit report written to {report}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    model: Annotated[Path, typer.Option("--model", "-m", help="Path to a saved joblib pipeline.")],
    data: Annotated[Path, typer.Option("--data", "-d", help="Validation CSV path.")],
    target: Annotated[str, typer.Option("--target", "-t", help="Label column name.")],
) -> None:
    """Evaluate a saved sklearn pipeline on a new CSV."""
    pipeline = joblib.load(model)
    X, y = _load_csv_xy(data, target)

    start = time.perf_counter()
    pred = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        classes = np.asarray(pipeline.classes_)
        pos_col = int(np.where(classes == classes[-1])[0][0])
        proba = pipeline.predict_proba(X)[:, pos_col]
    else:
        proba = None
    duration = time.perf_counter() - start

    typer.echo(f"Loaded {type(pipeline).__name__} from {model}")
    typer.echo(f"accuracy          = {accuracy_score(y, pred):.4f}")
    typer.echo(f"sensitivity       = {sensitivity(y, pred):.4f}")
    typer.echo(f"specificity       = {specificity(y, pred):.4f}")
    typer.echo(f"balanced_accuracy = {balanced_accuracy(y, pred):.4f}")
    if proba is not None:
        typer.echo(f"brier_score       = {brier_score(y, proba):.4f}")
        typer.echo(f"expected_cal_err  = {expected_calibration_error(y, proba):.4f}")
    typer.echo(f"duration_seconds  = {duration:.4f}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    leaderboard: Annotated[
        Path,
        typer.Option("--leaderboard", "-l", help="Leaderboard CSV path."),
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output HTML path.")],
    title: Annotated[str | None, typer.Option(help="Custom report title.")] = None,
) -> None:
    """Render an HTML report from a leaderboard CSV."""
    render_report_from_leaderboard(leaderboard, out, title=title)
    typer.echo(f"Report written to {out}")


if __name__ == "__main__":
    app()
