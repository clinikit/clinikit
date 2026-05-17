"""clinikit.cli — Typer-based command-line interface.

The CLI exposes five subcommands:

- ``clinikit train``      — train one model from a config file
- ``clinikit benchmark``  — run a model grid against bundled datasets
- ``clinikit audit``      — run leakage / fairness / documentation checks
- ``clinikit validate``   — evaluate a saved model on a new dataset
- ``clinikit report``     — render an HTML report from a leaderboard CSV

The subcommand implementations live in their respective modules
(``clinikit.protocols``, ``clinikit.audit``, ``clinikit.report``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from clinikit import __version__

app = typer.Typer(
    name="clinikit",
    help="A lightweight, sklearn-compatible toolkit for tabular ML.",
    no_args_is_help=True,
    add_completion=False,
)


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


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to training config.")],
) -> None:
    """Train a single model from a YAML/JSON configuration file."""
    typer.echo(f"[train] not yet implemented; config={config}")
    raise typer.Exit(code=2)


@app.command()
def benchmark(
    dataset: Annotated[str, typer.Option(help="Bundled dataset name.")] = "pima",
    models: Annotated[str, typer.Option(help="Comma-separated model names or 'all'.")] = "all",
) -> None:
    """Run a benchmark grid against a bundled dataset."""
    typer.echo(f"[benchmark] not yet implemented; dataset={dataset}, models={models}")
    raise typer.Exit(code=2)


@app.command()
def audit(
    data: Annotated[Path, typer.Option("--data", "-d", help="Input CSV path.")],
    report: Annotated[
        Path | None,
        typer.Option("--report", "-r", help="Output HTML report path."),
    ] = None,
) -> None:
    """Run leakage / fairness / documentation checks on a dataset."""
    typer.echo(f"[audit] not yet implemented; data={data}, report={report}")
    raise typer.Exit(code=2)


@app.command()
def validate(
    model: Annotated[Path, typer.Option("--model", "-m", help="Path to a saved joblib model.")],
    data: Annotated[Path, typer.Option("--data", "-d", help="Validation CSV path.")],
) -> None:
    """Evaluate a saved model on a new dataset."""
    typer.echo(f"[validate] not yet implemented; model={model}, data={data}")
    raise typer.Exit(code=2)


@app.command()
def report(
    leaderboard: Annotated[
        Path,
        typer.Option("--leaderboard", "-l", help="Leaderboard CSV path."),
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output HTML path.")],
) -> None:
    """Render an HTML report from a leaderboard CSV."""
    typer.echo(f"[report] not yet implemented; leaderboard={leaderboard}, out={out}")
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
