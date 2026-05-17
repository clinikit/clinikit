"""Experiment-tracking CSV with a fixed 38-column schema.

The leaderboard records one row per experiment run. The fixed schema
makes it straightforward to diff experiments, filter by model class,
and feed the table into a structured HTML report.

Three public helpers:

- :func:`new_leaderboard` — create an empty CSV with the schema header.
- :func:`append_run`      — append a single run row, validating keys.
- :func:`read_leaderboard`— read the CSV back as a pandas ``DataFrame``.

The 38 columns are exposed as the :data:`LEADERBOARD_COLUMNS` tuple
and are documented one-by-one inline in this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "LEADERBOARD_COLUMNS",
    "append_run",
    "new_leaderboard",
    "read_leaderboard",
]

# The fixed 38-column schema. Order matters — it defines the CSV
# column order and the field order in every dict returned by the
# helpers. Columns are grouped by concern; the comments in each
# block document the intended semantics.
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    # Run identity (5)
    "run_id",
    "created_at",
    "clinikit_version",
    "git_commit",
    "notes",
    # Dataset (6)
    "dataset_name",
    "dataset_hash",
    "n_samples_train",
    "n_samples_test",
    "n_features",
    "class_balance",
    # Model + split (7)
    "model_class",
    "model_params_hash",
    "random_state",
    "cv_strategy",
    "n_folds",
    "test_size",
    "duration_seconds",
    # Threshold + calibrator (4)
    "threshold_strategy",
    "threshold",
    "calibrator",
    "post_cal_ece",
    # Point metrics (8)
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1_score",
    "f2_score",
    "mcc",
    # Probability metrics (4)
    "balanced_accuracy",
    "brier_score",
    "ece",
    "roc_auc",
    # Bootstrap CI (3)
    "average_precision",
    "accuracy_ci_lower",
    "accuracy_ci_upper",
    # Schema version (1)
    "leaderboard_schema_version",
)

assert len(LEADERBOARD_COLUMNS) == 38, "LEADERBOARD_COLUMNS must contain 38 entries."

LEADERBOARD_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def new_leaderboard(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an empty leaderboard CSV with the schema header.

    Parameters
    ----------
    path : str or pathlib.Path
    overwrite : bool, default False
        Required to overwrite an existing file. Raises
        :class:`FileExistsError` otherwise.

    Returns
    -------
    pathlib.Path
        The resolved absolute path of the new file.
    """
    out = Path(path).expanduser().resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Leaderboard already exists at {out!s}; pass overwrite=True to replace."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(columns=list(LEADERBOARD_COLUMNS))
    empty.to_csv(out, index=False, lineterminator="\n")
    return out


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate keys and fill defaults for a new row."""
    extras = [k for k in row if k not in LEADERBOARD_COLUMNS]
    if extras:
        raise KeyError(
            f"Unknown leaderboard columns: {extras!r}. Allowed columns: {LEADERBOARD_COLUMNS!r}."
        )
    out = dict(row)
    out.setdefault("created_at", _utc_now_iso())
    out.setdefault("leaderboard_schema_version", LEADERBOARD_SCHEMA_VERSION)
    return out


def append_run(path: str | Path, row: dict[str, Any]) -> Path:
    """Append one experiment-run row to the leaderboard CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing leaderboard CSV created by :func:`new_leaderboard`.
    row : dict
        Mapping from a subset of :data:`LEADERBOARD_COLUMNS` to
        values. Missing columns are stored as empty cells; extra keys
        raise :class:`KeyError`.

    Returns
    -------
    pathlib.Path
        Resolved absolute path of the leaderboard.

    Examples
    --------
    >>> from clinikit.leaderboard import new_leaderboard, append_run, read_leaderboard
    >>> import tempfile, pathlib
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "runs.csv"
    >>> _ = new_leaderboard(tmp)
    >>> _ = append_run(tmp, {"run_id": "demo-1", "model_class": "LogReg", "accuracy": 0.81})
    >>> read_leaderboard(tmp).iloc[0]["model_class"]
    'LogReg'
    """
    out = Path(path).expanduser().resolve()
    if not out.is_file():
        raise FileNotFoundError(f"Leaderboard not found at {out!s}. Run new_leaderboard first.")

    coerced = _coerce_row(row)
    full = {col: coerced.get(col, "") for col in LEADERBOARD_COLUMNS}
    existing = pd.read_csv(out)
    # Guard against a stale CSV with the wrong header.
    if list(existing.columns) != list(LEADERBOARD_COLUMNS):
        raise ValueError(
            f"Leaderboard at {out!s} has a header that does not match "
            f"the current schema. Re-create with new_leaderboard."
        )
    new = pd.concat(
        [existing, pd.DataFrame([full], columns=list(LEADERBOARD_COLUMNS))],
        ignore_index=True,
    )
    new.to_csv(out, index=False, lineterminator="\n")
    return out


def read_leaderboard(path: str | Path) -> pd.DataFrame:
    """Read a leaderboard CSV into a pandas ``DataFrame``.

    The returned frame has the canonical column order whether or not
    the on-disk file is sorted that way.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    pandas.DataFrame
    """
    in_path = Path(path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Leaderboard not found at {in_path!s}.")
    df = pd.read_csv(in_path)
    extras = [c for c in df.columns if c not in LEADERBOARD_COLUMNS]
    missing = [c for c in LEADERBOARD_COLUMNS if c not in df.columns]
    if extras or missing:
        raise ValueError(
            f"Leaderboard schema mismatch at {in_path!s}. "
            f"Extra columns: {extras!r}; missing: {missing!r}."
        )
    return df[list(LEADERBOARD_COLUMNS)]
