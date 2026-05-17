"""Run manifest — combines dataset hash, config, libraries, and environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clinikit.reproducibility._environment import (
    collect_environment,
    collect_library_versions,
    get_git_commit,
)
from clinikit.reproducibility._hashing import (
    hash_array,
    hash_config,
    hash_dataframe,
)

__all__ = [
    "Manifest",
    "build_manifest",
    "load_manifest",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True)
class Manifest:
    """A reproducibility manifest for a single experiment run.

    Attributes
    ----------
    schema_version : int
        Version of the manifest schema. Bumped when the structure
        changes in a non-backwards-compatible way.
    created_at : str
        ISO 8601 UTC timestamp, ``"YYYY-MM-DDTHH:MM:SS+00:00"``.
    dataset_hash : str
        ``"sha256:<hex>"`` digest of the input dataset.
    dataset_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the input dataset.
    config_hash : str or None
        Digest of the ``config`` mapping, or ``None`` if none supplied.
    config : dict
        The configuration dict as-supplied (after JSON
        round-tripping for safety).
    libraries : dict[str, str]
        Tracked package versions.
    environment : dict[str, str]
        Python, OS, and architecture info.
    git_commit : str or None
        Current commit of the working repository, when available.
    random_seed : int or None
        The seed used for stochastic components, if any.
    extra : dict
        Free-form additional fields. Persisted alongside the manifest
        and ignored by ``build_manifest``.
    """

    schema_version: int
    created_at: str
    dataset_hash: str
    dataset_shape: tuple[int, int]
    config_hash: str | None
    config: dict[str, Any]
    libraries: dict[str, str]
    environment: dict[str, str]
    git_commit: str | None = None
    random_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _dataset_shape(dataset: pd.DataFrame | np.ndarray) -> tuple[int, int]:
    if isinstance(dataset, pd.DataFrame):
        return (int(dataset.shape[0]), int(dataset.shape[1]))
    arr = np.asarray(dataset)
    if arr.ndim == 1:
        return (int(arr.shape[0]), 1)
    if arr.ndim == 2:
        return (int(arr.shape[0]), int(arr.shape[1]))
    raise ValueError(f"dataset must be 1- or 2-D; got ndim={arr.ndim} (shape={arr.shape!r}).")


def _now_iso_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def build_manifest(
    *,
    dataset: pd.DataFrame | np.ndarray,
    config: dict[str, Any] | None = None,
    random_seed: int | None = None,
    extra: dict[str, Any] | None = None,
    tracked_packages: list[str] | None = None,
    repo_path: str | Path | None = None,
) -> Manifest:
    """Assemble a :class:`Manifest` from a dataset, config, and environment.

    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        The training (or evaluation) data the run consumes.
    config : dict, optional
        Run configuration. Values must be JSON-serialisable.
    random_seed : int, optional
        Seed used by the run's stochastic components.
    extra : dict, optional
        Free-form additional metadata recorded under ``manifest.extra``.
    tracked_packages : list[str], optional
        Packages whose versions to capture. Defaults to
        :data:`DEFAULT_TRACKED_PACKAGES`.
    repo_path : str or pathlib.Path, optional
        Path to a git repository whose HEAD commit should be captured.
        Defaults to the current working directory.

    Returns
    -------
    Manifest

    Examples
    --------
    >>> import pandas as pd
    >>> from clinikit.reproducibility import build_manifest
    >>> df = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
    >>> m = build_manifest(dataset=df, config={"model": "rule_augmented"}, random_seed=0)
    >>> m.dataset_shape
    (2, 2)
    >>> m.dataset_hash.startswith("sha256:")
    True
    """
    ds_hash = hash_dataframe(dataset) if isinstance(dataset, pd.DataFrame) else hash_array(dataset)

    cfg = dict(config) if config is not None else {}
    cfg_hash = hash_config(cfg) if config is not None else None

    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=_now_iso_utc(),
        dataset_hash=ds_hash,
        dataset_shape=_dataset_shape(dataset),
        config_hash=cfg_hash,
        config=cfg,
        libraries=collect_library_versions(tracked_packages),
        environment=collect_environment(),
        git_commit=get_git_commit(repo_path),
        random_seed=random_seed,
        extra=dict(extra) if extra is not None else {},
    )


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Serialise a manifest to JSON.

    The output is pretty-printed with sorted top-level keys for human
    reviewability and easy diffing.

    Parameters
    ----------
    manifest : Manifest
    path : str or pathlib.Path
        Destination file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        The resolved absolute path that was written.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    # Tuples become lists in JSON; do the conversion explicitly so the
    # round-trip is well-defined.
    payload["dataset_shape"] = list(payload["dataset_shape"])
    out.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest previously written by :func:`write_manifest`.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    Manifest

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If the schema version is newer than the package supports.
    """
    in_path = Path(path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"load_manifest: not a regular file: {in_path!s}")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    version = int(data.get("schema_version", 0))
    if version > MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema version {version} is newer than this "
            f"clinikit installation supports ({MANIFEST_SCHEMA_VERSION}). "
            "Upgrade clinikit or re-create the manifest."
        )

    data["dataset_shape"] = tuple(data["dataset_shape"])
    return Manifest(**data)
