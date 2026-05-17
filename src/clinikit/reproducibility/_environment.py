"""Collect Python, OS, library, and git-commit information for a manifest."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "DEFAULT_TRACKED_PACKAGES",
    "collect_environment",
    "collect_library_versions",
    "get_git_commit",
]

DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "clinikit",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced-learn",
    "matplotlib",
    "joblib",
    "jinja2",
    "typer",
)


def _safe_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_library_versions(
    packages: Iterable[str] | None = None,
) -> dict[str, str]:
    """Snapshot installed versions of the tracked Python packages.

    Parameters
    ----------
    packages : iterable of str, optional
        Distribution names to look up via :mod:`importlib.metadata`.
        Defaults to :data:`DEFAULT_TRACKED_PACKAGES`.

    Returns
    -------
    dict[str, str]
        Mapping of distribution name to installed version, or to the
        sentinel ``"not-installed"`` if the package is not present.

    Examples
    --------
    >>> from clinikit.reproducibility import collect_library_versions
    >>> versions = collect_library_versions(["clinikit"])
    >>> versions["clinikit"]
    '0.1.0'
    """
    pkgs = list(packages) if packages is not None else list(DEFAULT_TRACKED_PACKAGES)
    return {p: _safe_version(p) for p in pkgs}


def collect_environment() -> dict[str, str]:
    """Snapshot the Python interpreter, OS, and architecture.

    Returns
    -------
    dict[str, str]
        Keys: ``"python"``, ``"implementation"``, ``"platform"``,
        ``"machine"``, ``"processor"``.

    Examples
    --------
    >>> from clinikit.reproducibility import collect_environment
    >>> env = collect_environment()
    >>> set(env) >= {"python", "platform"}
    True
    """
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_git_commit(repo: str | Path | None = None) -> str | None:
    """Return the current git commit hash of ``repo``, or ``None``.

    Quietly returns ``None`` when git is not installed, the directory
    is not a git repository, or the command fails for any reason.

    Parameters
    ----------
    repo : str, pathlib.Path, or None, default None
        Directory to inspect. Defaults to the current working
        directory.

    Returns
    -------
    str or None
        40-character commit SHA, or ``None`` when unavailable.

    Notes
    -----
    The call uses ``subprocess.run`` with a fixed argv list (no
    ``shell=True``) and a 5-second timeout. It never raises — failures
    return ``None``.
    """
    repo_path = Path(repo).expanduser().resolve() if repo is not None else Path.cwd()
    if not (repo_path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        return None
    return sha
