"""Tests for clinikit.reproducibility."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinikit.reproducibility import (
    DEFAULT_TRACKED_PACKAGES,
    Manifest,
    build_manifest,
    collect_environment,
    collect_library_versions,
    get_git_commit,
    hash_array,
    hash_config,
    hash_dataframe,
    hash_file,
    load_manifest,
    write_manifest,
)

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_dataframe_returns_prefixed_sha256() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert SHA256_RE.match(hash_dataframe(df))


def test_hash_dataframe_is_deterministic() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert hash_dataframe(df) == hash_dataframe(df.copy())


def test_hash_dataframe_invariant_to_index_reset() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    assert hash_dataframe(df) == hash_dataframe(df.reset_index(drop=True))


def test_hash_dataframe_changes_when_value_changes() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_changes_when_column_order_changes() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = df1[["b", "a"]]
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        hash_dataframe([1, 2, 3])  # type: ignore[arg-type]


def test_hash_array_is_deterministic(rng: np.random.Generator) -> None:
    arr = rng.standard_normal((10, 4))
    assert hash_array(arr) == hash_array(arr.copy())
    assert SHA256_RE.match(hash_array(arr))


def test_hash_array_differs_when_shape_differs() -> None:
    arr1 = np.zeros((4, 4))
    arr2 = np.zeros((2, 8))  # same number of bytes, different shape
    assert hash_array(arr1) != hash_array(arr2)


def test_hash_file_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"clinikit test payload\n")
    assert hash_file(p) == hash_file(p)
    assert SHA256_RE.match(hash_file(p))


def test_hash_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "does_not_exist.bin")


def test_hash_config_is_key_order_invariant() -> None:
    c1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    c2 = {"c": {"d": 4}, "b": [2, 3], "a": 1}
    assert hash_config(c1) == hash_config(c2)
    assert SHA256_RE.match(hash_config(c1))


def test_hash_config_rejects_non_json() -> None:
    with pytest.raises(TypeError):
        hash_config({"bad": np.array([1, 2, 3])})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_collect_library_versions_includes_clinikit() -> None:
    versions = collect_library_versions()
    assert "clinikit" in versions
    assert re.match(r"^\d+\.\d+\.\d+$", versions["clinikit"])


def test_collect_library_versions_marks_missing_as_not_installed() -> None:
    versions = collect_library_versions(["this-package-does-not-exist-xyz"])
    assert versions["this-package-does-not-exist-xyz"] == "not-installed"


def test_default_tracked_packages_contains_core() -> None:
    assert "numpy" in DEFAULT_TRACKED_PACKAGES
    assert "pandas" in DEFAULT_TRACKED_PACKAGES
    assert "scikit-learn" in DEFAULT_TRACKED_PACKAGES
    assert "clinikit" in DEFAULT_TRACKED_PACKAGES


def test_collect_environment_has_expected_keys() -> None:
    env = collect_environment()
    assert {"python", "implementation", "platform", "machine", "processor"} <= env.keys()
    assert re.match(r"^\d+\.\d+\.\d+$", env["python"])


def test_get_git_commit_returns_none_in_non_repo(tmp_path: Path) -> None:
    # tmp_path is never a git repo, so this must return None without
    # raising.
    assert get_git_commit(tmp_path) is None


def test_get_git_commit_returns_sha_in_real_repo(tmp_path: Path) -> None:
    git_bin = shutil.which("git")
    if git_bin is None:
        pytest.skip("git executable not available on PATH")
    repo = tmp_path / "demo-repo"
    repo.mkdir()
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp_path), "PATH": os.environ["PATH"]}
    cmds = [
        [git_bin, "init", "-q", "--initial-branch=main", str(repo)],
        [git_bin, "-C", str(repo), "config", "user.email", "ci@example.com"],
        [git_bin, "-C", str(repo), "config", "user.name", "ci"],
        [git_bin, "-C", str(repo), "commit", "--allow-empty", "-m", "init", "-q"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, env=env, check=False, timeout=10)
        if result.returncode != 0:
            pytest.skip(f"git bootstrap failed: {cmd!r}: {result.stderr.decode(errors='replace')}")
    sha = get_git_commit(repo)
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": rng.standard_normal(20),
            "y": rng.integers(0, 2, size=20),
        }
    )


def test_build_manifest_basic(sample_df: pd.DataFrame) -> None:
    m = build_manifest(
        dataset=sample_df,
        config={"model": "rule_augmented", "C": 1.0},
        random_seed=42,
        extra={"note": "smoke run"},
    )
    assert isinstance(m, Manifest)
    assert m.schema_version == 1
    assert SHA256_RE.match(m.dataset_hash)
    assert m.dataset_shape == (20, 2)
    assert m.config == {"model": "rule_augmented", "C": 1.0}
    assert m.config_hash is not None and SHA256_RE.match(m.config_hash)
    assert m.random_seed == 42
    assert m.libraries.get("clinikit") is not None
    assert m.environment.get("python") is not None
    assert m.extra == {"note": "smoke run"}


def test_build_manifest_accepts_numpy_array(rng: np.random.Generator) -> None:
    arr = rng.standard_normal((30, 4))
    m = build_manifest(dataset=arr)
    assert m.dataset_shape == (30, 4)
    assert SHA256_RE.match(m.dataset_hash)


def test_build_manifest_handles_1d_array(rng: np.random.Generator) -> None:
    arr = rng.standard_normal(15)
    m = build_manifest(dataset=arr)
    assert m.dataset_shape == (15, 1)


def test_build_manifest_rejects_3d_array() -> None:
    arr = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="1- or 2-D"):
        build_manifest(dataset=arr)


def test_build_manifest_no_config_means_no_hash(sample_df: pd.DataFrame) -> None:
    m = build_manifest(dataset=sample_df)
    assert m.config_hash is None
    assert m.config == {}


def test_write_load_manifest_round_trip(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    m = build_manifest(
        dataset=sample_df,
        config={"a": 1, "b": [1, 2, 3]},
        random_seed=7,
    )
    out = write_manifest(m, tmp_path / "subdir" / "manifest.json")
    assert out.is_file()
    loaded = load_manifest(out)
    assert loaded.dataset_hash == m.dataset_hash
    assert loaded.dataset_shape == m.dataset_shape
    assert loaded.config == m.config
    assert loaded.config_hash == m.config_hash
    assert loaded.random_seed == m.random_seed
    assert loaded.libraries == m.libraries


def test_write_manifest_writes_sorted_keys(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = write_manifest(build_manifest(dataset=sample_df), tmp_path / "m.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data) == sorted(data)


def test_load_manifest_rejects_newer_schema(tmp_path: Path) -> None:
    bad = tmp_path / "future.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "created_at": "2026-01-01T00:00:00+00:00",
                "dataset_hash": "sha256:" + "0" * 64,
                "dataset_shape": [1, 1],
                "config_hash": None,
                "config": {},
                "libraries": {},
                "environment": {},
                "git_commit": None,
                "random_seed": None,
                "extra": {},
            }
        )
    )
    with pytest.raises(ValueError, match="schema version"):
        load_manifest(bad)


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")
