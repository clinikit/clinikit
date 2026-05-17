"""clinikit.reproducibility — run-manifest helpers.

A reproducibility manifest captures the inputs and environment for a
single experiment run:

- SHA256 hash of the dataset
- Hash of the configuration object
- Versions of all tracked Python libraries
- Random seed and run-level metadata
- Python / OS / architecture fingerprint
- Current git commit, when available

Public API
----------
Hashing
~~~~~~~
- :func:`hash_dataframe`
- :func:`hash_array`
- :func:`hash_file`
- :func:`hash_config`

Environment
~~~~~~~~~~~
- :data:`DEFAULT_TRACKED_PACKAGES`
- :func:`collect_library_versions`
- :func:`collect_environment`
- :func:`get_git_commit`

Manifest
~~~~~~~~
- :class:`Manifest`
- :func:`build_manifest`
- :func:`write_manifest`
- :func:`load_manifest`
"""

from __future__ import annotations

from clinikit.reproducibility._environment import (
    DEFAULT_TRACKED_PACKAGES,
    collect_environment,
    collect_library_versions,
    get_git_commit,
)
from clinikit.reproducibility._hashing import (
    hash_array,
    hash_config,
    hash_dataframe,
    hash_file,
)
from clinikit.reproducibility._manifest import (
    Manifest,
    build_manifest,
    load_manifest,
    write_manifest,
)

__all__ = [
    "DEFAULT_TRACKED_PACKAGES",
    "Manifest",
    "build_manifest",
    "collect_environment",
    "collect_library_versions",
    "get_git_commit",
    "hash_array",
    "hash_config",
    "hash_dataframe",
    "hash_file",
    "load_manifest",
    "write_manifest",
]
