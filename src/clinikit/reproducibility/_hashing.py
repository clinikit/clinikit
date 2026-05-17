"""Deterministic hashing of datasets, arrays, files, and configs.

Every public hash function returns a string of the form
``"sha256:<hex>"``. The ``"sha256:"`` prefix makes it explicit which
algorithm was used so future versions of the package can switch
without breaking older manifests (mismatch detection is then
straightforward).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "hash_array",
    "hash_config",
    "hash_dataframe",
    "hash_file",
]

_CHUNK_SIZE = 65536


def _prefixed(hexdigest: str) -> str:
    return f"sha256:{hexdigest}"


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA256-hash a ``pandas.DataFrame`` deterministically.

    The hash is invariant under index resets but sensitive to:
    column order, dtypes, and cell values (including NaN positions).

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame. The index is intentionally ignored; reset it
        beforehand if it carries information you want to capture.

    Returns
    -------
    str
        ``"sha256:<hex>"`` — 64 hex characters of digest.

    Examples
    --------
    >>> import pandas as pd
    >>> from clinikit.reproducibility import hash_dataframe
    >>> hash_dataframe(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))[:9]
    'sha256:b9'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"hash_dataframe expects a DataFrame, got {type(df).__name__}.")

    row_hashes = pd.util.hash_pandas_object(df, index=False, categorize=True)
    h = hashlib.sha256()
    h.update(b"columns:")
    h.update(",".join(map(str, df.columns)).encode("utf-8"))
    h.update(b"|dtypes:")
    h.update(",".join(str(d) for d in df.dtypes).encode("utf-8"))
    h.update(b"|rows:")
    h.update(row_hashes.to_numpy().tobytes())
    return _prefixed(h.hexdigest())


def hash_array(arr: ArrayLike) -> str:
    """SHA256-hash a numpy array.

    The hash captures shape, dtype, and raw byte content (so NaN
    positions, endianness, and contiguity all matter — arrays that
    differ in any of these will hash differently).

    Parameters
    ----------
    arr : array-like
        Anything ``numpy.asarray`` can coerce to a C-contiguous array.

    Returns
    -------
    str
        ``"sha256:<hex>"``.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"shape:{a.shape}|dtype:{a.dtype.str}|".encode())
    h.update(a.tobytes())
    return _prefixed(h.hexdigest())


def hash_file(path: str | Path) -> str:
    """Stream-hash a file with SHA256.

    Reads the file in 64 KiB chunks so very large datasets do not
    need to fit in memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a regular file. Relative paths are resolved against
        the current working directory.

    Returns
    -------
    str
        ``"sha256:<hex>"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a regular file.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"hash_file: not a regular file: {p!s}")

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return _prefixed(h.hexdigest())


def hash_config(config: dict[str, Any]) -> str:
    """SHA256-hash a JSON-serialisable configuration mapping.

    Uses ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` so
    re-ordering keys, whitespace, or trailing newlines do not change
    the hash.

    Parameters
    ----------
    config : dict
        Any mapping that JSON can serialise. Values that JSON cannot
        encode (e.g. numpy arrays, ``pathlib.Path``) must be converted
        by the caller — this function does not invent serialisers.

    Returns
    -------
    str
        ``"sha256:<hex>"``.

    Raises
    ------
    TypeError
        If ``config`` contains a value that cannot be JSON-encoded.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256()
    h.update(payload.encode("utf-8"))
    return _prefixed(h.hexdigest())
