"""clinikit.modelcard — Hugging Face Model Card generator.

Public functions
----------------
- :func:`render_modelcard`   — produce the YAML+Markdown string.
- :func:`generate_modelcard` — render and write to a file.

Reference: Mitchell et al., 2019, "Model Cards for Model Reporting"
(FAT* 2019). The output follows the Hugging Face Hub ``model-card``
schema so cards drop straight into a Hub repository.
"""

from __future__ import annotations

from clinikit.modelcard._card import (
    generate_modelcard,
    render_modelcard,
)

__all__ = [
    "generate_modelcard",
    "render_modelcard",
]
