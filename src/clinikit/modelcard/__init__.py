"""clinikit.modelcard — Hugging Face Model Card generator.

The generator emits a YAML-frontmatter + Markdown file compatible
with the Hugging Face Hub's model-card schema, populated from a
``clinikit`` run manifest and metrics object.

Public functions
----------------
- generate_modelcard(run, out_path)
"""

from __future__ import annotations

__all__: list[str] = []
