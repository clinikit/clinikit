"""clinikit.governance — audit-trail manifest templates.

Outputs are *documentation templates* whose first line explicitly
states that they are intended for research documentation and not
for regulated decision-making (see :data:`GOVERNANCE_DISCLAIMER`).

Public functions
----------------
- :func:`generate_manifest`        — write a JSON or Markdown manifest.
- :func:`list_required_fields`     — fields a complete manifest carries.
- :func:`render_manifest_markdown` — render a payload as Markdown.

Public data
-----------
- :data:`GOVERNANCE_DISCLAIMER`
- :data:`MANIFEST_REQUIRED_FIELDS`
"""

from __future__ import annotations

from clinikit.governance._manifests import (
    GOVERNANCE_DISCLAIMER,
    MANIFEST_REQUIRED_FIELDS,
    generate_manifest,
    list_required_fields,
    render_manifest_markdown,
)

__all__ = [
    "GOVERNANCE_DISCLAIMER",
    "MANIFEST_REQUIRED_FIELDS",
    "generate_manifest",
    "list_required_fields",
    "render_manifest_markdown",
]
