"""clinikit.governance — audit-trail manifest templates.

The governance module produces JSON / YAML manifests intended as
documentation templates for research workflows. Outputs include a
disclaimer line:

    "Templates are for research documentation only."

Public functions
----------------
- generate_manifest(run, out_path)
- list_required_fields()
"""

from __future__ import annotations

__all__: list[str] = []

GOVERNANCE_DISCLAIMER = "Templates are for research documentation only."
