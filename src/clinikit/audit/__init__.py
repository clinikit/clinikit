"""clinikit.audit — leakage / fairness / documentation aids.

Public functions
----------------
- :func:`detect_leakage`        — train/test contamination diagnostics.
- :func:`subgroup_fairness`     — per-group performance + disparity.
- :func:`documentation_checklist` — required-field check for a run record.

Public dataclasses
------------------
- :class:`LeakageReport`
- :class:`SubgroupFairnessReport`
- :class:`DocumentationCheckResult`

These helpers are documentation aids for research workflows; they do
not replace formal compliance review.
"""

from __future__ import annotations

from clinikit.audit._audit import (
    DocumentationCheckResult,
    LeakageReport,
    SubgroupFairnessReport,
    detect_leakage,
    documentation_checklist,
    subgroup_fairness,
)

__all__ = [
    "DocumentationCheckResult",
    "LeakageReport",
    "SubgroupFairnessReport",
    "detect_leakage",
    "documentation_checklist",
    "subgroup_fairness",
]
