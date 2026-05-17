"""clinikit.report — HTML structured report generator.

Reports are rendered from Jinja2 templates shipped under
``clinikit/report/templates``. The default template covers dataset
summary, model configuration, training/test metrics, calibration
diagnostics, and a reproducibility manifest.

Public functions
----------------
- render_report(run, out_path, *, template="default.html.j2")
"""

from __future__ import annotations

__all__: list[str] = []
