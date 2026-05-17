"""clinikit.report — HTML structured report generator.

Renders a single self-contained HTML file from a run record and an
optional leaderboard DataFrame. Templates ship under
``clinikit/report/templates`` and are resolved via
:mod:`importlib.resources`, so the package works after a clean
``pip install clinikit``.

Public functions
----------------
- :func:`render_report`                  — render a single-run report.
- :func:`render_report_from_leaderboard` — render a leaderboard-only
  report from a CSV path.
"""

from __future__ import annotations

from clinikit.report._report import (
    render_report,
    render_report_from_leaderboard,
)

__all__ = [
    "render_report",
    "render_report_from_leaderboard",
]
