"""clinikit — a lightweight, sklearn-compatible toolkit for tabular ML.

clinikit bundles 14 hybrid classifiers, calibration utilities,
label-noise diagnostics, fairness audits, experiment protocols, and
structured HTML reports behind a single drop-in package.

The package is intended for research and development use only. It
is not a regulated product. See ``CITATIONS.md`` in the project
root for the academic references behind every bundled method.

Submodules
----------
clinikit.models
    14 sklearn-compatible hybrid classifiers.
clinikit.preprocessing
    Imputers, scalers, outlier and missingness indicators.
clinikit.metrics
    Sensitivity, specificity, NPV, PPV, F2, MCC, Brier, ECE.
clinikit.curves
    ROC, PR, calibration, Decision Curve Analysis.
clinikit.calibration
    Platt, isotonic, temperature scaling, reliability diagrams.
clinikit.thresholds
    Five threshold-selection strategies.
clinikit.statistics
    DeLong, bootstrap CIs, McNemar.
clinikit.diagnostics
    Label-noise tools (Cleanlab integration, neighborhood conflict, LOO).
clinikit.cost_sensitive
    Asymmetric error weighting, Bayes-optimal threshold.
clinikit.cross_val
    Group-aware cross-validation utilities.
clinikit.protocols
    Five experiment protocols.
clinikit.leaderboard
    Experiment-tracking CSV with 38 columns.
clinikit.report
    HTML report generator (Jinja2 templates).
clinikit.audit
    Leakage detection, subgroup fairness, documentation checks.
clinikit.governance
    Audit-trail documentation templates.
clinikit.reproducibility
    Run-manifest helpers (data hash + config + library versions).
clinikit.datasets
    Bundled UCI benchmark datasets.
clinikit.monitor
    Drift detection (KS, Wasserstein, PSI) and performance monitoring.
clinikit.modelcard
    Hugging Face Model Card generator.
clinikit.plots
    Matplotlib helpers for common plots.
clinikit.explainability
    Optional SHAP / LIME wrappers and partial dependence.
clinikit.automl
    Optional thin wrappers around TabPFN, FLAML, AutoGluon.
clinikit.external_val
    Cross-dataset validation framework.
clinikit.time_split
    Chronological train/test split utilities.
clinikit.active_learning
    Optional modAL wrapper.
clinikit.synthetic
    Optional CTGAN/TVAE wrappers with a TSTR safety gate.
clinikit.cli
    Typer-based command-line interface.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Berat Kaan SEVEN"
__license__ = "MIT"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
]
