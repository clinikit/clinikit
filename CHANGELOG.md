# Changelog

All notable changes to `clinikit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-17

### Fixed
- README links to project files (`CITATIONS.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, `LICENSE`, and the quickstart
  notebook) now use absolute GitHub URLs so they resolve correctly
  from the PyPI project page. Relative paths were 404'ing on PyPI
  because the long description is rendered outside the repository
  context.

## [0.1.0] - 2026-05-17

### Added
- Initial public release of `clinikit`, a lightweight, sklearn-compatible
  Python toolkit for tabular machine learning.
- 14 hybrid classifiers (`clinikit.models`):
  `RuleAugmentedClassifier`, `BoundaryRefineClassifier`,
  `SubgroupThresholdClassifier`, `ErrorAwareCalibrator`,
  `MonotonicBooster`, `HardSampleWeightedEnsemble`,
  `ClassConditionalImputer`, `CrossDistributionDistiller`,
  `SelectiveClassifier`, `InstanceAdaptiveThreshold`,
  `DialecticalEnsemble`, `LatentSubtypeRouter`,
  `IterativeLabelRefiner`, `DualViewCoTrainer`.
- `preprocessing` — imputers (median, KNN, MICE, MissForest, DomainAware),
  scalers (Standard, Robust, YeoJohnson, Quantile, MinMax), outlier flag,
  missing indicator.
- `metrics` — sensitivity, specificity, NPV, PPV, F2, MCC, Brier, ECE,
  balanced accuracy.
- `curves` — ROC, PR, calibration curve, Decision Curve Analysis.
- `protocols` — 5 experiment protocols (Defensible, MaxScore,
  OriginalOnly, Deployment, Audit).
- `leaderboard` — experiment tracking CSV with 38 columns.
- `report` — HTML structured report generator via Jinja2.
- `audit` — leakage detection, subgroup fairness, documentation checks.
- `governance` — audit-trail manifest templates (documentation helpers).
- `reproducibility` — manifest files (data hash + config + library versions).
- `datasets` — UCI benchmarks (PIMA, Wisconsin, UCI Heart, Frankfurt).
- `cli` — Typer-based CLI with `train`, `benchmark`, `audit`, `validate`, `report`.
- `plots` — matplotlib wrappers for common plots.
- `thresholds` — 5 strategies (accuracy_max, recall_constrained,
  accuracy_constrained, cluster_specific, two_stage).
- `calibration` — Platt scaling, isotonic, temperature scaling,
  reliability diagrams.
- `statistics` — DeLong test, bootstrap CIs, McNemar test.
- `diagnostics` — Cleanlab integration, neighborhood conflict, LOO
  influence, seed stability.
- `cost_sensitive` — asymmetric error weighting, Bayes-optimal threshold.
- `monitor` — drift detection (KS, Wasserstein, PSI), performance monitoring.
- `modelcard` — Hugging Face Model Card generator.
- `cross_val` — group-aware cross-validation.
- `explainability` — SHAP and LIME wrappers, partial dependence (optional).
- `automl` — TabPFN, FLAML, AutoGluon thin wrappers (optional, safe-mode).
- `external_val` — cross-dataset validation framework.
- `time_split` — chronological train/test split utilities.
- `active_learning` — modAL wrapper for labeling loops (optional).
- `synthetic` — CTGAN, TVAE wrappers with TSTR safety gate (optional).
- Full sklearn estimator compatibility checks
  (`sklearn.utils.estimator_checks.check_estimator`) for every public classifier.
- GitHub Actions CI matrix across Python 3.10/3.11/3.12/3.13 on
  Ubuntu/macOS/Windows.
- Sphinx documentation scaffold, Read the Docs configuration.
- MIT license.

[Unreleased]: https://github.com/clinikit/clinikit/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/clinikit/clinikit/releases/tag/v0.1.1
[0.1.0]: https://github.com/clinikit/clinikit/releases/tag/v0.1.0
