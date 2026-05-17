# Roadmap

This document captures the planned trajectory of `clinikit`. The
scope of `v0.1.0` is fixed; everything after that is exploratory and
subject to community feedback.

> **Status legend** — :white_check_mark: shipped · :hourglass_flowing_sand:
> in progress · :seedling: planned · :no_entry_sign: deliberately
> out of scope.

---

## v0.1.0 — Foundation release (target 2026-Q3)

The first public release ships all 27 modules in the package.
Highlights:

- :seedling: 14 sklearn-compatible hybrid classifiers (`models`).
- :seedling: Preprocessing, metrics, curves, plotting utilities.
- :seedling: Five experiment protocols + leaderboard + structured
  HTML report.
- :seedling: Audit, governance, reproducibility, model-card helpers.
- :seedling: Optional groups for diagnostics, explainability, AutoML,
  synthetic data, conformal prediction, active learning.
- :seedling: Typer-based CLI: `train`, `benchmark`, `audit`, `validate`,
  `report`.
- :seedling: GitHub Actions CI matrix (Python 3.10–3.13 ·
  Ubuntu/macOS/Windows).
- :seedling: Sphinx docs hosted on Read the Docs.

### Quality gates for v0.1.0

- ruff + mypy clean.
- `pytest --cov-fail-under=80` on all CI platforms.
- `sklearn.utils.estimator_checks.check_estimator` passes for every
  public classifier.
- Portability greps (`beratkaanseven`, `/Users/`, hardcoded reference
  paths, `C:\\`) return zero matches.

---

## v0.2.0 — Stability and ergonomics (tentative, 2026-Q4)

Focus: polish, performance, and developer experience. No new modules.

- :seedling: Benchmark suite under `benchmarks/` with reproducible
  scripts.
- :seedling: Faster fit/predict paths via Cython/Numba where profiling
  warrants it.
- :seedling: Expanded tutorial notebooks (one per protocol).
- :seedling: Improved error messages, especially for sklearn API
  violations in user-supplied pipelines.
- :seedling: Optional `rich`-based progress bars in CLI and long-running
  jobs.

---

## v0.3.0 — Documentation and community (tentative, 2027-Q1)

Focus: lower the barrier for new contributors and users.

- :seedling: Public discussion forum / GitHub Discussions templates.
- :seedling: A "how to add a new estimator" cookbook.
- :seedling: Reproducibility manifest schema versioning.
- :seedling: Optional Turkish translation of the user guide.

---

## Out of scope (won't have)

The following have been explicitly **declined** for the foreseeable
future. Open a fork if you need them.

- :no_entry_sign: Mobile app, IDE extensions, video tutorials, chat
  rooms.
- :no_entry_sign: Direct EHR vendor adapters or anything that implies
  regulated deployment.
- :no_entry_sign: SaaS hosting / multi-tenant infrastructure.
- :no_entry_sign: Real-time / streaming inference.
- :no_entry_sign: Imaging, NLP, or audio modules.
- :no_entry_sign: Federated learning (Flower, FedML).
- :no_entry_sign: Distributed training (Dask, Ray).
- :no_entry_sign: Quantum or neuromorphic experimental modules.
- :no_entry_sign: Custom logo or brand identity work.
- :no_entry_sign: Any identifier (class, file, parameter) carrying the
  author's name.

---

## Versioning policy

`clinikit` follows [Semantic Versioning](https://semver.org).

- Patch (`0.X.Y`) — bug fixes, internal refactors, doc-only changes.
- Minor (`0.X.0`) — additive API changes, new optional modules.
- Major (`X.0.0`) — breaking API changes. Reserved until the library
  has stabilized in real-world use.

The leading `0.` is intentional. Until `1.0.0`, expect public APIs
to evolve based on user feedback.
