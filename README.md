# clinikit

Prepared by Berat Kaan SEVEN

A lightweight, sklearn-compatible Python toolkit for tabular machine
learning. `clinikit` bundles 14 hybrid classifiers, 5 experiment
protocols, calibration utilities, label-noise diagnostics, fairness
audits, and structured HTML reports behind a single drop-in package.

Research and development use only. This is an integration toolkit,
not a regulated product and not a research paper of original methods.
See [`CITATIONS.md`](CITATIONS.md) for source-method references.

[![CI](https://github.com/clinikit/clinikit/actions/workflows/ci.yml/badge.svg)](https://github.com/clinikit/clinikit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/clinikit/clinikit/branch/main/graph/badge.svg)](https://codecov.io/gh/clinikit/clinikit)
[![PyPI version](https://img.shields.io/pypi/v/clinikit.svg)](https://pypi.org/project/clinikit/)
[![Python versions](https://img.shields.io/pypi/pyversions/clinikit.svg)](https://pypi.org/project/clinikit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/clinikit/badge/?version=latest)](https://clinikit.readthedocs.io/en/latest/?badge=latest)

---

## Why clinikit

`clinikit` is a complement to existing libraries, not a competitor.

| Library              | Focus                                      | Why clinikit is different                                                              |
| -------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| scikit-learn         | General-purpose ML                         | Adds curated experiment protocols, audit utilities, and structured reporting           |
| Cleanlab             | Label noise only                           | Integrates Cleanlab plus neighborhood conflict and LOO into one diagnostics module     |
| MAPIE                | Conformal prediction only                  | Includes selective classification as one of 14 bundled models                          |
| Fairlearn / AIF360   | Fairness only                              | The `audit` module bundles fairness, leakage, and documentation helpers                |
| AutoGluon            | AutoML                                     | Library-first; thin AutoML wrappers exist but no auto-magic by default                 |
| PyHealth             | Deep learning for sequence / multimodal    | Tabular-only, classical ML focused, lightweight                                        |

---

## Installation

```bash
pip install clinikit
```

Optional dependency groups:

```bash
pip install "clinikit[diagnostics]"   # Cleanlab-based label-noise tools
pip install "clinikit[explain]"       # SHAP and LIME wrappers
pip install "clinikit[automl]"        # TabPFN, FLAML, AutoGluon wrappers
pip install "clinikit[synthetic]"     # CTGAN / TVAE wrappers
pip install "clinikit[conformal]"     # MAPIE conformal prediction
pip install "clinikit[all]"           # Everything
```

Supported Python versions: 3.10, 3.11, 3.12, 3.13.

---

## Quickstart

```python
from clinikit.datasets import load_pima
from clinikit.models import RuleAugmentedClassifier
from clinikit.metrics import sensitivity, specificity
from sklearn.model_selection import train_test_split

X, y = load_pima(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RuleAugmentedClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Sensitivity:", sensitivity(y_test, y_pred))
print("Specificity:", specificity(y_test, y_pred))
```

For a complete walkthrough, see [`examples/quickstart.ipynb`](examples/quickstart.ipynb)
or open it in
[Colab](https://colab.research.google.com/github/clinikit/clinikit/blob/main/examples/quickstart.ipynb).

---

## What is in the box

### 14 hybrid classifiers (`clinikit.models`)

All sklearn-compatible, all pass `sklearn.utils.estimator_checks.check_estimator`.

- `RuleAugmentedClassifier`
- `BoundaryRefineClassifier`
- `SubgroupThresholdClassifier`
- `ErrorAwareCalibrator`
- `MonotonicBooster`
- `HardSampleWeightedEnsemble`
- `ClassConditionalImputer`
- `CrossDistributionDistiller`
- `SelectiveClassifier`
- `InstanceAdaptiveThreshold`
- `DialecticalEnsemble`
- `LatentSubtypeRouter`
- `IterativeLabelRefiner`
- `DualViewCoTrainer`

### Supporting modules

- `preprocessing` — imputers, scalers, outlier flags, missing indicators
- `metrics` — sensitivity, specificity, NPV, PPV, F2, MCC, Brier, ECE
- `curves` — ROC, PR, calibration, Decision Curve Analysis
- `protocols` — 5 experiment protocols (Defensible, MaxScore, OriginalOnly, Deployment, Audit)
- `leaderboard` — experiment tracking CSV with 38 columns
- `report` — HTML structured report generator (Jinja2 templates)
- `audit` — leakage detection, subgroup fairness, documentation checks
- `governance` — audit-trail manifest templates (documentation only)
- `reproducibility` — manifest files (data hash + config + library versions)
- `datasets` — UCI benchmarks (PIMA, Wisconsin, UCI Heart, Frankfurt)
- `cli` — Typer-based CLI: `train`, `benchmark`, `audit`, `validate`, `report`
- `thresholds`, `calibration`, `statistics`, `diagnostics`, `cost_sensitive`,
  `monitor`, `modelcard`, `cross_val`, `explainability`, `automl`,
  `external_val`, `time_split`, `active_learning`, `synthetic`

---

## Command-line interface

```bash
clinikit train      --config config.yaml
clinikit benchmark  --dataset pima --models all
clinikit audit      --data data.csv --report audit.html
clinikit validate   --model model.joblib --data data.csv
clinikit report     --leaderboard runs.csv --out report.html
```

---

## Development Notes

`clinikit` was developed with AI assistance. Design discussions, code
generation, and documentation drafts were produced with Anthropic's
Claude. Final decisions, integration choices, testing, and the
public release have been reviewed and approved by the author
(Berat Kaan SEVEN).

This package is an integration toolkit. The methods it bundles are
adaptations of techniques published in the academic literature; see
[`CITATIONS.md`](CITATIONS.md) for source-method references. `clinikit`
is not a research paper of original methods, and it is not a regulated
product. It is a research and development library only.

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the development workflow, coding standards, and pull-request
process. By participating, you agree to abide by the
[`Code of Conduct`](CODE_OF_CONDUCT.md).

---

## Citation

If you use `clinikit` in academic work, please cite it via the
[`CITATION.cff`](CITATION.cff) file, or use:

```bibtex
@software{clinikit,
  author  = {SEVEN, Berat Kaan},
  title   = {clinikit: a tabular machine-learning toolkit},
  year    = {2026},
  url     = {https://github.com/clinikit/clinikit},
  version = {0.1.0}
}
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for the full text.
