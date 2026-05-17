# Contributing to clinikit

Thanks for considering a contribution. `clinikit` is a small, focused
toolkit and we keep the bar for new code high. This document explains
how to set up a development environment, what we expect from
contributions, and how the review process works.

> All identifiers, comments, docstrings, and examples are in **English**.
> Pull requests with Turkish (or other non-English) code are not
> accepted — translate them first.

---

## Table of Contents

1. [Ground rules](#ground-rules)
2. [Local development setup](#local-development-setup)
3. [Workflow](#workflow)
4. [Coding standards](#coding-standards)
5. [Tests](#tests)
6. [sklearn compatibility](#sklearn-compatibility)
7. [Documentation](#documentation)
8. [Commit and PR guidelines](#commit-and-pr-guidelines)
9. [Release process](#release-process)

---

## Ground rules

- `clinikit` is a **research and development library**. We do not
  ship features that imply production deployment, regulatory clearance,
  or specific domain certification.
- The package is **integration over invention**. We adapt and combine
  techniques from the literature, and we cite every source in
  [`CITATIONS.md`](CITATIONS.md). We do not claim novelty.
- The scope for v0.1 is fixed (27 modules, MoSCoW frozen). Open an
  issue before proposing a new module — most ideas belong in a fork
  or downstream package.
- Be kind. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## Local development setup

1. Fork the repo on GitHub and clone your fork.
2. Create a virtual environment inside the project folder:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   ```

3. Install in editable mode with the dev extras:

   ```bash
   pip install -e ".[dev,docs,diagnostics]"
   ```

4. Install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

   The hooks run `ruff`, `mypy`, and basic file checks. They must
   pass before a commit lands.

---

## Workflow

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feature/short-description
   ```
2. Make focused changes. One topic per branch; one PR per topic.
3. Add or update tests so coverage does not drop below 80%.
4. Run the full pre-flight check (see below).
5. Push and open a pull request against `main`.

### Pre-flight check (run before every PR)

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=clinikit --cov-fail-under=80
pytest tests/test_sklearn_compatibility.py
```

All five steps must pass.

---

## Coding standards

- **Python:** target `>=3.10, <3.14`. Use modern syntax (PEP 604
  `X | Y` unions, structural pattern matching where it helps).
- **Style:** `ruff` for linting and formatting. Configuration in
  [`pyproject.toml`](pyproject.toml). No personal style debates —
  let the formatter decide.
- **Types:** add type hints to every public function and method.
  `mypy` runs in the CI pipeline; aim for strictness where practical.
- **Docstrings:** NumPy style everywhere. Each public symbol must
  have a docstring with `Parameters`, `Returns`, and an `Examples`
  block. Examples must be runnable.
- **Comments:** explain *why*, not *what*. The code already says what.
- **Imports:** never use `from x import *`. Optional dependencies must
  be lazy-imported inside the function that needs them.
- **Naming:** `snake_case` for functions, methods, attributes, modules.
  `PascalCase` for classes. No abbreviations that obscure meaning.
- **Paths:** always `pathlib.Path`. Never raw strings, never
  `os.path.join`. Use `importlib.resources.files()` for package data.
- **No author-named identifiers.** No method, file, class, parameter,
  or variable may carry the author's name.

---

## Tests

- Tests live in `tests/` and mirror `src/clinikit/`.
- We use `pytest` with at least **80% line coverage**.
- File IO uses pytest's `tmp_path` or `tmp_path_factory` fixtures —
  never `/tmp` or hardcoded paths.
- Float comparisons go through `pytest.approx`, `np.isclose`, or
  `np.testing.assert_allclose`. Never `==` on floats.
- Every stochastic test fixes `random_state`.
- **No network calls** in tests. Mock or skip with a clear marker.

---

## sklearn compatibility

Every public classifier in `clinikit.models` MUST pass
`sklearn.utils.estimator_checks.check_estimator`. A dedicated test
file [`tests/test_sklearn_compatibility.py`](tests/test_sklearn_compatibility.py)
parameterizes over all public estimators.

Common pitfalls:

- `fit()` MUST return `self`.
- `__init__` MUST NOT mutate parameters. Store them unchanged as
  instance attributes with the same name.
- Fitted attributes end with a trailing underscore
  (`self.classes_`, `self.feature_names_in_`, `self.n_features_in_`).
- `predict_proba` returns shape `(n_samples, n_classes)`.
- Use `sklearn.utils.validation.check_X_y` and `check_array`.

A new public estimator is not accepted unless its `check_estimator`
test passes.

---

## Documentation

- API reference is auto-generated from docstrings via Sphinx.
- Tutorial notebooks live in `examples/`. Keep them small and
  reproducible.
- Update [`CHANGELOG.md`](CHANGELOG.md) in the same PR as any
  user-visible change (follow Keep a Changelog).

---

## Commit and PR guidelines

- Default branch: `main`.
- Commit messages: **Conventional Commits** in English. Examples:
  - `feat(models): add SelectiveClassifier`
  - `fix(metrics): correct ECE binning edge case`
  - `docs(readme): clarify install extras`
  - `refactor(audit): split fairness checks into submodule`
- Keep commits small and focused. Squash trivial fixups before review.
- Reference issues with `Fixes #123` or `Refs #123` in the PR body.
- PRs include: summary, motivation, test plan, breaking-change notes,
  CHANGELOG entry. The template is [here](.github/PULL_REQUEST_TEMPLATE.md).
- A PR is merged when CI is green, coverage holds, and at least one
  maintainer approves.

---

## Release process

Maintainers only:

1. Bump version in [`pyproject.toml`](pyproject.toml) and
   [`CITATION.cff`](CITATION.cff).
2. Update [`CHANGELOG.md`](CHANGELOG.md) — move `Unreleased` entries
   under a new version heading with the release date.
3. Tag the release: `git tag v0.X.Y && git push --tags`.
4. The `publish.yml` GitHub Actions workflow builds the wheel and
   uploads to PyPI via Trusted Publishing (OIDC, no API keys).
5. A Zenodo DOI is minted automatically from the GitHub release.

---

Thanks again. If you have questions, open a
[discussion](https://github.com/clinikit/clinikit/discussions) before
filing an issue.
