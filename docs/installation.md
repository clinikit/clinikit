# Installation

`clinikit` is published on PyPI. Install the core package:

```bash
pip install clinikit
```

Optional dependency groups bring in heavier or domain-specific
backends. Install them on demand:

```bash
pip install "clinikit[diagnostics]"   # Cleanlab-based label-noise tools
pip install "clinikit[explain]"       # SHAP, LIME
pip install "clinikit[automl]"        # TabPFN, FLAML, AutoGluon
pip install "clinikit[synthetic]"     # CTGAN, TVAE
pip install "clinikit[conformal]"     # MAPIE
pip install "clinikit[active]"        # modAL
pip install "clinikit[all]"           # everything above
```

Supported Python versions: **3.10, 3.11, 3.12, 3.13**, on Linux,
macOS, and Windows.

## Development install

For contributors:

```bash
git clone https://github.com/clinikit/clinikit.git
cd clinikit
python3.13 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev,docs]"
pre-commit install
```
