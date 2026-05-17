"""Sphinx configuration for clinikit documentation."""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "clinikit"
author = "Berat Kaan SEVEN"
copyright = f"{datetime.now(tz=timezone.utc).year}, {author}"

try:
    release = importlib.metadata.version("clinikit")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"clinikit {release}"

# nbsphinx — do not execute notebooks during the docs build.
nbsphinx_execute = "never"
