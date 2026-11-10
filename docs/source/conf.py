"""Sphinx configuration of the pyPALC documentation."""
import sys
from pathlib import Path

# repository root, so that ``pypalc`` is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypalc  # noqa: E402

project = "pyPALC"
author = "Arne Hölter, Florian Straube"
copyright = "2021-2026, Arne Hölter, Florian Straube"
release = pypalc.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# definition lists (term, then ": definition") in the Markdown pages
myst_enable_extensions = ["deflist"]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
# document the functions and classes of every module, not only its docstring
autodoc_default_options = {"members": True, "show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"pyPALC {release}"
