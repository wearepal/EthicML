"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
http://www.sphinx-doc.org/en/master/config
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from typing import List

import toml

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "EthicML"
copyright = "2022, O. Thomas, T. Kehrenberg, M. Bartlett"
author = "O. Thomas, T. Kehrenberg, M. Bartlett"

# The full version, including alpha/beta/rc tags
release = toml.load("../../pyproject.toml")["tool"]["poetry"]["version"]


# -- General configuration ---------------------------------------------------
autoclass_content = "class"  # ignore docstring of __init__
autodoc_default_options = {
    # Make sure that any autodoc declarations show the right members
    "members": True,
    "imported-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    # "autosummary": True,  # if True, every page has this small summary table at the top
    "autosummary-no-nesting": True,
    "autosummary-nosignatures": True,
    "special-members": "__len__",
}
autodoc_typehints = "description"
autodoc_type_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
}
autodoc_mock_imports = [
    "PIL",
    "folktables",
    "git",
    "joblib",
    "kit",
    "matplotlib",
    "numpy",
    "pandas",
    "ranzen",
    "scipy",
    "seaborn",
    "sklearn",
    "teext",
    "torch",
    "tqdm",
    "wandb",
]
add_module_names = False  # do not show classes and functions with full module name (ethicml.data..)

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Need the autodoc and autosummary packages to generate our docs.
    "sphinx.ext.autodoc",
    "autodocsumm",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
# pygments_style = "sphinx"  # syntax highlighting style to use

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
