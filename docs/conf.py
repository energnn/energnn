# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

print(sys.executable)
source_path = os.path.abspath("..")
sys.path.insert(0, source_path)
print(f"appended {source_path}")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "energnn"
copyright = "2024, Balthazar Donon, Hugo Kulesza"
author = "Balthazar Donon, Hugo Kulesza"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.viewcode", "sphinx.ext.todo", "sphinx.ext.mathjax"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_title = "EnerGNN"
html_short_title = "EnerGNN"

html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "energnn_title_black.png",
    "dark_logo": "energnn_title_white.png",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}
html_favicon = "_static/energnn_favicon_white.png"
html_context = {"default_mode": "dark"}

html_static_path = ["_static"]

# Autodoc options
add_module_names = False
autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "undoc-members": True,
    #    "show-inheritance": False,
    "inherited-members": False,
}

# So that dataframes appear as pandas.DataFrame and link to pandas site
autodoc_type_aliases = {"_DataFrame": "pandas.DataFrame", "_ArrayLike": "array-like"}

# No type hints in methods signature
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

todo_include_todos = True

# Generate one file per method
autosummary_generate = True

# Pour modifier l'affichage dans autosummary
modindex_common_prefix = ['energnn.']

# Utilise les templates personnalisés
autosummary_context = {
    'add_module_names': False,
}