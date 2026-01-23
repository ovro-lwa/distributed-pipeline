# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
project = 'orca'
copyright = '2026, Caltech OVRO-LWA Team'
author = 'Yuping Huang, Marin Anderson, Casey Law, Nikita Kosogorov'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# We use sphinx-autoapi instead of autodoc because it parses source files
# STATICALLY (no imports), avoiding issues with complex dependencies like
# CASA, Celery, configmanager, etc.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'myst_parser',
]

# -- AutoAPI configuration ---------------------------------------------------
# This parses Python source files without importing them
autoapi_type = 'python'
autoapi_dirs = ['../orca']  # Only document the main library package
autoapi_ignore = [
    '*/__pycache__/*',
    '*/tests/*',
    '*/extra/*',  # Contains older experimental code with syntax issues
]
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]
autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = False
autoapi_add_toctree_entry = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
# Intersphinx mapping for cross-referencing external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Create _static directory if it doesn't exist (needed for build)
os.makedirs(os.path.join(os.path.dirname(__file__), '_static'), exist_ok=True)
