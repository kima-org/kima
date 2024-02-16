import os
import sys
import spleaf

# -- Project information -----------------------------------------------------
project = 'S+LEAF'
copyright = '2019-2023, Jean-Baptiste Delisle'
author = 'Jean-Baptiste Delisle'

# -- General configuration ---------------------------------------------------
needs_sphinx = '1.1'
extensions = [
  'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.intersphinx',
  'sphinx.ext.coverage', 'numpydoc', 'matplotlib.sphinxext.plot_directive'
]
templates_path = ['_templates']
exclude_patterns = []
pygments_style = 'sphinx'
autosummary_generate = True
plot_include_source = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {"gitlab_url": "https://gitlab.unige.ch/delisle/spleaf"}
html_static_path = ['_static']
