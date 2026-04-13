import os
import subprocess
import sys

import spleaf

# -- Project information -----------------------------------------------------
project = 'S+LEAF'
copyright = '2019-2024, Jean-Baptiste Delisle'
author = 'Jean-Baptiste Delisle'

# -- General configuration ---------------------------------------------------
needs_sphinx = '1.1'
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.coverage',
  'numpydoc',
  'matplotlib.sphinxext.plot_directive',
  'nbsphinx',
  'nbsphinx_link',
]
templates_path = ['_templates']
exclude_patterns = []
pygments_style = 'sphinx'
autosummary_generate = True
plot_include_source = True
nbsphinx_execute_arguments = [
  "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {'gitlab_url': 'https://gitlab.unige.ch/delisle/spleaf'}
html_static_path = ['_static']

# -- Add links to original jupyter notebooks in doc --------------------------
try:
  git_rev = subprocess.check_output(
    ['git', 'describe', '--exact-match', 'HEAD'], universal_newlines=True
  )
except subprocess.CalledProcessError:
  try:
    git_rev = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], universal_newlines=True
    )
  except subprocess.CalledProcessError:
    git_rev = ''
if git_rev:
  git_rev = git_rev.splitlines()[0] + '/'

nbsphinx_link_target_root = os.path.join(os.path.dirname(__file__), '..', '..')
nbsphinx_prolog = (
  r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'] %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='doc/source/') %}
{% endif %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        `Download this notebook`__

    __ https://gitlab.unige.ch/delisle/spleaf/-/raw/
        """
  + git_rev
  + r'{{ docpath }}'
  + '?inline=false'
)
