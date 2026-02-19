import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'HybridRAG-Bench'
author = 'HybridRAG-Bench Team'
copyright = f"{datetime.datetime.now().year}, {author}"
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'hybridrag_logo',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
add_module_names = False
autodoc_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
html_title = 'HybridRAG-Bench Documentation'
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 2,
}
