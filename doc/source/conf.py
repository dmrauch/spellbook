# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'spellbook'
copyright = '2021, Daniel Rauch (dmrauch)'
# author = 'dmrauch'
# version = '0.1'
# release = '0.1b1'
import datetime


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# import sphinx_rtd_theme

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx_tabs.tabs',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc', # auto-generate source code documentation
    'sphinx.ext.autosummary', # auto-generate stub *.rst files for each *.py file
    'autodocsumm', # automatic summary tables at the top of each page
    'sphinx_autodoc_typehints', # automatic documentation of typehints,
                                # must come *after* 'sphinx.ext.napoleon'
    'nbsphinx',
    # 'sphinx_thebe',
    'sphinx.ext.githubpages', # creates .nojekyll file for GitHub Pages
    'doc.source._ext.code' # provides custom directive 'code-output'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# html_baseurl not needed for standard GitHub Pages site
# html_baseurl = 'https://dmrauch.github.io/spellbook'

html_title = 'spellbook'
html_logo = '_static/spellbook-logo.png'
html_favicon = '_static/spellbook-icon.ico'

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
html_last_updated_fmt = "%B %-d, %Y"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css', 'tooltip.css',
    # examples table based on DataTables and Bootstrap4
    # 'jquery.dataTables.min.css',
]
html_js_files = [
    # glossary tooltips
    ('tooltip.js', {'defer': 'defer'}),
    'glossary.json',

    # examples table based on DataTables and Bootstrap4
    # 'jquery-3.5.1.js', # is already present
    # 'jquery.dataTables.min.js',
]

# HTML templates/elements in the left sidebar
# - https://sphinx-book-theme.readthedocs.io/en/latest/configure.html#control-the-left-sidebar-items
# - https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_sidebars
# - double asterisk to include slashes: https://www.sphinx-doc.org/en/master/usage/configuration.html#id11
html_sidebars = {
    '**': ['sidebar-search-bs.html', # https://sphinx-book-theme.readthedocs.io/en/latest/configure.html?highlight=bootstrap#default-sidebar-elements
           'sbt-sidebar-nav.html',
           'sbt-sidebar-footer.html',
           'sidebar-social.html'],
}


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Book Theme
html_theme = 'sphinx_book_theme'

html_theme_options = {

    'path_to_docs': 'doc/source',

    'repository_url': 'https://github.com/dmrauch/spellbook',
    # 'repository_branch': 'gh-pages', # used by the 'edit' button
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,

    'use_download_button': True,
    # 'use_fullscreen_button': True, # documented but unsupported option

    # 'single_page': True, # remove the navigation sidebar
    'home_page_in_toc': False,     # landing page as first toc entry
    # 'toc_title': '{your-title}', # title of the right sidebar toc
    'extra_navbar': '',

    'launch_buttons': {
        'notebook_interface': 'jupyterlab',
        # 'thebe': True,
    },

    # 'show_prev_next': False, # hide the prev/next buttons

    # theme_extra_footer
    'extra_footer': '''Built with <a href="{sphinx}">Sphinx</a>
using the <a href="{sphinx_book_theme}">Sphinx Book Theme</a>'''.format(
    sphinx = 'https://www.sphinx-doc.org/en/master/index.html',
    sphinx_book_theme = 'https://sphinx-book-theme.readthedocs.io/en/latest/'
),

    'search_bar_text': 'Search' # change the text in the search bar
}



# -- Options for extensions --------------------------------------------------

todo_include_todos = True
todo_link_only = True

autodoc_default_options = {
    'autosummary': True,        # add summary to all autodoc-generated pages
}
autodoc_default_flags = ['members']

autosummary_generate = True

# sphinx_autodoc_typehints
typehints_fully_qualified = True   # show the module names before the object
                                   # names
simplify_optional_unions = True    # do not keep typing.Optional in Unions for
                                   # optional parameters


intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),

    'tensorflow': ( # https://github.com/GPflow/tensorflow-intersphinx/
                    # - mentioned in https://stackoverflow.com/a/37444321
        'https://www.tensorflow.org/api_docs/python',
        'https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv'
    )
}
