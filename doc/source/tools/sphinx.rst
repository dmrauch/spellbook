Sphinx
******


Tips & Tricks
=============


Useful Extensions
-----------------

============================ =====================================================================================================
Name                         Description
============================ =====================================================================================================
``sphinx.ext.intersphinx``   links to external *Sphinx* documentations of
                             functions and classes,
                             e.g. :class:`matplotlib.figure.Figure`
``sphinx_tabs.tabs``         provides container element with different tabs
                             beside each other
``sphinx.ext.todo``          create a list of all todos
``sphinx.ext.napoleon``      support for Google-style docstrings
``sphinx.ext.viewcode``      create HTML pages showing the source code and add
                             links to them to the respective objects in the
                             source code documentation
``sphinx.ext.doctest``       include code snippets/examples in ``*.rst`` files
                             and in the docstrings and test them to ensure
                             the documentation is up to date with the code
``sphinx.ext.autodoc``       given a *Python* module/class/function,
                             automatically create the source code documentation
``sphinx.ext.autosummary``   just point *Sphinx* to the *Python* modules and it
                             will generate the *stub* files for
                             ``sphinx.ext.autodoc`` automatically
``autodocsumm``              creates summary tables for the classes and
                             functions in a module
``sphinx_autodoc_typehints`` automatic documentation of typehints
``nbsphinx``                 add ``*.ipynb`` files to the toctree, automatically
                             execute them and generate pages that display the
                             notebook text and code cells and its outputs
``sphinx.ext.githubpages``   creates ``.nojekyll`` file for GitHub Pages
============================ =====================================================================================================

Built-in extensions are listed here:
https://www.sphinx-doc.org/en/master/usage/extensions/index.html

.. note:: Despite the different extensions, it is not possible in *Sphinx* to
          automatically generate a source code documentation where each of the
          different members has a toc entry associated to it. Therefore,
          unfortunately, the source code documentation can only be scrolled
          through or searched, but it is not possible to automatically
          generate page-level tables of content pointing to the individual
          classes, functions and attributes.
          This seems to be possible in :ref:`MkDocs`.
          
          It is, however, possible to generate summary tables for all members
          at the top of each page using the ``autodocsumm`` extension.



Text After Roles Without Intermediate Space
-------------------------------------------

When a role is inserted, such as ``:class:`pandas.DataFrame```, the backtick
ending the role has to be followed by whitespace. When this is not desired,
e.g. when using the plural, this can be avoided by using a backslash instead
of the space: ``:class:`pandas.DataFrame`\s`` will be rendered as
:class:`pandas.DataFrame`\s.



.. _sphinx-glossary:

Glossary
--------

The built-in glossary can be generated with the ``.. glossary::`` directive.
Underneath it, the individual terms can be defined like this:

.. code:: rst

   .. glossary::

      FPR
         False Positive Rate

      TPR
         True Positive Rate

On a normal *rst* page, an expression can be linked to the corresponding
glossary entry by means of the ``:term:`` role: The code ``:term:`FPR```
will result in a link like this: :term:`FPR`. You can also show different
text while still linking to the desired glossary entry: The code
``:term:`False Positive Rate <FPR>``` will be rendered as :term:`False Positive
Rate <FPR>`.

Unfortunately, since *rst* roles cannot be nested, vanilla *Sphinx* does not
allow for the combination of hovering tooltips with links to a central
glossary.



Numbered Footnotes
------------------

.. code:: rst

   I found this great paper [#fPaper]_

   .. rubric:: Links

   .. [#fPaper] Awesome Authors: *Awesome Paper*, Awesome Journal

is rendered as

.. highlights::

   I found this great paper [#fPaper]_

   .. rubric:: Links

   .. [#fPaper] Awesome Authors: *Awesome Paper*, Awesome Journal



Tables
------

In *simple tables*, coded like this

.. code:: rst

   ============ ============
   Column 1     Column 2
   ============ ============
   row 1 cell 1 row 1 cell 2
   row 2 cell 1 row 2 cell 2
   ============ ============

the relative size of the columns is defined in the ``<colgroup>`` and ``<col>``
tags in the generated HTML. The fractions that each column make up are
calculated from the relative length of the ``===`` sequences in the rst code.


The *Book* Theme
================

The *Book* theme is a responsive *Sphinx* theme with a file-based navigation
bar on the left and an in-document table of content in the right page margin.

- Homepage: https://sphinx-book-theme.readthedocs.io/en/latest/index.html
- Conda package ``sphinx-book-theme``:
  https://anaconda.org/conda-forge/sphinx-book-theme

.. warning:: At least ``sphinx-book-theme`` versions 0.0.40 and 0.0.41 do not
   work properly with *Sphinx* version 4: The ``.. margin::`` and
   ``.. sidebar::`` directives are not rendered correctly. This can be fixed
   by sticking to ``sphinx`` version 3.5.4.


Changing the Page Width
-----------------------

To increase the width of the overall page, add a custom ``*.css`` file to
the ``_static`` folder and specify it in ``conf.py``:

.. margin:: **File**

   ``doc/source/conf.py``

.. code:: python

   html_static_path = ['_static']
   html_css_files = ['custom-book.css']

The main container is ``container-xl``. The following snippet will extend the
page over the full width of the browser window. The left sidebar and the right
page margin are kept fixed and the increase in size benefits entirely the
central content pane.

.. margin:: **File**

   ``doc/source/_static/custom-book.rst``

.. code:: css

    .container-xl {
    max-width: none; /* 90% !important; */
   }
  


Page Elements
-------------

.. margin:: **My margin title**

   Here is my margin content, it is pretty cool!


Some text in between


.. sidebar:: **My sidebar title**

   Here is my sidebar content, it is pretty cool! Let's see how far this
   extends into the right page margin and what happens to the rest of the
   text...


:A Caption:

Some more text in between


.. margin:: Code blocks in margins

   Some text

   .. code:: python

      print('hello world!')


.. note::

   This is a note in the main text


.. margin:: **Notes in margins**

   .. note::

      This is a note in the margin


Let's write some more nonsensical text to simulate a meaningful document
containing really great content. Apparently, one has to be careful and watch
how the elements in the main text and in the right page margin are laid out.
As stated in the `Sphinx Book Theme documentation
<https://sphinx-book-theme.readthedocs.io/en/latest/layout.html>`_,
the elements can overlap.

.. code:: python
   
   # now let's see how source code is rendered

   import spellbook.python.plot as sb.plot

Any element can be made to extend fully from the main text into the right
page margin by adding ``:class: full-width``.

.. note::
   :class: full-width

   This is a full-width note


Now the main text continues.



Additional Container Elements
=============================

Tooltips
--------

There is built-in support for simple tooltips in Sphinx with the ``:abbr:``
role: ``:abbr:`normal text (tooltip text)``` will be rendered as
:abbr:`normal text (tooltip text)`.



*sphinx-tabs*
-------------

*sphinx-tabs* provides the ``.. tabs::`` directive which creates an element
with multiple tabs/pages beside each other

- https://github.com/executablebooks/sphinx-tabs
- https://anaconda.org/conda-forge/sphinx-tabs

.. tabs::

   .. tab:: First Tab

      Content of the first tab

      .. note:: Some information can go inside a note

   .. tab:: Second Tab

      There is some text here

      .. code:: python

         print('... and some code!')



Source Code Documentation
=========================

*sphinx.ext.intersphinx*
------------------------

When ``make html`` is run, *Sphinx* not only creates the HTML pages, but also
the ``objects.inv`` in the same directory. The ``objects.inv`` files of other
projects can be targeted with *intersphinx* and used to generate hyperlinks
to the source code documentation of other projects.

Add to ``conf.py``:

.. code:: python

   intersphinx_mapping = {
      'matplotlib': ('https://matplotlib.org/stable/', None),
      'numpy': ('https://numpy.org/doc/stable/', None),
      'pandas': ('https://pandas.pydata.org/docs/', None),
      'python': ('https://docs.python.org/3/', None),
      'seaborn': ('https://seaborn.pydata.org/', None),

      'tensorflow': ( # https://github.com/GPflow/tensorflow-intersphinx/
                      # - mentioned in https://stackoverflow.com/a/37444321
         'https://www.tensorflow.org/api_docs/python',
         'https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv'
      )
   }

Then, objects belonging to these other projects can be referenced and linked
using the ``:func:`` and ``:class:`` roles. The following naming prefixes
have to be used:

- ``matplotlib``
- ``numpy``
- ``pandas``
- no prefix for *Python*
- ``seaborn``
- ``sklearn`` for *scikit-learn*
- ``tf`` for *TensorFlow*, e.g. :class:`tf.data.Dataset`

.. note:: At least for *pandas* and *TensorFlow*, some object names are
          expanded in the auto-generated source code documentation based on
          the type hints / signatures (but not when the same objects are
          mentioned manually in the docstrings with ``:func:`` or ``:class:``,
          and neither in normal ``*.rst`` files!). As a result, the expanded
          object names cannot be found in the respective ``objects.inv``
          and no external documentation link is added.

          For example, ``:class:`tf.data.Dataset``` is rendered correctly
          as :class:`tf.data.Dataset`, but when a signature includes
          ``tf.data.Dataset``, this name is expanded to
          ``tensorflow.python.data.ops.dataset_ops.DatasetV2``.
          Another example is ``pd.DataFrame`` which is expanded to
          ``pandas.core.frame.DataFrame``.

          This is a known issue without any obvious solution on the
          implementation side:
          https://github.com/agronholm/sphinx-autodoc-typehints/issues/47

          It is possible, however, to fix these special cases by manually
          writing the types in the docstring:

          - For parameter types, add the reference in parentheses to the
            respective parameter (the others remain unaffected), e.g. like so:

            .. code:: rst

               Args:
                  data(:class:`pandas.DataFrame`): The imbalanced data

          - For the return type, just write something like this into the
            docstring:
            
            .. code:: rst
            
               Returns:
                  Tuple of :class:`tf.data.Dataset`: A tuple containing the
                  training and validation (and possibly test) datasets



*sphinx.ext.doctest*
--------------------

Directives:

- Test code separated from the output
 
  .. code:: rst
  
     .. testcode::

        import numpy as np
        a = np.arange(10)
        print(a.shape)
      
     Output:

     .. testoutput::

        (10,)

- Test code interleaved with the output

  .. code:: rst

     .. doctest::

        >>> print('hello world!')
        hello world!

        >>> print('hello again...')
        hello again...

Run with ``make doctest``.

.. rubric:: Links

- https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html
- https://sphinx-tutorial.readthedocs.io/step-3/



*sphinx-autodoc-typehints*
--------------------------

*sphinx-autodoc-typehints* automatically generates the documentation of the
typehints, thus eliminating the need to manually reproduce the typehints in
the docstrings.

.. note:: When used together with *sphinx.ext.napoleon*,
          *sphinx-autodoc-typehints* has to be included **after**
          *sphinx.ext.napoleon* in the configuration file ``conf.py``

Settings:

- ``typehints_fully_qualified = True``: show the module names before the
  object names
- ``simplify_optional_unions = False``: keep typing.Optional in Unions
  for optional parameters, I find this more explicit


.. rubric:: Links

- https://github.com/agronholm/sphinx-autodoc-typehints
- https://anaconda.org/conda-forge/sphinx-autodoc-typehints



Tools for Jupyter Notebooks
===========================


*nbsphinx*
----------

The *nbsphinx* extension provides support for Jupyter notebooks in *Sphinx*.
Notebooks can be included in toctrees and will be exectuted when *Sphinx* is run.
The rendered text and code cells along with the resulting output will be added
to the documentation.

- https://nbsphinx.readthedocs.io
- Conda package ``nbsphinx``: https://anaconda.org/conda-forge/nbsphinx

Quickstart:

#. Add ``'nbsphinx'`` to the ``extensions`` list in ``conf.py``
#. Add some ``*.ipynb`` files to a toctree
#. Run ``make html`` to create the documentation


*sphinx-thebe*
--------------

*sphinx-thebe* is a *Sphinx* extension for live code execution.

- https://sphinx-book-theme.readthedocs.io/en/latest/launch.html#live-code-cells-with-thebe
- https://sphinx-thebe.readthedocs.io/en/latest/index.html
- Conda package ``sphinx-thebe``: https://anaconda.org/conda-forge/sphinx-thebe



My Modifications and Additions
==============================


Admonitions
-----------


General Blue Admonition
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Blue Admonition
   :class: spellbook-admonition-blue

   .. code:: rst

      .. admonition:: Admonition Title
         :class: spellbook-admonition-blue

         Admonition content



General Orange Admonition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Orange Admonition
   :class: spellbook-admonition-orange

   .. code:: rst

      .. admonition:: Admonition Title
         :class: spellbook-admonition-orange

         Admonition content
         
         
         
Definition Admonition
^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Definition
   :class: spellbook-definition

   .. code:: rst

      .. admonition:: Definition
         :class: spellbook-definition

         Definition content



Glossary Tooltips
-----------------

.. admonition:: Definition
   :class: spellbook-definition

   Modified behaviour of the ``:term:`` and ``:abbr:`` roles.

   .. code:: rst

      The :term:`type-1 error` is related to the :abbr:`CL (confidence level)`.


As mentioned in :ref:`sphinx-glossary`, Vanilla *Sphinx* has the limitation
that reST roles cannot be nested and therefore a word or phrase cannot be
simultaneously given a tooltip with ``:abbr:`phrase``` and entered and linked
to the glossary with ``:term:`phrase```.
      
.. margin:: Source Files Involed

   - ``source/_static/glossary.py``
   - ``Makefile``
   - ``source/conf.py``
   - ``source/_static/tooltip.js``
   - ``source/_static/tooltip.css``
   - ``source/_templates/layout.html``

.. margin:: Build Files involved

   - ``build/html/glossary.html``

.. margin:: Files Generated

   - ``build/html/glossary.json``

To overcome this, I extended the behaviour of the ``:term:`` role.
The *Python* module ``source/_static/glossary.py`` is invoked in the
``Makefile`` after the ``sphinx-build`` command. It parses
the automatically created glossary in ``build/html/glossary.html`` and extracts
the terms and their definitions/explanations into a JSON dictionary which is
then written to ``build/html/_static/glossary.json``. Despite the name, this
file is actually a bit of *JavaScript* just containing the JSON dictionary.
``glossary.json`` is added to the ``html_js_files`` configuration parameter in
``source/conf.py`` so that this file is added as a script and read when an HTML
page is loaded. I also wrote a *JavaScript* script ``source/_static/tooltip.js``
that is also added to the HTML pages. When the HTML page is loaded, it reads
the JSON glossary dictionary from ``glossary.json`` and creates event handlers
connected to the all the appearances of the glossary terms on the HTML page.
When the mouse is then brought to hover over such a link to a glossary term,
the corresponding entry is retrieved from the glossary dictionary and displayed
in a custom tooltip. These tooltips are styled in
``source/_static/tooltip.css``. The regular hyperlinks of the terms/phrases to
their coresponding entries in ``glossary.html`` are retained, so when clicking
on a term/phrase, the full glossary is still loaded.

These glossary tooltips support all the normal *reST* containers, directives
and roles and therefore, the glossary entries can be written without
limitations. Since normally, links to the *MathJax* library are only included
in the HTML headers, when the underlying ``*.rst`` file contains a math
directive or role, I had to force the inclusion of the corresponding
``<script>`` tags via the ``extrahead`` template block in
``source/_templates/layout.html``. Now, math formulae and equation can be
displayed in the glossary tooltips even if the parent ``*.rst`` page does not
contain any math.

The glossary tooltips are positioned automatically in a way that they are
displayed within the viewport borders. However, since *MathJax* rendering takes
a moment, a glossary tooltip may subsequently grow beyond the viewport borders
after initial positioning.

Similarly-styled tooltips are also used to replace the normal plain ones for
``:abbr:``.

These glossary tooltips look like this in action:
The :term:`type-1 error` is related to the :abbr:`CL (confidence level)`.



Plot Galleries
--------------


Horizontally Scrolling Gallery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. admonition:: Definition
   :class: spellbook-definition

   .. code:: rst

      .. list-table::
         :class: spellbook-gallery-scroll

         * - .. figure:: /images/plot_grid_1D.png
                :height: 200px

           - .. figure:: /images/loss-acc.png
                :height: 200px

           - .. figure:: /images/true-false-pos-neg-rates.png
                :height: 200px

           - .. figure:: /images/rec-prec.png
                :height: 200px

           - .. figure:: /images/roc.png
                :height: 200px

           - .. figure:: /images/confusion-matrix-absolute.png
                :height: 200px

The table of plots will be scrollable horizontally if it is wider than the
window. Otherwise, the plots will be centered horizontally.

.. list-table::
   :class: spellbook-gallery-scroll

   * - .. figure:: /images/plot_grid_1D.png
          :height: 200px

     - .. figure:: /images/loss-acc.png
          :height: 200px

     - .. figure:: /images/true-false-pos-neg-rates.png
          :height: 200px

     - .. figure:: /images/rec-prec.png
          :height: 200px

     - .. figure:: /images/roc.png
          :height: 200px

     - .. figure:: /images/confusion-matrix-absolute.png
          :height: 200px



Wrapping Gallery
^^^^^^^^^^^^^^^^

.. admonition:: Definition
   :class: spellbook-definition

   .. code:: rst

      .. list-table::
         :class: spellbook-gallery-wrap

         * - .. figure:: /images/loss-acc.png
                :height: 200px

           - .. figure:: /images/true-false-pos-neg-rates.png
                :height: 200px

           - .. figure:: /images/roc.png
                :height: 200px

The table of plots will be wrapped into the next lines if it is wider than the
window. Otherwise, the plots will be centered horizontally.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: /images/loss-acc.png
          :height: 200px

     - .. figure:: /images/true-false-pos-neg-rates.png
          :height: 200px

     - .. figure:: /images/roc.png
          :height: 200px



Styling
-------

.. margin:: Files Involved

   - ``source/_static/custom.css``
   - ``source/_templates/autosummary/module.rst``
   - ``source/genindex.rst``
   - ``source/_templates/sidebar-social.html``

- page covering the full width of the viewport
- consistent custom colour scheme
- footnotes entries in the same line as the footnote mark in footnote lists
- horizontal lines underneath the ``<h2>`` and ``<h3>`` headers
- the *previous*/*next* buttons at the bottom of each page
- - borders around functions, classes and methods in the source code reference
- fully qualified names for modules, including the ``spellbook`` prefix, in the
  auto-generated source code documentation
- the *Extras* toctree in the left side bar with pointer to the ToDo list, the
  glossary and the index
- links to *GitHub* and *LinkedIn* at the bottom of the left sidebar
