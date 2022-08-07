.. https://stackoverflow.com/a/22885876

.. role:: bolditalic
   :class: bolditalic


.. toctree::
   :caption: The spellbook Library
   :maxdepth: 2
   :hidden:

   API
   Tools and Technologies <tools>


.. toctree::
   :caption: Projects & Tutorials
   :hidden:

   Table of Contents <examples/examples.rst>
   Binary Classification: Stroke Prediction <examples/1-binary-stroke-prediction/index>
   Decision Forests in TensorFlow <examples/2-tensorflow-decision-forests/index>
   _examples/3-tensorflow-serving-docker/code.rst
   examples/4-tf-serving-docker-aws/index


.. toctree::
   :caption: Extras
   :hidden:

   literature
   glossary
   genindex
   todo

.. genindex in toctree taken from https://stackoverflow.com/a/42310803



===============================================================================
Welcome to the :bolditalic:`spellbook` Data Science & Machine Learning Library!
===============================================================================


.. admonition:: Why *spellbook*?
   :class: spellbook-why

   - *spellbook* contains classes and functions to
     **boost productivity** in the area of data science and machine learning.
   - It also features a **projects & tutorials** section which will be further
     populated over time. These projects and tutorials serve both to explore and
     learn about data science and machine learning tools and techniques as well as to provide hands-on
     examples for how to use the *spellbook* library.


.. margin:: Gallery

   from left to right:

   - :func:`spellbook.plot.pairplot`
   - :func:`spellbook.plot.plot_confusion_matrix`
   - :func:`spellbook.train.ROCPlot.plot`
   - :func:`spellbook.inspect.PermutationImportance.plot`
   - :func:`spellbook.train.plot_history_binary`

     - true/false positives/negatives
     - loss and accuracy

   - :func:`spellbook.plot1D.histogram`
   - :func:`spellbook.plot2D.categorical_histogram`
   - :func:`spellbook.plot.parallel_coordinates`
   - :class:`spellbook.train.BinaryCalibrationPlot`


.. margin:: News
   :class: spellbook-news

   - *June 26, 2021*: :doc:`/examples/4-tf-serving-docker-aws/index`
   - *June 21, 2021*: :doc:`/_examples/3-tensorflow-serving-docker/code`
   - *June 16, 2021*: :doc:`/examples/2-tensorflow-decision-forests/index`
   - *June 13, 2021*: :doc:`/examples/1-binary-stroke-prediction/index`
   - *June 13, 2021*: Release


.. admonition:: Gallery
   :class: spellbook-gallery

   .. list-table::
      :class: spellbook-gallery-scroll

      * - .. figure:: /images/pairplot-5x5.png
             :height: 200px

        - .. figure:: /images/confusion-matrix-absolute.png
             :height: 200px

        - .. figure:: /images/roc.png
             :height: 200px

        - .. figure:: /images/permutation-importance.png
             :height: 200px

        - .. figure:: /images/true-false-pos-neg-rates.png
             :height: 200px

        - .. figure:: /images/loss-acc.png
             :height: 200px

        - .. figure:: /images/histogram.png
             :height: 200px

        - .. figure:: /images/categorical-histogram-stats.png
             :height: 200px

        - .. figure:: /images/parallel-coordinates.png
             :height: 200px

        - .. figure:: /images/calibration.png
             :height: 200px


.. admonition:: Projects & Tutorials
   :class: spellbook-projects

   - :doc:`/examples/1-binary-stroke-prediction/index`
   - :doc:`/examples/2-tensorflow-decision-forests/index`
   - :doc:`/_examples/3-tensorflow-serving-docker/code`
   - :doc:`/examples/4-tf-serving-docker-aws/index`


*spellbook* is where I collect functionality that I implement and expect to
reuse later. In this spirit, *spellbook* grows as needed and its development
does not aim to complete a specific set of features. The repository is
structured as follows:

- ``doc/``: Sphinx documentation including

  - source code / API documentation
  - assorted notes on tools and technologies

- ``examples/``: Projects and tutorials
- ``spellbook/``: Python modules

  - :mod:`spellbook.input`: functions for data preparation and input pipelining
  - :mod:`spellbook.inspect`: functions for model inspection
  - :mod:`spellbook.plot`: high-level functions for creating and saving plots
  - :mod:`spellbook.plot1D`: low-level functions for creating 1D plots
  - :mod:`spellbook.plot2D`: low-level functions for creating 2D plots
  - :mod:`spellbook.plotutils`: helper functions for the other plotting modules
  - :mod:`spellbook.stat`: statistics helpers
  - :mod:`spellbook.train`: functions for model training and validation



.. _Installation:

Installation
------------

*spellbook* is available via its `GitHub repository
<https://github.com/dmrauch/spellbook>`_. You can clone it with

.. code:: bash

   $ git clone git@github.com:dmrauch/spellbook.git

*spellbook* depends on *Python* as well as a number of tools and packages built
for and on top of *Python*, most notably

- *Matplotlib* (``matplotlib``) → https://matplotlib.org/
- *NumPy* (``numpy``) → https://numpy.org/
- *pandas* (``pandas``) → https://pandas.pydata.org/
- *scikit-learn* (``sklearn``) → https://scikit-learn.org/stable/
- *SciPy* (``scipy``) → https://scipy.org/
- *seaborn* (``seaborn``) → https://seaborn.pydata.org/
- *TensorFlow* (``tensorflow``) → https://www.tensorflow.org/

These and the other dependencies can be installed via the included *Anaconda*
environment requirement file ``spellbook.yml``. Therefore, it is
recommended to install `Anaconda <https://anaconda.org/>`_. Afterwards you can
do:

.. code:: bash

   $ cd spellbook
   $ conda env create --file spellbook.yml

which will create an *Anaconda* environment called ``spellbook`` and install
the configured packages into it. This environment will be located in the
``envs/`` folder in your *Anaconda* installation.

If you want to use *Jupyter* notebooks, please activate the environment and
register it with the *Jupyter* service:

.. code:: bash

   $ conda activate spellbook
   $ python -m ipykernel install --user --name=spellbook


To make this package available on your system, do the following:

- Add the repository's root folder to your
  ``$PYTHONPATH``, e.g. via your ``.bashrc``:

  .. code:: bash

     export PYTHONPATH=$PYTHONPATH:/path/to/repository/spellbook

  Then you can import *spellbook* modules with

  .. code:: python

     import spellbook as sb
  
- Alternatively, e.g. in a *Jupyter* notebook, add the repository's root folder
  to the system path:

  .. code:: python

     import sys
     sys.path.append('/path/to/repository/spellbook')


To compile the *Sphinx* documentation including the API reference and the notes,
do

.. code:: bash

   $ cd doc
   $ make html

The documentation is then built in ``doc/build/html/``.



Usage
-----

When you want to use *spellbook* after installation, just activate the
*Anaconda* environment:

.. code:: bash

   $ conda activate spellbook



Development
-----------

Some of the docstrings of the *Python* functions include *doctest* code
snippets with examples that are shown in the source code documentation.
These examples can be run as tests with

.. code:: bash

   $ cd doc
   $ make doctest


Publishing of the Compiled Documentation to *GitHub Pages*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   The Sphinx documentation is automatically built and published via the *GitHub Action*
   configured in ``.github/workflows/sphinx-publish.yml`` when a commit is pushed to the
   `publish` branch - i.e. changes should be merged from `master` into `publish` and then pushed
   to *GitHub*.

The automated build makes use of the ``requirements.txt``. This is due to the fact that the
*GitHub* action ``sphinx-notes/pages`` accepts a *pip* requirements file. Creating the conda
environment on the runner and listing it into a requirements file led to errors when
``sphinx-notes/pages`` tried to install the dependencies. Therefore, the ``requirements.txt``
now has to be kept in sync with conda's ``spellbook.yml`` manually until a better solution is
in place.


.. rubric:: The old manual procedure

- clone the repository to a different folder

  .. code:: bash

     git clone git@github.com:dmrauch/spellbook.git spellbook-gh-pages

- create the new branch ``gh-pages``
  
  .. code:: bash

     git branch gh-pages

  The new branch is created but the original branch remains active and
  checked out. Switch to the new branch with

  .. code:: bash

     git switch gh-pages

- compile the documentation in the original folder using the proper
  ``master`` or feature branch and then copy it over to the folder
  ``spellbook-gh-pages``
- push the compiled documentation to the branch ``gh-pages`` on *GitHub*

  .. code:: bash

     git push -u origin gh-pages
