******
Python
******


Profiling
=========

- Only installation of the package ``snakeviz`` is necessary. ``cProfile`` is a built-in module.

- If you would like to profile a notebook, export it to a Python script first

- Invoke the cProfile profiler with::

  $ python -m cProfile -o <profile-file-to-create> <script-to-profile>

- Run SnakeViz to visualise the profiling results::

  $ snakeviz vabe-engine-stateless-run.prof



Testing
=======

*pytest*
--------

*pytest-xdist*
^^^^^^^^^^^^^^

Running tests can be parallelized across the different CPUs on a system with the package
*pytest-xdist*. The option ``-n <N>`` will distribute the tests to *N* CPUs. Specifying
``-n auto`` will use all CPUs of the system.

This option can also be specified as part of different configuration files, e.g.

- ``pytest.ini`` ::

     [pytest]
     addopts = -n auto

- ``setup.cfg`` ::

     [tool:pytest]
     addopts = -n auto



Logging
=======

The ``logging`` module:

- `Source code reference <https://docs.python.org/3/library/logging.html>`_
- `Logging How-To <https://docs.python.org/3/howto/logging.html>`_


Parsing
=======

*Beautiful Soup*
----------------

HTML/XML parsing

- `Documentation <https://www.crummy.com/software/BeautifulSoup/bs4/doc/>`_
- `conda package <https://anaconda.org/conda-forge/beautifulsoup4>`_
