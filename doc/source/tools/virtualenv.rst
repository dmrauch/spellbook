*****************
virtualenv & venv
*****************

*virtualenv* needs to be installed, e.g. via conda.

*venv* is a subset of *virtualenv* that is integrated into the *Python*
standard library and therefore needs no further installation.


venv
====

- https://docs.python.org/3/tutorial/venv.html
- https://docs.python.org/3/library/venv.html

When the environment is created, the currently active *Python* version is
'inherited'. So make sure that the appropriate *Python* version is active
by having the appropriate corresponding *conda* environment activated.

- create a venv with the name ``.venv``::

  $ python -m venv .venv

- activate the environment::

  $ source .venv/bin/activate

- ``pip install`` packages or write a ``requirements.txt`` file for running
  ``pip install -r requirements.txt``.
