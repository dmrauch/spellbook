******************
Conda, Mamba & Pip
******************

.. tip::

   Use *mamba* (`repo <https://github.com/mamba-org/mamba>`_,
   `doc <https://mamba.readthedocs.io/en/latest/index.html>`_)
   for faster dependency resolution during package installation


*conda* Command Reference
=========================

- Install *mamba* into the *conda* ``base`` environment::

  $ conda install mamba -n base -c conda-forge

- To update *conda* itself, do something like this::

  $ conda update -n base -c defaults conda


Environment Activation, Inspection & Deactivation
-------------------------------------------------

- Show environments available:

  .. code:: bash

     $ conda env list

- Activate an environment:

  .. code:: bash

     $ conda activate <env_name>

- Deactivate the current environment:

  .. code:: bash

     $ conda deactivate

- List all packages (and their versions) in the current environment

  - *Option 1* -- Easily readable by eye

    .. code:: bash

       $ conda list

    creates output that looks like this::

       # packages in environment at /Users/drauch/Software/Conda/miniconda/miniconda3/envs/spellbook:
       #
       # Name                    Version                   Build  Channel
       abseil-cpp                20210324.2           h23ab428_0
       absl-py                   0.15.0             pyhd3eb1b0_0
       aiohttp                   3.8.1            py39hca72f7f_0
       aiosignal                 1.2.0              pyhd3eb1b0_0
       alabaster                 0.7.12             pyhd3eb1b0_0

       # [...] output truncated


  - *Option 2* -- For use by packaging tools

    .. code:: bash

       $ conda list -e

    creates the following output::

       # This file may be used to create an environment using:
       # $ conda create --name <env> --file <this file>
       # platform: osx-64
       abseil-cpp=20210324.2=h23ab428_0
       absl-py=0.15.0=pyhd3eb1b0_0
       aiohttp=3.8.1=py39hca72f7f_0
       aiosignal=1.2.0=pyhd3eb1b0_0
       alabaster=0.7.12=pyhd3eb1b0_0

       # [...] output truncated

    This can be used to create a file that can later be used to recreate the same environment:

    .. code:: bash
   
       $ conda list -e > conda-requirements.txt
       $ conda create --name <env> --file conda-requirements.txt

    Note that the ``conda-requirements.txt`` file does not work with *pip*, though.
    For *pip*, you should do

    .. code:: bash

       $ pip freeze > requirements.txt
       $ pip install -r requirements.txt

   

Environment Creation & Update
-----------------------------

- Create an environment and install packages:

  .. code:: bash

     $ conda create --name <env_name> python=3.8 tensorflow

- Create environment from requirements file:

  .. code:: bash

     $ conda env create --file spellbook.yml

  *or* - if *mamba* is installed -

  .. code:: bash

     $ mamba env create --file spellbook.yml

- Update an environment after changes to the requirements file:

  .. code:: bash

     $ conda env update --file spellbook.yml --prune

  *or* - if *mamba* is installed -

  .. code:: bash

     $ mamba env update --file spellbook.yml --prune

- Register an environment with Jupyter notebooks:

  .. code:: bash

     $ python -m ipykernel install --user --name=<env_name>

- List the environments registered with Jupyter:

  .. code:: bash

     $ jupyter kernelspec list



Environment Deletion
--------------------

- Delete an environment:

  .. code:: bash

     $ conda env remove -n <env_name>

- Remove an environment from Jupyter notebooks:

  .. code:: bash

     $ jupyter kernelspec uninstall <env_name>



*pip* Command Reference
=======================

- Install a package from a directory in *editable mode*, e.g. for development. This requires a
  ``setup.py``::

  $ pip install -e .

  or ::

  $ pip install -e some/folder


Mixing *conda* and *pip*
========================

General guidelines:
  
- First use *conda*, then *pip*
- When a *conda* environment is activate, *pip* will install packages into it
- If something goes wrong with *pip*, better don't bother trying to fix it,
  just delete the entire environment and reinstall from the *conda* yaml file and the *pip*
  ``requirements.txt``

You can add *pip* packages to be installed into the *conda* yaml file - when the environment is
created, *pip* is run and the specified packages are installed:

.. code:: yaml

   name: my-env-1
   channels:
       - conda-forge
       - anaconda

   dependencies:
       - python = 3.7
       - ipykernel # required to register the env as a kernel
       - numpy = 1.20.2

       - pip
       - pip:

          # add an additional package index / package repository
          # - --extra-index-url https://some-server.url/with/a/package/repository

          # add some packages
          - package1
          - package2




Links & Resources
=================

- `Homepage <https://anaconda.org/>`_
- `Anaconda package index/repository <https://anaconda.org/anaconda/repo>`_
