****************
Anaconda / Conda
****************


Command Reference
=================


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

- List all packages (and their versions) in the current environment:

  .. code:: bash

     $ conda list

   

Environment Creation & Update
-----------------------------

- Create an environment and install packages:

  .. code:: bash

     $ conda create --name <env_name> python=3.8 tensorflow

- Create environment from requirements file:

  .. code:: bash

     $ conda env create --file spellbook.yml

- Update an environment after changes to the requirements file:

  .. code:: bash

     $ conda env update --file spellbook.yml --prune

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



Mixing *conda* and *pip*
========================

.. todo::
  
   - First use conda, then pip
   - yml and requirements.txt files
   - conda env activated, pip installs into it
   - if something goes wrong with pip,don't bother trying to fix it,
     just delete everything and reinstall from the files



Links & Resources
=================

- `Homepage <https://anaconda.org/>`_
- `Anaconda package index/repository <https://anaconda.org/anaconda/repo>`_
