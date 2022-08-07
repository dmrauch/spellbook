*******
Jupyter
*******


Handy Stuff
===========

- Automatically update modules imported into a notebook when the code inside a module changes:

  .. code::

     # autoreload imports: https://stackoverflow.com/a/57245926
     %reload_ext autoreload
     %autoreload 2



Jupyter Notebooks and Git
=========================


Jupytext
--------

In order to avoid excessive file sizes from the cell outputs in Jupyter notebooks as well as
non-vanishing file diffs from changed execution counters, it can be handy to not check
the notebooks themselves into Git but rather the corresponding exported Python scripts.
Jupytext is a handy two-way syncing tool both for exporting the notebooks as scripts
as well as for updating existing notebooks with changes to the tracked Python scripts pushed to
the repo by others. The upshot of this mechanism is that the output of the Jupyter notebooks
does not have to be cleaned or deleted. In particular, when changes to a notebook are merged from
the tracked Python scripts, only the output of the modified cells is lost.
The workflow is as follows:

- Create a Jupyter notebook file and fill it with your code

- From the notebook, create a corresponding Python script file::

  $ jupytext --set-formats ipynb,py:percent notebook.ipynb

- Edit the notebook, save it and update the Python script::

  $ jupytext --sync notebook.ipynb

- Add/stage, commit and push the updates to the Python script::

  $ git add notebook.py

- When somebody else updates a Python script, you can just pull and merge their changes
  and then run the same sync command::

  $ jupytext --sync notebook.ipynb
