******************
Visual Studio Code
******************


- To control, when the active *Python* interpreter is shown in the status bar, edit the setting
  ``python.interpreter.infoVisibility`` and set it to ``always``

- To use *VS Code* as a three-way merge editor for *git*, add the following lines to your
  ``~/.gitconfig``::

     [merge]
       tool = code
     [mergetool "code"]
       cmd = code --wait --merge $REMOTE $LOCAL $BASE $MERGED



Extensions
==========


The list of installed extensions can be exported with

.. code:: bash

    $ code --list-extensions

Extensions can be installed with

.. code:: bash

    $ code --install-extension <ExtensionName>

Taken from https://stackoverflow.com/a/49398449.


The *Python* Extension
----------------------

With the *Python* extension in Visual Studio Code, it is possible to
create *cells* in normal ``*.py`` files by putting

.. code:: python

    # %%

(or ``#%%``) in a line. The cell can be run with ``Shift + Enter`` from within
VS Code. Since this is just a plain comment, it does not change in any way,
how the file behaves when it is run as a script with ``python <filename>``.


The *env* File
^^^^^^^^^^^^^^

A plain text file can be specified that contains environment variables and their values
which should be used when debugging or when running tests. The default file path is
``${workspaceFolder}/.env``, which is stored in the setting *Python: Env File*.
The file format is::

   # comment
   ENV_VAR=VALUE



The *Python Docstring Generator* Extension
------------------------------------------

Apply the following settings:

- Set *Auto Docstring: Docstring Format* to ``google``
- Set *Auto Docstring: Quote Style* to ``'''``
- Check *Auto Docstring: Start On New Line*

When using *Sphinx* with the ``sphinx_autodoc_typehints`` extension, it is sufficient to
include the types in the type hints and not necessary to include them in the docstrings as well.
In the spirit of avoiding duplication, I have created a ``google-no-types.mustache`` docstring
template based on `NilsJPWerner's <https://github.com/NilsJPWerner/autoDocstring>`_ original
`google.mustache <https://github.com/NilsJPWerner/autoDocstring/blob/master/src/docstring/templates/google.mustache>`_:

.. code::

    {{! Google Docstring Template Without Type Hints }}
    {{summaryPlaceholder}}

    {{extendedSummaryPlaceholder}}

    {{#parametersExist}}
    Args:
    {{#args}}
        {{var}}: {{descriptionPlaceholder}}
    {{/args}}
    {{#kwargs}}
        {{var}} (optional): {{descriptionPlaceholder}}. Defaults to {{&default}}.
    {{/kwargs}}
    {{/parametersExist}}

    {{#exceptionsExist}}
    Raises:
    {{#exceptions}}
        {{type}}: {{descriptionPlaceholder}}
    {{/exceptions}}
    {{/exceptionsExist}}

    {{#returnsExist}}
    Returns:
    {{#returns}}
        {{descriptionPlaceholder}}
    {{/returns}}
    {{/returnsExist}}

    {{#yieldsExist}}
    Yields:
    {{#yields}}
        {{descriptionPlaceholder}}
    {{/yields}}
    {{/yieldsExist}}

When the path to this template file is specified in the *Auto Docstring: Custom Template Path*
setting, google-style docstrings without type specifications are generated.



Tasks
=====

Run a task within a specific conda environment
----------------------------------------------

- Create an entry in ``tasks.json``
- Activate the desired *conda* environment before executing the actual command, e.g.::

    "command": "conda activate spellbook && make html"

- VS Code tasks are run in non-interactive shells, e.g. ``zsh -c``. Therefore, for the
  *conda* environment activation to work, *conda* has to be initialized in a file that is
  read by the non-interactive shell, e.g. ``~/.zhenv``.



Tests
=====

*pytest*
--------

- Open the command palette and select "Python: Configure Tests" [#VSCodeConfigureTests]_
- Follow the instructions, select the folder where the tests reside and select *pytest*.
  This will create or modify the file ``.vscode/settings.json`` in the project folder and add
  the following lines::

     {
         "python.testing.pytestArgs": [
             "<SOME-FOLDER>"
         ],
         "python.testing.unittestEnabled": false,
         "python.testing.pytestEnabled": true
     }

- If environment variables need to be specified, this can be done in a ``.env`` file in the
  project folder, e.g. ::

     MY_VAR1=hello
     MY_VAR2=123

- ``.env`` is the default filename that *VS Code* expects. If the filename and/or location is
  different, *VS Code* can be made aware of this via a setting in the ``.vscode/settings.json``,
  e.g.::

    "python.envFile": "${workspaceFolder}/<SOME-OTHER-FOLDER>/.env2",


*pytest-xdist*
--------------

*pytest-xdist* enables running of *pytest* unittests in parallel on multiple CPU cores.

- Install *pytest-xdist*
- In the project folder, create a file ``pytest.ini`` and specify the number of CPU cores that
  should be used [#VSCodePyTestXDist]_::

     [pytest]
     addopts=-n auto

  ``-n auto`` will attempt to detect the number of physical CPUs of the machine. If this fails,
  *pytest* will fall back to ``-n logical`` which corresponds to the number of logical CPUs.
  *pytest-xdist* can be deactivated completely with ``-n0 --dist no`` [#PyTestXDistDoc]_

  Alternatively, such settings can also be added to a ``setup.cfg``, e.g. [#PyTestXDistExample]_::

     [tool:pytest]
     addopts = --verbose --numprocesses auto --dist=loadscope
     python_files = unit_testing/test_*.py unit_testing/cli/test_*.py



.. rubric:: Links & References

.. [#VSCodeConfigureTests] https://code.visualstudio.com/docs/python/testing#_configure-tests
.. [#VSCodePyTestXDist] https://code.visualstudio.com/docs/python/testing#_run-tests-in-parallel
.. [#PyTestXDistDoc] https://pytest-xdist.readthedocs.io/en/latest/distribution.html
.. [#PyTestXDistExample] https://github.com/pytest-dev/pytest-xdist/issues/231#issuecomment-762959356