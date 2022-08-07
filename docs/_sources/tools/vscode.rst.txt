******************
Visual Studio Code
******************


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
