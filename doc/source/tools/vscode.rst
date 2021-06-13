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
