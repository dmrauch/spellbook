****
Rich
****

`Rich <https://rich.readthedocs.io/en/stable/index.html>`_
is a library for rich - i.e. coloured, styled and formatted - printing to the console.


TLDR
====

Just copy and paste this to the beginning of your code:

.. code:: python

   import rich
   from rich import print        # replace default print
   from rich import print_json   # pretty-print JSON
   from rich import pretty
   pretty.install()              # add rich to the REPL


If you would like to record the terminal output, use the following:

.. code:: python

   from rich.console import Console
   console = Console(record=True)

   # console.log(...)
   # console.print(...)
   # console.print_json(...)

   console.save_html('export.html')



Other Features
==============

- `progress bars <https://rich.readthedocs.io/en/stable/progress.html>`_
- `logging handler <https://rich.readthedocs.io/en/stable/logging.html>`_

  - also see :ref:`Tools and Technologies > Python > Logging <tools-python-logging>`



Demo
====

The following code

.. code:: python

    from rich.console import Console
    console = Console(record=True)

    # console message
    console.log('This is a console.log() message\n')

    # pretty printing
    console.print("Let's define and print a dictionary 'd':")
    d = {'a': 1, 'b': 2, 'c': list(range(3))}
    console.print(f'd = {d}\n')

    # JSON pretty printing
    import json
    console.print("And now let's convert it to JSON and pretty-print it")
    console.print_json(json.dumps(d))

    # export and save the console output as html
    console.save_html('rich-demo.html')

will lead to output that looks like this:

.. figure:: /images/tools/rich-1.png

And it will also create a file ``rich-demo.html`` that contains an export of the console as
HTML - which looks like this:

.. margin::

   ``rich-demo.html``

.. raw:: html
   :file: rich-demo.html
