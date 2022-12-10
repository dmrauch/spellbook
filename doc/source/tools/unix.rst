****************
Unix & CLI tools
****************


curl
====

- run a POST request::

   $ curl -X POST -d name=somebody -H <HEADER> localhost:5000/some-endpoint

- bearer tokens go into the header - this can be done with something like this::

   $ -H $(gcloud auth generate token)

- files can be uploaded with::

    $ -d data=@audio.wav

  The ``@`` ensures that the content of the file is uploaded



zsh
===

The following files can be used to influence the behaviour of *zsh*
(based on https://unix.stackexchange.com/a/71258) - listed in the order in which
they are read:

- ``~/.zshenv``: Is always sourced, i.e. both for interactive and non-interactive (``zsh -c``)
  shells
- ``~/.zprofile``: Is only sourced for interactive shells
- ``~/.zshrc``: Is only sourced for interactive shells
- ``~/.zlogin``: Is only sourced for interactive shells
