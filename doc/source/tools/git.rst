***
Git
***


Git
===

Configuration
-------------

- Global git settings - stored in ``~/.gitconfig``::

  $ git config --global user.name "My Name"
  $ git config --global user.emal "my@email.xyz"

- Per repository settings - stored in ``.git/config``::

  $ git config user.name "My Name"
  $ git config user.email "my@email.xyz"


Branches
--------

- Switch to an existing branch::

  $ git checkout <branch-name>

- Create a new branch and check it out::

  $ git checkout -b <branch-name>


Revert Changes
--------------

- Create new changes that roll back the changes from a specific older commit::

  $ git revert <commit-hash>



GitHub
======

Git Credentials for GitHub
--------------------------

.. code:: bash

    $ git config --local user.name dmrauch
    $ git config --local user.email 17890191+dmrauch@users.noreply.github.com


SSH Authentication
------------------

From August 2021, GitHub will no longer allow password-based authentication for git operations: https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/

- Connecting to GitHub with SSH: https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

  - Adding a new SSH key to your GitHub account: https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account
  - Testing the SSH connection: https://docs.github.com/en/github/authenticating-to-github/testing-your-ssh-connection
  - GitHub's SSH key fingerprints: https://docs.github.com/en/github/authenticating-to-github/githubs-ssh-key-fingerprints
  - Working with SSH key passphrases: https://docs.github.com/en/github/authenticating-to-github/working-with-ssh-key-passphrases


GitHub Pages
------------

There are two kinds of GitHub Pages [#fGitHubPages]_:

- **user/organisation site**: Hosted at ``<username>.github.io``. For these,
  you need to create a repository names ``<username>.github.io``
- **project site**: Hosted at ``<username>.github.io/<project>``. These can
  be created for any repository.

GitHub only allows specific folders where the documentation can be located
for it to be served as a GitHub Pages site [#fGitHubPagesSources]_. For project
sites, these are:

- the repository's root/main folder
- the ``docs`` folder

However, it allows selection of the branch that should be used as the source.
Typical choices are the ``docs`` folder on a branch called ``gh-pages``. This
way, the project source code (including the source code of documentation) and
the (compiled) website can be kept separately.

A good step-by-step introduction and tutorial is given in
[#fGitHubPagesTutorial]_.

A published *GitHub Pages* site can be removed/unpublished by selecting `None`
as the publishing source [#fGitHubPagesUnpublish]_. If the site was published from
the ``gh-pages`` branch, it may have to be deleted as well?


.. rubric:: Links

.. [#fGitHubPages] https://pages.github.com/
.. [#fGitHubPagesSources] https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
.. [#fGitHubPagesTutorial] https://www.docslikecode.com/articles/github-pages-python-sphinx/
.. [#fGitHubPagesUnpublish] https://docs.github.com/en/pages/getting-started-with-github-pages/unpublishing-a-github-pages-site
