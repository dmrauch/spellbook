***
Git
***

.. toctree::
   :hidden:
   :maxdepth: 1

   github


Git Command Reference
=====================


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

- Remove stale branches, i.e. branches that were deleted in the remote
  repository and in the local repository, but that are still listed as remote
  branches in the local repository::

  $ git fetch -p

  or ::

  $ git remote prune origin


Remotes
-------

- List all remotes::

  $ git remote -v

- Rename a remote::

  $ git remote rename <old-name> <new-name>


Stashing
--------

================================================= ==============================
Command              Description
================================================= ==============================
``git stash list``                                List the existing stashes
``git stash push -m <message> <file1> <file2>``   Create a stash entry on the current branch
                                                  from the changes in the specified files
``git stash show stash@{0}``                      Show the diffstat of the most recent patch
``git stash show -p stash@{0}``                   Show the most recent patch as a diff / patch
================================================= ==============================


Revert Changes
--------------

- Create new changes that roll back the changes from a specific older commit::

  $ git revert <commit-hash>



Git How-Tos
===========

How to Split Out a Folder Into a Separate Repository
----------------------------------------------------

... and keep the *git* history!

Very heavily based on https://ao.ms/how-to-split-a-subdirectory-to-a-new-git-repository-and-keep-the-history/

1. Make a duplicate clone of the old repository. This copy of the repository will eventually become
   the new repository that holds only the selected folder that is split out. If there is a copy of
   the old repository sitting in the same directory, direct the duplicate clone to a folder with
   a different name (`<new-repository>`)::
   
   $ git clone <old-repository> <new-repository>
   $ cd <new-repository>

2. Check out the appropriate branch::

   $ git checkout <branch>

3. Filter to keep only the contents of the desired folder::

   $ fit filter-branch --prune-empty --subdirectory-filter <subdirectory>

4. Create a new *git* repository at your hoser of choice. This will be the new repository holding
   only the desired folder.

5. Configure the new repository as the remote::

   $ git remote set-url origin <url-to-new-repository>

6. Push the contents of the new repository::

   $ git push -u origin <branch>

7. Optionally, delete the folder from the old repository
