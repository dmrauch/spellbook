***************************
GCP (Google Cloud Platform)
***************************

General
=======

- List the current configuration, including the current project

  .. code:: bash

     $ gcloud config list

- Set the active project

  .. code:: bash

     $ gcloud config set project <PROJECT_ID>

- View information about your Google Cloud SDK installation and the active configuration

  .. code:: bash

     $ gcloud info

- Update your Google Cloud SDK installation

  .. code:: bash

     $ gcloud components update

- List the available regions

  .. code:: bash

     $ gcloud compute regions list

  e.g. ``europe-west1`` (Belgium, low CO2)

- List the roles granted to a resource

  .. code:: bash

     $ gcloud iam list-grantable-roles


Authentication
==============

- View credentialed accounts::

  $ gcloud auth list

- Set the active account::

  $ gcloud config set account <ACCOUNT>



Tips & Tricks
=============

Persistently Store Configurations in Cloud Shell
------------------------------------------------

- Create a plain text file somwehere and write your stuff in there ::

     PROJECT_ID=12312212
     REGION=europe-west1

- This file can be sourced with

  .. code:: bash

     $ source some/path/config

  and the correct setting of the environment variables can be checked with

  .. code:: bash

     $ echo $PROJECT_ID
     $ echo $REGION

- To make this automatic for each new Cloud Shell that you open,
  modify your `~/.profile` and append ::

     source some/path/config



Compute
=======

Compute Engine
--------------

- list the CPU platforms/types in a particular zone ::

  $ gcloud compute zone describe



Storage
=======


Cloud Storage
-------------


Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

- create a bucket ::

  $ gsutil mb gs://<BUCKET_NAME>

- copy a file into a bucket ::

  $ gsutil cp <LOCAL_FILE_NAME> gs://<BUCKET_NAME>

- sync the contents of two buckets ::

  $ gsutil rsync ...


Python SDK
^^^^^^^^^^

- create a bucket

  .. code:: python

    from google.cloud import storage

    # connect the client
    client = storage.Client()

    # create a bucket
    bucket = client.create_bucket('bucket-name')

- upload a file to an existing bucket

  .. code:: python

    from google.cloud import storage

    # connect the client
    client = storage.Client()

    # upload the file
    bucket = client.get_bucket('bucket-name')
    new_blob = bucket.blob('remote/filename')
    new_blob.upload_from_filename(filename='local/filename')



Docker
======

https://cloud.google.com/deep-learning-containers/docs/getting-started-local

- Use `gcloud` as the credential helper for Docker

  .. code:: bash

     $ gcloud auth configure-docker
