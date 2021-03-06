*******************************************
*TensorFlow Serving* with *Docker* on *AWS*
*******************************************

.. raw:: html

   <div class="tag-list">
   <div class="tag-cell tag-date">June 26, 2021</div>
   <div class="tag-cell tag-center"></div>
   <div class="tag-cell tag-right">
      <span class="tag-text">tags:</span>
      <a class="tag right" href="../../search.html?q=tags+AWS">
         AWS</a>
      <a class="tag right" href="../../search.html?q=tags+Docker">
         Docker</a>
      <a class="tag right" href="../../search.html?q=tags+EC2">
         EC2</a>
      <a class="tag right" href="../../search.html?q=tags+ECR">
         ECR</a>
   </div>
   </div>


.. admonition:: In this project/tutorial, we will
   :class: spellbook-admonition-orange

   - Deploy a **TensorFlow** model on **AWS** and serve it with
     **TensorFlow Serving** in a **Docker container**

The source code file for this tutorial are located in
``examples/4-tensorflow-serving-docker-aws/``.



Deploying on *Amazon EC2*
=========================

In this tutorial, we will take the containerised example *TensorFlow* model
from the previous tutorial :doc:`/_examples/3-tensorflow-serving-docker/code`
and deploy it on *Amazon EC2*. To achieve this, we will

- create an *AWS Identity & Access Management* (:term:`IAM`)
  user with the required permission policies
- install the *AWS Command Line Interface* (CLI)
- upload our *Docker* image to Amazon *Elastic Container Registry*
  (:term:`ECR`)
- create a virtual server based on one of the *Deep Learning AMIs* on
  Amazon *Elastic Cloud Compute* (:term:`EC2`)
- pull the *Docker* images from *ECR* to the server instance and run it there
- expose the server instance to the internet for answering HTTP
  prediction requests via *TensorFlow Serving*'s REST API

The *Deep Learning AMIs* are container images with a range of machine learning
frameworks and tools installed, including *TensorFlow*, *PyTorch* and
*Apache MXNet*.

In this tutorial, we will issue commands both from our host machine
terminal as well as the terminal of virtual servers running on *EC2*
in the Amazon cloud. To mark the difference, the commands for the local
machine are prepended with ``$``, while the commands to be run on the
virtual server are prepended with ``(ec2) $``.

.. rubric:: Links & Resources

- Amazon ECR: `Using Amazon ECR with the AWS CLI
  <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_
- AWS Deep Learning Containers

  - `What are AWS Deep Learning Containers?
    <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html>`_
  - `Amazon EC2 Tutorials
    <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html>`_
  - `Deep Learning Containers Images
    <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_
  - GitHub: `Available Deep Learning Containers Images
    <https://github.com/aws/deep-learning-containers/blob/master/available_images.md>`_
  - `Release Notes for Deep Learning Containers
    <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`_



Preparations
------------


Storage Encryption by Default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Amazon EBS encryption
  <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSEncryption.html>`_

In the *EC2* management console, go to
*EC2 Dashboard > Account attributes > EBS encryption*
and set *Always encrypt new EBS volumes* to *Enabled*.
If the alias ``aws/ebs`` does not work, go to the *AWS Key Management Service*
(:term:`KMS`) management console and copy over the proper ARN.



Creating an *IAM* User
^^^^^^^^^^^^^^^^^^^^^^

Either extend an existing user's permissions or create a new *IAM* user
with the following permission policies attached

- ``AmazonEC2ContainerRegistryFullAccess``
- ``AmazonECS_FullAccess``
  
If it doesn't yet exist, create an access key and note down the
*access key ID* and the *secret access key*.

In the following, I will assume a user with the name ``tf-fmnist-ec2``.



Installing the *AWS CLI*
^^^^^^^^^^^^^^^^^^^^^^^^

The `installation instructions
<https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>`_
are given in the user guide in the `AWS CLI documentation
<https://docs.aws.amazon.com/cli/>`_.
Download the ``*.zip``-file, uncompress it, install it with

.. code:: bash

   $ ./aws/install --install-dir /path/to/aws-installdir --bin-dir /path/to/aws-bin-dir

and add ``/path/to/aws-bin-dir`` to the ``PATH`` environment variable
in ``.bashrc``:

.. code:: bash

   $ export PATH=$PATH:/path/to/aws-bin-dir

In case the *AWS CLI* is already installed, check that it is up to date
and if not, install the latest version as described `here
<https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-upgrade>`_.

Run ``aws configure`` and add the access key created earlier as described
`here <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html>`_.
To add the user as the default profile, run

.. code:: bash

   $ aws configure

Here, however, we will add the user ``tf-fmnist-ec2`` under a specific profile
with the same name:

.. code:: bash

   $ aws configure --profile tf-fmnist-ec2

After adding the access key ID and the secret access key, you will be asked
for a region (``<AWS_REGION>``) and a default output format - in my case
``eu-central-1`` and ``json``.
A list of the regional endpoints and their names can be found
`here <https://docs.aws.amazon.com/general/latest/gr/rande.html>`_.

Profiles with a specific name, i.e. the non-default profiles, can be
used in ``aws`` commands like this:

.. code:: bash

   $ aws s3 ls --profile <PROFILE>

The profile information is stored in ``~/.aws/credentials`` and
``~/.aws/config``. More information on named profiles is given `here
<https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html>`_.



Uploading the *Docker* Image to *ECR*
-------------------------------------

- `Using Amazon ECR with the AWS CLI
  <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_

On :term:`ECR`, create a private or public repository named
``tf-serving-fmnist`` and enable *Scan on push*.
The commands needed to push images to this repository
can be displayed in *ECR* by selecting the repository and clicking on
*View push commands*.

Next, we have to authenticate *Docker* on our local machine to this repository:

.. code:: bash

   $ aws ecr get-login-password --profile tf-fmnist-ec2 --region <AWS_REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com

Tag the image and push it to the ``tf-serving-fmnist`` repository on *ECR*:

- list the images

  .. code:: bash

     $ docker images

  .. code-output::

     REPOSITORY           TAG       IMAGE ID       CREATED        SIZE
     tf-serving-fmnist    latest    f34fefc2ee4c   27 hours ago   411MB
     tensorflow/serving   latest    e874bf5e4700   5 weeks ago    406MB

- tag the image

  .. code:: bash

     $ docker tag tf-serving-fmnist:latest <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist:latest

- push the image

  .. code:: bash

     $ docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist:latest

  .. code-output::

     The push refers to repository [<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist]
     75d124fb0170: Pushed 
     bb4423850a27: Pushed 
     b60ba33781cd: Pushed 
     547f89523b17: Pushed 
     bd91f28d5f3c: Pushed 
     8cafc6d2db45: Pushed 
     a5d4bacb0351: Pushed 
     5153e1acaabc: Pushed 
     latest: digest: sha256:8807f835d9beadfa22679630bcd75f9555695272245b11201bc39f9c8c55d6e0 size: 1991

The latest image from this :term:`ECR` repository can be pulled with

.. code:: bash

   $ docker pull <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist:latest

as described `here
<https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-pull-image>`_.



Serving the Model on *EC2*
--------------------------

Now, we are going to create an :term:`EC2` instance based on the
`Deep Learning Base AMI
<https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html>`_.

In the *EC2* management console, go to *Network & Security > Key Pairs*,
create a key pair called ``tf-fmnist-ec2`` and download it as a ``*.pem``-file.

Then, go to *Instances* and click on *Launch instances*.
In this tutorial, we are using the
`AWS Deep Learning AMI (Ubuntu 18.04)
<https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y>`_
from the AWS Marketplace.

Next, we have to choose an instance type. I have been running on a
``t2.medium`` which comes with 2 vCPUs, 4 GiB of memory and 'low to
moderate' network performance, which is more than enough for testing.
Also, it has storage on *EBS*, which can be configured to persist
after an *EC2* instance is terminated. To achieve this, when launching
an *EC2* instance, we have to deselect *Delete on Termination* in
*Step 4: Add Storage*.

In *Step 5: Add Tags*, you can add a tag with

- **key**: Name
- **value**: tf-fmnist-ec2

In *Step 6: Configure Security Group*, create a new security group with
the name ``tf-fmnist-ec2`` and description *SSH and TCP:8501 HTTP REST access*
and add the following rule:

- **type**: custom TCP
- **protocol**: TCP
- **port range**: 8501
- **source**: anywhere
- **description**: tensorflow-serving predict REST API

As the description suggests, opening port 8501 is necessary to allow
internet traffic requesting a prediction from the model via
*TensorFlow Serving*'s REST API to reach the *EC2* instance.

After confirming the configuration in *Step 7: Review Instance Launch* by
clicking on *Launch*, a dialog window will ask you to either select an existing
key pair or create a new one. Here we can choose the existing key pair
``tf-fmnist-ec2``. This concludes the instance configuration and will spin
up the requested server.

However, as long as no internet gateway is configured in *VPC*, the instance
cannot be reached via SSH or otherwise. Therefore, in the *VPC* management
console go to *Virtual Private Cloud > Internet Gateways*, click on
*Create internet gateway* and set the name tag to ``tf-fmnist-ec2``.
Once it is created, attach it to the existing VPC that is used by the
*EC2* instance: *Actions > Attach to VPC*. Next, in the *VPC* management
console, go to *Virtual Private Cloud > Route Tables*, select the appropriate
one and check that in the *Routes* tab, there is a route with the newly
created internet gateway as a target. If this is not the case, click on
*Edit routes* and add a route with the following configuration:

- **destination**: 0.0.0.0/0
- **target**: the newly created internet gateway

Now, if the instance has started and is marked as *running*, we can connect
to it from the terminal via *SSH* as described `here
<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html#AccessingInstancesLinuxSSHClient>`_:

.. code:: bash

   $ ssh -i /path/to/tf-fmnist-ec2.pem ubuntu@<Public_IPv4_DNS>

The command with the adequate IP address can be displayed in the
*EC2 management console* by selecting the instance and clicking on
*Actions > Connect > SSH client*.

.. note::

   In case of connectivity issues, be it reaching the instance via SSH
   or querying the deployed model via *TensorFlow Serving*, the
   *VPC Reachability Analyzer* is a very helpful tool. In the *VPC*
   management console, go to *Reachability > Reachability Analyzer*
   and click on *Create and analyze path* to configure an analysis.
   If the connection fails, the error message will give hints as to
   what should be fixed or set up differently.

When connected to the *EC2* instance, update the *AWS CLI* as described
`here
<https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-upgrade>`_
and then run ``aws configure``

.. code:: bash

   (ec2) $ aws configure --profile tf-fmnist-ec2

and specify the credentials of the ``tf-fmnist-ec2`` *IAM* user.

We can then pull the *Docker* image from *ECR* as described `here
<https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-pull-image>`_.
But before, we have to authenticate to the *ECR* repository - otherwise
we will get the ``no basic auth credentials`` error from *Docker*.
Since we are not on a tty console, the ``aws ecr ... | docker login ...``
command from before will not work. Instead, we can do

.. code:: bash

   (ec2) $ docker login -u AWS -p $(aws ecr get-login-password --profile tf-fmnist-ec2 --region <AWS_REGION>) <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com

as suggested `here <https://stackoverflow.com/a/61854312>`_.

Now we can pull the image with

.. code:: bash

   (ec2) $ docker pull <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist:latest

.. code-output::

   latest: Pulling from tf-serving-fmnist
   01bf7da0a88c: Pull complete 
   f3b4a5f15c7a: Pull complete 
   57ffbe87baa1: Pull complete 
   e72e6208e893: Pull complete 
   6ea3f464ef73: Pull complete 
   01e9bf86544b: Pull complete 
   68f6bba3dc50: Pull complete 
   dafe84328936: Pull complete 
   Digest: sha256:8807f835d9beadfa22679630bcd75f9555695272245b11201bc39f9c8c55d6e0
   Status: Downloaded newer image for 009984474629.dkr.ecr.eu-central-1.amazonaws.com/tf-serving-fmnist:latest
   009984474629.dkr.ecr.eu-central-1.amazonaws.com/tf-serving-fmnist:latest

show the list of all images with

.. code:: bash

   (ec2) $ docker images

.. code-output::

   REPOSITORY                                                              TAG       IMAGE ID       CREATED      SIZE
   <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist   latest    f34fefc2ee4c   2 days ago   411MB

and run it with the same command already used in the previous tutorial
:doc:`/_examples/3-tensorflow-serving-docker/code`:

.. code:: bash

   (ec2) $ docker run -p 8501:8501 -e MODEL_NAME=fmnist-model -t --name tf-serving-fmnist <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/tf-serving-fmnist

.. code-output::

   [...]

   2021-06-26 09:12:09.808270: I tensorflow_serving/model_servers/server.cc:414] Exporting HTTP/REST API at:localhost:8501 ...
   [evhttp_server.cc : 245] NET_LOG: Entering the event loop ...

Now we can query the deployed model with an HTTP request to the REST API
like in the last tutorial :doc:`/examples/3-tensorflow-serving-docker/code`.
For this, the script ``1-request`` is provided in
``examples/4-tf-serving-docker-aws/``. In it, set the ``IPv4`` variable to
the appropriate address for your *EC2* instance.
Download a few example images of different items of clothing from the
internet and include their filenames in the script.


.. margin:: from **1-request.py**

   in ``examples/4-tf-serving-docker-aws/``

.. code:: python

   def load_images(images: Union[str, List[str]]):

       if isinstance(images, str): images = [images]

       array = np.empty(shape=(len(images), 28, 28, 1))
       for i, image in enumerate(images):
           img = tf.keras.preprocessing.image.load_img(
               path = image,
               color_mode = 'grayscale',
               target_size = (28, 28))

           array[i] = tf.keras.preprocessing.image.img_to_array(img=img)

       return (255 - array) / 255


   tshirts = [
       'images/test/tshirt-1.jpg', 'images/test/tshirt-2.png',
       'images/test/tshirt-3.jpg', 'images/test/tshirt-4.png'
   ]
   sandals = [f'images/test/sandal-{i}.jpg' for i in range(1, 5)]
   sneakers = [f'images/test/sneaker-{i}.jpg' for i in range(1, 5)]

   # test_images = load_images(tshirts)
   test_images = load_images(sandals)
   # test_images = load_images(sneakers)

   IPv4 = 'ec2-54-93-96-215.eu-central-1.compute.amazonaws.com'
   # IPv4 = '54.93.96.215'


   data = json.dumps({
       'signature_name': 'serving_default',
       'instances': test_images.tolist()   # *either* 'instances'
       # 'inputs': test_images.tolist()    # *or* 'inputs'
   })

   headers = {'content-type': 'application/json'}
   json_response = requests.post(
       f'http://{IPv4}:8501/v1/models/fmnist-model:predict',
       headers=headers,
       data=data
   )

   predictions = json.loads(json_response.text)['predictions'] # for 'instances'
   # predictions = json.loads(json_response.text)['outputs']   # for 'inputs'

   for i, prediction in enumerate(predictions):
       print('prediction {}: {} -> predicted class: {}'.format(
           i, prediction, np.argmax(prediction)))


Runing the script on a few example pictures of sandals yields

.. code:: bash

   $ python 1-request.py 

.. code-output::

   prediction 0: [6.30343275e-05, 1.85037115e-05, 7.5565149e-06, 2.83105837e-05, 1.74145697e-07, 0.998879, 3.21875377e-05, 2.88182633e-07, 2.93241665e-05, 0.000941563747] -> predicted class: 5
   prediction 1: [0.00130418374, 2.53213402e-05, 3.95829147e-06, 7.95430169e-05, 1.08047243e-06, 0.975131869, 0.000146661841, 0.0195651725, 2.74548784e-05, 0.00371479546] -> predicted class: 5
   prediction 2: [0.000241088521, 3.73763069e-05, 1.56952246e-05, 0.000210506681, 5.54936341e-05, 0.992369056, 9.06738e-05, 0.00564925186, 0.00128137472, 4.94648739e-05] -> predicted class: 5
   prediction 3: [9.74363502e-06, 7.06914216e-06, 6.15942408e-05, 0.00042412043, 0.000195012341, 0.910350859, 1.11019081e-05, 0.0710703135, 0.0176780988, 0.000191997562] -> predicted class: 5

so all example images were correctly classified as sandals.
