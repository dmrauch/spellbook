'''
***************************************
Serving *TensorFlow* Models in *Docker*
***************************************

.. raw:: html

    <div class="tag-list">
       <div class="tag-cell tag-date">June 21, 2021</div>
       <div class="tag-cell tag-center"></div>
       <div class="tag-cell tag-right">
          <span class="tag-text">tags:</span>
          <a class="tag right" href="../../search.html?q=tags+CNN">
             CNN</a>
          <a class="tag right" href="../../search.html?q=tags+Docker">
             Docker</a>
          <a class="tag right" href="../../search.html?q=tags+image+augmentation">
             image augmentation</a>
          <a class="tag right" href="../../search.html?q=tags+image+classification">
             image classification</a>
          <a class="tag right" href="../../search.html?q=tags+multi+class+classification">
             multi-class classification</a>
        </div>
    </div>


.. admonition:: In this project/tutorial, we will
   :class: spellbook-admonition-orange
 
   - Train a **convolutional neural network** (**CNN**) to do
     **multi-class classification** on Zalando's **Fashion-MNIST** dataset
   - Use **image augmentation** to make the model more general
   - Serve the model with *TensorFlow Serving* in a **Docker container**

The source code files for this tutorial are located in
``examples/3-tensorflow-serving-docker/``.

In the first sections, we will have a look at the *Fashion-MNIST* dataset and
set up and train a convolutional neural network. If you are just interested
in the part about serving a model in *Docker*, please skip ahead to the
final section :ref:`ex3-serving-docker`.
'''


# %%
# The *Fashion-MNIST* Dataset
# ===========================
# 
# Zalando's `Fashion-MNIST
# <https://www.kaggle.com/zalando-research/fashionmnist>`_ dataset is one of
# the standard benchmarking datasets in computer vision, designed to be a
# drop-in replacement for the original *MNIST* dataset of handwritten digits.
# *Fashion-MNIST* consists of greyscale images of different items of clothing.
# It includes 60 000 images for training and 10 000 for validation and testing,
# 28 pixels in height and 28 pixels in width, divided into 10 classes
# indicating the type of clothing:
# 
# - **0**: t-shirt, top
# - **1**: trouser
# - **2**: pullover
# - **3**: dress
# - **4**: coat
# - **5**: sandal
# - **6**: shirt
# - **7**: sneaker
# - **8**: bag
# - **9**: ankle boot
# 
# *TensorFlow* and *Keras* provide the *Fashion-MNIST* dataset as
# :class:`numpy.ndarray`\s containing the 28x28 pixel values and the labels.
# They can be loaded with
#
# .. margin:: from **1-plot.py**
#
#    in ``examples/3-tensorflow-serving-docker/``

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (val_images, val_labels) \
    = fashion_mnist.load_data()

print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape)


# %%
# Each image is a 28x28 array of values between 0 and 255, with the background
# filled with zeros and the object itself given with values larger than 0:
#
# .. margin:: from **1-plot.py**
#
#    in ``examples/3-tensorflow-serving-docker/``

import numpy as np
print(np.array2string(train_images[0], max_line_width=150))


# %%
# While it is possible to use just these arrays to train a neural network
# classifier, let's go a step further and use
# :class:`tf.keras.preprocessing.image.ImageDataGenerator` for creating
# a flow of images for training and validation.
# With :class:`tf.keras.preprocessing.image.ImageDataGenerator` it is 
# possible to flow images from :class:`numpy.ndarray`\s, but also to load
# and flow them from directories or :class:`pandas.DataFrame`\s containing
# the image filepaths. At the same time,
# :class:`tf.keras.preprocessing.image.ImageDataGenerator` is able to
# prepare the labels, batch the images and the labels as well as apply
# :term:`image augmentation`. In image augmentation, transformations are
# applied to the images, including, but not limited to
#
# - horizontally or vertically flipping
# - rotating
# - zooming
# - shearing
#
# them randomly within configurable ranges where applicable.
# The randomness and the effective increase in the number of training
# images reduce the likelihood of overtraining and the widened range of
# positions, orientations and appearances of the objects in the images
# can help make the model more applicable to a larger number of images
# in inference.
# We can set up the generator and obtain the flow iterator with
#
# .. margin:: from **helpers.py**
#
#    in ``examples/3-tensorflow-serving-docker/``

import numpy as np

def get_generator(images, labels, batch_size=32, **kwargs):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        rescale=1/255)

    gen = datagen.flow(
        x = np.expand_dims(images, axis=3),
        y = tf.keras.utils.to_categorical(labels, num_classes=10),
        batch_size = batch_size,
        **kwargs
    )

    return gen


# %%
# Note that we also rescaled the images with a factor of 1/255 to restrict
# the values to the range between 0 and 1. This is a customary input
# normalisation since neural networks generally perform better with data
# of the order of one.
# We also turned the simple integer class labels into one-hot encoded
# label vectors using :func:`tf.keras.utils.to_categorical`.
#
# We can then instantiate a generator, e.g. for the training set, and
# retrieve the first batch of augmented images
#
# .. margin:: from **1-plot.py**
#
#    in ``examples/3-tensorflow-serving-docker/``

import helpers

n = 7

train_gen = helpers.get_generator(train_images, train_labels, batch_size=n,
    shuffle=False)

train_images_augmented = next(train_gen)


# %%
# Here, we set ``shuffle=False`` because we want to preserve the order of
# the images for comparing the augmented to the original images.
# So let's proceed to plot a few images in two rows - the original images
# in the upper row and the corresponding augmented images in the lower
# row. The result is shown in Figure 1.
#
# .. margin:: from **1-plot.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    import matplotlib as mpl
#    import matplotlib.pyplot as plt
#    import spellbook as sb
#    
#    fig = plt.figure(figsize=(7,2))
#    grid = mpl.gridspec.GridSpec(nrows=2, ncols=n, wspace=0.1, hspace=0.1)
#    
#    for i in range(n):
#        ax = plt.Subplot(fig, grid[0,i])
#        fig.add_subplot(ax)
#        ax.set_axis_off()
#        ax = plt.imshow(train_images[i], cmap='gray')
#    
#        ax = plt.Subplot(fig, grid[1,i])
#        fig.add_subplot(ax)
#        ax.set_axis_off()
#        ax = plt.imshow(train_images_augmented[0][i], cmap='gray')
#    
#    sb.plot.save(fig, 'images.png')
#
# .. figure:: /images/examples/3-tensorflow-serving-docker/images.png
#
#    Figure 1: The first few training images before and after augmentation
#
# As we can see, the clothes are flipped and rotated as specified.


# %%
# Training a Neural Network Classifier
# ====================================
#
# We are now going to set up a neural network for classifying the
# *Fashion-MNIST* images according to their respective labels.
#
# .. margin:: from **2-train.py**
#
#    in ``examples/3-tensorflow-serving-docker/``

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu',
        input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
model.summary()


# %%
# The networks begins with a 2D convolutional layer (:term:`CNN`)
# with 30 filters of 3x3 pixels each, followed by a max-pooling layer.
# Since each filter has 3x3=9 pixels and a bias, the total of 30 filters
# correspond to 300 trainable parameters. Sliding a 3x3 pixel filter across
# 28x28 pixel images yields 26x26 pixel images and applying 2x2 pixel
# max-pooling cuts the image size down to 13x13 pixels.
# The flattening layer turns the 2D pixel arrays into a vector and
# feeds them to the final part of the network, consisting of a dense
# layer with 50 nodes and the output layer.
# Since we turned the labels into one-hot encoded label vectors, we
# use a dense output layer with 10 nodes, i.e. one node per class,
# and :term:`softmax` activation. This ensures that the sum of all 10
# outputs of the last layer sum up to unity, which at least numerically
# corresponds to properties expected from discreet probabilities. However,
# as long as a classifier is not calibrated, one cannot be sure that its
# outputs really give the probabilities for the different classes.
#
# This network is deliberately simple because the main focus of this
# tutorial is on serving the model in *Docker* rather than achieving the
# best possible performance. Improved performance can be achieved by
# adding more filters, more convolutional layers, followed by a larger
# dense network, while paying attention to overtraining and keeping it
# in check, e.g. by using :term:`dropout` layers.
#
# Instead of pursuing this approach and the correspondingly larger
# computational complexity, we will keep it fast and simple and proceed to
# configure the model with appropriate loss and metrics and finally train
# it for only just 3 epochs:
#
# .. margin:: from **2-train.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    model.compile(
#        loss = 'categorical_crossentropy',
#        optimizer = tf.keras.optimizers.Adam(),
#        metrics = [
#            tf.keras.metrics.CategoricalCrossentropy(),
#            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
#        ]
#    )
#
#    epochs = 3
#    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
#        callbacks = [
#            tf.keras.callbacks.CSVLogger(filename='fmnist-model-history.csv'),
#            sb.train.ModelSavingCallback(foldername='fmnist-model')
#        ]
#    )
#
# .. code-output::
#
#    Epoch 1/3
#    938/938 [==============================] - 56s 59ms/step - loss: 0.5996 - categorical_crossentropy: 0.5996 - accuracy: 0.7853 - val_loss: 0.4732 - val_categorical_crossentropy: 0.4732 - val_accuracy: 0.8255
#    Epoch 2/3
#    938/938 [==============================] - 55s 58ms/step - loss: 0.4286 - categorical_crossentropy: 0.4286 - accuracy: 0.8444 - val_loss: 0.4286 - val_categorical_crossentropy: 0.4286 - val_accuracy: 0.8442
#    Epoch 3/3
#    938/938 [==============================] - 55s 58ms/step - loss: 0.3829 - categorical_crossentropy: 0.3829 - accuracy: 0.8601 - val_loss: 0.4052 - val_categorical_crossentropy: 0.4052 - val_accuracy: 0.8517
#    2021-06-20 13:17:43.055082: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
#    Training finished after 3 epochs: Saved model to folder 'fmnist-model'</pre>
#
# ``'categorical_crossentropy'`` sets an instance of
# :class:`tf.keras.losses.CategoricalCrossentropy` configured with the
# default values as the loss function. This is the appropriate loss function
# when using one-hot encoded labels. Lkewise, we are using the
# :class:`tf.keras.metrics.CategoricalCrossentropy` metric.
# Finally, the model is trained with ``model.fit``, passing instances
# of :class:`tf.keras.callbacks.CSVLogger` for saving the metrics to a
# ``*.csv``-file during training and
# :class:`spellbook.train.ModelSavingCallback` for saving the model
# during and at the end of the training. As we can see from the ouput, a
# classification accuracy of about 85% is achieved during validation, which
# doesn't seem too great, but is enough for our purposes in this tutorial.



# %%
# Evaluating the Model
# ====================
#
# But before we go on, let's have just a slightly closer look at the
# model's performance. We begin by loading the saved model, preparing an
# image generator for the validation dataset and use the model to calculate
# the predictions:
#
# .. margin:: from **3-evaluate.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    model = tf.keras.models.load_model('fmnist-model')
#    val_gen = helpers.get_generator(val_images, val_labels, batch_size, shuffle=False)
#    val_predictions = model.predict(val_gen)
#    val_predicted_labels = np.argmax(val_predictions, axis=1)
#
# Again we use ``shuffle=False`` so that the order of the predictions
# corresponds to the order of the orginal labels in ``val_labels``.
#
# We can then go on to determine and plot the confusion matrix with
# :func:`spellbook.plot.plot_confusion_matrix` - one version
# with the absolute datapoint counts given in Figure 2 and one version
# normalised across each true class in Figure 3:
#
# .. margin:: from **3-evaluate.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    class_ids = list(helpers.label_dict.keys())
#    class_names = list(helpers.label_dict.values())
#    val_confusion = tf.math.confusion_matrix(
#        val_labels, val_predicted_labels, num_classes=len(class_ids))
#    
#    sb.plot.save(
#        sb.plot.plot_confusion_matrix(
#            confusion_matrix = val_confusion,
#            class_names = class_names,
#            class_ids = class_ids,
#            fontsize = 9.0,
#            fontsize_annotations = 'x-small'
#        ),
#        filename = 'fmnist-model-confusion.png'
#    )
#    sb.plot.save(
#        sb.plot.plot_confusion_matrix(
#            confusion_matrix = val_confusion,
#            class_names = class_names,
#            class_ids = class_ids,
#            normalisation = 'norm-true',
#            crop = False,
#            figsize = (6.4, 4.8),
#            fontsize = 9.0,
#            fontsize_annotations = 'x-small'
#        ),
#        filename = 'fmnist-model-confusion-norm-true.png'
#    )
#
# .. list-table::
#    :class: spellbook-gallery-wrap
#
#    * - .. figure:: /images/examples/3-tensorflow-serving-docker/fmnist-model-confusion.png
#           :height: 350px
#
#           Figure 2: Absolute datapoint counts
#
#      - .. figure:: /images/examples/3-tensorflow-serving-docker/fmnist-model-confusion-norm-true.png
#           :height: 350px
#
#           Figure 3: Relative frequencies normalised in each true category
#
# We can see that by and large at least 75% of the items of each category are
# correctly classified, except for shirts which are most often confused with
# t-shirts/tops (15.8%) and to a lesser extent coats (8.7%) and pullovers
# (7.4%).



# %%
# Making Predictions About Other Pictures
# =======================================
#
# While evaluating and benchmarking a model's performance is still part of
# the development process, the eventual interest is of course in deploying
# and using the model in production to obtain the predictions for images
# that are not part of the training and validation sets.
# To simulate this, I downloaded a few random images of different pieces
# of clothing and loaded them into :class:`numpy.ndarray`\s using
# :func:`tf.keras.preprocessing.image.load_img` and
# :func:`tf.keras.preprocessing.image.img_to_array`.
#
# .. margin:: from **4-predict.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    import numpy as np
#    import tensorflow as tf
#    
#    import helpers
#    
#    
#    model = tf.keras.models.load_model('fmnist-model')
#    
#    test_images = helpers.load_images(helpers.tshirts)
#    # test_images = helpers.load_images(helpers.sandals)
#    # test_images = helpers.load_images(helpers.sneakers)
#
#
# When loading the images, it is important to size the arrays
# in accordance with the model's architecture and the images used during
# training and validation. Therefore, in this example, we choose a target
# size of 28x28 pixels and use the ``'grayscale'`` colour mode:
#
# .. margin:: from **helpers.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    tshirts = [
#        'images/test/tshirt-1.jpg', 'images/test/tshirt-2.png',
#        'images/test/tshirt-3.jpg', 'images/test/tshirt-4.png'
#    ]
#    sandals = [f'images/test/sandal-{i}.jpg' for i in range(1, 5)]
#    sneakers = [f'images/test/sneaker-{i}.jpg' for i in range(1, 5)]
#
#
#    def load_images(images: Union[str, List[str]]):
#    
#        if isinstance(images, str): images = [images]
#    
#        array = np.empty(shape=(len(images), 28, 28, 1))
#        for i, image in enumerate(images):
#            img = tf.keras.preprocessing.image.load_img(
#                path = image,
#                color_mode = 'grayscale',
#                target_size = (28, 28))
#    
#            array[i] = tf.keras.preprocessing.image.img_to_array(img=img)
#    
#        return (255 - array) / 255
#
#
# Once, the images are loaded, the ``model.predict`` function can be used
# to apply the model to the data:
#
# .. margin:: from **4-predict.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    predictions = model.predict(test_images)
#    for prediction in predictions:
#        print('prediction:', prediction,
#            '-> predicted class:', np.argmax(prediction))
#
# .. code-output::
#
#    prediction: [5.1984423e-01 4.8716479e-06 2.3578823e-04 4.8502727e-04 7.7355535e-06
#     1.2823233e-06 4.7876367e-01 1.8715612e-06 6.5485924e-04 6.3797620e-07] -> predicted class: 0
#    prediction: [2.1379247e-02 2.8888881e-04 5.3068418e-03 4.2415579e-04 2.2449503e-05
#     5.5147725e-04 5.9862167e-02 2.1699164e-07 9.1194308e-01 2.2140094e-04] -> predicted class: 8
#    prediction: [8.70128453e-01 1.32829140e-04 1.44031774e-02 2.06211829e-04
#     8.69949628e-03 4.22267794e-06 1.03566416e-01 5.67484494e-05
#     2.79841595e-03 4.09945960e-06] -> predicted class: 0
#    prediction: [9.6439028e-01 7.7955298e-07 1.6881671e-02 6.1920066e-03 2.3770966e-05
#     5.2960712e-07 1.2419004e-02 1.9606419e-09 9.1841714e-05 1.0729976e-09] -> predicted class: 0
#
# As we can see, three of the four t-shirt images are correctly classified,
# while the second one is mistaken for a bag - which is perhaps as much as
# can be expected with such a simple and only extremely briefly trained
# model.



# %%
# .. _ex3-serving-docker:
#
# Serving the Model in *Docker*
# =============================
#
# Finally, let's see how we can serve the model inside
# a *Docker* container using *TensorFlow Serving*.
# To do that, first install *Docker* and get the ``tensorflow/serving``
# image
#
# .. code:: bash
#
#    $ docker pull tensorflow/serving
#
# from `Docker Hub <https://hub.docker.com/r/tensorflow/serving>`_.
# When started, a container created from this image will run
# ``tensorflow_model_server`` and expose the REST API on port 8501
# The model to serve can be specified via the ``MODEL_NAME`` and
# ``MODEL_BASE_PATH`` environment variables.
# By default, ``MODEL_BASE_PATH=/models`` and ``MODEL_NAME=model``.


# %%
# Creating a Custom Image for the Model
# -------------------------------------
#
# To serve our own model, we first need to create a custom *Docker* image
# based on the ``tensorflow/serving`` image.
# First, create a container from the ``tensorflow/serving`` image and start it:
#
# .. code:: bash
#
#    $ sudo docker run -d --name tf-serving-base tensorflow/serving
#
# .. code-output::
#
#    4f9109df18bcace745d108dd1fba0659b65db62e5f4104f99ca4ed5536d194c6
#
# This creates a container named ``tf-serving-base`` and returns the
# container ID. We can verify that the container is running with
#
# .. code:: bash
#
#    $ sudo docker ps
#
# .. code-output::
#
#    CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS           NAMES
#    4f9109df18bc   tensorflow/serving   "/usr/bin/tf_serving…"   2 minutes ago   Up 2 minutes   8500-8501/tcp   tf-serving-base
#
# Next, we have to create a folder for the model inside the container.
# This folder will later hold one subfolder for each version of the model
# so that the model can be seamlessly updated.
#
# .. code:: bash
#
#    $ sudo docker exec tf-serving-base mkdir /models/fmnist-model
#
# Now we can copy the saved model over to the container, creating the first
# version subfolder at the same time:
#
# .. code:: bash
#
#    $ sudo docker cp fmnist-model tf-serving-base:/models/fmnist-model/1
#
# Instead of copying the model, we can also mount it into the container
# from the host filesystem when we start the container. While this will work
# when developing locally, it is of course not a good idea when the container
# is to be sent elsewhere.
#
# .. note::
# 
#    If no version folder is created and the model is copied or mounted into
#    the model folder (``/models/fmnist-model/``) directly, the model cannot
#    be served and the following message will be shown:
#
#    .. code-output::
#
#       2021-06-17 19:17:06.689731: W tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc:268] No versions of servable fmnist-model found under base path /models/fmnist-model. Did you forget to name your leaf directory as a number (eg. '/1/')?
#
#
# Once this is done, we can create a new image ``tf-serving-fmnist``
# from this container, specifying the name of the model folder as the
# environment variable ``MODEL_NAME``
#
# .. code:: bash
#
#    $ sudo docker commit --change "ENV MODEL_NAME fmnist-model" tf-serving-base tf-serving-fmnist
#
# .. code-output::
#
#    sha256:f34fefc2ee4ccc5d0b8a9ab756a48062fe23bcd4bc493b1fe165f66d7bfd3318
#
# and double-check the list of images
#
# .. code:: bash
#
#    $ sudo docker images
#
# .. code-output::
#
#    REPOSITORY           TAG       IMAGE ID       CREATED          SIZE
#    tf-serving-fmnist    latest    f34fefc2ee4c   57 seconds ago   411MB
#    tensorflow/serving   latest    e874bf5e4700   5 weeks ago      406MB
#
# Finally, we can stop the ``tf-serving-base`` container by doing
#
# - either
#
#   .. code:: bash
#
#      $ sudo docker stop tf-serving-base
#
# - or
#
#   .. code:: bash
#
#      $ sudo docker kill tf-serving-base
#
# which can readily be verified with ``sudo docker ps``.


# %%
# Serving and Querying Our Own Model
# ----------------------------------
#
# We can create a container from the ``tf-serving-fmnist`` image and run it
# with
#
# .. code:: bash
#
#    $ sudo docker run -p 8501:8501 -e MODEL_NAME=fmnist-model -t tf-serving-fmnist
#
# In case we didn't copy the model into the container, we have to mount it
# from the host filesystem with
#
# .. code:: bash
#
#    $ sudo docker run -p 8501:8501 \
#          --mount \
#              type=bind,\
#              source=/home/daniel/Computing/Programming/spellbook/examples/3-model-internal-image-preprocessing-pipeline/fmnist-model/,\
#              target=/models/fmnist-model/1 \
#          -e MODEL_NAME=fmnist-model \
#          -t \
#          tf-serving-fmnist
#
# We can then query the model and obtain its predictions for some images by
# means of a ``POST`` request from the command line using
# ``curl`` or from a python script using the ``requests`` module/library.
# The example script ``5-request.py`` does both - it prints out a ``curl``
# command that can be copied, pasted and run in a terminal and it also submits
# a request directly from the *Python* code and prints out the resulting
# predictions:
#
# .. margin:: from **5-request.py**
#
#    in ``examples/3-tensorflow-serving-docker/``
#
# .. code:: python
#
#    import json
#    import numpy as np
#    import requests
#    
#    import helpers
#    
#    
#    test_images = helpers.load_images(helpers.tshirts)
#    # test_images = helpers.load_images(helpers.sandals)
#    # test_images = helpers.load_images(helpers.sneakers)
#    
#    data = json.dumps({
#        'signature_name': 'serving_default',
#        'instances': test_images.tolist()   # *either* 'instances'
#        # 'inputs': test_images.tolist()    # *or* 'inputs'
#    })
#    
#    print('------------------------------------------------------------')
#    print("for querying the served model from the terminal with 'curl',"
#          " use the following command\n")
#    print("curl -d '{}' -X POST {}".format(
#        data, 'http://localhost:8501/v1/models/fmnist-model:predict'))
#    print('\n------------------------------------------------------------')
#    
#    
#    headers = {'content-type': 'application/json'}
#    json_response = requests.post(
#        'http://localhost:8501/v1/models/fmnist-model:predict',
#        headers=headers,
#        data=data
#    )
#    
#    print(json_response.text)
#    predictions = json.loads(json_response.text)['predictions'] # for 'instances'
#    # predictions = json.loads(json_response.text)['outputs']   # for 'inputs'
#    for i, prediction in enumerate(predictions):
#        print('prediction {}: {} -> predicted class: {}'.format(
#            i, prediction, np.argmax(prediction)))
#
# For using a specific version of the model, e.g. version 2, the URL
# ``http://localhost:8501/v1/models/fmnist-model/versions/2:predict``
# can be used.
#
# Once it has been created, the container can be stopped and started again
# with
#
# .. code:: bash
#
#    $ sudo docker stop <CONTAINER-ID>
#
# and
#
# .. code:: bash
#
#    $ sudo docker start <CONTAINER-ID>
#
# If a name was specified for the container with ``--name <NAME>``
# when creating it with the ``docker run`` command, then this name can be
# used to refer to the container instead of the ID.
#
#
# .. rubric:: Links
#
# - https://www.tensorflow.org/tfx/serving/docker
# - https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image
# - https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#make_a_request_to_your_model_in_tensorflow_serving
# - https://www.tensorflow.org/tfx/serving/api_rest#predict_api
#