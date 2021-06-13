**********
TensorFlow
**********


.. toctree::
   :maxdepth: 2
   :hidden:

   tf-save



Layers
======

Module: `tf.keras.layers <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_

- InputLayer: ``tf.keras.layers.InputLayer(input_shape=(10,))``
- Dense: ``tf.keras.layers.Dense(units, activation=None)``
- Flatten: ``tf.keras.layers.Flatten()``
- Dropout: ``tf.keras.layers.Dropout(rate)``
- Conv2D: ``tf.keras.layers.Conv2D(filters, kernel_size, activation=None, input_shape)``

  - ``input_shape`` is expected to be a 3D tuple, with the last dimension being the colour channels, e.g. ``(28,28,3)`` for 28 x 28 pixels and 3 colour channels (RGB). An alpha channel would probably be a fourth channel?

- MaxPooling2D: ``tf.keras.layers.MaxPooling2D(pool_size=(2, 2))``


Activation Functions
====================

Module: `tf.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_

The activation functions are also available as individual layers, e.g. when you would like the model to output logits and would like to build a separate probability model with an additional softmax layer appended.

- ReLu: ``"relu"``, ``tf.keras.activations.relu``
- Softmax: ``"softmax"``, ``tf.keras.activations.softmax``
- Sigmoid: ``"sigmoid"``, ``tf.keras.activations.sigmoid``

  - Normally, the number of output neurons should match the number of classes
    in a classification problem. One exception is binary classification, where
    it is possible to use one single output neuron with sigmoid activation.
   

Optimizers
==========

Module: :mod:`tf.keras.optimizers`

- SGD: ``optimizer='sgd'``, ``tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)``
- Adam: ``optimizer='adam'``, ``tf.keras.optimizers.Adam(learning_rate=0.001)``
- RMSProp: ``tf.keras.optimizers.RMSprop(learning_rate=0.001)``


Loss Functions
==============

Module: `tf.keras.losses <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_

- Binary Crossentropy: ``tf.keras.losses.BinaryCrossentropy(from_logits=False)``
- Categorical Crossentropy: ``tf.keras.losses.CategoricalCrossentropy(from_logits=False)``
- Sparse Categorical Crossentropy: ``tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)``
- Mean Squared Error: ``tf.keras.losses.MeanSquaredError``


Metrics
=======

- ``"accuracy"``


Inspection / Information about the Model
========================================

- ``model.summary()``
- ``model.input_shape``
- ``model.output_shape``
- ``model.layers``


Regularisation
==============

Module: :mod:`tf.keras.regularizers`

- L1 / lasso: ``kernel_regularizer='l1'``, :class:`tf.keras.regularizers.L1`
- L2 / ridge: ``kernel_regularizer='l2'``, :class:`tf.keras.regularizers.L2`
- L1L2: ``kernel_regularizer='l1_l2'``, :class:`tf.keras.regularizers.L1L2`


Treatment of Inputs
===================

- `Load NumPy data <https://www.tensorflow.org/tutorials/load_data/numpy>`_
- `Load a pandas.DataFrame <https://www.tensorflow.org/tutorials/load_data/pandas_dataframe>`_
- `Dataset.shuffle <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle>`_
- Split into training and testing datasets

  - `Dataset.take <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take>`_
  - `Dataset.skip <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip>`_

- `Dataset.batch <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch>`_



Links & Resources
=================

- `Google Machine Learning Glossary <https://developers.google.com/machine-learning/glossary>`_

   