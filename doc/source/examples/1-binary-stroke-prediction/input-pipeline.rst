.. _BinaryStrokePrediction-InputPipeline:

***********************************
Data Preparation and Input Pipeline
***********************************


Now let's move on to bringing the data into a form that can be processed by
the neural network!

We'll have a look at the three scripts

- ``2-stroke-prediction-naive.py``
- ``3-stroke-prediction-oversampling.py``
- ``4-stroke-prediction-oversampling-norm.py``

in ``examples/1-binary-stroke-prediction/``.



Treatment of Categorical Variables
==================================


After data loading and cleaning, we start off by processing the categorical
variables and manipulating them so that they can be digested by the network.

.. margin:: from **2-stroke-prediction-naive.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    # inplace convert string category labels to numerical indices
    categories = sb.inputs.encode_categories(data)

Behind the scenes, the function :func:`spellbook.inputs.encode_categories`
loops over all categorical variables in the dataset and converts them to
:class:`pandas.Categorical`\s.
A dictionary containing the mapping of the category names to numerical indices
for each categorical variable is returned::

    categories: {
        'gender': {0: 'female', 1: 'male', 2: 'other'},
        'hypertension': {0: 'no', 1: 'yes'},
        'heart_disease': {0: 'no', 1: 'yes'},
        'ever_married': {0: 'no', 1: 'yes'},
        'work_type': {0: 'children', 1: 'govt', 2: 'never', 3: 'private', 4: 'self'},
        'residence_type': {0: 'rural', 1: 'urban'},
        'smoking_status': {0: 'formerly', 1: 'never', 2: 'smokes', 3: 'unknown'},
        'stroke': {0: 'no', 1: 'yes'}
    }

Finally, for each categorical
variable ``var``, an additional column ``var_codes`` is added to the dataset,
containing the numerical category codes for each datapoint. These are the 
columns that we will later feed into the network.

The corresponding code in :func:`spellbook.inputs.encode_categories` looks
like this:

.. margin:: from :func:`spellbook.input.encode_categories`

    in :mod:`spellbook.input`

.. code:: python

    categories = {}
    for var in data:
        if data[var].dtype == 'object':
            print("variable '{}' is categorical".format(var))
            data[var] = pd.Categorical(data[var])

            # taken from https://stackoverflow.com/a/51102402
            categories[var] = dict(enumerate(data[var].cat.categories))

            # use numerical values instead of strings
            data[var+'_codes'] = data[var].cat.codes

Next, we are going to shuffle the dataset and order the rows randomly.
As we could see before, the original dataset is actually ordered in a way that
datapoints for patients with strokes come first. This is not what we want here
because when splitting the dataset into a training and a validation set, this
would mean that stroke cases would only be present in the training but not
the validation set. We do the shuffling with :meth:`pandas.DataFrame.sample`
and afterwards continue to adjust our list of the feature variables and the
name of the target variable to point to the new columns holding the integer
category indices:

.. margin:: from **2-stroke-prediction-naive.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    # shuffle data either now in pandas or later in TensorFlow
    data = data.sample(frac = 1.0)

    # use new numerical columns for the features
    for var in categories:
        if var == target:
            target = target + '_codes'
        else:
            features[features.index(var)] = var + '_codes'



Feeding the Dataset into *TensorFlow*
=====================================

Now it is time to split the data into a *training set*, which is used to
adjust the model parameters so as to describe the datapoints with
:meth:`tf.keras.Model.fit`, and a *validation set*. The latter is used to
evaluate the model performance and benchmark different models against each
other when changing model :term:`hyperparameter`\s such as the number of
layers, the number of nodes in a layer or the number of training epochs.

While doing the split, we are also going to convert the training and validation
datasets into objects that can be fed into the network. There are at least
three different ways of doing this in terms of the objects and datatypes
involved:

#. from a :class:`pandas.DataFrame` to a :class:`tensorflow.data.Dataset`
   where features and labels are combined in one object
#. from a :class:`pandas.DataFrame` to two :class:`tensorflow.Tensor`\s,
   one for the features and one for the labels
#. from a :class:`pandas.DataFrame` to two :class:`numpy.ndarray`\s,
   one for the features and one for the labels

Since columns in a :class:`pandas.DataFrame` can be accessed in much the
same way as entries in a *Python* :class:`dict`, it is actually also possible
to directly feed features stored in a :class:`pandas.DataFrame` and labels
stored in a :class:`pandas.Series` into *TensorFlow* networks.
The separation of a single :class:`pandas.DataFrame` into separate feature
and label sets for training, validation and testing is implemented in
:func:`spellbook.inputs.split_pddataframe_to_pddataframes`.



Option 1: Using *TensorFlow* Datasets
-------------------------------------

This approach is taken in ``2-stroke-prediction-naive.py``.
It is implemented in :func:`spellbook.inputs.split_pddataframe_to_tfdatasets`
and can be used like this:

.. margin:: from **2-stroke-prediction-naive.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    train, val = sb.inputs.split_pddataframe_to_tfdatasets(data, target, features, n_train=3500)
    print(train.cardinality()) # print the size of the dataset
    print(val.cardinality())

Using 3500 datapoints for the training set and the remaining 1409 for the
validation set corresponds to reserving a fraction of 71.3% of all data for
training. A typical recommendation is to use about 70% for training when
dealing with datasets containing a few thousand datapoints. As the size of
the total dataset increases, the fraction
reserved for training can increase and when a million datapoints are available,
it is perhaps sufficient to only use about 1% of them for validation.
Basically, the logic behind this is to use as many datapoints as possible for
training while at the same time ensuring that the validation (and possibly the
:term:`testing`) set have sufficient size to provide numerically stable
results for the metrics used to quantify the model performance. The smaller
the validation set is, the larger the statistical uncertainty on the metrics
will be.

Under the hood, in :func:`spellbook.inputs.split_pddataframe_to_tfdatasets`,
the :class:`pandas.DataFrame` is split into two separate frames,
one for the features and one for the target labels. These are converted to
two ``numpy.ndarray``\s which are then used to initialise the
:class:`tensorflow.data.Dataset` using the
:func:`tensorflow.data.Dataset.from_tensor_slices` function. Finally, the
split is applied:

.. margin:: from :func:`spellbook.input.split_pddataframe_to_tfdatasets`

    in :mod:`spellbook.input`

.. code:: python

    n = len(data)
    n_train, n_val, n_test = calculate_splits(n, frac_train, frac_val,
                                              n_train, n_val)

    # separate features and labels
    data_features = data[features]
    data_labels = data[target]

    # create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_features.values, data_labels.values))

    # split it into train/val/test
    train = dataset.take(n_train)
    val = dataset.skip(n_train).take(n_val)
    if n_test:
        test = dataset.skip(n_train).skip(n_val).take(n_test)
        return((train, val, test))
    else:
        return((train, val))

Note that this does not shuffle and batch the resulting dataset. Shuffling
may be done

- before in *pandas*: ``data = data.sample(frac = 1.0)``
- or afterwards in *TensorFlow*:
  ``train = train.shuffle(buffer_size = train.cardinality())``

Since we shuffled the data before converting and splitting them, we can
proceed to divide them into batches in *TensorFlow* with

.. margin:: from **2-stroke-prediction-naive.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    train = train.batch(batch_size = 100)
    val = val.batch(batch_size = 100)

The :term:`batch` size of ``100`` is chosen so that each batch contains at
least some stroke cases.



Option 2: Using *TensorFlow* Tensors
------------------------------------

Used in ``3-stroke-prediction-oversampling.py``
and implemented in :func:`spellbook.inputs.separate_tfdataset_to_tftensors`.

Just like before, *TensorFlow* :class:`tf.data.Dataset`\s for training and
validation are created. This time, they are each split into two separate
:class:`tf.Tensor`\s - one for the features and one for the labels:

.. margin:: from **3-stroke-prediction-oversampling.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    train, val = sb.inputs.split_pddataframe_to_tfdatasets(data, target, features, n_train=7000)
    print(train.cardinality()) # print the size of the dataset
    print(val.cardinality())
    
    # separate features and labels
    train_features, train_labels = sb.inputs.separate_tfdataset_to_tftensors(train)
    val_features, val_labels = sb.inputs.separate_tfdataset_to_tftensors(val)

Please don't mind the increased size of the training set for now - this script
uses *oversampling* to deal with the imbalance in the ``stroke`` categories.
We will look into this in more detail later.
    
The advantage of this :class:`tf.Tensor`-based approach is that it yields
a separate object containing just the target
labels, which can then be used when evaluating the model performance, e.g.
when calculating a confusion matrix or a :term:`ROC` curve from comparisons
of the predicted labels against the actual true target labels.

Internally, :func:`spellbook.inputs.separate_tfdataset_to_tftensors` proceeds as follows:

.. margin:: from :func:`spellbook.input.separate_tfdataset_to_tftensors`

    in :mod:`spellbook.input`

.. code:: python

    # unpack the feature and label tensors from the dataset
    # and reshape them into two separate tuples of tensors
    features, labels = zip(*dataset)

    # [...]

    # which are then stacked to form the features and labels tensors
    features = tf.stack(features)
    labels = tf.stack(labels)

Finally, since :class:`tf.Tensor`\s cannot be batched, in this approach, the
batching is left to the call to :meth:`tf.keras.Model.fit` later after model
setup.



.. _BinaryStrokePrediction-NumpyArrays:

Option 3: Using *NumPy* Arrays
------------------------------

Used in ``4-stroke-prediction-oversampling-norm.py``
and implemented in :func:`spellbook.inputs.split_pddataframe_to_nparrays`.

The third option is to split the dataset into :class:`numpy.ndarray`\s:

.. margin:: from **4-stroke-prediction-oversampling-norm.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

    train_features, train_labels, val_features, val_labels \
        = sb.inputs.split_pddataframe_to_nparrays(
            data, target, features, n_train=7000)

    print(train_features.shape) # print the size of the dataset
    print(train_labels.shape)
    print(val_features.shape)
    print(val_labels.shape)

The function :func:`spellbook.inputs.split_pddataframe_to_nparrays` works
like this in principle

.. margin:: from :func:`spellbook.input.split_pddataframe_to_nparrays`

    in :mod:`spellbook.input`

.. code:: python

    train = data.iloc[:n_train]
    val = data.iloc[n_train:].iloc[:n_val]
    result = [train[features].values, train[target].values,
              val[features].values, val[target].values]
    
    # [...]
    return tuple(result)

Like with :class:`tf.Tensor`\s in the previous approach, batching is left
the call to :meth:`tf.keras.Model.fit` later after model setup.
