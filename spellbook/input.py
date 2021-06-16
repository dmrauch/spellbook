'''
Functions for handling and preprocessing input data
'''

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


def encode_categories(data: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    '''
    Turns all string variables into categorical variables and adds columns with
    the corresponding numerical values

    Args:
        data (:class:`pandas.DataFrame`): The dataset

    Returns:
        Dictionary of dictionaries with the encodings of each category.
        For each categorical variable there is a dictionary with the numerical
        codes as the keys and the category names/labels as the values
    '''

    categories = {}
    for var in data:
        if data[var].dtype == 'object':
            print("variable '{}' is categorical".format(var))
            data[var] = pd.Categorical(data[var])

            # taken from https://stackoverflow.com/a/51102402
            categories[var] = dict(enumerate(data[var].cat.categories))

            # use numerical values instead of strings
            data[var+'_codes'] = data[var].cat.codes
    
    return(categories)


def oversample(data: pd.DataFrame,
               target: str,
               shuffle: bool = True) -> pd.DataFrame:
    '''
    Oversample the data to increase the sizes of the minority classes/categories

    The target variable is assumed to be of type *categorical*, so
    this function should only be called after converting the target variable
    from type *object* (when the values are strings) to type *categorical* with
    something like

    .. code:: python

        import pandas as pd
        data[target] = pd.Categorical(data[target])

    The original datapoints in the minority classes/categories are retained
    and oversampling is only used to fill in the missing datapoints in
    order to ensure that for small imbalances all the original datapoints are
    kept. Otherwise, because of random fluctuations, some datapoints may be
    sampled twice and others never, losing some of the original data.

    Args:
        data(:class:`pandas.DataFrame`): The imbalanced data
        target: The name of the variable that should be balanced
        shuffle: Optional, whether or not that oversampled DataFrame should
            be shuffled. If set to ``False``, then the DataFrame will be
            a concatenation of the different classes/categories, i.e. all
            datapoints belonging to the same class/category will be grouped
            together
    
    Returns:
        :class:`pandas.DataFrame`: A dataset with each class/category populated
        with the same number of datapoints
    
    Example:

    .. doctest::

        >>> import numpy as np
        >>> import pandas as pd
        >>> import spellbook as sb
        >>> np.random.seed(0) # only for the sake of reproducibility of the
        >>>                   # doctests here in the documentation

        >>> data_dict = {'cat': ['p']*5 + ['n']*2}
        >>> data = pd.DataFrame(data_dict)
        >>> sb.input.encode_categories(data)
        variable 'cat' is categorical
        {'cat': {0: 'n', 1: 'p'}}

        >>> print('before oversampling:', data.head)
        before oversampling: <bound method NDFrame.head of   cat  cat_codes
        0   p          1
        1   p          1
        2   p          1
        3   p          1
        4   p          1
        5   n          0
        6   n          0>

        >>> # Oversampling without shuffling returns a dataset sorted by category
        >>> data_oversampled = sb.input.oversample(data, target='cat', shuffle=False)
        >>> print('after oversampling:', data_oversampled.head)
        after oversampling: <bound method NDFrame.head of   cat  cat_codes
        5   n          0
        6   n          0
        5   n          0
        6   n          0
        6   n          0
        0   p          1
        1   p          1
        2   p          1
        3   p          1
        4   p          1>

        >>> # Therefore, shuffling is activated by default
        >>> data_oversampled = sb.input.oversample(data, target='cat')
        >>> print('after oversampling:', data_oversampled.head)
        after oversampling: <bound method NDFrame.head of   cat  cat_codes
        3   p          1
        1   p          1
        6   n          0
        5   n          0
        5   n          0
        0   p          1
        4   p          1
        6   n          0
        2   p          1
        6   n          0>
    '''

    assert isinstance(data, pd.DataFrame)
    assert isinstance(target, str)

    # data needs to be of type pandas.Categorical
    assert data[target].dtype == 'category'

    # get the different categories
    cats = data[target].cat.categories

    # split data by target category
    datapoints = {}
    for i in range(len(cats)):
        datapoints[cats[i]] = data[data[target] == cats[i]]

    # count datapoints in each category
    counts = np.zeros(shape = len(cats), dtype=int)
    for i in range(len(cats)):
        counts[i] = datapoints[cats[i]][target].count()

    # get number of datapoints in the largest category
    nmax = np.amax(counts)

    resampled = pd.DataFrame()
    for i, cat in enumerate(cats):

        # how many datapoints are missing?
        nmiss = nmax - len(datapoints[cat])
        assert nmiss >= 0

        if nmiss > 0:

            # ordered array of indices of the underrepresented category
            indices = np.arange(counts[i])

            # draw nmiss times with replacement from the indices array
            choices = np.random.choice(indices, size=nmiss)

            datapoints[cat] = datapoints[cat].append(
                datapoints[cat].iloc[choices])

        resampled = resampled.append(datapoints[cat])
    
    if shuffle: resampled = resampled.sample(frac = 1.0)
    return(resampled)


def calculate_splits(n: int,
                     frac_train: float = 0.7,
                     frac_val: float = None,
                     n_train: int = None,
                     n_val: int = None) -> Tuple[int, int, int]:
    '''
    Calculate the numbers of datapoints in the training, validation and test
    sets

    The size of the test dataset is calculated so as to use the remaining
    datapoints after filling the training and validation sets. If **frac_val**
    or **n_val** are not given, or if the training and validation sets use
    up all datapoints from the original full dataset, then the size of the test
    set will be set to zero.

    Args:
        n: The size of the full dataset
        frac_train: Optional, the fractional size of the training set
        frac_val: Optional, the fractional size of the validation set
        n_train: Optional, the absolute size of the training set
        n_val: Optional, the absolute size of the validation set

    Returns:
        The absolute sizes of the training, validation and test sets
    
    Examples:

    - Default training/validation split defined by fractional sizes

      .. testcode::

         import spellbook as sb
         sizes = sb.input.calculate_splits(n=1000)
         print('train:', sizes[0])
         print('validation:', sizes[1])
         print('test:', sizes[2])
    
      Output:

      .. testoutput::

         train: 700
         validation: 300
         test: 0
    
    - Training/validation/test split defined by fractional sizes

      .. testcode::

         import spellbook as sb
         sizes = sb.input.calculate_splits(n=1000, frac_train=0.7, frac_val=0.2)
         print('train:', sizes[0])
         print('validation:', sizes[1])
         print('test:', sizes[2])
    
      Output:

      .. testoutput::

         train: 700
         validation: 200
         test: 100
    
    - Training/validation/test split defined by absolute sizes

      .. testcode::

         import spellbook as sb
         sizes = sb.input.calculate_splits(n=1000, n_train=600, n_val=250)
         print('train:', sizes[0])
         print('validation:', sizes[1])
         print('test:', sizes[2])
    
      Output:

      .. testoutput::

         train: 600
         validation: 250
         test: 150
    
    '''

    if frac_train:
        assert isinstance(frac_train, float)
        assert frac_train >= 0.0 and frac_train <= 1.0
    if frac_val:
        assert isinstance(frac_val, float)
        assert frac_val >= 0.0 and frac_val <= 1.0
        assert frac_train and not (n_train or n_val)
    if n_train:
        assert isinstance(n_train, int)
        assert n_train >= 0 and n_train <= n
        assert not frac_val
    if n_val:
        assert isinstance(n_val, int)
        assert n_val >= 0 and n_val <= n
        assert n_train and not frac_val # frac_train has default value

    if n_train:
        if not n_val: n_val = n - n_train
    else:
        n_train = round(n * frac_train)
        if frac_val:
            if frac_train + frac_val == 1.0:
                n_val = n - n_train
            else:
                n_val = round(n * frac_val)
        else:
            n_val = n - n_train

    # now we can assume that n_train and n_val are given
    n_test = n - n_train - n_val
    assert isinstance(n_train, int)
    assert isinstance(n_val, int)
    assert isinstance(n_test, int)
    assert n_train >= 0 and n_val >= 0 and n_test >= 0
    return((n_train, n_val, n_test))


def split_pddataframe_to_pddataframes(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    frac_train: float = 0.7,
    frac_val: float = None,
    n_train: int = None,
    n_val: int = None) -> Union[
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
              pd.DataFrame, pd.Series]
    ]:
    '''
    Get separate :class:`pandas.DataFrame`\s and :class:`pandas.Series`
    for training/validation/test features and labels

    - If *either* **frac_train** and **frac_val** *or* **n_train** and **n_val**
      are given, six datasets are returned: the training features and labels,
      the validation features and labels and the test features and labels,
      with the test sets sized so as to use the remaining
      datapoints after the training and validation sets were filled
    - If no **frac_val** or **n_val** is given, the output will contain no test
      datasets, but rather just the training features and labels and
      the validation features and labels,
      with the validation sets sized so as to use the remaining datapoints
      after the training datasets were filled

    .. note:: This function does not include shuffling and batching of the
              data, so these should be done separately!

    Args:
        data (:class:`pandas.DataFrame`): The dataset
        target: The name of the target variable containing the labels
        features: The names of the feature variables. Not all variables have
            to be extracted from the dataset, e.g. the numerical codes of a
            categorical variable should be extracted, whereas the variable
            containing the original strings should be skipped.
        frac_train: Optional, the fractional size of the training dataset
        frac_val: Optional, the fractional size of the validation dataset
        n_train: Optional, the number of datapoints in the training set
        n_val: Optional, the number of datapoints in the validation set

    Returns:
        Tuple of :class:`pandas.DataFrame` and :class:`pandas.Series`:
        A tuple containing the different
        datasets requested, separated into features and labels

        - (**train_features**: :class:`pandas.DataFrame`, \
           **train_labels**: :class:`pandas.Series`, \
           **validation_features**: :class:`pandas.DataFrame`, \
           **validation_labels**: :class:`pandas.Series`)
        - (**train_features**: :class:`pandas.DataFrame`, \
           **train_labels**: :class:`pandas.Series`, \
           **validation_features**: :class:`pandas.DataFrame`, \
           **validation_labels**: :class:`pandas.Series`, \
           **test_features**: :class:`pandas.DataFrame`, \
           **test_labels**: :class:`pandas.Series`)

    Example:

    - Training/validation split with the default relative sizes of 70%/30%:

      .. testcode::
  
         import numpy as np
         import pandas as pd
         import spellbook as sb

         data_dict = {
             'x': np.arange(100),
             'y': np.arange(100),
             'target': np.arange(100)
         }
         data = pd.DataFrame(data_dict)

         train_features, train_labels, val_features, val_labels \\
             = sb.input.split_pddataframe_to_pddataframes(
                 data, target='target', features=['x', 'y'])
         
         print('train_features:', type(train_features), train_features.shape)
         print('val_labels:', type(val_labels), val_labels.shape)

      Output:

      .. testoutput::

         train_features: <class 'pandas.core.frame.DataFrame'> (70, 2)
         val_labels: <class 'pandas.core.series.Series'> (30,)
    
    See also:
    
    - :func:`calculate_splits`
    '''

    assert isinstance(target, str)
    assert isinstance(features, list)
    for feature in features:
        assert isinstance(feature, str)

    n = len(data)
    n_train, n_val, n_test = calculate_splits(n, frac_train, frac_val,
                                              n_train, n_val)

    train = data.iloc[:n_train]
    val = data.iloc[n_train:].iloc[:n_val]
    result = [train[features], train[target], val[features], val[target]]

    if n_test > 0:
        test = data.iloc[n_val:]
        result += [test[features], test[target]]

    return(tuple(result))


def split_pddataframe_to_nparrays(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    frac_train: float = 0.7,
    frac_val: float = None,
    n_train: int = None,
    n_val: int = None) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray]
    ]:
    '''
    Get separate :class:`numpy.ndarray`\s for training/validation/test
    features and labels

    - If *either* **frac_train** and **frac_val** *or* **n_train** and **n_val**
      are given, six datasets are returned: the training features and labels,
      the validation features and labels and the test features and labels,
      with the test sets sized so as to use the remaining
      datapoints after the training and validation sets were filled
    - If no **frac_val** or **n_val** is given, the output will contain no test
      datasets, but rather just the training features and labels and
      the validation features and labels,
      with the validation sets sized so as to use the remaining datapoints
      after the training datasets were filled

    .. note:: This function does not include shuffling and batching of the
              data, so these should be done separately!

    Args:
        data (:class:`pandas.DataFrame`): The dataset
        target: The name of the target variable containing the labels
        features: The names of the feature variables. Not all variables have
            to be extracted from the dataset, e.g. the numerical codes of a
            categorical variable should be extracted, whereas the variable
            containing the original strings should be skipped.
        frac_train: Optional, the fractional size of the training dataset
        frac_val: Optional, the fractional size of the validation dataset
        n_train: Optional, the number of datapoints in the training set
        n_val: Optional, the number of datapoints in the validation set

    Returns:
        Tuple of :class:`numpy.ndarray`: A tuple containing the different
        datasets requested, separated into features and labels

        - (**train_features**: :class:`numpy.ndarray`, \
           **train_labels**: :class:`numpy.ndarray`, \
           **validation_features**: :class:`numpy.ndarray`, \
           **validation_labels**: :class:`numpy.ndarray`)
        - (**train_features**: :class:`numpy.ndarray`, \
           **train_labels**: :class:`numpy.ndarray`, \
           **validation_features**: :class:`numpy.ndarray`, \
           **validation_labels**: :class:`numpy.ndarray`, \
           **test_features**: :class:`numpy.ndarray`, \
           **test_labels**: :class:`numpy.ndarray`)

    Example:

    - Training/validation split with the default relative sizes of 70%/30%:

      .. testcode::
  
         import numpy as np
         import pandas as pd
         import spellbook as sb

         data_dict = {
             'x': np.arange(100),
             'y': np.arange(100),
             'target': np.arange(100)
         }
         data = pd.DataFrame(data_dict)

         train_features, train_labels, val_features, val_labels \\
             = sb.input.split_pddataframe_to_nparrays(
                 data, target='target', features=['x', 'y'])
         
         print('train_features:', type(train_features), train_features.shape)
         print('val_labels:', type(val_labels), val_labels.shape)

      Output:

      .. testoutput::

         train_features: <class 'numpy.ndarray'> (70, 2)
         val_labels: <class 'numpy.ndarray'> (30,)
    
    See also:
    
    - :func:`calculate_splits`
    '''

    assert isinstance(target, str)
    assert isinstance(features, list)
    for feature in features:
        assert isinstance(feature, str)

    n = len(data)
    n_train, n_val, n_test = calculate_splits(n, frac_train, frac_val,
                                              n_train, n_val)

    train = data.iloc[:n_train]
    val = data.iloc[n_train:].iloc[:n_val]
    result = [train[features].values, train[target].values,
              val[features].values, val[target].values]

    if n_test > 0:
        test = data.iloc[n_val:]
        result += [test[features].values, test[target].values]

    return tuple(result)


def split_pddataframe_to_tfdatasets(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    frac_train: float = 0.7,
    frac_val: float = None,
    n_train: int = None,
    n_val: int = None) -> Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
    ]:
    '''
    Get separate :class:`tf.data.Dataset`\s for training, validation and
    testing

    - If *either* **frac_train** and **frac_val** *or* **n_train** and **n_val**
      are given, three datasets are returned: the training, the validation and
      the test sets, with the test set sized so as to use the remaining
      datapoints after the training and validation sets were filled
    - If no **frac_val** or **n_val** is given, the output will contain no test
      dataset, but rather just the training and the validation sets,
      with the validation set sized so as to use the remaining datapoints
      after the training dataset was filled

    .. note:: This function does not include shuffling and batching of the
              data, so these should be done separately!

    Args:
        data (:class:`pandas.DataFrame`): The data
        target: The name of the target variable containing the labels
        features: The names of the feature variables. Not all variables have
            to be extracted from the dataset, e.g. the numerical codes of a
            categorical variable should be extracted, whereas the variable
            containing the original strings should be skipped.
        frac_train: Optional, the fractional size of the training dataset
        frac_val: Optional, the fractional size of the validation dataset
        n_train: Optional, the number of datapoints in the training set
        n_val: Optional, the number of datapoints in the validation set

    Returns:
        Tuple of :class:`tf.data.Dataset`\s: A tuple containing the training,
        validation and possibly also the test datasets:

        - (:class:`tf.data.Dataset`, :class:`tf.data.Dataset`): the training
          and the validation set
        - (:class:`tf.data.Dataset`, :class:`tf.data.Dataset`,
          :class:`tf.data.Dataset`): the training, the validation and the test
          set

    Example:

    - Training/validation split with the default relative sizes of 70%/30%:

      .. testcode::
  
         import numpy as np
         import pandas as pd
         import spellbook as sb

         data_dict = {
             'x': np.arange(100),
             'y': np.arange(100),
             'target': np.arange(100)
         }
         data = pd.DataFrame(data_dict)

         train, val = sb.input.split_pddataframe_to_tfdatasets(
             data, target='target', features=['x', 'y'])
         
         print('train:', type(train), train.cardinality().numpy())
         print('val:', type(val), val.cardinality().numpy())

      Output:

      .. testoutput::

         train: <class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'> 70
         val: <class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'> 30
    
    See also:
    
    - :func:`calculate_splits`
    '''

    assert isinstance(target, str)
    assert isinstance(features, list)
    for feature in features:
        assert isinstance(feature, str)

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


def separate_tfdataset_to_tftensors(
    dataset: tf.data.Dataset
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    '''
    Separate a :class:`tf.data.Dataset` into :class:`tf.Tensor`\s for the
    features and the labels

    Args:
        dataset (:class:`tf.data.Dataset`): The dataset to separate, may
            be batched or unbatched

    Returns:
        Tuple of :class:`tf.Tensor`: A tuple holding the two separate
        tensors for the features and the labels:
        (**features**: :class:`tf.Tensor`, **labels**: :class:`tf.Tensor`)
    '''
    
    # dataset is tf.data.Dataset and has format [[f, l], [f, l], ...]
    # need to reshape this into two separate tuples of tensors (f, f, f), (l, l, l)
    # if type(dataset).__name__ == 'BatchDataset':
    if dataset.element_spec[0].shape[0] is None \
       and dataset.element_spec[1].shape[0] is None:
        # dataset is batched -> remove batching first
        features, labels = zip(*dataset.unbatch())
    else:
        features, labels = zip(*dataset)

    # stack tuple of tensors of rank d into single tensor of rank d+1
    # before:
    # - train_features is a tuple of tensors of rank/ndim 1 and shape (10,),
    #   i.e. for each datapoint there is a vector entry in the tuple
    # - train_labels is a tuple of tensors of rank/ndim 0 and shape (),
    #   i.e. for each datapoint there is a scalar entry in the tuple
    # after:
    # - train_features is a tensor of rank/ndim 2 and shape (2500, 10)
    # - train_labels is a tensor of rank/ndim 1 and shape (2500,)
    features = tf.stack(features)
    labels = tf.stack(labels)

    return((features, labels))
