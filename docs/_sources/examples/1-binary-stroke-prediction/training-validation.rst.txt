.. _BinaryStrokePrediction-TrainingValidation:

*****************************
Model Training and Validation
*****************************

Now that the inputs are prepared, we can proceed to set up and configure a
model. I tried different configurations with up to three hidden layers with
different numbers of nodes in each layer and eventually settled on the following
model layout:

.. margin:: from **2-stroke-prediction-naive.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(10,)),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(100, activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(50, activation='relu'),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(1, activation='sigmoid')
   ])
   model.summary()

This sets up a neural network with three fully connected hidden layers,
interleaved with two dropout layers and with :term:`L2 regularisation`
activated in the middle hidden layer.

Since there are 10 input variables, the input layer of the network has
``input_shape=(10,)``. The following dense layers have 10, 100 and 50 neurons,
respectively. Dropout is a regularisation technique
aimed at eliminating overtraining by introducing an element of randomness.
During training, a :term:`dropout` layer randomly discards a configurable
fraction of the inputs received from the preceding layer during each
feed-forward step. In this case, 20% of the previous layer's outputs are thrown
away by each of the two dropout layers. Finally, the network ends with a single
sigmoid-activated output neuron. This is a typical situation in binary
classification - low values near 0 denote one class, larger values near 1
the other.

This model has a total of 6311 parameters as can be seen from the ``summary()``
command:

- The first hidden dense layer with 10 neurons, which are fed from the 10
  input nodes plus a bias node, has (10 + 1) · 10 = 110 parameters
- The second dense layer with 100 neurons, which are fed from the 10 neurons
  of the previous layer plus a bias node, has (10 + 1) · 100 = 1100
  parameters
- Dropout layers do not have any free parameters that can be optimised during
  training
- Similarly, the third dense layer and the single-neuron output layer have
  5050 and 51 parameters, respectively.

Before we can begin training the model, we need to equip the model with a
few more tools, in particular, a *loss function* and an *optimiser*.
Metrics that quantify certain aspects of the model's performance are not
strictly necessary but highly useful:

.. margin:: from **2-stroke-prediction-naive.py**

   in ``examples/1-binary-stroke-prediction/``

.. code:: python

   model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=[
                     tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
                     tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                     tf.keras.metrics.TruePositives(name='tp'),
                     tf.keras.metrics.TrueNegatives(name='tn'),
                     tf.keras.metrics.FalsePositives(name='fp'),
                     tf.keras.metrics.FalseNegatives(name='fn'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.Precision(name='precision')
                 ])

The optimiser is the algorithm responsible for minimising the loss
function and the loss function quantifies how much the predictions of the
model generated during training differ from the training labels.
The loss function of choice for binary classification tasks is
*binary crossentropy* (:class:`tf.keras.losses.BinaryCrossentropy`)
and *Adam* (:class:`tf.keras.optimizers.Adam`) is one of several optimisers
that can be used.

In binary classification, in particular, there is a wide range of different,
but related metrics. Let us take this opportunity and briefly discuss some of
the metrics specifically geared towards binary classification - a thorough
discussion can be found on `Wikipedia: Sensitivity and specificity
<https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_:

- **True Positives** (:term:`TP`): The number of datapoints that were classified
  as belonging to the *positive* class, corresponding to the presence of a
  stroke in our example, by the model and that in reality do belong to the
  positive class
- **True Negatives** (:term:`TN`): The number of datapoints that were classified
  as belonging to the *negative* class, corresponding to the absence of a
  stroke in our example, by the model and that in reality do belong to the
  negative class
- **False Positives** (:term:`FP`): The number of datapoints that were
  classified as belonging to the *positive* class but truly belong to the
  *negative* class
- **False Negatives** (:term:`FN`): The number of datapoints that were
  classified as belonging to the *negative* class but truly belong to the
  *positive* class
- **Recall**, also called *sensitivity* and *True Positive Rate*
  (:term:`TPR`) is defined as

  .. math:: \frac{\text{TP}}{\text{TP} + \text{FN}}

  and quantifies the fraction of the truly *positve* datapoints that were
  correctly predicted/classified by the model.
- **Precision** is defined as

  .. math:: \frac{\text{TP}}{\text{TP} + \text{FP}}

  and quantifies the fraction of all positively classified datapoints that are
  in fact truly positive.
- **Accuracy**: The fraction of datapoints that were correctly
  classified, i.e. the predicted class was identical to the true class.
  Accuracy is not a specifically binary metrics but is also very useful in
  multi-class classification problems.

Finally, we start the training as follows:

.. margin:: from **2-stroke-prediction-naive.py**

   in ``examples/1-binary-stroke-prediction/``

.. code:: python

   epochs = 100
   history = model.fit(train, epochs=epochs,
       validation_data=val,
       callbacks = [
           tf.keras.callbacks.CSVLogger(
               filename='naive-e{}-history.csv'.format(epochs)),
           sb.train.ModelSavingCallback(
               foldername='model-naive-e{}'.format(epochs))
       ])

This configures 100 training epochs, during each of which the full training
set is digested and processed by the network in its search for ever-improving
model parameters. During training not only the loss is calculated, but also
the values of the other quantities specified in the metrics list. At the end
of each epoch, the complete validation dataset is consumed to calculate the
same metrics on a disjoint dataset that is not used for adjusting the model
parameters. The values of all metrics, both those calculated from the training
set and those evaluated from the test set, are written to a ``*.csv`` file by
the :class:`tf.keras.callbacks.CSVLogger` callback at the end of each epoch.
Similarly, the :class:`spellbook.train.ModelSavingCallback` callback saves
the entire model including its architecture as well as all parameter values
every 10 epochs. This can be very handy because it preserves the model and
its state in case the training has to be cancelled.

.. note:: I will use this exact model setup in the rest of this
          tutorial to demonstrate the effect that different training strategies
          can have on models with exactly the same layout. The only difference
          will be in the batch size, which is taken to be 100 here in the
          first approach and is reduced to the default value of 32 in the
          later versions.


Naive Approach
==============

Let's have a look at the training and validation metrics we obtain when
training the model in this 'naive' setup using the imbalanced dataset with
96% negative (*no stroke*) and 4% positive (*stroke*) cases.

Plots showing the evolution of the binary metrics can be generated with
*spellbook*'s :func:`spellbook.train.plot_history_binary` function:

.. margin:: from **2-stroke-prediction-naive.py**

   in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # inspect/plot the training history
   sb.train.plot_history_binary(history,
      '{}-e{}-history'.format(prefix, epochs))

The resulting plot will be shown in Figure 24.

In order to determine the confusion matrix, it is necessary to retrieve the
predictions of the trained model for the validation data. These can be
obtained after some reorganisation and type conversion gymnastics:

.. margin:: from **2-stroke-prediction-naive.py**

   in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # separate the datasets into features and labels
   _, train_labels = sb.input.separate_tfdataset_to_tftensors(train)
   _, val_labels = sb.input.separate_tfdataset_to_tftensors(val)

   # obtain the predictions of the model
   train_predictions = model.predict(train)
   val_predictions = model.predict(val)
   
   # not strictly necessary: remove the superfluous inner nesting
   train_predictions = tf.reshape(train_predictions, train_predictions.shape[0])
   val_predictions = tf.reshape(val_predictions, val_predictions.shape[0])
   
   # until now the predictions are still continuous in [0, 1] and this breaks the
   # calculation of the confusion matrix, so we need to set them to either 0 or 1
   # according to a default intermediate threshold of 0.5
   train_predicted_labels = sb.train.get_binary_labels(
       train_predictions, threshold=0.5)
   val_predicted_labels = sb.train.get_binary_labels(
       val_predictions, threshold=0.5)
   
   # calculate and plot the confusion matrix
   class_names = list(categories['stroke'].values())
   class_ids = list(categories['stroke'].keys())
   val_confusion_matrix = tf.math.confusion_matrix(val_labels,
                                                   val_predicted_labels,
                                                   num_classes=len(class_names))

The confusion matrix relates the predicted classes to the true classes and
shows how often the model was accurate and in what way it erred. It can
be plotted with the :func:`spellbook.plot.plot_confusion_matrix` function:

.. margin:: from **2-stroke-prediction-naive.py**

   in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # absolute datapoint counts
   sb.plot.save(
      sb.plot.plot_confusion_matrix(
         confusion_matrix = val_confusion_matrix,
         class_names = class_names,
         class_ids = class_ids),
      filename = '{}-e{}-confusion.png'.format(prefix, epochs))

The resulting plot is shown in Figure 25.

.. _ex1-fig-naive:

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/naive-e100-history-loss-acc.png
          :height: 300px

          Figure 24: Evolution of the loss and the classification accuracy
          during training in the *naive* approach

     - .. figure:: images/naive-e100-confusion.png
          :height: 300px

          Figure 25: Confusion matrix with absolute datapoint counts in the
          *naive* approach

As we can see from Figure 24, the loss improves only in the beginning, but
then only decreases very slightly after about 10 epochs. Similarly, the
accuracy reaches 95-96% at about the same time and does not improve beyond
that. This accuracy very closely resembles the fractions of the two target
classes or labels. Recall from :ref:`Figure 1 <ex1-fig-variables>` in
:doc:`/examples/1-binary-stroke-prediction/index` that 95.7% of the
people in the dataset have not had a stroke
compared to 4.3% who have. Looking at the confusion matrix in
:ref:`Figure 25 <ex1-fig-naive>`, we
can confirm what has happened: The model has learned to practically only and
exclusively categorise the data as *negative* or *healthy*. Since it predicts
all datapoints to be *negative*, its accuracy simply reflects how often this
category appears in the dataset and despite the 95% accuracy it is in fact
completely ignorant. Looking at the evolution of the true/false positive/
negative counts as well as the precision and recall in Figures 26 and 27,
we can indeed confirm that this is not an accidental fluctuation:

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/naive-e100-history-pos-neg.png
          :height: 300px

          Figure 26: Evolution of the true/false positive/negative counts
          during training in the *naive* approach

     - .. figure:: images/naive-e100-history-rec-prec.png
          :height: 300px

          Figure 27: Evolution of precision and recall during training
          in the *naive* approach

All of this is despite the fact that the batch size was increased to 100
so as to make it probable that at least *some* positive cases were present in
each batch. However, the 4% fraction of the positive cases is simply just
much smaller than the 96% fraction of negative cases.

This is a typical situation when dealing with *imbalanced data*, where one
target class appears substantially more often than the other(s). Note that
simply training longer does not change the fundamental challenge. Instead, when
facing this situation, two approaches come to mind:

- **Class/event weights**: Assigning larger weights to the minority events, thus
  making them more 'important' effectively. This is similar to the *boosting*
  approach commonly used when training sets of decision trees (even in balanced
  situations).
- **Oversampling**: Balancing the datasets by passing the same minority events
  to the model multiple times during each epoch

In this tutorial, we will apply the oversampling method.



Oversampling the Minority Class
===============================

The idea of oversampling is to divide the dataset into two parts according the
positive and
negative cases, thus separating the majority and minority classes and then to
repeatedly sample with replacement from the minority classes to create a
'pseudo' minority dataset that is equal in size to the majority class. This
resampled minority dataset will naturally contain events multiple times and the
oversampling factor is given by the ratio of the sizes of the majority and the
original minority sets. Oversampling is implemented in
:func:`spellbook.input.oversample` and can be applied as follows:

.. margin:: from **3-stroke-prediction-oversampling.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # oversampling (including shuffling of the data)
   data = sb.input.oversample(data, target)

Under the hood, oversampling is realised as follows:

.. margin:: from :func:`spellbook.input.oversample`

    in :mod:`spellbook.input`

.. code:: python

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

The data is grouped and separated according to the different categories of
the target variable and the difference in size between the largest
class/category and the minority classes/categories is made up by sampling
them with replacement until the size gaps are closed. This way, it is ensured
that all original datapoints in the minority classes are preserved and only
the datapoints needed to fill the gaps are randomly sampled.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/oversampling-e500-history-loss-acc.png
          :height: 300px

          Figure 28: Evolution of the loss and the classification accuracy
          during training when *oversampling* is used

     - .. figure:: images/oversampling-e500-history-pos-neg.png
          :height: 300px

          Figure 29: Evolution of the true/false positives/negatives during training when *oversampling* is used

     - .. figure:: images/oversampling-e500-confusion-norm-true.png
          :height: 300px

          Figure 30: Confusion matrix, normalised for each true target class

In Figure 28 we can see that the loss is now somewhat increased compared to
the *naive* approach, but decreases steadily over a longer range of training
epochs. Likewise, the initial classification accuracy has decreased to about
70% and continually improves from there to about 86%. Given that the training
and validation sets now, thanks to oversampling, are made up of very similar
amounts of *positive* and *negative* datapoints, this accuracy is
actually meaningful. This can also be seen in Figures 29 and 30, which show
the evolution of the numbers of true/false positives/negatives and the
truth-normalised confusion matrix, respectively. Now, the model is actually
learning to detect *positive* cases and reaches a performance with 0.5% of
the truly *positive* cases wrongly classified as *negative* and 27.6% of the
truly *negative* cases wrongly classified as *positive*.



Normalising the Inputs
======================

Now, we are going to apply an additional technique on top -
*input normalisation*. Normally, this should be done earlier, but I wanted to
evaluate the benefit this brings by benchmarking against the previous
model based on un-normalised inputs. The strategy of input normalisation is
based on the sensitivity of neural networks to the scale of numerical
input values - variables of the order of magnitude 1 are handled much better
than values in the range of hundreds, thousands or even larger.

One common approach is *standardisation*, where a variable is shifted and
scaled so that after the transformation its mean is 0 and its variance 1.
Here, in order to see how powerful even a very simplistic normalisation is,
we are just dividing the continuous numerical variables by some constant
factors to rescale them to intervals roughly between 0 and 1:

.. margin:: from **4-stroke-prediction-oversampling-norm.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # normalisation
   data['age_norm'] = data['age'] / 100.0
   data['avg_glucose_level_norm'] = data['avg_glucose_level'] / 300.0
   data['bmi_norm'] = data['bmi'] / 100.0
   # replace unnormalised variable names with their normalised counterparts
   features[features.index('age')] = 'age_norm'
   features[features.index('avg_glucose_level')] = 'avg_glucose_level_norm'
   features[features.index('bmi')] = 'bmi_norm'

We are keeping the original unnormalised variables, created new, normalised
ones with the suffix ``_norm`` and replace the original variables with them
in the list of feature variables which is later used for splitting the full
dataset into separate sets for the features and the labels.

After training for 2000 epochs, we get the following results:

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/oversampling-normalised-e2000-history-loss-acc.png
          :height: 300px

          Figure 31: Evolution of the loss and the classification accuracy
          during training when both *oversampling* and *input normalisation*
          are used

     - .. figure:: images/oversampling-normalised-e2000-history-pos-neg.png
          :height: 300px

          Figure 32: Evolution of the true/false positives/negatives
          during training when both *oversampling* and *input normalisation*
          are used

     - .. figure:: images/oversampling-normalised-e2000-confusion-norm-true.png
          :height: 300px

          Figure 33: Confusion matrix, normalised for each true target class

As we can see from Figure 33, the performance of the model has improved further
as reflected in reduced classification errors shown in the off-diagonal
cells of the confusion matrix: The fraction of true negatives wrongly
classified as positive has decreased to 9.5% and the fraction of true positives
wrongly classified as negative has even gone down to zero.



Comparison: ROC Curves
======================

Finally, let's compare all three approaches in yet another way: by means of
the *Receiver Operator Characteristic* (:term:`ROC`) curves.
ROC curves are a commonly used tool to benchmark the performance of different
models against each other. They show the *False Positive Rate* (:term:`FPR`)
on the x-axis, indicating how often truly negative datapoints are
wrongly classified as positive, and the *True Positive Rate* (:term:`TPR`) on
the y-axis, indicating what fraction of the truly positive datapoints are
correctly classified.

ROC curves are implemented in :class:`spellbook.train.ROCPlot`.
At the end of each training, we determine the ROC curves for the respective
models and save them to disk using *Python*'s *pickle* mechanism:

.. margin:: from **4-stroke-prediction-oversampling-norm.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   # calculate and plot the ROC curve
   roc = sb.train.ROCPlot()
   roc.add_curve('{} / {} epochs (training)'.format(name, epochs),
       train_labels, train_predictions.numpy(),
       plot_args = dict(color='C0', linestyle='--'))
   roc.add_curve('{} / {} epochs (validation)'.format(name, epochs),
       val_labels, val_predictions.numpy(),
       plot_args = dict(color='C0', linestyle='-'))
   sb.plot.save(roc.plot(), '{}-e{}-roc.png'.format(prefix, epochs))
   roc.pickle_save('{}-e{}-roc.pickle'.format(prefix, epochs))

We can subsequently load these pickle files with

.. margin:: from **5-roc.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   roc_naive100 = ROCPlot.pickle_load('naive-e100-roc.pickle')
   roc_oversampling2000 = ROCPlot.pickle_load('oversampling-e2000-roc.pickle')
   roc_norm2000 = ROCPlot.pickle_load('oversampling-normalised-e2000-roc.pickle')

and combine the ROC curves of different models in a single plot with

.. margin:: from **5-roc.py**

    in ``examples/1-binary-stroke-prediction/``

.. code:: python

   roc = ROCPlot()
   roc += roc_naive100
   WP = roc.get_WP('naive / 100 epochs (validation)', threshold=0.5)
   roc.draw_WP(WP, linestyle='-', linecolor='C1')
   roc.curves['naive / 100 epochs (training)']['line'].set_color('C1')
   roc.curves['naive / 100 epochs (validation)']['line'].set_color('C1')
   
   roc += roc_oversampling2000
   WP = roc.get_WP('oversampling / 2000 epochs (validation)', threshold=0.5)
   roc.draw_WP(WP, linestyle='-', linecolor='C0')
   
   roc += roc_norm2000
   WP = roc.get_WP('oversampling normalised / 2000 epochs (validation)', threshold=0.5)
   roc.draw_WP(WP, linestyle='-', linecolor='black')
   roc.curves['oversampling normalised / 2000 epochs (training)']['line'].set_color('black')
   roc.curves['oversampling normalised / 2000 epochs (validation)']['line'].set_color('black')

   sb.plot.save(roc.plot(), prefix+'roc-2000-naive-oversampling-normalised.png')

This code also serves to draw specific working points on the ROC curves,
corresponding to picking the values of the model output (the sigmoid-activated
output of the last model layer) that mark the boundary between classifying
a datapoint as *negative* or as *positive*. Defining different working points
with different threshold values can decrease the number of false negatives
at the cost of increasing the number of false positives and vice versa.
Here, we are simply going to stick to the default working points with
threshold values of 0.5.

In order to have a fair benchmark, we plot the ROC curves for the second and
third model after training for 2000 epochs in both cases. Since it was
obvious that the first, *naive* model does not perform well at all for
systematic reasons and that this does not change when training longer, we
stick to 100 training epochs for this model and only include it for the
sake of completeness.

.. figure:: images/roc-2000-naive-oversampling-normalised.png
   :align: center
   :height: 450px

We can see that the third model, trained with both *oversampling* and
*input normalisatoin* outperforms the other two, as indicated by its ROC
curve extending further to the top left corner of the plot, corresponding
to lower FPR and higher TPR. The *area under the curve* (:term:`AUC`) metric
condenses this into a single number and we can see that the third model
reaches an AUC of about 0.98 on the validation set as opposed to 0.94 for the
second model.



Summary
=======

In this tutorial, we used *TensorFlow* to set up and train a neural network
for binary classification, detecting whether or not patients were suffering
from a stroke. We encountered the problem of *imbalanced data*, saw how it
prevented a first naive model from learning to distinguish between both
target classes and then used *oversampling* to train a better classifier.
We also saw the impact of *input normalisation* and used different metics
as well as ROC curves to compare the performance of our different models.
Our final classifier reached an AUC of about 0.98 with a TPR of 100% and an FPR
of about 11% for the default working point with a treshold of 0.5 on the
sigmoid-activated output of the single node in the last network layer.
