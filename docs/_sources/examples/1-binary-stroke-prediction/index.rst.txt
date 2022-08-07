********************************************************
Binary Classification with the Stroke Prediction Dataset
********************************************************


.. toctree::
   :hidden:

   exploration-cleaning
   input-pipeline
   training-validation


.. raw:: html

   <div class="tag-list">
      <div class="tag-cell tag-date">June 13, 2021</div>
      <div class="tag-cell tag-center"></div>
      <div class="tag-cell tag-right">
         <span class="tag-text">tags:</span>
         <a class="tag right" href="../../search.html?q=tags+binary+classification">
            binary classification</a>
         <a class="tag right" href="../../search.html?q=tags+imbalanced+data">
            imbalanced data</a>
         <a class="tag right" href="../../search.html?q=tags+oversampling">
            oversampling</a>
         <a class="tag right" href="../../search.html?q=tags+supervised+learning">
            supervised learning</a>
      </div>
   </div>



.. admonition:: In this project/tutorial, we will
   :class: spellbook-admonition-orange

   - **Explore** the `Stroke Prediction Dataset
     <https://www.kaggle.com/fedesoriano/stroke-prediction-dataset>`_ and
     inspect and **plot** its variables and their correlations by means of the
     *spellbook* library
   - Set up an **input pipeline** that loads the data from the original ``*.csv``
     file, preprocesses them and feeds them into a neural network
   - Use **TensorFlow** to train a **neural network / multi-layer perceptron**
     to do **binary classification**
   - Encounter the problem of **imbalanced data** and explore the strategy of
     **oversampling** to address it
   - See what effect even a very basic **normalisation of the input data** can
     have on the model performance

Here on this page, I will give a summary of the key points and ideas of this
project without any code snippets. The technical and implementation details
are then laid out in depth on the next pages of this tutorial:

- :doc:`/examples/1-binary-stroke-prediction/exploration-cleaning`
- :doc:`/examples/1-binary-stroke-prediction/input-pipeline`
- :doc:`/examples/1-binary-stroke-prediction/training-validation`



Setup
=====

- Install the *spellbook* library as detailed in :ref:`Installation`
- The source code for this tutorial is located in
  ``examples/1-binary-stroke-prediction/``
- Download the `Stroke Prediction Dataset
  <https://www.kaggle.com/fedesoriano/stroke-prediction-dataset>`_ from
  Kaggle and extract the file ``healthcare-dataset-stroke-data.csv``



Summary without Implementation Details
======================================


This dataset contains a total of 5110 datapoints, each of them describing a
patient, whether they have had a stroke or not, as well as 10 other variables,
ranging from gender, age and type of work to the existence of medical
conditions such as heart problems or hypertension.

The task we are going to pursue is :term:`binary classification`, specifically
the prediction of the presence of a stroke based on the other variables.
The key characteristic of this dataset is the stark imbalance in the fractions
of patients with a stroke vs. the ones without.
This situation of very differently populated target classes is called
:term:`imbalanced data`.

For 201 datapoints, no BMI was given. While it may be
worthwhile to consider imputing these values, we will drop the affected
datapoints for now. 4909 datapoints remain with 4.3% of the people having
suffered a stroke.

.. _ex1-fig-variables:

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/stroke.png
          :height: 250px

          Figure 1: The binary target variable

     - .. figure:: images/variables.png
          :height: 250px

          Figure 2: Distributions of all variables

Let us first follow the *naive* approach by ignoring the imbalance and train
a simple fully-connected neural network / multi-layer perceptron on a training
set of 3500 datapoints, leaving us with a validation dataset of 1409 datapoints.
In this tutorial, I am using a network with three hidden dense layers of 10,
100 and 50 units/nodes/neurons, respectively, and an output layer, interleaved
with two :term:`dropout` layers, and with :term:`L2 regularisation` activated
in the largest hidden dense layer. The output layer consists of a
single sigmoid-activated
neuron. This architecture is the result of manually trying out different model
configurations in the later stages of this tutorial, but we are using the same
network throughout in order to have a fair comparison of the different training
strategies.

Very quickly, already in the first 10 epochs, the loss and the accuracy flatten
off, as can be seen in Figure 3. The accuracy settles at
around 96%, which just corresponds to the fraction of healthy people in the
dataset. Looking at the confusion matrix calculated from the predictions for
the validation dataset and shown in Figure 4, we can confirm that the model is
not at all picking up the positive cases and is purely and only predicting the
negative class, i.e. the absence of a stroke. Therefore, the accuracy, although
appearing to be good, cannot capture the fact that this model is inadequate.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/naive-e100-history-loss-acc.png
          :height: 250px

          Figure 3: Naive approach: loss and accuracy

     - .. figure:: images/naive-e100-confusion-edited.png
          :height: 250px

          Figure 4: Naive approach: Confusion matrix

So let's try an approach aimed at dealing with imbalanced data and *oversample
the minority class*, i.e. the positive class containing the patients with a
stroke in this example. In this strategy, we create an artificial dataset
that is populated in equal numbers with the positive and the negative class.
This is achieved by separating the full dataset into patients with a stroke
and patients without a stroke and then drawing with replacement from the
``stroke = yes`` class as many times as there are datapoints in the
``stroke = no`` class (4700 datapoints). Of course, this means that the same
datapoints of patients with a stroke will be included multiple times but the
point is to create a balanced dataset (with 2 * 4700 datapoints)
so that the model can learn the characteristics of both classes with equal
emphasis. Also, adopting the opposite approach of downsampling the majority
class would throw away a lot of good data and leave us with a very limited
dataset of only 2 * 209 datapoints.

When training the same model on this new, balanced dataset, we can see a lower
initial accuracy of about 70%, increasing slowly but steadily as the training
continues, as depicted in Figure 5. Likewise the loss is somewhat
larger than in the naive case, but steadily decreases over time. When looking
at the evolution of the numbers of true and false positives and negatives,
given in Figure 6, we can see that the model is now definitely predicting
datapoints to belong to the positive class as well. Moreover, the false
predictions decrease during training and the confusion matrix based on the
validation dataset given in Figure 7 shows that of all the patients with a
stroke less than 1% were predicted to be healthy. These are the *false
negatives* (:term:`FN`), which are of particular interest in the context of
tests or models in medicine because a falsely negative classification means
that a patient's condition is not recognised. On the other side, the
confusion matrix indicates that 27.6% of all truly negative cases were
predicted to be positive during validation. This is the
*false positive rate* (:term:`FPR`). A false positive corresponds to worrying
a patient with an alarming test result, normally triggering more tests.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/oversampling-e500-history-loss-acc.png
          :height: 250px

          Figure 5: Oversampling: loss and accuracy

     - .. figure:: images/oversampling-e500-history-pos-neg.png
          :height: 250px

          Figure 6: Oversampling: true/false positives/negatives

     - .. figure:: images/oversampling-e500-confusion-norm-true.png
          :height: 250px

          Figure 7: Oversampling: confusion matrix normalised by the true labels

But while, especially from the patients' perspective, it is preferable to
beat down the false negatives as much as possible in order to minimise the
risk of not detecting a true condition, it would also be nice to simultaneously
minimise the probability of false positives. For patients, this will lead to
less false alarms, and for the healthcare sector, this corresponds to less
follow-up tests and therefore less cost. For a given model, however, false
positives can only be traded against false negatives and
decreasing one will result in an increase of the other. This can be seen on
the Receiver Operator Characteristic (:term:`ROC`) curves, into which we will
look later. Therefore, decreasing both the false positives and negatives at
the same time can only be achieved by creating a better-performing model.

So let's explore the impact of another technique typically used to improve
model performance, *normalisation of the input data*. It consists of
scaling down the numerical values of continuous variables and restricting
them to some interval close to or around zero, typically with an interval
length of the order of one. This strategy is based on the experience that
models can achieve better performance when fed with small numbers as opposed
to values of the order of tens, hundreds or more. While there are more
involved variants of this principle, such as *standardisation*, we will just
perform a very simple normalisation by division and scale down age and BMI
by a factor of 100, and the average glucose level by a factor of 300.
The resulting distributions retain their original shape as can be seen in
Figure 8.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/variables-normalised.png
          :height: 250px

          Figure 8: Normalised continuous variables

After training for 2000 epochs with both oversampling and input normalisation,
we should check the various metrics for signs of :term:`overtraining` which
would show up as a significantly better model performance on the training
data compared to the validation data. As an example, the evolution of the
loss and the accuracy is shown in Figure 9, indicating that there is no
overtraining. The confusion matrix is given in Figure 10. The false negatives
have gone down even further and we can now see that the false positives have
improved as well,
from 27.6% before to now 9.5%. This is also reflected in the :term:`ROC` curves
given in Figure 11 which compare the performance of all three models:

- the original naive model, trained over 100 epochs
- the model trained on the oversampled dataset for 2000 epochs
- the final model with both oversampling and normalisation of the inputs,
  trained over 2000 epochs

The standard working points based on a
sigmoid-activated classifier output of 0.5 are indicated with filled red
circles. The better the performance of a model, the more its :term:`ROC` curve extends
to the top left corner, corresponding to simultaneously low false positives
and false negatives, the latter indicated by a high true positive rate
(:term:`TPR`). Therefore, one metric typically used to compare the performance
of different models is the *area under the curve* (:term:`AUC`) of the :term:`ROC`
curves. As already seen from the confusion matrices, the last model with both
oversampling and normalisation of the inputs performs best, reaching AUCs of
0.99 in both training and validation.

.. list-table::
   :class: spellbook-gallery-wrap

   * - .. figure:: images/oversampling-normalised-e2000-history-loss-acc.png
          :height: 250px

          Figure 9: Evolution of loss and accuracy

     - .. figure:: images/oversampling-normalised-e2000-confusion-norm-true.png
          :height: 250px

          Figure 10: Confusion matrix for normalisation and oversampling

     - .. figure:: images/roc-2000-naive-oversampling-normalised.png
          :height: 250px

          Figure 11: :term:`ROC` curves of the different models
