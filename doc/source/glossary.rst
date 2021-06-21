Glossary
========


A
-

.. glossary::

    AUC
        *Area Under the Curve*

        The area under a ROC curve. AUC is a metric that can be used to
        benchmark different models against each other, with larger values
        corresponding to better model performance.

        See :term:`ROC`



B
-

.. glossary::

    batch
        A batch is a subset of datapoints in a dataset that is processed
        in sequence by a neural network before the loss function is evaluated,
        the gradients are calculated and the model weights are updated
        using backprop.
        
        The larger the batch size is, the less frequently
        these updates will happen and therefore, the less time each epoch will
        take. On the flip side, the model parameters will not be updated
        very often and the progress from epoch to the next may be slower.
        Choosing smaller batch sizes on the other hand will lead to more
        gradient evaluations, more frequent updates of the model weights and
        therefore a longer duration of each epoch.

    bias
        The deviation of the predictions from the true labels in classification
        and of the predicted values from the true values in regression

    binary classification
        Classification problem with only two classes. These are often
        *positive* and *negative*, e.g. when a test for a disease is performed,
        and the *negative* class is associated with the null hypothesis, i.e.
        the hypothesis that is assumed to be valid until evidence of the
        contrary is presented.

        Contrast to :term:`multi-class classification`.



C
-

.. glossary::

    calibration
        *Calibration* refers to the process of checking and correcting
        the :term:`score` of a model in terms of its possible interpretation
        as a probability. For example, in :term:`binary classification`, the
        :term:`sigmoid`-activated model output is often interpreted as the
        probability that the datapoint belongs to the *positive* class.
        However, the fact that the :term:`sigmoid` function returns values
        in [0, 1] does not yet guarantee that its values accurately quantify
        the probability for a datapoint to belong to a certain class. These
        probabilities and their relation to the model :term:`score` are
        determined during *calibration*.

        The principle roughly is the following, illustrated with the example
        of :term:`binary classification` in mind:

        - Choose a certain threshold for the model :term:`score`, above which
          a datapoint is sorted into the positive category
        - Apply this threshold and

          - average the predicted scores over all datapoints that exceed the
            threshold
          - among the datapoints exceeding the threshold, determine the
            fraction that truly belong to the *positive* class

        - Add a point to the calibration plot with the average predicted score
          on the x-axis and the fraction of true positives on the y-axis
        - Repeat this with a number of different thresholds

        This results in a *calibration curve*, e.g. such as the one in
        https://scikit-learn.org/stable/modules/calibration.html#calibration-curves.

    CL
        :term:`confidence level`

    CNN
        *Convolutional Neural Network*

        A neural network containing one or more convolutional layers.

        In a 2D convolutional layer, typically used for images, :math:`n_f`
        two-dimensional filters of size :math:`f_1 \times f_2` are slid
        across the two-dimensional data arrays of size :math:`n_1 \times n_2`
        to create :math:`n_f` output data arrays of size
        :math:`(n_1-f_1+1) \times (n_2-f_2+1)`. Analogous convolutional
        layers of different dimensionalities exist as well.
        Convolutional layers are often followed by pooling layers that
        aggregate neighbouring pixels or voxels by calculating their maximum
        or average.

        By training and adjusting the filters, the neural network can
        distill particular patterns in the data and feed them to the
        following dense layers.

    confidence level
        The probability of *not* making a
        :term:`type-1 error`, i.e. the probability of *not* wrongly rejecting
        the null hypothesis and therefore rightly accepting the null hypothesis.

        .. math:: \text{confidence level} = \text{CL} = 1 - \alpha



D
-

.. glossary::

    data augmentation
        Techniques for artificially increasing the size of the dataset

        For example, in computer vision, images in the training set
        may be subjected to random shifts, rotations, shearing, horizontal
        flipping, changes in brightness, contrast, saturation and other
        properties.

        This can help to increase the model performance by allowing for
        more and longer training while at the same time avoiding
        :term:`overtraining`.

    dropout
        A *dropout layer* in a neural network randomly sets some of the values
        passed into it from the preceding layer to zero, i.e. randomly drops or
        deactivates some of its inputs. The fraction of dropped nodes, usually
        called *dropout rate*, is a model :term:`hyperparameter`.

        The original paper: G.E. Hinton et al: *Improving neural networks by
        preventing co-adaptation of feature detectors*, `arXiv:1207.0580
        <https://arxiv.org/abs/1207.0580>`_



F
-

.. glossary::

    FN
        *False Negative*: The label of a datapoint is predicted to be
        *negative*, but is *positive* in reality.

        .. caution:: False negatives can be particularly dangerous as e.g. a
           patient who really has a condition is not detected as sick and
           therefore is not treated.

    FP
        *False Positive*: The label of a datapoint is predicted to be
        *positive*, but is *negative* in reality.

    FPR
        *False Positive Rate*

        Defined as

        .. math:: \text{FPR} := \frac{\text{FP}}{\text{TN} + \text{FP}}
                              = 1 - \text{specificity} \approx \alpha

        where :math:`\text{FP}` are the false positives and :math:`\text{TN}`
        the true negatives.

        It specifies what fraction of the truly negative datapoints were
        incorrectly classified / predicted to be positive. Therefore, it is
        related to the :term:`type-1 error` and its probability :math:`\alpha`.



H
-

.. glossary::

    hyperparameter
        *Hyperparameters* characterise the layout and architecture of the
        model and its associated functions and algorithms. As such,
        *hyperparameters* are not and cannot be changed
        during training. There are two different types of *hyperparameters*:

        - **model hyperparameters**: e.g. the number of hidden layers in a
          neural network, the :term:`dropout` rate, the activation function
          of a specific layer
        - **algorithm hyperparameters**: e.g. the optimiser, its learning
          rate, the batch size
        
        *Hyperparameters* can be searched and optimised to maximise model
        performance. This process is called :term:`hyperparameter tuning` or
        :term:`hyperparameter optimisation`.

        Contrast against :term:`model parameter`.
    
    hyperparameter optimisation
        see :term:`hyperparameter tuning`
    
    hyperparameter tuning
        Evaluation of the achievable model performance when
        trying out different values for one or more hyperparameters. Normally,
        *hyperparameter tuning* refers to automated strategies for scanning
        different :term:`hyperparameter` values and ranges.

        Since evaluating a single point in hyperparameter space involves
        training and validating a model, *hyperparameter tuning* can be quite
        time-consuming and resource-intensive. Therefore, normally, not the
        full hyperparameter space is scanned for a model, but rather a
        subset.

        Tuning strategies broadly fall into three basic categories:

        - **grid searches**: All possible combinations of the selected
          hyperparameters and their values are tried out systematically.
          For categorical hyperparameters, e.g. the choice of the optimiser,
          all specified options are tried, and for continuous and ordinal
          hyperparameters, linearly or logarithmically equidistant points
          within configured ranges may be tried.
        - **random searches**: Points in the configured hyperparameter space
          are picked randomly
        - **advanced searches**: Advanced searches try to make informed
          decisions on which hyperparameter point to evaluate next, based
          on which hyperparameter points were scanned before and how they
          performed. A typical strategy is Bayesian optimisation together with
          Gaussian random processes.



I
-

.. glossary::

    image augmentation
        In *image augmentation*, transformations are applied to images
        before feeding them into a model. These transformations can serve
        to normalise the images, e.g. by rescaling them with a common
        factor, as well as to effectively increase the size of the datasets by
        applying random flips, rotations, brightness changes and other
        transformations. While these random transformations can help protect
        against :term:`overtraining`, they can also help the trained model
        in generalising to other images. For example, this would be the
        case with the *Fashion-MNIST* dataset which, among other types of
        clothes, contains shoes which are all pointing with their tips to
        the left.

    imbalanced data
        When the data contain significantly more datapoints in one class than
        the other(s), in :term:`binary classification` or
        :term:`multi-class classification`.

        See :doc:`examples/1-binary-stroke-prediction/index`



L
-

.. glossary::

    L1 regularisation
        When *L1* (or *lasso*) *regularisation* is activated for a layer, a
        penalty term *proportional to the sum of the absolute values* of the
        weights of that layer is added to the loss function. The strength
        of the regularisation can be adjusted by scaling the penalty term
        with a factor.

    L2 regularisation
        When *L2* (or *ridge*) *regularisation* is activated for a layer, a
        penalty term *quadratic in the sum* of the weights of that layer is
        added to the loss function. The strength of the regularisation can
        be adjusted by scaling the penalty term with a factor.

    lasso regularisation
        see :term:`L1 regularisation`



M
-

.. glossary::

    model parameter
        *Model parameters* are the parameters adjusted during training to
        minimise the loss function and fit the model to the training data,
        e.g. the weights of the edges between the nodes in a neural network.

        Contrast against :term:`hyperparameter`.

    multi-class classification
        Classification problem involving more than two classes

        Contrast to :term:`binary classification`.



O
-

.. glossary::

    overfitting
        see :term:`overtraining`

    oversampling
        Method for addressing :term:`imbalanced data`

        See :doc:`examples/1-binary-stroke-prediction/index`

    overtraining
        Also called *overfitting*

        When the model memorises specific random fluctuations in the training
        data. Since the validation does not contain the exact same datapoints,
        but rather others with different random fluctuations, the model fails
        to generalise to the validation data. Therefore, when *overtraining*
        occurs, the model performance is worse during validation than in
        training.

        In training, the *predicted* values lie close to the *true* values,
        but the model fails to generalise beyond the specific datapoints,
        corresponding to a low :term:`bias` but high :term:`variance`.

        *Overtraining* may occur when

        - the model is too complex, i.e. it has too many parameters
        - training continues for too long on a too limited dataset

        There are several strategies aimed at avoiding *overtraining*:

        - more training data
        - early stopping of the training, when the loss and accuracy do not
          improve anymore
        - a less complex model with fewer parameters
        - regularisation techniques

          - :term:`dropout` layers
          - :term:`L1 regularisation` or :term:`L2 regularisation`

        - :term:`data augmentation`


P
-

.. glossary::

    power
        The *power* of a test or classifier quantifies its capability of
        detecting a *positive* result. Therefore, it is related to the
        probability of the :term:`type-2 error` :math:`\beta` by:

        .. math::
        
           \text{power} = 1 - \beta

        See also: :term:`TPR`
    
    precision
        Defined as

        .. math:: \text{precision} := \frac{\text{TP}}{\text{TP} + \text{FP}}

        where :math:`\text{TP}` are the true positives and :math:`\text{FP}`
        the false positives.

        It specifies what fraction of the datapoints that were
        classified/predicted to be *positive* are in fact truly *positive*,
        i.e. which fraction of the *positive* classifications/predictions
        is correct. Therefore, e.g. in the context of medical tests,
        the *precision* is of special interest to the tested person or
        patient because it gives the probability for the *positive* result
        to be actually true.


R
-

.. glossary::

    recall
        see *True Positive Rate* (:term:`TPR`)

    ridge regularisation
        see :term:`L2 regularisation`
    
    ROC
        *Receiver Operator Characteristic*

        The ROC curve shows the *true positive rate* (:term:`TPR`) vs. the
        *false positive rate* (:term:`FPR`) for a given model. So it
        essentially gives the balance between type-1 and type-2 errors and
        visualises to what extent decreasing one will increase the other.
        Choosing a certain threshold value of the activated classifier
        output (and thereby defining the rule for associating datapoints with
        classes) corresponds to picking a working point somewhere on a given
        ROC curve and moving the threshold value scans the ROC curve so that
        a working point with the desired balance of error rates can be picked.

        The more the ROC curve extends to the top left corner, i.e. towards
        high TPRs at low FPRs, the better the performance of a model.
        Therefore, the *area under the curve* (:term:`AUC`) of a ROC curve can
        be used to benchmark different models against each other.



S
-

.. glossary::

    sample
        In datascience, *sample* refers to a single datapoint.

        Since I have a background in particle physics, where the term *sample*
        usually refers to a set of generated/simulated datapoints, I tend to
        avoid it and usually prefer *datapoint*.

        The vocabulary 'confusion matrix' that translates between data science
        and particle physics is the following:

        =============== ============ =================================================
        object          data science particle physics
        =============== ============ =================================================
        single entity   sample       datapoint
        set of entities dataset      - if *measured*: dataset
                                     - if *generated/simulated*: (Monte Carlo) sample
        =============== ============ =================================================

    score
        The *score of a model* is the activated output of a model, e.g. the
        activated output of the last layer in a neural network.

        The unactivated outputs are called *logits*.
        
        Commonly used activation functions are

        - :term:`sigmoid` activation in :term:`binary classification`
        - :term:`softmax` activation in :term:`multi-class classification`

    sensitivity
        see *True Positive Rate* (:term:`TPR`)

    sigmoid
        Sigmoid functions follow a characteristic 'S'-shape. In machine
        learning, *sigmoid activation* usually refers to using the
        *logistic function*

        .. math::
        
           f(x) = \frac{1}{1 + e^{-x}}
        
        as the activation function.
        
        Since the *sigmoid function* maps all real numbers to
        the interval (0, 1), *sigmoid activation* is typically used in
        :term:`binary classification`, with outputs close to 0 associated to
        one category and outputs close to 1 to the other. The sigmoid-
        activated network output is also often interpreted as the probability
        of a datapoint to belong to the second class, but this interpretation
        has to be taken with a grain of salt, see :term:`calibration`.

    softmax
        The *softmax* function is typically used as the activation function
        in :term:`multi-class classification` problems with one-hot
        encoded labels. It is defined as

        .. math::

           \sigma(\vec{x})_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}

        Each of the :math:`n` target classes corresponds to one entry in
        the classification vector :math:`\vec{x}` and the *softmax* function
        provides a mapping :math:`\mathbb{R}^n \to [0,1]^n`. Furthermore,
        it provides a normalisation such that the activated entries of the
        classification vector sum up to unity, i.e.
        :math:`\sum_{i=1}^n \sigma(\vec{x})_i = 1`.
        This is what is naturally expected for discreet probabilities.
        However, as long as a classifier is not calibrated, it cannot be
        guaranteed that the activated output of the last layer gives the
        probabilities for a datapoint to belong to each of the involved
        target classes.

    specificity
        Defined as

        .. math:: \text{specificity} := \frac{\text{TN}}{\text{TN} + \text{FP}}
                                      = 1 - \text{FPR}

        where :math:`\text{TN}` are the true negatives and :math:`\text{FP}`
        the false positives.

        It specifies what fraction of the truly *negative* datapoints was
        correctly classified/predicted to be *negative*. Therefore, the
        *specificity* is related to the :term:`FPR`.



T
-

.. glossary::

    testing
        The determination of the *unbiased* model performance.

        To this end, the full dataset available during model design, training
        and development is split up into three distinct parts:

        - the *training* dataset
        - the *validation* / *hold-out cross-validation* or
          *development* dataset and
        - the *test* dataset
        
        While the model parameters are adjusted on the *training* dataset, the
        performance of the model during the development phase is estimated from
        the *validation* dataset. Between the training runs, the
        hyperparameters are changed so as to maximise the performance metrics
        evaluated from the *validation* dataset. Finally, at the end of the
        development phase, a specific model and a set of hyperparameters is
        chosen and afterwards, the model performance is evaluated based on the
        *test* dataset. This is an unbiased estimate since the *test* data
        were never previously used to make choices regarding the model.

        Many times, when getting a proper unbiased estimate of the model
        performance is not crucial, no separate testing is performed. In such
        cases, the model performance is simply quantified with the validation
        results. In practice, this validation stage is then often referred to
        as 'testing'. 

    TN
        *True Negative*: The label of a datapoint is predicted to be
        *negative* and also is *positive* in reality

    TP
        *True Positive*: The label of a datapoint is predicted to be
        *positive* and also is *positive* in reality

    TPR
        *True Positive Rate*

        Defined as

        .. math:: \text{TPR} := \frac{\text{TP}}{\text{TP} + \text{FN}}
                              = \text{sensitivity}
                              = \text{recall} \approx 1 - \beta

        where :math:`\text{TP}` are the true positives and :math:`\text{FN}`
        the false negatives.

        It specifies what fraction of the truly positive datapoints were
        correctly classified / predicted to be positive. Therefore, it is
        related to the :term:`type-2 error` and its probability :math:`\beta`,
        or, more specifically the :term:`power` :math:`1 - \beta`.

    type-1 error
        The error of wrongly rejecting the null hypothesis and accepting the
        alternative hypothesis. Its probability is denoted with :math:`\alpha`:

        .. math:: \alpha := P(\text{type-1 error})

        It is related to the :term:`confidence level` (CL) by

        .. math:: \alpha = 1 - \text{confidence level}

        In :term:`binary classification`, where the null hypothesis is usually
        taken to be
        
        - a negative test
        - the patient is healthy
        -  the absence of new physics effects and the validity of the currently
           established model
        
        or a similarly *normal* situation, making type-1 errors results in
        *false positives* (:term:`FP`).

    type-2 error
        The error of wrongly accepting the null hypothesis and rejecting the
        alternative hypothesis. Its probability is denoted with :math:`\beta`:

        .. math:: \beta := P(\text{type-2 error})

        It is related to the :term:`power` by

        .. math:: \beta = 1 - \text{power}

        In :term:`binary classification`, where the null hypothesis is usually
        taken to be

        - a negative test
        - the patient is healthy
        -  the absence of new physics effects and the validity of the currently
           established model
        
        or a similarly *normal* situation, making type-2 errors results in
        *false negatives* (:term:`FN`).



U
-

.. glossary::

    underfitting
        see :term:`undertraining`

    undertraining
        Also called *underfitting*.

        When the model fails to learn the characteristic properties of the
        data during training. It is indicated by a bad model performance in
        both training and validation and the *predicted* values deviate from
        the *true* values, corresponding to a high :term:`bias`.

        *Undertraining* may occur when

        - there is not enough training data
        - there is too much noise in the training data, hiding the
          real characteristics and dependencies
        - training does not continue long enough
        - the model is inadequate and perhaps too simple to capture the
          characteristics of the data (e.g. as when trying to fit a linear
          function to datapoints following a sinus function)
        
        Possible strategies:

        - more training data
        - cleaner training data with less noise and statistical fluctuations
        - a more sophisticated or flexible model (e.g. more parameters,
          different types of layers)



V
-

.. glossary::

    variance
        The amount of variation in the model itself.
        
        For example, in
        function regression, a model may have very low bias, i.e. approximate
        the given *true values* very well, but at the same time oscillate
        and fluctuate wildly in between those true values. Such a model
        will generalise poorly to new data, see :term:`overtraining`.
