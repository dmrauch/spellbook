import numpy as np
import pandas as pd
import tensorflow as tf

import spellbook as sb

import helpers


# run name and prefix for filenames
name = 'oversampling normalised'
prefix = name.replace(' ', '-')

# data loading and cleaning
data, vars, target, features = helpers.load_data()

# normalisation
data['age_norm'] = data['age'] / 100.0
data['avg_glucose_level_norm'] = data['avg_glucose_level'] / 300.0
data['bmi_norm'] = data['bmi'] / 100.0
# replace unnormalised variable names with their normalised counterparts
features[features.index('age')] = 'age_norm'
features[features.index('avg_glucose_level')] = 'avg_glucose_level_norm'
features[features.index('bmi')] = 'bmi_norm'

# inplace convert string category labels to numerical indices
categories = sb.input.encode_categories(data)

# oversampling (including shuffling of the data)
data = sb.input.oversample(data, target)

# use new numerical columns for the features
for var in categories:
    if var == target:
        target = target + '_codes'
    else:
        features[features.index(var)] = var + '_codes'

sb.plot.save(
    sb.plot.plot_grid_1D(nrows=2, ncols=3, data=data,
        features = [
            'age', 'avg_glucose_level', 'bmi',
            'age_norm', 'avg_glucose_level_norm', 'bmi_norm'
        ],
        stats=False),
    filename = 'variables-normalised.png'
)

# input pipeline: pandas.DataFrame -> numpy.ndarray
# total is 9400 datapoints (4700 for each class)
# split at 7000 = 74.5% of all data
train_features, train_labels, val_features, val_labels \
    = sb.input.split_pddataframe_to_nparrays(
        data, target, features, n_train=7000)

print(train_features.shape) # print the size of the dataset
print(train_labels.shape)
print(val_features.shape)
print(val_labels.shape)


# ------------------------------------------------------------------------------
# MODEL SETUP AND TRAINING
# ------------------------------------------------------------------------------

# prepare the model
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

# train the model
epochs = 10
# epochs = 500
# epochs = 2000
history = model.fit(train_features, train_labels,
    batch_size=32, epochs=epochs,
    validation_data=(val_features, val_labels),
    callbacks = [
        tf.keras.callbacks.CSVLogger(
            filename='{}-e{}-history.csv'.format(prefix, epochs)),
        sb.train.ModelSavingCallback(
            foldername='model-{}-e{}'.format(prefix, epochs))
    ])


# ------------------------------------------------------------------------------
# MODEL VALIDATION
# ------------------------------------------------------------------------------

# inspect/plot the training history
sb.train.plot_history_binary(history,
    '{}-e{}-history'.format(prefix, epochs))


# obtain the predictions of the model
train_predictions = model.predict(train_features)
val_predictions = model.predict(val_features)

# not strictly necessary: remove the superfluous inner nesting
train_predictions = tf.reshape(train_predictions, train_predictions.shape[0])
val_predictions = tf.reshape(val_predictions, val_predictions.shape[0])

# until now the predictions are still continuous in [0, 1] and this breaks the
# calculation of the confusion matrix, so we need to set them to either 0 or 1
# according to an intermediate threshold of 0.5
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
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids),
    filename = '{}-e{}-confusion.png'.format(prefix, epochs))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-all'),
    filename = '{}-e{}-confusion-norm-all.png'.format(prefix, epochs))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-true',
        crop = True,
        figsize = (6.4, 4.8)),
    filename = '{}-e{}-confusion-norm-true.png'.format(prefix, epochs))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-pred',
        crop = True,
        figsize = (5.3, 5.8)),
    filename = '{}-e{}-confusion-norm-pred.png'.format(prefix, epochs))


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
