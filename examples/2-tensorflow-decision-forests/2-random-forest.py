import tensorflow as tf
import tensorflow_decision_forests as tfdf

import spellbook as sb

import helpers

prefix = 'random-forest'

# data loading and cleaning
data, vars, target, features = helpers.load_data()

# inplace convert string category labels to numerical indices
categories = sb.input.encode_categories(data)

# oversampling (including shuffling of the data)
data = sb.input.oversample(data, target)

# split into training and validation data
n_split = 7000
train = tfdf.keras.pd_dataframe_to_tf_dataset(
    data[features+[target]].iloc[:n_split], label=target)
val = tfdf.keras.pd_dataframe_to_tf_dataset(
    data[features+[target]].iloc[n_split:], label=target)

# define binary metrics
metrics = [
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.Precision(name='precision')
]

# prepare and train the model
model = tfdf.keras.RandomForestModel()
model.compile(metrics=metrics)
model.fit(train)
# model.summary()
model.save('model-{}'.format(prefix))

eval = model.evaluate(val, return_dict=True)
print('eval:', eval)
# print(model.make_inspector().variable_importances())


# separate the datasets into features and labels
train_features, train_labels = zip(*train.unbatch())
val_features, val_labels = zip(*val.unbatch())

# obtain the predictions of the model
train_predictions = model.predict(train)
val_predictions = model.predict(val)

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
val_confusion_matrix = tf.math.confusion_matrix(
    val_labels, val_predicted_labels, num_classes=len(class_names))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids),
    filename = '{}-confusion.png'.format(prefix))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-all'),
    filename = '{}-confusion-norm-all.png'.format(prefix))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-true',
        crop = True,
        figsize = (6.4, 4.8)),
    filename = '{}-confusion-norm-true.png'.format(prefix))
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion_matrix,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-pred',
        crop = True,
        figsize = (5.3, 5.8)),
    filename = '{}-confusion-norm-pred.png'.format(prefix))

# calculate and plot the ROC curve
roc = sb.train.ROCPlot()
roc.add_curve('random forest (training)',
    train_labels, train_predictions.numpy(),
    plot_args = dict(color='C0', linestyle='--'))
roc.add_curve('random forest (validation)',
    val_labels, val_predictions.numpy(),
    plot_args = dict(color='C0', linestyle='-'))
WP = roc.get_WP('random forest (validation)', threshold=0.5)
roc.draw_WP(WP)
sb.plot.save(roc.plot(), '{}-roc.png'.format(prefix))
roc.pickle_save('{}-roc.pickle'.format(prefix))

# calculate feature importance
importance = sb.inspect.PermutationImportance(
    data, features, target, model, metrics, n_repeats=10, tfdf=True)
sb.plot.save(
    importance.plot(
        metric_name='accuracy',
        annotations_alignment = 'left',
        xmin = 0.56),
    filename='{}-permutation-importance-accuracy.png'.format(prefix))
