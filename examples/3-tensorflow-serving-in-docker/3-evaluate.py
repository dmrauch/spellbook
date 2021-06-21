import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import spellbook as sb

import helpers


batch_size = 250

model = tf.keras.models.load_model('fmnist-model')

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (val_images, val_labels) \
    = fashion_mnist.load_data()

val_gen = helpers.get_generator(val_images, val_labels, batch_size, shuffle=False)

history = model.evaluate(val_gen, return_dict=True)
print(history)

val_predictions = model.predict(val_gen)
val_predicted_labels = np.argmax(val_predictions, axis=1)

class_ids = list(helpers.label_dict.keys())
class_names = list(helpers.label_dict.values())
val_confusion = tf.math.confusion_matrix(
    val_labels, val_predicted_labels, num_classes=len(class_ids))

sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion,
        class_names = class_names,
        class_ids = class_ids,
        fontsize = 9.0,
        fontsize_annotations = 'x-small'
    ),
    filename = 'fmnist-model-confusion.png'
)
sb.plot.save(
    sb.plot.plot_confusion_matrix(
        confusion_matrix = val_confusion,
        class_names = class_names,
        class_ids = class_ids,
        normalisation = 'norm-true',
        crop = False,
        figsize = (6.4, 4.8),
        fontsize = 9.0,
        fontsize_annotations = 'x-small'
    ),
    filename = 'fmnist-model-confusion-norm-true.png'
)
