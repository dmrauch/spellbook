import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import spellbook as sb

import helpers


n = 7

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (val_images, val_labels) \
    = fashion_mnist.load_data()

print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape)

print(np.array2string(train_images[0], max_line_width=150))

train_gen = helpers.get_generator(train_images, train_labels, batch_size=n,
    shuffle=False)

train_images_augmented = next(train_gen)

fig = plt.figure(figsize=(7,2))
grid = mpl.gridspec.GridSpec(nrows=2, ncols=n, wspace=0.1, hspace=0.1)

for i in range(n):
    ax = plt.Subplot(fig, grid[0,i])
    fig.add_subplot(ax)
    ax.set_axis_off()
    ax = plt.imshow(train_images[i], cmap='gray')

    ax = plt.Subplot(fig, grid[1,i])
    fig.add_subplot(ax)
    ax.set_axis_off()
    ax = plt.imshow(train_images_augmented[0][i], cmap='gray')

sb.plot.save(fig, 'images.png')
