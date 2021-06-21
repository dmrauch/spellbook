import tensorflow as tf

import spellbook as sb

import helpers

batch_size = 64
epochs = 3

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (val_images, val_labels) \
    = fashion_mnist.load_data()

train_gen = helpers.get_generator(train_images, train_labels, batch_size)
val_gen = helpers.get_generator(val_images, val_labels, batch_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu',
        input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
model.summary()
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(),
    metrics = [
        tf.keras.metrics.CategoricalCrossentropy(),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    ]
)

history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
    callbacks = [
        tf.keras.callbacks.CSVLogger(filename='fmnist-model-history.csv'),
        sb.train.ModelSavingCallback(foldername='fmnist-model')
    ]
)
