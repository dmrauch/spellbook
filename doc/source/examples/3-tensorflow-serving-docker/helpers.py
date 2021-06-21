import numpy as np
import tensorflow as tf

from typing import List
from typing import Union


label_dict = {
    0: 't-shirt/top',
    1: 'trouser',
    2: 'pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'ankle boot'
}

tshirts = [
    'images/test/tshirt-1.jpg',
    'images/test/tshirt-2.png',
    'images/test/tshirt-3.jpg',
    'images/test/tshirt-4.png'
]
sandals = [f'images/test/sandal-{i}.jpg' for i in range(1, 5)]
sneakers = [f'images/test/sneaker-{i}.jpg' for i in range(1, 5)]


def get_generator(images, labels, batch_size=32, **kwargs):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        rescale=1/255)

    gen = datagen.flow(
        x = np.expand_dims(images, axis=3),
        y = tf.keras.utils.to_categorical(labels, num_classes=10),
        batch_size = batch_size,
        **kwargs
    )

    return gen


def load_images(images: Union[str, List[str]]):

    if isinstance(images, str): images = [images]

    array = np.empty(shape=(len(images), 28, 28, 1))
    for i, image in enumerate(images):
        img = tf.keras.preprocessing.image.load_img(
            path = image,
            color_mode = 'grayscale',
            target_size = (28,28))

        array[i] = tf.keras.preprocessing.image.img_to_array(img=img)

    return (255 - array) / 255