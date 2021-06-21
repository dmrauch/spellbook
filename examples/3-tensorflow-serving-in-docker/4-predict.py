import numpy as np
import tensorflow as tf

import helpers


model = tf.keras.models.load_model('fmnist-model')

test_images = helpers.load_images(helpers.tshirts)
# test_images = helpers.load_images(helpers.sandals)
# test_images = helpers.load_images(helpers.sneakers)

predictions = model.predict(test_images)
for prediction in predictions:
    print('prediction:', prediction,
        '-> predicted class:', np.argmax(prediction))
