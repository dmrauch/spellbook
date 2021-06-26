import json
import numpy as np
import requests
import tensorflow as tf

from typing import List
from typing import Union


def load_images(images: Union[str, List[str]]):

    if isinstance(images, str): images = [images]

    array = np.empty(shape=(len(images), 28, 28, 1))
    for i, image in enumerate(images):
        img = tf.keras.preprocessing.image.load_img(
            path = image,
            color_mode = 'grayscale',
            target_size = (28, 28))

        array[i] = tf.keras.preprocessing.image.img_to_array(img=img)

    return (255 - array) / 255


tshirts = [
    'images/test/tshirt-1.jpg', 'images/test/tshirt-2.png',
    'images/test/tshirt-3.jpg', 'images/test/tshirt-4.png'
]
sandals = [f'images/test/sandal-{i}.jpg' for i in range(1, 5)]
sneakers = [f'images/test/sneaker-{i}.jpg' for i in range(1, 5)]

# test_images = load_images(tshirts)
test_images = load_images(sandals)
# test_images = load_images(sneakers)

IPv4 = 'ec2-54-93-96-215.eu-central-1.compute.amazonaws.com'
# IPv4 = '54.93.96.215'


data = json.dumps({
    'signature_name': 'serving_default',
    'instances': test_images.tolist()   # *either* 'instances'
    # 'inputs': test_images.tolist()    # *or* 'inputs'
})

headers = {'content-type': 'application/json'}
json_response = requests.post(
    f'http://{IPv4}:8501/v1/models/fmnist-model:predict',
    headers=headers,
    data=data
)

predictions = json.loads(json_response.text)['predictions'] # for 'instances'
# predictions = json.loads(json_response.text)['outputs']   # for 'inputs'

for i, prediction in enumerate(predictions):
    print('prediction {}: {} -> predicted class: {}'.format(
        i, prediction, np.argmax(prediction)))
