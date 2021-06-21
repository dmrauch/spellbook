import json
import numpy as np
import requests

import helpers


test_images = helpers.load_images(helpers.tshirts)
# test_images = helpers.load_images(helpers.sandals)
# test_images = helpers.load_images(helpers.sneakers)

data = json.dumps({
    'signature_name': 'serving_default',
    'instances': test_images.tolist()   # *either* 'instances'
    # 'inputs': test_images.tolist()    # *or* 'inputs'
})

print('------------------------------------------------------------')
print("for querying the served model from the terminal with 'curl',"
      " use the following command\n")
print("curl -d '{}' -X POST {}".format(
    data, 'http://localhost:8501/v1/models/fmnist-model:predict'))
print('\n------------------------------------------------------------')


headers = {'content-type': 'application/json'}
json_response = requests.post(
    'http://localhost:8501/v1/models/fmnist-model:predict',
    headers=headers,
    data=data
)

predictions = json.loads(json_response.text)['predictions'] # for 'instances'
# predictions = json.loads(json_response.text)['outputs']   # for 'inputs'

for i, prediction in enumerate(predictions):
    print('prediction {}: {} -> predicted class: {}'.format(
        i, prediction, np.argmax(prediction)))
