.. _tools-tf-save:

Saving and Loading Models
=========================

There are three ways of saving and reloading models:

#. saving the model weights manually
#. saving the model weights automatically during training using a callback function
#. saving the full model

At any point you can save the model weights manually by doing

.. code:: python

   model.save_weights('filename.h5')

You can then later restore the weights by loading them from the file with

.. code:: python

   model.load_weights('filename.h5')

Note that for reloading the weights an instance of the model is needed.

This can also be done automatically during the training loop using a callback function:

.. code:: python

  checkpoint_path = 'training/model-e{epoch:04d}.ckpt'
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
  model.fit(..., callbacks=[checkpoint_callback])

By default, this will save the model weights after every single epoch. The path of the latest checkpoint can then be retrieved with

.. code:: python

  tf.train.latest_checkpoint('training')

This mechanism is handy for keeping persistent copies of the model so that an interrupted training can later be resumed.

Finally, it is possible to save the full model, including its definition and architecture, with

.. code:: python

   model.save('model.h5')

Then, the model can be reused even without knowing the exact architecture using

.. code:: python

   model = tf.keras.models.load_model('model.h5')
