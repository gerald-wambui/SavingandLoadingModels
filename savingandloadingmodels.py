# We take a trained model, save it, then load it back.
# We will save it as a HDF5 file, which is the format used by Keras
# concepts
"""
        saving models in HDF5 format
        saving models in Tensorflow SavedModel format
        loading models
        download models to Local Disk for deployment to different platforms
"""

# Imports
!pip install -U tensorflow_hub
!pip install -U tensorflow_datasets

import time
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from tensorflow.keras import layers

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)
# reformat images to have the same res 224*224
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label
num_examples = info.splits['train'].num_examples
BATCH_SIZE = 32
IMAGE_RES = 224
train_batches = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)