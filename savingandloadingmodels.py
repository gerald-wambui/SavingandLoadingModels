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

