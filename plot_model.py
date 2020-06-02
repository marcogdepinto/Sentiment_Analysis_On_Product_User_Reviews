"""
File used to make a plot of the model.
"""

import tensorflow.keras as tf
from tensorflow.keras.utils import plot_model

from config import MODEL_PATH
from config import MODEL_NAME

restored_keras_model = tf.models.load_model(MODEL_PATH + '/2' + MODEL_NAME)

plot_model(restored_keras_model, to_file='models/2_model.png')