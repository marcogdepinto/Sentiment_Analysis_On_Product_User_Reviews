"""
File used to make a plot of the model.
"""

import tensorflow.keras as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

from config import MODEL_PATH
from config import MODEL_NAME


def root_mean_squared_error(tmp_y_train: list, tmp_y_test: list) -> list:
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(tmp_y_test - tmp_y_train)))


restored_keras_model = tf.models.load_model(MODEL_PATH + '/' + MODEL_NAME,
                                            custom_objects={'root_mean_squared_error': root_mean_squared_error})

plot_model(restored_keras_model, to_file='models/model.png')
