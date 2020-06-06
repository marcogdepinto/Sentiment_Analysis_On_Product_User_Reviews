"""
Train.py trains a deep learning regressor from a matrix created using the Train class.
"""
from typing import List
from numpy import zeros
from numpy import asarray
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_FEATURES_PATH
from config import TRAIN_PROCESSED_DATAFRAME_FILENAME
from config import DEV_PROCESSED_DATAFRAME_FILENAME
from config import EMBEDDINGS_PATH
from config import EMBEDDINGS_FILENAME
from config import MODEL_PATH
from config import MODEL_NAME


class Train:
    """
    Trains a deep learning regressor from a matrix created within the class itself.
    """

    def __init__(self):
        self.dataframe_path = PROCESSED_FEATURES_PATH
        self.dataframe = pd.DataFrame()
        self.embeddings_path = EMBEDDINGS_PATH
        self.embeddings_filename = EMBEDDINGS_FILENAME
        self.embeddings_dictionary = dict()

    def task_3_dataframe_loader(self, dataframe_name: str) -> pd.DataFrame:
        """
        Loading the data. As the scope is task 3 of EVALITA,
        columns not needed for sentiment analysis are dropped.
        """
        print('Starting to load ', dataframe_name)
        self.dataframe = joblib.load(self.dataframe_path + '/' + dataframe_name)
        print(dataframe_name, ' loaded.')
        return self.dataframe

    def create_x_y_features(self) -> tuple:
        """
        Scaling the reviews from 0 to 4. This is needed as
        Keras models classify starting from 0 and not from one.
        :return: x, y
        """

        print('Starting to create x and y features..')
        tmp_x = self.dataframe['processed_sentence']
        tmp_y = self.dataframe['score']
        print('x and y features created.')

        return tmp_x, tmp_y

    def matrix_creation(self, tmp_x_train: list, tmp_x_test: list) -> tuple:
        """
        Creates the matrix that will be used to train the model.
        """

        print('Starting to load the word embeddings and create the matrices..')

        tokenizer = Tokenizer(num_words=6000)

        tokenizer.fit_on_texts(tmp_x_train)

        tmp_x_train = tokenizer.texts_to_sequences(tmp_x_train)
        tmp_x_test = tokenizer.texts_to_sequences(tmp_x_test)

        # glove_file = open(EMBEDDINGS_PATH + EMBEDDINGS_FILENAME, encoding="utf8")
        glove_file = open(EMBEDDINGS_PATH + EMBEDDINGS_FILENAME)

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            self.embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 300

        tmp_x_train = pad_sequences(tmp_x_train, padding='post', maxlen=maxlen)
        tmp_x_test = pad_sequences(tmp_x_test, padding='post', maxlen=maxlen)

        embedding_matrix = zeros((vocab_size, 300))
        for word, index in tokenizer.word_index.items():
            embedding_vector = self.embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        print("Matrix is ready to feed the model.")

        return embedding_matrix, vocab_size, tmp_x_train, tmp_x_test

    @staticmethod
    def root_mean_squared_error(tmp_y_train: list, tmp_y_test: list) -> List:
        """
        RMSE loss function
        """
        return K.sqrt(K.mean(K.square(tmp_y_test - tmp_y_train)))

    @staticmethod
    def rmse(predictions: list, targets: list) -> List:
        """
        RMSE evaluation function.
        """
        return np.sqrt(((predictions - targets) ** 2).mean())

    def model(self, embedding_matrix: list, vocab_size: int,
              tmp_x_train: list, tmp_y_train: list,
              tmp_x_test: list, tmp_y_test: list) -> list:
        """
        Trains the deep learning model and prints metrics.
        """

        model = Sequential()
        maxlen = 300
        embedding_layer = Embedding(vocab_size,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        model.add(embedding_layer)
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='rmsprop',
                      loss=self.root_mean_squared_error,
                      metrics=[self.root_mean_squared_error])
        print(model.summary())

        print('Starting model training..')

        history = model.fit(tmp_x_train, tmp_y_train,
                            batch_size=300, epochs=50,
                            verbose=1, validation_split=0.2)

        # Loss plotting
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        predicts = model.predict(tmp_x_test)

        predictions = []

        for pred in predicts:
            predictions.append(pred[0])

        rmse_val = self.rmse(tmp_y_test, predictions)
        print("RMS error is: " + str(rmse_val))

        model.save(MODEL_PATH + '/' + MODEL_NAME)

        print('Routine completed.')

        return rmse_val


if __name__ == '__main__':
    # Train features creation
    TRAIN = Train()
    TRAIN_DATAFRAME = TRAIN.task_3_dataframe_loader(TRAIN_PROCESSED_DATAFRAME_FILENAME)
    x_train, y_train = TRAIN.create_x_y_features()

    # Dev features creation
    DEV = Train()
    DEV_DATAFRAME = DEV.task_3_dataframe_loader(DEV_PROCESSED_DATAFRAME_FILENAME)
    x_test, y_test = DEV.create_x_y_features()

    # Matrix creation
    INSTANCE = Train()
    MATRIX, VOCAB_SIZE, x_train, x_test = INSTANCE.matrix_creation(x_train, x_test)
    print(MATRIX.size, MATRIX.shape)

    # Model training
    MODEL = Train()
    MODEL.model(MATRIX, VOCAB_SIZE,
                x_train, y_train,
                x_test, y_test)
