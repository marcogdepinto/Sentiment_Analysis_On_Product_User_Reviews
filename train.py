import sys
import joblib
import pandas as pd
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import PROCESSED_FEATURES_PATH
from config import TRAIN_PROCESSED_DATAFRAME_FILENAME
from config import DEV_PROCESSED_DATAFRAME_FILENAME
from config import EMBEDDINGS_PATH
from config import EMBEDDINGS_FILENAME
from config import NUMBER_OF_CLASSES
from config import MODEL_PATH
from config import MODEL_NAME


class Train:

    def __init__(self):
        self.dataframe_path = PROCESSED_FEATURES_PATH
        self.dataframe = pd.DataFrame()
        self.embeddings_path = EMBEDDINGS_PATH
        self.embeddings_filename = EMBEDDINGS_FILENAME
        self.embeddings_dictionary = dict()
        self.number_of_classes = NUMBER_OF_CLASSES

    def task_3_dataframe_loader(self, dataframe_name) -> pd.DataFrame:
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
        :return: X, y
        """

        print('Starting to create X and y features..')

        X = self.dataframe['processed_sentence']

        if self.number_of_classes == 2:
            y = self.dataframe['score']
            y = y.replace(1, 0)
            y = y.replace(5, 1)
            y = y.replace(4, 0)
            y = y.replace(3, 0)
            y = y.replace(2, 0)

        elif self.number_of_classes == 5:
            y = self.dataframe['score'] - 1

        else:
            print('NUMBER_OF_CLASSES in config.py can be only 2 or 5')
            sys.exit(0)

        print('X and y features created.')

        return X, y

    def matrix_creation(self, X_train, X_test) -> None:
        """
        Creates the matrix that will be used to train the model.
        """

        print('Starting to load the word embeddings and create the matrices..')

        tokenizer = Tokenizer(num_words=6000)

        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        glove_file = open(EMBEDDINGS_PATH + EMBEDDINGS_FILENAME, encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            self.embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 300

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        embedding_matrix = zeros((vocab_size, 300))
        for word, index in tokenizer.word_index.items():
            embedding_vector = self.embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        print("Matrix is ready to feed the model.")

        return embedding_matrix, vocab_size, X_train, X_test

    def model(self, embedding_matrix, vocab_size,
              X_train, y_train, X_test, y_test) -> tuple:
        """
        Trains the deep learning model and prints metrics.
        """

        loss_function = 'sparse_categorical_crossentropy' if\
            self.number_of_classes == 5 else 'binary_crossentropy'
        dense_out = 5 if self.number_of_classes == 5 else 1

        print('The loss that will be used is: ', loss_function)
        print('The number of classes that will be used is ', self.number_of_classes)
        print('The final dense layer that will be used is ', dense_out)

        model = Sequential()
        maxlen = 300
        embedding_layer = Embedding(vocab_size,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        model.add(embedding_layer)
        model.add(Flatten())
        model.add(Dense(dense_out, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss=loss_function,
                      metrics=['accuracy'])
        print(model.summary())

        print('Starting model training..')

        history = model.fit(X_train, y_train, batch_size=300, epochs=5, verbose=1, validation_split=0.2)

        # Loss plotting
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')

        score = model.evaluate(X_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        predictions = model.predict_classes(X_test)

        matrix = confusion_matrix(y_test, predictions)
        print(matrix)

        report = classification_report(y_test, predictions)
        print(report)

        model.save(MODEL_PATH + '/' + str(NUMBER_OF_CLASSES) + MODEL_NAME)

        print('Routine completed.')


if __name__ == '__main__':
    # Train features creation
    TRAIN = Train()
    TRAIN_DATAFRAME = TRAIN.task_3_dataframe_loader(TRAIN_PROCESSED_DATAFRAME_FILENAME)
    X_train, y_train = TRAIN.create_x_y_features()

    # Dev features creation
    DEV = Train()
    DEV_DATAFRAME = DEV.task_3_dataframe_loader(DEV_PROCESSED_DATAFRAME_FILENAME)
    X_test, y_test = DEV.create_x_y_features()

    # Matrix creation
    INSTANCE = Train()
    MATRIX, VOCAB_SIZE, X_train, X_test = INSTANCE.matrix_creation(X_train, X_test)
    print(MATRIX.size, MATRIX.shape)

    # Model training
    MODEL = Train()
    MODEL.model(MATRIX, VOCAB_SIZE,
                X_train, y_train,
                X_test, y_test)
