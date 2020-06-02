"""
Text preprocessing.
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import os
import re
import joblib
import pandas as pd

from config import CLEAN_DATAFRAME_PATH
from config import PROCESSED_FEATURES_PATH
from config import TRAIN_DATAFRAME_FILE_NAME
from config import DEV_DATAFRAME_FILE_NAME


class Preprocessing:
    """
    1) Loading the dataframe
    2) Preprocessing text input (stopwords, punctuations, spaces and so on)
    3) Creating a new joblib with the processed feature set
    """

    def __init__(self):
        self.dataframe_path = CLEAN_DATAFRAME_PATH
        self.dataframe = pd.DataFrame()

    def task_3_dataframe_loader(self, dataframe_name) -> pd.DataFrame:
        """
        Loading the data. As the scope is task 3 of EVALITA,
        columns not needed for sentiment analysis are dropped.
        """
        self.dataframe = joblib.load(self.dataframe_path + '/' + dataframe_name)
        self.dataframe = self.dataframe.drop(columns=['id_sentence',
                                                      'polarities',
                                                      'aspects_position',
                                                      'aspects'])
        return self.dataframe

    @staticmethod
    def preprocess_text(dataframe, replace_column) -> pd.DataFrame:
        """
        Removes punctuation, numbers, single characters,
        multiple spaces and stopwords.
        """

        dataframe_column = dataframe[replace_column].tolist()
        new_values_list = list()

        for sentence in dataframe_column:
            # Remove punctuations and numbers
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)
            # Single character removal
            sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence)
            # Tokenize sentence
            sentence = word_tokenize(sentence)
            # Remove stopwords
            sentence = [word for word in sentence if word not in stopwords.words('italian')]
            # Merge processed sentence
            sentence = ' '.join(sentence)
            new_values_list.append(sentence)

        # Creating the column with the processed sentences
        dataframe['processed_sentence'] = new_values_list

        # Dropping the old dataframe column
        del dataframe['sentence']

        return dataframe

    def save_dataframe_to_joblib(self, file_name):
        """
        Function to save the created dataframe into a joblib file.
        """
        print('Calling dataframe from word processing', self.dataframe)
        joblib.dump(self.dataframe, os.path.join(PROCESSED_FEATURES_PATH, file_name))


if __name__ == '__main__':

    PREPROCESSING_TRAIN = Preprocessing()
    # Loading train dataframe
    TRAIN_DATAFRAME = PREPROCESSING_TRAIN.task_3_dataframe_loader(TRAIN_DATAFRAME_FILE_NAME)
    # Creating a new column elaborated column: processed_sentence
    TRAIN_PROCESSED_DATAFRAME = PREPROCESSING_TRAIN.preprocess_text(TRAIN_DATAFRAME, 'sentence')
    PREPROCESSING_TRAIN.save_dataframe_to_joblib('processed_' + TRAIN_DATAFRAME_FILE_NAME)

    PREPROCESSING_DEV = Preprocessing()
    # Loading dev dataframe
    DEV_DATAFRAME = PREPROCESSING_DEV.task_3_dataframe_loader(DEV_DATAFRAME_FILE_NAME)
    # Creating a new column elaborated column: processed_sentence
    DEV_PROCESSED_DATAFRAME = PREPROCESSING_DEV.preprocess_text(DEV_DATAFRAME, 'sentence')
    PREPROCESSING_DEV.save_dataframe_to_joblib('processed_' + DEV_DATAFRAME_FILE_NAME)
