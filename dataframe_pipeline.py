"""
Creates the dataframe to be used to load the features.
"""

import os
import joblib
import ndjson
import pandas as pd

from config import TRAINING_SET_PATH
from config import DEV_SET_PATH
from config import CLEAN_DATAFRAME_PATH
from config import TRAIN_DATAFRAME_FILE_NAME
from config import DEV_DATAFRAME_FILE_NAME


class DataframePipeline:
    """
    Create a dataframe from source file and save it into a joblib archive.
    """
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def feature_loader(self, path: str) -> pd.DataFrame:
        """
        Function to create a dataframe from source files.
        """
        with open(path) as f:
            reader = ndjson.reader(f)

            for post in reader:
                df = pd.DataFrame([post], columns=post.keys())
                self.dataframe = pd.concat([self.dataframe, df],
                                           axis=0,
                                           ignore_index=True)

        return self.dataframe

    def save_dataframe_to_joblib(self, file_name: str) -> pd.DataFrame:
        """
        Function to save the created dataframe into a joblib file.
        """
        print('Calling dataframe from word processing', self.dataframe)
        joblib.dump(self.dataframe, os.path.join(CLEAN_DATAFRAME_PATH, file_name))
        return self.dataframe


if __name__ == '__main__':
    TRAIN_PIPELINE = DataframePipeline()
    print('Starting training dataframe creation...')
    TRAIN_PIPELINE.feature_loader(path=TRAINING_SET_PATH)
    TRAIN_PIPELINE.save_dataframe_to_joblib(TRAIN_DATAFRAME_FILE_NAME)
    print('Training dataframe created.')
    print('Starting dev dataframe creation...')
    DEV_PIPELINE = DataframePipeline()
    DEV_PIPELINE.feature_loader(path=DEV_SET_PATH)
    DEV_PIPELINE.save_dataframe_to_joblib(DEV_DATAFRAME_FILE_NAME)
    print('Dev dataframe created.')
    print('Procedure completed.')
