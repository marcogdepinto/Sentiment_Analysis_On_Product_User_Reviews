"""
This file
- Creates additional features using CSV scraped files;
- Clean and transforms the above features into a pandas dataframe;
- Saves the dataframe into a defined path.
"""
import glob
import joblib
import pandas as pd

from config import ADDITIONAL_FEATURES_PATH
from config import ADDITIONAL_FEATURES_DF_JOBLIB_PATH
from config import ADDITIONAL_FEATURES_DF_JOBLIB_FILE_NAME


class AdditionalFeaturesProcessing:
    """
    Additional features scraping and preparation.
    """

    def __init__(self):
        self.features_path = ADDITIONAL_FEATURES_PATH + '/*.csv'
        self.joblib_features_path = ADDITIONAL_FEATURES_DF_JOBLIB_PATH
        self.joblib_features_filename = ADDITIONAL_FEATURES_DF_JOBLIB_FILE_NAME
        self.dataframe = pd.DataFrame()
        self.tmp_list = list()

    def feature_loader(self) -> pd.DataFrame:
        """
        Function to load from a directory full of CSVs
        and create a pandas dataframe
        """
        for file_name in glob.glob(self.features_path):
            data = pd.read_csv(file_name)
            self.tmp_list.append(data)
        self.dataframe = self.dataframe.append(self.tmp_list)
        self.dataframe['comment'].str.strip()

        return self.dataframe

    def data_cleaner(self) -> pd.DataFrame:
        """
        Function to remove leading and trailing characters
        in the comment column.
        """
        self.dataframe['comment'] = self.dataframe['comment'].str.strip()
        return self.dataframe

    def stars_column_int_converter(self) -> pd.DataFrame:
        """
        Function to convert the sentences to integer for model purposes.
        Example input: "4,0 su 5 stelle"
        Output: 4
        """
        self.dataframe['stars'] = self.dataframe['stars'].str[0].astype(int)
        return self.dataframe

    def five_stars_remover(self) -> pd.DataFrame:
        """
        Function to remove the 5 stars reviews from the additional data.
        This is needed as the dataset is unbalanced (very high on 5 stars review).
        Hence, we want to increment the amount of 1-4 stars reviews only.
        """
        self.dataframe.drop(self.dataframe.index[self.dataframe['stars'] == 5],
                            inplace=True)
        self.dataframe = self.dataframe.reset_index(drop=True)
        return self.dataframe

    def rename_dataframe_columns(self) -> pd.DataFrame:
        """
        Function to rename the dataframe columns according to
        the dataset we are going to merge the output of this script with.
        In details, the column names need to be processed_sentence and score
        """
        self.dataframe.rename(columns={'stars': 'score',
                                       'comment': 'processed_sentence'},
                              inplace=True)
        return self.dataframe

    def save_dataframe_to_joblib(self) -> None:
        """
        Function to save the created dataframe into a joblib file.
        """
        print('Calling dataframe from word processing', self.dataframe)
        joblib.dump(self.dataframe, self.joblib_features_path + self.joblib_features_filename)


if __name__ == '__main__':
    PROCESSING = AdditionalFeaturesProcessing()
    PROCESSING.feature_loader()
    PROCESSING.data_cleaner()
    PROCESSING.stars_column_int_converter()
    PROCESSING.five_stars_remover()
    PROCESSING.rename_dataframe_columns()
    PROCESSING.save_dataframe_to_joblib()
