"""
Configuration file: includes constants, paths and other information.
"""
import os

TRAINING_SET_PATH = './ATE_ABSITA_training_set/ate_absita_training.ndjson'
DEV_SET_PATH = './ATE_ABSITA_dev_set/ate_absita_dev.ndjson'

CLEAN_DATAFRAME_PATH = './joblib_not_processed_dataframe'
TRAIN_DATAFRAME_FILE_NAME = 'train_dataframe.joblib'
DEV_DATAFRAME_FILE_NAME = 'dev_dataframe.joblib'

PROCESSED_FEATURES_PATH = './joblib_processed_features'
TRAIN_PROCESSED_DATAFRAME_FILENAME = 'processed_train_dataframe.joblib'
DEV_PROCESSED_DATAFRAME_FILENAME = 'processed_dev_dataframe.joblib'

EMBEDDINGS_PATH = './embeddings/'
EMBEDDINGS_FILENAME = '/cc.it.300.vec'

MODEL_PATH = './models'
MODEL_NAME = 'model.h5'

ADDITIONAL_FEATURES_PATH = './additional_scraped_reviews'
ADDITIONAL_FEATURES_DF_JOBLIB_PATH = './joblib_additional_reviews'
ADDITIONAL_FEATURES_DF_JOBLIB_FILE_NAME = '/additional_features.joblib'
