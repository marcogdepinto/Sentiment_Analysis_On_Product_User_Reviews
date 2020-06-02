"""
Configuration file: includes constants, paths and other information.
"""

TRAINING_SET_PATH = './ATE_ABSITA_training_set/ate_absita_training.ndjson'
DEV_SET_PATH = './ATE_ABSITA_dev_set/ate_absita_dev.ndjson'

CLEAN_DATAFRAME_PATH = './joblib_not_processed_dataframe'
TRAIN_DATAFRAME_FILE_NAME = 'train_dataframe.joblib'
DEV_DATAFRAME_FILE_NAME = 'dev_dataframe.joblib'

PROCESSED_FEATURES_PATH = './joblib_processed_features'
TRAIN_PROCESSED_DATAFRAME_FILENAME = 'processed_train_dataframe.joblib'
DEV_PROCESSED_DATAFRAME_FILENAME = 'processed_dev_dataframe.joblib'

EMBEDDINGS_PATH = './glove_embeddings/glove.6B'
EMBEDDINGS_FILENAME = '/glove.6B.300d.txt'

# Valid values: 2, 5.
NUMBER_OF_CLASSES = 5

MODEL_PATH = './models'
MODEL_NAME = '_classes_model.h5'