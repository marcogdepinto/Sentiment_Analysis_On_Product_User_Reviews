from config import ADDITIONAL_FEATURES_PATH


class AdditionalFeaturesProcessing:

    def __init__(self):
        self.features_path = ADDITIONAL_FEATURES_PATH

    def processing(self):
        pass


if __name__ == '__main__':
    PROCESSING = AdditionalFeaturesProcessing()
    PROCESSING.processing()
