from enum import Enum


class Filenames(Enum):
    RAW_IRIS_DATA = "iris.csv"
    PROCESSED_IRIS_DATA = "iris_processed.csv"
    IRIS_PREDICTIONS = "iris_predictions.csv"


class Model(Enum):
    NAME = "iris"
