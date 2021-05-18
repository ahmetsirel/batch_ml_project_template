import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

file = Path(__file__).resolve()
src_directory = file.parents[1]
root_directory = file.parents[2]
MODEL_PATH = os.path.join(root_directory, "models")

sys.path.append(str(src_directory))

from configuration import Filenames, Model
from models.model_utils import get_latest_model, evaluate


def make_predictions():
    logger = logging.getLogger(__name__)
    logger.info('Make predictions')

    model = get_latest_model("iris")

    processed_data_dir = os.path.join(root_directory, "data/processed")
    data = pd.read_csv(os.path.join(processed_data_dir, Filenames.PROCESSED_IRIS_DATA.value), sep=";")

    x = data.drop("species", axis=1)
    y = data["species"]

    data["Predictions"] = model.predict(x)

    processed_data_dir = os.path.join(root_directory, "data/processed")
    data.to_csv(os.path.join(processed_data_dir, Filenames.IRIS_PREDICTIONS.value), sep=";", index=False)

    logger.info('Make predictions is finished.')
    return data


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    make_predictions()
