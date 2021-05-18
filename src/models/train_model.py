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


def train_model():
    logger = logging.getLogger(__name__)
    logger.info('Train model')

    processed_data_dir = os.path.join(root_directory, "data/processed")
    data = pd.read_csv(os.path.join(processed_data_dir, Filenames.PROCESSED_IRIS_DATA.value), sep=";")

    x = data.drop("species", axis=1)
    y = data["species"]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

    # Build the Model
    scaler = StandardScaler()
    model = LogisticRegression()
    model_pipeline = Pipeline([("scale", scaler), ("model", model)])

    model_pipeline.fit(x_train, y_train)

    y_test_pred = model_pipeline.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)
    print("accuracy_score", acc)

    training_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_location = f"{MODEL_PATH}/{Model.NAME.value} | {training_date_time}.pkl"
    pickle.dump(model_pipeline, open(model_location, "wb"))

    msg = f"Model with {acc} accuracy, saved into {model_location}."
    logger.info('Train model finished.')
    logger.info(msg)
    return msg


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model()
