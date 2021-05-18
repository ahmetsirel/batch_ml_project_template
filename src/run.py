import argparse
import sys
import os
import logging
from pathlib import Path
import datetime

file = Path(__file__).resolve()
src_directory = file.parents[1]
root_directory = file.parents[2]

sys.path.append(str(src_directory))

from data.data_loading import load_main_data
from features.build_features import prepare_features
from models.train_model import train_model
from models.predict_model import make_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="?", default="production")
    parser.add_argument("--prefix", nargs="?", default="")
    parser.add_argument("--run_date", nargs="?", default=datetime.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--prediction", default=False, action="store_true")
    args = parser.parse_args()

    return (args.env,
            args.prefix,
            args.run_date,
            args.train,
            args.prediction
            )


def train_pipeline():
    load_main_data()
    prepare_features()
    train_model()


def prediction_pipeline():
    load_main_data()
    prepare_features()
    make_predictions()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    (ENV, PREFIX, RUN_DATE, TRAIN, PREDICTION) = parse_args()

    if TRAIN:
        train_pipeline()

    if PREDICTION:
        prediction_pipeline()
