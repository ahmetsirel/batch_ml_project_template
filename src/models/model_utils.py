import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

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

from configuration import Filenames

MODEL_PATH = os.path.join(root_directory, "models")


def get_latest_model(model_name: str, model_path=MODEL_PATH, run_date=None):
    models = [filename for filename in os.listdir(model_path) if "pkl" in filename and filename.startswith(model_name)]
    models.sort()
    latest_model_name = models[-1]
    latest_model = pickle.load(open(os.path.join(model_path, latest_model_name), 'rb'))
    print(latest_model_name, "is loaded.")
    return latest_model


def evaluate(predictions):
    return None
