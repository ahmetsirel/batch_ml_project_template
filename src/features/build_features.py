import sys
import os
import logging
from pathlib import Path
import pandas as pd

file = Path(__file__).resolve()
src_directory = file.parents[1]
root_directory = file.parents[2]

sys.path.append(str(src_directory))

from configuration import Filenames

class Erdem:
    pass

def prepare_features():
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    raw_data_dir = os.path.join(root_directory, "data/raw")
    data = pd.read_csv(os.path.join(raw_data_dir, Filenames.RAW_IRIS_DATA.value), sep=";")

    processed_data_dir = os.path.join(root_directory, "data/processed")
    data.to_csv(os.path.join(processed_data_dir, Filenames.PROCESSED_IRIS_DATA.value), sep=";", index=False)

    logger.info('making final data set from raw data is finished.')

    return data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    prepare_features()
