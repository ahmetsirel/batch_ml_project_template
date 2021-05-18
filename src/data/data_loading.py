# -*- coding: utf-8 -*-

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


def load_main_data():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading raw dataset.')

    data = pd.read_csv(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

    raw_data_dir = os.path.join(root_directory, "data/raw")
    data.to_csv(os.path.join(raw_data_dir, Filenames.RAW_IRIS_DATA.value), sep=";", index=False)
    logger.info('Loading raw dataset is finished.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_main_data()
