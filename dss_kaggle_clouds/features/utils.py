"""General utilities for managing/processing files"""

import os
import glob

def expect_path_or_error(fp: str):
    """
    :fp - filepath to check if file exists
    """
    if not os.path.exists(fp):
        print("Expected path at {} but it didn't exist.".format(fp))
        print("Make sure that the data dir you're passing like 'kaggle_clouds data/raw/' has the unzipped files in that dir; at data/raw/train_images, data/raw/train.csv")
        sys.exit(1)

def verify_files_exist(data_dir: str):

    data_path = os.path.abspath(data_dir)
    expect_path_or_error(data_path)
    expect_path_or_error(os.path.join(data_path, 'train.csv'))
    expect_path_or_error(os.path.join(data_path, 'train_images'))
    return True
